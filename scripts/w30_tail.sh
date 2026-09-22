#!/usr/bin/env bash
# Takes over the GPU lane after the 1.7B one-gold-per-group run: same remaining items, but each checkpoint's ladder
# decode is followed by queueing the tag for scripts/w30_scorer.sh instead of scoring it inline, so the GPU starts the
# next training run immediately.
#   nohup caffeinate -dims ./scripts/w30_tail.sh >> logs/w30_tail.log 2>&1 &
#   A. 360M SSD at 8x steps, checkpoints at 126/252/504/1008        (training-budget sweep)
#   B. 360M SSD, group 16, 4x rollouts, checkpoints at 126/252/500  (rollout-budget sweep)
#   C. 1.7B SSD + one gold per group, seed 1
#   D. 1.7B GRPO (unmasked rollouts) + portability and StableHLO
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M360=HuggingFaceTB/SmolLM2-360M-Instruct; R360=a10cc1512eabd3dde888204e902eca88bddb4951
M17=HuggingFaceTB/SmolLM2-1.7B-Instruct;  R17=31b70e2e869a7173562077fd711b654946d38674
C17="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25 --lr 1e-4 --lora --lora-rank 32 --lora-scale 2 --max-tokens 300 --micro-batch 8"
log() { echo "[w30] $(date '+%m-%d %H:%M') $*"; }
ckpt_of() { ls -d "models/ckpt/$1/best" "results/w14_ablations/ckpt/$1/best" "results/w03_train/$1/ckpt/best" 2>/dev/null | head -1; }
decode() { # MODEL CKPT TAG METHOD SEED: ladder decode on the GPU, then hand the tag to the scorer
  $PY scripts/w04_eval_ckpt.py --model "$1" --ckpt "$2" --tag "$3" --method "$4" --train-seed "$5" && echo "$3" >> logs/score_queue.txt; }
# Budget-sweep checkpoints need free acceptance and full-stack correctness on arith+func only, so they decode two
# stacks on one pool instead of four on two: a quarter of the decode time, and the scope and shape residuals still come out.
sweep() { # TAG STEP...: decode every saved step checkpoint on the reduced grid
  local T=$1; shift
  for S in "$@"; do C=$(ls -d models/ckpt/$T/step$S results/w14_ablations/ckpt/$T/step$S 2>/dev/null | head -1)
    [ -n "$C" ] || continue
    $PY scripts/w04_eval_ckpt.py --model $M360 --ckpt "$C" --tag "${T}__step=$S" --method ssd --train-seed 0 \
        --pools arith_func_200 --constraints none,c1_c2_c3 && echo "${T}__step=$S" >> logs/score_queue.txt
  done; }

log "waiting for the 1.7B one-gold-per-group evaluation to finish"
until grep -q "2: 360M SSD at 8x steps" logs/w29_gpu_lane.log; do sleep 120; done
pkill -f "w29_gpu_lane.sh" || true; sleep 3; pkill -f "scripts/w03_train.py" || true; sleep 5
log "took over from the previous lane"

log "A: 360M SSD at 8x steps"
T=smollm2-360m-instruct_ssd_s0__epochs=16
[ -f results/w14_ablations/$T/summary.json ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd --seed 0 --epochs 16 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25 --lr 2e-5 --save-steps 126,252,504,1008 --out-dir results/w14_ablations --tag $T
sweep $T 126 252 504 1008

log "B: 360M SSD, group 16, 4x rollouts"
T=smollm2-360m-instruct_ssd_s0__group=16__epochs=2
[ -f results/w14_ablations/$T/summary.json ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd --seed 0 --epochs 2 --n-train 2000 --group 16 --batch-prompts 8 --eval-every 25 --lr 2e-5 --save-steps 126,252,500 --out-dir results/w14_ablations --tag $T
sweep $T 126 252 500

log "C: 1.7B SSD + one gold per group, seed 1"
T=smollm2-1.7b-instruct_ssd_s1__gold_in_group=1
[ -f results/w14_ablations/$T/summary.json ] || $PY scripts/w03_train.py --model $M17 --revision $R17 --method ssd --seed 1 $C17 --gold-in-group 1 --out-dir results/w14_ablations --tag $T
decode $M17 "$(ckpt_of $T)" $T ssd 1

log "D: 1.7B GRPO (unmasked rollouts)"
T=smollm2-1.7b-instruct_grpo_s0
[ -f results/w03_train/$T/summary.json ] || $PY scripts/w03_train.py --model $M17 --revision $R17 --method grpo --seed 0 $C17 --tag $T
decode $M17 "$(ckpt_of $T)" $T grpo 0
$PY scripts/w12_portability.py --tags $T || log "portability failed"
$PY scripts/w10_stablehlo.py --tags $T || log "stablehlo failed"
touch logs/score_queue.done
log "W30 TAIL DONE"
