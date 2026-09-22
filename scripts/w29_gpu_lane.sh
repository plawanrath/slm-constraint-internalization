#!/usr/bin/env bash
# GPU lane for the follow-up experiments. Runs alone on the GPU; functional scoring of the
# pass@k samples happens in scripts/w29_score_lane.sh, and the two lanes share one scorer through a mkdir lock.
#   nohup caffeinate -dims ./scripts/w29_gpu_lane.sh >> logs/w29_gpu_lane.log 2>&1 &
# 1. base 360M masked pass@256 samples on the 150 gold-gate prompts        (support test at k = 256)
# 3. sampled correctness (k = 8) at 1.7B (base, SSD s0/s1, RFT s0) and 3B (base, SSD, RFT)
# 4. 1.7B: base pass@16 samples; SFT-gold; SSD + one gold per group (s0); evaluations
# 2. 360M SSD at 8x steps with checkpoints at 1x/2x/4x/8x; group 16 at 4x rollouts with the same checkpoints
# 4b. 1.7B SSD + one gold per group, seed 1
# 5. 1.7B GRPO (unmasked on-policy rollouts) + portability
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M360=HuggingFaceTB/SmolLM2-360M-Instruct; R360=a10cc1512eabd3dde888204e902eca88bddb4951
M17=HuggingFaceTB/SmolLM2-1.7B-Instruct;  R17=31b70e2e869a7173562077fd711b654946d38674
M3B=HuggingFaceTB/SmolLM3-3B;             R3B=a07cc9a04f16550a088caea529712d1d335b0ac1
C360="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25 --lr 2e-5"
C17="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25 --lr 1e-4 --lora --lora-rank 32 --lora-scale 2 --max-tokens 300 --micro-batch 8"
log() { echo "[w29-gpu] $(date '+%m-%d %H:%M') $*"; }
lock() { until mkdir logs/.scorer.lock 2>/dev/null; do sleep 30; done; }
unlock() { rmdir logs/.scorer.lock 2>/dev/null || true; }
ckpt_of() { ls -d "models/ckpt/$1/best" "results/w14_ablations/ckpt/$1/best" "results/w03_train/$1/ckpt/best" 2>/dev/null | head -1; }
evalckpt() { # MODEL CKPT TAG METHOD SEED
  $PY scripts/w04_eval_ckpt.py --model "$1" --ckpt "$2" --tag "$3" --method "$4" --train-seed "$5"
  lock; $PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$3"; unlock; }

log "1: base 360M pass@256 samples (gate prompts)"
$PY scripts/w16_passk.py --model $M360 --revision $R360 --k 256 --gate-only --gen-only

log "3: sampled correctness k=8 at 1.7B and 3B"
$PY scripts/w16_passk.py --model $M17 --revision $R17 --k 8 --gate-only --gen-only --tag smollm2-1.7b-instruct
for T in smollm2-1.7b-instruct_ssd_s0 smollm2-1.7b-instruct_ssd_s1 smollm2-1.7b-instruct_rft_s0; do
  $PY scripts/w16_passk.py --model $M17 --revision $R17 --k 8 --gate-only --gen-only --ckpt "$(ckpt_of $T)" --tag "$T"; done
$PY scripts/w16_passk.py --model $M3B --revision $R3B --k 8 --gate-only --gen-only --tag smollm3-3b
for T in smollm3-3b_ssd_s0 smollm3-3b_rft_s0; do
  $PY scripts/w16_passk.py --model $M3B --revision $R3B --k 8 --gate-only --gen-only --ckpt "$(ckpt_of $T)" --tag "$T"; done
echo "STAGE3 GEN DONE" >> logs/w29_gpu_lane.log

log "4: 1.7B base pass@16 samples"
$PY scripts/w16_passk.py --model $M17 --revision $R17 --k 16 --gate-only --gen-only --tag smollm2-1.7b-instruct
echo "STAGE4 GEN DONE" >> logs/w29_gpu_lane.log
log "4: SFT-gold at 1.7B"
T=smollm2-1.7b-instruct_sft_s0
[ -f results/w03_train/$T/summary.json ] || $PY scripts/w03_train.py --model $M17 --revision $R17 --method sft --seed 0 $C17 --tag $T
evalckpt $M17 "$(ckpt_of $T)" $T sft 0
log "4: SSD + one gold per group at 1.7B, seed 0"
T=smollm2-1.7b-instruct_ssd_s0__gold_in_group=1
[ -f results/w14_ablations/$T/summary.json ] || $PY scripts/w03_train.py --model $M17 --revision $R17 --method ssd --seed 0 $C17 --gold-in-group 1 --out-dir results/w14_ablations --tag $T
evalckpt $M17 "$(ckpt_of $T)" $T ssd 0

log "2: 360M SSD at 8x steps (checkpoints at 126/252/504/1008)"
T=smollm2-360m-instruct_ssd_s0__epochs=16
[ -f results/w14_ablations/$T/summary.json ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd --seed 0 --epochs 16 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25 --lr 2e-5 --save-steps 126,252,504,1008 --out-dir results/w14_ablations --tag $T
for S in 126 252 504 1008; do C=$(ls -d models/ckpt/$T/step$S results/w14_ablations/ckpt/$T/step$S 2>/dev/null | head -1); [ -n "$C" ] && evalckpt $M360 "$C" "${T}__step=$S" ssd 0; done
log "2: 360M SSD group 16 at 4x rollouts (checkpoints at 126/252/500)"
T=smollm2-360m-instruct_ssd_s0__group=16__epochs=2
[ -f results/w14_ablations/$T/summary.json ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd --seed 0 --epochs 2 --n-train 2000 --group 16 --batch-prompts 8 --eval-every 25 --lr 2e-5 --save-steps 126,252,500 --out-dir results/w14_ablations --tag $T
for S in 126 252 500; do C=$(ls -d models/ckpt/$T/step$S results/w14_ablations/ckpt/$T/step$S 2>/dev/null | head -1); [ -n "$C" ] && evalckpt $M360 "$C" "${T}__step=$S" ssd 0; done

log "4b: SSD + one gold per group at 1.7B, seed 1"
T=smollm2-1.7b-instruct_ssd_s1__gold_in_group=1
[ -f results/w14_ablations/$T/summary.json ] || $PY scripts/w03_train.py --model $M17 --revision $R17 --method ssd --seed 1 $C17 --gold-in-group 1 --out-dir results/w14_ablations --tag $T
evalckpt $M17 "$(ckpt_of $T)" $T ssd 1

log "5: 1.7B GRPO (unmasked on-policy rollouts) + portability"
T=smollm2-1.7b-instruct_grpo_s0
[ -f results/w03_train/$T/summary.json ] || $PY scripts/w03_train.py --model $M17 --revision $R17 --method grpo --seed 0 $C17 --tag $T
evalckpt $M17 "$(ckpt_of $T)" $T grpo 0
$PY scripts/w12_portability.py --tags $T || log "portability failed"
$PY scripts/w10_stablehlo.py --tags $T || log "stablehlo failed"
log "W29 GPU LANE DONE"
