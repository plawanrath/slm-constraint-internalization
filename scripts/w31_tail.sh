#!/usr/bin/env bash
# Takes over after the training-budget sweep. Inserts the adapter control ahead of the remaining steps, because it
# decides whether the gold-injection price is a scale effect or an adapter effect, and the later steps do not.
#   nohup caffeinate -dims ./scripts/w31_tail.sh >> logs/w31_tail.log 2>&1 &
#   L. 360M SSD + one gold per group under rank-32 adapters (the 360M run in the main tables is a full fine-tune,
#      the 1.7B run uses adapters, so scale and adapter capacity are otherwise confounded)
#   P. held-out-dialect transfer for the 1.7B full fine-tune (is the loss an adapter effect?)
#   K. per-token KL to the untrained base for the checkpoints the transfer claims rest on
#   C. 1.7B SSD + one gold per group, seed 1
#   D. 1.7B GRPO (unmasked rollouts) + portability and StableHLO
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M360=HuggingFaceTB/SmolLM2-360M-Instruct; R360=a10cc1512eabd3dde888204e902eca88bddb4951
M17=HuggingFaceTB/SmolLM2-1.7B-Instruct;  R17=31b70e2e869a7173562077fd711b654946d38674
C17="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25 --lr 1e-4 --lora --lora-rank 32 --lora-scale 2 --max-tokens 300 --micro-batch 8"
log() { echo "[w31] $(date '+%m-%d %H:%M') $*"; }
ckpt_of() { ls -d "models/ckpt/$1/best" "results/w14_ablations/ckpt/$1/best" "results/w03_train/$1/ckpt/best" 2>/dev/null | head -1; }
decode() { $PY scripts/w04_eval_ckpt.py --model "$1" --ckpt "$2" --tag "$3" --method "$4" --train-seed "$5" && echo "$3" >> logs/score_queue.txt; }
sweep() { local T=$1; shift
  for S in "$@"; do C=$(ls -d models/ckpt/$T/step$S results/w14_ablations/ckpt/$T/step$S 2>/dev/null | head -1)
    [ -n "$C" ] || continue
    $PY scripts/w04_eval_ckpt.py --model $M360 --ckpt "$C" --tag "${T}__step=$S" --method ssd --train-seed 0 \
        --pools arith_func_200 --constraints none,c1_c2_c3 && echo "${T}__step=$S" >> logs/score_queue.txt
  done; }

log "waiting for the training-budget sweep to finish"
until grep -q "^\[w30\].*B:" logs/w30_tail.log; do sleep 120; done
pkill -f "w30_tail.sh" || true; sleep 3; pkill -f "scripts/w03_train.py" || true; sleep 5
log "took over from the previous lane"

log "L: 360M SSD + one gold per group, rank-32 adapters (adapter control)"
T=smollm2-360m-instruct_ssd_s0__gold_in_group=1__lora
[ -f results/w14_ablations/$T/summary.json ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd --seed 0 \
  --epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25 --lr 1e-4 --lora --lora-rank 32 --lora-scale 2 \
  --gold-in-group 1 --out-dir results/w14_ablations --tag $T
C=$(ckpt_of $T); [ -n "$C" ] && $PY scripts/w04_eval_ckpt.py --model $M360 --ckpt "$C" --tag "$T" --method ssd --train-seed 0 \
  --pools arith_func_200 --constraints none,c1,c1_c2_c3 && echo "$T" >> logs/score_queue.txt
log "L done"

log "P: held-out-dialect transfer for the 1.7B full fine-tune"
T=smollm2-1.7b-instruct_ssd_s0__full_ft
$PY scripts/w12_portability.py --tags $T || log "portability failed"
$PY scripts/w10_stablehlo.py --cell "$T=$(ckpt_of $T)" --method ssd || log "stablehlo failed"

log "K: per-token KL to the untrained base"
kl() { $PY scripts/w32_kl_base.py --model "$1" --revision "$2" --ckpt "$3" --tag "$4" || log "kl $4 failed"; }
kl $M360 $R360 "$(ckpt_of smollm2-360m-instruct_ssd_s0)" smollm2-360m-instruct_ssd_s0
kl $M17 $R17 "$(ckpt_of smollm2-1.7b-instruct_ssd_s0)" smollm2-1.7b-instruct_ssd_s0
kl $M17 $R17 "$(ckpt_of smollm2-1.7b-instruct_ssd_s0__full_ft)" smollm2-1.7b-instruct_ssd_s0__full_ft
kl $M17 $R17 "$(ckpt_of smollm2-1.7b-instruct_rft_s0)" smollm2-1.7b-instruct_rft_s0
kl $M17 $R17 "$(ckpt_of smollm2-1.7b-instruct_sft_s0)" smollm2-1.7b-instruct_sft_s0

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
log "W31 TAIL DONE"
