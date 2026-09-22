#!/usr/bin/env bash
# Final experiment queue: three additions that test the "source of the signal" mechanism.
#   A. gold-guided SSD at 360M: one template gold replaces the lowest-reward rollout of every group
#   B. masked pass@16 of the untrained 360M model on arith_func_200 (the on-policy ceiling)
#   C. dense scope mask in SSD at 360M (per-token C3 mass loss, weight 1)
# Each step is idempotent. One GPU job at a time; the functional scorer runs alone.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M=HuggingFaceTB/SmolLM2-360M-Instruct
REV=a10cc1512eabd3dde888204e902eca88bddb4951
COMMON="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25 --lr 2e-5 --out-dir results/w14_ablations"
log() { echo "[final] $(date '+%m-%d %H:%M') $*"; }

run_abl() {  # TAG extra-flags...
  local TAG=$1; shift
  if [ ! -f "results/w14_ablations/$TAG/summary.json" ]; then
    $PY scripts/w03_train.py --model $M --revision $REV --method ssd --seed 0 $COMMON --tag "$TAG" "$@"
  fi
  for c in "results/w14_ablations/ckpt/$TAG/best" "models/ckpt/$TAG/best"; do
    [ -d "$c" ] && $PY scripts/w04_eval_ckpt.py --model $M --ckpt "$c" --tag "$TAG" --method ssd --train-seed 0 && break
  done
  $PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"
}

log "A: gold-guided SSD (gold_in_group=1)"
run_abl smollm2-360m-instruct_ssd_s0__gold_in_group=1 --gold-in-group 1
CK=$(ls -d results/w14_ablations/ckpt/smollm2-360m-instruct_ssd_s0__gold_in_group=1/best models/ckpt/smollm2-360m-instruct_ssd_s0__gold_in_group=1/best 2>/dev/null | head -1)
[ -n "$CK" ] && $PY scripts/w14_dev_functional.py --model $M --revision $REV --ckpt "$CK" || log "dev functional (A) failed"

log "B: masked pass@16 of the untrained 360M model"
$PY scripts/w16_passk.py --model $M --revision $REV --k 16 || log "pass@k failed"

log "C: dense scope mask SSD (c3_weight=1)"
run_abl smollm2-360m-instruct_ssd_s0__c3_mask=1 --c3-mask --c3-weight 1.0
$PY scripts/w06_r3_conditional.py || log "r3 conditional failed"
log "FINAL QUEUE DONE"
