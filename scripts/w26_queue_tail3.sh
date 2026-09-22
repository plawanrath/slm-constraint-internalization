#!/usr/bin/env bash
# Third queue tail: evaluates the SFT-gold -> SSD checkpoint that tail2 skipped (its checkpoint lives under models/ckpt,
# not results/w14_ablations/ckpt), then continues tail2's remaining steps (G, D, S) with checkpoint lookup in both places.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M360=HuggingFaceTB/SmolLM2-360M-Instruct; R360=a10cc1512eabd3dde888204e902eca88bddb4951
M17=HuggingFaceTB/SmolLM2-1.7B-Instruct
COMMON="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25"
log() { echo "[tail26] $(date '+%m-%d %H:%M') $*"; }
ckpt_of() { ls -d "models/ckpt/$1/best" "results/w14_ablations/ckpt/$1/best" 2>/dev/null | head -1; }
eval360() {  # TAG: ladder eval + functional scoring (+ dev functional)
  local TAG=$1 C; C=$(ckpt_of "$TAG")
  [ -n "$C" ] && $PY scripts/w04_eval_ckpt.py --model $M360 --ckpt "$C" --tag "$TAG" --method ssd --train-seed "${2:-0}"
  $PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"
}
abl360() {  # TAG extra-flags...
  local TAG=$1; shift
  [ -f "results/w14_ablations/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd $COMMON --lr 2e-5 --out-dir results/w14_ablations --tag "$TAG" "$@"
  eval360 "$TAG"
}

log "waiting for tail2 step G"
until grep -q "^\[tail25\].*G:" logs/w25_queue_tail2.log; do sleep 120; done
pkill -f "w25_queue_tail2.sh" || true; sleep 3; pkill -f "scripts/w03_train.py" || true; sleep 5
log "tail2 stopped at step G; tail3 takes over"

log "B2: evaluate SFT-gold -> SSD"
TAG=smollm2-360m-instruct_ssd_s0__init=sft_s0
eval360 "$TAG"
$PY scripts/w14_dev_functional.py --model $M360 --revision $R360 --ckpt "$(ckpt_of $TAG)" || log "dev functional (B2) failed"

log "G: group size 16, functional reward, same rollout budget"
TAG=smollm2-360m-instruct_ssd_s0__group=16
[ -f "results/w14_ablations/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd --seed 0 --epochs 2 --n-train 500 --group 16 --batch-prompts 8 --eval-every 25 --lr 2e-5 --reward functional --out-dir results/w14_ablations --tag "$TAG"
eval360 "$TAG"

log "D: KL-to-unmasked-base control at 1.7B"
TAG=smollm2-1.7b-instruct_ssd_s0__kl_base=1.0
[ -f "results/w03_train/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M17 --method ssd --seed 0 $COMMON --lr 1e-4 --lora --lora-scale 2.0 --beta 1.0 --no-teacher-masked --tag "$TAG"
C=$(ckpt_of "$TAG"); [ -n "$C" ] && $PY scripts/w04_eval_ckpt.py --model $M17 --ckpt "$C" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"
$PY scripts/w12_portability.py --tags "$TAG" --stablehlo || log "portability (D) failed"

log "S: seeds 1,2 for the coverage-pool and one-gold-per-group ablations"
for s in 1 2; do
  abl360 "smollm2-360m-instruct_ssd_s${s}__pool=v1c" --seed $s --train-pool data/pools/train_v1c.jsonl --dev-pool data/pools/dev_v1c.jsonl
  abl360 "smollm2-360m-instruct_ssd_s${s}__gold_in_group=1" --seed $s --gold-in-group 1
done
$PY scripts/w19_passk_analysis.py || true
echo "TAIL DONE" >> logs/w21_queue_tail.log
log "TAIL3 DONE"
