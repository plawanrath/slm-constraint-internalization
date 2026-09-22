#!/usr/bin/env bash
# Second queue tail: re-inserts the two steps that failed in w21 (interface control, gold-in-group pass@16) ahead of
# the 1.7B full fine-tune, then continues w21's remaining steps (C, G, D, S). Takes over when w21 reaches step C.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M360=HuggingFaceTB/SmolLM2-360M-Instruct; R360=a10cc1512eabd3dde888204e902eca88bddb4951
M17=HuggingFaceTB/SmolLM2-1.7B-Instruct
COMMON="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25"
CK=models/ckpt
log() { echo "[tail25] $(date '+%m-%d %H:%M') $*"; }
abl360() {
  local TAG=$1; shift
  [ -f "results/w14_ablations/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd $COMMON --lr 2e-5 --out-dir results/w14_ablations --tag "$TAG" "$@"
  [ -d "results/w14_ablations/ckpt/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model $M360 --ckpt "results/w14_ablations/ckpt/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
  $PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"
}

log "waiting for w21 step C"
until grep -q "^\[tail21\].*C:" logs/w21_queue_tail.log; do sleep 120; done
pkill -f "w21_queue_tail.sh" || true; sleep 3; pkill -f "scripts/w03_train.py" || true; sleep 5
log "w21 stopped at step C; tail2 takes over"

log "I: interface-given control (rerun)"
$PY scripts/w20_interface_given.py --model $M360 --revision $R360 --cell base --cell $CK/smollm2-360m-instruct_ssd_s0/best \
  --cell $CK/smollm2-360m-instruct_rft_s0/best --cell $CK/smollm2-360m-instruct_sft_s0/best || log "interface control failed"

log "A2: pass@16 for the gold-in-group student (rerun) + analyses"
G=$(ls -d results/w14_ablations/ckpt/smollm2-360m-instruct_ssd_s0__gold_in_group=1/best $CK/smollm2-360m-instruct_ssd_s0__gold_in_group=1/best 2>/dev/null | head -1)
[ -n "$G" ] && $PY scripts/w16_passk.py --model $M360 --revision $R360 --ckpt "$G" --tag smollm2-360m-instruct_ssd_s0__gold_in_group=1 --k 16 || log "passk gold failed"
$PY scripts/w19_passk_analysis.py || log "passk analysis failed"

log "C: full fine-tuning SSD at 1.7B"
TAG=smollm2-1.7b-instruct_ssd_s0__full_ft
[ -f "results/w03_train/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M17 --method ssd --seed 0 $COMMON --lr 2e-5 --micro-batch 4 --tag "$TAG"
[ -d "$CK/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model $M17 --ckpt "$CK/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"

log "G: group size 16, functional reward, same rollout budget"
TAG=smollm2-360m-instruct_ssd_s0__group=16
[ -f "results/w14_ablations/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd --seed 0 --epochs 2 --n-train 500 --group 16 --batch-prompts 8 --eval-every 25 --lr 2e-5 --reward functional --out-dir results/w14_ablations --tag "$TAG"
[ -d "results/w14_ablations/ckpt/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model $M360 --ckpt "results/w14_ablations/ckpt/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"

log "D: KL-to-unmasked-base control at 1.7B"
TAG=smollm2-1.7b-instruct_ssd_s0__kl_base=1.0
[ -f "results/w03_train/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M17 --method ssd --seed 0 $COMMON --lr 1e-4 --lora --lora-scale 2.0 --beta 1.0 --no-teacher-masked --tag "$TAG"
[ -d "$CK/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model $M17 --ckpt "$CK/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"
$PY scripts/w12_portability.py --tags "$TAG" --stablehlo || log "portability (D) failed"

log "S: seeds 1,2 for the coverage-pool and one-gold-per-group ablations"
for s in 1 2; do
  abl360 "smollm2-360m-instruct_ssd_s${s}__pool=v1c" --seed $s --train-pool data/pools/train_v1c.jsonl --dev-pool data/pools/dev_v1c.jsonl
  abl360 "smollm2-360m-instruct_ssd_s${s}__gold_in_group=1" --seed $s --gold-in-group 1
done
$PY scripts/w19_passk_analysis.py || true
echo "TAIL DONE" >> logs/w21_queue_tail.log
log "TAIL2 DONE"
