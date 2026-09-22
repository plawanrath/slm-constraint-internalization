#!/usr/bin/env bash
# Pass@k and gold-supervision experiments:
#   A. pass@16: released pool for SSD s0 and SFT-gold s0 (harness); dev pool for base, SSD s0, SFT-gold s0 (one-trial match)
#   B. SFT-gold -> SSD sequenced recipe at 360M (init from the SFT-gold s0 checkpoint)
#   C. full fine-tuning SSD at 1.7B (no adapter), the adapter-regime control
#   D. forgetting control at 1.7B: SSD with a KL term to the unmasked base (beta 1.0), then StableHLO portability
# Idempotent; one GPU job at a time; the functional scorer runs alone.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M360=HuggingFaceTB/SmolLM2-360M-Instruct; R360=a10cc1512eabd3dde888204e902eca88bddb4951
M17=HuggingFaceTB/SmolLM2-1.7B-Instruct
COMMON="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25"
log() { echo "[queue18] $(date '+%m-%d %H:%M') $*"; }

log "A: pass@16 curves"
$PY scripts/w16_passk.py --model $M360 --revision $R360 --ckpt models/ckpt/smollm2-360m-instruct_ssd_s0/best --k 16 || log "passk ssd failed"
$PY scripts/w16_passk.py --model $M360 --revision $R360 --ckpt models/ckpt/smollm2-360m-instruct_sft_s0/best --k 16 || log "passk sft failed"
$PY scripts/w16_passk.py --model $M360 --revision $R360 --dev --k 16 || log "passk dev base failed"
$PY scripts/w16_passk.py --model $M360 --revision $R360 --ckpt models/ckpt/smollm2-360m-instruct_ssd_s0/best --dev --k 16 || log "passk dev ssd failed"
$PY scripts/w16_passk.py --model $M360 --revision $R360 --ckpt models/ckpt/smollm2-360m-instruct_sft_s0/best --dev --k 16 || log "passk dev sft failed"

log "B: SFT-gold -> SSD at 360M"
TAG=smollm2-360m-instruct_ssd_s0__init=sft_s0
if [ ! -f "results/w14_ablations/$TAG/summary.json" ]; then
  $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd --seed 0 $COMMON --lr 2e-5 --init-ckpt models/ckpt/smollm2-360m-instruct_sft_s0/best --out-dir results/w14_ablations --tag "$TAG"
fi
[ -d "results/w14_ablations/ckpt/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model $M360 --ckpt "results/w14_ablations/ckpt/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"
$PY scripts/w14_dev_functional.py --model $M360 --revision $R360 --ckpt "results/w14_ablations/ckpt/$TAG/best" || log "dev functional (B) failed"

log "C: full fine-tuning SSD at 1.7B"
TAG=smollm2-1.7b-instruct_ssd_s0__full_ft
if [ ! -f "results/w03_train/$TAG/summary.json" ]; then
  $PY scripts/w03_train.py --model $M17 --method ssd --seed 0 $COMMON --lr 2e-5 --micro-batch 4 --tag "$TAG"
fi
[ -d "models/ckpt/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model $M17 --ckpt "models/ckpt/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"

log "D: forgetting control at 1.7B (KL to the unmasked base, beta 1.0)"
TAG=smollm2-1.7b-instruct_ssd_s0__kl_base=1.0
if [ ! -f "results/w03_train/$TAG/summary.json" ]; then
  $PY scripts/w03_train.py --model $M17 --method ssd --seed 0 $COMMON --lr 1e-4 --lora --lora-scale 2.0 --beta 1.0 --no-teacher-masked --tag "$TAG"
fi
[ -d "models/ckpt/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model $M17 --ckpt "models/ckpt/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"
$PY scripts/w12_portability.py --tags "$TAG" --stablehlo || log "portability (D) failed"
log "QUEUE18 DONE"
