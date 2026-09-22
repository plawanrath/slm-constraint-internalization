#!/usr/bin/env bash
# Tail of the pivot queue: takes over from w07_pivot_queue.sh when it reaches step 6, so the
# 1.7B functional-reward replication runs before the synthetic sweep. Idempotent like the main queue.
#   A. functional-reward ablation at 1.7B (LoRA scale 2, lr 1e-4) + eval + functional scoring
#   B. synthetic scope sweep 135M/360M × D {2,8,32}
#   C. probes 360M base vs SSD s0
#   D. 360M SSD seeds 1,2 rerun (first attempt was killed by an out-of-memory event)
#   E. Gemma-270M SSD + RFT s0 (deferred from the campaign)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
COMMON="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25"
log() { echo "[pivot-tail] $(date '+%m-%d %H:%M') $*"; }

log "waiting for the main queue to reach step 6"
until grep -q "step 6: synthetic sweep" logs/w07_pivot_queue.log; do sleep 120; done
pkill -f "w07_pivot_queue.sh" || true; sleep 3
pkill -f "scripts/w08_synth_law.py" || true; pkill -f "scripts/w03_train.py" || true; sleep 5
log "main queue stopped at step 6; tail takes over"

log "A: functional-reward ablation at 1.7B"
TAG=smollm2-1.7b-instruct_ssd_s0__reward=functional
if [ ! -f "results/w14_ablations/$TAG/summary.json" ]; then
  $PY scripts/w03_train.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --revision 31b70e2e869a7173562077fd711b654946d38674 \
    --method ssd --seed 0 $COMMON --lr 1e-4 --lora --lora-scale 2.0 --reward functional --out-dir results/w14_ablations --tag "$TAG"
fi
[ -d "models/ckpt/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --ckpt "models/ckpt/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"
$PY scripts/w14_dev_functional.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --revision 31b70e2e869a7173562077fd711b654946d38674 --ckpt base
$PY scripts/w14_dev_functional.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --revision 31b70e2e869a7173562077fd711b654946d38674 --ckpt models/ckpt/smollm2-1.7b-instruct_ssd_s0/best
$PY scripts/w14_dev_functional.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --revision 31b70e2e869a7173562077fd711b654946d38674 --ckpt "models/ckpt/$TAG/best"

log "B: synthetic sweep"
$PY scripts/w08_synth_law.py --models SmolLM2-135M-Instruct,SmolLM2-360M-Instruct --D 2,8,32 --seeds 0 --epochs 1 || log "synthetic sweep failed"

log "C: probes"
$PY scripts/w09_probes.py --pair HuggingFaceTB/SmolLM2-360M-Instruct:base --pair HuggingFaceTB/SmolLM2-360M-Instruct:smollm2-360m-instruct_ssd_s0 --ladder results/w04_residuals/ladder.jsonl || log "probes failed"

log "D: 360M SSD seeds 1,2 rerun"
$PY scripts/w04_campaign.py --models SmolLM2-360M-Instruct --methods ssd --seeds 1,2 $COMMON
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional

log "E: Gemma-270M SSD + RFT"
$PY scripts/w04_campaign.py --models gemma-3-270m-it --methods ssd,rft --seeds 0 $COMMON
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional
log "PIVOT QUEUE TAIL DONE"
