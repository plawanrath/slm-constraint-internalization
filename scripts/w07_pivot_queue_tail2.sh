#!/usr/bin/env bash
# Second tail: runs after scripts/w07_pivot_queue_tail.sh prints "PIVOT QUEUE TAIL DONE".
#   E2. synthetic sweep re-evaluation with the C3 arm (checkpoints exist; decode only)
#   F. scf portability (Scf-Spec-60) for the four seed-0 SSD checkpoints and their bases
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
log() { echo "[pivot-tail2] $(date '+%m-%d %H:%M') $*"; }
log "waiting for the first tail to finish"
until grep -q "PIVOT QUEUE TAIL DONE" logs/w07_pivot_queue_tail.log; do sleep 300; done
log "E2: synthetic re-evaluation (C3 arm)"
$PY scripts/w08_synth_law.py --models SmolLM2-135M-Instruct,SmolLM2-360M-Instruct --D 2,8,32 --seeds 0 --epochs 1 --no-train --reeval || log "synthetic re-eval failed"
log "F: scf portability"
$PY scripts/w11_scf.py \
  --cell base360 HuggingFaceTB/SmolLM2-360M-Instruct \
  --cell ssd360 HuggingFaceTB/SmolLM2-360M-Instruct models/ckpt/smollm2-360m-instruct_ssd_s0/best \
  --cell base1p7 HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --cell ssd1p7 HuggingFaceTB/SmolLM2-1.7B-Instruct models/ckpt/smollm2-1.7b-instruct_ssd_s0/best \
  --cell base3b HuggingFaceTB/SmolLM3-3B \
  --cell ssd3b HuggingFaceTB/SmolLM3-3B models/ckpt/smollm3-3b_ssd_s0/best \
  --cell baseg1b google/gemma-3-1b-it \
  --cell ssdg1b google/gemma-3-1b-it models/ckpt/gemma-3-1b-it_ssd_s0/best || log "scf failed"
log "PIVOT QUEUE TAIL2 DONE"
