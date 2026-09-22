#!/usr/bin/env bash
# Rerun of final-queue step B (masked pass@16) after the dense-mask step, once the few-shot dialect key fix is in.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
log() { echo "[final-tail] $(date '+%m-%d %H:%M') $*"; }
log "waiting for the final queue"
until grep -q "FINAL QUEUE DONE" logs/w17_final_queue.log; do sleep 300; done
log "B (rerun): masked pass@16 of the untrained 360M model"
$PY scripts/w16_passk.py --model HuggingFaceTB/SmolLM2-360M-Instruct --revision a10cc1512eabd3dde888204e902eca88bddb4951 --k 16 || log "pass@k failed"
log "FINAL TAIL DONE"
