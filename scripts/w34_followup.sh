#!/usr/bin/env bash
# Reruns the held-out-dialect transfer for the 1.7B full fine-tune, which failed earlier on a malformed --cell
# argument (the flag takes TAG MODEL CKPT_DIR as separate words, not TAG=CKPT). Waits for the experiment lane so
# only one job touches the GPU.
#   nohup caffeinate -dims ./scripts/w34_followup.sh >> logs/w34_followup.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M17=HuggingFaceTB/SmolLM2-1.7B-Instruct
T=smollm2-1.7b-instruct_ssd_s0__full_ft
log() { echo "[w34] $(date '+%m-%d %H:%M') $*"; }
log "waiting for the experiment lane to finish"
until grep -q "W31 TAIL DONE" logs/w31_tail.log 2>/dev/null; do sleep 300; done
log "running held-out StableHLO for the 1.7B full fine-tune"
$PY scripts/w10_stablehlo.py --cell "$T" "$M17" "models/ckpt/$T/best" --method ssd || log "stablehlo failed again"
log "W34 DONE"
