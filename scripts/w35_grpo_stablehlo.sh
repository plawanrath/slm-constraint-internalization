#!/usr/bin/env bash
# Held-out StableHLO for the 1.7B unmasked GRPO student. This is the control for the suppression reading in the
# portability section: the student trains without the grammar mask and without the legal-mass term, so if it also
# loses StableHLO validity, the loss is not attributable to that term. The earlier lane skipped it after the
# malformed --cell argument. Waits for the preceding job so only one job touches the GPU.
#   nohup caffeinate -dims ./scripts/w35_grpo_stablehlo.sh >> logs/w35_grpo_stablehlo.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M17=HuggingFaceTB/SmolLM2-1.7B-Instruct
T=smollm2-1.7b-instruct_grpo_s0
log() { echo "[w35] $(date '+%m-%d %H:%M') $*"; }
log "waiting for the StableHLO rerun to finish"
until grep -q "W34 DONE" logs/w34_followup.log 2>/dev/null; do sleep 120; done
log "running held-out StableHLO for the unmasked GRPO student"
$PY scripts/w10_stablehlo.py --cell "$T" "$M17" "models/ckpt/$T/best" --method grpo || log "stablehlo failed"
log "W35 DONE"
