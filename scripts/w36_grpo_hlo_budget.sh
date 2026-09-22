#!/usr/bin/env bash
# Reruns held-out StableHLO for the unmasked GRPO student at a larger token budget. At the default 600 tokens the
# masked stacks truncated on 130 of 200 prompts, so their verify rate measured the decode cap rather than the
# student; free decoding was unaffected (unfinished 0) and is already reported.
#   nohup caffeinate -dims ./scripts/w36_grpo_hlo_budget.sh >> logs/w36_grpo_hlo_budget.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M17=HuggingFaceTB/SmolLM2-1.7B-Instruct
T=smollm2-1.7b-instruct_grpo_s0
log() { echo "[w36] $(date '+%m-%d %H:%M') $*"; }
log "rerunning held-out StableHLO for the GRPO student at 1600 tokens"
$PY scripts/w10_stablehlo.py --cell "$T" "$M17" "models/ckpt/$T/best" --method grpo \
    --sets stablehlo_held_out_200 --constraints c1,c1_c2,c1_c2_c3 --max-tokens 1600 || log "stablehlo failed"
log "W36 DONE"
