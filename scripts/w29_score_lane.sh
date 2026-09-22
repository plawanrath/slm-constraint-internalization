#!/usr/bin/env bash
# CPU scoring lane: scores pass@k samples as the GPU lane produces them (one scorer at a time, per-cell mkdir lock
# shared with scripts/w29_gpu_lane.sh), then refreshes the pass@k analyses.
#   nohup ./scripts/w29_score_lane.sh >> logs/w29_score_lane.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M360=HuggingFaceTB/SmolLM2-360M-Instruct; M17=HuggingFaceTB/SmolLM2-1.7B-Instruct; M3B=HuggingFaceTB/SmolLM3-3B
log() { echo "[w29-score] $(date '+%m-%d %H:%M') $*"; }
lock() { until mkdir logs/.scorer.lock 2>/dev/null; do sleep 30; done; }
unlock() { rmdir logs/.scorer.lock 2>/dev/null || true; }
score() { $PY scripts/w16_passk.py --model "$1" --k "$2" --gate-only --score-only --tag "$3" --lock-dir logs/.scorer.lock; }
waitfor() { until [ -f "results/w16_passk/$1/GEN_DONE" ] && [ "$(cat results/w16_passk/$1/GEN_DONE)" -ge "$2" ]; do sleep 300; done; }

log "waiting for the base 360M k=256 samples"
waitfor smollm2-360m-instruct 256; log "scoring base 360M k=256 (240 new cells)"
score $M360 256 smollm2-360m-instruct
$PY scripts/w19_passk_analysis.py || true
log "waiting for the 1.7B/3B k=8 samples"
until grep -q "STAGE3 GEN DONE" logs/w29_gpu_lane.log; do sleep 300; done
for T in smollm2-1.7b-instruct smollm2-1.7b-instruct_ssd_s0 smollm2-1.7b-instruct_ssd_s1 smollm2-1.7b-instruct_rft_s0; do score $M17 8 $T; done
for T in smollm3-3b smollm3-3b_ssd_s0 smollm3-3b_rft_s0; do score $M3B 8 $T; done
log "waiting for the 1.7B base k=16 samples"
until grep -q "STAGE4 GEN DONE" logs/w29_gpu_lane.log; do sleep 300; done
score $M17 16 smollm2-1.7b-instruct
$PY scripts/w19_passk_analysis.py || true
log "W29 SCORE LANE DONE"
