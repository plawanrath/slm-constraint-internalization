#!/usr/bin/env bash
# Third tail: after tail2, re-evaluate the synthetic sweep with a token budget that does not truncate depth-32 programs
# (the 600-token budget left 19–38 % of D32 free outputs unfinished, which the residual counted as scope failures).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
log() { echo "[pivot-tail3] $(date '+%m-%d %H:%M') $*"; }
log "waiting for tail2"
until grep -q "PIVOT QUEUE TAIL2 DONE" logs/w07_pivot_queue_tail2.log; do sleep 300; done
log "G: synthetic re-evaluation with max-tokens 1500"
cp results/w08_synth_law/sweep.jsonl results/w08_synth_law/sweep_600tok.jsonl
cp -r results/w08_synth_law/evals results/w08_synth_law/evals_600tok
$PY scripts/w08_synth_law.py --models SmolLM2-135M-Instruct,SmolLM2-360M-Instruct --D 2,8,32 --seeds 0 --epochs 1 --no-train --reeval --max-tokens 1500 || log "synthetic re-eval failed"
log "PIVOT QUEUE TAIL3 DONE"
