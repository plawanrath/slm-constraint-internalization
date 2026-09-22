#!/usr/bin/env bash
# Drains the functional-scoring queue: one tag per line in logs/score_queue.txt, scored with scripts/w02_functional.py
# under the shared scorer lock so only one harness runs at a time. Exits when logs/score_queue.done exists and the
# queue is empty, so the producing queue controls its lifetime.
#   nohup ./scripts/w30_scorer.sh >> logs/w30_scorer.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
Q=logs/score_queue.txt; LOCK=logs/.scorer.lock
log() { echo "[w30-score] $(date '+%m-%d %H:%M') $*"; }
touch "$Q"
while :; do
  TAG=$(head -1 "$Q" 2>/dev/null)
  if [ -z "$TAG" ]; then
    [ -f logs/score_queue.done ] && { log "queue drained and closed"; break; }
    sleep 60; continue
  fi
  until mkdir "$LOCK" 2>/dev/null; do sleep 20; done
  log "scoring $TAG"
  $PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG" || log "scoring $TAG failed"
  rmdir "$LOCK" 2>/dev/null || true
  sed -i '' '1d' "$Q"
  log "done $TAG"
  sleep 30   # let another scorer take the lock
done
log "W30 SCORER DONE"
