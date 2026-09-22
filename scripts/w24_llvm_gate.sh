#!/usr/bin/env bash
# Starts the LLVM IR replication as soon as the preceding queue tail finishes.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[llvm-gate] $(date '+%m-%d %H:%M') $*"; }
until grep -q "TAIL DONE" logs/w21_queue_tail.log; do sleep 120; done
log "queue tail done; starting the LLVM replication"
caffeinate -dims ./scripts/w23_llvm_queue.sh >> logs/w23_llvm_queue.log 2>&1
log "LLVM GATE DONE"
