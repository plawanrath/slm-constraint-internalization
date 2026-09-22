#!/usr/bin/env bash
# Second-target replication on textual LLVM IR.
#   nohup caffeinate -dims ./scripts/w23_llvm_queue.sh >> logs/w23_llvm_queue.log 2>&1 &
#   1. build the LLVM IR pool by lowering the template golds (scripts/w22_llvm_pool.py)
#   2. SSD and SFT-gold at 360M on that pool (--task llvm), same budget as the MLIR runs
#   3. LLVM-Spec-60 ladder + lli functional scoring for both checkpoints (scripts/w12_portability.py; base rows exist)
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M=HuggingFaceTB/SmolLM2-360M-Instruct; REV=a10cc1512eabd3dde888204e902eca88bddb4951
COMMON="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25 --lr 2e-5 --task llvm --train-pool data/pools/train_llvm.jsonl --dev-pool data/pools/dev_llvm.jsonl"
log() { echo "[llvm] $(date '+%m-%d %H:%M') $*"; }

log "1: pool"
[ -f data/pools/train_llvm.jsonl ] || $PY scripts/w22_llvm_pool.py --n-train 2000 --n-dev 100
N=$(wc -l < data/pools/train_llvm.jsonl); log "train rows: $N"
[ "$N" -ge 1000 ] || { log "pool too small ($N < 1000); stopping"; exit 0; }

for METHOD in ssd sft; do
  TAG="smollm2-360m-instruct_${METHOD}_s0__task=llvm"
  log "2: $METHOD"
  [ -f "results/w03_train/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M --revision $REV --method $METHOD --seed 0 $COMMON --tag "$TAG"
done
log "3: LLVM-Spec-60 evaluation"
$PY scripts/w12_portability.py --tags smollm2-360m-instruct_ssd_s0__task=llvm,smollm2-360m-instruct_sft_s0__task=llvm || log "portability failed"
log "LLVM QUEUE DONE"
