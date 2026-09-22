#!/usr/bin/env bash
# Fourth queue tail: reruns the group-16 ablation (its first attempt deadlocked in the parallel functional reward;
# functional rewards now run sequentially) after tail3's KL control, then the ablation seeds.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M360=HuggingFaceTB/SmolLM2-360M-Instruct; R360=a10cc1512eabd3dde888204e902eca88bddb4951
COMMON="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25"
log() { echo "[tail27] $(date '+%m-%d %H:%M') $*"; }
ckpt_of() { ls -d "models/ckpt/$1/best" "results/w14_ablations/ckpt/$1/best" 2>/dev/null | head -1; }
eval360() { local TAG=$1 C; C=$(ckpt_of "$TAG"); [ -n "$C" ] && $PY scripts/w04_eval_ckpt.py --model $M360 --ckpt "$C" --tag "$TAG" --method ssd --train-seed "${2:-0}"; $PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"; }
abl360() { local TAG=$1; shift; [ -f "results/w14_ablations/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd $COMMON --lr 2e-5 --out-dir results/w14_ablations --tag "$TAG" "$@"; eval360 "$TAG"; }

log "waiting for tail3 step S"
until grep -q "^\[tail26\].*S:" logs/w26_queue_tail3.log; do sleep 120; done
pkill -f "w26_queue_tail3.sh" || true; sleep 3; pkill -f "scripts/w03_train.py" || true; sleep 5
log "tail3 stopped at step S; tail4 takes over"

log "G: group size 16, functional reward (rerun, sequential rewards)"
TAG=smollm2-360m-instruct_ssd_s0__group=16
D=results/w14_ablations/$TAG; [ -f "$D/summary.json" ] || { [ -d "$D" ] && mv "$D" "${D}__hung_attempt"; }
$PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd --seed 0 --epochs 2 --n-train 500 --group 16 --batch-prompts 8 --eval-every 25 --lr 2e-5 --reward functional --out-dir results/w14_ablations --tag "$TAG"
eval360 "$TAG"

log "S: seeds 1,2 for the coverage-pool and one-gold-per-group ablations"
for s in 1 2; do
  abl360 "smollm2-360m-instruct_ssd_s${s}__pool=v1c" --seed $s --train-pool data/pools/train_v1c.jsonl --dev-pool data/pools/dev_v1c.jsonl
  abl360 "smollm2-360m-instruct_ssd_s${s}__gold_in_group=1" --seed $s --gold-in-group 1
done
$PY scripts/w19_passk_analysis.py || true
echo "TAIL DONE" >> logs/w21_queue_tail.log
log "TAIL4 DONE"
