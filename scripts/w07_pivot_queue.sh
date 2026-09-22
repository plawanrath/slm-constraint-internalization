#!/usr/bin/env bash
# Pivot queue: runs after the gate block, one GPU job at a time.
#   nohup caffeinate -dims ./scripts/w07_pivot_queue.sh >> logs/w07_pivot_queue.log 2>&1 &
# Every step is idempotent (skips when its summary/result exists). Order:
#   0. wait for "[campaign] GATE SCALES DONE", stop the campaign's continuation
#   1. RFT reruns with kept-fraction step scaling at 360M and 1.7B (old runs renamed __rft_unscaled)
#   2. functional-reward ablation at 360M (claim 1)
#   3. seeds: 360M SSD/RFT/SFT × {1,2}, 1.7B SSD × {1}
#   4. coverage ablation: 360M SSD on pool v1c (claim 2)   [needs data/pools/train_v1c.jsonl]
#   5. portability eval: StableHLO + LLVM-Spec-60 for the SSD s0 checkpoints and bases (claim 2)
#   6. synthetic scope sweep 135M/360M × D {2,8,32}          (claim 3; first to cut)
#   7. probes 360M base vs SSD s0
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
L=logs/w04_campaign.log
COMMON="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25"
log() { echo "[pivot] $(date '+%m-%d %H:%M') $*"; }

log "waiting for the gate block"
until grep -q "GATE SCALES DONE" "$L"; do sleep 300; done
pkill -f "w04_campaign_all.sh" || true; pkill -f "scripts/w04_campaign.py" || true; sleep 5
pkill -f "scripts/w03_train.py" || true; sleep 5
log "gate block done; campaign continuation stopped"

# 1. RFT reruns with kept-fraction step scaling
for tag in smollm2-360m-instruct_rft_s0 smollm2-1.7b-instruct_rft_s0; do
  if [ -f "results/w03_train/$tag/summary.json" ] && [ ! -d "results/w03_train/${tag}__rft_unscaled" ] && ! grep -q '"rft_scale_by_kept": true' "results/w03_train/$tag/summary.json"; then
    log "renaming unscaled $tag"
    mv "results/w03_train/$tag" "results/w03_train/${tag}__rft_unscaled"
    [ -d "models/ckpt/$tag" ] && mv "models/ckpt/$tag" "models/ckpt/${tag}__rft_unscaled"
    [ -d "results/w04_functional/$tag" ] && mv "results/w04_functional/$tag" "results/w04_functional/${tag}__rft_unscaled"
    $PY - "$tag" <<'EOF'
import json, sys
from pathlib import Path
tag = sys.argv[1]; new = tag + "__rft_unscaled"
lad = Path("results/w04_residuals/ladder.jsonl"); rows = [json.loads(l) for l in lad.read_text().splitlines() if l.strip()]
for r in rows:
    if r["model"] == tag: r["model"] = new
lad.write_text("".join(json.dumps(r) + "\n" for r in rows))
for f in ("results/w04_residuals/residuals.json", "results/w04_functional/functional_summary.json"):
    p = Path(f); d = json.loads(p.read_text()); d = {(new if k == tag else k): v for k, v in d.items()}; p.write_text(json.dumps(d, indent=2))
EOF
  fi
done
log "step 1: RFT reruns"
$PY scripts/w04_campaign.py --models SmolLM2-360M-Instruct,SmolLM2-1.7B-Instruct --methods rft --seeds 0 $COMMON

# 2. functional-reward ablation at 360M
log "step 2: functional-reward ablation"
TAG=smollm2-360m-instruct_ssd_s0__reward=functional
if ! grep -q 'kind.*functional' sci/train/spine.py; then
  log "step 2 skipped: reward kind 'functional' not implemented in sci/train/spine.py yet (rerun this script later)"
elif [ ! -f "results/w14_ablations/$TAG/summary.json" ]; then
  $PY scripts/w03_train.py --model HuggingFaceTB/SmolLM2-360M-Instruct --revision a10cc1512eabd3dde888204e902eca88bddb4951 \
    --method ssd --seed 0 $COMMON --lr 2e-5 --reward functional --out-dir results/w14_ablations --tag "$TAG"
fi
[ -d "results/w14_ablations/ckpt/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model HuggingFaceTB/SmolLM2-360M-Instruct --ckpt "results/w14_ablations/ckpt/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
[ -d "models/ckpt/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model HuggingFaceTB/SmolLM2-360M-Instruct --ckpt "models/ckpt/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional

# 3. seeds
log "step 3: seeds"
$PY scripts/w04_campaign.py --models SmolLM2-360M-Instruct --methods ssd,rft,sft --seeds 1,2 $COMMON
$PY scripts/w04_campaign.py --models SmolLM2-1.7B-Instruct --methods ssd --seeds 1 $COMMON
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional

# 4. coverage ablation
log "step 4: coverage ablation"
TAG=smollm2-360m-instruct_ssd_s0__pool=v1c
if [ -f data/pools/train_v1c.jsonl ]; then
  if [ ! -f "results/w14_ablations/$TAG/summary.json" ]; then
    $PY scripts/w03_train.py --model HuggingFaceTB/SmolLM2-360M-Instruct --revision a10cc1512eabd3dde888204e902eca88bddb4951 \
      --method ssd --seed 0 $COMMON --lr 2e-5 --train-pool data/pools/train_v1c.jsonl --dev-pool data/pools/dev_v1c.jsonl --out-dir results/w14_ablations --tag "$TAG"
  fi
  for c in "results/w14_ablations/ckpt/$TAG/best" "models/ckpt/$TAG/best"; do
    [ -d "$c" ] && $PY scripts/w04_eval_ckpt.py --model HuggingFaceTB/SmolLM2-360M-Instruct --ckpt "$c" --tag "$TAG" --method ssd --train-seed 0
  done
  $PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional
else
  log "step 4 skipped: data/pools/train_v1c.jsonl missing"
fi

# 5. portability eval
log "step 5: portability"
$PY scripts/w12_portability.py --tags smollm2-360m-instruct_ssd_s0,smollm2-1.7b-instruct_ssd_s0,smollm3-3b_ssd_s0,gemma-3-1b-it_ssd_s0 \
  --base HuggingFaceTB/SmolLM2-360M-Instruct --base HuggingFaceTB/SmolLM2-1.7B-Instruct --base HuggingFaceTB/SmolLM3-3B --base google/gemma-3-1b-it --stablehlo || log "portability failed"

# 6. synthetic sweep
log "step 6: synthetic sweep"
$PY scripts/w08_synth_law.py --models SmolLM2-135M-Instruct,SmolLM2-360M-Instruct --D 2,8,32 --seeds 0 --epochs 1 || log "synthetic sweep failed"

# 7. probes
log "step 7: probes"
$PY scripts/w09_probes.py --pair HuggingFaceTB/SmolLM2-360M-Instruct:base --pair HuggingFaceTB/SmolLM2-360M-Instruct:smollm2-360m-instruct_ssd_s0 --ladder results/w04_residuals/ladder.jsonl || log "probes failed"
log "PIVOT QUEUE DONE"
