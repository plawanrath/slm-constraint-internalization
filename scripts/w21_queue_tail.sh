#!/usr/bin/env bash
# Re-sequenced experiment queue. Takes over from w18 when its step A (pass@k)
# finishes, then runs:
#   A2. pass@16 for the gold-in-group student (released pool) + analyses (scripts/w19_passk_analysis.py)
#   I.  interface-given control at 360M (base, SSD, RFT, SFT-gold; decode + score)
#   B.  SFT-gold -> SSD at 360M
#   C.  full fine-tuning SSD at 1.7B
#   G.  group size 16 with the functional reward at 360M (same rollout budget: 500 prompts x 16)
#   D.  KL-to-unmasked-base control at 1.7B + StableHLO portability
#   S.  seeds 1,2 for the coverage-pool and one-gold-per-group ablations
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
M360=HuggingFaceTB/SmolLM2-360M-Instruct; R360=a10cc1512eabd3dde888204e902eca88bddb4951
M17=HuggingFaceTB/SmolLM2-1.7B-Instruct
COMMON="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25"
CK=models/ckpt
log() { echo "[tail21] $(date '+%m-%d %H:%M') $*"; }

abl360() {  # TAG extra-flags...   (train + eval + score + dev functional at 360M, out-dir results/w14_ablations)
  local TAG=$1; shift
  [ -f "results/w14_ablations/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd $COMMON --lr 2e-5 --out-dir results/w14_ablations --tag "$TAG" "$@"
  [ -d "results/w14_ablations/ckpt/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model $M360 --ckpt "results/w14_ablations/ckpt/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
  $PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"
}

log "waiting for w18 step A"
until grep -q "^\[queue18\].*B:" logs/w18_experiment_queue.log; do sleep 300; done
pkill -f "w18_experiment_queue.sh" || true; sleep 3; pkill -f "scripts/w03_train.py" || true; sleep 5
log "w18 stopped after step A; tail takes over"

log "A2: pass@16 for the gold-in-group student + analyses"
$PY scripts/w16_passk.py --model $M360 --revision $R360 --ckpt results/w14_ablations/ckpt/smollm2-360m-instruct_ssd_s0__gold_in_group=1/best --k 16 || log "passk gold failed"
$PY scripts/w19_passk_analysis.py || log "passk analysis failed"

log "I: interface-given control"
$PY scripts/w20_interface_given.py --model $M360 --revision $R360 --cell base --cell $CK/smollm2-360m-instruct_ssd_s0/best \
  --cell $CK/smollm2-360m-instruct_rft_s0/best --cell $CK/smollm2-360m-instruct_sft_s0/best || log "interface control failed"

log "B: SFT-gold -> SSD at 360M"
abl360 smollm2-360m-instruct_ssd_s0__init=sft_s0 --seed 0 --init-ckpt $CK/smollm2-360m-instruct_sft_s0/best
$PY scripts/w14_dev_functional.py --model $M360 --revision $R360 --ckpt "results/w14_ablations/ckpt/smollm2-360m-instruct_ssd_s0__init=sft_s0/best" || log "dev functional (B) failed"

log "C: full fine-tuning SSD at 1.7B"
TAG=smollm2-1.7b-instruct_ssd_s0__full_ft
[ -f "results/w03_train/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M17 --method ssd --seed 0 $COMMON --lr 2e-5 --micro-batch 4 --tag "$TAG"
[ -d "$CK/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model $M17 --ckpt "$CK/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"

log "G: group size 16, functional reward, same rollout budget"
TAG=smollm2-360m-instruct_ssd_s0__group=16
[ -f "results/w14_ablations/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M360 --revision $R360 --method ssd --seed 0 --epochs 2 --n-train 500 --group 16 --batch-prompts 8 --eval-every 25 --lr 2e-5 --reward functional --out-dir results/w14_ablations --tag "$TAG"
[ -d "results/w14_ablations/ckpt/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model $M360 --ckpt "results/w14_ablations/ckpt/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"

log "D: KL-to-unmasked-base control at 1.7B"
TAG=smollm2-1.7b-instruct_ssd_s0__kl_base=1.0
[ -f "results/w03_train/$TAG/summary.json" ] || $PY scripts/w03_train.py --model $M17 --method ssd --seed 0 $COMMON --lr 1e-4 --lora --lora-scale 2.0 --beta 1.0 --no-teacher-masked --tag "$TAG"
[ -d "$CK/$TAG/best" ] && $PY scripts/w04_eval_ckpt.py --model $M17 --ckpt "$CK/$TAG/best" --tag "$TAG" --method ssd --train-seed 0
$PY scripts/w02_functional.py --ladder results/w04_residuals/ladder.jsonl --out results/w04_functional --models "$TAG"
$PY scripts/w12_portability.py --tags "$TAG" --stablehlo || log "portability (D) failed"

log "S: seeds 1,2 for the coverage-pool and one-gold-per-group ablations"
for s in 1 2; do
  abl360 "smollm2-360m-instruct_ssd_s${s}__pool=v1c" --seed $s --train-pool data/pools/train_v1c.jsonl --dev-pool data/pools/dev_v1c.jsonl
  abl360 "smollm2-360m-instruct_ssd_s${s}__gold_in_group=1" --seed $s --gold-in-group 1
done
$PY scripts/w19_passk_analysis.py || true
log "TAIL DONE"
