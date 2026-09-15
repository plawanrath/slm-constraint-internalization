#!/usr/bin/env bash
# Weeks 3–6 campaign, gate-first order: SSD + RFT at every framing-gate scale, then the rest.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/scripts/env/bin:$PATH"
PY=.venv/bin/python
COMMON="--epochs 2 --n-train 2000 --group 4 --batch-prompts 32 --eval-every 25"
$PY scripts/w04_campaign.py --models SmolLM2-360M-Instruct --methods ssd,rft,grpo,sft --seeds 0 $COMMON
$PY scripts/w04_campaign.py --models SmolLM2-1.7B-Instruct,gemma-3-1b-it,SmolLM3-3B --methods ssd,rft --seeds 0 $COMMON
$PY scripts/w04_campaign.py --models SmolLM2-135M-Instruct,gemma-3-270m-it --methods ssd,rft --seeds 0 $COMMON
echo "[campaign] GATE SCALES DONE"
$PY scripts/w04_campaign.py --models SmolLM2-1.7B-Instruct,gemma-3-1b-it --methods grpo,sft --seeds 0 $COMMON
$PY scripts/w04_campaign.py --models SmolLM2-135M-Instruct,gemma-3-270m-it --methods grpo,sft --seeds 0 $COMMON
$PY scripts/w04_campaign.py --models SmolLM2-360M-Instruct,SmolLM2-135M-Instruct,gemma-3-270m-it --methods ssd,rft,grpo,sft --seeds 1,2 $COMMON
$PY scripts/w04_campaign.py --models SmolLM2-1.7B-Instruct,gemma-3-1b-it --methods ssd,rft,grpo,sft --seeds 1 $COMMON
echo "[campaign] ALL DONE"
