"""Weeks 3–6 campaign: train (model × method × seed) then evaluate each best checkpoint.
  python scripts/w04_campaign.py --models SmolLM2-135M-Instruct,SmolLM2-360M-Instruct --methods ssd,rft,grpo,sft --seeds 0,1,2
Skips runs whose summary.json already exists; skips evals whose tag is already in the residual ladder.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv/bin/python")
REPOS = {"SmolLM2-135M-Instruct": "HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
         "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM3-3B": "HuggingFaceTB/SmolLM3-3B",
         "gemma-3-270m-it": "google/gemma-3-270m-it", "gemma-3-1b-it": "google/gemma-3-1b-it"}
LORA = {"SmolLM2-1.7B-Instruct", "SmolLM3-3B", "gemma-3-1b-it"}


def revision_for(repo: str) -> str:
    for line in (REPO / "scripts/env/models.lock.txt").read_text().splitlines():
        if line.startswith(repo + "@"):
            return line.split("@", 1)[1].strip()
    raise KeyError(repo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--methods", default="ssd,rft,grpo,sft")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--batch-prompts", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-dir", default="results/w03_train")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    for m in args.models.split(","):
        repo = REPOS[m]; rev = revision_for(repo); lora = m in LORA
        for method in args.methods.split(","):
            for seed in [int(s) for s in args.seeds.split(",")]:
                tag = f"{m.lower()}_{method}_s{seed}"
                run_dir = REPO / args.out_dir / tag
                if not (run_dir / "summary.json").exists():
                    lr = args.lr or (1e-4 if lora else 2e-5)
                    cmd = [PY, "scripts/w03_train.py", "--model", repo, "--revision", rev, "--method", method, "--seed", str(seed),
                           "--epochs", str(args.epochs), "--group", str(args.group), "--batch-prompts", str(args.batch_prompts),
                           "--eval-every", str(args.eval_every), "--lr", str(lr), "--out-dir", args.out_dir, "--tag", tag]
                    if lora:
                        cmd.append("--lora")
                    if args.n_train:
                        cmd += ["--n-train", str(args.n_train)]
                    print("[campaign] train", tag, file=sys.stderr)
                    subprocess.run(cmd, check=False)
                ckpt = REPO / "models" / "ckpt" / tag / "best"
                if args.no_eval or not ckpt.exists():
                    continue
                print("[campaign] eval", tag, file=sys.stderr)
                subprocess.run([PY, "scripts/w04_eval_ckpt.py", "--model", repo, "--ckpt", str(ckpt), "--tag", tag,
                                "--method", method, "--train-seed", str(seed)], check=False)


if __name__ == "__main__":
    main()
