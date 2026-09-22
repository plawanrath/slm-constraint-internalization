"""Evaluate a trained checkpoint (or the base model) on the frozen pools under all four stacks,
append rows to results/w04_residuals/ladder.jsonl, and write per-layer residuals.
  python scripts/w04_eval_ckpt.py --model HuggingFaceTB/SmolLM2-135M-Instruct --ckpt results/w03_train/<tag>/best --tag <tag>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sci.eval.ladder import CONSTRAINTS, run_model, summarize
from sci.eval.residual import residuals_from_jsonl

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/w04_residuals"


def revision_for(repo: str) -> str | None:
    for line in (REPO / "scripts/env/models.lock.txt").read_text().splitlines():
        if line.startswith(repo + "@"):
            return line.split("@", 1)[1].strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", default="none")
    ap.add_argument("--train-seed", type=int, default=None)
    ap.add_argument("--pools", default="arith_func_200,linalg_125")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--constraints", default=None, help="comma list of stacks to decode (default all four); a partial stack yields rates for the stacks decoded and residuals only for the layers whose stack is present")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stacks = tuple(args.constraints.split(",")) if args.constraints else CONSTRAINTS
    run_model(args.model, revision_for(args.model), args.tag, args.pools.split(","), stacks, OUT / "ladder.jsonl",
              method=args.method, ckpt=args.ckpt or "base", ckpt_dir=Path(args.ckpt) if args.ckpt else None, limit=args.limit)
    summarize(OUT / "ladder.jsonl", OUT / "summary.json")
    res = {pool: residuals_from_jsonl(OUT / "ladder.jsonl", args.tag, pool) for pool in args.pools.split(",")}
    rp = OUT / "residuals.json"
    allres = json.loads(rp.read_text()) if rp.exists() else {}
    allres[args.tag] = res
    rp.write_text(json.dumps(allres, indent=2))
    print(json.dumps({args.tag: res}, indent=1)[:3000])


if __name__ == "__main__":
    main()
