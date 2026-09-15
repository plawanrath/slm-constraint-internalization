"""Table 1: zero-training constraint ladder for every pinned model on both frozen pools.

Usage: python scripts/w02_table1.py [model-substr ...] [--limit N]
Writes results/w02_table1/ladder.jsonl (per row) and summary.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sci.eval.ladder import CONSTRAINTS, run_model, summarize

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/w02_table1"


def pinned() -> list[tuple[str, str]]:
    out = []
    for line in (REPO / "scripts/env/models.lock.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            repo, sha = line.strip().split("@", 1); out.append((repo, sha))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pools", default="arith_func_200,linalg_125")
    args = ap.parse_args()
    for repo, sha in pinned():
        if args.only and not any(o.lower() in repo.lower() for o in args.only):
            continue
        tag = repo.split("/")[-1].lower()
        print(f"[table1] {tag}@{sha[:8]}", file=sys.stderr)
        run_model(repo, sha, tag, args.pools.split(","), CONSTRAINTS, OUT / "ladder.jsonl", limit=args.limit)
        summarize(OUT / "ladder.jsonl", OUT / "summary.json")
    summarize(OUT / "ladder.jsonl", OUT / "summary.json")


if __name__ == "__main__":
    main()
