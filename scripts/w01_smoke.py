"""Week-1 smoke: one prompt per constraint stack per pinned model.

Checks (a) each model loads and generates under the C1 and C1+C2 masks, (b) the offline
mask replay accepts every sampled token, (c) parse validity and tokens/s, (d) that the
chat-template prompt for SmolLM2 equals the literal ChatML prompt.
Writes results/w01_smoke/smoke.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from sci.constraints.parser import is_parse_valid
from sci.eval.generate import Generator, build_prompt, chatml_prompt
from sci.masks.llg import replay_allowed_sets

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "scripts/env/models.lock.txt"
NL = "Write a function that multiplies two i32 values and returns the product."
OUT = REPO / "results/w01_smoke/smoke.json"


def pinned_models() -> list[tuple[str, str]]:
    out = []
    for line in LOCK.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        repo, sha = line.split("@", 1)
        out.append((repo, sha))
    return out


def main() -> None:
    only = sys.argv[1:] or None
    results = json.loads(OUT.read_text()) if OUT.exists() else {}  # merge across partial runs
    for repo, sha in pinned_models():
        if only and not any(o in repo for o in only):
            continue
        print(f"[smoke] {repo}@{sha[:8]}", file=sys.stderr)
        t0 = time.perf_counter()
        gen = Generator(repo, revision=sha)
        load_s = time.perf_counter() - t0
        prompt = build_prompt(gen.tokenizer, NL, "arith+func")
        row = {"revision": sha, "load_s": round(load_s, 1), "vocab": gen.llg_tok.vocab_size,
               "prompt_is_literal_chatml": prompt == chatml_prompt(NL, "arith+func"), "cells": {}}
        for c in ("none", "c1", "c1_c2", "c1_c2_c3"):
            try:
                g = gen.generate(prompt, constraint=c, max_tokens=300)
                cell = {"parse_valid": is_parse_valid(g.text), "n_tokens": len(g.tokens),
                        "tok_s": round(len(g.tokens) / max(g.dt, 1e-6), 1), "dt": round(g.dt, 2),
                        "finished": g.finished, "attempts": g.attempts, "mask_errors": g.mask_errors[:2],
                        "text": g.text[:400]}
                if c in ("c1", "c1_c2"):
                    try:
                        replay_allowed_sets(gen.llg_tok, gen.grammar("mlir_gen_c1" if c == "c1" else "mlir_gen_c1c2"), g.tokens)
                        cell["replay_ok"] = True
                    except ValueError as e:
                        cell["replay_ok"] = False; cell["replay_err"] = str(e)[:200]
            except Exception as e:  # noqa: BLE001
                cell = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            row["cells"][c] = cell
            print(f"  {c}: {json.dumps({k: v for k, v in cell.items() if k != 'text'})}", file=sys.stderr)
        results[repo] = row
        del gen
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"[smoke] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
