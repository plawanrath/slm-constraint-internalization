"""In-distribution functional match on the dev pool (template distribution) for a base model and checkpoints.

Complements the released-pool functional table: shows whether a reward moved semantics *inside* the training
distribution even when the released pools do not move.
  python scripts/w14_dev_functional.py --model HuggingFaceTB/SmolLM2-360M-Instruct --ckpt base --ckpt models/ckpt/<tag>/best ... [--n-dev 100]
Writes results/w14_ablations/dev_functional.json (appends per checkpoint tag).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sci.eval.generate import Generator, build_prompt
from sci.train.functional_reward import functional_match
from sci.train.run import load_ckpt

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--ckpt", action="append", required=True, help="'base' or a checkpoint dir; repeatable")
    ap.add_argument("--dev-pool", default="data/pools/dev_v1.jsonl")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--out", default="results/w14_ablations/dev_functional.json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (REPO / args.dev_pool).read_text().splitlines() if l.strip()][: args.n_dev]
    out_p = REPO / args.out; out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for ck in args.ckpt:
        gen = Generator(args.model, revision=args.revision)
        tag = "base:" + args.model.split("/")[-1].lower() if ck == "base" else Path(ck).parent.name
        if ck != "base":
            load_ckpt(gen, REPO / ck)
        res = {}
        for constraint in ("none", "c1_c2"):
            items = [(i, build_prompt(gen.tokenizer, r["nl"], r["dialect"]), r["mlir"]) for i, r in enumerate(rows)]
            # sequential greedy decoding: one KV cache at a time, so this can run beside a training job
            from sci.train.spine import reward_fn
            match = verify = 0; status = {}
            for i, r in enumerate(rows):
                g = gen.generate(items[i][1], constraint=constraint, max_tokens=300)
                _, parts = reward_fn(g.text, r["mlir"], "task")
                if not parts.get("verify"):
                    status["not_verified"] = status.get("not_verified", 0) + 1; continue
                verify += 1
                m = functional_match(g.text, r["mlir"], trial=0)
                status[m["status"]] = status.get(m["status"], 0) + 1
                match += int(m["match"])
            res[constraint] = {"n": len(rows), "verify": verify, "match": match, "match_rate": round(match / len(rows), 3), "status": status}
        out[tag] = res
        print(tag, json.dumps(res))
        out_p.parent.mkdir(parents=True, exist_ok=True); out_p.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
