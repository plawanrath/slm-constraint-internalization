"""Per-token KL from a trained checkpoint to its untrained base, estimated on the training distribution.

For each dev prompt the checkpoint decodes one continuation without a mask, and the same tokens are scored under
both the checkpoint and the frozen base model. The estimate is

    KL(pi_trained || pi_base) ~= mean_t [ log pi_trained(x_t | x_<t) - log pi_base(x_t | x_<t) ],  x ~ pi_trained

which is the quantity the forgetting literature relates to drift on other distributions: a checkpoint that moved a
long way on its own task has a large value here, whichever training signal produced the move.

  python scripts/w32_kl_base.py --model <hf id> --ckpt <dir> --tag <name> [--n-prompts 100] [--lora]

Appends one row per checkpoint to results/w32_kl_base/kl.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir; 'base' reports 0 by definition")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--dev-pool", default="data/pools/dev_v1.jsonl")
    ap.add_argument("--out", default="results/w32_kl_base/kl.json")
    args = ap.parse_args()

    from sci.eval.generate import Generator, build_prompt
    from sci.train.run import load_ckpt
    from sci.train.spine import collect_rollouts_batched, seq_logprob_per_token, teacher_logprobs

    rows = [json.loads(l) for l in (REPO / args.dev_pool).read_text().splitlines() if l.strip()][: args.n_prompts]
    t0 = time.perf_counter()

    # the trained policy proposes; both models score the same tokens
    gen = Generator(args.model, revision=args.revision)
    if args.ckpt != "base":
        load_ckpt(gen, REPO / args.ckpt)
    prompts = [(i, build_prompt(gen.tokenizer, r["nl"], r.get("dialect", "arith+func"))) for i, r in enumerate(rows)]
    ros = collect_rollouts_batched(gen, prompts, 1, constraint="none", temp=0.0,
                                   max_tokens=args.max_tokens, seed=0, with_masks=False, reward_kind="verify")
    lp_trained = [np.asarray(seq_logprob_per_token(gen.model, ro)) for ro in ros]
    del gen

    base = Generator(args.model, revision=args.revision)   # frozen, no checkpoint loaded
    V = base.model.args.vocab_size if hasattr(base.model, "args") else len(base.tokenizer)
    lp_base = [np.asarray(teacher_logprobs(base.model, ro, V, masked=False)) for ro in ros]
    del base

    per_prompt = [float(np.mean(a - b)) for a, b in zip(lp_trained, lp_base) if len(a) and len(a) == len(b)]
    arr = np.asarray(per_prompt)
    rng = np.random.default_rng(0)
    boots = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(2000)] if len(arr) else [0.0]
    out_p = REPO / args.out
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    out[args.tag] = {"model": args.model, "ckpt": args.ckpt, "n_prompts": len(per_prompt),
                     "kl_per_token": round(float(arr.mean()) if len(arr) else 0.0, 4),
                     "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                     "mean_tokens": round(float(np.mean([len(a) for a in lp_trained])), 1),
                     "elapsed_s": round(time.perf_counter() - t0, 1)}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=1))
    print(json.dumps({args.tag: out[args.tag]}, indent=1))


if __name__ == "__main__":
    main()
