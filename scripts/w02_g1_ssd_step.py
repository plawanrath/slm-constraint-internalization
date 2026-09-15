"""G1 gate: SSD runs end to end at 135M on 64 prompts; loss decreases; no NaN.
Writes results/w02_g1/summary.json."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

from sci.eval.generate import Generator, build_prompt
from sci.train.spine import collect_rollouts, ssd_step

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--pool", default="data/pools/train_v1.jsonl")
    ap.add_argument("--out", default="results/w02_g1/summary.json")
    args = ap.parse_args()
    pool_path = REPO / args.pool
    if pool_path.exists():
        rows = [json.loads(l) for l in pool_path.read_text().splitlines()[: args.n_prompts]]
    else:
        from sci.data.templates import generate
        rows = generate(args.n_prompts, 0, seed=7)
    gen = Generator(args.model)
    prompts = [(i, build_prompt(gen.tokenizer, r["nl"], r["dialect"])) for i, r in enumerate(rows)]
    V = gen.llg_tok.vocab_size
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    history = []
    for step in range(args.steps):
        ros = collect_rollouts(gen, prompts, args.group, max_tokens=args.max_tokens, seed=step)
        st = ssd_step(gen.model, opt, ros, V, lam=1.0, mode="ssd", micro_batch=4)
        rec = {"step": step, "loss": st.loss, "loss_mask": st.loss_mask, "loss_grpo": st.loss_grpo,
               "n_rollouts": st.n_rollouts, "mean_reward": st.mean_reward, "mean_legal_mass": st.mean_legal_mass,
               "dt_s": round(st.dt, 1)}
        history.append(rec); print(json.dumps(rec), file=sys.stderr)
    losses = [h["loss_mask"] for h in history]
    ok = all(math.isfinite(h["loss"]) for h in history) and losses[-1] < losses[0]
    out = REPO / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "n_prompts": args.n_prompts, "group": args.group,
                               "history": history, "g1_pass": ok}, indent=2))
    print(f"[G1] pass={ok}", file=sys.stderr)


if __name__ == "__main__":
    main()
