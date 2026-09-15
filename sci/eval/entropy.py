"""Token-entropy analysis under successive constraint levels.

For each (dialect, constraint_level) pair:
  - Compute per-step next-token entropy on a sample of held-out prompts.
  - Aggregate into a single scalar (mean entropy per generated token) + curve.

The intuition: as we tighten from base → C1 → C1+C2,
the entropy should collapse, and the entropy collapse should correlate with
pass@1 recovery. This is the mechanism artifact.

Implementation is MLX-only (entropy needs access to per-step logits, which
Ollama doesn't expose). We write results/entropy.json keyed by
(dialect, constraint_level, model).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def token_entropies_mlx(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_tokens: int = 256,
    constraint_mask_fn=None,
) -> list[list[float]]:
    """Return per-prompt per-step entropy lists.

    `constraint_mask_fn(logits, state) -> masked_logits` is called before
    softmax; pass None for unconstrained measurement.
    """
    try:
        import mlx.core as mx  # type: ignore
    except ImportError as e:
        raise RuntimeError("mlx unavailable — entropy analysis requires Apple Silicon.") from e

    all_entropies: list[list[float]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        x = mx.array(ids)
        entropies: list[float] = []
        for _ in range(max_tokens):
            logits = model(x[None])[:, -1, :]
            if constraint_mask_fn is not None:
                logits = constraint_mask_fn(logits, {})
            probs = mx.softmax(logits, axis=-1)
            # Entropy of the softmax distribution, in bits.
            p = np.asarray(probs[0], dtype=np.float64)
            p = p[p > 0]
            h = float(-(p * np.log2(p)).sum())
            entropies.append(h)
            # Sample greedily for the entropy walk (we want a stable trajectory).
            next_id = int(mx.argmax(logits, axis=-1)[0])
            x = mx.concatenate([x, mx.array([next_id])])
            if next_id == tokenizer.eos_token_id:
                break
        all_entropies.append(entropies)
    return all_entropies


def summarize(entropies: list[list[float]]) -> dict:
    if not entropies:
        return {"n_prompts": 0, "mean_per_token": float("nan"), "curve": []}
    max_len = max(len(e) for e in entropies)
    padded = np.full((len(entropies), max_len), np.nan)
    for i, e in enumerate(entropies):
        padded[i, : len(e)] = e
    mean_per_token = float(np.nanmean(padded))
    curve = np.nanmean(padded, axis=0).tolist()
    return {
        "n_prompts": len(entropies),
        "mean_per_token": mean_per_token,
        "curve": curve,
        "per_prompt_mean": [float(np.nanmean(e)) if e else float("nan") for e in entropies],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--dialect", required=True)
    ap.add_argument("--constraint", default="none")
    ap.add_argument("--out", type=Path, default=Path("results/entropy.json"))
    ap.add_argument("--n-prompts", type=int, default=500)
    args = ap.parse_args()

    from mlx_lm import load  # lazy: not needed for unit tests
    model, tokenizer = load(args.model)
    prompts = [
        json.loads(line)["prompt"] if line.startswith("{") else line.strip()
        for line in args.prompts.read_text().splitlines()
        if line.strip()
    ][: args.n_prompts]

    ents = token_entropies_mlx(model, tokenizer, prompts)
    summary = summarize(ents)
    out_key = f"{args.model}|{args.dialect}|{args.constraint}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.out.exists():
        existing = json.loads(args.out.read_text())
    existing[out_key] = summary
    args.out.write_text(json.dumps(existing, indent=2))
    print(f"[entropy] wrote {out_key} → {args.out}")


if __name__ == "__main__":
    main()
