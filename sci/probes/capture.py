"""Block-wise residual-stream capture and activation patching for mlx_lm models.

The transformer blocks of an mlx_lm model live in a plain Python list (`model.model.layers`
for Llama-style models such as SmolLM2, exposed as `model.layers` for Gemma-3). We run the
model's own `__call__` unchanged and temporarily replace each block in that list by a thin
proxy that forwards the call, records (or overwrites) the block's output, and delegates every
attribute access to the block. Nothing in mlx_lm is edited, the model's mask/scaling logic
is reused verbatim for every family, and the list is restored in a `finally`.

Layer index `l` denotes the output of block `l` (0-based), i.e. the residual stream entering
block `l + 1`. Batches are right-padded; the models are causal, so real positions are
unaffected by the padding.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


def blocks(model) -> list:
    """The list of transformer blocks of an mlx_lm model."""
    inner = getattr(model, "model", None)
    if inner is not None and isinstance(getattr(inner, "layers", None), list):
        return inner.layers
    if isinstance(getattr(model, "layers", None), list):
        return model.layers
    raise TypeError("model exposes no transformer block list")


@dataclass
class Patch:
    layer: int
    batch: int
    pos: int
    vector: np.ndarray   # (d,) replaces the residual at (batch, pos) after block `layer`


class _Tap:
    """Proxy for one block: forwards the call, applies patches, records the output."""

    def __init__(self, block, idx: int, store: dict | None, patches: list[Patch]):
        self._block, self._idx, self._store, self._patches = block, idx, store, patches

    def __getattr__(self, name):
        return getattr(self._block, name)

    def __call__(self, x, *args, **kwargs):
        h = self._block(x, *args, **kwargs)
        for p in self._patches:
            h[p.batch, p.pos] = mx.array(p.vector).astype(h.dtype)
        if self._store is not None:
            self._store[self._idx] = h
        return h


def _pad(seqs: list[list[int]]) -> tuple[mx.array, list[int]]:
    L = max(len(s) for s in seqs)
    return mx.array([s + [0] * (L - len(s)) for s in seqs]), [len(s) for s in seqs]


def run_tapped(model, seqs: list[list[int]], layers: list[int] | None = None,
               patches: list[Patch] | None = None, logit_positions: list[tuple[int, int]] | None = None,
               ) -> tuple[dict[int, mx.array], mx.array | None]:
    """One right-padded forward. Returns ({layer: (B, L, d) residuals}, logits at `logit_positions`
    as (n, V) float32 or None). `layers=None` captures every block; `layers=[]` captures none."""
    blks = blocks(model)
    n = len(blks)
    want = set(range(n)) if layers is None else set(layers)
    by_layer: dict[int, list[Patch]] = {}
    for p in patches or []:
        by_layer.setdefault(p.layer, []).append(p)
    if any(l < 0 or l >= n for l in want | set(by_layer)):
        raise IndexError(f"layer out of range for a {n}-block model")
    store: dict[int, mx.array] = {}
    originals = list(blks)
    try:
        for i, b in enumerate(originals):
            if i in want or i in by_layer:
                blks[i] = _Tap(b, i, store if i in want else None, by_layer.get(i, []))
        x, _ = _pad(seqs)
        logits = model(x)
        out_logits = None
        if logit_positions:
            bi = mx.array([b for b, _ in logit_positions]); pi = mx.array([p for _, p in logit_positions])
            out_logits = logits[bi, pi].astype(mx.float32)
        mx.eval(list(store.values()) + ([out_logits] if out_logits is not None else []))
    finally:
        blks[:] = originals
    return store, out_logits


def capture_residuals(model, token_ids: list[int], layers: list[int] | None = None) -> dict[int, np.ndarray]:
    """{layer: (T, d) float32} residual stream of one sequence after each requested block."""
    store, _ = run_tapped(model, [token_ids], layers)
    return {l: np.asarray(h[0].astype(mx.float32)) for l, h in store.items()}


def capture_residuals_batch(model, seqs: list[list[int]], layers: list[int] | None = None,
                            positions: list[list[int]] | None = None) -> list[dict[int, np.ndarray]]:
    """Per-sequence {layer: (T_i, d)} residuals from one padded forward; with `positions` (one
    list per sequence) only those positions are returned, as (len(positions[i]), d)."""
    store, _ = run_tapped(model, seqs, layers)
    out = []
    for i, s in enumerate(seqs):
        if positions is None:
            out.append({l: np.asarray(h[i, :len(s)].astype(mx.float32)) for l, h in store.items()})
        else:
            idx = mx.array(positions[i], dtype=mx.int32)
            out.append({l: np.asarray(h[i][idx].astype(mx.float32)) for l, h in store.items()})
    return out


def logits_at(model, seqs: list[list[int]], positions: list[tuple[int, int]],
              patches: list[Patch] | None = None) -> np.ndarray:
    """(n, V) float32 logits at the given (batch, pos) pairs, optionally with residual patches."""
    _, lg = run_tapped(model, seqs, [], patches, positions)
    return np.asarray(lg)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
