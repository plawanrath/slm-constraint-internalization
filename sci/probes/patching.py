"""Activation patching at SSA-use positions.

A pair is (clean, corrupted): the clean program uses an in-scope name at a use site; the
corrupted program is the same text with that name's definition and earlier uses renamed
(`positions.corrupt_use`), so at the same site the upcoming name is now out of scope while
the surface form of the use itself is unchanged. For each layer l we run the corrupted
program with the residual stream after block l at the use position replaced by the clean
run's residual and read the next-token distribution there:

  target    log-prob of the first token of the originally used name (in scope in clean, out of
            scope in corrupted); `recovery = (patched - corrupted) / (clean - corrupted)`.
  margin    log-mass on first tokens of names in scope in the *corrupted* program minus
            log-mass on first tokens of names out of scope there (the renamed original, names
            defined later); patching should lower it if the layer carries the scope set.

Per-layer effects are averaged over pairs with bootstrap CIs. Forwards are batched: one
padded forward per (layer, micro-batch of pairs).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from sci.eval.stats import bootstrap_ci
from sci.probes.capture import Patch, blocks, log_softmax, run_tapped
from sci.probes.positions import (DefSite, UseSite, attach_token_positions, corrupt_use, find_sites,
                                  first_name_token, match_site, tokenize_with_offsets)


@dataclass
class PatchPair:
    clean_ids: list[int]
    corrupt_ids: list[int]
    clean_pos: int
    corrupt_pos: int
    target_tok: int
    in_toks: list[int]
    out_toks: list[int]
    meta: dict = field(default_factory=dict)


def build_pair(tokenizer, prompt_ids: list[int], text: str, site: UseSite, uses: list[UseSite],
               defs: list[DefSite], meta: dict | None = None) -> PatchPair:
    """Clean/corrupted pair for one in-scope use site of `text` (program tokens follow `prompt_ids`)."""
    if site.label != 1:
        raise ValueError("patching pairs need an in-scope use")
    c_text, new_name = corrupt_use(text, site, uses, defs)
    c_uses, c_defs = find_sites(c_text)
    c_site = match_site(c_uses, uses, site)
    assert c_site.label == 0
    ids, off = tokenize_with_offsets(tokenizer, text)
    c_ids, c_off = tokenize_with_offsets(tokenizer, c_text)
    attach_token_positions(uses, defs, off, len(prompt_ids))
    attach_token_positions(c_uses, c_defs, c_off, len(prompt_ids))
    ft = lambda n: first_name_token(tokenizer, n)  # noqa: E731
    in_toks = sorted({ft(n) for n in c_site.in_scope})
    out_toks = sorted({ft(n) for n in c_site.out_of_scope | {site.name}} - set(in_toks))
    return PatchPair(prompt_ids + ids, prompt_ids + c_ids, site.pred_pos, c_site.pred_pos, ft(site.name),
                     in_toks, out_toks, dict(meta or {}, name=site.name, renamed_to=new_name,
                                             n_in_scope=len(c_site.in_scope)))


def _margin(lp: np.ndarray, in_toks: list[int], out_toks: list[int]) -> float:
    lse = lambda v: float(np.logaddexp.reduce(v)) if len(v) else -np.inf  # noqa: E731
    return lse(lp[in_toks]) - lse(lp[out_toks])


def run_patching(model, pairs: list[PatchPair], layers: list[int] | None = None, micro_batch: int = 8,
                 log=None) -> dict:
    """Per-layer patching effects over `pairs`. Returns {"layers": [...], "per_layer": {l: {...}},
    "pairs": [...per-pair raw numbers...]}."""
    n_layers = len(blocks(model))
    layers = list(range(n_layers)) if layers is None else list(layers)
    n = len(pairs)
    t_clean = np.zeros(n); t_corr = np.zeros(n); m_clean = np.zeros(n); m_corr = np.zeros(n)
    t_patch = np.zeros((len(layers), n)); m_patch = np.zeros((len(layers), n))
    clean_res: list[dict[int, np.ndarray]] = [None] * n
    for s in range(0, n, micro_batch):
        chunk = list(range(s, min(n, s + micro_batch)))
        # clean pass: residuals at the use position (all patch layers) + logits there
        store, lg = run_tapped(model, [pairs[i].clean_ids for i in chunk], layers,
                               logit_positions=[(b, pairs[i].clean_pos) for b, i in enumerate(chunk)])
        lp = log_softmax(np.asarray(lg))
        for b, i in enumerate(chunk):
            clean_res[i] = {l: np.asarray(store[l][b, pairs[i].clean_pos].astype(mx.float32)) for l in layers}
            t_clean[i] = lp[b, pairs[i].target_tok]; m_clean[i] = _margin(lp[b], pairs[i].in_toks, pairs[i].out_toks)
        # corrupted baseline
        _, lg = run_tapped(model, [pairs[i].corrupt_ids for i in chunk], [],
                           logit_positions=[(b, pairs[i].corrupt_pos) for b, i in enumerate(chunk)])
        lp = log_softmax(np.asarray(lg))
        for b, i in enumerate(chunk):
            t_corr[i] = lp[b, pairs[i].target_tok]; m_corr[i] = _margin(lp[b], pairs[i].in_toks, pairs[i].out_toks)
        # patched runs, one forward per layer
        for li, l in enumerate(layers):
            patches = [Patch(l, b, pairs[i].corrupt_pos, clean_res[i][l]) for b, i in enumerate(chunk)]
            _, lg = run_tapped(model, [pairs[i].corrupt_ids for i in chunk], [], patches,
                               [(b, pairs[i].corrupt_pos) for b, i in enumerate(chunk)])
            lp = log_softmax(np.asarray(lg))
            for b, i in enumerate(chunk):
                t_patch[li, i] = lp[b, pairs[i].target_tok]
                m_patch[li, i] = _margin(lp[b], pairs[i].in_toks, pairs[i].out_toks)
        for i in chunk:
            clean_res[i] = None
        if log:
            log(f"[patch] {chunk[-1] + 1}/{n} pairs")
    gap = t_clean - t_corr
    valid = gap > 0.1   # pairs where the corruption actually moved the target (recovery is undefined otherwise)
    per_layer = {}
    for li, l in enumerate(layers):
        rec = (t_patch[li] - t_corr) / np.where(valid, gap, 1.0)
        dm = m_patch[li] - m_corr
        r_ci = bootstrap_ci(rec[valid], n_resamples=2000, seed=0) if valid.any() else None
        m_ci = bootstrap_ci(dm, n_resamples=2000, seed=0)
        per_layer[int(l)] = {
            "target_recovery": r_ci.point if r_ci else float("nan"),
            "target_recovery_ci": [r_ci.ci_low, r_ci.ci_high] if r_ci else [float("nan")] * 2,
            "target_logprob_delta": float(np.mean(t_patch[li] - t_corr)),
            "margin_delta": m_ci.point, "margin_delta_ci": [m_ci.ci_low, m_ci.ci_high],
        }
    return {
        "layers": [int(l) for l in layers], "n_pairs": n, "n_valid": int(valid.sum()),
        "target_logprob_clean": float(t_clean.mean()), "target_logprob_corrupt": float(t_corr.mean()),
        "margin_clean": float(m_clean.mean()), "margin_corrupt": float(m_corr.mean()),
        "per_layer": per_layer,
        "pairs": [dict(pairs[i].meta, t_clean=float(t_clean[i]), t_corr=float(t_corr[i]),
                       m_clean=float(m_clean[i]), m_corr=float(m_corr[i]),
                       t_patch=[float(v) for v in t_patch[:, i]]) for i in range(n)],
    }
