"""Probe datasets from MLIR programs: examples, paired corruptions and batched feature extraction.

`build_examples` turns programs (gold or model-generated, each with its NL prompt and dialect)
into token sequences (prompt + program) and probe examples for the designs in
`sci.probes.probe` plus the clean/corrupted pairs used for activation patching.
`extract_features` runs the batched residual capture once per micro-batch and gathers only
the positions the examples need.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from sci.eval.generate import build_prompt
from sci.probes.capture import capture_residuals_batch
from sci.probes.patching import PatchPair, build_pair
from sci.probes.positions import attach_token_positions, find_sites, tokenize_with_offsets


@dataclass
class ProbeExample:
    design: str      # use | name | def_diff
    seq: int         # index into the sequence list
    pos: int         # residual position (use position)
    label: int
    group: int       # program index (corrupted copies share it with their clean program)
    pos2: int = -1   # def position for def_diff (feature = h[pos] - h[pos2])
    meta: dict = field(default_factory=dict)


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    """Prompt ids the way `Generator.encode` produces them (BOS unless the template has it)."""
    add_bos = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
    return tokenizer.encode(prompt, add_special_tokens=add_bos)


def build_examples(tokenizer, programs: list[dict], max_cands: int = 3, seed: int = 0, corrupt: bool = True,
                   max_len: int = 900, checked_only: bool = True,
                   ) -> tuple[list[list[int]], list[ProbeExample], list[PatchPair]]:
    """programs: dicts with `text`, `nl`, `dialect` (+ optional `source`). Returns (sequences,
    examples, patch pairs). Programs with no function or over `max_len` tokens are skipped."""
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    examples: list[ProbeExample] = []
    pairs: list[PatchPair] = []
    for gi, prog in enumerate(programs):
        text = prog["text"]
        try:
            uses, defs = find_sites(text)
        except (ValueError, AssertionError):
            continue
        if not uses:
            continue
        prompt_ids = encode_prompt(tokenizer, build_prompt(tokenizer, prog["nl"], prog["dialect"]))
        ids, offsets = tokenize_with_offsets(tokenizer, text)
        if len(prompt_ids) + len(ids) > max_len:
            continue
        attach_token_positions(uses, defs, offsets, len(prompt_ids))
        s = len(seqs)
        seqs.append(prompt_ids + ids)
        meta = {"program": gi, "source": prog.get("source", "gold")}
        for u in uses:
            if checked_only and not u.checked:
                continue
            m = dict(meta, name=u.name, stmt=u.stmt_index)
            examples.append(ProbeExample("use", s, u.pred_pos, u.label, gi, meta=m))
            examples.append(ProbeExample("name", s, u.name_pos, u.label, gi, meta=m))
            # def_diff: candidates defined by a statement of the same function (parameters are excluded so
            # that positives and negatives read the same kind of token, an LHS `%`, and differ only in
            # whether that definition precedes the use)
            fdefs = {d.name: d for d in defs if d.func_index == u.func_index and d.stmt_index >= 0}
            pos_c = [n for n in sorted(u.in_scope) if n in fdefs]
            neg_c = [n for n in sorted(u.out_of_scope) if n in fdefs]
            rng.shuffle(pos_c); rng.shuffle(neg_c)
            k = min(max_cands, len(pos_c), len(neg_c)) if neg_c else 0
            for n in pos_c[:k]:
                examples.append(ProbeExample("def_diff", s, u.pct_pos, 1, gi, fdefs[n].pct_pos, dict(m, cand=n)))
            for n in neg_c[:k]:
                examples.append(ProbeExample("def_diff", s, u.pct_pos, 0, gi, fdefs[n].pct_pos, dict(m, cand=n)))
            if corrupt and u.label == 1:
                try:
                    pair = build_pair(tokenizer, prompt_ids, text, u, uses, defs, meta=m)
                except (ValueError, AssertionError):
                    continue
                if len(pair.corrupt_ids) > max_len:
                    continue
                sc = len(seqs)
                seqs.append(pair.corrupt_ids)
                pairs.append(pair)
                mc = dict(m, source=meta["source"] + "+corrupt", renamed_to=pair.meta["renamed_to"])
                examples.append(ProbeExample("use", sc, pair.corrupt_pos, 0, gi, meta=mc))
                examples.append(ProbeExample("name", sc, pair.corrupt_pos + 1, 0, gi, meta=mc))
    return seqs, examples, pairs


def extract_features(model, seqs: list[list[int]], examples: list[ProbeExample], layers: list[int] | None,
                     micro_batch: int = 8, log=None) -> tuple[dict[int, np.ndarray], dict[tuple[int, int], int]]:
    """Residual vectors at every (seq, pos) an example needs: ({layer: (n_rows, d)}, {(seq, pos): row})."""
    need: dict[int, set[int]] = {}
    for e in examples:
        need.setdefault(e.seq, set()).add(e.pos)
        if e.pos2 >= 0:
            need[e.seq].add(e.pos2)
    row: dict[tuple[int, int], int] = {}
    for s in sorted(need):
        for p in sorted(need[s]):
            row[(s, p)] = len(row)
    order = sorted(need, key=lambda s: len(seqs[s]))
    feats: dict[int, np.ndarray] = {}
    done = 0
    for i in range(0, len(order), micro_batch):
        chunk = order[i:i + micro_batch]
        pos_lists = [sorted(need[s]) for s in chunk]
        outs = capture_residuals_batch(model, [seqs[s] for s in chunk], layers, pos_lists)
        for s, plist, out in zip(chunk, pos_lists, outs):
            for l, arr in out.items():
                if l not in feats:
                    feats[l] = np.zeros((len(row), arr.shape[-1]), np.float32)
                for p, vec in zip(plist, arr):
                    feats[l][row[(s, p)]] = vec
        done += len(chunk)
        if log:
            log(f"[capture] {done}/{len(order)} sequences")
    return feats, row


def design_matrix(feats: dict[int, np.ndarray], row: dict[tuple[int, int], int], examples: list[ProbeExample],
                  design: str) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """({layer: X}, y, groups) for one probe design."""
    ex = [e for e in examples if e.design == design]
    y = np.array([e.label for e in ex], int)
    groups = np.array([e.group for e in ex], int)
    r1 = np.array([row[(e.seq, e.pos)] for e in ex], int)
    X: dict[int, np.ndarray] = {}
    if design == "def_diff":
        r2 = np.array([row[(e.seq, e.pos2)] for e in ex], int)
        for l, F in feats.items():
            X[l] = F[r1] - F[r2]
    else:
        for l, F in feats.items():
            X[l] = F[r1]
    return X, y, groups
