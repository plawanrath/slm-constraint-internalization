"""Disjointness tooling for the training pool.

Checks every candidate (nl, mlir) against a set of protected records (released benchmarks,
functional references, evaluation pools):
  * exact match on normalized NL, exact match on normalized MLIR;
  * NL n-gram overlap (default 8-gram): a candidate is flagged if any 8-gram of its normalized
    NL occurs in any protected NL;
  * MLIR near-duplicate: normalized MLIR with SSA names and function names canonicalized.
Returns a report and the filtered candidates.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROTECTED_FILES = [
    "data/benchmarks/mlir_spec_150.jsonl", "data/benchmarks/linalg_spec_30.jsonl",
    "data/benchmarks/stablehlo_spec_30.jsonl", "data/benchmarks/stablehlo_held_out_200.jsonl",
    "data/benchmarks/stablehlo_outofgrammar_25.jsonl", "data/functional/references.jsonl",
    "data/pools/arith_func_200.jsonl", "data/pools/linalg_125.jsonl",
]


def norm_nl(s: str) -> str:
    s = s.lower().replace("`", "")
    s = re.sub(r"[^a-z0-9%?<>\.\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_mlir(s: str) -> str:
    s = re.sub(r"//.*", "", s)
    s = re.sub(r"@[A-Za-z_][A-Za-z0-9_]*", "@F", s)
    names = {}
    def ssa(m):
        n = m.group(0)
        names.setdefault(n, f"%v{len(names)}")
        return names[n]
    s = re.sub(r"%[A-Za-z0-9_#$.]+", ssa, s)
    s = re.sub(r"\s*([,:=\[\]\(\)\{\}<>])\s*", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def ngrams(s: str, n: int) -> set[tuple[str, ...]]:
    w = s.split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_protected(files: list[str] = PROTECTED_FILES) -> list[dict]:
    out = []
    for f in files:
        p = REPO / f
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out.append({"src": f, "nl": r.get("nl", ""), "mlir": r.get("mlir") or r.get("gold_mlir") or ""})
    return out


def check(candidates: list[dict], protected: list[dict] | None = None, n: int = 8) -> tuple[list[dict], dict]:
    protected = protected if protected is not None else load_protected()
    p_nl = {norm_nl(r["nl"]) for r in protected if r["nl"]}
    p_mlir = {norm_mlir(r["mlir"]) for r in protected if r["mlir"]}
    p_ngrams: set = set()
    for r in protected:
        if r["nl"]:
            p_ngrams |= ngrams(norm_nl(r["nl"]), n)
    kept, reasons = [], Counter()
    seen_nl, seen_mlir = set(), set()
    for c in candidates:
        cn, cm = norm_nl(c["nl"]), norm_mlir(c["mlir"])
        if cn in p_nl:
            reasons["exact_nl_vs_protected"] += 1; continue
        if cm in p_mlir:
            reasons["exact_mlir_vs_protected"] += 1; continue
        if ngrams(cn, n) & p_ngrams:
            reasons[f"nl_{n}gram_overlap_vs_protected"] += 1; continue
        if cn in seen_nl:
            reasons["dup_nl_within_pool"] += 1; continue
        seen_nl.add(cn); seen_mlir.add(cm)
        kept.append(c)
    report = {"n_candidates": len(candidates), "n_kept": len(kept), "dropped": dict(reasons),
              "n_protected": len(protected), "ngram": n,
              "unique_mlir_in_kept": len(seen_mlir), "unique_nl_in_kept": len(seen_nl)}
    return kept, report
