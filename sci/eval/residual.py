"""Per-constraint-layer violation rates and residuals (ADR-0004).

Given ladder rows for one (model, ckpt) on one pool:
  v1 = parse-invalid rate on free decoding
  v2 = `type` error bucket rate on free decoding (C2 domain)
  v3 = scope-validator failure ∪ `type_ssa` bucket on free decoding (C3)
  v4 = shape/interface bucket on free decoding (mlir-opt shape diagnostics)
Residual_k = v_k(free) − v_k(under stack S_k), with paired bootstrap CIs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from sci.constraints.parser import is_parse_valid
from sci.constraints.scope import accept_or_reject
from sci.eval.error_categories import categorize
from sci.eval.stats import bootstrap_ci, paired_bootstrap_diff
from sci.eval.verify import verify

SHAPE_RE = re.compile(r"(shape|rank|dimension|dim |inner dimension|incompatible|expected .*tensor|expected .*memref|number of operands)", re.I)
_OP_RE = re.compile(r"\b((?:arith|memref|linalg|func|stablehlo|scf|math|cf|tensor|vector|affine)\.[a-z_]+)\b")
_KNOWN_OPS: set[str] | None = None


def known_ops() -> set[str]:
    """Op mnemonics that appear in the bundled generation grammars (the dialect vocabulary)."""
    global _KNOWN_OPS
    if _KNOWN_OPS is None:
        from sci.constraints.parser import generation_grammar_path
        ops = set()
        for g in ("mlir_gen_c1c2", "mlir_gen_stablehlo"):
            ops |= set(_OP_RE.findall(generation_grammar_path(g).read_text()))
        _KNOWN_OPS = ops | {"func.func", "func.return"}
    return _KNOWN_OPS


def unknown_ops(text: str) -> list[str]:
    return sorted({o for o in _OP_RE.findall(text) if o not in known_ops()})


def violations(text: str) -> dict[str, bool]:
    """Layer-wise violation flags for one generation."""
    pv = is_parse_valid(text)
    if not pv:
        vocab = bool(unknown_ops(text))
        return {"v1": True, "v1_vocab": vocab, "v1_struct": not vocab, "v2": False, "v3": False, "v4": False,
                "verify": False, "cat": "vocab" if vocab else "syntax"}
    scope_ok, _ = accept_or_reject(text)
    r = verify(text)
    ok = r["returncode"] == 0
    cat = "passed" if ok else categorize(r["stderr"])
    v2 = (cat == "type")
    v3 = (not scope_ok) or (cat == "type_ssa")
    v4 = (not ok) and bool(SHAPE_RE.search(r["stderr"] or "")) and cat not in ("type", "type_ssa")
    return {"v1": False, "v1_vocab": False, "v1_struct": False, "v2": v2, "v3": v3, "v4": v4, "verify": ok, "cat": cat}


SPEC_N = {"arith_func_200": 150, "linalg_125": 30}


def residuals(rows: list[dict], subset: str = "all") -> dict:
    """rows: ladder rows for one model on one pool (all constraints). Returns per-layer rates + residuals.
    subset: 'all' | 'spec' (hand-authored benchmark prompts only) | 'mined' (test-suite pad prompts only)."""
    by = {}
    for r in rows:
        n_spec = SPEC_N.get(r["pool"], 10**9)
        if subset == "spec" and r["prompt_id"] >= n_spec:
            continue
        if subset == "mined" and r["prompt_id"] < n_spec:
            continue
        by.setdefault(r["constraint"], {})[r["prompt_id"]] = r
    out = {"n": {c: len(v) for c, v in by.items()}, "rates": {}, "residual_pp": {}}
    flags = {c: {pid: violations(r["generated"]) for pid, r in v.items()} for c, v in by.items()}
    for c, f in flags.items():
        ids = sorted(f)
        out["rates"][c] = {k: round(sum(f[i][k] for i in ids) / len(ids), 4) for k in ("v1", "v1_vocab", "v1_struct", "v2", "v3", "v4", "verify")}
        out["rates"][c]["cats"] = {}
        for i in ids:
            out["rates"][c]["cats"][f[i]["cat"]] = out["rates"][c]["cats"].get(f[i]["cat"], 0) + 1
    stack_for = {"v1": "c1", "v1_vocab": "c1", "v1_struct": "c1", "v2": "c1_c2", "v3": "c1_c2_c3", "v4": "c1_c2_c3"}
    if "none" in flags:
        for k, sc in stack_for.items():
            if sc not in flags:
                continue
            ids = sorted(set(flags["none"]) & set(flags[sc]))
            d = paired_bootstrap_diff([int(flags["none"][i][k]) for i in ids], [int(flags[sc][i][k]) for i in ids])
            out["residual_pp"][k] = {"vs": sc, "pp": round(100 * d.point, 1), "ci95": [round(100 * d.ci_low, 1), round(100 * d.ci_high, 1)], "n": d.n}
    return out


def residuals_from_jsonl(path: Path, model: str, pool: str) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["model"] == model and r["pool"] == pool]
    out = residuals(rows, "all")
    out["spec"] = residuals(rows, "spec")
    out["mined"] = residuals(rows, "mined")
    return out
