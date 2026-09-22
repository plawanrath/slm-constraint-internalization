"""Resumable constraint-ladder runner for held-out transfer targets.

Target-agnostic counterpart of `sci.eval.ladder.run_model` for benchmark JSONL sets
(rows `{id, nl, mlir}`) whose parse / scope / verify tools differ from the training
dialects: a `Target` bundles those callables and the runner handles resume, row
schema, per-layer violation flags (v1 parse, v2 type bucket, v3 scope or
type_ssa bucket, v4 shape/interface) and the summary. Used by
`scripts/w10_stablehlo.py` (StableHLO) and `scripts/w11_scf.py` (scf).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sci.eval.error_categories import categorize
from sci.eval.ladder import _done_keys, summarize
from sci.eval.residual import SHAPE_RE
from sci.eval.stats import paired_bootstrap_diff

CONSTRAINTS = ("none", "c1", "c1_c2", "c1_c2_c3")
LAYER_STACK = {"v1": "c1", "v2": "c1_c2", "v3": "c1_c2_c3", "v4": "c1_c2_c3"}


@dataclass
class Cell:
    tag: str
    model: str
    revision: str | None = None
    ckpt_dir: Path | None = None
    method: str = "none"


@dataclass
class Target:
    dialect: str
    make_generator: Callable[[str, str | None], object]   # (model_repo, revision) -> Generator-like
    build_prompt: Callable[[object, str], str]            # (tokenizer, nl) -> prompt
    parse_valid: Callable[[str], bool]
    scope_ok: Callable[[str], bool]
    verify: Callable[[str], dict]                         # -> {returncode, stderr, ...}
    functional: Callable[[str, dict], dict] | None = None  # (generated, benchmark_row) -> result dict


def load_set(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for i, r in enumerate(rows):
        r.setdefault("prompt_id", i)
    return rows


def violation_flags(target: Target, text: str) -> dict:
    """Layer-wise violation flags for one generation (mirrors `sci.eval.residual.violations`)."""
    pv = target.parse_valid(text)
    if not pv:
        return {"parse_valid": False, "scope_valid": False, "verify_valid": False, "verify_stderr": "",
                "cat": "syntax", "v1": True, "v2": False, "v3": False, "v4": False}
    scope_ok = target.scope_ok(text)
    r = target.verify(text)
    ok = r["returncode"] == 0
    err = r.get("stderr") or ""
    cat = "passed" if ok else categorize(err)
    return {"parse_valid": True, "scope_valid": scope_ok, "verify_valid": ok, "verify_stderr": err[:300],
            "cat": cat, "v1": False, "v2": cat == "type", "v3": (not scope_ok) or cat == "type_ssa",
            "v4": (not ok) and bool(SHAPE_RE.search(err)) and cat not in ("type", "type_ssa")}


def run_cell(cell: Cell, target: Target, pools: dict[str, list[dict]], constraints: tuple[str, ...],
             out_jsonl: Path, max_tokens: int = 600, seed: int = 0, limit: int | None = None) -> None:
    done = _done_keys(out_jsonl)
    todo = [(p, c, r["prompt_id"]) for p, rows in pools.items() for c in constraints
            for r in (rows[:limit] if limit else rows)]
    if all((cell.tag, p, c, i) in done for p, c, i in todo):
        print(f"[ladder] {cell.tag}: all cells present, skipping", file=sys.stderr)
        return
    gen = target.make_generator(cell.model, cell.revision)
    if cell.ckpt_dir is not None:
        from sci.train.run import load_ckpt
        load_ckpt(gen, Path(cell.ckpt_dir))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a") as fout:
        for pool_name, rows in pools.items():
            if limit:
                rows = rows[:limit]
            for constraint in constraints:
                n_ok = n_done = 0; t_cell = time.perf_counter()
                for r in rows:
                    key = (cell.tag, pool_name, constraint, r["prompt_id"])
                    if key in done:
                        continue
                    base = {"model": cell.tag, "method": cell.method, "ckpt": str(cell.ckpt_dir or "base"),
                            "train_seed": None, "constraint": constraint, "pool": pool_name, "dialect": target.dialect,
                            "seed": seed, "prompt_id": r["prompt_id"], "id": r.get("id"), "nl": r["nl"]}
                    try:
                        prompt = target.build_prompt(gen.tokenizer, r["nl"])
                        g = gen.generate(prompt, constraint=constraint, max_tokens=max_tokens, seed=seed)
                        row = {**base, "generated": g.text, **violation_flags(target, g.text),
                               "attempts": g.attempts, "scope_passed_per_try": g.scope_passed_per_try,
                               "finished": g.finished, "n_tokens": len(g.tokens), "dt": round(g.dt, 3)}
                        if target.functional is not None and row["verify_valid"]:
                            row["functional"] = target.functional(g.text, r)
                    except Exception as e:  # noqa: BLE001
                        row = {**base, "generated": "", "parse_valid": False, "scope_valid": False,
                               "verify_valid": False, "verify_stderr": "", "cat": "error", "v1": True, "v2": False,
                               "v3": False, "v4": False, "attempts": 0, "scope_passed_per_try": [], "finished": False,
                               "n_tokens": 0, "dt": 0.0, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                    fout.write(json.dumps(row) + "\n"); fout.flush()
                    n_done += 1; n_ok += row["verify_valid"]
                    if n_done % 25 == 0:
                        print(f"  [{cell.tag}|{pool_name}|{constraint}] {n_done}/{len(rows)} verify={n_ok} "
                              f"{(time.perf_counter()-t_cell)/n_done:.2f}s/gen", file=sys.stderr)
                print(f"[ladder] {cell.tag}|{pool_name}|{constraint}: verify {n_ok}/{n_done} "
                      f"in {(time.perf_counter()-t_cell)/60:.1f} min", file=sys.stderr)


def residuals(rows: list[dict]) -> dict:
    """Per-layer violation rates per constraint and Residual_k = v_k(free) - v_k(S_k) with paired CIs."""
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["constraint"], {})[r["prompt_id"]] = r
    out = {"n": {c: len(v) for c, v in by.items()}, "rates": {}, "residual_pp": {}, "functional_rate": {}}
    for c, byid in by.items():
        ids = sorted(byid)
        out["rates"][c] = {k: round(sum(bool(byid[i].get(k)) for i in ids) / len(ids), 4)
                           for k in ("v1", "v2", "v3", "v4", "verify_valid")}
        fn = [byid[i].get("functional") for i in ids]
        out["functional_rate"][c] = round(sum(1 for f in fn if f and f.get("status") == "match") / len(ids), 4)
    if "none" in by:
        for k, sc in LAYER_STACK.items():
            if sc not in by:
                continue
            ids = sorted(set(by["none"]) & set(by[sc]))
            if not ids:
                continue
            d = paired_bootstrap_diff([int(bool(by["none"][i][k])) for i in ids], [int(bool(by[sc][i][k])) for i in ids])
            out["residual_pp"][k] = {"vs": sc, "pp": round(100 * d.point, 1),
                                     "ci95": [round(100 * d.ci_low, 1), round(100 * d.ci_high, 1)], "n": d.n}
    return out


def write_summary(out_jsonl: Path, out_json: Path) -> dict:
    cells = summarize(out_jsonl, out_json)
    rows = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l.strip()]
    res: dict = {}
    for model in cells:
        for pool in cells[model]:
            res.setdefault(model, {})[pool] = residuals([r for r in rows if r["model"] == model and r["pool"] == pool])
    summary = {"cells": cells, "residuals": res}
    out_json.write_text(json.dumps(summary, indent=2))
    return summary
