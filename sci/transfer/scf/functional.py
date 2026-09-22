"""Gold-differential functional scoring for the scf target.

Reuses the driver construction, lowering pipeline, runner and output comparison of
`sci.eval.differential` unchanged: that pipeline already lowers `scf` (`--convert-
scf-to-cf` sits between `--convert-linalg-to-loops` and the LLVM conversions), so
loops over memrefs execute under `mlir-cpu-runner` exactly like the linalg golds.
`SCF_PIPELINE` is asserted to contain the scf lowering so a future pipeline edit
cannot silently break this target.

Two entry points:
  * `differential(candidate, gold, ref)`  — K seeded trials, `differential.py`
    semantics (signature compatibility, memref final state + scalar result);
    scalar inputs honor the reference's `input_domains` when present.
  * `run_scf(generated, ref)`             — `sci.eval.functional.DISPATCH`-shaped
    {verify_valid, lower_ok, exec_ok, output_match, detail} for one candidate.
Importing this module registers the "scf" dialect in `differential.DIALECTS` and
`run_scf` in `functional.DISPATCH`, so both existing CLIs can score the set.

Reference rows (`data/functional/scf_references.jsonl`):
  {id, prompt_id, dialect: "scf", source_benchmark: "scf_spec_60", source_id, nl,
   canonical_fn_name, canonical_signature, signature: {args, ret},
   input_domains: [{type, min, max}, ...]}   (one entry per argument, in order)
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import sci.eval.differential as _diff
import sci.eval.functional as _func
from sci.eval.differential import (
    K_TRIALS,
    build_driver,
    classify_signature,
    gen_trial,
    lower_and_run,
    outputs_equal,
    parse_first_fn,
    parse_stdout,
    signature_compatible,
    strip_outer_module,
)
from sci.eval.verify import verify

REPO = Path(__file__).resolve().parents[3]
BENCHMARK = REPO / "data" / "benchmarks" / "scf_spec_60.jsonl"
REFERENCES = REPO / "data" / "functional" / "scf_references.jsonl"

SCF_PIPELINE = list(_diff.PIPELINE)
assert "--convert-scf-to-cf" in SCF_PIPELINE, "differential pipeline lost the scf lowering"

_diff.DIALECTS.setdefault("scf", {"gold_jsonl": "data/benchmarks/scf_spec_60.jsonl",
                                  "cand_dialect": "scf", "n_spec": 60})


def load_benchmark(path: Path = BENCHMARK) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_references(path: Path = REFERENCES) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def gold_for(ref: dict, benchmark: list[dict] | None = None) -> str:
    rows = benchmark if benchmark is not None else load_benchmark()
    for r in rows:
        if r["id"] == ref["source_id"]:
            return r["mlir"]
    raise KeyError(ref["source_id"])


# ---------------- inputs ----------------

def _domain_rng(prompt_key: str, trial: int, k: int) -> random.Random:
    h = hashlib.sha256(f"scf-domain:{prompt_key}:{trial}:{k}".encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def trial_inputs(sig: dict, ref: dict, prompt_key: str, trial: int) -> dict:
    """`differential.gen_trial` data with scalar values redrawn from `input_domains`
    (memref shapes and contents are left to the harness so its in-bounds guarantees hold)."""
    td = gen_trial(sig, prompt_key, trial)
    domains = ref.get("input_domains") or []
    for k, (t, d) in enumerate(zip(sig["arg_types"], domains)):
        if t.startswith("memref") or "min" not in d or "max" not in d:
            continue
        rng = _domain_rng(prompt_key, trial, k)
        if t in _diff.FLOAT_TYPES:
            lo, hi = float(d["min"]), float(d["max"])
            td["values"][k] = rng.randrange(int(lo * 4), int(hi * 4) + 1) * 0.25
        else:
            lo, hi = int(d["min"]), int(d["max"])
            if td["shapes"] and any(s is not None for s in td["shapes"]):
                # index-like scalars must stay in bounds for every memref argument
                min_dim = min(min(s) for s in td["shapes"] if s is not None)
                hi = min(hi, min_dim - 1)
            td["values"][k] = rng.randint(lo, max(lo, hi))
    return td


# ---------------- gate + differential ----------------

def gold_gate(gold: str, ref: dict, k: int = K_TRIALS) -> dict:
    """Run the gold twice per trial; {ok, sig, outputs: {trial: parsed}, err}."""
    sig = parse_first_fn(gold)
    reason, detail = classify_signature(sig)
    if reason:
        return {"ok": False, "sig": sig, "outputs": {}, "err": f"excluded:{reason}:{detail}"}
    key = f"scf:{ref['source_id']}"
    body = strip_outer_module(gold)
    outs = {}
    for t in range(k):
        td = trial_inputs(sig, ref, key, t)
        driver = build_driver(body, sig["name"], sig, td)
        ra = lower_and_run(driver)
        if not ra["exec_ok"]:
            return {"ok": False, "sig": sig, "outputs": {}, "err": f"trial{t}:{'lower' if not ra['lower_ok'] else 'exec'}:{ra['err'][:200]}"}
        rb = lower_and_run(driver)
        if not rb["exec_ok"] or not outputs_equal(parse_stdout(ra["stdout"]), parse_stdout(rb["stdout"]), exact=True):
            return {"ok": False, "sig": sig, "outputs": {}, "err": f"trial{t}:nondeterministic"}
        outs[t] = parse_stdout(ra["stdout"])
    return {"ok": True, "sig": sig, "outputs": outs, "err": ""}


def differential(candidate: str, gold: str, ref: dict, gate: dict | None = None, k: int = K_TRIALS) -> dict:
    """Score `candidate` against `gold` on K seeded trials.

    status: match | mismatch | signature_mismatch | no_parseable_fn | fn_named_main | gold_gate_fail.
    """
    if gate is None:
        gate = gold_gate(gold, ref, k)
    if not gate["ok"]:
        return {"status": "gold_gate_fail", "err": gate["err"], "n_trials_matched": 0, "trials": []}
    sig = gate["sig"]
    csig = parse_first_fn(candidate)
    if csig is None:
        return {"status": "no_parseable_fn", "n_trials_matched": 0, "trials": []}
    if csig["name"] == "main":
        return {"status": "fn_named_main", "n_trials_matched": 0, "trials": []}
    compat, dim_relaxed = signature_compatible(sig, csig)
    if not compat:
        return {"status": "signature_mismatch", "cand_sig": csig, "n_trials_matched": 0, "trials": []}
    key = f"scf:{ref['source_id']}"
    body = strip_outer_module(candidate)
    trials, n_match, first_err = [], 0, ""
    for t in range(k):
        td = trial_inputs(sig, ref, key, t)
        rc = lower_and_run(build_driver(body, csig["name"], sig, td, callee_types=csig["arg_types"]))
        ok = rc["exec_ok"] and outputs_equal(parse_stdout(rc["stdout"]), gate["outputs"][t])
        n_match += ok
        if not ok and not first_err:
            first_err = "lower_fail" if not rc["lower_ok"] else "exec_fail" if not rc["exec_ok"] else "output_mismatch"
        trials.append({"trial": t, "lower_ok": rc["lower_ok"], "exec_ok": rc["exec_ok"], "match": ok, "err": rc["err"][:120]})
    return {"status": "match" if n_match == k else "mismatch", "n_trials_matched": n_match,
            "first_divergence": first_err, "dim_relaxed": dim_relaxed, "trials": trials}


def run_scf(generated: str, ref: dict) -> dict:
    """`sci.eval.functional` runner contract for one scf candidate."""
    detail: dict = {"dialect": "scf"}
    v = verify(generated)
    if v["returncode"] != 0:
        detail["verify_stderr"] = (v["stderr"] or "")[:300]
        return {"verify_valid": False, "lower_ok": False, "exec_ok": False, "output_match": False, "detail": detail}
    res = differential(generated, gold_for(ref), ref)
    detail.update({k: v for k, v in res.items() if k != "trials"})
    trials = res.get("trials") or []
    lower_ok = bool(trials) and all(t["lower_ok"] for t in trials)
    exec_ok = bool(trials) and all(t["exec_ok"] for t in trials)
    return {"verify_valid": True, "lower_ok": lower_ok, "exec_ok": exec_ok,
            "output_match": res["status"] == "match", "detail": detail}


_func.DISPATCH.setdefault("scf", run_scf)
