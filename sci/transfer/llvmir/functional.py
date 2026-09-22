"""Gold-differential functional scoring of LLVM IR functions under `lli`.

Protocol per prompt (mirrors `sci.eval.differential`), K=5 seeded trials:
  1. Gold gate: for each trial draw inputs from the reference's input domains,
     wrap the gold function in a driver module whose `main` calls it with the
     literal arguments and prints the scalar result through `printf`, run it
     twice under `lli`, and require both runs to succeed with identical
     output. Prompts whose gold fails the gate are not scorable.
  2. Differential: a candidate whose `define` signature (argument types and
     result type) equals the gold's is run with the same drivers; it matches
     iff every trial executes and agrees with the gold (ints exact, floats
     rel 1e-4 / abs 1e-5). A candidate with a different signature is a
     `signature_mismatch`; one without a parseable `define` is `no_define`.

Reference rows (`data/functional/llvm_references.jsonl`):
  {prompt_id, id, fn_name, signature: {args: [ty...], ret: ty},
   input_domains: [{type, min, max, nonzero?} | {type: "float"/"double", min, max}]}
"""
from __future__ import annotations

import hashlib
import random
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LLI = REPO / "scripts" / "env" / "bin" / "lli"

K_TRIALS = 5
FLOAT_REL = 1e-4
FLOAT_ABS = 1e-5
INT_TYPES = ("i1", "i32", "i64")
FLOAT_TYPES = ("float", "double")

_DEFINE_RE = re.compile(r"define\s+(?:[a-z_]+\s+)*?(i1|i32|i64|float|double|void)\s+@([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)")
_NUM_RE = re.compile(r"-?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)")


# ---------------- signatures ----------------

def parse_define(text: str) -> dict | None:
    """{name, ret, arg_types, arg_names} of the first `define`, or None."""
    m = _DEFINE_RE.search(text)
    if not m:
        return None
    arg_types, arg_names = [], []
    for piece in [p.strip() for p in m.group(3).split(",") if p.strip()]:
        parts = piece.split()
        if not parts:
            return None
        arg_types.append(parts[0])
        arg_names.append(parts[1] if len(parts) > 1 else "")
    return {"name": m.group(2), "ret": m.group(1), "arg_types": arg_types, "arg_names": arg_names}


def signature_of_ref(ref: dict) -> dict:
    return {"name": ref.get("fn_name", "f"), "ret": ref["signature"]["ret"],
            "arg_types": list(ref["signature"]["args"]), "arg_names": []}


def signature_compatible(gold_sig: dict, cand_sig: dict) -> bool:
    return gold_sig["ret"] == cand_sig["ret"] and gold_sig["arg_types"] == cand_sig["arg_types"]


# ---------------- inputs ----------------

def _rng_for(prompt_key: str, trial: int) -> random.Random:
    h = hashlib.sha256(f"llvm-differential:{prompt_key}:{trial}".encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def gen_inputs(ref: dict, trial: int) -> list:
    """Deterministic literal argument values for one trial (ints, quarter-step floats, bools)."""
    rng = _rng_for(ref.get("id", str(ref.get("prompt_id"))), trial)
    domains = ref.get("input_domains") or [{"type": t} for t in ref["signature"]["args"]]
    values = []
    for d in domains:
        ty = d["type"]
        if ty == "i1":
            values.append(rng.randint(0, 1))
        elif ty in INT_TYPES:
            lo, hi = int(d.get("min", -20)), int(d.get("max", 20))
            v = rng.randint(lo, hi)
            while d.get("nonzero") and v == 0:
                v = rng.randint(lo, hi)
            values.append(v)
        elif ty in FLOAT_TYPES:
            lo, hi = float(d.get("min", -8.0)), float(d.get("max", 8.0))
            v = rng.randrange(int(lo * 4), int(hi * 4) + 1) * 0.25
            while d.get("nonzero") and v == 0.0:
                v = rng.randrange(int(lo * 4), int(hi * 4) + 1) * 0.25
            values.append(v)
        else:
            raise ValueError(f"unsupported input type {ty}")
    return values


def literal(ty: str, v) -> str:
    if ty == "i1":
        return "true" if int(v) else "false"
    if ty in INT_TYPES:
        return str(int(v))
    return repr(float(v)) if "." in repr(float(v)) else f"{float(v)}.0"


# ---------------- driver ----------------

def build_driver(callee: str, sig: dict, values: list) -> str:
    """Module = callee + `main` that calls it with literal args and prints the result."""
    args = ", ".join(f"{t} {literal(t, v)}" for t, v in zip(sig["arg_types"], values))
    ret = sig["ret"]
    lines = [f"  %r = call {ret} @{sig['name']}({args})"]
    if ret in FLOAT_TYPES:
        src = "%r" if ret == "double" else "%w"
        if ret == "float":
            lines.append("  %w = fpext float %r to double")
        lines.append(f"  %c = call i32 (ptr, ...) @printf(ptr @.fmt, double {src})")
        fmt = '@.fmt = private unnamed_addr constant [6 x i8] c"%.9g\\0A\\00"'
    else:
        src = "%r" if ret == "i64" else "%w"
        if ret == "i1":
            lines.append("  %w = zext i1 %r to i64")
        elif ret == "i32":
            lines.append("  %w = sext i32 %r to i64")
        lines.append(f"  %c = call i32 (ptr, ...) @printf(ptr @.fmt, i64 {src})")
        fmt = '@.fmt = private unnamed_addr constant [6 x i8] c"%lld\\0A\\00"'
    return (f"{fmt}\ndeclare i32 @printf(ptr, ...)\n\n{callee.strip()}\n\n"
            "define i32 @main() {\nentry:\n" + "\n".join(lines) + "\n  ret i32 0\n}\n")


def run_lli(driver: str, timeout: float = 30.0) -> dict:
    """{exec_ok, stdout, err} from `lli` on the driver module (stdin)."""
    try:
        r = subprocess.run([str(LLI), "-"], input=driver, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"exec_ok": False, "stdout": "", "err": "exec_timeout"}
    return {"exec_ok": r.returncode == 0, "stdout": r.stdout or "", "err": (r.stderr or "")[:300]}


def parse_result(stdout: str) -> float | None:
    m = _NUM_RE.search(stdout.strip())
    return float(m.group(0)) if m else None


def _num_eq(a: float, b: float, exact: bool) -> bool:
    if a != a and b != b:
        return True
    if exact:
        return a == b
    return abs(a - b) <= FLOAT_ABS + FLOAT_REL * max(abs(a), abs(b))


# ---------------- gate + differential ----------------

def gold_gate(gold: str, ref: dict, k: int = K_TRIALS) -> dict:
    """Run the gold twice per trial; {ok, outputs: [float per trial], err}."""
    sig = parse_define(gold)
    if sig is None or sig["name"] == "main":
        return {"ok": False, "outputs": [], "err": "gold_no_define"}
    outputs = []
    for t in range(k):
        driver = build_driver(gold, sig, gen_inputs(ref, t))
        ra = run_lli(driver)
        if not ra["exec_ok"]:
            return {"ok": False, "outputs": [], "err": f"trial{t}:exec:{ra['err'][:200]}"}
        rb = run_lli(driver)
        if not rb["exec_ok"] or ra["stdout"] != rb["stdout"]:
            return {"ok": False, "outputs": [], "err": f"trial{t}:nondeterministic"}
        v = parse_result(ra["stdout"])
        if v is None:
            return {"ok": False, "outputs": [], "err": f"trial{t}:no_output"}
        outputs.append(v)
    return {"ok": True, "outputs": outputs, "err": ""}


def differential(candidate: str, gold: str, ref: dict, gold_outputs: list[float] | None = None,
                 k: int = K_TRIALS) -> dict:
    """Score `candidate` against `gold` on K seeded trials.

    status: match | mismatch | signature_mismatch | no_define | fn_named_main | gold_gate_fail.
    """
    if gold_outputs is None:
        gate = gold_gate(gold, ref, k)
        if not gate["ok"]:
            return {"status": "gold_gate_fail", "err": gate["err"], "n_trials_matched": 0, "trials": []}
        gold_outputs = gate["outputs"]
    gsig = parse_define(gold)
    csig = parse_define(candidate)
    if csig is None:
        return {"status": "no_define", "n_trials_matched": 0, "trials": []}
    if csig["name"] == "main":
        return {"status": "fn_named_main", "n_trials_matched": 0, "trials": []}
    if not signature_compatible(gsig, csig):
        return {"status": "signature_mismatch", "cand_sig": csig, "n_trials_matched": 0, "trials": []}
    exact = gsig["ret"] in INT_TYPES
    trials, n_match, first_err = [], 0, ""
    for t in range(k):
        rc = run_lli(build_driver(candidate, csig, gen_inputs(ref, t)))
        got = parse_result(rc["stdout"]) if rc["exec_ok"] else None
        ok = got is not None and _num_eq(got, gold_outputs[t], exact)
        n_match += ok
        if not ok and not first_err:
            first_err = "exec_fail" if not rc["exec_ok"] else "output_mismatch"
        trials.append({"trial": t, "exec_ok": rc["exec_ok"], "match": ok, "got": got,
                       "want": gold_outputs[t], "err": rc["err"][:120]})
    return {"status": "match" if n_match == k else "mismatch", "n_trials_matched": n_match,
            "first_divergence": first_err, "trials": trials}
