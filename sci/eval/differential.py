"""Gold-differential functional scoring on randomized inputs.

Scores candidate generations against gold reference modules by executing both
on the same seeded random inputs and comparing outputs.

Protocol per prompt, K=5 seeded trials (trial t uses square size n = 3 + t for
every dynamic dim; static dims come from the gold signature):

  1. HARD GATE (gold-vs-gold): build a driver module around the gold function
     with seeded random inputs (memref.global dense data, in-bounds index
     scalars, nonzero small ints, quarter-step floats), lower it
     (linalg->loops, scf->cf, math/arith/cf/memref/func->llvm), execute twice
     under mlir-cpu-runner in the pinned LLVM container, and require both runs
     to succeed with exactly equal parsed outputs on ALL K trials. Prompts that
     fail the gate are EXCLUDED with a counted reason; no candidate is scored
     against an unvalidated reference.
  2. DIFFERENTIAL: for a verify-valid candidate whose parsed signature is
     compatible with the gold's, run the SAME drivers (same input bytes) around
     the candidate function and compare parsed outputs to the gold's (ints
     exact; floats rel 1e-4 / abs 1e-5; every memref argument's final state is
     compared, plus the scalar result if any). A candidate is functionally
     correct on the prompt iff all K trials execute and match. Compatibility:
     scalar arg types and the result type must be identical; memref args must
     match in rank and dtype, and each candidate dim must be '?' or equal to the
     gold's dim (static-to-dynamic generalization is accepted and flagged
     dim_relaxed; a narrower candidate is not). Anything else counts as
     signature_mismatch: interface non-conformance the textual verifier cannot
     detect.

Observability rule: a prompt is in scope iff its gold function has a scalar
result and/or at least one memref argument (whose final state is printed).
Void-to-void prompts and unsupported types are excluded with counted reasons.

Output (under --out-dir): gold_gate.jsonl (one row per gold prompt),
differential.jsonl (one row per candidate trial), summary.json (rates + CIs).
"""
from __future__ import annotations

import argparse
import os
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LLVM_CONTAINER = os.environ.get("SCI_LLVM_CONTAINER", "sci-llvm")

K_TRIALS = 5
FLOAT_REL = 1e-4
FLOAT_ABS = 1e-5

PIPELINE = [
    "--convert-linalg-to-loops",
    "--convert-scf-to-cf",
    "--convert-math-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-cf-to-llvm",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
]

RUNNER = [
    "mlir-cpu-runner", "-e", "main", "-entry-point-result=i32",
    "--shared-libs=/opt/llvm/lib/libmlir_runner_utils.so",
    "--shared-libs=/opt/llvm/lib/libmlir_c_runner_utils.so",
]

INT_TYPES = {"i1", "i8", "i16", "i32", "i64"}
FLOAT_TYPES = {"f16", "f32", "f64"}
PRINTABLE_MEMREF_DTYPES = {"f32", "f64", "i32", "i64"}

_FN_RE = re.compile(
    r"func\.func\s+(?!private\b)@(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^\{]+?))?\s*\{",
)
_MEMREF_RE = re.compile(r"^memref<([0-9?x]*?)x?([fi]\d+)>$")


# ---------------- signature parsing ----------------

def parse_first_fn(mlir: str):
    """Return {name, arg_types, ret} for the first non-private func, or None.
    Types are whitespace-normalized strings."""
    m = _FN_RE.search(mlir)
    if not m:
        return None
    name = m.group(1)
    args_src = m.group(2).strip()
    arg_types = []
    if args_src:
        for piece in args_src.split(","):
            if ":" not in piece:
                return None
            arg_types.append(piece.rsplit(":", 1)[1].replace(" ", ""))
    ret = (m.group(3) or "").strip().replace(" ", "")
    return {"name": name, "arg_types": arg_types, "ret": ret}


def parse_memref_type(t: str):
    """'memref<?x?xf32>' -> (['?','?'], 'f32'); 'memref<32xf32>' -> (['32'],'f32').
    Returns None if not a supported ranked memref."""
    m = _MEMREF_RE.match(t)
    if not m:
        return None
    dims_src, dtype = m.group(1), m.group(2)
    if dims_src == "":
        return None  # 0-d memref unsupported
    dims = dims_src.split("x")
    if any(d != "?" and not d.isdigit() for d in dims):
        return None
    return dims, dtype


def signature_compatible(gold_sig, cand_sig) -> tuple[bool, bool]:
    """(compatible, dim_relaxed). Scalars/result identical; memrefs same rank
    and dtype with each candidate dim '?' or equal to the gold dim."""
    if cand_sig["ret"] != gold_sig["ret"]:
        return False, False
    if len(cand_sig["arg_types"]) != len(gold_sig["arg_types"]):
        return False, False
    relaxed = False
    for gt, ct in zip(gold_sig["arg_types"], cand_sig["arg_types"]):
        if gt == ct:
            continue
        gm, cm = parse_memref_type(gt), parse_memref_type(ct)
        if gm is None or cm is None:
            return False, False
        (gdims, gdt), (cdims, cdt) = gm, cm
        if gdt != cdt or len(gdims) != len(cdims):
            return False, False
        if not all(cd == "?" or cd == gd for gd, cd in zip(gdims, cdims)):
            return False, False
        relaxed = True
    return True, relaxed


def classify_signature(sig) -> tuple[str | None, str | None]:
    """Return (None, None) if supported, else (exclude_reason, detail)."""
    if sig is None:
        return "no_public_fn", None
    if sig["name"] == "main":
        return "fn_named_main", None
    ret = sig["ret"]
    if ret and (ret.startswith("(") or "," in ret):
        return "multi_result", ret
    if ret and ret not in INT_TYPES | FLOAT_TYPES | {"index"}:
        return "unsupported_ret_type", ret
    has_memref = False
    for t in sig["arg_types"]:
        if t.startswith("memref"):
            pm = parse_memref_type(t)
            if pm is None:
                return "unsupported_memref_type", t
            if pm[1] not in PRINTABLE_MEMREF_DTYPES:
                return "unprintable_memref_dtype", t
            has_memref = True
        elif t not in INT_TYPES | FLOAT_TYPES | {"index"}:
            return "unsupported_arg_type", t
    if not ret and not has_memref:
        return "no_observable_output", None
    return None, None


# ---------------- trial input generation ----------------

def _rng_for(prompt_key: str, trial: int) -> random.Random:
    h = hashlib.sha256(f"differential:{prompt_key}:{trial}".encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def gen_trial(sig, prompt_key: str, trial: int) -> dict:
    """Concrete shapes + input data for one trial. Deterministic in
    (prompt_key, trial); identical bytes are fed to gold and candidate."""
    rng = _rng_for(prompt_key, trial)
    n = 3 + trial  # every dynamic dim in this trial

    shapes = []       # per arg: list[int] for memrefs else None
    min_dim = None
    for t in sig["arg_types"]:
        if t.startswith("memref"):
            dims, _ = parse_memref_type(t)
            conc = [n if d == "?" else int(d) for d in dims]
            shapes.append(conc)
            for d in conc:
                min_dim = d if min_dim is None else min(min_dim, d)
        else:
            shapes.append(None)

    def rand_float():
        return rng.randrange(2, 33) * 0.25  # [0.5, 8.0], exactly representable

    def rand_int_scalar(ty):
        if ty == "i1":
            return rng.randint(0, 1)
        if min_dim is not None:
            return rng.randint(0, min_dim - 1)  # in-bounds if used as index
        return rng.randint(1, 7)               # nonzero: div/shift safe

    values = []
    for t, shape in zip(sig["arg_types"], shapes):
        if shape is not None:
            _, dtype = parse_memref_type(t)
            count = 1
            for d in shape:
                count *= d
            if dtype.startswith("f"):
                values.append([rand_float() for _ in range(count)])
            else:
                values.append([rng.randint(1, 7) for _ in range(count)])
        elif t in FLOAT_TYPES:
            values.append(rand_float())
        elif t == "index" or t in INT_TYPES:
            values.append(rand_int_scalar(t))
        else:
            raise ValueError(t)
    return {"n": n, "shapes": shapes, "values": values}


# ---------------- driver module construction ----------------

def _dense_literal(shape: list[int], flat: list, is_float: bool) -> str:
    def fmt(v):
        return (f"{float(v)}" if is_float else str(int(v)))

    def build(dims, offset, stride_done):
        if len(dims) == 1:
            return "[" + ", ".join(fmt(flat[offset + i]) for i in range(dims[0])) + "]"
        inner = 1
        for d in dims[1:]:
            inner *= d
        return "[" + ", ".join(build(dims[1:], offset + i * inner, None)
                               for i in range(dims[0])) + "]"

    return build(shape, 0, None)


def strip_outer_module(mlir: str) -> str:
    s = mlir.strip()
    if not s.startswith("module"):
        return s
    i = s.find("{")
    if i < 0:
        return s
    depth, end = 0, -1
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    return s[i + 1:end].strip() if end > 0 else s


def build_driver(callee_body: str, fn_name: str, sig, trial_data: dict,
                 callee_types: list[str] | None = None) -> str:
    """Full module: globals with the trial's input data + the callee + a main
    that calls it, prints the scalar result (if any), then prints every
    memref argument's final state. Shapes/data always come from the GOLD
    signature (sig + trial_data); callee_types (default: gold's) are the types
    the call is cast to and invoked with, so a dim-relaxed candidate receives
    the exact same buffers through its own declared interface."""
    if callee_types is None:
        callee_types = sig["arg_types"]
    globals_lines: list[str] = []
    main_lines: list[str] = []
    decls: set[str] = set()
    arg_names: list[str] = []
    memref_prints: list[str] = []

    for k, (t, shape, val) in enumerate(
            zip(callee_types, trial_data["shapes"], trial_data["values"])):
        if shape is not None:
            dims, dtype = parse_memref_type(t)
            static_ty = f"memref<{'x'.join(str(d) for d in shape)}x{dtype}>"
            lit = _dense_literal(shape, val, dtype.startswith("f"))
            globals_lines.append(
                f'  memref.global "private" @g{k} : {static_ty} = dense<{lit}>')
            main_lines.append(f"    %m{k} = memref.get_global @g{k} : {static_ty}")
            if t == static_ty.replace(" ", ""):
                arg_names.append(f"%m{k}")
            else:
                main_lines.append(f"    %a{k} = memref.cast %m{k} : {static_ty} to {t}")
                arg_names.append(f"%a{k}")
            pfn = "printMemref" + dtype.upper()[0] + dtype[1:]  # printMemrefF32 etc.
            decls.add(f"  func.func private @{pfn}(memref<*x{dtype}>) "
                      "attributes {llvm.emit_c_interface}")
            memref_prints.append(
                f"    %u{k} = memref.cast %m{k} : {static_ty} to memref<*x{dtype}>")
            memref_prints.append(
                f"    func.call @{pfn}(%u{k}) : (memref<*x{dtype}>) -> ()")
        elif t in FLOAT_TYPES:
            main_lines.append(f"    %s{k} = arith.constant {float(val)} : {t}")
            arg_names.append(f"%s{k}")
        else:  # int / index
            main_lines.append(f"    %s{k} = arith.constant {int(val)} : {t}")
            arg_names.append(f"%s{k}")

    ret = sig["ret"]
    call_sig = "(" + ", ".join(callee_types) + ")"
    call_args = ", ".join(arg_names)
    ret_lines: list[str] = []
    if ret:
        main_lines.append(
            f"    %r = func.call @{fn_name}({call_args}) : {call_sig} -> {ret}")
        if ret in FLOAT_TYPES:
            if ret == "f64":
                src = "%r"
            else:
                ret_lines.append(f"    %rw = arith.extf %r : {ret} to f64")
                src = "%rw"
            ret_lines.append(f"    func.call @printF64({src}) : (f64) -> ()")
            decls.add("  func.func private @printF64(f64)")
        else:
            if ret == "i64":
                src = "%r"
            elif ret == "index":
                ret_lines.append("    %rw = arith.index_cast %r : index to i64")
                src = "%rw"
            else:
                ret_lines.append(f"    %rw = arith.extsi %r : {ret} to i64")
                src = "%rw"
            ret_lines.append(f"    func.call @printI64({src}) : (i64) -> ()")
            decls.add("  func.func private @printI64(i64)")
        ret_lines.append("    func.call @printNewline() : () -> ()")
        decls.add("  func.func private @printNewline()")
    else:
        main_lines.append(
            f"    func.call @{fn_name}({call_args}) : {call_sig} -> ()")

    main = (
        "  func.func @main() -> i32 {\n"
        + "\n".join(main_lines) + "\n"
        + ("\n".join(ret_lines) + "\n" if ret_lines else "")
        + ("\n".join(memref_prints) + "\n" if memref_prints else "")
        + "    %zero = arith.constant 0 : i32\n"
        + "    return %zero : i32\n"
        + "  }"
    )
    return (
        "module {\n"
        + ("\n".join(sorted(decls)) + "\n" if decls else "")
        + ("\n".join(globals_lines) + "\n" if globals_lines else "")
        + "\n" + callee_body.rstrip() + "\n\n"
        + main + "\n}\n"
    )


# ---------------- docker lower + execute ----------------

def _docker(cmd: list[str], stdin_text: str, timeout: float):
    return subprocess.run(["docker", "exec", "-i", LLVM_CONTAINER] + cmd,
                          input=stdin_text, capture_output=True, text=True,
                          timeout=timeout)


def lower_and_run(driver: str) -> dict:
    """{lower_ok, exec_ok, stdout, err}"""
    try:
        r1 = _docker(["mlir-opt"] + PIPELINE, driver, 60.0)
    except subprocess.TimeoutExpired:
        return {"lower_ok": False, "exec_ok": False, "stdout": "",
                "err": "lower_timeout"}
    if r1.returncode != 0:
        return {"lower_ok": False, "exec_ok": False, "stdout": "",
                "err": (r1.stderr or "")[:300]}
    try:
        r2 = _docker(RUNNER, r1.stdout, 60.0)
    except subprocess.TimeoutExpired:
        return {"lower_ok": True, "exec_ok": False, "stdout": "",
                "err": "exec_timeout"}
    if r2.returncode != 0:
        return {"lower_ok": True, "exec_ok": False, "stdout": r2.stdout or "",
                "err": (r2.stderr or "")[:300]}
    return {"lower_ok": True, "exec_ok": True, "stdout": r2.stdout or "", "err": ""}


# ---------------- output parsing + comparison ----------------

_NUM_RE = re.compile(r"-?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)")


def parse_stdout(s: str) -> dict:
    """{'scalars': [floats], 'memrefs': [{'sizes': [...], 'data': [...]}]}
    Scalar result (if any) is printed before the first memref block."""
    parts = s.split("Unranked Memref")
    scalars = [float(x) for x in _NUM_RE.findall(parts[0])]
    memrefs = []
    for block in parts[1:]:
        m = re.search(r"sizes = \[([^\]]*)\]", block)
        sizes = [int(x) for x in m.group(1).split(",")] if m and m.group(1).strip() else []
        data = []
        di = block.find("data =")
        if di >= 0:
            seg = block[di + 6:]
            # take only the balanced bracket block: trailing text (e.g. the
            # runner echoing main's exit code) must not leak into the data
            start = seg.find("[")
            if start >= 0:
                depth, end = 0, -1
                for j in range(start, len(seg)):
                    if seg[j] == "[":
                        depth += 1
                    elif seg[j] == "]":
                        depth -= 1
                        if depth == 0:
                            end = j
                            break
                seg = seg[start:end + 1] if end > 0 else seg
            data = [float(x) for x in _NUM_RE.findall(seg)]
        memrefs.append({"sizes": sizes, "data": data})
    return {"scalars": scalars, "memrefs": memrefs}


def _num_eq(a: float, b: float, exact: bool) -> bool:
    if a != a and b != b:  # both NaN
        return True
    if exact:
        return a == b
    return abs(a - b) <= FLOAT_ABS + FLOAT_REL * max(abs(a), abs(b))


def outputs_equal(x: dict, y: dict, exact: bool = False) -> bool:
    if len(x["scalars"]) != len(y["scalars"]):
        return False
    if len(x["memrefs"]) != len(y["memrefs"]):
        return False
    for a, b in zip(x["scalars"], y["scalars"]):
        if not _num_eq(a, b, exact):
            return False
    for ma, mb in zip(x["memrefs"], y["memrefs"]):
        if ma["sizes"] != mb["sizes"] or len(ma["data"]) != len(mb["data"]):
            return False
        for a, b in zip(ma["data"], mb["data"]):
            if not _num_eq(a, b, exact):
                return False
    return True


# ---------------- stats ----------------

def bootstrap_ci(flags: list[bool], n_boot: int = 10_000, seed: int = 0):
    rng = random.Random(seed)
    n = len(flags)
    if n == 0:
        return [0.0, 0.0]
    rates = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return [rates[int(0.025 * n_boot)], rates[int(0.975 * n_boot)]]


# ---------------- benchmark + candidate loading ----------------

DIALECTS = {
    "arith": {
        "gold_jsonl": "data/benchmarks/mlir_spec_150.jsonl",
        "cand_dialect": "arith+func",
        "n_spec": 150,
    },
    "linalg": {
        "gold_jsonl": "data/benchmarks/linalg_spec_30.jsonl",
        "cand_dialect": "linalg",
        "n_spec": 30,
    },
}


def load_candidates(path: Path, cand_dialect: str, model_substr: str = "",
                    seed: int | None = None) -> dict[int, dict]:
    """Candidate rows keyed by int prompt_id (row schema: model, dialect, seed,
    prompt_id, generated, verify_valid, ...). prompt_id i corresponds to the
    i-th gold row of the dialect's benchmark file."""
    out: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if model_substr and model_substr not in str(r.get("model", "")):
            continue
        if r.get("dialect") != cand_dialect:
            continue
        if seed is not None and int(r.get("seed", -1)) != seed:
            continue
        out[int(r["prompt_id"])] = r
    return out


# ---------------- main loop ----------------

def run(out_dir: Path, dialects: list[str], limit: int | None, candidates: Path,
        model_substr: str = "", seed: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_path = out_dir / "gold_gate.jsonl"
    diff_path = out_dir / "differential.jsonl"
    summary_path = out_dir / "summary.json"

    done_gate = set()
    if gate_path.exists():
        for line in gate_path.read_text().splitlines():
            r = json.loads(line)
            done_gate.add((r["dialect"], r["id"]))
    gate_f = gate_path.open("a")
    diff_f = diff_path.open("a")

    summary: dict = {"k_trials": K_TRIALS, "float_rel": FLOAT_REL,
                     "float_abs": FLOAT_ABS, "candidates": str(candidates),
                     "dialects": {}}
    t0 = time.time()

    for dkey in dialects:
        cfg = DIALECTS[dkey]
        examples = [json.loads(l) for l in (REPO / cfg["gold_jsonl"]).read_text().splitlines() if l.strip()]
        if limit:
            examples = examples[:limit]
        cands = load_candidates(candidates, cfg["cand_dialect"], model_substr, seed)
        print(f"[differential] {dkey}: {len(examples)} gold prompts, "
              f"{len(cands)} candidate rows", file=sys.stderr)

        gate_rows: list[dict] = []
        diff_prompt_rows: list[dict] = []

        for idx, ex in enumerate(examples):
            ex_id = ex["id"]
            if (dkey, ex_id) in done_gate:
                # already gated in a previous (resumed) pass; reload the row
                for line in gate_path.read_text().splitlines():
                    r = json.loads(line)
                    if r["dialect"] == dkey and r["id"] == ex_id:
                        gate_rows.append(r)
                        break
                continue

            gold = ex["mlir"]
            sig = parse_first_fn(gold)
            reason, detail = classify_signature(sig)
            row = {"dialect": dkey, "id": ex_id, "idx": idx}

            if reason:
                row.update({"gate": "excluded", "reason": reason, "detail": detail})
                gate_rows.append(row)
                gate_f.write(json.dumps(row) + "\n"); gate_f.flush()
                print(f"  [{dkey}] {ex_id}: EXCLUDED ({reason})", file=sys.stderr)
                continue

            prompt_key = f"{dkey}:{ex_id}"
            gold_body = strip_outer_module(gold)
            gate_ok, gate_err = True, ""
            gold_outs = {}
            for t in range(K_TRIALS):
                td = gen_trial(sig, prompt_key, t)
                driver = build_driver(gold_body, sig["name"], sig, td)
                ra = lower_and_run(driver)
                if not ra["exec_ok"]:
                    gate_ok, gate_err = False, f"trial{t}:{'lower' if not ra['lower_ok'] else 'exec'}:{ra['err'][:200]}"
                    break
                rb = lower_and_run(driver)
                if not rb["exec_ok"]:
                    gate_ok, gate_err = False, f"trial{t}:rerun:{rb['err'][:200]}"
                    break
                pa, pb = parse_stdout(ra["stdout"]), parse_stdout(rb["stdout"])
                if not outputs_equal(pa, pb, exact=True):
                    gate_ok, gate_err = False, f"trial{t}:nondeterministic"
                    break
                gold_outs[t] = pa
            row.update({"gate": "pass" if gate_ok else "fail",
                        "reason": None if gate_ok else gate_err,
                        "sig": sig})
            gate_rows.append(row)
            gate_f.write(json.dumps(row) + "\n"); gate_f.flush()
            print(f"  [{dkey}] {ex_id}: gate={'PASS' if gate_ok else 'FAIL ' + (gate_err or '')[:80]}",
                  file=sys.stderr)
            if not gate_ok:
                continue

            # ---- differential against the frozen candidate ----
            cand = cands.get(idx)
            prow = {"dialect": dkey, "id": ex_id, "idx": idx}
            if cand is not None and cand.get("nl") and cand["nl"] != ex["nl"]:
                prow["cand_status"] = "nl_mismatch"
            elif cand is None:
                prow["cand_status"] = "no_candidate_row"
            elif not cand.get("verify_valid"):
                prow["cand_status"] = "not_verify_valid"
            else:
                csig = parse_first_fn(cand["generated"])
                compat, dim_relaxed = (signature_compatible(sig, csig)
                                       if csig else (False, False))
                if csig is None:
                    prow["cand_status"] = "no_parseable_fn"
                elif not compat:
                    prow["cand_status"] = "signature_mismatch"
                    prow["cand_sig"] = csig
                elif csig["name"] == "main":
                    prow["cand_status"] = "fn_named_main"
                else:
                    cbody = strip_outer_module(cand["generated"])
                    prow["dim_relaxed"] = dim_relaxed
                    n_match = 0
                    first_err = ""
                    for t in range(K_TRIALS):
                        td = gen_trial(sig, prompt_key, t)
                        driver = build_driver(cbody, csig["name"], sig, td,
                                              callee_types=csig["arg_types"])
                        rc = lower_and_run(driver)
                        trial_match = (rc["exec_ok"]
                                       and outputs_equal(parse_stdout(rc["stdout"]),
                                                         gold_outs[t]))
                        if trial_match:
                            n_match += 1
                        elif not first_err:
                            first_err = ("lower_fail" if not rc["lower_ok"]
                                         else "exec_fail" if not rc["exec_ok"]
                                         else "output_mismatch")
                        diff_f.write(json.dumps({
                            "dialect": dkey, "id": ex_id, "idx": idx, "trial": t,
                            "lower_ok": rc["lower_ok"], "exec_ok": rc["exec_ok"],
                            "match": trial_match, "err": rc["err"][:200],
                        }) + "\n")
                        diff_f.flush()
                    prow["cand_status"] = "scored"
                    prow["n_trials_matched"] = n_match
                    prow["functional_match"] = (n_match == K_TRIALS)
                    prow["first_divergence"] = first_err
            diff_prompt_rows.append(prow)
            status = prow.get("cand_status")
            extra = (f" match={prow.get('n_trials_matched')}/{K_TRIALS}"
                     if status == "scored" else "")
            print(f"  [{dkey}] {ex_id}: cand={status}{extra}", file=sys.stderr)

            # live summary after every prompt
            summary["dialects"][dkey] = _aggregate(gate_rows, diff_prompt_rows)
            summary["elapsed_min"] = round((time.time() - t0) / 60.0, 1)
            summary_path.write_text(json.dumps(summary, indent=2))

        summary["dialects"][dkey] = _aggregate(gate_rows, diff_prompt_rows)
        summary_path.write_text(json.dumps(summary, indent=2))

    summary["elapsed_min"] = round((time.time() - t0) / 60.0, 1)
    summary_path.write_text(json.dumps(summary, indent=2))
    gate_f.close(); diff_f.close()
    print(f"[differential] DONE in {summary['elapsed_min']} min", file=sys.stderr)
    print(json.dumps(summary, indent=2), file=sys.stderr)


def _aggregate(gate_rows: list[dict], prompt_rows: list[dict]) -> dict:
    from collections import Counter
    n_gold = len(gate_rows)
    excluded = Counter(r["reason"] for r in gate_rows if r["gate"] == "excluded")
    gate_fail = [r["id"] for r in gate_rows if r["gate"] == "fail"]
    gate_pass = sum(1 for r in gate_rows if r["gate"] == "pass")
    cstat = Counter(r.get("cand_status") for r in prompt_rows)
    scored = [r for r in prompt_rows if r.get("cand_status") == "scored"]
    matched = [r for r in scored if r.get("functional_match")]
    match_flags = [bool(r.get("functional_match")) for r in scored]
    # conservative view: every gate-passed prompt counts; non-scored = no match
    all_flags = [bool(r.get("functional_match")) for r in prompt_rows]
    out = {
        "n_gold": n_gold,
        "gold_excluded": dict(excluded),
        "gold_gate_fail": gate_fail,
        "gold_gate_pass": gate_pass,
        "cand_status_counts": dict(cstat),
        "n_scored": len(scored),
        "n_scored_dim_relaxed": sum(1 for r in scored if r.get("dim_relaxed")),
        "n_functional_match": len(matched),
        "match_rate_among_scored": (len(matched) / len(scored)) if scored else None,
        "match_ci95_among_scored": bootstrap_ci(match_flags) if scored else None,
        "match_rate_among_gate_passed": (sum(all_flags) / len(all_flags))
                                         if all_flags else None,
        "match_ci95_among_gate_passed": bootstrap_ci(all_flags) if all_flags else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--candidates", required=True, help="JSONL of candidate generations")
    ap.add_argument("--model-substr", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dialects", default="arith,linalg")
    ap.add_argument("--limit", type=int, default=None,
                    help="only first N gold prompts per dialect (smoke)")
    args = ap.parse_args()
    run(Path(args.out_dir), args.dialects.split(","), args.limit, Path(args.candidates),
        args.model_substr, args.seed)


if __name__ == "__main__":
    main()
