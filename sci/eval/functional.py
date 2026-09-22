"""Functional execution runner for the n=30 mini-benchmark.

For each (candidate generation, reference) pair, reports four columns:
  - verify_valid: parses + verifies under dialect-specific verify
  - lower_ok:    lowers cleanly through the dialect-specific pipeline
  - exec_ok:     compiled + executed without runtime error
  - output_match: stdout matches the reference's expected_stdout_regex

The candidate generation is taken from the SmolLM2 + C1+C2+C3 cell
(greedy, seed=0). For arith / linalg cells we read from the day-39-PM
multiseed file (seed 1 — not seed 0; seed 0 is the un-uniform-n run).
For StableHLO we read from the day-42 (held_out) or day-32 (spec_30)
cells. Caller can override via --candidates-jsonl.

This is evaluation evidence, NOT a benchmark release. See
eval/functional/__init__.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LLVM_CONTAINER = os.environ.get("SCI_LLVM_CONTAINER", "sci-llvm")


# ---------------- generation parsing ----------------

# Match `func.func @<name>(<args>) -> <ret>` (or no return) — first non-private.
_FN_RE = re.compile(
    r"func\.func\s+(?!private\b)@(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^\s{]+))?",
    re.MULTILINE,
)


def detect_first_function(mlir: str) -> tuple[str, str, str | None] | None:
    """Return (fn_name, args_str, return_type_str_or_None), or None if not found."""
    m = _FN_RE.search(mlir)
    if not m:
        return None
    return m.group(1), m.group(2).strip(), (m.group(3).strip() if m.group(3) else None)


def strip_outer_module(mlir: str) -> str:
    """If the generation is `module { ... }`, return the inner body. Otherwise
    return as-is."""
    s = mlir.strip()
    if s.startswith("module"):
        # find first { after 'module'
        i = s.find("{")
        if i < 0:
            return mlir
        # find matching closing brace at the end
        # count braces
        depth = 0
        end = -1
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end < 0:
            return mlir
        return s[i + 1:end].strip()
    return mlir


# ---------------- arith+func runner ----------------

ARITH_PIPELINE = [
    "--convert-arith-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    "--reconcile-unrealized-casts",
]


def _arith_input_lit(ty: str, val: Any) -> str:
    """Generate `arith.constant` SSA + return its name. Caller emits the lines
    and uses the returned name."""
    raise NotImplementedError("inlined below")


def _build_arith_wrapper(generated_body: str, fn_name: str,
                         inputs: list[dict], result_type: str) -> str:
    """Build a complete `module { ... }` with the candidate function plus a
    main that calls it with hardcoded input constants and prints the result
    via printI64 / printF64."""
    arg_lines = []
    arg_names = []
    for k, inp in enumerate(inputs):
        name = f"%i{k}"
        ty = inp["type"]
        v = inp["value"]
        if ty.startswith("i"):
            arg_lines.append(f"  {name} = arith.constant {int(v)} : {ty}")
        else:  # f-types
            # MLIR float literal needs a dot.
            vs = repr(float(v)) if "." in repr(float(v)) else f"{float(v)}"
            arg_lines.append(f"  {name} = arith.constant {vs} : {ty}")
        arg_names.append(name)

    arg_types = ", ".join(inp["type"] for inp in inputs)
    arg_uses = ", ".join(arg_names) if arg_names else ""
    has_args = bool(arg_names)

    # Result printing
    rt = result_type
    if rt.startswith("i"):
        # widen to i64
        if rt == "i64":
            cast_lines = [f"  %r64 = arith.bitcast %r : i64 to i64"]
            cast_lines = []  # no cast needed; %r is already i64
            print_call = "  func.call @printI64(%r) : (i64) -> ()"
        else:
            # extend to i64
            cast_lines = [f"  %r64 = arith.extsi %r : {rt} to i64"]
            print_call = "  func.call @printI64(%r64) : (i64) -> ()"
        print_decl = "  func.func private @printI64(i64) "
    elif rt.startswith("f"):
        if rt == "f64":
            cast_lines = []
            print_call = "  func.call @printF64(%r) : (f64) -> ()"
        else:
            cast_lines = [f"  %r64 = arith.extf %r : {rt} to f64"]
            print_call = "  func.call @printF64(%r64) : (f64) -> ()"
        print_decl = "  func.func private @printF64(f64) "
    else:
        raise ValueError(f"unsupported result type {rt}")

    sig = f"({arg_types}) -> {rt}"
    if not has_args:
        sig = f"() -> {rt}"

    main = (
        "  func.func @main() -> i32 {\n"
        + ("\n".join(arg_lines) + "\n" if arg_lines else "")
        + f"    %r = func.call @{fn_name}({arg_uses}) : {sig}\n"
        + ("\n".join("  " + l for l in cast_lines) + "\n" if cast_lines else "")
        + "  " + print_call + "\n"
        + "    %z = arith.constant 0 : i32\n"
        + "    return %z : i32\n"
        + "  }"
    )

    return (
        "module {\n"
        + print_decl + "\n\n"
        + generated_body.rstrip() + "\n\n"
        + main + "\n"
        + "}\n"
    )


def _docker_exec(cmd_argv: list[str], stdin_text: str | None = None,
                 timeout: float = 60.0) -> subprocess.CompletedProcess:
    """Run a command inside the pinned LLVM container, streaming stdin if given."""
    full = ["docker", "exec", "-i", LLVM_CONTAINER] + cmd_argv
    return subprocess.run(
        full, input=stdin_text, capture_output=True, text=True, timeout=timeout,
    )


def run_arith(generated: str, ref: dict) -> dict:
    """Returns {verify_valid, lower_ok, exec_ok, output_match, detail}."""
    detail: dict[str, Any] = {}

    # 1. verify_valid
    detail["dialect"] = "arith+func"
    r1 = _docker_exec(["mlir-opt", "--verify-diagnostics", "--allow-unregistered-dialect"],
                      stdin_text=generated, timeout=15.0)
    verify_valid = (r1.returncode == 0)
    if not verify_valid:
        detail["verify_stderr"] = (r1.stderr or "")[:300]
        return {"verify_valid": False, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": detail}

    # Detect function name
    det = detect_first_function(generated)
    if det is None:
        detail["err"] = "no func.func found"
        return {"verify_valid": True, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": detail}
    fn_name, _args, _rt = det

    # 2. Build the wrapper module + try to lower
    body = strip_outer_module(generated)
    try:
        wrapper = _build_arith_wrapper(body, fn_name, ref["inputs"], ref["result_type"])
    except Exception as e:
        detail["wrapper_err"] = str(e)
        return {"verify_valid": True, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": detail}

    r2 = _docker_exec(["mlir-opt"] + ARITH_PIPELINE,
                      stdin_text=wrapper, timeout=20.0)
    if r2.returncode != 0:
        detail["lower_stderr"] = (r2.stderr or "")[:300]
        return {"verify_valid": True, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": detail}
    lowered_mlir = r2.stdout

    # 3. mlir-translate to LLVM IR + lli execute
    r3 = _docker_exec(["mlir-translate", "--mlir-to-llvmir"],
                      stdin_text=lowered_mlir, timeout=20.0)
    if r3.returncode != 0:
        detail["translate_stderr"] = (r3.stderr or "")[:300]
        return {"verify_valid": True, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": detail}
    llvm_ir = r3.stdout

    # lli expects LLVM IR via stdin (or file). Pass via -.
    r4 = _docker_exec(
        ["lli", "-load", "/opt/llvm/lib/libmlir_c_runner_utils.so",
         "-load", "/opt/llvm/lib/libmlir_runner_utils.so", "-"],
        stdin_text=llvm_ir, timeout=20.0,
    )
    exec_ok = (r4.returncode == 0)
    stdout = r4.stdout or ""
    if not exec_ok:
        detail["exec_stderr"] = (r4.stderr or "")[:300]
        return {"verify_valid": True, "lower_ok": True, "exec_ok": False,
                "output_match": False, "detail": detail}

    # 4. Match
    rx = re.compile(ref["expected_stdout_regex"], re.MULTILINE)
    output_match = bool(rx.search(stdout))
    detail["stdout"] = stdout[:300]
    return {"verify_valid": True, "lower_ok": True, "exec_ok": True,
            "output_match": output_match, "detail": detail}


# ---------------- linalg+memref runner ----------------

LINALG_PIPELINE = [
    "--convert-linalg-to-loops",
    "--convert-scf-to-cf",
    "--convert-arith-to-llvm",
    "--convert-cf-to-llvm",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
]


def _build_linalg_wrapper(generated_body: str, fn_name: str, ref: dict) -> str:
    """Build a wrapper with memref.alloc + linalg.fill seeding inputs, calling
    the candidate, then printing the requested output memrefs."""
    inputs = ref.get("memref_inputs", [])
    scalars = ref.get("scalar_inputs", [])
    print_targets = ref.get("memref_print", [])

    # Static or dynamic? Use static if all inputs declared static.
    def _shape_str(shape: list[int], dyn: bool) -> str:
        if dyn:
            return "x".join("?" * len(shape))
        return "x".join(str(s) for s in shape)

    # Candidate signature (?-shaped if any input has dynamic)
    # For matching the function signature, we use the dynamic form where the
    # canonical signature uses ? — else static. To stay compatible with the
    # generated function (which uses ? if the spec uses ?), we always use
    # dynamic memrefs in the wrapper unless a memref input is marked static.
    arg_lines: list[str] = []
    arg_names: list[str] = []
    arg_types: list[str] = []
    cleanup_lines: list[str] = []

    # Scalars first (per canonical signature ordering — fp value as first param
    # for fill_value variants).
    for k, sc in enumerate(scalars):
        name = f"%s{k}"
        ty = sc["type"]
        v = sc["value"]
        if ty.startswith("f"):
            vs = repr(float(v))
            if "." not in vs:
                vs += ".0"
            arg_lines.append(f"  {name} = arith.constant {vs} : {ty}")
        else:
            arg_lines.append(f"  {name} = arith.constant {int(v)} : {ty}")
        arg_names.append(name)
        arg_types.append(ty)

    # Then memref inputs.
    for k, mi in enumerate(inputs):
        name = f"%m{k}"
        shape = mi["shape"]
        dtype = mi["dtype"]
        static = mi.get("static", False)
        memty = f"memref<{_shape_str(shape, dyn=not static)}x{dtype}>"
        if static:
            arg_lines.append(f"  {name} = memref.alloc() : {memty}")
        else:
            # dynamic alloc: pass dim sizes as %c<n>
            dim_args = []
            for d, sz in enumerate(shape):
                dn = f"%c{name[1:]}d{d}"
                arg_lines.append(f"  {dn} = arith.constant {sz} : index")
                dim_args.append(dn)
            arg_lines.append(f"  {name} = memref.alloc({', '.join(dim_args)}) : {memty}")
        arg_names.append(name)
        arg_types.append(memty)
        cleanup_lines.append(f"  memref.dealloc {name} : {memty}")

        init = mi.get("init", "alloc_only")
        if init == "fill":
            fv = mi["fill_value"]
            fname = f"%fv{k}"
            if dtype.startswith("f"):
                vs = repr(float(fv))
                if "." not in vs:
                    vs += ".0"
                arg_lines.append(f"  {fname} = arith.constant {vs} : {dtype}")
            else:
                arg_lines.append(f"  {fname} = arith.constant {int(fv)} : {dtype}")
            arg_lines.append(
                f"  linalg.fill ins({fname} : {dtype}) outs({name} : {memty})"
            )

    # Call signature
    call_args = ", ".join(arg_names) if arg_names else ""
    call_sig = "(" + ", ".join(arg_types) + ") -> ()"

    # Print targets (always cast to unranked for printMemref*).
    # The unranked-memref print intrinsics require the C-interface wrapper
    # because the memref descriptor needs the C-friendly struct layout.
    EMITC = "attributes {llvm.emit_c_interface}"
    print_decls = set()
    print_lines = []
    for k, pt in enumerate(print_targets):
        # Find the matching input by name (must already be declared above).
        target_idx = None
        for i, mi in enumerate(inputs):
            if mi["name"] == pt["name"]:
                target_idx = i
                break
        if target_idx is None:
            raise ValueError(f"print target {pt['name']} not in memref_inputs")
        target_name = f"%m{target_idx}"
        target_shape = inputs[target_idx]["shape"]
        target_static = inputs[target_idx].get("static", False)
        target_dtype = inputs[target_idx]["dtype"]
        memty = f"memref<{_shape_str(target_shape, dyn=not target_static)}x{target_dtype}>"
        if target_dtype == "f32":
            print_decls.add(f"  func.func private @printMemrefF32(memref<*xf32>) {EMITC}")
            cast_name = f"%pc{k}"
            print_lines.append(f"  {cast_name} = memref.cast {target_name} : {memty} to memref<*xf32>")
            print_lines.append(f"  func.call @printMemrefF32({cast_name}) : (memref<*xf32>) -> ()")
        elif target_dtype == "f64":
            print_decls.add(f"  func.func private @printMemrefF64(memref<*xf64>) {EMITC}")
            cast_name = f"%pc{k}"
            print_lines.append(f"  {cast_name} = memref.cast {target_name} : {memty} to memref<*xf64>")
            print_lines.append(f"  func.call @printMemrefF64({cast_name}) : (memref<*xf64>) -> ()")
        elif target_dtype == "i32":
            print_decls.add(f"  func.func private @printMemrefI32(memref<*xi32>) {EMITC}")
            cast_name = f"%pc{k}"
            print_lines.append(f"  {cast_name} = memref.cast {target_name} : {memty} to memref<*xi32>")
            print_lines.append(f"  func.call @printMemrefI32({cast_name}) : (memref<*xi32>) -> ()")
        elif target_dtype == "i64":
            print_decls.add(f"  func.func private @printMemrefI64(memref<*xi64>) {EMITC}")
            cast_name = f"%pc{k}"
            print_lines.append(f"  {cast_name} = memref.cast {target_name} : {memty} to memref<*xi64>")
            print_lines.append(f"  func.call @printMemrefI64({cast_name}) : (memref<*xi64>) -> ()")
        else:
            raise ValueError(f"no print intrinsic for memref<{target_dtype}>")

    main = (
        "  func.func @main() -> i32 {\n"
        + "\n".join(arg_lines) + "\n"
        + f"    func.call @{fn_name}({call_args}) : {call_sig}\n"
        + "\n".join(print_lines) + "\n"
        + "\n".join(cleanup_lines) + "\n"
        + "    %z = arith.constant 0 : i32\n"
        + "    return %z : i32\n"
        + "  }"
    )

    decls = "\n".join(sorted(print_decls))
    return (
        "module {\n"
        + decls + "\n\n"
        + generated_body.rstrip() + "\n\n"
        + main + "\n"
        + "}\n"
    )


def run_linalg(generated: str, ref: dict) -> dict:
    detail: dict[str, Any] = {"dialect": "linalg+memref"}

    r1 = _docker_exec(["mlir-opt", "--verify-diagnostics"],
                      stdin_text=generated, timeout=15.0)
    verify_valid = (r1.returncode == 0)
    if not verify_valid:
        detail["verify_stderr"] = (r1.stderr or "")[:300]
        return {"verify_valid": False, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": detail}

    det = detect_first_function(generated)
    if det is None:
        detail["err"] = "no func.func found"
        return {"verify_valid": True, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": detail}
    fn_name, _args, _rt = det

    body = strip_outer_module(generated)
    try:
        wrapper = _build_linalg_wrapper(body, fn_name, ref)
    except Exception as e:
        detail["wrapper_err"] = str(e)
        return {"verify_valid": True, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": detail}

    r2 = _docker_exec(["mlir-opt"] + LINALG_PIPELINE,
                      stdin_text=wrapper, timeout=30.0)
    if r2.returncode != 0:
        detail["lower_stderr"] = (r2.stderr or "")[:300]
        return {"verify_valid": True, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": detail}
    lowered = r2.stdout

    # mlir-cpu-runner accepts MLIR via stdin; needs --shared-libs for
    # printMemref*.
    r3 = _docker_exec(
        ["mlir-cpu-runner", "-e", "main", "-entry-point-result=i32",
         "--shared-libs=/opt/llvm/lib/libmlir_runner_utils.so",
         "--shared-libs=/opt/llvm/lib/libmlir_c_runner_utils.so"],
        stdin_text=lowered, timeout=30.0,
    )
    exec_ok = (r3.returncode == 0)
    stdout = r3.stdout or ""
    if not exec_ok:
        detail["exec_stderr"] = (r3.stderr or "")[:300]
        return {"verify_valid": True, "lower_ok": True, "exec_ok": False,
                "output_match": False, "detail": detail}

    rx = re.compile(ref["expected_stdout_regex"], re.MULTILINE | re.DOTALL)
    output_match = bool(rx.search(stdout))
    detail["stdout"] = stdout[:600]
    return {"verify_valid": True, "lower_ok": True, "exec_ok": True,
            "output_match": output_match, "detail": detail}


# ---------------- stablehlo runner ----------------

def run_stablehlo(generated: str, ref: dict) -> dict:
    """iree-compile + iree-run-module path. Uses host venv tools (no docker)."""
    detail: dict[str, Any] = {"dialect": "stablehlo"}

    venv_bin = REPO / ".venv" / "bin"
    iree_compile = venv_bin / "iree-compile"
    iree_run = venv_bin / "iree-run-module"

    # 1. verify_valid: iree-compile --compile-to=input
    from sci.eval.verify_stablehlo import verify_stablehlo
    vr = verify_stablehlo(generated)
    verify_valid = (vr["returncode"] == 0)
    if not verify_valid:
        detail["verify_stderr"] = vr.get("stderr", "")[:300]
        return {"verify_valid": False, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": detail}

    det = detect_first_function(generated)
    if det is None:
        detail["err"] = "no func.func found"
        return {"verify_valid": True, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": detail}
    fn_name = det[0]

    # 2. Compile to a vmfb bytecode for llvm-cpu
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as fmlir:
        fmlir.write(generated)
        mlir_path = fmlir.name
    vmfb_path = mlir_path.replace(".mlir", ".vmfb")
    try:
        rc = subprocess.run(
            [str(iree_compile),
             "--iree-input-type=stablehlo",
             "--iree-hal-target-backends=llvm-cpu",
             mlir_path, "-o", vmfb_path],
            capture_output=True, text=True, timeout=60.0,
        )
        if rc.returncode != 0:
            detail["compile_stderr"] = (rc.stderr or "")[:300]
            return {"verify_valid": True, "lower_ok": False, "exec_ok": False,
                    "output_match": False, "detail": detail}

        # 3. iree-run-module with the canonical inputs
        cmd = [str(iree_run), f"--module={vmfb_path}",
               f"--function={fn_name}", "--device=local-task"]
        for inp in ref["iree_inputs"]:
            cmd.append(f"--input={inp}")
        rr = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0)
        exec_ok = (rr.returncode == 0)
        stdout = rr.stdout or ""
        if not exec_ok:
            detail["run_stderr"] = (rr.stderr or "")[:300]
            return {"verify_valid": True, "lower_ok": True, "exec_ok": False,
                    "output_match": False, "detail": detail}

        rx = re.compile(ref["expected_output_pattern"], re.MULTILINE | re.DOTALL)
        output_match = bool(rx.search(stdout))
        detail["stdout"] = stdout[:600]
        return {"verify_valid": True, "lower_ok": True, "exec_ok": True,
                "output_match": output_match, "detail": detail}
    finally:
        for p in (mlir_path, vmfb_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


# ---------------- entry point ----------------

DISPATCH = {
    "arith+func": run_arith,
    "linalg+memref": run_linalg,
    "stablehlo": run_stablehlo,
}


def _load_candidates_from_jsonl(path: Path) -> dict[str, str]:
    """Map (source_benchmark_id) → generated MLIR. Best-effort key match."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        # Use prompt_id (int) as the key when possible.
        out[str(r.get("prompt_id"))] = r.get("generated", "")
    return out


def _load_candidate_for(ref: dict, candidates_by_dialect: dict) -> str | None:
    """Look up the candidate generation for this reference. Returns the
    generated MLIR string, or None if no candidate found."""
    src = ref["source_benchmark"]
    src_id = ref["source_id"]
    pool = candidates_by_dialect.get(src, {})
    # Try exact prompt_id (int prefix from source_id like "001_..." → 0).
    # Spec-150 ids are 1-indexed; jsonl prompt_id is 0-indexed → subtract 1.
    if "_" in src_id:
        head = src_id.split("_", 1)[0]
        try:
            n = int(head)
            return pool.get(str(n - 1))
        except ValueError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--references", default=str(REPO / "data/functional/references.json"))
    ap.add_argument("--candidates-arith", required=True, help="JSONL of arith+func candidate generations")
    ap.add_argument("--candidates-linalg", required=True, help="JSONL of linalg candidate generations")
    ap.add_argument("--candidates-stablehlo", required=True, help="JSONL of stablehlo candidate generations")
    ap.add_argument("--model-substr", default="", help="only rows whose model field contains this substring")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-filter", type=int, default=0,
                    help="Only consider rows where seed==this (0 = first seed in jsonl)")
    args = ap.parse_args()

    # Load candidates per dialect, filtered to SmolLM2 c1c2c3 cell at the chosen seed.
    def _filter_candidates(jsonl_path: Path, dialect: str) -> dict[str, str]:
        out: dict[str, str] = {}
        if not jsonl_path.exists():
            return out
        for line in jsonl_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if args.model_substr and args.model_substr not in str(r.get("model", "")):
                continue
            # Seed filter: only enforced if the jsonl row has a "seed" key.
            # Rows without a seed column are taken as-is.
            if "seed" in r and r["seed"] != args.seed_filter:
                continue
            # For multi-seed jsonls require the c1_c3 SmolLM2 cell
            # we want, not other co-located cells.
            if "constraint" in r and r["constraint"] not in ("c1_c3", "c1_c2_c3"):
                continue
            row_dialect = r.get("dialect", "")
            if dialect == "arith+func" and row_dialect != "arith+func":
                continue
            if dialect == "linalg+memref" and row_dialect != "linalg":
                continue
            if dialect == "stablehlo" and "stablehlo" not in row_dialect:
                continue
            out[str(r.get("prompt_id"))] = r.get("generated", "")
        return out

    # The benchmark layout: each ref names a source_benchmark. We map:
    #   mlir_spec_150  → arith candidates jsonl
    #   linalg_spec_30 → linalg candidates jsonl
    #   stablehlo_spec_30 → stablehlo candidates jsonl
    candidates_by_dialect = {
        "mlir_spec_150":     _filter_candidates(Path(args.candidates_arith), "arith+func"),
        "linalg_spec_30":    _filter_candidates(Path(args.candidates_linalg), "linalg+memref"),
        "stablehlo_spec_30": _filter_candidates(Path(args.candidates_stablehlo), "stablehlo"),
    }

    refs = json.loads(Path(args.references).read_text())["references"]
    print(f"[functional] {len(refs)} references", file=sys.stderr)

    per_prompt = []
    for ref in refs:
        rid = ref["id"]
        cand = _load_candidate_for(ref, candidates_by_dialect)
        if not cand:
            per_prompt.append({
                "id": rid, "dialect": ref["dialect"],
                "verify_valid": False, "lower_ok": False, "exec_ok": False,
                "output_match": False, "detail": {"err": "no candidate generation found for source_id"},
            })
            print(f"  {rid}: NO_CAND", file=sys.stderr)
            continue
        runner = DISPATCH[ref["dialect"]]
        try:
            res = runner(cand, ref)
        except Exception as e:
            res = {"verify_valid": False, "lower_ok": False, "exec_ok": False,
                   "output_match": False, "detail": {"runner_err": str(e)}}
        res["id"] = rid
        res["dialect"] = ref["dialect"]
        per_prompt.append(res)
        flags = "".join("✓" if res[k] else "✗"
                        for k in ("verify_valid", "lower_ok", "exec_ok", "output_match"))
        print(f"  {rid}: V/L/E/M = {flags}", file=sys.stderr)

    # Aggregate
    n = len(per_prompt)
    def rate(k):
        return sum(1 for r in per_prompt if r.get(k)) / n if n else 0.0
    summary = {
        "n": n,
        "verify_valid_rate": rate("verify_valid"),
        "lower_ok_rate":     rate("lower_ok"),
        "exec_ok_rate":      rate("exec_ok"),
        "output_match_rate": rate("output_match"),
        "by_dialect": {},
    }
    for d in ("arith+func", "linalg+memref", "stablehlo"):
        rows = [r for r in per_prompt if r["dialect"] == d]
        if not rows:
            continue
        summary["by_dialect"][d] = {
            "n": len(rows),
            "verify_valid_rate": sum(r["verify_valid"] for r in rows) / len(rows),
            "lower_ok_rate":     sum(r["lower_ok"]     for r in rows) / len(rows),
            "exec_ok_rate":      sum(r["exec_ok"]      for r in rows) / len(rows),
            "output_match_rate": sum(r["output_match"] for r in rows) / len(rows),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"per_prompt": per_prompt, "summary": summary}, indent=2))
    print(f"\n[functional] summary: {summary}", file=sys.stderr)
    print(f"[functional] → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
