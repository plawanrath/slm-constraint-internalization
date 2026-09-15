"""Extract ODSOp records from `llvm-tblgen --dump-json` output.

This is the primary ODS source. The regex-based parser in `ods_parse.py` is
retained for unit tests on tiny .td snippets but misses everything that uses
TableGen's class-hierarchy features (Arguments<(ins ...)>, multiclass, etc.).

Flow:
  1. Inside the pinned LLVM container, run:
        llvm-tblgen --dump-json \\
            -I /opt/llvm/include -I /opt/llvm/include/mlir \\
            /opt/llvm/include/mlir/Dialect/X/IR/XOps.td
     This resolves inheritance, multiclass expansion, and everything else.
  2. We parse the JSON and identify Op records via `!superclasses` containing
     "Op" or "Op<...>".
  3. Each op yields an ODSOp with operands/results/attributes classified by
     checking the referenced def's `!superclasses` (TypeConstraint vs Attr).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .ods_parse import ODSArgument, ODSOp, ODSResult

REPO_ROOT = Path(__file__).resolve().parent.parent
TBLGEN = REPO_ROOT / "scripts" / "env" / "bin" / "llvm-tblgen"


def _is_attr_def(record_name: str, records: dict) -> bool:
    """True iff the record with name `record_name` extends an Attr class."""
    rec = records.get(record_name)
    if not isinstance(rec, dict):
        return False
    supers = rec.get("!superclasses", [])
    return any("Attr" in s for s in supers)


def _is_variadic_wrapper(record_name: str, records: dict) -> tuple[str | None, bool]:
    """If `record_name` is a `Variadic<X>` or similar wrapper, return (X, True).
    Else return (record_name, False)."""
    rec = records.get(record_name)
    if not isinstance(rec, dict):
        return record_name, False
    supers = rec.get("!superclasses", [])
    if any(s.startswith("Variadic") or "Variadic" in s for s in supers):
        # A Variadic<X> record has an anonymous name; look up its "baseType" field.
        base = rec.get("baseType")
        if isinstance(base, dict) and "def" in base:
            return base["def"], True
    return record_name, False


def _extract_dag_args(dag: dict | None, records: dict) -> list[tuple[str, str, bool]]:
    """Extract (type_constraint, arg_name, variadic) triples from an arguments
    or results dag. Skips entries whose type doesn't have a resolvable class."""
    out: list[tuple[str, str, bool]] = []
    if not isinstance(dag, dict):
        return out
    for item in dag.get("args", []):
        if not isinstance(item, list) or len(item) != 2:
            continue
        spec, name = item
        if not isinstance(spec, dict):
            continue
        tc_name = spec.get("def", "")
        if not tc_name:
            continue
        resolved, variadic = _is_variadic_wrapper(tc_name, records)
        out.append((resolved, str(name), variadic))
    return out


def _is_op_record(name: str, rec: dict) -> bool:
    if not isinstance(rec, dict):
        return False
    supers = rec.get("!superclasses", [])
    return any(s == "Op" or s.startswith("Op<") for s in supers)


def parse_tblgen_json(json_text: str) -> list[ODSOp]:
    """Parse `llvm-tblgen --dump-json` output into ODSOp records."""
    data = json.loads(json_text)
    if not isinstance(data, dict):
        return []
    ops: list[ODSOp] = []
    for name, rec in data.items():
        if name.startswith("!") or name.startswith("__"):
            continue
        if not _is_op_record(name, rec):
            continue
        mnemonic = rec.get("opName", "")
        if not mnemonic:
            continue
        # Dialect comes from the dialect reference (rec["opDialect"]["def"]).
        dialect_ref = rec.get("opDialect")
        if isinstance(dialect_ref, dict) and "def" in dialect_ref:
            dialect_rec_name = dialect_ref["def"]
            dialect_rec = data.get(dialect_rec_name, {})
            dialect = dialect_rec.get("name") if isinstance(dialect_rec, dict) else ""
            if not dialect:
                dialect = dialect_rec_name.lower().rstrip("_dialect").replace("_", "")
        else:
            dialect = ""
        if not dialect:
            continue
        full_name = f"{dialect}.{mnemonic}"

        summary = rec.get("summary", "") or ""
        description = rec.get("description", "") or ""

        args_triples = _extract_dag_args(rec.get("arguments"), data)
        results_triples = _extract_dag_args(rec.get("results"), data)

        arguments = [
            ODSArgument(
                type_constraint=tc, name=nm,
                is_attribute=_is_attr_def(tc, data),
                variadic=variadic,
            )
            for tc, nm, variadic in args_triples
        ]
        results = [
            ODSResult(type_constraint=tc, name=nm, variadic=variadic)
            for tc, nm, variadic in results_triples
        ]

        # Traits: `traits` is a list of class refs.
        traits: list[str] = []
        for tref in rec.get("traits", []) or []:
            if isinstance(tref, dict) and "def" in tref:
                traits.append(tref["def"])

        ops.append(ODSOp(
            dialect=dialect,
            mnemonic=mnemonic,
            full_name=full_name,
            summary=summary.strip(),
            description=description.strip(),
            arguments=arguments,
            results=results,
            traits=traits,
            source_file="",
        ))
    return ops


def tblgen_dump_for_dialect(
    ops_td_path: str,
    include_paths: tuple[str, ...] = ("/opt/llvm/include", "/opt/llvm/include/mlir"),
) -> str:
    """Run `llvm-tblgen --dump-json` inside the container and return JSON text."""
    cmd = [str(TBLGEN), "--dump-json"]
    for p in include_paths:
        cmd.extend(["-I", p])
    cmd.append(ops_td_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout
