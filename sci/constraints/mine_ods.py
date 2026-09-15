"""L0 ODS miner — uses `llvm-tblgen --dump-json` for full inheritance resolution.

Flow:
  1. For each target dialect, locate the primary *Ops.td file in the pinned
     LLVM container (/opt/llvm/include/mlir/Dialect/X/IR/XOps.td) and also walk
     any additional .td files passed via --extra-td.
  2. Run `llvm-tblgen --dump-json` to get resolved ODS as machine-readable JSON.
  3. Parse into ODSOp records.
  4. Write (NL, op_signature) jsonl + per-dialect lattice JSONs.

Usage:
    python -m data.pipelines.l0_ods.mine \\
        --out data/processed/l0_ods.jsonl \\
        --lattice-out grammar/lattices \\
        --target-dialects arith,func,linalg,memref

The container must be running (docker compose up -d).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sci.constraints.ods_parse import ODSOp
from sci.constraints.ods_tblgen import parse_tblgen_json, tblgen_dump_for_dialect
from sci.constraints.ods_lattice import write_lattice


# Map target dialect name → primary *Ops.td path inside the container.
_DIALECT_TD_PATHS: dict[str, tuple[str, ...]] = {
    "arith":   ("/opt/llvm/include/mlir/Dialect/Arith/IR/ArithOps.td",),
    "func":    ("/opt/llvm/include/mlir/Dialect/Func/IR/FuncOps.td",),
    "linalg":  (
        "/opt/llvm/include/mlir/Dialect/Linalg/IR/LinalgOps.td",
        "/opt/llvm/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td",
    ),
    "memref":  ("/opt/llvm/include/mlir/Dialect/MemRef/IR/MemRefOps.td",),
    # Extra dialects we'd mine for corpus volume but not for lattices.
    "scf":     ("/opt/llvm/include/mlir/Dialect/SCF/IR/SCFOps.td",),
    "affine":  ("/opt/llvm/include/mlir/Dialect/Affine/IR/AffineOps.td",),
    "tensor":  ("/opt/llvm/include/mlir/Dialect/Tensor/IR/TensorOps.td",),
    "vector":  ("/opt/llvm/include/mlir/Dialect/Vector/IR/VectorOps.td",),
    "tosa":    ("/opt/llvm/include/mlir/Dialect/Tosa/IR/TosaOps.td",),
    "gpu":     ("/opt/llvm/include/mlir/Dialect/GPU/IR/GPUOps.td",),
}


def _mine_dialect(dialect: str) -> list[ODSOp]:
    paths = _DIALECT_TD_PATHS.get(dialect, ())
    ops: list[ODSOp] = []
    for p in paths:
        try:
            js = tblgen_dump_for_dialect(p)
        except Exception as e:  # noqa: BLE001
            print(f"[l0] warn: tblgen failed for {dialect} ({p}): {e}",
                  file=sys.stderr)
            continue
        dialect_ops = parse_tblgen_json(js)
        for op in dialect_ops:
            op.source_file = p
        ops.extend(dialect_ops)
        print(f"[l0] {dialect}: +{len(dialect_ops)} ops from {p.split('/')[-1]}",
              file=sys.stderr)
    return ops


def _format_signature(op: ODSOp) -> str:
    ins = ", ".join(f"{a.type_constraint}:${a.name}" for a in op.arguments)
    outs = ", ".join(f"{r.type_constraint}:${r.name}" for r in op.results)
    return f"{op.full_name} ({ins}) -> ({outs})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/processed/ods_docs.jsonl"))
    ap.add_argument("--lattice-out", type=Path, default=Path(__file__).resolve().parent / "lattices")
    ap.add_argument("--target-dialects", default="arith,func,linalg,memref")
    ap.add_argument(
        "--extra-dialects",
        default="scf,affine,tensor,vector",
        help="Extra dialects to mine for corpus volume (no lattice written).",
    )
    args = ap.parse_args()

    target = [d.strip() for d in args.target_dialects.split(",") if d.strip()]
    extra = [d.strip() for d in args.extra_dialects.split(",") if d.strip()]

    ops_target: list[ODSOp] = []
    ops_extra: list[ODSOp] = []
    for d in target:
        ops_target.extend(_mine_dialect(d))
    for d in extra:
        ops_extra.extend(_mine_dialect(d))

    all_ops = ops_target + ops_extra

    # Write JSONL corpus (all ops; target + extra).
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for op in all_ops:
            if not op.summary and not op.description:
                continue
            record = {
                "source": "l0_ods",
                "dialect": op.dialect,
                "op_name": op.full_name,
                "summary": op.summary,
                "description": op.description,
                "op_signature": _format_signature(op),
                "traits": op.traits,
                "source_file": op.source_file,
                "in_target": op.dialect in target,
            }
            f.write(json.dumps(record) + "\n")

    print(f"[l0] wrote {len(all_ops)} ops ({len(ops_target)} target, {len(ops_extra)} extra) → {args.out}",
          file=sys.stderr)

    # Write lattices only for target dialects.
    written = write_lattice(ops_target, args.lattice_out, target_dialects=tuple(sorted(target)))
    for dialect, path in written.items():
        n_ops = len(json.loads(path.read_text()))
        print(f"[l0] lattice[{dialect}]: {n_ops} ops → {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
