"""Build a per-op type lattice from parsed ODS records.

Output schema (per dialect, written to sci/constraints/lattices/{dialect}.json):

    {
      "arith.addi": {
        "operand_types": [["AnyInteger"], ["AnyInteger"]],
        "result_types":  [["AnyInteger"]],
        "operand_arity": [2, 2],     # [min, max]
        "result_arity":  [1, 1],
        "attributes":    [],
        "same_type_classes": [[0, 1, 0]],   # groups: operands[0], operands[1], result[0] must share type
        "source_file": ".../ArithOps.td"
      },
      ...
    }

The "allowed-type" vocabulary used by C2 is intentionally small and closed:

    PRIMITIVE: i1, i8, i16, i32, i64, i128, f16, bf16, f32, f64, index
    COMPOUND:  memref<...>, tensor<...>, vector<...>

At mask time C2 expands type-constraint class names into the matching
primitive + compound set. `_CLASS_EXPANSION` below is the canonical map.
Unknown classes fall back to "ANY" (no restriction), which C2 treats as an
abstain-this-op signal.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ods_parse import ODSOp


# Closed vocabulary of MLIR types that C2 can enforce at mask time.
PRIMITIVE_TYPES = [
    "i1", "i8", "i16", "i32", "i64", "i128",
    "f16", "bf16", "f32", "f64", "index",
]
COMPOUND_HEADS = ["memref", "tensor", "vector"]
ANY = "ANY"  # sentinel — C2 disables the type-prefix automaton for this slot


_CLASS_EXPANSION: dict[str, list[str]] = {
    # Integer types
    "AnyInteger": ["i1", "i8", "i16", "i32", "i64", "i128"],
    "AnySignlessInteger": ["i1", "i8", "i16", "i32", "i64", "i128"],
    "SignlessInteger": ["i1", "i8", "i16", "i32", "i64", "i128"],
    "AnyNonZeroBitwidthSignlessIntegerOrIndex": [
        "i1", "i8", "i16", "i32", "i64", "i128", "index"
    ],
    "I1": ["i1"],
    "I8": ["i8"],
    "I16": ["i16"],
    "I32": ["i32"],
    "I64": ["i64"],
    "I128": ["i128"],
    "Index": ["index"],
    "IndexOrSignlessInteger": ["i1", "i8", "i16", "i32", "i64", "i128", "index"],
    # Float types
    "AnyFloat": ["f16", "bf16", "f32", "f64"],
    "F16": ["f16"],
    "BF16": ["bf16"],
    "F32": ["f32"],
    "F64": ["f64"],
    # Catch-alls we keep as ANY (C2 skips)
    "AnyType": [ANY],
    "AnyShaped": [ANY],
    "AnyRankedTensor": [ANY],
    "AnyRankedOrUnrankedMemRef": ["memref"],
    # Compound containers — represented as a head token; C2 recurses on the
    # element type but treats the shape as free-form.
    "AnyMemRef": ["memref"],
    "MemRefOf": ["memref"],
    "AnyTensor": ["tensor"],
    "TensorOf": ["tensor"],
    "AnyVector": ["vector"],
    "VectorOf": ["vector"],
    "AnyVectorOfAnyRank": ["vector"],
    # "Like" constraints: elementwise-mappable (scalar OR tensor-of-X OR vector-of-X)
    "SignlessIntegerLike": ["i1", "i8", "i16", "i32", "i64", "i128", "index"],
    "IntegerLike": ["i1", "i8", "i16", "i32", "i64", "i128", "index"],
    "FloatLike": ["f16", "bf16", "f32", "f64"],
    "SignlessFixedWidthIntegerLike": ["i1", "i8", "i16", "i32", "i64", "i128"],
    "SignlessIntegerOrIndexLike": [
        "i1", "i8", "i16", "i32", "i64", "i128", "index"
    ],
    # Arith dialect introduces its own constraint class; we treat as ANY.
    "Arith_SignlessIntegerOrIndexLike": [
        "i1", "i8", "i16", "i32", "i64", "i128", "index"
    ],
    # Arbitrary range/step/bound types — treat as ANY.
    "AnyIntegerAttr": [ANY],
    "TypedAttrInterface": [ANY],
}


def expand_class(tc: str) -> list[str]:
    """Map an ODS type-constraint class to a closed-vocab list.

    Unknown classes → [ANY] (C2 will skip the mask for that slot)."""
    tc = tc.strip()
    if tc in _CLASS_EXPANSION:
        return list(_CLASS_EXPANSION[tc])
    # Heuristic: Variadic< X > is unwrapped upstream; handle nested wrappers.
    m = re.match(r"(\w+)\s*<(.+)>\s*$", tc)
    if m:
        outer, _inner = m.groups()
        if outer in _CLASS_EXPANSION:
            return list(_CLASS_EXPANSION[outer])
    return [ANY]


@dataclass
class OpLatticeEntry:
    operand_types: list[list[str]]
    result_types: list[list[str]]
    operand_arity: tuple[int, int]
    result_arity: tuple[int, int]
    attributes: list[str] = field(default_factory=list)
    same_type_classes: list[list[int]] = field(default_factory=list)
    source_file: str = ""

    def to_dict(self) -> dict:
        return {
            "operand_types": self.operand_types,
            "result_types": self.result_types,
            "operand_arity": list(self.operand_arity),
            "result_arity": list(self.result_arity),
            "attributes": self.attributes,
            "same_type_classes": self.same_type_classes,
            "source_file": self.source_file,
        }


def build_entry(op: ODSOp) -> OpLatticeEntry:
    operand_args = [a for a in op.arguments if not a.is_attribute]
    attr_args = [a for a in op.arguments if a.is_attribute]
    operand_types = [expand_class(a.type_constraint) for a in operand_args]
    result_types = [expand_class(r.type_constraint) for r in op.results]
    operand_min = sum(0 if a.variadic else 1 for a in operand_args)
    operand_max = (
        10**6 if any(a.variadic for a in operand_args) else len(operand_args)
    )
    result_min = sum(0 if r.variadic else 1 for r in op.results)
    result_max = (
        10**6 if any(r.variadic for r in op.results) else len(op.results)
    )
    # Derive same-type constraints from traits.
    same_type_classes: list[list[int]] = []
    if "SameOperandsAndResultType" in op.traits:
        group = list(range(len(operand_args))) + [
            len(operand_args) + i for i in range(len(op.results))
        ]
        # Encode: for each slot, what equivalence class it belongs to.
        # We flatten to a single class covering all operand + result slots.
        same_type_classes.append(group)
    elif "SameOperandsType" in op.traits:
        same_type_classes.append(list(range(len(operand_args))))
    return OpLatticeEntry(
        operand_types=operand_types,
        result_types=result_types,
        operand_arity=(operand_min, operand_max),
        result_arity=(result_min, result_max),
        attributes=[a.type_constraint for a in attr_args],
        same_type_classes=same_type_classes,
        source_file=op.source_file,
    )


def build_lattice(ops: list[ODSOp]) -> dict[str, dict]:
    """Group ODSOp records by dialect, emit a per-op lattice dict."""
    out: dict[str, dict] = {}
    for op in ops:
        out[op.full_name] = build_entry(op).to_dict()
    return out


def write_lattice(
    ops: list[ODSOp],
    out_dir: Path,
    target_dialects: tuple[str, ...] = ("arith", "func", "linalg", "memref"),
) -> dict[str, Path]:
    """Write one lattice JSON per target dialect. Returns {dialect: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_dialect: dict[str, list[ODSOp]] = {d: [] for d in target_dialects}
    for op in ops:
        if op.dialect in by_dialect:
            by_dialect[op.dialect].append(op)
    written = {}
    for dialect, ops_for_d in by_dialect.items():
        path = out_dir / f"{dialect}.json"
        lattice = build_lattice(ops_for_d)
        path.write_text(json.dumps(lattice, indent=2, sort_keys=True))
        written[dialect] = path
    return written


def load_lattice(path: Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text())
