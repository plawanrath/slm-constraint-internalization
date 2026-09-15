"""Minimal TableGen (.td) parser for MLIR ODS records.

We parse enough of ODS to extract, per operation record:
  - the op mnemonic (dialect + name)
  - the `summary` string (NL)
  - the `description` string (NL)
  - the `arguments` dag (operand types + attribute types)
  - the `results` dag (result types)

We deliberately do not implement the full TableGen language. We treat .td as
a mostly-regular grammar for our purposes and fall back to ignoring unknown
records. This is enough for L0 and for the C2 type lattice.

Tested surface:
  def ArithAddIOp : Arith_IntBinaryOp<"addi"> { let summary = "..."; ... }

Missing (tracked as TODO):
  - multiclass expansion (we only read direct `def` records)
  - include-chain resolution (we parse a single .td file; the miner walks)
  - TypeConstraint composition semantics (we record the class name verbatim;
    the type-lattice builder interprets common ones explicitly)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"', re.DOTALL)
_BRACED_RE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)  # triple-braced strings


@dataclass
class ODSArgument:
    """One operand or attribute of an op."""
    type_constraint: str  # e.g. "I32", "AnyInteger", "AnyFloat", "F32", "Index", "AnyMemRef"
    name: str             # e.g. "lhs"
    is_attribute: bool = False  # True if declared as an attribute, not an operand
    variadic: bool = False


@dataclass
class ODSResult:
    type_constraint: str
    name: str
    variadic: bool = False


@dataclass
class ODSOp:
    dialect: str          # e.g. "arith"
    mnemonic: str         # e.g. "addi"  (no dialect prefix)
    full_name: str        # e.g. "arith.addi"
    summary: str
    description: str
    arguments: list[ODSArgument] = field(default_factory=list)
    results: list[ODSResult] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)
    source_file: str = ""


def _strip_comments(src: str) -> str:
    # Strip // line comments and /* ... */ block comments.
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return src


def _extract_string(body: str, field_name: str) -> str:
    """Pull `let <field_name> = "<value>"` or `= [{ ... }]` from a record body."""
    # Triple-brace form: let summary = [{ ... }];
    m = re.search(
        rf"let\s+{field_name}\s*=\s*\[\{{(.+?)\}}\]\s*;",
        body,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    # Plain string form: let summary = "...";
    m = re.search(
        rf'let\s+{field_name}\s*=\s*"((?:[^"\\]|\\.)*)"\s*;',
        body,
        re.DOTALL,
    )
    if m:
        return m.group(1)
    return ""


def _parse_dag(body: str, field_name: str) -> list[tuple[str, str]]:
    """Parse `let arguments = (ins I32:$a, I32:$b);` into [(type, name), ...]."""
    m = re.search(
        rf"let\s+{field_name}\s*=\s*\(\s*(?:ins|outs)\s*(.*?)\)\s*;",
        body,
        re.DOTALL,
    )
    if not m:
        return []
    inner = m.group(1).strip()
    if not inner:
        return []
    # Split on top-level commas.
    items = _split_top_level(inner, ",")
    out: list[tuple[str, str]] = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        # Form: `TypeConstraint:$name` or `Variadic<TypeConstraint>:$name`.
        if ":$" in item:
            tc, name = item.split(":$", 1)
            out.append((tc.strip(), name.strip()))
        else:
            out.append((item, ""))
    return out


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split `s` on `sep` respecting <>, (), [], {} nesting."""
    parts = []
    depth_a = depth_r = depth_s = depth_c = 0
    buf = []
    for ch in s:
        if ch == "<":
            depth_a += 1
        elif ch == ">":
            depth_a -= 1
        elif ch == "(":
            depth_r += 1
        elif ch == ")":
            depth_r -= 1
        elif ch == "[":
            depth_s += 1
        elif ch == "]":
            depth_s -= 1
        elif ch == "{":
            depth_c += 1
        elif ch == "}":
            depth_c -= 1
        if ch == sep and depth_a == depth_r == depth_s == depth_c == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


_OP_DEF_RE = re.compile(
    # def RecordName : ParentClass<"mnemonic"[, [Trait1, Trait2]]> { ... }
    # ParentClass is any identifier (Arith_Op, Arith_IntBinaryOp, LinalgStructured_Op, etc.);
    # we derive dialect from its leading underscore-separated token.
    r"def\s+(\w+)\s*:\s*(\w+)\s*<\s*\"([^\"]+)\"\s*(?:,\s*\[([^\]]*)\])?\s*>\s*\{(.*?)\n\}",
    re.DOTALL,
)


def parse_td(src: str, source_file: str = "") -> list[ODSOp]:
    """Parse a .td source string, return all ODSOp records we recognize."""
    src = _strip_comments(src)
    ops: list[ODSOp] = []
    for m in _OP_DEF_RE.finditer(src):
        _record_name, dialect_ident, mnemonic, traits_s, body = m.groups()
        dialect = dialect_ident.lower().rstrip("_")
        # Normalize common parent-class names:
        #   Arith_IntBinaryOp → arith
        #   Func_Op → func
        #   Linalg_Op → linalg
        dialect = dialect.split("_")[0]
        full_name = f"{dialect}.{mnemonic}"
        summary = _extract_string(body, "summary")
        description = _extract_string(body, "description")
        arguments_raw = _parse_dag(body, "arguments")
        results_raw = _parse_dag(body, "results")
        arguments = [
            ODSArgument(
                type_constraint=_unwrap_variadic(tc)[0],
                name=name,
                is_attribute=_looks_like_attribute(tc),
                variadic=_unwrap_variadic(tc)[1],
            )
            for tc, name in arguments_raw
        ]
        results = [
            ODSResult(
                type_constraint=_unwrap_variadic(tc)[0],
                name=name,
                variadic=_unwrap_variadic(tc)[1],
            )
            for tc, name in results_raw
        ]
        traits = [t.strip() for t in (traits_s or "").split(",") if t.strip()]
        ops.append(
            ODSOp(
                dialect=dialect,
                mnemonic=mnemonic,
                full_name=full_name,
                summary=summary.strip(),
                description=description.strip(),
                arguments=arguments,
                results=results,
                traits=traits,
                source_file=source_file,
            )
        )
    return ops


def _unwrap_variadic(tc: str) -> tuple[str, bool]:
    m = re.match(r"Variadic\s*<\s*(.+?)\s*>\s*$", tc)
    if m:
        return m.group(1).strip(), True
    return tc.strip(), False


_ATTR_HEADS = ("I32Attr", "I64Attr", "F32Attr", "F64Attr", "StrAttr",
               "SymbolNameAttr", "FlatSymbolRefAttr", "TypeAttr", "UnitAttr",
               "BoolAttr", "AnyAttr", "DictionaryAttr", "ArrayAttr")


def _looks_like_attribute(tc: str) -> bool:
    return any(tc.startswith(head) or tc.endswith("Attr") for head in _ATTR_HEADS)


def parse_td_file(path: Path) -> list[ODSOp]:
    src = Path(path).read_text()
    return parse_td(src, source_file=str(path))


def walk_td(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.td")
