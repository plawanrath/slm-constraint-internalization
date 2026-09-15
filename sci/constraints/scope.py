"""C3: SSA-scope + type-consistency validator for generated MLIR.

The context-free layers (C1 syntax, C2 type domains) cannot see across ops: a
function parameter declared as `i32` may be used in an op whose trailing type
annotation is `f32`, or an SSA value defined in one op may be referenced later
under a different type or before its definition. This validator closes that
gap post hoc; it is the context-sensitive layer of the constraint stack and is
applied as a rejection-sampling filter (and, offline, as a per-token oracle
for training signals).

Scope:
- Per-function symbol tables. Seeded with `func.func @name(%a : T, %b : T)`
  parameters. A bare `return` closes the function.
- Each `%name = <op> ... : TYPE` adds `%name → type_of_def(op, trailing_type)`
  to the scope.
- For every use on an op line, the use's type-obligation under that op pattern
  is compared against the scope entry.

Supported op patterns (cover the gen grammar's 22 alternatives):
- arith.constant (int/float/index)            — no uses
- arith binop int/float (SameOperandsAndResult)— all uses share trailing type
- arith.cmpi / arith.cmpf                     — operands share trailing type,
                                                 result is i1
- arith.select int/float                      — cond %s1 : i1, %s2,%s3 : T
- arith casts (5 families)                    — src-type source, tgt-type result
- arith.bitcast                               — src-type source, tgt-type result
- memref.alloc() : memref<...>                — no uses
- memref.load %m[%i...] : memref<SHAPExE>     — %m: memref<...>,
                                                 %i... : index, def : E
- memref.store %v, %m[%i...] : memref<SHAPExE>— %v : E, %m : memref<...>,
                                                 %i... : index
- memref.dealloc %m : memref<...>             — %m : memref<...>
- memref.dim %m, %i : memref<...>             — %m : memref<...>, %i : index,
                                                 def : index
- return %x : T                               — %x : T
- return (void)                               — no uses

Any unrecognized op line is abstained on (neither adds a def nor validates
uses). This is safe: the worst case is letting through a generation that
would have been caught; it never causes false rejects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


# ---------------- lexical helpers ----------------

# SSA ref: %name or %name:N or %name#N. We normalize to the bare name.
_SSA_RE = re.compile(r"%([A-Za-z0-9_][A-Za-z0-9_]*)(?::\d+)?(?:#\d+)?")

# Matches a parameter like  %a : i32  or  %a: memref<?xf32>  (whitespace-tolerant).
_PARAM_RE = re.compile(
    r"%(?P<name>[A-Za-z0-9_][A-Za-z0-9_]*)\s*:\s*(?P<type>[^,)]+?)\s*(?=,|\))"
)

# func.func header. Captures the parenthesized param group; we rescan it with
# _PARAM_RE rather than risk nested-bracket regex woes.
_FUNC_HEADER_RE = re.compile(
    r"func\.func\s+(?:private\s+)?@[A-Za-z_][A-Za-z0-9_]*\s*\((?P<params>[^)]*)\)"
)

_INT_TYPES = {"i1", "i4", "i8", "i16", "i32", "i64", "i128", "index"}
_FLOAT_TYPES = {"f16", "bf16", "f32", "f64", "f80", "f128"}


def _ssa_refs(s: str) -> list[str]:
    """All %names in s, in order, deduplicated on first occurrence."""
    seen: list[str] = []
    out: list[str] = []
    for m in _SSA_RE.finditer(s):
        name = m.group(1)
        if name not in seen:
            seen.append(name)
            out.append(name)
    return out


def _normalize(t: str) -> str:
    """Strip whitespace and trailing punctuation from a type string."""
    return t.strip().rstrip(",;)")


def _stablehlo_type_sig(rhs: str) -> tuple[str | None, str | None]:
    """For stablehlo ops with `(tensor<IN>) -> tensor<OUT>` signatures, return
    (in_type, out_type) or (None, None) if not matched."""
    m = re.search(r"\(\s*(tensor<[^)]*?>)\s*\)\s*->\s*(tensor<[^>]*?>)", rhs)
    if m is None:
        return None, None
    return _normalize(m.group(1)), _normalize(m.group(2))


def _stablehlo_binary_sig(rhs: str) -> tuple[list[str] | None, str | None]:
    """For stablehlo ops with `(tensor<A>, tensor<B>) -> tensor<C>` signatures,
    return ([in_a, in_b], out_type) or (None, None) if not matched."""
    m = re.search(
        r"\(\s*(tensor<[^,]*?>)\s*,\s*(tensor<[^)]*?>)\s*\)\s*->\s*(tensor<[^>]*?>)",
        rhs,
    )
    if m is None:
        return None, None
    return [_normalize(m.group(1)), _normalize(m.group(2))], _normalize(m.group(3))


def _memref_element_type(memref_type: str) -> str | None:
    """Pull the element type out of `memref<SHAPExELEM>` or `memref<ELEM>`.

    Handles nested `<...>` inside tuples/vectors by tracking bracket depth.
    Returns None if the type isn't a memref.
    """
    s = memref_type.strip()
    if not s.startswith("memref<") or not s.endswith(">"):
        return None
    inner = s[len("memref<"):-1]
    # Walk from the right, splitting on the last top-level "x" that separates
    # shape dims from the element type. Shape dims are digits or "?".
    depth = 0
    last_x = -1
    for i, ch in enumerate(inner):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "x" and depth == 0:
            # "x" only separates shape dims if what precedes is a dim token
            # (digits or ? or "[N]" for scalable).
            prev = inner[:i].rstrip()
            if prev and (prev[-1].isdigit() or prev.endswith("?") or prev.endswith("]")):
                last_x = i
    elem = inner[last_x + 1:] if last_x >= 0 else inner
    return elem.strip()


# ---------------- data classes ----------------

@dataclass
class ScopeViolation:
    kind: str            # "undef_use" | "type_mismatch" | "parse"
    detail: str
    line: str

    def __str__(self) -> str:  # nice for logging
        return f"[{self.kind}] {self.detail} || {self.line!r}"


@dataclass
class ScopeReport:
    passed: bool
    violations: list[ScopeViolation] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


# ---------------- op pattern dispatch ----------------

# Each handler receives (def_name_or_None, rest_of_line, scope, violations).
# It may extend `violations` and register a def in `scope`.

_ARITH_BIN_INT = {
    "arith.addi", "arith.subi", "arith.muli",
    "arith.divsi", "arith.divui",
    "arith.andi", "arith.ori", "arith.xori",
    "arith.shli", "arith.shrsi", "arith.shrui",
    "arith.maxsi", "arith.maxui", "arith.minsi", "arith.minui",
    "arith.remsi", "arith.remui",
}
_ARITH_BIN_FLOAT = {
    "arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.remf",
    "arith.maxf", "arith.minf",
}
_INT_WIDEN = {"arith.extsi", "arith.extui", "arith.trunci"}
_I2F = {"arith.sitofp", "arith.uitofp"}
_F2I = {"arith.fptosi", "arith.fptoui"}
_FLOAT_WIDEN = {"arith.extf", "arith.truncf"}
_INDEX_CAST = {"arith.index_cast", "arith.index_castui"}

# Linalg named ops (memref semantics, 12 supported named ops).
# All use an ins(...) / outs(...) clause structure.
_LINALG_TWO_IN_ONE_OUT = {
    # 2 memref ins + 1 memref outs, all element-type-compatible.
    "linalg.matmul", "linalg.matvec",
    "linalg.add", "linalg.sub", "linalg.mul", "linalg.div",
}
_LINALG_ONE_IN_ONE_OUT = {
    # 1 memref ins + 1 memref outs, element-type-compatible.
    "linalg.copy", "linalg.transpose", "linalg.broadcast",
    "linalg.exp", "linalg.abs",
}
_LINALG_SCALAR_IN_ONE_OUT = {
    # 1 scalar ins (matches memref element type) + 1 memref outs.
    "linalg.fill",
}
_LINALG_OPS = _LINALG_TWO_IN_ONE_OUT | _LINALG_ONE_IN_ONE_OUT | _LINALG_SCALAR_IN_ONE_OUT

# StableHLO named ops (tensor semantics).
# Unlike linalg, StableHLO uses direct operand syntax rather than ins/outs.
_STABLEHLO_EW_BIN = {
    "stablehlo.add", "stablehlo.subtract", "stablehlo.multiply", "stablehlo.divide",
}
_STABLEHLO_EW_UN = {
    "stablehlo.abs", "stablehlo.exponential",
}
# Ops that carry a (tensor<IN>) -> tensor<OUT> type signature:
_STABLEHLO_TYPED_SIG = {
    "stablehlo.transpose", "stablehlo.broadcast_in_dim",
    "stablehlo.reshape", "stablehlo.convert",
}
# Ops with a (tensor<A>, tensor<B>) -> tensor<C> signature:
_STABLEHLO_BINARY_TYPED = {
    "stablehlo.dot_general",
}
_STABLEHLO_OPS = (
    _STABLEHLO_EW_BIN | _STABLEHLO_EW_UN
    | _STABLEHLO_TYPED_SIG | _STABLEHLO_BINARY_TYPED
)


def _extract_clause(line: str, keyword: str) -> str | None:
    """Return the contents of `keyword(...)` clause (linalg-style), or None.

    Finds `keyword` followed by `(`, then returns the substring between the
    matched parens (respecting nested `<>` for memref type). For example,
    `_extract_clause("linalg.add ins(%a, %b : memref<?xi32>, memref<?xi32>) outs(%c : memref<?xi32>)", "ins")`
    returns `"%a, %b : memref<?xi32>, memref<?xi32>"`.
    """
    idx = line.find(keyword + "(")
    if idx < 0:
        # tolerate whitespace between keyword and (
        import re as _re
        m = _re.search(r"\b" + _re.escape(keyword) + r"\s*\(", line)
        if m is None:
            return None
        idx = m.end() - 1  # idx of '('
    else:
        idx += len(keyword)  # idx of '('
    depth_par = 0
    depth_ang = 0
    start = idx + 1
    for i in range(idx, len(line)):
        ch = line[i]
        if ch == "<":
            depth_ang += 1
        elif ch == ">":
            depth_ang = max(0, depth_ang - 1)
        elif ch == "(":
            depth_par += 1
        elif ch == ")":
            depth_par -= 1
            if depth_par == 0 and depth_ang == 0:
                return line[start:i]
    return None


def _parse_linalg_clause(clause: str) -> list[tuple[str, str]]:
    """Parse `%a, %b : T1, T2` into [(name, T1), (name, T2)]; `%a : T` → [(a, T)].

    Counts are matched (one type per SSA name). If counts disagree we return
    as many pairs as we can — rejection sampling will fail the generation
    anyway, but we avoid hard crashes.
    """
    if ":" not in clause:
        return []
    lhs, rhs = clause.split(":", 1)
    names = [m.group(1) for m in _SSA_RE.finditer(lhs)]
    # Split rhs on top-level commas (respecting < > nesting).
    types: list[str] = []
    depth_ang = 0
    cur: list[str] = []
    for ch in rhs:
        if ch == "<":
            depth_ang += 1
        elif ch == ">":
            depth_ang = max(0, depth_ang - 1)
        if ch == "," and depth_ang == 0:
            types.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        types.append(tail)
    return list(zip(names, types))


def _memref_or_scalar_element(t: str) -> str:
    """For a memref<...>E, return E; for scalar type, return t unchanged."""
    if t.strip().startswith("memref<"):
        e = _memref_element_type(t.strip())
        return e or t.strip()
    return t.strip()


def _last_colon_type(line: str) -> str | None:
    """Return the type substring after the final ' : ' on the line (strip trailing
    loc/attr clutter). For `arith.extsi %x : i32 to i64`, returns 'i32 to i64'
    — callers that need the target type should split on ' to '."""
    # Find the LAST ` : ` outside any < > or ( ) pair.
    depth_ang = depth_par = 0
    last = -1
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "<":
            depth_ang += 1
        elif ch == ">":
            depth_ang = max(0, depth_ang - 1)
        elif ch == "(":
            depth_par += 1
        elif ch == ")":
            depth_par = max(0, depth_par - 1)
        elif (
            ch == ":"
            and depth_ang == 0
            and depth_par == 0
            and (i == 0 or line[i - 1] in " \t")
            and (i + 1 < len(line) and line[i + 1] in " \t")
        ):
            last = i
        i += 1
    if last < 0:
        return None
    return line[last + 1:].strip()


# ---------------- core validator ----------------

def _check_use(
    name: str,
    expected: str | None,
    scope: dict[str, str],
    line: str,
    violations: list[ScopeViolation],
) -> None:
    """Check a single SSA use against the scope.

    - undef_use if `name` is not in scope.
    - type_mismatch if `expected` is not None and differs from scope[name].
    """
    if name not in scope:
        violations.append(ScopeViolation("undef_use", f"%{name}", line))
        return
    if expected is None:
        return
    have = _normalize(scope[name])
    want = _normalize(expected)
    if have != want:
        violations.append(
            ScopeViolation("type_mismatch", f"%{name}: have={have} want={want}", line)
        )


def _handle_op_line(
    line: str,
    scope: dict[str, str],
    violations: list[ScopeViolation],
) -> None:
    """Dispatch on op name and update scope/violations.

    Lines we don't recognize are abstained on (no def added, no uses checked).
    """
    stripped = line.strip()
    if not stripped:
        return

    # LHS = RHS split.
    def_name: str | None = None
    rhs = stripped
    if "=" in stripped:
        lhs, rhs = stripped.split("=", 1)
        m = _SSA_RE.match(lhs.strip())
        if m is not None:
            def_name = m.group(1)
        rhs = rhs.strip()

    # Op mnemonic = first whitespace-separated token that looks like DIALECT.OP
    # or "return".
    head = rhs.split(None, 1)[0] if rhs else ""
    if not head:
        return

    # return
    if head == "return":
        # two forms: `return` or `return %x : T`
        tail = rhs[len("return"):].strip()
        if not tail:
            return
        # pull the first SSA ref and the trailing type
        refs = _ssa_refs(tail)
        ty = _last_colon_type(tail)
        if refs:
            _check_use(refs[0], ty, scope, line, violations)
        return

    # arith.constant (no uses)
    if head == "arith.constant":
        ty = _last_colon_type(rhs)
        if def_name and ty:
            scope[def_name] = ty
        return

    # arith binary ops (SameOperandsAndResultType)
    if head in _ARITH_BIN_INT or head in _ARITH_BIN_FLOAT:
        ty = _last_colon_type(rhs)
        refs = _ssa_refs(rhs)
        for r in refs:
            _check_use(r, ty, scope, line, violations)
        if def_name and ty:
            scope[def_name] = ty
        return

    # arith.cmpi / arith.cmpf — operands share trailing type, result is i1
    if head in ("arith.cmpi", "arith.cmpf"):
        ty = _last_colon_type(rhs)
        refs = _ssa_refs(rhs)
        for r in refs:
            _check_use(r, ty, scope, line, violations)
        if def_name:
            scope[def_name] = "i1"
        return

    # arith.select — first operand is i1, next two share trailing type
    if head == "arith.select":
        ty = _last_colon_type(rhs)
        refs = _ssa_refs(rhs)
        if refs:
            _check_use(refs[0], "i1", scope, line, violations)
        for r in refs[1:]:
            _check_use(r, ty, scope, line, violations)
        if def_name and ty:
            scope[def_name] = ty
        return

    # casts: source type from `: SRC to TGT`
    if (
        head in _INT_WIDEN or head in _I2F or head in _F2I
        or head in _FLOAT_WIDEN or head in _INDEX_CAST or head == "arith.bitcast"
    ):
        ty_tail = _last_colon_type(rhs)
        src = tgt = None
        if ty_tail and " to " in ty_tail:
            src, tgt = (t.strip() for t in ty_tail.split(" to ", 1))
        refs = _ssa_refs(rhs)
        for r in refs:
            _check_use(r, src, scope, line, violations)
        if def_name and tgt:
            scope[def_name] = tgt
        return

    # memref.alloc() — no uses, def is memref<...>
    if head == "memref.alloc":
        ty = _last_colon_type(rhs)
        if def_name and ty:
            scope[def_name] = ty
        return

    # memref.load %m[%i...] : memref<SHAPExELEM>
    if head == "memref.load":
        ty = _last_colon_type(rhs)
        refs = _ssa_refs(rhs)
        if refs:
            _check_use(refs[0], ty, scope, line, violations)
            for r in refs[1:]:
                _check_use(r, "index", scope, line, violations)
        if def_name and ty:
            elem = _memref_element_type(ty)
            if elem is not None:
                scope[def_name] = elem
        return

    # memref.store %v, %m[%i...] : memref<SHAPExELEM>
    if head == "memref.store":
        ty = _last_colon_type(rhs)
        refs = _ssa_refs(rhs)
        elem = _memref_element_type(ty) if ty else None
        if len(refs) >= 2:
            _check_use(refs[0], elem, scope, line, violations)
            _check_use(refs[1], ty, scope, line, violations)
            for r in refs[2:]:
                _check_use(r, "index", scope, line, violations)
        return

    # memref.dealloc %m : memref<...>
    if head == "memref.dealloc":
        ty = _last_colon_type(rhs)
        refs = _ssa_refs(rhs)
        if refs:
            _check_use(refs[0], ty, scope, line, violations)
        return

    # memref.dim %m, %i : memref<...>
    if head == "memref.dim":
        ty = _last_colon_type(rhs)
        refs = _ssa_refs(rhs)
        if refs:
            _check_use(refs[0], ty, scope, line, violations)
            for r in refs[1:]:
                _check_use(r, "index", scope, line, violations)
        if def_name:
            scope[def_name] = "index"
        return

    # Linalg named ops (memref semantics, 12 supported named ops).
    # Structure: linalg.OP ins(%a[, %b] : T[, T]) outs(%c : T) [attrs]
    # Validation:
    #   - Each ins/outs SSA name must be in scope.
    #   - Each SSA's scope-recorded type must match the declared clause type
    #     (or its element type for scalar-ins, e.g., linalg.fill).
    #   - For elemwise ops, all declared types should match — falls out of
    #     the per-slot type check since each is compared to its clause type
    #     and the clause types themselves are not enforced to be equal
    #     (the grammar already splits by op domain).
    #   - No new defs (memref outs is an existing buffer being written).
    if head in _LINALG_OPS:
        ins_clause  = _extract_clause(rhs, "ins")
        outs_clause = _extract_clause(rhs, "outs")
        if ins_clause is not None:
            for name, ty in _parse_linalg_clause(ins_clause):
                # linalg.fill takes a scalar ins whose type must match the
                # OUT memref element type. For the in-scope check we verify
                # the ins SSA was declared with exactly the clause type
                # (scalar-vs-scalar); element-type consistency between ins
                # and outs is covered by per-slot checks falling through.
                _check_use(name, ty, scope, line, violations)
        if outs_clause is not None:
            for name, ty in _parse_linalg_clause(outs_clause):
                _check_use(name, ty, scope, line, violations)
        # For linalg.fill: additionally check that ins scalar type matches
        # outs memref element type.
        if head == "linalg.fill" and ins_clause and outs_clause:
            ins_pairs  = _parse_linalg_clause(ins_clause)
            outs_pairs = _parse_linalg_clause(outs_clause)
            if ins_pairs and outs_pairs:
                scalar_ty = _normalize(ins_pairs[0][1])
                elem_ty   = _memref_or_scalar_element(outs_pairs[0][1])
                if scalar_ty != _normalize(elem_ty):
                    violations.append(ScopeViolation(
                        "type_mismatch",
                        f"linalg.fill scalar={scalar_ty} vs memref_elem={elem_ty}",
                        line,
                    ))
        return

    # StableHLO named ops (tensor semantics).
    if head in _STABLEHLO_OPS:
        if head in _STABLEHLO_EW_BIN:
            # %c = stablehlo.add %a, %b : tensor<NxT>
            refs = _ssa_refs(rhs)
            ty = _last_colon_type(rhs)
            if ty and len(refs) >= 2:
                _check_use(refs[0], ty, scope, line, violations)
                _check_use(refs[1], ty, scope, line, violations)
            if def_name and ty:
                scope[def_name] = ty
            return
        if head in _STABLEHLO_EW_UN:
            # %c = stablehlo.abs %a : tensor<NxT>
            refs = _ssa_refs(rhs)
            ty = _last_colon_type(rhs)
            if ty and len(refs) >= 1:
                _check_use(refs[0], ty, scope, line, violations)
            if def_name and ty:
                scope[def_name] = ty
            return
        if head in _STABLEHLO_TYPED_SIG:
            # %c = stablehlo.transpose %a, attr = ... : (tensor<IN>) -> tensor<OUT>
            refs = _ssa_refs(rhs)
            in_ty, out_ty = _stablehlo_type_sig(rhs)
            if in_ty and len(refs) >= 1:
                _check_use(refs[0], in_ty, scope, line, violations)
            if def_name and out_ty:
                scope[def_name] = out_ty
            return
        if head in _STABLEHLO_BINARY_TYPED:
            # %c = stablehlo.dot_general %a, %b, attr = ... : (tensor<A>, tensor<B>) -> tensor<C>
            refs = _ssa_refs(rhs)
            in_tys, out_ty = _stablehlo_binary_sig(rhs)
            if in_tys and len(refs) >= 2:
                _check_use(refs[0], in_tys[0], scope, line, violations)
                _check_use(refs[1], in_tys[1], scope, line, violations)
            if def_name and out_ty:
                scope[def_name] = out_ty
            return

    # Abstain: unknown op. If there's an LHS, record the def with the trailing
    # type so downstream uses don't spuriously fire undef_use.
    if def_name:
        ty = _last_colon_type(rhs)
        if ty:
            scope[def_name] = ty


def _split_op_lines(body: str) -> list[str]:
    """Split a function body into one logical op per line.

    Handles op lines that the model split across newlines (the gen grammar's
    `WS: /[ \\t\\n]{1,8}/` lets whitespace include \\n, so a valid op may be
    emitted as `%0 = arith.addi %a ,\\n  %b : i32`). We glue lines together
    when the previous line ends in a continuation indicator (`,`, `=`, `(`,
    `[`, `<`) OR when bracket/angle/paren depth from prior lines is non-zero.
    """
    raw = [ln for ln in body.splitlines() if ln.strip()]
    out: list[str] = []
    pending = ""
    depth_par = depth_br = depth_ang = 0
    for ln in raw:
        if pending:
            pending = pending.rstrip() + " " + ln.strip()
        else:
            pending = ln
        # recompute depth from pending so multi-line memref<...>x</...> works.
        depth_par = depth_br = depth_ang = 0
        for ch in pending:
            if ch == "(":
                depth_par += 1
            elif ch == ")":
                depth_par = max(0, depth_par - 1)
            elif ch == "[":
                depth_br += 1
            elif ch == "]":
                depth_br = max(0, depth_br - 1)
            elif ch == "<":
                depth_ang += 1
            elif ch == ">":
                depth_ang = max(0, depth_ang - 1)
        stripped = pending.rstrip()
        last = stripped[-1:] if stripped else ""
        continuation = last in ",=([<:" or stripped.endswith("->")
        if continuation or depth_par or depth_br or depth_ang:
            continue
        out.extend(_split_statements(pending))
        pending = ""
    if pending.strip():
        out.extend(_split_statements(pending))
    return out


# A statement starts at `%res = ...`, at `return`, or at a bare dialect op
# (no result). Whitespace in the generation grammar may include newlines, so a
# model can legally emit `... : f32 return %0 : i32` on one physical line; the
# scope checker must still see two statements.
_STMT_START_RE = re.compile(
    r"(?<=[^\s=,(:])\s+(?=(?:%[\w#$.]+(?:\s*,\s*%[\w#$.]+)*\s*=\s|return\b|"
    r"(?:arith|memref|linalg|func|stablehlo|scf|math|cf|tensor|vector|affine)\.[a-z_]+\b))"
)


def _split_statements(line: str) -> list[str]:
    parts = [p for p in _STMT_START_RE.split(line) if p and p.strip()]
    return parts if parts else [line]


def _extract_functions(text: str) -> list[tuple[str, str]]:
    """Return (header, body) for each func.func in `text`.

    Body is the content between the matching `{` and `}` of the function.
    Tracks brace depth to handle nested regions (blocks, SCF ops).
    """
    out: list[tuple[str, str]] = []
    for m in _FUNC_HEADER_RE.finditer(text):
        # Find the `{` that opens this function's body.
        i = text.find("{", m.end())
        if i < 0:
            continue
        depth = 1
        j = i + 1
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth != 0:
            continue  # unbalanced braces — treat as parse error elsewhere
        header = text[m.start():i]
        body = text[i + 1:j - 1]
        out.append((header, body))
    return out


def validate(mlir_text: str) -> ScopeReport:
    """Post-hoc SSA-scope + type-consistency check.

    passed=True iff every function in `mlir_text` is free of undefined-SSA
    uses and trailing-type mismatches under our op pattern dispatcher. Any
    unrecognized op is abstained on and cannot cause a reject.
    """
    violations: list[ScopeViolation] = []
    funcs = _extract_functions(mlir_text)
    if not funcs:
        # No func.func found. Don't reject — we'd be penalizing generations
        # that emit dialect-only IR for downstream verification.
        return ScopeReport(passed=True, violations=[])
    for header, body in funcs:
        scope: dict[str, str] = {}
        hdr_match = _FUNC_HEADER_RE.search(header)
        if hdr_match:
            for pm in _PARAM_RE.finditer(hdr_match.group("params") + ","):
                scope[pm.group("name")] = _normalize(pm.group("type"))
        for line in _split_op_lines(body):
            _handle_op_line(line, scope, violations)
    return ScopeReport(passed=not violations, violations=violations)


# ---------------- rejection-sampling hook ----------------

def accept_or_reject(mlir_text: str) -> tuple[bool, ScopeReport]:
    """Thin wrapper around `validate` that returns (accept_bool, report).

    accept_bool is the rejection-sampling decision signal: True → keep this
    generation; False → resample with a different seed.
    """
    rep = validate(mlir_text)
    return rep.passed, rep


def summarize_violations(violations: Iterable[ScopeViolation]) -> dict[str, int]:
    """Count violations by kind (useful for per-run diagnostics)."""
    counts: dict[str, int] = {}
    for v in violations:
        counts[v.kind] = counts.get(v.kind, 0) + 1
    return counts
