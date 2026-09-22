"""Four-layer checker for the synthetic SSA language.

`check(program)` parses the text with a small recursive-descent parser over the C1 surface syntax
(any op with any type, so type errors are reported as C2 rather than parse failures) and then
runs the three context-sensitive layers over the statement tree:

  C1 parse_valid — the text is one well-formed `prog { ... ret %x : T }`
  C2 type_ok     — op family matches the trailing type's domain (`addi` ↔ `int[...]`,
                   `addf` ↔ `flt[...]`); integer literals pair with int types, decimals with flt
  C3 scope_ok    — every use names a value defined earlier in an enclosing region (names defined
                   inside a `{ }` block are dead after it), no value is defined twice while in
                   scope, and operand domains agree with the op's trailing type
  C4 shape_ok    — binary operands and the result carry the same shape

The shape-domain split mirrors the MLIR stack: the cross-statement type-domain check is the
`type_mismatch` half of v3 there, and shapes are the value-dependent layer that has no mask.
Dependency distance is measured on the flattened statement order (blocks contribute their
statements, braces are not statements).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from sci.synth.grammar import DOMAINS, FLT_OPS, INT_OPS

_TOKEN_RE = re.compile(
    r"(?P<ws>[ \t\n]+)|(?P<ssa>%[a-z][a-z0-9]{0,7})|(?P<type>(?:int|flt)\[[1-9][0-9]?(?:x[1-9][0-9]?)?\])"
    r"|(?P<flt>-?[0-9]{1,4}\.[0-9]{1,3})|(?P<int>-?[0-9]{1,4})|(?P<word>[a-z]+)|(?P<punct>[{}=,:])"
)
_TYPE_RE = re.compile(r"^(int|flt)(\[[0-9x]+\])$")
OP_DOMAIN = {**{o: "int" for o in INT_OPS}, **{o: "flt" for o in FLT_OPS}}


@dataclass
class Stmt:
    """One statement: `const`, a binary op, or `ret`."""
    kind: str                 # const | bin | ret
    op: str                   # const / addi ... / ret
    result: str | None        # defined name (None for ret)
    operands: list[str]       # used names
    literal: str | None       # const literal text (None otherwise)
    type: str                 # trailing type text, e.g. int[4]
    depth: int                # nesting depth (0 = program body)
    index: int = -1           # flat statement index
    line: str = ""


@dataclass
class Block:
    depth: int
    items: list = field(default_factory=list)  # Stmt | Block


@dataclass
class Violation:
    layer: int                # 1..4
    kind: str                 # parse | type_domain | literal | undef_use | dead_use | redef | type_mismatch | shape_mismatch
    detail: str
    line: str = ""

    def __str__(self) -> str:
        return f"[C{self.layer}:{self.kind}] {self.detail} || {self.line!r}"


@dataclass
class CheckReport:
    parse_valid: bool
    type_ok: bool
    scope_ok: bool
    shape_ok: bool
    violations: list[Violation] = field(default_factory=list)
    tree: Block | None = None

    @property
    def passed(self) -> bool:
        return self.parse_valid and self.type_ok and self.scope_ok and self.shape_ok

    def __bool__(self) -> bool:
        return self.passed

    def as_dict(self) -> dict:
        return {"parse_valid": self.parse_valid, "type_ok": self.type_ok, "scope_ok": self.scope_ok,
                "shape_ok": self.shape_ok, "passed": self.passed, "violations": [str(v) for v in self.violations]}


def split_type(t: str) -> tuple[str, str]:
    """`int[2x3]` → ("int", "[2x3]")."""
    m = _TYPE_RE.match(t)
    if not m:
        raise ValueError(f"bad type {t!r}")
    return m.group(1), m.group(2)


# ---------------- parsing (C1) ----------------

class _ParseError(Exception):
    pass


class _Parser:
    def __init__(self, text: str):
        self.toks: list[tuple[str, str]] = []
        pos = 0
        while pos < len(text):
            m = _TOKEN_RE.match(text, pos)
            if m is None:
                raise _ParseError(f"unexpected character {text[pos]!r} at {pos}")
            pos = m.end()
            if m.lastgroup != "ws":
                self.toks.append((m.lastgroup, m.group()))
        self.i = 0
        self.flat: list[Stmt] = []

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, kind: str | None = None, value: str | None = None) -> str:
        t = self.peek()
        if t is None:
            raise _ParseError(f"unexpected end of input (wanted {value or kind})")
        if (kind and t[0] != kind) or (value and t[1] != value):
            raise _ParseError(f"expected {value or kind}, got {t[1]!r}")
        self.i += 1
        return t[1]

    def program(self) -> Block:
        self.take("word", "prog"); self.take("punct", "{")
        body = self.block(0, top=True)
        if self.peek() is not None:
            raise _ParseError(f"trailing input {self.peek()[1]!r}")
        return body

    def block(self, depth: int, top: bool = False) -> Block:
        b = Block(depth)
        while True:
            t = self.peek()
            if t is None:
                raise _ParseError("unterminated block")
            if t == ("punct", "}"):
                self.i += 1
                if top and not (b.items and isinstance(b.items[-1], Stmt) and b.items[-1].kind == "ret"):
                    raise _ParseError("program body must end with `ret`")
                if not b.items:
                    raise _ParseError("empty block")
                return b
            if t == ("punct", "{"):
                self.i += 1
                b.items.append(self.block(depth + 1))
                continue
            b.items.append(self.stmt(depth, top))
            if b.items[-1].kind == "ret" and self.peek() != ("punct", "}"):
                raise _ParseError("`ret` must be the last statement")

    def stmt(self, depth: int, top: bool) -> Stmt:
        start = self.i
        t = self.peek()
        if t == ("word", "ret"):
            if not top:
                raise _ParseError("`ret` inside a block")
            self.i += 1
            name = self.take("ssa"); self.take("punct", ":"); ty = self.take("type")
            st = Stmt("ret", "ret", None, [name], None, ty, depth)
        else:
            res = self.take("ssa"); self.take("punct", "=")
            op = self.take("word")
            if op == "const":
                lt = self.peek()
                if lt is None or lt[0] not in ("int", "flt"):
                    raise _ParseError("const needs a literal")
                self.i += 1
                self.take("punct", ":"); ty = self.take("type")
                st = Stmt("const", "const", res, [], lt[1], ty, depth)
            elif op in OP_DOMAIN:
                a = self.take("ssa"); self.take("punct", ","); b = self.take("ssa")
                self.take("punct", ":"); ty = self.take("type")
                st = Stmt("bin", op, res, [a, b], None, ty, depth)
            else:
                raise _ParseError(f"unknown op {op!r}")
        st.line = " ".join(v for _, v in self.toks[start:self.i])
        st.index = len(self.flat)
        self.flat.append(st)
        return st


def parse(program: str) -> tuple[Block | None, list[Stmt], str | None]:
    """(tree, flat statements, error) — tree is None on a parse error."""
    try:
        p = _Parser(program)
        tree = p.program()
        return tree, p.flat, None
    except _ParseError as e:
        return None, [], str(e)


# ---------------- context-sensitive layers ----------------

def _walk(block: Block, scope: dict[str, str], stmts_out: list[tuple[Stmt, dict[str, str]]]) -> None:
    """Pre-order walk recording, for every statement, a snapshot of the names in scope before it."""
    local: dict[str, str] = dict(scope)
    for it in block.items:
        if isinstance(it, Block):
            _walk(it, local, stmts_out)
        else:
            stmts_out.append((it, dict(local)))
            if it.result is not None:
                local[it.result] = it.type


def check(program: str) -> CheckReport:
    """Run all four layers; every violation is tagged with its layer."""
    tree, flat, err = parse(program)
    if tree is None:
        return CheckReport(False, False, False, False, [Violation(1, "parse", err or "parse error")])
    vio: list[Violation] = []
    # C2: op domain vs trailing type; literal vs type
    for st in flat:
        dom, _ = split_type(st.type)
        if st.kind == "bin" and OP_DOMAIN[st.op] != dom:
            vio.append(Violation(2, "type_domain", f"{st.op} with {st.type}", st.line))
        if st.kind == "const":
            lit_dom = "flt" if "." in st.literal else "int"
            if lit_dom != dom:
                vio.append(Violation(2, "literal", f"{st.literal} : {st.type}", st.line))
    # C3 / C4 with region scoping
    seen_all: set[str] = set()
    ordered: list[tuple[Stmt, dict[str, str]]] = []
    _walk(tree, {}, ordered)
    for st, scope in ordered:
        dom, shape = split_type(st.type)
        for name in st.operands:
            if name not in scope:
                kind = "dead_use" if name in seen_all else "undef_use"
                vio.append(Violation(3, kind, f"{name} not in scope", st.line))
                continue
            odom, oshape = split_type(scope[name])
            if odom != dom:
                vio.append(Violation(3, "type_mismatch", f"{name} is {scope[name]}, used as {st.type}", st.line))
            elif oshape != shape:
                vio.append(Violation(4, "shape_mismatch", f"{name} is {scope[name]}, used as {st.type}", st.line))
        if st.result is not None:
            if st.result in scope:
                vio.append(Violation(3, "redef", f"{st.result} already defined", st.line))
            seen_all.add(st.result)
    return CheckReport(True, not any(v.layer == 2 for v in vio), not any(v.layer == 3 for v in vio),
                       not any(v.layer == 4 for v in vio), vio, tree)


def accept_or_reject(program: str) -> tuple[bool, CheckReport]:
    """(accept, report): accept iff C1–C3 hold (the C3 rejection-sampling filter; C4 has no mask and
    is reported, not filtered, exactly as in the MLIR stack)."""
    rep = check(program)
    return rep.parse_valid and rep.type_ok and rep.scope_ok, rep


def scope_ok(program: str) -> bool:
    """C3 verdict alone (parse failures count as C3 failures for v3 accounting: nothing to scope)."""
    rep = check(program)
    return rep.parse_valid and rep.scope_ok


# ---------------- structural features ----------------

def dep_distances(program: str) -> list[tuple[str, int, int, int]]:
    """(name, def_index, use_index, distance) for every use whose def is in scope, in flat statement
    order; distance = statements strictly between def and use."""
    tree, flat, err = parse(program)
    if tree is None:
        return []
    ordered: list[tuple[Stmt, dict[str, str]]] = []
    _walk(tree, {}, ordered)
    def_index: dict[str, int] = {}
    out = []
    for st, scope in ordered:
        for name in st.operands:
            if name in scope and name in def_index:
                out.append((name, def_index[name], st.index, st.index - def_index[name] - 1))
        if st.result is not None:
            def_index[st.result] = st.index
    return out


def max_dep_distance(program: str) -> int:
    ds = [d for _, _, _, d in dep_distances(program)]
    return max(ds) if ds else -1


def op_multiset(program: str) -> Counter:
    """Multiset of op mnemonics (`const`, `addi`, ... ; `ret` excluded)."""
    _, flat, _ = parse(program)
    return Counter(st.op for st in flat if st.kind != "ret")


def ret_type(program: str) -> str | None:
    _, flat, _ = parse(program)
    return flat[-1].type if flat and flat[-1].kind == "ret" else None


def op_jaccard(a: Counter, b: Counter) -> float:
    inter = sum((a & b).values()); union = sum((a | b).values())
    return inter / union if union else 1.0


def reward_fn(text: str, gold: str | None = None, kind: str = "task") -> tuple[float, dict]:
    """Sequence reward with the MLIR spine's contract: kind='verify' → all four layers hold;
    kind='task' → verify · [ret type matches gold] · Jaccard(op multiset, gold's). `parts` carries the
    keys the training driver reads (`parse`, `scope`, `verify`) plus the four layer flags."""
    rep = check(text)
    vv = rep.passed
    parts = {"parse": rep.parse_valid, "scope": rep.parse_valid and rep.scope_ok, "verify": vv,
             "type_ok": rep.type_ok, "shape_ok": rep.shape_ok}
    if kind == "verify" or gold is None:
        return float(vv), parts
    sig = float(ret_type(text) == ret_type(gold)) if vv else 0.0
    jac = op_jaccard(op_multiset(text), op_multiset(gold)) if vv else 0.0
    parts.update({"sig": round(sig, 3), "op_jaccard": round(jac, 3)})
    return float(vv) * sig * jac, parts


def signature_task_reward(gen: str, gold: str) -> float:
    """Scalar task-conditional reward (validity × structural agreement with the gold)."""
    return reward_fn(gen, gold, "task")[0]


__all__ = ["check", "accept_or_reject", "scope_ok", "dep_distances", "max_dep_distance", "op_multiset",
           "ret_type", "reward_fn", "signature_task_reward", "CheckReport", "Violation", "DOMAINS"]
