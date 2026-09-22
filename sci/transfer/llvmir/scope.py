"""C3 for LLVM IR: SSA def-before-use with dominance over the function's CFG.

Within a basic block a use must follow its definition. Across blocks a use in
block B is legal iff the defining block strictly dominates B (dominators are
computed over the CFG built from `br` terminators, entry = first block). A
`phi` is legal iff it sits at the top of its block, names exactly the block's
predecessors, and each incoming value is defined in (or in a block dominating)
the matching predecessor. Uses inside unreachable blocks are only checked for
existence, as the LLVM verifier does.

Unrecognized lines are abstained on: they neither add a definition (unless a
`%name =` LHS is present, which is registered so later uses do not fire
spuriously) nor have their uses checked, and are listed in
`LlvmScopeReport.abstained`. Abstention can never cause a reject.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SSA_RE = re.compile(r"%([A-Za-z0-9_.][A-Za-z0-9_.]*)")
_DEFINE_RE = re.compile(r"^\s*define\s+(?:\S+\s+)*?@[A-Za-z_][A-Za-z0-9_]*\s*\(([^)]*)\)")
_LABEL_RE = re.compile(r"^\s*([A-Za-z_.][A-Za-z0-9_.]*):\s*(.*)$")
_LHS_RE = re.compile(r"^\s*%([A-Za-z0-9_.][A-Za-z0-9_.]*)\s*=\s*(.*)$")
_PHI_IN_RE = re.compile(r"\[\s*([^,\]]+?)\s*,\s*%([A-Za-z0-9_.]+)\s*\]")
_LABEL_REF_RE = re.compile(r"label\s+%([A-Za-z0-9_.]+)")
_STMT_START_RE = re.compile(
    r"^\s*(?:%[A-Za-z0-9_.]+\s*=|store\b|br\b|ret\b|define\b|[A-Za-z_.][A-Za-z0-9_.]*:|\}|declare\b|;)"
)

_VALUE_OPS = {
    "add", "sub", "mul", "sdiv", "udiv", "srem", "urem", "and", "or", "xor", "shl", "lshr", "ashr",
    "fadd", "fsub", "fmul", "fdiv", "frem", "icmp", "fcmp", "select", "sext", "zext", "trunc",
    "sitofp", "uitofp", "fptosi", "fptoui", "fpext", "fptrunc", "alloca", "load", "phi",
}
_NO_RESULT_OPS = {"store", "ret", "br"}


@dataclass
class ScopeViolation:
    kind: str    # undef_use | use_before_def | not_dominated | bad_phi | undef_label | redefinition
    detail: str
    line: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.detail} || {self.line!r}"


@dataclass
class LlvmScopeReport:
    passed: bool
    violations: list[ScopeViolation] = field(default_factory=list)
    abstained: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class _Instr:
    line: str
    op: str
    lhs: str | None
    uses: list[str]                       # SSA value uses (labels excluded)
    phi_in: list[tuple[str | None, str]]  # (value name or None for literal, pred label)
    labels: list[str]


@dataclass
class _Block:
    name: str
    instrs: list[_Instr] = field(default_factory=list)
    succs: list[str] = field(default_factory=list)


# ---------------- statement splitting ----------------

def _statements(text: str) -> list[str]:
    """Logical statements: physical lines glued to the previous one unless they start a
    statement (the grammar's WS may include newlines inside an instruction)."""
    out: list[str] = []
    for raw in text.splitlines():
        ln = raw.split(";", 1)[0].rstrip()
        if not ln.strip():
            continue
        if out and not _STMT_START_RE.match(ln):
            out[-1] = out[-1] + " " + ln.strip()
        else:
            out.append(ln)
    return out


def _parse_instr(line: str) -> _Instr | None:
    """Structured view of one instruction line, or None if it is not recognized."""
    lhs = None
    rest = line.strip()
    m = _LHS_RE.match(rest)
    if m:
        lhs, rest = m.group(1), m.group(2).strip()
    op = rest.split(None, 1)[0] if rest else ""
    if lhs is not None and op not in _VALUE_OPS:
        return None
    if lhs is None and op not in _NO_RESULT_OPS:
        return None
    body = rest[len(op):]
    labels: list[str] = []
    phi_in: list[tuple[str | None, str]] = []
    if op == "br":
        labels = _LABEL_REF_RE.findall(body)
        body = _LABEL_REF_RE.sub(" ", body)
    if op == "phi":
        for val, pred in _PHI_IN_RE.findall(body):
            mv = _SSA_RE.fullmatch(val.strip())
            phi_in.append((mv.group(1) if mv else None, pred))
        uses: list[str] = []
    else:
        uses = [m.group(1) for m in _SSA_RE.finditer(body)]
    return _Instr(line=line, op=op, lhs=lhs, uses=uses, phi_in=phi_in, labels=labels)


# ---------------- function extraction ----------------

def _functions(text: str) -> list[tuple[list[str], list[_Block], list[str]]]:
    """(params, blocks, abstained_lines) per `define` in `text`."""
    out = []
    stmts = _statements(text)
    i = 0
    while i < len(stmts):
        m = _DEFINE_RE.match(stmts[i])
        if not m:
            i += 1
            continue
        params = [pm.group(1) for pm in _SSA_RE.finditer(m.group(1))]
        blocks: list[_Block] = []
        abstained: list[str] = []
        i += 1
        while i < len(stmts) and not stmts[i].strip().startswith("}"):
            ln = stmts[i]
            i += 1
            lm = _LABEL_RE.match(ln)
            if lm:
                blocks.append(_Block(lm.group(1)))
                ln = lm.group(2)
                if not ln.strip():
                    continue
            if not blocks:  # unlabelled entry block
                blocks.append(_Block("%entry"))
            ins = _parse_instr(ln)
            if ins is None:
                abstained.append(ln.strip())
                lm2 = _LHS_RE.match(ln.strip())
                if lm2:  # register the def so later uses do not fire spuriously
                    blocks[-1].instrs.append(_Instr(ln, "?", lm2.group(1), [], [], []))
                continue
            blocks[-1].instrs.append(ins)
        for b in blocks:
            if b.instrs and b.instrs[-1].op == "br":
                b.succs = list(b.instrs[-1].labels)
        out.append((params, blocks, abstained))
        i += 1
    return out


# ---------------- dominators ----------------

def _dominators(blocks: list[_Block]) -> dict[str, set[str]]:
    """dom[b] = set of blocks dominating b (including b). Unreachable blocks are absent."""
    if not blocks:
        return {}
    names = [b.name for b in blocks]
    succ = {b.name: [s for s in b.succs if s in names] for b in blocks}
    entry = names[0]
    reach: set[str] = set()
    stack = [entry]
    while stack:
        n = stack.pop()
        if n in reach:
            continue
        reach.add(n)
        stack.extend(succ[n])
    preds = {n: [p for p in reach if n in succ[p]] for n in reach}
    dom = {n: set(reach) for n in reach}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for n in names:
            if n not in reach or n == entry:
                continue
            new = set(reach)
            for p in preds[n]:
                new &= dom[p]
            new |= {n}
            if new != dom[n]:
                dom[n] = new
                changed = True
    return dom


# ---------------- validation ----------------

def validate(text: str) -> LlvmScopeReport:
    """Dominance-aware SSA scope check over every `define` in `text`."""
    violations: list[ScopeViolation] = []
    abstained_all: list[str] = []
    for params, blocks, abstained in _functions(text):
        abstained_all.extend(abstained)
        defs: dict[str, tuple[str, int]] = {}   # value -> (block, index)
        for b in blocks:
            for k, ins in enumerate(b.instrs):
                if ins.lhs is None:
                    continue
                if ins.lhs in defs or ins.lhs in params:
                    violations.append(ScopeViolation("redefinition", f"%{ins.lhs}", ins.line))
                defs[ins.lhs] = (b.name, k)
        block_names = {b.name for b in blocks}
        dom = _dominators(blocks)
        preds = {b.name: [p.name for p in blocks if b.name in p.succs] for b in blocks}

        def defined_at_end_of(name: str, blk: str) -> bool:
            """Value `name` is available at the end of block `blk` (params, or a def whose
            block dominates `blk`)."""
            if name in params:
                return True
            if name not in defs:
                return False
            return blk not in dom or defs[name][0] in dom[blk]

        for b in blocks:
            reachable = b.name in dom
            for k, ins in enumerate(b.instrs):
                for lab in ins.labels:
                    if lab not in block_names:
                        violations.append(ScopeViolation("undef_label", f"%{lab}", ins.line))
                if ins.op == "phi":
                    if any(prev.op != "phi" for prev in b.instrs[:k]):
                        violations.append(ScopeViolation("bad_phi", "phi not at top of block", ins.line))
                    incoming = [p for _, p in ins.phi_in]
                    if reachable and sorted(incoming) != sorted(preds[b.name]):
                        violations.append(ScopeViolation(
                            "bad_phi", f"incoming {sorted(incoming)} != predecessors {sorted(preds[b.name])}", ins.line))
                    for val, pred in ins.phi_in:
                        if val is None:
                            continue
                        if val not in defs and val not in params:
                            violations.append(ScopeViolation("undef_use", f"%{val}", ins.line))
                        elif reachable and pred in dom and not defined_at_end_of(val, pred):
                            violations.append(ScopeViolation(
                                "bad_phi", f"%{val} does not reach the end of predecessor %{pred}", ins.line))
                    continue
                for u in ins.uses:
                    if u in params:
                        continue
                    if u not in defs:
                        violations.append(ScopeViolation("undef_use", f"%{u}", ins.line))
                        continue
                    dblk, didx = defs[u]
                    if dblk == b.name:
                        if didx >= k:
                            violations.append(ScopeViolation("use_before_def", f"%{u}", ins.line))
                    elif reachable and dblk not in dom[b.name]:
                        violations.append(ScopeViolation(
                            "not_dominated", f"%{u} defined in %{dblk} does not dominate %{b.name}", ins.line))
    return LlvmScopeReport(passed=not violations, violations=violations, abstained=abstained_all)


def accept_or_reject(text: str) -> tuple[bool, LlvmScopeReport]:
    """Rejection-sampling hook: (accept, report), like `sci.constraints.scope.accept_or_reject`."""
    rep = validate(text)
    return rep.passed, rep
