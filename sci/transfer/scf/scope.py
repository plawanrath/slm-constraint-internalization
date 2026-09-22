"""C3 for the scf target: region-aware SSA scope + type consistency.

Extends the flat, per-function validator in `sci.constraints.scope` (whose op
handlers, header parsing and report types are reused by import) with a scope
stack driven by region braces:

- `scf.for %i = %lb to %ub step %s [iter_args(%a = %x, ...)] [-> (T, ...)] {`
  checks %lb/%ub/%s are in scope as `index` and each init %x matches its result
  type, then opens a region in which %i (`index`) and the iter_args are visible.
- `scf.if %c [-> (T, ...)] {` checks %c is `i1` and opens the then-region;
  `} else {` closes it and opens the else-region with a fresh scope.
- `scf.yield %v, ... : T, ...` operands must be in scope with the declared types,
  and the declared types must equal the enclosing region op's result types
  (`yield_mismatch`); a bare `scf.yield` is only legal without results.
- `}` closes the innermost region: block arguments and every value defined
  inside it go out of scope; the region op's results (LHS names) become defined
  in the enclosing scope once the op is complete. A result-bearing region that
  closes without a yield, or an `scf.if` with results but no else, is a
  `yield_mismatch`.

Simple op lines are delegated to `sci.constraints.scope._handle_op_line` over a
`ChainMap` of the scope stack, so undefined uses and trailing-type mismatches
are reported exactly as in the flat validator. Unrecognized lines are abstained
on. Same `(ok, report)` contract as `sci.constraints.scope.accept_or_reject`.
"""
from __future__ import annotations

import re
from collections import ChainMap
from dataclasses import dataclass, field

from sci.constraints.scope import (
    _FUNC_HEADER_RE,
    _PARAM_RE,
    ScopeReport,
    ScopeViolation,
    _check_use,
    _extract_functions,
    _handle_op_line,
    _normalize,
    _split_statements,
)

_NAME = r"%[A-Za-z0-9_$.\-]+(?:#\d+)?(?::\d+)?"
_LHS = rf"(?:(?P<lhs>{_NAME}(?:\s*,\s*{_NAME})*)\s*=\s*)?"
_FOR_RE = re.compile(
    _LHS + r"scf\.for\s+%(?P<iv>[\w$.\-]+)\s*=\s*%(?P<lb>[\w$.\-]+)\s+to\s+%(?P<ub>[\w$.\-]+)"
    r"\s+step\s+%(?P<step>[\w$.\-]+)(?:\s+iter_args\s*\((?P<iters>[^)]*)\))?"
    r"(?:\s*->\s*(?P<rtypes>\([^)]*\)|[^\s{:]+))?(?:\s*:\s*(?P<ivtype>[^\s{]+))?\s*\{?\s*$"
)
_IF_RE = re.compile(
    _LHS + r"scf\.if\s+%(?P<cond>[\w$.\-]+)(?:\s*->\s*(?P<rtypes>\([^)]*\)|[^\s{]+))?\s*\{?\s*$"
)
_YIELD_RE = re.compile(r"scf\.yield(?:\s+(?P<vals>%[^:]*?)\s*:\s*(?P<types>.+?))?\s*$")
_SSA_NAME_RE = re.compile(r"%([A-Za-z0-9_$.\-]+)")


def _split_types(s: str) -> list[str]:
    """Top-level comma split of a type list (parens stripped), respecting `<...>`."""
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    tail = "".join(cur)
    if tail.strip():
        out.append(tail)
    return [_normalize(t) for t in out if t.strip()]


def _names(s: str | None) -> list[str]:
    return _SSA_NAME_RE.findall(s or "")


@dataclass
class _Frame:
    kind: str                       # "for" | "if"
    line: str
    results: list[str]              # LHS names (may be empty)
    rtypes: list[str]               # declared result types
    yielded: bool = False           # a yield was seen in the current branch
    in_else: bool = False
    then_closed: bool = False       # if-frame: then-region closed, else may follow
    branches_yielded: list[bool] = field(default_factory=list)


class _Walker:
    def __init__(self, params: dict[str, str]):
        self.scopes: list[dict[str, str]] = [dict(params)]
        self.frames: list[_Frame] = []
        self.violations: list[ScopeViolation] = []

    # -- scope helpers --
    @property
    def view(self) -> ChainMap:
        return ChainMap(*reversed(self.scopes))  # innermost first for writes and lookups

    def _use(self, name: str, ty: str | None, line: str) -> None:
        _check_use(name, ty, self.view, line, self.violations)

    def _bad(self, kind: str, detail: str, line: str) -> None:
        self.violations.append(ScopeViolation(kind, detail, line))

    # -- region ops --
    def open_for(self, m: re.Match, line: str) -> None:
        for r in (m.group("lb"), m.group("ub"), m.group("step")):
            self._use(r, m.group("ivtype") or "index", line)
        rtypes = _split_types(m.group("rtypes")) if m.group("rtypes") else []
        results = _names(m.group("lhs"))
        iters: list[tuple[str, str]] = []
        for piece in (m.group("iters") or "").split(","):
            if "=" not in piece:
                continue
            a, x = piece.split("=", 1)
            iters.append((_names(a)[0] if _names(a) else "", _names(x)[0] if _names(x) else ""))
        if m.group("iters") is not None and len(iters) != len(rtypes):
            self._bad("yield_mismatch", f"{len(iters)} iter_args vs {len(rtypes)} result types", line)
        if results and len(results) != len(rtypes):
            self._bad("yield_mismatch", f"{len(results)} results vs {len(rtypes)} result types", line)
        inner: dict[str, str] = {m.group("iv"): m.group("ivtype") or "index"}
        for k, (arg, init) in enumerate(iters):
            ty = rtypes[k] if k < len(rtypes) else None
            if init:
                self._use(init, ty, line)
            if arg and ty:
                inner[arg] = ty
        self.frames.append(_Frame("for", line, results, rtypes))
        self.scopes.append(inner)

    def open_if(self, m: re.Match, line: str) -> None:
        self._use(m.group("cond"), "i1", line)
        rtypes = _split_types(m.group("rtypes")) if m.group("rtypes") else []
        results = _names(m.group("lhs"))
        if results and len(results) != len(rtypes):
            self._bad("yield_mismatch", f"{len(results)} results vs {len(rtypes)} result types", line)
        self.frames.append(_Frame("if", line, results, rtypes))
        self.scopes.append({})

    def yield_(self, m: re.Match, line: str) -> None:
        if not self.frames:
            self._bad("yield_mismatch", "scf.yield outside any region", line)
            return
        fr = self.frames[-1]
        vals, types = _names(m.group("vals")), (_split_types(m.group("types")) if m.group("types") else [])
        for v, t in zip(vals, types):
            self._use(v, t, line)
        if len(vals) != len(types):
            self._bad("yield_mismatch", f"{len(vals)} operands vs {len(types)} types", line)
        elif types != fr.rtypes:
            self._bad("yield_mismatch", f"yield types {types} vs region results {fr.rtypes}", line)
        fr.yielded = True

    def _finish_branch(self, fr: _Frame) -> None:
        if fr.rtypes and not fr.yielded:
            self._bad("yield_mismatch", f"region with results {fr.rtypes} has no scf.yield", fr.line)
        fr.branches_yielded.append(fr.yielded)
        fr.yielded = False
        self.scopes.pop()

    def _complete(self, fr: _Frame) -> None:
        """Region op finished: define its results in the enclosing scope."""
        if fr.kind == "if" and fr.rtypes and not fr.in_else:
            self._bad("yield_mismatch", "scf.if with results has no else region", fr.line)
        for name, ty in zip(fr.results, fr.rtypes):
            self.scopes[-1][name] = ty

    def close(self, line: str) -> None:
        if not self.frames:
            self._bad("parse", "unbalanced '}'", line)
            return
        fr = self.frames[-1]
        self._finish_branch(fr)
        if fr.kind == "if" and not fr.in_else:
            fr.then_closed = True     # wait for a possible `else {`
            return
        self.frames.pop()
        self._complete(fr)

    def else_(self, line: str) -> None:
        fr = self.frames[-1] if self.frames else None
        if fr is None or fr.kind != "if" or not fr.then_closed:
            self._bad("parse", "else without a closed scf.if then-region", line)
            return
        fr.then_closed = False
        fr.in_else = True
        self.scopes.append({})

    def settle_pending_if(self) -> None:
        """A closed then-region not followed by `else` completes the scf.if."""
        if self.frames and self.frames[-1].kind == "if" and self.frames[-1].then_closed:
            fr = self.frames.pop()
            self._complete(fr)

    # -- driver --
    def line(self, raw: str) -> None:
        s = raw.strip()
        if not s:
            return
        if s.startswith("else"):
            self.else_(s)
            return
        self.settle_pending_if()
        if s == "}":
            self.close(s)
            return
        if s == "{":
            return
        m = _FOR_RE.match(s)
        if m:
            self.open_for(m, s); return
        m = _IF_RE.match(s)
        if m:
            self.open_if(m, s); return
        m = _YIELD_RE.match(s)
        if m:
            self.yield_(m, s); return
        _handle_op_line(s, self.view, self.violations)

    def finish(self) -> None:
        self.settle_pending_if()
        for fr in self.frames:
            self._bad("parse", f"unclosed {fr.kind} region", fr.line)


def _region_lines(body: str) -> list[str]:
    """One logical statement per line; region braces on their own lines."""
    text = body.replace("{", "{\n").replace("}", "\n}\n")
    out: list[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.endswith("{") and ln != "{":
            # statements before the region opener are separate; the opener itself
            # (`%r = scf.for %i = %lb ...`) must not be split at its inner `=`s
            parts = _split_statements(ln[:-1].rstrip())
            k = next((i for i, p in enumerate(parts) if "scf.for" in p or "scf.if" in p), len(parts) - 1)
            out.extend(parts[:k])
            out.append(" ".join(parts[k:]) + " {")
        else:
            out.extend(_split_statements(ln))
    return out


def validate(mlir_text: str) -> ScopeReport:
    """Region-aware SSA-scope + type-consistency check over every `func.func`."""
    violations: list[ScopeViolation] = []
    funcs = _extract_functions(mlir_text)
    if not funcs:
        return ScopeReport(passed=True, violations=[])
    for header, body in funcs:
        params: dict[str, str] = {}
        hm = _FUNC_HEADER_RE.search(header)
        if hm:
            for pm in _PARAM_RE.finditer(hm.group("params") + ","):
                params[pm.group("name")] = _normalize(pm.group("type"))
        w = _Walker(params)
        for line in _region_lines(body):
            w.line(line)
        w.finish()
        violations.extend(w.violations)
    return ScopeReport(passed=not violations, violations=violations)


def accept_or_reject(mlir_text: str) -> tuple[bool, ScopeReport]:
    """Rejection-sampling decision: (accept, report)."""
    rep = validate(mlir_text)
    return rep.passed, rep
