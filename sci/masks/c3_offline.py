"""Offline per-token C3 (SSA scope) mask for the training loss.

The inline C1/C2 mask is context-free; the scope layer only enters training as a sparse
sequence reward. This module builds a *dense* C3 signal after the fact: for every generated
token that is emitting an SSA operand use (`%name` on the right-hand side of an op, in a
`return`, or inside `ins(...)`/`outs(...)`), the allowed set is the set of vocabulary tokens
that keep the partially emitted name a prefix of some name currently in scope (function
parameters plus the left-hand sides of statements that ended before this use). Positions that
are not inside an operand name carry no constraint. The resulting rows feed the same
legal-mass loss as the grammar mask (see `sci.train.spine`), which decouples the *class* of the
constraint (context-sensitive) from the *density* of its training signal.

Scope semantics are deliberately the simple ones the generation grammars admit: one function
per module, flat body, a name is in scope from the end of its defining line to the end of the
function. `sci.constraints.scope` remains the oracle for rewards and evaluation.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

_IDENT = r"[A-Za-z0-9_$.]"
_USE_RE = re.compile(r"%(?P<name>[A-Za-z0-9_]+)")
_LHS_RE = re.compile(r"^\s*%(?P<name>[A-Za-z0-9_]+)\s*=", re.M)
_FUNC_HEADER_RE = re.compile(r"func\.func\s+(?:private\s+)?@[A-Za-z_][A-Za-z0-9_]*\s*\((?P<params>[^)]*)\)")
_IDENT_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_$.")


def scope_events(text: str) -> tuple[list[tuple[int, int, str]], dict[str, int]]:
    """(uses, defs): uses = [(start, end, name)] character spans of `%name` operand uses (span covers
    the `%` and the name); defs = {name: offset} where the name becomes visible (end of its line,
    or the header's `{` for parameters)."""
    defs: dict[str, int] = {}
    header_end = -1
    m = _FUNC_HEADER_RE.search(text)
    if m:
        header_end = m.end()
        brace = text.find("{", header_end)
        avail = brace + 1 if brace >= 0 else header_end
        for pm in _USE_RE.finditer(m.group("params")):
            defs.setdefault(pm.group("name"), avail)
    for lm in _LHS_RE.finditer(text):
        eol = text.find("\n", lm.end())
        defs.setdefault(lm.group("name"), (eol + 1) if eol >= 0 else len(text))
    lhs_spans = {(lm.start(1) - 1, lm.end(1)) for lm in _LHS_RE.finditer(text)}
    uses = []
    for um in _USE_RE.finditer(text):
        if um.start() < header_end or (um.start(), um.end()) in lhs_spans:
            continue
        uses.append((um.start(), um.end(), um.group("name")))
    return uses, defs


def in_scope_at(defs: dict[str, int], offset: int) -> list[str]:
    return sorted(n for n, o in defs.items() if o <= offset)


class C3MaskBuilder:
    """Builds per-token allowed sets at SSA-use positions for one tokenizer."""

    def __init__(self, tokenizer, vocab_size: int | None = None):
        self.tok = tokenizer
        V = vocab_size or len(getattr(tokenizer, "_tokenizer", tokenizer))
        self.vocab_size = V
        self.by_str: dict[str, list[int]] = {}
        self.by_close: dict[str, list[int]] = {}
        for i in range(V):
            s = tokenizer.decode([i])
            if not s or "�" in s:
                continue
            self.by_str.setdefault(s, []).append(i)
            # every position where an identifier run ends before the end of the string
            j = 0
            while j < len(s):
                if s[j] in _IDENT_CHARS:
                    k = j
                    while k < len(s) and s[k] in _IDENT_CHARS:
                        k += 1
                    if k < len(s):
                        self.by_close.setdefault(s[:k], []).append(i)
                    j = k
                else:
                    j += 1

    @lru_cache(maxsize=65536)
    def allowed_for(self, pre: str, remainders: tuple[str, ...]) -> np.ndarray:
        """Token ids whose string starts with `pre` (the non-name text emitted by this token before the
        name) and then either continues one of the `remainders` (name tail still open, token ends inside
        the name) or completes it and continues with a non-identifier character."""
        ids: set[int] = set()
        for r in remainders:
            for k in range(1, len(r) + 1):
                ids.update(self.by_str.get(pre + r[:k], ()))
            ids.update(self.by_close.get(pre + r, ()))
        return np.array(sorted(ids), dtype=np.int32)

    def token_offsets(self, gen_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
        """Decoded text and the character span of each token (prefix-decoding, exact for ASCII output)."""
        spans, prev, text = [], 0, ""
        for i in range(1, len(gen_ids) + 1):
            text = self.tok.decode(gen_ids[:i])
            spans.append((prev, len(text))); prev = len(text)
        return text, spans

    def build(self, gen_ids: list[int]) -> list[tuple[int, np.ndarray]]:
        """[(token position, allowed ids)] for every generated token that emits part of an operand name
        with at least one name in scope. Tokens that emit part of a name when nothing is in scope are
        skipped (no legal continuation exists; the sequence reward handles those)."""
        text, spans = self.token_offsets(gen_ids)
        uses, defs = scope_events(text)
        out = []
        for a, b, _name in uses:
            names = in_scope_at(defs, a)
            if not names:
                continue
            for pos, (ts, te) in enumerate(spans):
                if te <= a or ts >= b:
                    continue
                name_start = a + 1
                if ts < name_start:
                    pre, partial = text[ts:name_start], ""
                else:
                    pre, partial = "", text[name_start:ts]
                if te <= name_start:
                    continue  # token ends at or before the name (e.g. emits just " %"): nothing to constrain yet
                rem = tuple(sorted({n[len(partial):] for n in names if n.startswith(partial) and len(n) > len(partial)}))
                if not rem:
                    continue
                ids = self.allowed_for(pre, rem)
                if len(ids):
                    out.append((pos, ids))
        return out


def rows_to_bool(rows: list[tuple[int, np.ndarray]], T: int, vocab_size: int) -> np.ndarray:
    """Dense (T, V) bool with all-True rows at unconstrained positions."""
    m = np.ones((T, vocab_size), dtype=bool)
    for pos, ids in rows:
        m[pos] = False; m[pos, ids] = True
    return m
