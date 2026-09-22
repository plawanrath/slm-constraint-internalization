"""SSA-use positions for the scope probes.

A *use site* is a `%name` operand on the right-hand side of a statement (or in a `return`,
or in a linalg `ins(...)`/`outs(...)` clause). For every use site we record the name, the
set of names in scope just before the statement, a candidate set of names that are *not*
in scope there, and the token positions the probes read:

  pct_pos   token containing the `%` of the use (the model is about to emit the name);
  name_pos  token containing the first character of the name (the model has read it);
  pred_pos  position whose next-token logits predict the first name token (= name_pos - 1;
            equals pct_pos unless the tokenizer merges `%` with the name).

Scope semantics are those of `sci.constraints.scope.validate`: the scope is replayed with
`_handle_op_line` statement by statement, so `label == 0` for a *checked* site iff the
validator would report `undef_use` for it. "Checked" mirrors which operands the validator
actually inspects (unknown ops and a few degenerate forms are abstained on and flagged
`checked=False`). Out-of-scope candidates are the names the function defines later (the
validator has no liveness notion, so "dead" names do not exist for it) plus names defined
only in other functions of the same program.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from sci.constraints.scope import (
    _ARITH_BIN_FLOAT,
    _ARITH_BIN_INT,
    _extract_clause,
    _extract_functions,
    _F2I,
    _FLOAT_WIDEN,
    _FUNC_HEADER_RE,
    _handle_op_line,
    _I2F,
    _INDEX_CAST,
    _INT_WIDEN,
    _last_colon_type,
    _LINALG_OPS,
    _normalize,
    _PARAM_RE,
    _parse_linalg_clause,
    _split_op_lines,
    _SSA_RE,
    _ssa_refs,
    _STABLEHLO_BINARY_TYPED,
    _STABLEHLO_EW_BIN,
    _STABLEHLO_EW_UN,
    _STABLEHLO_TYPED_SIG,
    _stablehlo_binary_sig,
    _stablehlo_type_sig,
)

_ALL_REFS_HEADS = (
    _ARITH_BIN_INT | _ARITH_BIN_FLOAT | {"arith.cmpi", "arith.cmpf", "arith.select"}
    | _INT_WIDEN | _I2F | _F2I | _FLOAT_WIDEN | _INDEX_CAST | {"arith.bitcast"}
    | {"memref.load", "memref.dim"}
)


@dataclass
class DefSite:
    name: str
    char_pos: int            # offset of the `%` in the program text
    func_index: int
    stmt_index: int          # -1 for function parameters
    pct_pos: int = -1        # token containing the `%` (set by attach_token_positions)
    name_pos: int = -1


@dataclass
class UseSite:
    name: str
    char_pos: int
    func_index: int
    stmt_index: int
    in_scope: set[str] = field(default_factory=set)
    out_of_scope: set[str] = field(default_factory=set)
    checked: bool = True     # the validator inspects this operand
    pct_pos: int = -1
    name_pos: int = -1
    pred_pos: int = -1

    @property
    def label(self) -> int:
        return int(self.name in self.in_scope)


# ---------------- validator mirror: which operands does a statement check? ----------------

def _split_def(stmt: str) -> tuple[str | None, str, str]:
    """(def_name, rhs, head) exactly as `_handle_op_line` derives them."""
    stripped = stmt.strip()
    def_name = None
    rhs = stripped
    if "=" in stripped:
        lhs, rhs = stripped.split("=", 1)
        m = _SSA_RE.match(lhs.strip())
        if m is not None:
            def_name = m.group(1)
        rhs = rhs.strip()
    head = rhs.split(None, 1)[0] if rhs else ""
    return def_name, rhs, head


def _checked_names(rhs: str, head: str) -> set[str]:
    """Names on `rhs` whose scope membership `_handle_op_line` checks (mirrors its dispatch)."""
    refs = _ssa_refs(rhs)
    if head == "return":
        tail = rhs[len("return"):].strip()
        r = _ssa_refs(tail)
        return {r[0]} if tail and r else set()
    if head in _ALL_REFS_HEADS:
        return set(refs)
    if head == "memref.store":
        return set(refs) if len(refs) >= 2 else set()
    if head == "memref.dealloc":
        return {refs[0]} if refs else set()
    if head in _LINALG_OPS:
        out: set[str] = set()
        for kw in ("ins", "outs"):
            clause = _extract_clause(rhs, kw)
            if clause is not None:
                out.update(n for n, _ in _parse_linalg_clause(clause))
        return out
    if head in _STABLEHLO_EW_BIN:
        return set(refs[:2]) if _last_colon_type(rhs) and len(refs) >= 2 else set()
    if head in _STABLEHLO_EW_UN:
        return {refs[0]} if _last_colon_type(rhs) and refs else set()
    if head in _STABLEHLO_TYPED_SIG:
        in_ty, _ = _stablehlo_type_sig(rhs)
        return {refs[0]} if in_ty and refs else set()
    if head in _STABLEHLO_BINARY_TYPED:
        in_tys, _ = _stablehlo_binary_sig(rhs)
        return set(refs[:2]) if in_tys and len(refs) >= 2 else set()
    return set()


# ---------------- site finder ----------------

def find_sites(text: str) -> tuple[list[UseSite], list[DefSite]]:
    """All operand uses and definitions of `text` with character offsets and validator scope."""
    uses: list[UseSite] = []
    defs: list[DefSite] = []
    cursor = 0
    per_func_names: list[set[str]] = []
    for fi, (header, body) in enumerate(_extract_functions(text)):
        hs = text.index(header, cursor)
        bs = hs + len(header) + 1
        assert text[bs - 1] == "{" and text[bs:bs + len(body)] == body
        cursor = bs + len(body)
        scope: dict[str, str] = {}
        hdr = _FUNC_HEADER_RE.search(header)
        if hdr:
            p0 = hs + hdr.start("params")
            for pm in _PARAM_RE.finditer(hdr.group("params") + ","):
                scope[pm.group("name")] = _normalize(pm.group("type"))
                defs.append(DefSite(pm.group("name"), p0 + pm.start("name") - 1, fi, -1))
        refs = _SSA_RE.finditer(body)
        f_uses: list[UseSite] = []
        for si, stmt in enumerate(_split_op_lines(body)):
            names = [m.group(1) for m in _SSA_RE.finditer(stmt)]
            occ = []
            for n in names:
                m = next(refs)
                if m.group(1) != n:
                    raise ValueError(f"reference sequence mismatch at {n!r} / {m.group(1)!r}")
                occ.append((n, bs + m.start()))
            def_name, rhs, head = _split_def(stmt)
            checked = _checked_names(rhs, head)
            before = set(scope)
            if def_name is not None:
                defs.append(DefSite(def_name, occ[0][1], fi, si))
                occ = occ[1:]
            for n, cp in occ:
                f_uses.append(UseSite(n, cp, fi, si, in_scope=before, checked=n in checked))
            _handle_op_line(stmt, scope, [])
        # scope after the whole function = every name the validator ever registered
        all_names = set(scope)
        per_func_names.append(all_names)
        for u in f_uses:
            u.out_of_scope = all_names - u.in_scope
        uses.extend(f_uses)
    for u in uses:
        for fj, names in enumerate(per_func_names):
            if fj != u.func_index:
                u.out_of_scope |= names - u.in_scope
    return uses, defs


# ---------------- tokenization with offsets ----------------

def tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids of `text` (no special tokens) and per-token character spans.

    Uses the fast tokenizer's offset mapping when available, else reconstructs spans by
    decoding prefixes (robust to multi-byte merges but slower)."""
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    try:
        if getattr(hf, "is_fast", False):
            enc = hf(text, add_special_tokens=False, return_offsets_mapping=True)
            return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    except (TypeError, ValueError, NotImplementedError):
        pass
    ids = list(hf.encode(text, add_special_tokens=False))
    offsets: list[tuple[int, int]] = []
    prev = 0
    for i in range(1, len(ids) + 1):
        dec = hf.decode(ids[:i])
        # the decoded prefix should be a prefix of `text`; fall back to its length otherwise
        end = len(dec) if text.startswith(dec) else min(len(text), prev + max(1, len(dec) - prev))
        offsets.append((prev, end))
        prev = end
    return ids, offsets


def char_to_token(offsets: list[tuple[int, int]], char_pos: int) -> int:
    """Index of the token whose span contains `char_pos` (spans are half-open; gaps map to the
    next token, which is what we want for whitespace absorbed into the following token)."""
    starts = [s for s, _ in offsets]
    i = bisect.bisect_right(starts, char_pos) - 1
    while i >= 0 and offsets[i][1] <= char_pos:
        i -= 1
    if i < 0 or offsets[i][0] > char_pos:
        i = bisect.bisect_left(starts, char_pos)
    if i >= len(offsets):
        raise ValueError(f"char {char_pos} beyond tokenized text")
    return i


def attach_token_positions(uses: list[UseSite], defs: list[DefSite], offsets: list[tuple[int, int]],
                           base: int = 0) -> None:
    """Fill the token positions of every site; `base` is the number of prompt tokens preceding the
    program so positions index the full (prompt + program) sequence."""
    for d in defs:
        d.pct_pos = base + char_to_token(offsets, d.char_pos)
        d.name_pos = base + char_to_token(offsets, d.char_pos + 1)
    for u in uses:
        u.pct_pos = base + char_to_token(offsets, u.char_pos)
        u.name_pos = base + char_to_token(offsets, u.char_pos + 1)
        u.pred_pos = u.name_pos - 1


def first_name_token(tokenizer, name: str, prefix: str = " %") -> int:
    """Vocabulary id of the first token that carries `name` when `%name` follows a space."""
    text = prefix + name
    ids, offsets = tokenize_with_offsets(tokenizer, text)
    return ids[char_to_token(offsets, len(prefix))]


# ---------------- corruption for paired probes and patching ----------------

def fresh_name(taken: set[str], like: str) -> str:
    """An unused SSA name of the same flavour as `like` (digits stay digits)."""
    if like.isdigit():
        k = 0
        while str(k) in taken:
            k += 1
        return str(k)
    for k in range(100):
        cand = f"v{k}"
        if cand not in taken:
            return cand
    raise RuntimeError("no fresh name")


def corrupt_use(text: str, site: UseSite, uses: list[UseSite], defs: list[DefSite],
                new_name: str | None = None) -> tuple[str, str]:
    """Return (corrupted_text, new_name) where the definition of `site.name` and every use of it
    *before* `site` in the same function are renamed, so the use at `site` now refers to a name that
    is not in scope while the prefix stays a valid program. The text from `site.char_pos` on is
    unchanged, so the model's view up to the use is identical except for the renamed prefix."""
    if new_name is None:
        taken = {d.name for d in defs} | {u.name for u in uses}
        new_name = fresh_name(taken, site.name)
    spans = [d.char_pos for d in defs if d.func_index == site.func_index and d.name == site.name
             and d.char_pos < site.char_pos]
    spans += [u.char_pos for u in uses if u.func_index == site.func_index and u.name == site.name
              and u.char_pos < site.char_pos]
    if not spans:
        raise ValueError(f"%{site.name} has no definition before the use; nothing to rename")
    out = text
    for cp in sorted(spans, reverse=True):
        assert out[cp] == "%" and out[cp + 1:cp + 1 + len(site.name)] == site.name
        out = out[:cp + 1] + new_name + out[cp + 1 + len(site.name):]
    return out, new_name


def match_site(uses: list[UseSite], ref_uses: list[UseSite], ref: UseSite) -> UseSite:
    """The site in `uses` (a renamed copy of the program) corresponding to `ref` in `ref_uses`:
    same function, statement, name and occurrence order."""
    key = (ref.func_index, ref.stmt_index, ref.name)
    k = sum(1 for u in ref_uses[:ref_uses.index(ref)] if (u.func_index, u.stmt_index, u.name) == key)
    cands = [u for u in uses if (u.func_index, u.stmt_index, u.name) == key]
    if k >= len(cands):
        raise ValueError("site not found in corrupted program")
    return cands[k]
