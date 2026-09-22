"""(prompt, gold) generator for the synthetic SSA language with controlled dependency distance.

A program is built around one *probe* value `%x`: it is defined among the first N constants, and
its first use comes exactly D statements later; every other use is kept within the same window
(operands are drawn only from values defined at most D statements back), so the program's maximum
def-use distance equals D. N bounds the generator's working set (the values it may draw operands
from at any point); with `nesting > 0` some of the D intervening statements are wrapped in `{ }`
blocks whose definitions leave the working set when the block closes, so region scoping matters
for C3. Constants are typed (C2) and binary operands always share the result's type and shape (C4).

The prompt is a templated description that fixes N, the value types, the probe's position and
distance, the op multiset, the block count and the return type, which is what the task reward
scores (validity × return-type match × op-multiset Jaccard).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from sci.synth.grammar import FLT_OPS, INT_OPS, MAX_DEPTH, MAX_STMTS

SHAPES = ("[4]", "[8]", "[2x3]", "[3x2]")
PREFIXES = ("Write a prog", "Write an SSA prog", "Emit a prog", "Produce a prog", "Define a prog")
OPS_BY_DOMAIN = {"int": INT_OPS, "flt": FLT_OPS}


@dataclass
class Value:
    name: str
    dom: str
    shape: str
    index: int        # flat statement index of the definition

    @property
    def type(self) -> str:
        return f"{self.dom}{self.shape}"


@dataclass
class Program:
    lines: list[str] = field(default_factory=list)   # rendered body lines (statements and braces)
    n_stmts: int = 0
    n_blocks: int = 0
    ops: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "prog {\n" + "\n".join(self.lines) + "\n}\n"


class _Builder:
    def __init__(self, D: int, N: int, nesting: int, rng: random.Random):
        self.D, self.N, self.nesting, self.rng = D, N, min(nesting, MAX_DEPTH), rng
        self.prog = Program()
        self.counter = 0
        self.scope: list[Value] = []      # names in scope (region-aware)
        self.working: list[Value] = []    # ≤ N candidates for operands
        self.probe: Value | None = None
        self.probe_used = False

    # ---- emission helpers ----
    def fresh(self) -> str:
        self.counter += 1
        return f"%v{self.counter - 1}"

    def emit(self, line: str, depth: int, value: Value | None, op: str) -> None:
        self.prog.lines.append("  " * (depth + 1) + line)
        self.prog.n_stmts += 1
        self.prog.ops.append(op)
        if value is not None:
            self.scope.append(value); self.working.append(value)
            if len(self.working) > self.N:
                keep = next((v for v in self.working if v is not self.probe), self.working[0])
                self.working.remove(keep)

    def literal(self, dom: str) -> str:
        if dom == "int":
            return str(self.rng.randint(-99, 99))
        return f"{self.rng.randint(-9, 9)}.{self.rng.randint(0, 9)}"

    def candidates(self, exclude_probe: bool) -> list[Value]:
        """Working-set values whose use here stays within the distance window."""
        here = self.prog.n_stmts
        out = [v for v in self.working if here - v.index - 1 <= self.D]
        if exclude_probe:
            out = [v for v in out if v is not self.probe]
        return out

    def const(self, depth: int, dom: str | None = None, shape: str | None = None) -> Value:
        dom = dom or self.rng.choice(("int", "flt")); shape = shape or self.rng.choice(SHAPES)
        v = Value(self.fresh(), dom, shape, self.prog.n_stmts)
        self.emit(f"{v.name} = const {self.literal(dom)} : {v.type}", depth, v, "const")
        return v

    def binop(self, depth: int, a: Value, b: Value | None = None) -> Value:
        b = b or a
        op = self.rng.choice(OPS_BY_DOMAIN[a.dom])
        v = Value(self.fresh(), a.dom, a.shape, self.prog.n_stmts)
        x, y = (a, b) if self.rng.random() < 0.5 else (b, a)
        self.emit(f"{v.name} = {op} {x.name} , {y.name} : {v.type}", depth, v, op)
        return v

    def filler(self, depth: int) -> None:
        cands = self.candidates(exclude_probe=not self.probe_used)
        if not cands or self.rng.random() < 0.25:
            self.const(depth); return
        a = self.rng.choice(cands)
        same = [v for v in cands if v.dom == a.dom and v.shape == a.shape]
        self.binop(depth, a, self.rng.choice(same))

    def fillers(self, count: int, depth: int) -> None:
        """`count` statements at `depth`, some wrapped in nested blocks (blocks are not statements)."""
        while count > 0:
            if depth < self.nesting and count >= 2 and self.rng.random() < 0.5:
                k = self.rng.randint(1, min(count - 1, 6))
                saved_scope, saved_work = list(self.scope), list(self.working)
                self.prog.lines.append("  " * (depth + 1) + "{"); self.prog.n_blocks += 1
                self.fillers(k, depth + 1)
                self.prog.lines.append("  " * (depth + 1) + "}")
                self.scope, self.working = saved_scope, saved_work
                count -= k
            else:
                self.filler(depth); count -= 1

    # ---- program assembly ----
    def build(self, n_ops: int | None) -> tuple[Program, dict]:
        D, N = self.D, self.N
        p = self.rng.randint(max(0, N - 1 - D), N - 1)
        prologue: list[Value] = []
        for i in range(N):
            v = self.const(0)
            prologue.append(v)
            if i == p:
                self.probe = v
        n_fill = D - (N - 1 - p)
        self.fillers(n_fill, 0)
        assert self.prog.n_stmts - self.probe.index - 1 == D
        same = [v for v in self.candidates(exclude_probe=True)
                if v.dom == self.probe.dom and v.shape == self.probe.shape]
        partner = self.rng.choice(same) if same else self.probe
        use = self.binop(0, self.probe, partner)
        self.probe_used = True
        use_op = self.prog.ops[-1]
        tail = 0 if n_ops is None else max(0, n_ops - self.prog.n_stmts - 1)
        outer_items = sum(1 for l in self.prog.lines if l.startswith("  ") and not l.startswith("   "))
        self.fillers(min(tail, MAX_STMTS - 1 - outer_items), 0)
        ret = next(v for v in reversed(self.scope))
        self.prog.lines.append(f"  ret {ret.name} : {ret.type}")
        self.prog.n_stmts += 1
        meta = {"probe": self.probe.name, "probe_type": self.probe.type, "probe_pos": p + 1, "use_op": use_op,
                "use": use.name, "ret": ret.name, "ret_type": ret.type, "prologue": [v.type for v in prologue],
                "n_blocks": self.prog.n_blocks, "n_stmts": self.prog.n_stmts}
        return self.prog, meta


def _describe(meta: dict, D: int, N: int, ops: list[str], rng: random.Random) -> str:
    counts: dict[str, int] = {}
    for t in meta["prologue"]:
        counts[t] = counts.get(t, 0) + 1
    types = ", ".join(f"{c} {t}" for t, c in counts.items())
    opc: dict[str, int] = {}
    for o in ops:
        opc[o] = opc.get(o, 0) + 1
    op_str = ", ".join(f"{o} x{c}" for o, c in sorted(opc.items()))
    nb = meta["n_blocks"]
    blocks = f" The program has {nb} nested block{'s' if nb != 1 else ''} whose values are dropped afterwards." if nb else ""
    return (f"{rng.choice(PREFIXES)} that keeps {N} values live at once, starting with {N} constants ({types}). "
            f"The constant at position {meta['probe_pos']} is a {meta['probe_type']}; leave it untouched for "
            f"exactly {D} statements, then combine it with {meta['use_op']}.{blocks} "
            f"Ops used in total: {op_str}. Return the last value, a {meta['ret_type']}.")


def generate(D: int, N: int, nesting: int, seed: int, n_ops: int | None = None) -> tuple[str, str]:
    """(prompt_text, gold_program). Deterministic in (D, N, nesting, seed). `n_ops` is a target total
    statement count (padding after the probe use); at least N + D + 2 statements are emitted."""
    if D < 0 or N < 1:
        raise ValueError("need D >= 0 and N >= 1")
    if N + D + 2 > MAX_STMTS:
        raise ValueError(f"N + D + 2 must be <= {MAX_STMTS} (grammar statement cap)")
    rng = random.Random(f"synth-{D}-{N}-{nesting}-{seed}")
    prog, meta = _Builder(D, N, nesting, rng).build(n_ops)
    return _describe(meta, D, N, prog.ops, rng), prog.text()


def make_pool(n: int, D: int, N: int, nesting: int, seed: int, path: str | Path | None = None,
              n_ops: int | None = None, id_base: int = 0) -> list[dict]:
    """`n` rows {prompt_id, nl, gold, D, N, nesting, seed}; written as JSONL when `path` is given.
    Row i uses generator seed `seed * 1_000_003 + i`; prompt_id = id_base + i."""
    rows = []
    for i in range(n):
        s = seed * 1_000_003 + i
        nl, gold = generate(D, N, nesting, s, n_ops)
        rows.append({"prompt_id": id_base + i, "nl": nl, "gold": gold, "D": D, "N": N, "nesting": nesting, "seed": s})
    if path is not None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows
