"""Parametric training-pool generator: (NL prompt, gold MLIR) pairs for arith+func+memref and
linalg, emitted in the generation grammar's surface format (spaces around `,` and `:`).

Every program is built from a small dataflow model so the NL description and the gold are
derived from the same structure. Golds are checked against the parse grammar, the
generation grammar (llguidance), and `mlir-opt` by `scripts/w02_build_pool.py`; this module
only generates.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

INT_TYPES = ["i8", "i16", "i32", "i64"]
FLOAT_TYPES = ["f16", "f32", "f64"]
FN_NAMES = ["f", "g", "h", "compute", "kernel", "op", "calc", "fn", "eval", "step", "apply", "run",
            "combine", "reduce", "update", "transform", "main_op", "func1", "work", "process"]

INT_BIN = {  # op: (verb phrase, symbol)
    "arith.addi": ("adds", "+"), "arith.subi": ("subtracts the second from the first of", "-"),
    "arith.muli": ("multiplies", "*"), "arith.divsi": ("divides (signed) the first by the second of", "/"),
    "arith.divui": ("divides (unsigned) the first by the second of", "/u"),
    "arith.remsi": ("computes the signed remainder of the first by the second of", "%"),
    "arith.remui": ("computes the unsigned remainder of the first by the second of", "%u"),
    "arith.andi": ("computes the bitwise AND of", "&"), "arith.ori": ("computes the bitwise OR of", "|"),
    "arith.xori": ("computes the bitwise XOR of", "^"), "arith.shli": ("shifts the first left by the second of", "<<"),
    "arith.shrsi": ("shifts the first right (arithmetic) by the second of", ">>"),
    "arith.shrui": ("shifts the first right (logical) by the second of", ">>>"),
    "arith.maxsi": ("returns the signed maximum of", "max"), "arith.minsi": ("returns the signed minimum of", "min"),
    "arith.maxui": ("returns the unsigned maximum of", "umax"), "arith.minui": ("returns the unsigned minimum of", "umin"),
}
FLOAT_BIN = {"arith.addf": ("adds", "+"), "arith.subf": ("subtracts the second from the first of", "-"),
             "arith.mulf": ("multiplies", "*"), "arith.divf": ("divides the first by the second of", "/"),
             "arith.remf": ("computes the floating-point remainder of the first by the second of", "%")}
ICMP = {"eq": "equal to", "ne": "not equal to", "slt": "less than", "sle": "less than or equal to",
        "sgt": "greater than", "sge": "greater than or equal to", "ult": "unsigned less than", "ugt": "unsigned greater than"}
FCMP = {"oeq": "equal to", "one": "not equal to", "olt": "less than", "ole": "less than or equal to",
        "ogt": "greater than", "oge": "greater than or equal to"}
LINALG_BIN = {"linalg.add": "adds", "linalg.sub": "subtracts", "linalg.mul": "multiplies elementwise",
              "linalg.div": "divides elementwise"}
LINALG_UN = {"linalg.exp": "exponential", "linalg.abs": "absolute value"}

PREFIXES = ["Write a function that", "Write an MLIR function that", "Implement a function that",
            "Define a function that", "Create a function that", "Write a function `{name}` that",
            "Implement `{name}`, a function that", "Write MLIR for a function that"]


@dataclass
class Program:
    name: str
    params: list[tuple[str, str]]
    ops: list[str]
    ret: tuple[str, str] | None  # (ssa, type) or None
    family: str
    nl: str
    dialect: str
    meta: dict = field(default_factory=dict)

    def mlir(self) -> str:
        ps = " , ".join(f"{n} : {t}" for n, t in self.params)
        head = f"  func.func @{self.name}({ps})"
        if self.ret:
            head += f" -> {self.ret[1]}"
        body = "\n".join(f"    {o}" for o in self.ops)
        ret = f"    return {self.ret[0]} : {self.ret[1]}" if self.ret else "    return"
        return f"module {{\n{head} {{\n{body}\n{ret}\n  }}\n}}\n"


class Gen:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    # ---------- helpers ----------
    def prefix(self, name: str) -> str:
        return self.rng.choice(PREFIXES).format(name=name)

    def name(self) -> str:
        return self.rng.choice(FN_NAMES) + (str(self.rng.randint(0, 9)) if self.rng.random() < 0.3 else "")

    def pnames(self, k: int) -> list[str]:
        pools = [["%a", "%b", "%c", "%d"], ["%x", "%y", "%z", "%w"], ["%lhs", "%rhs", "%p", "%q"],
                 ["%arg0", "%arg1", "%arg2", "%arg3"], ["%u", "%v", "%s", "%t"]]
        return self.rng.choice(pools)[:k]

    def article(self, t: str) -> str:
        return ("an " if t[0] in "if" else "a ") + t

    def plural(self, t: str) -> str:
        return f"{t} values"

    # ---------- arith families ----------
    def binop(self) -> Program:
        is_int = self.rng.random() < 0.6
        t = self.rng.choice(INT_TYPES if is_int else FLOAT_TYPES)
        op, (verb, _) = self.rng.choice(list((INT_BIN if is_int else FLOAT_BIN).items()))
        a, b = self.pnames(2); nm = self.name()
        nl = f"{self.prefix(nm)} {verb} two {t} values and returns the result."
        return Program(nm, [(a, t), (b, t)], [f"%0 = {op} {a} , {b} : {t}"], ("%0", t), "binop", nl, "arith+func")

    def chain(self) -> Program:
        is_int = self.rng.random() < 0.6
        t = self.rng.choice(INT_TYPES if is_int else FLOAT_TYPES)
        table = INT_BIN if is_int else FLOAT_BIN
        simple = [k for k in table if table[k][1] in "+-*/"]
        k = self.rng.choice([2, 2, 3, 3, 4, 5])
        n_params = min(k + 1, 4)
        names = self.pnames(n_params); nm = self.name()
        avail = list(names); ops = []; expr = {n: n[1:] for n in names}
        for i in range(k):
            op = self.rng.choice(simple); x, y = self.rng.sample(avail, 2)
            r = f"%{i}"; ops.append(f"{r} = {op} {x} , {y} : {t}")
            expr[r] = f"({expr[x]} {table[op][1]} {expr[y]})"; avail.append(r)
        last = f"%{k-1}"
        nl = f"{self.prefix(nm)} takes {n_params} {t} inputs ({', '.join(n[1:] for n in names)}) and returns {expr[last]}."
        return Program(nm, [(n, t) for n in names], ops, (last, t), f"chain{k}", nl, "arith+func")

    def constant(self) -> Program:
        is_int = self.rng.random() < 0.6
        t = self.rng.choice(INT_TYPES if is_int else FLOAT_TYPES)
        hi = {"i8": 127, "i16": 32767, "i32": 100000, "i64": 100000}.get(t, 127)
        v = str(self.rng.randint(-hi - 1, hi)) if is_int else f"{self.rng.uniform(-10, 100):.2f}"
        nm = self.name()
        if self.rng.random() < 0.5:
            nl = f"{self.prefix(nm)} takes no arguments and returns the {t} constant {v}."
            return Program(nm, [], [f"%0 = arith.constant {v} : {t}"], ("%0", t), "const", nl, "arith+func")
        op, (verb, sym) = self.rng.choice([(k, INT_BIN[k]) for k in ("arith.addi", "arith.muli", "arith.subi")]
                                          if is_int else [(k, FLOAT_BIN[k]) for k in ("arith.addf", "arith.mulf", "arith.subf")])
        a = self.pnames(1)[0]
        nl = f"{self.prefix(nm)} takes {self.article(t)} input {a[1:]} and returns {a[1:]} {sym} {v}."
        return Program(nm, [(a, t)], [f"%c = arith.constant {v} : {t}", f"%0 = {op} {a} , %c : {t}"], ("%0", t),
                       "const_op", nl, "arith+func")

    def compare(self) -> Program:
        is_int = self.rng.random() < 0.7
        t = self.rng.choice(INT_TYPES if is_int else FLOAT_TYPES)
        pred, words = self.rng.choice(list((ICMP if is_int else FCMP).items()))
        cmp = "arith.cmpi" if is_int else "arith.cmpf"
        a, b = self.pnames(2); nm = self.name()
        if self.rng.random() < 0.5:
            nl = f"{self.prefix(nm)} returns an i1 that is true when {a[1:]} is {words} {b[1:]}, for two {t} inputs."
            return Program(nm, [(a, t), (b, t)], [f"%0 = {cmp} {pred} , {a} , {b} : {t}"], ("%0", "i1"), "cmp", nl, "arith+func")
        nl = f"{self.prefix(nm)} returns {a[1:]} if {a[1:]} is {words} {b[1:]}, and {b[1:]} otherwise, for two {t} inputs (compare then select)."
        return Program(nm, [(a, t), (b, t)], [f"%0 = {cmp} {pred} , {a} , {b} : {t}", f"%1 = arith.select %0 , {a} , {b} : {t}"],
                       ("%1", t), "cmp_select", nl, "arith+func")

    def cast(self) -> Program:
        kind = self.rng.choice(["extsi", "extui", "trunci", "sitofp", "uitofp", "fptosi", "fptoui", "extf", "truncf", "index_cast", "index_cast_to"])
        a = self.pnames(1)[0]; nm = self.name()
        if kind in ("extsi", "extui"):
            i = self.rng.randrange(0, 3); src, dst = INT_TYPES[i], self.rng.choice(INT_TYPES[i + 1:])
            how = "sign-extends" if kind == "extsi" else "zero-extends"
            nl = f"{self.prefix(nm)} {how} {self.article(src)} value to {dst}."
        elif kind == "trunci":
            i = self.rng.randrange(1, 4); src, dst = INT_TYPES[i], self.rng.choice(INT_TYPES[:i])
            nl = f"{self.prefix(nm)} truncates {self.article(src)} value to {dst}."
        elif kind in ("sitofp", "uitofp"):
            src, dst = self.rng.choice(INT_TYPES), self.rng.choice(FLOAT_TYPES)
            nl = f"{self.prefix(nm)} converts {'a signed' if kind == 'sitofp' else 'an unsigned'} {src} integer to {dst}."
        elif kind in ("fptosi", "fptoui"):
            src, dst = self.rng.choice(FLOAT_TYPES), self.rng.choice(INT_TYPES)
            nl = f"{self.prefix(nm)} converts {self.article(src)} value to {'a signed' if kind == 'fptosi' else 'an unsigned'} {dst} integer."
        elif kind == "extf":
            i = self.rng.randrange(0, 2); src, dst = FLOAT_TYPES[i], self.rng.choice(FLOAT_TYPES[i + 1:])
            nl = f"{self.prefix(nm)} extends {self.article(src)} value to {dst}."
        elif kind == "truncf":
            i = self.rng.randrange(1, 3); src, dst = FLOAT_TYPES[i], self.rng.choice(FLOAT_TYPES[:i])
            nl = f"{self.prefix(nm)} truncates {self.article(src)} value to {dst}."
        elif kind == "index_cast":
            src, dst = "index", self.rng.choice(INT_TYPES); kind = "index_cast"
            nl = f"{self.prefix(nm)} converts an index value to {dst}."
        else:
            src, dst = self.rng.choice(INT_TYPES), "index"; kind = "index_cast"
            nl = f"{self.prefix(nm)} converts {self.article(src)} value to an index."
        return Program(nm, [(a, src)], [f"%0 = arith.{kind} {a} : {src} to {dst}"], ("%0", dst), "cast", nl, "arith+func")

    def minmax3(self) -> Program:
        t = self.rng.choice(INT_TYPES); op = self.rng.choice(["arith.minsi", "arith.maxsi"])
        a, b, c = self.pnames(3); nm = self.name(); word = "minimum" if "min" in op else "maximum"
        nl = f"{self.prefix(nm)} returns the signed {word} of three {t} values using {op}."
        return Program(nm, [(a, t), (b, t), (c, t)], [f"%0 = {op} {a} , {b} : {t}", f"%1 = {op} %0 , {c} : {t}"], ("%1", t),
                       "minmax3", nl, "arith+func")

    def shape(self, rank: int, static_p: float = 0.5) -> str:
        dims = [str(self.rng.choice([2, 4, 8, 16, 32, 64])) if self.rng.random() < static_p else "?" for _ in range(rank)]
        return "x".join(dims)

    def memref_load(self) -> Program:
        t = self.rng.choice(INT_TYPES + FLOAT_TYPES); rank = self.rng.choice([1, 1, 2]); sh = self.shape(rank)
        mt = f"memref<{sh}x{t}>"; nm = self.name()
        idx = ["%i", "%j"][:rank]
        params = [("%m", mt)] + [(i, "index") for i in idx]
        variant = self.rng.choice(["load", "load", "load_op", "store", "load_store", "dim"])
        if variant == "load":
            nl = f"{self.prefix(nm)} loads the element at index {', '.join(i[1:] for i in idx)} from a {mt} and returns it."
            return Program(nm, params, [f"%0 = memref.load %m[{' , '.join(idx)}] : {mt}"], ("%0", t), "load", nl, "arith+func")
        if variant == "load_op":
            is_int = t.startswith("i"); tbl = INT_BIN if is_int else FLOAT_BIN
            op = self.rng.choice(["arith.addi", "arith.muli"] if is_int else ["arith.addf", "arith.mulf"])
            nl = f"{self.prefix(nm)} loads the element at index {', '.join(i[1:] for i in idx)} from a {mt}, {tbl[op][0]} it and the {t} argument v, and returns the result."
            return Program(nm, params + [("%v", t)], [f"%0 = memref.load %m[{' , '.join(idx)}] : {mt}", f"%1 = {op} %0 , %v : {t}"],
                           ("%1", t), "load_op", nl, "arith+func")
        if variant == "store":
            nl = f"{self.prefix(nm)} stores the {t} value v at index {', '.join(i[1:] for i in idx)} of a {mt}."
            return Program(nm, params + [("%v", t)], [f"memref.store %v , %m[{' , '.join(idx)}] : {mt}"], None, "store", nl, "arith+func")
        if variant == "load_store":
            is_int = t.startswith("i")
            op = self.rng.choice(["arith.addi", "arith.muli", "arith.subi"] if is_int else ["arith.addf", "arith.mulf", "arith.subf"])
            tbl = INT_BIN if is_int else FLOAT_BIN
            nl = f"{self.prefix(nm)} loads the element at index {', '.join(i[1:] for i in idx)} of a {mt}, {tbl[op][0]} it and the {t} argument v, and stores the result back at the same index."
            return Program(nm, params + [("%v", t)], [f"%0 = memref.load %m[{' , '.join(idx)}] : {mt}", f"%1 = {op} %0 , %v : {t}",
                                                      f"memref.store %1 , %m[{' , '.join(idx)}] : {mt}"], None, "load_store", nl, "arith+func")
        d = self.rng.randrange(rank)
        nl = f"{self.prefix(nm)} returns the size of dimension {d} of a {mt} as an index."
        return Program(nm, [("%m", mt)], [f"%c = arith.constant {d} : index", f"%0 = memref.dim %m , %c : {mt}"], ("%0", "index"), "dim", nl, "arith+func")

    # ---------- linalg families ----------
    def linalg(self) -> Program:
        fam = self.rng.choice(["matmul", "matmul", "matvec", "fill", "copy", "transpose", "broadcast", "ewbin", "ewbin", "ewun"])
        t = self.rng.choice(["f32", "f32", "f64", "f16"]); nm = self.name(); static = self.rng.random() < 0.4
        if fam == "matmul":
            m, k, n = [str(self.rng.choice([4, 8, 16, 32])) if static else "?" for _ in range(3)]
            A, B, C = f"memref<{m}x{k}x{t}>", f"memref<{k}x{n}x{t}>", f"memref<{m}x{n}x{t}>"
            nl = f"{self.prefix(nm)} performs a matrix multiplication of a {A} and a {B}, accumulating into a pre-allocated {C} output."
            return Program(nm, [("%A", A), ("%B", B), ("%C", C)], [f"linalg.matmul ins(%A , %B : {A}, {B}) outs(%C : {C})"], None, "linalg_matmul", nl, "linalg")
        if fam == "matvec":
            m, k = [str(self.rng.choice([4, 8, 16, 32])) if static else "?" for _ in range(2)]
            A, x, y = f"memref<{m}x{k}x{t}>", f"memref<{k}x{t}>", f"memref<{m}x{t}>"
            nl = f"{self.prefix(nm)} performs a matrix-vector product of a {A} with a {x} into a pre-allocated {y}."
            return Program(nm, [("%A", A), ("%x", x), ("%y", y)], [f"linalg.matvec ins(%A , %x : {A}, {x}) outs(%y : {y})"], None, "linalg_matvec", nl, "linalg")
        rank = self.rng.choice([1, 2]); sh = self.shape(rank, 0.5 if static else 0.0); M = f"memref<{sh}x{t}>"
        if fam == "fill":
            nl = f"{self.prefix(nm)} fills a {M} with the {t} scalar value v."
            return Program(nm, [("%v", t), ("%m", M)], [f"linalg.fill ins(%v : {t}) outs(%m : {M})"], None, "linalg_fill", nl, "linalg")
        if fam == "copy":
            nl = f"{self.prefix(nm)} copies the contents of a {M} into another {M}."
            return Program(nm, [("%src", M), ("%dst", M)], [f"linalg.copy ins(%src : {M}) outs(%dst : {M})"], None, "linalg_copy", nl, "linalg")
        if fam == "transpose":
            a, b = (self.shape(1, 0.5 if static else 0.0), self.shape(1, 0.5 if static else 0.0))
            S, D = f"memref<{a}x{b}x{t}>", f"memref<{b}x{a}x{t}>"
            nl = f"{self.prefix(nm)} transposes a {S} into a pre-allocated {D} (permutation [1, 0])."
            return Program(nm, [("%in", S), ("%out", D)], [f"linalg.transpose ins(%in : {S}) outs(%out : {D}) permutation = [1 , 0]"], None, "linalg_transpose", nl, "linalg")
        if fam == "broadcast":
            a = self.shape(1, 0.5 if static else 0.0); b = self.shape(1, 0.5 if static else 0.0); d = self.rng.choice([0, 1])
            S = f"memref<{a}x{t}>"; D = f"memref<{b}x{a}x{t}>" if d == 0 else f"memref<{a}x{b}x{t}>"
            nl = f"{self.prefix(nm)} broadcasts a {S} into a pre-allocated {D} along dimension {d}."
            return Program(nm, [("%in", S), ("%out", D)], [f"linalg.broadcast ins(%in : {S}) outs(%out : {D}) dimensions = [{d}]"], None, "linalg_broadcast", nl, "linalg")
        if fam == "ewbin":
            op, verb = self.rng.choice(list(LINALG_BIN.items()))
            nl = f"{self.prefix(nm)} {verb} two {M} inputs elementwise and writes the result into a pre-allocated {M} output."
            return Program(nm, [("%x", M), ("%y", M), ("%out", M)], [f"{op} ins(%x , %y : {M}, {M}) outs(%out : {M})"], None, "linalg_ewbin", nl, "linalg")
        op, word = self.rng.choice(list(LINALG_UN.items()))
        nl = f"{self.prefix(nm)} computes the elementwise {word} of a {M} into a pre-allocated {M} output."
        return Program(nm, [("%x", M), ("%out", M)], [f"{op} ins(%x : {M}) outs(%out : {M})"], None, "linalg_ewun", nl, "linalg")

    ARITH_FAMILIES = ("binop", "binop", "chain", "chain", "chain", "constant", "compare", "cast", "minmax3", "memref_load", "memref_load")

    def sample(self, dialect: str) -> Program:
        if dialect == "linalg":
            return self.linalg()
        return getattr(self, self.rng.choice(self.ARITH_FAMILIES))()


def generate(n_arith: int, n_linalg: int, seed: int = 0) -> list[dict]:
    g = Gen(seed); out = []; seen = set()
    for dialect, n in (("arith+func", n_arith), ("linalg", n_linalg)):
        tries = 0
        while sum(1 for r in out if r["dialect"] == dialect) < n and tries < 50 * n:
            tries += 1
            p = g.sample(dialect); m = p.mlir()
            key = (p.nl, m)
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": f"tpl_{seed}_{len(out):06d}", "dialect": dialect, "family": p.family, "nl": p.nl, "mlir": m})
    return out
