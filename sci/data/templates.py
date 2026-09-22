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
    """`families="v1"` reproduces the v1 pool exactly (same rng call sequence). `families="v1c"` mixes in
    the coverage families below with probability P_NEW_* per draw."""

    FAMILY_SETS = ("v1", "v1c")

    def __init__(self, seed: int = 0, families: str = "v1"):
        if families not in self.FAMILY_SETS:
            raise ValueError(f"unknown family set {families!r}; expected one of {self.FAMILY_SETS}")
        self.rng = random.Random(seed)
        self.families = families

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

    # ---------- v1c coverage families ----------
    # Concepts the students invent op mnemonics for on the released benchmarks :
    # alloc/dealloc, negation and abs, clamp, compare-with-constant, remainder/parity, shifts and masks
    # by a constant, cast chains, bitcast, float min/max via cmpf+select, i1-conditioned select,
    # element copy/swap between memrefs, dim+index_cast, square/polynomial, and the elementwise linalg
    # ops whose v1 phrasing the disjointness filter removed. Every gold stays within the generation
    # grammar (static `memref.alloc()`, at most 5 ops before `return`).

    STATIC_DIMS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 100, 128, 256]

    def static_shape(self, rank: int) -> list[int]:
        return [self.rng.choice(self.STATIC_DIMS) for _ in range(rank)]

    def zero(self, t: str) -> str:
        return "0" if t.startswith("i") else "0.0"

    def alloc_dealloc(self) -> Program:
        t = self.rng.choice(INT_TYPES + FLOAT_TYPES); rank = self.rng.choice([1, 1, 2]); dims = self.static_shape(rank)
        mt = f"memref<{'x'.join(str(d) for d in dims)}x{t}>"; nm = self.name()
        variant = self.rng.choice(["plain", "store", "load", "store_load", "store_load"])
        alloc, dealloc = f"%m = memref.alloc() : {mt}", f"memref.dealloc %m : {mt}"
        if variant == "plain":
            nl = f"{self.prefix(nm)} allocates a {mt} buffer with memref.alloc and immediately frees it with memref.dealloc."
            return Program(nm, [], [alloc, dealloc], None, "alloc_dealloc", nl, "arith+func")
        if rank == 1:
            k = self.rng.randrange(dims[0]); idx = "%i"; pre = [f"%i = arith.constant {k} : index"]; params = []
            where = f"index {k}"
        else:
            idx = "%i , %j"; pre = []; params = [("%i", "index"), ("%j", "index")]
            where = "position (i, j)"
        if variant == "store":
            nl = f"{self.prefix(nm)} allocates a {mt}, stores the {t} argument v at {where}, and deallocates the buffer."
            return Program(nm, params + [("%v", t)], [alloc] + pre + [f"memref.store %v , %m[{idx}] : {mt}", dealloc], None, "alloc_dealloc", nl, "arith+func")
        if variant == "load":
            nl = f"{self.prefix(nm)} allocates a {mt}, loads the element at {where}, frees the buffer with memref.dealloc, and returns the loaded {t}."
            return Program(nm, params, [alloc] + pre + [f"%0 = memref.load %m[{idx}] : {mt}", dealloc], ("%0", t), "alloc_dealloc", nl, "arith+func")
        nl = f"{self.prefix(nm)} allocates a {mt}, stores the {t} argument v at {where}, reads it back, deallocates the buffer, and returns the value."
        return Program(nm, params + [("%v", t)], [alloc] + pre + [f"memref.store %v , %m[{idx}] : {mt}", f"%0 = memref.load %m[{idx}] : {mt}", dealloc],
                       ("%0", t), "alloc_dealloc", nl, "arith+func")

    def negate(self) -> Program:
        is_int = self.rng.random() < 0.5
        t = self.rng.choice(INT_TYPES if is_int else FLOAT_TYPES); a = self.pnames(1)[0]; nm = self.name()
        sub = "arith.subi" if is_int else "arith.subf"; z = self.zero(t)
        if self.rng.random() < 0.6:
            nl = f"{self.prefix(nm)} returns the negation of the {t} input {a[1:]}, computed as 0 - {a[1:]} with {sub}."
            return Program(nm, [(a, t)], [f"%c = arith.constant {z} : {t}", f"%0 = {sub} %c , {a} : {t}"], ("%0", t), "negate", nl, "arith+func")
        cmp, pred = ("arith.cmpi", "slt") if is_int else ("arith.cmpf", "olt")
        nl = f"{self.prefix(nm)} computes the magnitude (absolute value) of the {t} input {a[1:]}: when {a[1:]} is below zero it returns 0 - {a[1:]}, otherwise {a[1:]} itself."
        return Program(nm, [(a, t)], [f"%c = arith.constant {z} : {t}", f"%0 = {sub} %c , {a} : {t}", f"%1 = {cmp} {pred} , {a} , %c : {t}",
                                      f"%2 = arith.select %1 , %0 , {a} : {t}"], ("%2", t), "negate", nl, "arith+func")

    def clamp(self) -> Program:
        t = self.rng.choice(INT_TYPES); nm = self.name(); signed = self.rng.random() < 0.8
        mx, mn = ("arith.maxsi", "arith.minsi") if signed else ("arith.maxui", "arith.minui")
        word = "signed" if signed else "unsigned"
        x, lo, hi = self.rng.choice([("%x", "%lo", "%hi"), ("%v", "%min", "%max"), ("%a", "%b", "%c")])
        if self.rng.random() < 0.5:
            nl = f"{self.prefix(nm)} takes three {t} values {x[1:]}, {lo[1:]} and {hi[1:]} and returns {x[1:]} clamped to the interval [{lo[1:]}, {hi[1:]}] ({word} maximum with {lo[1:]}, then {word} minimum with {hi[1:]})."
            return Program(nm, [(x, t), (lo, t), (hi, t)], [f"%0 = {mx} {x} , {lo} : {t}", f"%1 = {mn} %0 , {hi} : {t}"], ("%1", t), "clamp", nl, "arith+func")
        top = {"i8": 127, "i16": 1000, "i32": 1000, "i64": 1000}[t]
        L = self.rng.randint(0 if not signed else -top // 2, top // 2); H = self.rng.randint(L + 1, top)
        nl = f"{self.prefix(nm)} clamps the {t} input {x[1:]} to the interval [{L}, {H}] using a {word} maximum with {L} followed by a {word} minimum with {H}."
        return Program(nm, [(x, t)], [f"%clo = arith.constant {L} : {t}", f"%chi = arith.constant {H} : {t}", f"%0 = {mx} {x} , %clo : {t}", f"%1 = {mn} %0 , %chi : {t}"],
                       ("%1", t), "clamp", nl, "arith+func")

    def const_lit(self, t: str) -> str:
        if t.startswith("f"):
            return f"{self.rng.uniform(-10, 100):.1f}"
        hi = {"i8": 127, "i16": 1000, "i32": 1000, "i64": 1000}[t]
        return str(self.rng.randint(-hi // 4, hi))

    def cmp_const(self) -> Program:
        is_int = self.rng.random() < 0.7
        t = self.rng.choice(INT_TYPES if is_int else FLOAT_TYPES)
        pred, words = self.rng.choice(list((ICMP if is_int else FCMP).items()))
        cmp = "arith.cmpi" if is_int else "arith.cmpf"; a = self.pnames(1)[0]; nm = self.name(); v = self.const_lit(t)
        variant = self.rng.choice(["cmp", "cmp", "select_self", "select_consts"])
        head = [f"%c = arith.constant {v} : {t}", f"%0 = {cmp} {pred} , {a} , %c : {t}"]
        if variant == "cmp":
            nl = f"{self.prefix(nm)} returns an i1 that is true when the {t} input {a[1:]} is {words} the constant {v}."
            return Program(nm, [(a, t)], head, ("%0", "i1"), "cmp_const", nl, "arith+func")
        if variant == "select_self":
            nl = f"{self.prefix(nm)} returns the {t} input {a[1:]} when {a[1:]} is {words} {v}, and the constant {v} otherwise."
            return Program(nm, [(a, t)], head + [f"%1 = arith.select %0 , {a} , %c : {t}"], ("%1", t), "cmp_const", nl, "arith+func")
        p, q = self.const_lit(t), self.const_lit(t)
        nl = f"{self.prefix(nm)} returns the {t} constant {p} if the input {a[1:]} is {words} {v}, and the {t} constant {q} otherwise."
        return Program(nm, [(a, t)], head + [f"%p = arith.constant {p} : {t}", f"%q = arith.constant {q} : {t}", f"%1 = arith.select %0 , %p , %q : {t}"],
                       ("%1", t), "cmp_const", nl, "arith+func")

    CONST_OPS = {  # op: (phrase before the constant, value kind)
        "arith.shli": ("shifts the {t} input {a} left by", "shift"), "arith.shrsi": ("arithmetically shifts the {t} input {a} right by", "shift"),
        "arith.shrui": ("logically shifts the {t} input {a} right by", "shift"),
        "arith.andi": ("computes the bitwise AND of the {t} input {a} and the mask", "mask"),
        "arith.ori": ("computes the bitwise OR of the {t} input {a} and", "mask"), "arith.xori": ("computes the bitwise XOR of the {t} input {a} and", "mask"),
        "arith.remsi": ("returns the signed remainder of the {t} input {a} divided by", "div"),
        "arith.remui": ("returns the unsigned remainder of the {t} input {a} divided by", "div"),
        "arith.divsi": ("divides (signed) the {t} input {a} by", "div"), "arith.divui": ("divides (unsigned) the {t} input {a} by", "div"),
        "arith.maxsi": ("returns the signed maximum of the {t} input {a} and", "any"), "arith.minsi": ("returns the signed minimum of the {t} input {a} and", "any"),
    }

    def const_op2(self) -> Program:
        t = self.rng.choice(INT_TYPES); a = self.pnames(1)[0]; nm = self.name(); bits = int(t[1:])
        if self.rng.random() < 0.25:
            k = self.rng.choice([2, 2, 3, 4, 5, 7, 8, 10])
            what = "is even (divisible by 2)" if k == 2 else f"is divisible by {k}"
            nl = f"{self.prefix(nm)} returns an i1 that is true when the {t} input {a[1:]} {what}: it computes the signed remainder modulo {k} and compares it with zero."
            return Program(nm, [(a, t)], [f"%k = arith.constant {k} : {t}", f"%r = arith.remsi {a} , %k : {t}", f"%z = arith.constant 0 : {t}",
                                          f"%0 = arith.cmpi eq , %r , %z : {t}"], ("%0", "i1"), "const_op2", nl, "arith+func")
        op, (phrase, kind) = self.rng.choice(list(self.CONST_OPS.items()))
        if kind == "shift":
            v = self.rng.randint(1, bits - 1)
        elif kind == "mask":
            v = self.rng.choice([m for m in (1, 3, 7, 15, 31, 63, 127, 255, 4095, 65535) if m < 2 ** (bits - 1)])
        elif kind == "div":
            v = self.rng.randint(2, 100 if bits > 8 else 20)
        else:
            v = self.rng.randint(-100 if bits > 8 else -50, 100 if bits > 8 else 50)
        nl = f"{self.prefix(nm)} {phrase.format(t=t, a=a[1:])} {v} and returns the {t} result."
        return Program(nm, [(a, t)], [f"%c = arith.constant {v} : {t}", f"%0 = {op} {a} , %c : {t}"], ("%0", t), "const_op2", nl, "arith+func")

    def cast_chain(self) -> Program:
        a = self.pnames(1)[0]; nm = self.name()
        variant = self.rng.choice(["int2", "float2", "int2fp_cmp", "cast_op", "index_load", "index_store", "fp2int_op"])
        if variant == "int2":
            kind = self.rng.choice(["extsi", "extui"]); how = "sign-extends" if kind == "extsi" else "zero-extends"
            i = self.rng.randrange(0, 2); j = self.rng.randrange(i + 1, 3); src, mid, dst = INT_TYPES[i], INT_TYPES[j], self.rng.choice(INT_TYPES[j + 1:])
            nl = f"{self.prefix(nm)} {how} the {src} input {a[1:]} to {mid} and then {how} that to {dst}, returning the {dst} result."
            return Program(nm, [(a, src)], [f"%0 = arith.{kind} {a} : {src} to {mid}", f"%1 = arith.{kind} %0 : {mid} to {dst}"], ("%1", dst), "cast_chain", nl, "arith+func")
        if variant == "float2":
            if self.rng.random() < 0.5:
                kind, src, mid, dst, how = "extf", "f16", "f32", "f64", "widens"
            else:
                kind, src, mid, dst, how = "truncf", "f64", "f32", "f16", "narrows"
            nl = f"{self.prefix(nm)} {how} the {src} input {a[1:]} to {mid} and then to {dst} with two arith.{kind} ops, returning the {dst} value."
            return Program(nm, [(a, src)], [f"%0 = arith.{kind} {a} : {src} to {mid}", f"%1 = arith.{kind} %0 : {mid} to {dst}"], ("%1", dst), "cast_chain", nl, "arith+func")
        if variant == "int2fp_cmp":
            src, dst = self.rng.choice(INT_TYPES), self.rng.choice(FLOAT_TYPES); pred, words = self.rng.choice(list(FCMP.items())); v = self.const_lit(dst)
            nl = f"{self.prefix(nm)} converts the signed {src} input {a[1:]} to {dst} and returns an i1 that is true when the converted value is {words} {v}."
            return Program(nm, [(a, src)], [f"%0 = arith.sitofp {a} : {src} to {dst}", f"%c = arith.constant {v} : {dst}", f"%1 = arith.cmpf {pred} , %0 , %c : {dst}"],
                           ("%1", "i1"), "cast_chain", nl, "arith+func")
        if variant == "cast_op":
            i = self.rng.randrange(0, 3); src, dst = INT_TYPES[i], self.rng.choice(INT_TYPES[i + 1:]); b = "%b" if a != "%b" else "%y"
            op = self.rng.choice(["arith.addi", "arith.muli", "arith.subi"])
            nl = f"{self.prefix(nm)} sign-extends the {src} input {a[1:]} to {dst}, then {INT_BIN[op][0]} it and the {dst} input {b[1:]}, returning the {dst} result."
            return Program(nm, [(a, src), (b, dst)], [f"%0 = arith.extsi {a} : {src} to {dst}", f"%1 = {op} %0 , {b} : {dst}"], ("%1", dst), "cast_chain", nl, "arith+func")
        if variant in ("index_load", "index_store"):
            t = self.rng.choice(INT_TYPES + FLOAT_TYPES); it = self.rng.choice(["i32", "i64"]); mt = f"memref<?x{t}>"
            cast = f"%i = arith.index_cast %n : {it} to index"
            if variant == "index_load":
                nl = f"{self.prefix(nm)} takes a {mt} and an {it} value n, casts n to index with arith.index_cast, and returns the element of the memref at that position."
                return Program(nm, [("%m", mt), ("%n", it)], [cast, f"%0 = memref.load %m[%i] : {mt}"], ("%0", t), "cast_chain", nl, "arith+func")
            nl = f"{self.prefix(nm)} takes a {mt}, an {it} value n and a {t} value v, casts n to index, and stores v into the memref at that position."
            return Program(nm, [("%m", mt), ("%n", it), ("%v", t)], [cast, f"memref.store %v , %m[%i] : {mt}"], None, "cast_chain", nl, "arith+func")
        src, dst = self.rng.choice(FLOAT_TYPES), self.rng.choice(INT_TYPES); v = self.const_lit(dst); op = self.rng.choice(["arith.addi", "arith.muli"])
        nl = f"{self.prefix(nm)} converts the {src} input {a[1:]} to a signed {dst} integer with arith.fptosi, then {INT_BIN[op][0]} it and the constant {v}, returning the {dst} result."
        return Program(nm, [(a, src)], [f"%0 = arith.fptosi {a} : {src} to {dst}", f"%c = arith.constant {v} : {dst}", f"%1 = {op} %0 , %c : {dst}"], ("%1", dst), "cast_chain", nl, "arith+func")

    def bitcast(self) -> Program:
        a = self.pnames(1)[0]; nm = self.name()
        src, dst = self.rng.choice([("i16", "f16"), ("i32", "f32"), ("i64", "f64"), ("f16", "i16"), ("f32", "i32"), ("f64", "i64")])
        nl = f"{self.prefix(nm)} reinterprets the bits of the {src} input {a[1:]} as {dst} with arith.bitcast (no numeric conversion) and returns the {dst} value."
        return Program(nm, [(a, src)], [f"%0 = arith.bitcast {a} : {src} to {dst}"], ("%0", dst), "bitcast", nl, "arith+func")

    def fminmax(self) -> Program:
        t = self.rng.choice(FLOAT_TYPES); nm = self.name(); is_max = self.rng.random() < 0.5
        pred, big, sup = ("ogt", "larger", "largest") if is_max else ("olt", "smaller", "smallest")
        variant = self.rng.choice(["two", "two", "three", "zero"])
        if variant == "two":
            a, b = self.pnames(2)
            nl = f"{self.prefix(nm)} picks the {big} of the two {t} values {a[1:]} and {b[1:]} using an ordered arith.cmpf followed by arith.select."
            return Program(nm, [(a, t), (b, t)], [f"%0 = arith.cmpf {pred} , {a} , {b} : {t}", f"%1 = arith.select %0 , {a} , {b} : {t}"], ("%1", t), "fminmax", nl, "arith+func")
        if variant == "three":
            a, b, c = self.pnames(3)
            nl = f"{self.prefix(nm)} returns the {sup} of three {t} values {a[1:]}, {b[1:]} and {c[1:]} by comparing and selecting pairwise (cmpf then select, twice)."
            return Program(nm, [(a, t), (b, t), (c, t)], [f"%0 = arith.cmpf {pred} , {a} , {b} : {t}", f"%1 = arith.select %0 , {a} , {b} : {t}",
                                                          f"%2 = arith.cmpf {pred} , %1 , {c} : {t}", f"%3 = arith.select %2 , %1 , {c} : {t}"], ("%3", t), "fminmax", nl, "arith+func")
        a = self.pnames(1)[0]
        nl = f"{self.prefix(nm)} returns the {big} of the {t} input {a[1:]} and 0.0 (a floating-point {'max' if is_max else 'min'} with zero via cmpf and select)."
        return Program(nm, [(a, t)], [f"%c = arith.constant 0.0 : {t}", f"%0 = arith.cmpf {pred} , {a} , %c : {t}", f"%1 = arith.select %0 , {a} , %c : {t}"],
                       ("%1", t), "fminmax", nl, "arith+func")

    def cond_select(self) -> Program:
        t = self.rng.choice(INT_TYPES + FLOAT_TYPES); a, b = self.pnames(2); nm = self.name()
        if self.rng.random() < 0.6:
            nl = f"{self.prefix(nm)} given an i1 flag cond and two {t} values {a[1:]} and {b[1:]}, returns {a[1:]} when cond is true and {b[1:]} otherwise."
            return Program(nm, [("%cond", "i1"), (a, t), (b, t)], [f"%0 = arith.select %cond , {a} , {b} : {t}"], ("%0", t), "cond_select", nl, "arith+func")
        is_int = t.startswith("i"); op = self.rng.choice(["arith.addi", "arith.muli"] if is_int else ["arith.addf", "arith.mulf"])
        tbl = INT_BIN if is_int else FLOAT_BIN
        nl = f"{self.prefix(nm)} selects {a[1:]} when the i1 flag cond is true and {b[1:]} otherwise, then {tbl[op][0]} the selected value and the {t} input k."
        return Program(nm, [("%cond", "i1"), (a, t), (b, t), ("%k", t)], [f"%0 = arith.select %cond , {a} , {b} : {t}", f"%1 = {op} %0 , %k : {t}"],
                       ("%1", t), "cond_select", nl, "arith+func")

    def mem_elem(self) -> Program:
        t = self.rng.choice(INT_TYPES + FLOAT_TYPES); nm = self.name()
        sh = str(self.rng.choice(self.STATIC_DIMS)) if self.rng.random() < 0.5 else "?"; mt = f"memref<{sh}x{t}>"
        variant = self.rng.choice(["copy", "copy_same", "copy2", "swap", "store2"])
        idx = [("%i", "index"), ("%j", "index")]
        if variant == "copy":
            nl = f"{self.prefix(nm)} reads position i of the {mt} src and writes that element to position j of the {mt} dst."
            return Program(nm, [("%src", mt), ("%dst", mt)] + idx, [f"%0 = memref.load %src[%i] : {mt}", f"memref.store %0 , %dst[%j] : {mt}"], None, "mem_elem", nl, "arith+func")
        if variant == "copy_same":
            nl = f"{self.prefix(nm)} duplicates the entry at index i of the {mt} src into index i of the {mt} dst."
            return Program(nm, [("%src", mt), ("%dst", mt), idx[0]], [f"%0 = memref.load %src[%i] : {mt}", f"memref.store %0 , %dst[%i] : {mt}"], None, "mem_elem", nl, "arith+func")
        if variant == "copy2":
            nl = f"{self.prefix(nm)} transfers the entries at positions i and j of the {mt} src to the same positions of the {mt} dst."
            return Program(nm, [("%src", mt), ("%dst", mt)] + idx, [f"%0 = memref.load %src[%i] : {mt}", f"memref.store %0 , %dst[%i] : {mt}",
                                                                   f"%1 = memref.load %src[%j] : {mt}", f"memref.store %1 , %dst[%j] : {mt}"], None, "mem_elem", nl, "arith+func")
        if variant == "swap":
            nl = f"{self.prefix(nm)} exchanges the elements at positions i and j of the {mt} m in place (two loads, then two stores)."
            return Program(nm, [("%m", mt)] + idx, [f"%0 = memref.load %m[%i] : {mt}", f"%1 = memref.load %m[%j] : {mt}",
                                                    f"memref.store %1 , %m[%i] : {mt}", f"memref.store %0 , %m[%j] : {mt}"], None, "mem_elem", nl, "arith+func")
        nl = f"{self.prefix(nm)} stores the {t} value v1 at position i and the {t} value v2 at position j of the {mt} m."
        return Program(nm, [("%m", mt)] + idx + [("%v1", t), ("%v2", t)], [f"memref.store %v1 , %m[%i] : {mt}", f"memref.store %v2 , %m[%j] : {mt}"], None, "mem_elem", nl, "arith+func")

    def dim_cast(self) -> Program:
        t = self.rng.choice(INT_TYPES + FLOAT_TYPES); rank = self.rng.choice([1, 2, 2, 3]); nm = self.name()
        sh = self.shape(rank, 0.3); mt = f"memref<{sh}x{t}>"
        variant = self.rng.choice(["dim_param", "dim_cast", "dim_cast", "dim_mul" if rank >= 2 else "dim_cast"])
        if variant == "dim_param":
            nl = f"{self.prefix(nm)} takes a {mt} and an index d and returns the size of the memref along dimension d with memref.dim."
            return Program(nm, [("%m", mt), ("%d", "index")], [f"%0 = memref.dim %m , %d : {mt}"], ("%0", "index"), "dim_cast", nl, "arith+func")
        if variant == "dim_cast":
            d = self.rng.randrange(rank); it = self.rng.choice(["i32", "i64"])
            nl = f"{self.prefix(nm)} reports the extent of dimension {d} of a {mt}, converted from index to {it} with arith.index_cast."
            return Program(nm, [("%m", mt)], [f"%c = arith.constant {d} : index", f"%0 = memref.dim %m , %c : {mt}", f"%1 = arith.index_cast %0 : index to {it}"],
                           ("%1", it), "dim_cast", nl, "arith+func")
        nl = f"{self.prefix(nm)} returns the product of the sizes of dimensions 0 and 1 of a {mt} as an index (two memref.dim ops and an index multiply)."
        return Program(nm, [("%m", mt)], [f"%c0 = arith.constant 0 : index", f"%c1 = arith.constant 1 : index", f"%0 = memref.dim %m , %c0 : {mt}",
                                          f"%1 = memref.dim %m , %c1 : {mt}", f"%2 = arith.muli %0 , %1 : index"], ("%2", "index"), "dim_cast", nl, "arith+func")

    def poly(self) -> Program:
        is_int = self.rng.random() < 0.6
        t = self.rng.choice(INT_TYPES if is_int else FLOAT_TYPES); nm = self.name()
        mul, add = ("arith.muli", "arith.addi") if is_int else ("arith.mulf", "arith.addf")
        variant = self.rng.choice(["square", "square", "square_plus", "cube", "quadratic"])
        x = self.rng.choice(["%x", "%v", "%a", "%n"])
        if variant == "square":
            nl = f"{self.prefix(nm)} returns the square of the {t} input {x[1:]} ({x[1:]} times itself)."
            return Program(nm, [(x, t)], [f"%0 = {mul} {x} , {x} : {t}"], ("%0", t), "poly", nl, "arith+func")
        if variant == "cube":
            nl = f"{self.prefix(nm)} returns the cube of the {t} input {x[1:]} (two multiplications of {x[1:]} by itself)."
            return Program(nm, [(x, t)], [f"%0 = {mul} {x} , {x} : {t}", f"%1 = {mul} %0 , {x} : {t}"], ("%1", t), "poly", nl, "arith+func")
        if variant == "square_plus":
            y = "%y" if x != "%v" else "%w"
            nl = f"{self.prefix(nm)} returns {x[1:]} squared plus {y[1:]} for two {t} inputs {x[1:]} and {y[1:]}."
            return Program(nm, [(x, t), (y, t)], [f"%0 = {mul} {x} , {x} : {t}", f"%1 = {add} %0 , {y} : {t}"], ("%1", t), "poly", nl, "arith+func")
        nl = f"{self.prefix(nm)} evaluates the quadratic p times x squared, plus q times x, plus r, for {t} inputs p, q, r and x."
        return Program(nm, [("%p", t), ("%q", t), ("%r", t), ("%x", t)], [f"%0 = {mul} %x , %x : {t}", f"%1 = {mul} %p , %0 : {t}", f"%2 = {mul} %q , %x : {t}",
                                                                          f"%3 = {add} %1 , %2 : {t}", f"%4 = {add} %3 , %r : {t}"], ("%4", t), "poly", nl, "arith+func")

    LINALG_EW_NOUN = {"linalg.add": ("sum", "+"), "linalg.sub": ("difference", "-"), "linalg.mul": ("product", "*"), "linalg.div": ("quotient", "/")}

    def linalg_ew2(self) -> Program:
        t = self.rng.choice(["f32", "f32", "f64", "f16"]); nm = self.name(); rank = self.rng.choice([1, 2])
        dims = ["?" if self.rng.random() < 0.3 else str(self.rng.choice(self.STATIC_DIMS)) for _ in range(rank)]
        M = f"memref<{'x'.join(dims)}x{t}>"
        if self.rng.random() < 0.75:
            op, (noun, sym) = self.rng.choice(list(self.LINALG_EW_NOUN.items()))
            a, b, c = self.rng.choice([("%x", "%y", "%out"), ("%a", "%b", "%c"), ("%lhs", "%rhs", "%res")])
            if self.rng.random() < 0.5:
                nl = f"{self.prefix(nm)} computes the elementwise {noun} of the {M} buffers {a[1:]} and {b[1:]} with {op}, writing each result into the {M} buffer {c[1:]}."
            else:
                nl = f"{self.prefix(nm)} takes two {M} operands {a[1:]} and {b[1:]} and an output {M} {c[1:]}, and stores {a[1:]} {sym} {b[1:]} elementwise into {c[1:]}."
            return Program(nm, [(a, M), (b, M), (c, M)], [f"{op} ins({a} , {b} : {M}, {M}) outs({c} : {M})"], None, "linalg_ew2", nl, "linalg")
        op, word = self.rng.choice(list(LINALG_UN.items())); a, b = self.rng.choice([("%x", "%y"), ("%in", "%out"), ("%src", "%dst")])
        nl = f"{self.prefix(nm)} applies {op} (elementwise {word}) to every element of the {M} buffer {a[1:]}, storing the results in the {M} buffer {b[1:]}."
        return Program(nm, [(a, M), (b, M)], [f"{op} ins({a} : {M}) outs({b} : {M})"], None, "linalg_ewun2", nl, "linalg")

    ARITH_FAMILIES = ("binop", "binop", "chain", "chain", "chain", "constant", "compare", "cast", "minmax3", "memref_load", "memref_load")
    ARITH_FAMILIES_V1C = ("alloc_dealloc", "alloc_dealloc", "negate", "clamp", "cmp_const", "const_op2", "cast_chain", "cast_chain", "bitcast",
                          "fminmax", "cond_select", "mem_elem", "mem_elem", "dim_cast", "poly")
    P_NEW_ARITH = 0.2
    P_NEW_LINALG = 0.15

    def sample(self, dialect: str) -> Program:
        if self.families == "v1c":
            if dialect == "linalg":
                return self.linalg_ew2() if self.rng.random() < self.P_NEW_LINALG else self.linalg()
            if self.rng.random() < self.P_NEW_ARITH:
                return getattr(self, self.rng.choice(self.ARITH_FAMILIES_V1C))()
        if dialect == "linalg":
            return self.linalg()
        return getattr(self, self.rng.choice(self.ARITH_FAMILIES))()


V1C_FAMILIES = frozenset(Gen.ARITH_FAMILIES_V1C) | {"linalg_ew2", "linalg_ewun2"}


def generate(n_arith: int, n_linalg: int, seed: int = 0, families: str = "v1") -> list[dict]:
    g = Gen(seed, families); out = []; seen = set()
    tag = f"tpl_{seed}" if families == "v1" else f"tpl_{families}_{seed}"
    for dialect, n in (("arith+func", n_arith), ("linalg", n_linalg)):
        tries = 0
        while sum(1 for r in out if r["dialect"] == dialect) < n and tries < 50 * n:
            tries += 1
            p = g.sample(dialect); m = p.mlir()
            key = (p.nl, m)
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": f"{tag}_{len(out):06d}", "dialect": dialect, "family": p.family, "nl": p.nl, "mlir": m})
    return out
