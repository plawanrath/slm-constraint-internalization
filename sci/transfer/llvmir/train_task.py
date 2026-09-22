"""Training-task adapter for textual LLVM IR (second training target).

Same protocol as `sci.synth.task.SynthTask` so `scripts/w03_train.py --task llvm` runs SSD, RFT, GRPO and
SFT-gold on an LLVM IR pool (built by `scripts/w22_llvm_pool.py`) with no change to the training driver:

    build_prompt(tokenizer, nl) -> str      3-shot prompt (sci.transfer.llvmir.task)
    grammar_name / c1_grammar_name          llvm_gen_c1c2 / llvm_gen_c1 (sci.transfer.llvmir.grammar)
    reward(text, gold, kind) -> (r, parts)  kind 'task': verify · signature · opcode-Jaccard, gated on the
                                            dominance-aware scope validator; 'verify'; 'functional' adds one
                                            seeded lli differential trial against the gold; *_noc3 drop the gate
    scope_ok(text) -> bool                  C3 verdict
    gold_key = "gold"                       pool column holding the gold program
"""
from __future__ import annotations

import re
from collections import Counter

from sci.transfer.llvmir import functional as F
from sci.transfer.llvmir import task as P
from sci.transfer.llvmir.grammar import grammar_text, is_parse_valid
from sci.transfer.llvmir.scope import accept_or_reject
from sci.transfer.llvmir.verify import passes

_OPCODE_RE = re.compile(r"^\s*(?:%[\w.]+\s*=\s*)?([a-z][a-z0-9_.]*)", re.M)
_SKIP = {"define", "entry", "ret", "br", "declare"}


def opcodes(text: str) -> Counter:
    return Counter(op for op in _OPCODE_RE.findall(text) if op not in _SKIP and not op.endswith(":"))


def signature_score(text: str, gold: str) -> float:
    """Graded agreement of the first `define` with the gold's: name, return type, argument types (thirds)."""
    g, c = F.parse_define(gold), F.parse_define(text)
    if g is None or c is None:
        return 0.0
    return ((g["name"] == c["name"]) + (g["ret"] == c["ret"]) + (g["arg_types"] == c["arg_types"])) / 3.0


def op_jaccard(a: Counter, b: Counter) -> float:
    inter = sum((a & b).values()); union = sum((a | b).values())
    return inter / union if union else 0.0


def ref_from_gold(gold: str) -> dict | None:
    """A differential reference (signature + default input domains) derived from the gold's define line."""
    sig = F.parse_define(gold)
    if sig is None:
        return None
    return {"fn_name": sig["name"], "signature": {"args": list(sig["arg_types"]), "ret": sig["ret"]},
            "input_domains": [{"type": t, "min": -50, "max": 50} for t in sig["arg_types"]]}


class LlvmTask:
    name = "llvm"
    grammar_name = "llvm_gen_c1c2"
    c1_grammar_name = "llvm_gen_c1"
    gold_key = "gold"

    @staticmethod
    def grammar_lark(name: str = "llvm_gen_c1c2") -> str:
        return grammar_text(name)

    @staticmethod
    def build_prompt(tokenizer, nl: str) -> str:
        return P.build_prompt(tokenizer, nl)

    @staticmethod
    def scope_ok(text: str) -> bool:
        return accept_or_reject(text)[0]

    @staticmethod
    def accept_or_reject(text: str):
        return accept_or_reject(text)

    @staticmethod
    def check(text: str) -> dict:
        pv = is_parse_valid(text, "llvm_gen_c1")
        sc = bool(pv) and accept_or_reject(text)[0]
        vv = bool(pv) and passes(text)
        return {"parse_valid": pv, "scope_ok": sc, "verify_valid": vv, "passed": pv and sc and vv}

    @staticmethod
    def reward(text: str, gold: str | None = None, kind: str = "task") -> tuple[float, dict]:
        pv = is_parse_valid(text, "llvm_gen_c1")
        sc = bool(pv) and accept_or_reject(text)[0]
        gate = pv if kind.endswith("_noc3") else sc
        vv = bool(gate) and passes(text)
        parts = {"parse": pv, "scope": sc, "verify": vv}
        if kind in ("verify", "verify_noc3") or gold is None:
            return float(vv), parts
        sig = signature_score(text, gold) if vv else 0.0
        jac = op_jaccard(opcodes(text), opcodes(gold)) if vv else 0.0
        parts.update(sig=sig, jaccard=jac)
        r = float(vv) * sig * jac
        if kind.startswith("functional"):
            fm = False
            if vv:
                ref = ref_from_gold(gold)
                if ref is not None:
                    d = F.differential(text, gold, ref, k=1)
                    fm = d.get("status") == "match" or d.get("n_trials_matched", 0) >= 1
            parts["functional"] = fm
            r += 1.0 * float(fm)
        return r, parts
