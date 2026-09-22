"""LLVM IR generation grammars: file access, llguidance compilation, lark parse check."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

GRAMMAR_DIR = Path(__file__).resolve().parent / "grammars"
GRAMMAR_NAMES = ("llvm_gen_c1", "llvm_gen_c1c2")

# Constraint stack -> grammar name (mirrors `sci.eval.generate.GRAMMAR_FOR`).
GRAMMAR_FOR = {"c1": "llvm_gen_c1", "c1_c2": "llvm_gen_c1c2", "c1_c2_c3": "llvm_gen_c1c2"}


def grammar_path(name: str = "llvm_gen_c1c2") -> Path:
    if name not in GRAMMAR_NAMES:
        raise KeyError(f"unknown LLVM grammar {name!r}; expected one of {GRAMMAR_NAMES}")
    return GRAMMAR_DIR / f"{name}.lark"


def grammar_text(name: str = "llvm_gen_c1c2") -> str:
    """Raw LARK text of a bundled LLVM generation grammar."""
    return grammar_path(name).read_text()


@lru_cache(maxsize=None)
def llg_grammar(name: str = "llvm_gen_c1c2") -> str:
    """Compiled llguidance grammar spec (cached per name)."""
    from llguidance import LLMatcher
    return LLMatcher.grammar_from_lark(grammar_text(name))


@lru_cache(maxsize=None)
def _parser(name: str):
    from lark import Lark
    return Lark(grammar_text(name), parser="earley", lexer="dynamic_complete", start="start")


def is_parse_valid(text: str, name: str = "llvm_gen_c1") -> bool:
    """True iff `text` is accepted by the named grammar (default: C1 syntax)."""
    from lark.exceptions import LarkError
    try:
        _parser(name).parse(text)
        return True
    except LarkError:
        return False
