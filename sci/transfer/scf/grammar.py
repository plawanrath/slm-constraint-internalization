"""scf grammars: generation grammars (C1 / C1+C2), the parse grammar, lattice path.

The generation grammars are bundled here rather than under `sci.constraints.grammars`
so the training-dialect grammar set stays untouched; `sci.masks.llg.lark_grammar`
is bypassed by `ScfGenerator.grammar`. The parse grammar is `mlir.lark` plus the
scf extension in `mlir_scf.lark`, concatenated at load time.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

GRAMMAR_DIR = Path(__file__).resolve().parent / "grammars"
LATTICE_PATH = Path(__file__).resolve().parent / "lattices" / "scf.json"
GRAMMAR_NAMES = ("mlir_gen_scf_c1", "mlir_gen_scf")

# Constraint stack -> grammar name (mirrors `sci.eval.generate.GRAMMAR_FOR`).
GRAMMAR_FOR = {"c1": "mlir_gen_scf_c1", "c1_c2": "mlir_gen_scf", "c1_c2_c3": "mlir_gen_scf"}


def grammar_path(name: str = "mlir_gen_scf") -> Path:
    if name not in GRAMMAR_NAMES:
        raise KeyError(f"unknown scf grammar {name!r}; expected one of {GRAMMAR_NAMES}")
    return GRAMMAR_DIR / f"{name}.lark"


def grammar_text(name: str = "mlir_gen_scf") -> str:
    """Raw LARK text of a bundled scf generation grammar."""
    return grammar_path(name).read_text()


@lru_cache(maxsize=None)
def llg_grammar(name: str = "mlir_gen_scf") -> str:
    """Compiled llguidance grammar spec (cached per name)."""
    from llguidance import LLMatcher
    return LLMatcher.grammar_from_lark(grammar_text(name))


def parse_grammar_text() -> str:
    """`mlir.lark` followed by the scf `%extend` block from `mlir_scf.lark`."""
    from sci.constraints.parser import grammar_path as base_path
    return base_path().read_text() + "\n" + (GRAMMAR_DIR / "mlir_scf.lark").read_text()


@lru_cache(maxsize=1)
def _parser():
    from lark import Lark
    return Lark(parse_grammar_text(), parser="earley", lexer="dynamic_complete", start="start",
                propagate_positions=False, maybe_placeholders=False)


def is_parse_valid_scf(text: str) -> bool:
    """True iff the scf-extended parse grammar accepts `text` (no semantic check)."""
    from lark.exceptions import LarkError
    try:
        _parser().parse(text)
        return True
    except LarkError:
        return False
