"""Synthetic SSA language family: grammar (C1/C2), generator with tunable dependency distance,
four-layer checker (C1–C4), task adapter for the training driver, and the residual scaling-law fit."""
from sci.synth.checker import accept_or_reject, check, dep_distances, max_dep_distance, reward_fn, scope_ok
from sci.synth.generator import generate, make_pool
from sci.synth.grammar import compile_grammar, grammar_path, grammar_text

__all__ = ["accept_or_reject", "check", "dep_distances", "max_dep_distance", "reward_fn", "scope_ok",
           "generate", "make_pool", "compile_grammar", "grammar_path", "grammar_text"]
