"""Token-level grammar masks via llguidance.

Two uses share one matcher:
  * online: a logits processor for `mlx_lm` generation that forbids tokens the LARK
    generation grammar cannot continue (the C1 / C1+C2 masks);
  * offline: `replay_allowed_sets` reconstructs, for an already sampled token sequence,
    the allowed-token bitmask at every position by replaying the tokens through a fresh
    matcher. Training losses consume these bitmasks; nothing is captured during sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from llguidance import LLMatcher, LLTokenizer
from llguidance.hf import from_tokenizer
from llguidance.mlx import apply_token_bitmask
from llguidance.numpy import allocate_token_bitmask, fill_next_token_bitmask

from sci.constraints.parser import generation_grammar_path


def lark_grammar(name: str = "mlir_gen_c1c2") -> str:
    """Compiled llguidance grammar spec from a bundled LARK generation grammar."""
    return LLMatcher.grammar_from_lark(generation_grammar_path(name).read_text())


def llg_tokenizer(hf_tokenizer) -> LLTokenizer:
    """llguidance tokenizer from an HF (or mlx_lm TokenizerWrapper) tokenizer."""
    tok = getattr(hf_tokenizer, "_tokenizer", hf_tokenizer)
    return from_tokenizer(tok)


@dataclass
class MaskState:
    """Per-sequence matcher plus the number of tokens it has consumed."""
    matcher: LLMatcher
    consumed: int = 0


class GrammarLogitsProcessor:
    """`mlx_lm` logits processor: masks logits with the grammar's allowed set.

    mlx_lm calls `processor(tokens, logits)` with `logits` of shape (1, vocab) and
    `tokens` = the tokens fed to the model since the prefill (mlx_lm prefills all but
    the last prompt token without processors, so the first call sees one prompt token)
    followed by everything generated. The length at the first call is the anchor; the
    matcher is advanced by tokens generated since the previous call.
    """

    def __init__(self, llg_tok: LLTokenizer, grammar: str, prompt_len: int | None = None):
        self.llg_tok = llg_tok
        self.grammar = grammar
        self.matcher = LLMatcher(llg_tok, grammar)
        self.base_len: int | None = None
        self.consumed = 0
        self.bitmask = allocate_token_bitmask(1, llg_tok.vocab_size)
        self.errors: list[str] = []

    def reset(self) -> None:
        self.matcher.reset(); self.consumed = 0; self.base_len = None; self.errors.clear()

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.base_len is None:
            self.base_len = int(tokens.shape[-1])
        n_gen = int(tokens.shape[-1]) - self.base_len
        if n_gen > self.consumed:
            new = tokens[self.base_len + self.consumed:].tolist()
            for t in new:
                self.matcher.consume_token(int(t))
                if self.matcher.is_error():
                    self.errors.append(self.matcher.get_error()); break
            self.consumed = n_gen
        if self.matcher.is_stopped():
            return logits
        fill_next_token_bitmask(self.matcher, self.bitmask, 0)
        shape = logits.shape
        biased = apply_token_bitmask(logits.reshape(-1, shape[-1]), self.bitmask)  # returns (1, vocab)
        return biased.reshape(shape)


def replay_allowed_sets(llg_tok: LLTokenizer, grammar: str, gen_tokens: list[int]) -> np.ndarray:
    """Bitmask (len(gen_tokens), ceil(vocab/32)) int32: row t is the allowed set *before*
    emitting gen_tokens[t]. Raises if a token was never legal (the sequence was not
    produced under this grammar)."""
    m = LLMatcher(llg_tok, grammar)
    n = len(gen_tokens)
    out = allocate_token_bitmask(max(n, 1), llg_tok.vocab_size)
    for t, tok in enumerate(gen_tokens):
        if m.is_stopped():
            out[t:] = 0; break
        fill_next_token_bitmask(m, out, t)
        m.consume_token(int(tok))
        if m.is_error():
            raise ValueError(f"token {tok} at position {t} is illegal under the grammar: {m.get_error()}")
    return out[:n]


def bitmask_to_bool(bitmask_row: np.ndarray, vocab_size: int) -> np.ndarray:
    """Unpack one int32 bitmask row into a bool vector of length vocab_size."""
    bits = np.unpackbits(bitmask_row.view(np.uint8), bitorder="little")
    return bits[:vocab_size].astype(bool)


def legal_mass(logits: mx.array, bitmask_rows: np.ndarray, vocab_size: int) -> mx.array:
    """log Z_t = log Σ_{v∈A_t} softmax(logits_t)[v] for each row t (shape (T,)).

    logits: (T, vocab) student logits at the generated positions; bitmask_rows: (T, W)
    from `replay_allowed_sets`. Returns log legal mass per position (≤ 0)."""
    allowed = np.stack([bitmask_to_bool(r, vocab_size) for r in bitmask_rows])  # (T, V)
    allowed_mx = mx.array(allowed)
    logz_all = mx.logsumexp(logits, axis=-1)
    masked = mx.where(allowed_mx, logits, mx.array(-1e30, dtype=logits.dtype))
    logz_allowed = mx.logsumexp(masked, axis=-1)
    return logz_allowed - logz_all
