"""Mask module tests (need llguidance + a tokenizer + MLX; marked mlx)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")
mx = pytest.importorskip("mlx.core")

from sci.masks.llg import (GrammarLogitsProcessor, bitmask_to_bool, lark_grammar, legal_mass,
                           llg_tokenizer, replay_allowed_sets)

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _hf():
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer
    return load_tokenizer(Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.txt"])))


@pytest.fixture(scope="module")
def hf():
    return _hf()


@pytest.fixture(scope="module")
def tok(hf):
    return llg_tokenizer(hf)


@pytest.mark.mlx
def test_replay_accepts_grammar_string(tok, hf):
    text = "module {\n  func.func @f(%a : i32 , %b : i32) -> i32 {\n    %0 = arith.addi %a , %b : i32\n    return %0 : i32\n  }\n}\n"
    ids = hf.encode(text, add_special_tokens=False)
    g = lark_grammar("mlir_gen_c1c2")
    masks = replay_allowed_sets(tok, g, ids)
    assert masks.shape[0] == len(ids)
    for t, tid in enumerate(ids):
        allowed = bitmask_to_bool(masks[t], tok.vocab_size)
        assert allowed[tid], f"token {tid} ({hf.decode([tid])!r}) at {t} not in its own allowed set"


@pytest.mark.mlx
def test_replay_rejects_illegal_prefix(tok, hf):
    ids = hf.encode("modulemodule", add_special_tokens=False)
    with pytest.raises(ValueError):
        replay_allowed_sets(tok, lark_grammar("mlir_gen_c1"), ids)


@pytest.mark.mlx
def test_legal_mass_bounds(tok, hf):
    ids = hf.encode("module {\n", add_special_tokens=False)
    masks = replay_allowed_sets(tok, lark_grammar("mlir_gen_c1"), ids)
    V = tok.vocab_size
    rng = np.random.default_rng(0)
    logits = mx.array(rng.normal(size=(len(ids), V)).astype(np.float32))
    lm = legal_mass(logits, masks, V)
    assert lm.shape == (len(ids),)
    assert bool(mx.all(lm <= 1e-5))
    # each sampled token's own probability is a lower bound on the legal mass
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    own = mx.array([float(lp[t, tid]) for t, tid in enumerate(ids)])
    assert bool(mx.all(lm >= own - 1e-5))


@pytest.mark.mlx
def test_processor_matches_replay(tok, hf):
    """Online processor masks and offline replay agree at every step."""
    ids = hf.encode("module {\n  func.func @f() -> i32 {\n", add_special_tokens=False)
    g = lark_grammar("mlir_gen_c1")
    proc = GrammarLogitsProcessor(tok, g)
    V = tok.vocab_size
    offline = replay_allowed_sets(tok, g, ids)
    seen = [999]  # one "last prompt token" as mlx_lm would pass
    for t, tid in enumerate(ids):
        out = proc(mx.array(seen), mx.zeros((1, V)))
        online = np.array(out[0] > -1e9)
        assert np.array_equal(online, bitmask_to_bool(offline[t], V)), f"mismatch at step {t}"
        seen.append(tid)
