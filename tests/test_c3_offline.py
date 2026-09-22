"""Offline per-token C3 mask: scope events and allowed sets (tokenizer-dependent parts marked mlx)."""
from __future__ import annotations

import numpy as np
import pytest

from sci.masks.c3_offline import in_scope_at, rows_to_bool, scope_events

PROG = """module {
  func.func @f(%a : i32 , %b : i32) -> i32 {
    %0 = arith.addi %a , %b : i32
    %1 = arith.muli %0 , %b : i32
    %2 = arith.subi %1 , %zz : i32
    return %2 : i32
  }
}"""


def test_scope_events_defs_and_uses():
    uses, defs = scope_events(PROG)
    assert set(defs) == {"a", "b", "0", "1", "2"}
    assert [u[2] for u in uses] == ["a", "b", "0", "b", "1", "zz", "2"]
    # params visible from the body's opening brace; %0 visible only after its line ends
    first_use = uses[0][0]
    assert in_scope_at(defs, first_use) == ["a", "b"]
    assert "0" in in_scope_at(defs, uses[2][0]) and "1" not in in_scope_at(defs, uses[2][0])
    assert "2" in in_scope_at(defs, uses[-1][0])


def test_linalg_uses_no_lhs():
    prog = """module {
  func.func @mm(%A : memref<?x?xf32> , %B : memref<?x?xf32> , %C : memref<?x?xf32>) {
    linalg.matmul ins(%A , %B : memref<?x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
    return
  }
}"""
    uses, defs = scope_events(prog)
    assert [u[2] for u in uses] == ["A", "B", "C"] and set(defs) == {"A", "B", "C"}


@pytest.fixture(scope="module")
def builder():
    pytest.importorskip("mlx.core")
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer
    from sci.masks.c3_offline import C3MaskBuilder
    path = Path(snapshot_download("HuggingFaceTB/SmolLM2-135M-Instruct", allow_patterns=["*.json", "*.txt"]))
    tok = load_tokenizer(path)
    return tok, C3MaskBuilder(tok)


@pytest.mark.mlx
def test_allowed_sets_follow_scope(builder):
    tok, b = builder
    ids = tok.encode(PROG, add_special_tokens=False)
    rows = b.build(ids)
    assert len(rows) == 7
    decoded = [tok.decode([ids[p]]) for p, _ in rows]
    assert decoded == ["a", "b", "0", "b", "1", "zz", "2"]
    for (pos, allowed), name in zip(rows, decoded):
        assert (ids[pos] in set(allowed.tolist())) == (name != "zz")
    # allowed sets grow as names come into scope, and never include a not-yet-defined name
    sizes = [len(a) for _, a in rows]
    assert sizes[0] < sizes[2] < sizes[4] <= sizes[-1]
    strs = {tok.decode([i]) for i in rows[0][1]}
    assert "a" in strs and "b" in strs and "0" not in strs and "2" not in strs
    dense = rows_to_bool(rows, len(ids), b.vocab_size)
    assert dense.shape == (len(ids), b.vocab_size) and dense.all(axis=1).sum() == len(ids) - 7


@pytest.mark.mlx
def test_multi_token_names(builder):
    tok, b = builder
    prog = """module {
  func.func @f(%value_left : i64 , %value_right : i64) -> i64 {
    %result = arith.addi %value_left , %value_right : i64
    return %result : i64
  }
}"""
    ids = tok.encode(prog, add_special_tokens=False)
    rows = b.build(ids)
    assert rows, "operand names spanning several tokens must produce rows"
    for pos, allowed in rows:
        assert ids[pos] in set(allowed.tolist())  # the gold's own tokens are always legal
