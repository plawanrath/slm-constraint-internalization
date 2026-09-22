"""Probe tests: use-site finder vs the C3 validator, the probe trainer, and an MLX smoke test."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from sci.constraints.scope import validate
from sci.probes.positions import (attach_token_positions, char_to_token, corrupt_use, find_sites,
                                  match_site)
from sci.probes.probe import LogisticProbe, evaluate_probe

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REPO = Path(__file__).resolve().parents[1]

ARITH = textwrap.dedent("""\
    module {
      func.func @fma(%a : i32 , %b : i32 , %c : i32) -> i32 {
        %0 = arith.addi %a , %b : i32
        %1 = arith.muli %0 , %c : i32
        %2 = arith.subi %3 , %1 : i32
        %3 = arith.constant 7 : i32
        return %2 : i32
      }
    }
    """)

MEMREF = textwrap.dedent("""\
    module {
      func.func @ld(%m : memref<?xf32> , %i : index) -> f32 {
        %0 = memref.load %m[%i] : memref<?xf32>
        memref.store %0 , %m[%j] : memref<?xf32>
        %j = arith.constant 0 : index
        return %0 : f32
      }
    }
    """)

LINALG = textwrap.dedent("""\
    module {
      func.func @mm(%A : memref<?x?xf32> , %B : memref<?x?xf32> , %C : memref<?x?xf32>) {
        linalg.matmul ins(%A , %B : memref<?x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
        linalg.fill ins(%v : f32) outs(%C : memref<?x?xf32>)
        return
      }
    }
    """)

TWO_FUNCS = textwrap.dedent("""\
    module {
      func.func @f(%a : i32) -> i32 {
        %0 = arith.addi %a , %a : i32
        return %0 : i32
      }
      func.func @g(%b : i32) -> i32 {
        %1 = arith.addi %b , %0 : i32
        return %1 : i32
      }
    }
    """)


def _undef(text: str) -> set[str]:
    return {v.detail[1:] for v in validate(text).violations if v.kind == "undef_use"}


@pytest.mark.parametrize("text", [ARITH, MEMREF, LINALG, TWO_FUNCS])
def test_labels_agree_with_validator(text):
    uses, _ = find_sites(text)
    assert uses
    assert {u.name for u in uses if u.checked and u.label == 0} == _undef(text)
    for u in uses:
        assert text[u.char_pos] == "%" and text[u.char_pos + 1:].startswith(u.name)


def test_arith_scope_sets():
    uses, defs = find_sites(ARITH)
    by = {(u.stmt_index, u.name): u for u in uses}
    assert by[(0, "a")].in_scope == {"a", "b", "c"}
    assert by[(1, "0")].in_scope == {"a", "b", "c", "0"} and by[(1, "0")].out_of_scope == {"1", "2", "3"}
    assert by[(2, "3")].label == 0 and by[(2, "1")].label == 1
    assert by[(4, "2")].in_scope == {"a", "b", "c", "0", "1", "2", "3"}
    assert [d.name for d in defs] == ["a", "b", "c", "0", "1", "2", "3"]
    assert [d.stmt_index for d in defs[:3]] == [-1, -1, -1]


def test_memref_and_linalg_forms():
    uses, _ = find_sites(MEMREF)
    store = [u for u in uses if u.stmt_index == 1]
    assert [(u.name, u.label) for u in store] == [("0", 1), ("m", 1), ("j", 0)]
    uses, _ = find_sites(LINALG)
    assert [(u.name, u.label, u.checked) for u in uses] == [("A", 1, True), ("B", 1, True), ("C", 1, True),
                                                            ("v", 0, True), ("C", 1, True)]


def test_two_functions_scope_resets():
    uses, _ = find_sites(TWO_FUNCS)
    g = [u for u in uses if u.func_index == 1]
    assert {(u.name, u.label) for u in g} == {("b", 1), ("0", 0), ("1", 1)}
    assert "0" in [u for u in g if u.name == "0"][0].out_of_scope


def test_unchecked_ops_are_flagged():
    text = ("module {\n  func.func @g(%in : memref<?x4xf64> , %out : memref<4x?xf64>) {\n"
            "    linalg.transpose ins(%in : memref<?x4xf64>) outs(%out : memref<4x?xf64>) permutation = [1 , 0]\n"
            "    return\n  }\n}\n")
    uses, _ = find_sites(text)
    assert [u.checked for u in uses] == [False, False] and validate(text).passed


def test_corruption_makes_use_out_of_scope():
    uses, defs = find_sites(ARITH)
    site = [u for u in uses if u.stmt_index == 1 and u.name == "0"][0]
    c_text, new = corrupt_use(ARITH, site, uses, defs)
    assert c_text[site.char_pos:] == ARITH[site.char_pos:]      # unchanged from the use on
    assert f"%{new} = arith.addi" in c_text
    c_uses, _ = find_sites(c_text)
    assert match_site(c_uses, uses, site).label == 0 and _undef(c_text) == {"0", "3"}  # %3 was already undefined


def test_dev_pool_agreement():
    rows = [json.loads(l) for l in (REPO / "data/pools/dev_v1.jsonl").read_text().splitlines()[:150]]
    for r in rows:
        uses, defs = find_sites(r["mlir"])
        assert {u.name for u in uses if u.checked and u.label == 0} == _undef(r["mlir"])
        for u in uses:
            if u.checked and u.label == 1:
                c, _ = corrupt_use(r["mlir"], u, uses, defs)
                assert _undef(c) == {u.name}


def test_char_to_token_and_attach():
    offsets = [(0, 3), (3, 5), (5, 6), (6, 8)]   # '   ', ' %', '0', ' ='
    assert char_to_token(offsets, 4) == 1 and char_to_token(offsets, 5) == 2 and char_to_token(offsets, 0) == 0
    text = "%0 = arith.addi %a , %b : i32"
    from sci.probes.positions import DefSite, UseSite
    u = UseSite("a", 16, 0, 0); d = DefSite("0", 0, 0, 0)
    offs = [(i, i + 1) for i in range(len(text))]
    attach_token_positions([u], [d], offs, base=10)
    assert (u.pct_pos, u.name_pos, u.pred_pos, d.pct_pos) == (26, 27, 26, 10)


def test_probe_recovers_separable_data_and_control_is_chance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 10)); w = rng.normal(size=10)
    y = (X @ w > 0).astype(int)
    res = evaluate_probe(X, y, n_boot=500)
    assert res["acc"] > 0.95 and res["ci_low"] > 0.9
    assert abs(res["control_acc"] - 0.5) < 0.08
    p = LogisticProbe().fit(X, y)
    assert (p.predict(X) == y).mean() > 0.95


def test_probe_groups_keep_programs_together():
    from sci.probes.probe import group_folds
    groups = np.repeat(np.arange(20), 5)
    folds = group_folds(groups, 5)
    assert sorted(np.concatenate(folds).tolist()) == list(range(100))
    for f in folds:
        assert len(set(groups[f])) == len(f) // 5


@pytest.mark.mlx
def test_capture_and_patching_smoke():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm import load
    from sci.probes.capture import capture_residuals, capture_residuals_batch
    from sci.probes.dataset import build_examples, design_matrix, extract_features
    from sci.probes.patching import run_patching

    model, tok = load(MODEL)
    model.set_dtype(mx.float32)
    d = model.args.hidden_size
    a = tok.encode(ARITH, add_special_tokens=False)
    b = tok.encode("module {\n  func.func @g() {\n", add_special_tokens=False)
    single = capture_residuals(model, a, [2, 29])
    assert set(single) == {2, 29} and all(v.shape == (len(a), d) for v in single.values())
    batch = capture_residuals_batch(model, [a, b], [2, 29])
    for l in (2, 29):
        tol = 1e-2 * float(np.abs(single[l]).max())
        assert np.allclose(batch[0][l], single[l], atol=tol)
        assert batch[1][l].shape == (len(b), d)
    progs = [{"text": ARITH, "nl": "Write fma.", "dialect": "arith+func"},
             {"text": MEMREF, "nl": "Load and store.", "dialect": "arith+func"}]
    seqs, examples, pairs = build_examples(tok, progs)
    assert pairs and any(e.design == "def_diff" for e in examples)
    feats, row = extract_features(model, seqs, examples, [2, 29], micro_batch=4)
    X, y, groups = design_matrix(feats, row, examples, "use")
    assert X[2].shape == (len(y), d) and set(y.tolist()) == {0, 1}
    res = run_patching(model, pairs[:1], layers=[2, 29], micro_batch=2)
    assert res["n_pairs"] == 1 and set(res["per_layer"]) == {2, 29}
    # patching the last block reproduces the clean logits exactly
    assert abs(res["per_layer"][29]["target_logprob_delta"] - (res["target_logprob_clean"] - res["target_logprob_corrupt"])) < 1e-3
