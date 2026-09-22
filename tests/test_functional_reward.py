"""Tests for the functional (gold-differential) reward kind.

The docker-marked tests lower and execute in the pinned LLVM container; the fast tests monkeypatch the
verifier so `reward_fn` needs no container.
"""
from __future__ import annotations

import time

import pytest

from sci.train import functional_reward as fr
from sci.train import spine
from sci.train.spine import reward_fn

GOLD = """module {
  func.func @f(%a : i32) -> i32 {
    %c = arith.constant -82932 : i32
    %0 = arith.subi %a , %c : i32
    return %0 : i32
  }
}
"""
# operands of the (non-commutative) subtraction swapped: same signature and op multiset, different semantics
MUTATED = GOLD.replace("arith.subi %a , %c", "arith.subi %c , %a")
# valid program with a different interface (i64 instead of i32)
WRONG_SIG = GOLD.replace("i32", "i64")

LINALG_GOLD = """module {
  func.func @calc2(%src : memref<16xf64> , %dst : memref<16xf64>) {
    linalg.copy ins(%src : memref<16xf64>) outs(%dst : memref<16xf64>)
    return
  }
}
"""
LINALG_MUTATED = LINALG_GOLD.replace("linalg.copy ins(%src : memref<16xf64>) outs(%dst : memref<16xf64>)",
                                     "%c = arith.constant 1.0 : f64\n    linalg.fill ins(%c : f64) outs(%dst : memref<16xf64>)")


@pytest.fixture
def tmp_cache(tmp_path):
    fr.set_cache_path(tmp_path / "functional_reward_cache.sqlite")
    yield tmp_path
    fr.set_cache_path(None)


@pytest.mark.docker
def test_gold_matches_itself(tmp_cache):
    r = fr.functional_match(GOLD, GOLD)
    assert r["match"] is True and r["status"] == "match", r
    r = fr.functional_match(LINALG_GOLD, LINALG_GOLD)
    assert r["match"] is True and r["status"] == "match", r


@pytest.mark.docker
def test_mutated_gold_mismatches(tmp_cache):
    r = fr.functional_match(MUTATED, GOLD)
    assert r["match"] is False and r["status"] == "mismatch", r
    r = fr.functional_match(LINALG_MUTATED, LINALG_GOLD)
    assert r["match"] is False and r["status"] == "mismatch", r


@pytest.mark.docker
def test_wrong_signature(tmp_cache):
    r = fr.functional_match(WRONG_SIG, GOLD)
    assert r["match"] is False and r["status"] == "signature_mismatch", r


def test_unsupported_never_raises(tmp_cache):
    r = fr.functional_match("garbage", "func.func @g(%m : memref<2xi8>) { return }")
    assert r == {"match": False, "status": "unsupported", "dt": r["dt"]}
    r = fr.functional_match("garbage", "")
    assert r["status"] == "unsupported" and r["match"] is False


@pytest.mark.docker
def test_reward_fn_functional_kind(tmp_cache):
    r_gold, p_gold = reward_fn(GOLD, GOLD, "functional")
    r_task, p_task = reward_fn(GOLD, GOLD, "task")
    assert p_task["verify"] is True
    assert r_gold == pytest.approx(r_task + 1.0)
    assert p_gold["functional"]["status"] == "match" and p_gold["functional"]["match"] is True
    r_mut, p_mut = reward_fn(MUTATED, GOLD, "functional")
    r_mut_task, _ = reward_fn(MUTATED, GOLD, "task")
    assert r_mut == pytest.approx(r_mut_task) and r_mut_task > 0
    assert p_mut["functional"]["status"] == "mismatch"
    # the _noc3 variant follows the same rule
    r_noc3, _ = reward_fn(GOLD, GOLD, "functional_noc3")
    assert r_noc3 == pytest.approx(r_gold)


@pytest.mark.docker
def test_cache_hit_is_fast(tmp_cache):
    first = fr.functional_match(GOLD, GOLD)
    assert first["status"] == "match" and not first.get("cached")
    t0 = time.perf_counter(); second = fr.functional_match(GOLD, GOLD); dt = time.perf_counter() - t0
    assert second["match"] is True and second.get("cached") is True and dt < 0.05
    # sqlite hit after the in-memory cache is dropped (as a fresh process would see it)
    fr.clear_memory_cache()
    t0 = time.perf_counter(); third = fr.functional_match(GOLD, GOLD); dt = time.perf_counter() - t0
    assert third["match"] is True and third.get("cached") is True and dt < 0.05


@pytest.mark.docker
def test_parallel_rewards(tmp_cache):
    texts = [GOLD, MUTATED, WRONG_SIG, LINALG_GOLD, LINALG_MUTATED] * 2
    golds = [GOLD, GOLD, GOLD, LINALG_GOLD, LINALG_GOLD] * 2
    out = spine.rewards_parallel(texts, golds, "functional", workers=6)
    statuses = [p["functional"]["status"] for _, p in out]
    assert statuses == ["match", "mismatch", "signature_mismatch", "match", "mismatch"] * 2, statuses


# ---------------- fast (no container): existing kinds unchanged ----------------

def _patch_verify(monkeypatch, rc: int):
    monkeypatch.setattr(spine, "verify", lambda text, *a, **k: {"returncode": rc})


def test_task_kind_unchanged(monkeypatch):
    _patch_verify(monkeypatch, 0)
    r, parts = reward_fn(GOLD, GOLD, "task")
    assert r == 1.0
    assert parts == {"parse": True, "scope": True, "verify": True, "sig": 1.0, "op_jaccard": 1.0}
    r, parts = reward_fn(WRONG_SIG, GOLD, "task")
    assert r == 0.0 and parts["sig"] == 0.0 and parts["op_jaccard"] == 1.0
    r, parts = reward_fn(GOLD, GOLD, "verify")
    assert (r, parts) == (1.0, {"parse": True, "scope": True, "verify": True})
    _patch_verify(monkeypatch, 1)
    r, parts = reward_fn(GOLD, GOLD, "task")
    assert (r, parts) == (0.0, {"parse": True, "scope": True, "verify": False, "sig": 0.0, "op_jaccard": 0.0})


def test_functional_kind_skips_trial_when_not_verified(monkeypatch):
    _patch_verify(monkeypatch, 1)
    calls = []
    monkeypatch.setattr(spine, "functional_match", lambda *a, **k: calls.append(a) or {"match": True, "status": "match", "dt": 0.0})
    r, parts = reward_fn(GOLD, GOLD, "functional")
    assert r == 0.0 and calls == []
    assert parts["functional"] == {"match": False, "status": "not_verified", "dt": 0.0}
    _patch_verify(monkeypatch, 0)
    r, parts = reward_fn(GOLD, GOLD, "functional")
    assert r == 2.0 and len(calls) == 1 and parts["functional"]["status"] == "match"
