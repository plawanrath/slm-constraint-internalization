"""Unit tests for sci/eval/functional.py.

Marked `docker` because the actual exec smoke-test requires the
pinned LLVM container. The fast subset (function-name detection +
wrapper construction) runs without docker.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sci.eval.functional import (
    detect_first_function,
    strip_outer_module,
    _build_arith_wrapper,
    DISPATCH,
    run_arith,
)

REPO = Path(__file__).resolve().parents[1]
REFERENCES = json.loads(
    (REPO / "data/functional/references.json").read_text()
)["references"]


def test_detect_first_function_simple():
    src = "module {\n  func.func @add(%a: i32, %b: i32) -> i32 {\n    return %a : i32\n  }\n}"
    assert detect_first_function(src) == ("add", "%a: i32, %b: i32", "i32")


def test_detect_first_function_skips_private():
    src = (
        "module {\n"
        "  func.func private @printI64(i64)\n"
        "  func.func @add(%a: i32, %b: i32) -> i32 { return %a : i32 }\n"
        "}"
    )
    assert detect_first_function(src) == ("add", "%a: i32, %b: i32", "i32")


def test_strip_outer_module():
    src = "module {\n  func.func @f() { return }\n}\n"
    inner = strip_outer_module(src)
    assert "module" not in inner
    assert "func.func @f()" in inner


def test_strip_outer_module_no_module():
    src = "func.func @f() { return }"
    assert strip_outer_module(src) == src


def test_arith_wrapper_construction_addi():
    body = "  func.func @add(%a: i32, %b: i32) -> i32 {\n    %0 = arith.addi %a, %b : i32\n    return %0 : i32\n  }"
    inputs = [{"type": "i32", "value": 7}, {"type": "i32", "value": 3}]
    wrapper = _build_arith_wrapper(body, "add", inputs, "i32")
    assert "func.func private @printI64(i64)" in wrapper
    assert "arith.constant 7 : i32" in wrapper
    assert "arith.constant 3 : i32" in wrapper
    assert "func.call @add(%i0, %i1)" in wrapper
    assert "arith.extsi" in wrapper


def test_references_count_and_dialect_mix():
    assert len(REFERENCES) == 30
    by_dialect = {"arith+func": 0, "linalg+memref": 0, "stablehlo": 0}
    for r in REFERENCES:
        by_dialect[r["dialect"]] = by_dialect.get(r["dialect"], 0) + 1
    assert by_dialect == {"arith+func": 10, "linalg+memref": 10, "stablehlo": 10}


def test_references_have_required_fields():
    for r in REFERENCES:
        assert "id" in r and "dialect" in r and "source_id" in r
        if r["dialect"] == "arith+func":
            assert "inputs" in r and "result_type" in r and "expected_stdout_regex" in r
        elif r["dialect"] == "linalg+memref":
            assert "memref_inputs" in r and "memref_print" in r and "expected_stdout_regex" in r
        elif r["dialect"] == "stablehlo":
            assert "iree_inputs" in r and "expected_output_pattern" in r


@pytest.mark.docker
def test_runner_exec_canonical_arith_add():
    """Smoke test: feed the canonical reference MLIR for F01 through the runner;
    expect all 4 columns true."""
    canonical = (
        "module {\n"
        "  func.func @add(%a: i32, %b: i32) -> i32 {\n"
        "    %0 = arith.addi %a, %b : i32\n"
        "    return %0 : i32\n"
        "  }\n"
        "}"
    )
    ref = next(r for r in REFERENCES if r["id"] == "F01_arith_add")
    res = run_arith(canonical, ref)
    assert res["verify_valid"] is True, res
    assert res["lower_ok"] is True, res
    assert res["exec_ok"] is True, res
    assert res["output_match"] is True, res
