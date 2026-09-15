"""Unit tests for error categorization regex rules."""
from __future__ import annotations

from sci.eval.error_categories import categorize, summarize, categorize_rows


def test_type_error() -> None:
    msg = "error: 'arith.addi' op requires all operands to have the same type"
    assert categorize(msg) == "type"


def test_arity_error() -> None:
    msg = "error: expected 2 operands but found 3"
    assert categorize(msg) == "arity"


def test_dialect_misuse() -> None:
    msg = "error: operation 'foo.bar' not registered in dialect"
    assert categorize(msg) == "dialect_misuse"


def test_syntax_error() -> None:
    msg = "error: expected '('"
    assert categorize(msg) == "syntax"


def test_unknown_falls_back_to_other() -> None:
    assert categorize("something totally unrelated") == "other"
    assert categorize("") == "other"


def test_summarize_groups_by_cell() -> None:
    rows = [
        {"model": "M", "constraint": "c1", "dialect": "arith", "passed": True, "verify_stderr": ""},
        {"model": "M", "constraint": "c1", "dialect": "arith", "passed": False,
         "verify_stderr": "error: expected 2 operands but found 3"},
    ]
    cat = categorize_rows(rows)
    summary = summarize(cat)
    assert "M|c1|arith" in summary
    counts = summary["M|c1|arith"]
    assert counts["passed"] == 1
    assert counts["arity"] == 1
