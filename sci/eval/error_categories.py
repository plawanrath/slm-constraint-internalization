"""Categorize mlir-opt --verify failures into {type, arity, dialect-misuse, syntax, other}.

This powers the "Error-category analysis" ablation (design §5.3 item 4): show
that C2 specifically collapses the {type, arity} buckets.

Usage:
    python -m eval.error_categories \
        --matrix results/matrix.jsonl \
        --out results/error_categories.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CATEGORIES = ("type_ssa", "type", "arity", "dialect_misuse", "syntax", "other")


@dataclass
class CategoryRule:
    category: str
    pattern: re.Pattern
    priority: int = 0


# Ordered: earlier rules win ties. Tuned against typical mlir-opt diagnostics.
#
# Design note: type_ssa captures cross-SSA-value type mismatches ("use of
# value '%X' expects different type than prior uses"), which is what C3 is
# meant to address. type captures within-op type mismatches (operands or
# operand-vs-result inside one op), which is what C2 is meant to address.
# Splitting these lets the error-category figure show that C2 collapses
# type and C3 collapses type_ssa, distinctly.
_RULES: tuple[CategoryRule, ...] = (
    CategoryRule(
        category="type_ssa",
        pattern=re.compile(
            r"(use of value .*? expects different type than prior uses|"
            r"SSA value .*? has different type|"
            r"block argument .*? has different type|"
            r"prior use here)",
            re.IGNORECASE,
        ),
        priority=12,
    ),
    CategoryRule(
        category="type",
        pattern=re.compile(
            r"(type mismatch|expected .*? but got|incompatible (?:operand|result) types|"
            r"mismatched operand types|op requires all (?:operands|results) to have the same type|"
            r"op requires the same (?:element )?type|operand type .*? does not match|"
            r"result #\d+ type .*? does not match|"
            r"failed to verify that .*?(types? match|same type))",
            re.IGNORECASE,
        ),
        priority=10,
    ),
    CategoryRule(
        category="arity",
        pattern=re.compile(
            r"(expected \d+ (?:operand|result|argument)s?|op has \d+ operands? but expected|"
            r"too few operands|too many operands|incorrect number of (?:operand|result|argument)s?|"
            r"number of (?:operand|result|argument)s? does not match)",
            re.IGNORECASE,
        ),
        priority=9,
    ),
    CategoryRule(
        category="dialect_misuse",
        pattern=re.compile(
            r"(unregistered operation|unknown operation|operation .*? not (?:registered|defined) in dialect|"
            r"use of unregistered dialect|operation has no verifier)",
            re.IGNORECASE,
        ),
        priority=8,
    ),
    CategoryRule(
        category="syntax",
        pattern=re.compile(
            r"(expected '[^']+'|custom op .*? parsing failed|error: expected|"
            r"parse error|expected (?:SSA|attribute|type|operation))",
            re.IGNORECASE,
        ),
        priority=5,
    ),
)


def categorize(stderr: str) -> str:
    if not stderr.strip():
        return "other"
    best: tuple[int, str] | None = None
    for rule in _RULES:
        if rule.pattern.search(stderr):
            if best is None or rule.priority > best[0]:
                best = (rule.priority, rule.category)
    return best[1] if best else "other"


def categorize_rows(rows: Iterable[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("passed"):
            out.append({**row, "category": "passed"})
        else:
            out.append({**row, "category": categorize(row.get("verify_stderr", ""))})
    return out


def summarize(categorized: list[dict]) -> dict:
    by_cell: dict[tuple, Counter] = {}
    for row in categorized:
        key = (row.get("model", ""), row.get("constraint", ""), row.get("dialect", ""))
        by_cell.setdefault(key, Counter())[row["category"]] += 1
    return {
        f"{m}|{c}|{d}": dict(counts)
        for (m, c, d), counts in by_cell.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/error_categories.json"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.matrix.read_text().splitlines() if l.strip()]
    cat = categorize_rows(rows)
    summary = summarize(cat)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[error-cats] wrote → {args.out}")


if __name__ == "__main__":
    main()
