"""L3 miner — walk llvm-project/mlir/test/**/*.mlir, use pathname + FileCheck
comments as weak natural-language descriptions.

Emits (weak_nl, mlir) pairs to data/processed/l3_tests.jsonl.

Usage:
    python -m data.pipelines.l3_tests.mine \
        --llvm-src /path/to/llvm-project \
        --out data/processed/l3_tests.jsonl \
        --target-dialects arith,func,linalg,memref
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_FILECHECK_RE = re.compile(r"//\s*(CHECK|CHECK-LABEL|CHECK-NEXT|CHECK-SAME)[-:](.*)")
_RUN_LINE_RE = re.compile(r"//\s*RUN:")


def _strip_test_directives(src: str) -> str:
    """Remove RUN, CHECK, and the `// -----` splitter markers so the residual
    is closer to real MLIR. Does NOT split — see _split_test_cases for that."""
    out_lines = []
    for line in src.splitlines():
        if _RUN_LINE_RE.search(line):
            continue
        if _FILECHECK_RE.match(line.strip()):
            continue
        if line.strip() == "// -----":
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip() + "\n"


def _split_test_cases(src: str) -> list[str]:
    """MLIR test files use `// -----` (on its own line) to separate independent
    test cases within one file. Split the file, strip RUN/CHECK from each case,
    and return non-empty cases."""
    cases = re.split(r"^\s*//\s*-----\s*$", src, flags=re.MULTILINE)
    out: list[str] = []
    for case in cases:
        cleaned = _strip_test_directives(case)
        if cleaned.strip():
            out.append(cleaned)
    return out


def _extract_filecheck_hints(src: str) -> list[str]:
    hints = []
    for line in src.splitlines():
        m = _FILECHECK_RE.match(line.strip())
        if m:
            hints.append(m.group(2).strip())
    return hints


def _dialects_mentioned(src: str) -> set[str]:
    return set(re.findall(r"([a-z_][a-z_0-9]*)\.[a-z_]", src))


def _weak_nl_from_path(path: Path, llvm_root: Path) -> str:
    rel = path.relative_to(llvm_root).as_posix()
    # Example: mlir/test/Dialect/Arith/canonicalize.mlir → "arith canonicalize"
    stem = path.stem.replace("_", " ").replace("-", " ")
    parts = [p.lower() for p in rel.split("/") if p]
    dialect_hints = [p for p in parts if p in {"arith", "func", "linalg", "memref", "scf", "affine"}]
    return " ".join([*dialect_hints, stem])


def _iter_test_files(llvm_root: Path):
    for path in (llvm_root / "mlir" / "test").rglob("*.mlir"):
        # Skip invalid-by-design tests (they won't verify).
        if "invalid" in path.name.lower():
            continue
        yield path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvm-src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed/mlir_tests.jsonl"))
    ap.add_argument("--target-dialects", default="arith,func,linalg,memref")
    ap.add_argument(
        "--require-verify",
        action="store_true",
        help="Only keep pairs that pass mlir-opt --verify (slower; costs Docker).",
    )
    args = ap.parse_args()

    target = {s.strip() for s in args.target_dialects.split(",") if s.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    verifier = None
    if args.require_verify:
        from sci.eval.verify import verify as verify_fn
        verifier = verify_fn

    n = 0
    n_files = 0
    with args.out.open("w") as f:
        for path in _iter_test_files(args.llvm_src):
            try:
                src = path.read_text()
            except UnicodeDecodeError:
                continue
            n_files += 1
            cases = _split_test_cases(src)
            for case_idx, case in enumerate(cases):
                dialects = _dialects_mentioned(case)
                if not (dialects & target):
                    continue
                if verifier is not None and verifier(case)["returncode"] != 0:
                    continue
                weak_nl = _weak_nl_from_path(path, args.llvm_src)
                f.write(json.dumps({
                    "source": "l3_tests",
                    "weak_nl": weak_nl,
                    "mlir": case,
                    "filecheck_hints": _extract_filecheck_hints(src),
                    "dialects": sorted(dialects),
                    "source_path": str(path.relative_to(args.llvm_src)),
                    "case_idx": case_idx,
                }) + "\n")
                n += 1
            if n_files % 500 == 0:
                print(f"[l3] scanned {n_files} files, kept {n} cases", file=sys.stderr)
    print(f"[l3] wrote {n} pairs from {n_files} files → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
