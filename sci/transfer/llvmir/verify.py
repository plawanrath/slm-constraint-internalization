"""`llvm-as` verification of textual LLVM IR, sqlite-cached.

Same subprocess + cache pattern as `sci.eval.verify` (whose `VerifyCache` class is
reused with a separate database, `results/verify_cache_llvm.sqlite`). `llvm-as`
parses the module and runs the IR verifier; exit code 0 iff the text is
well-formed LLVM IR. The host wrapper in `scripts/env/bin/` execs into the
long-lived `sci-llvm` container.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from sci.eval.verify import REPO_ROOT, VerifyCache

LLVM_CACHE = REPO_ROOT / "results" / "verify_cache_llvm.sqlite"
LLVM_AS = REPO_ROOT / "scripts" / "env" / "bin" / "llvm-as"
DEFAULT_FLAGS: tuple[str, ...] = ("-o", "/dev/null", "-")

# Diagnostic buckets on `llvm-as` stderr (per-layer violation flags in the runner).
SCOPE_RE = re.compile(
    r"(use of undefined value|does not dominate all uses|PHINode should have one entry|"
    r"PHI node entries do not match predecessors|PHI nodes not grouped|"
    r"multiple definition of local value|only PHI nodes may reference their own value|"
    r"PHI node has multiple entries|Instruction does not dominate)",
    re.I,
)
TYPE_RE = re.compile(
    r"(defined with type .* but expected|invalid operand type|both operands .* same type|"
    r"expected (?:icmp|fcmp) predicate|floating point constant invalid for type|"
    r"integer constant must have integer type|constant expression type mismatch|"
    r"invalid cast opcode|cast .* invalid|type mismatch|must be (?:integer|floating.point)|"
    r"invalid type for|explicit pointee type doesn't match|stored value and pointer type|"
    r"expected .* type|does not match|is not a valid .* type|requires .* type)",
    re.I,
)


def verify(text: str, flags: tuple[str, ...] = DEFAULT_FLAGS, cache: VerifyCache | None = None,
           timeout: float = 30.0) -> dict:
    """Run `llvm-as <flags>` on `text`; returns {returncode, stdout, stderr, duration_ms, ts, cached}."""
    own = cache is None
    if own:
        cache = VerifyCache(LLVM_CACHE)
    try:
        hit = cache.get(text, flags)
        if hit is not None:
            return hit
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([str(LLVM_AS), *flags], input=text, text=True, capture_output=True,
                                  timeout=timeout)
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            rc, out, err = -1, "", f"TIMEOUT after {timeout}s: {e}"
        result = {"returncode": rc, "stdout": out, "stderr": err,
                  "duration_ms": int((time.perf_counter() - t0) * 1000), "ts": time.time(), "cached": False}
        if rc != 127:  # a missing container is not a property of the text; do not cache it
            cache.put(text, flags, result)
        return result
    finally:
        if own:
            cache.close()


def passes(text: str, cache: VerifyCache | None = None) -> bool:
    return verify(text, cache=cache)["returncode"] == 0


def categorize(stderr: str) -> str:
    """Coarse bucket of an `llvm-as` failure: scope | type | syntax."""
    if SCOPE_RE.search(stderr or ""):
        return "scope"
    if TYPE_RE.search(stderr or ""):
        return "type"
    return "syntax"
