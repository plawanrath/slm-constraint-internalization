"""StableHLO verify via iree-compile.

The pinned `mlir-opt` lacks the StableHLO dialect. `iree-compile` (installed
via `pip install iree-compiler`) ships StableHLO support and can verify a
fragment by compiling it to IREE's `input` stage — this parses the
module, type-checks all ops, and confirms the dialect is well-formed
without running the full codegen pipeline.

Usage:
  from sci.eval.verify_stablehlo import verify_stablehlo
  r = verify_stablehlo(mlir_text)
  # r["returncode"] == 0 iff verify-valid
"""
from __future__ import annotations

import subprocess
from pathlib import Path

IREE_COMPILE = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "iree-compile"


def verify_stablehlo(mlir_text: str, timeout: float = 20.0) -> dict:
    """Return {'returncode': int, 'stdout': str, 'stderr': str}.

    returncode == 0 iff iree-compile's `--compile-to=input` stage accepts
    the fragment (parse + type-check + dialect validation OK) AND the
    fragment contains at least one non-empty function body.

    Phase F fix: iree-compile treats an empty stdin as a valid empty
    module (stdout = `module {\\n}`, returncode 0). That false-positive
    inflated our Day-32 and Day-42 "100% verify" numbers for C1-
    constrained generations that produced empty output. We now require
    the input to contain a `func.func @` definition — a minimal
    non-triviality check that rejects empty / whitespace-only inputs.
    """
    if "func.func @" not in mlir_text:
        return {
            "returncode": -3,
            "stdout": "",
            "stderr": "phase-F non-triviality check: no func.func definition in input",
        }
    try:
        r = subprocess.run(
            [
                str(IREE_COMPILE),
                "--iree-input-type=stablehlo",
                "--compile-to=input",
                "-",
            ],
            input=mlir_text, capture_output=True, text=True, timeout=timeout,
        )
        return {
            "returncode": r.returncode,
            "stdout": r.stdout[:2000],
            "stderr": r.stderr[:2000],
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": "timeout"}
    except FileNotFoundError:
        return {"returncode": -2, "stdout": "", "stderr": "iree-compile not found"}
