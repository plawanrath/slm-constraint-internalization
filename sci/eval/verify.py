"""SQLite-backed cache for `mlir-opt --verify` results.

Eval runs thousands of identical verifications (same generation retried across
seeds, same gold sample used as sanity check). Without a cache, each call shells
out to `docker exec`. With this cache, cold verify calls go through the container;
warm calls hit sqlite.

Key: sha256 of (mlir_text, extra_flags).
Value: (returncode, stdout, stderr, duration_ms, timestamp).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "results" / "verify_cache.sqlite"
MLIR_OPT = REPO_ROOT / "scripts" / "env" / "bin" / "mlir-opt"


def _key(mlir_text: str, flags: Iterable[str]) -> str:
    h = hashlib.sha256()
    h.update(mlir_text.encode("utf-8"))
    for f in flags:
        h.update(b"\x00")
        h.update(f.encode("utf-8"))
    return h.hexdigest()


class VerifyCache:
    def __init__(self, path: Path = DEFAULT_CACHE):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS verify (
                key TEXT PRIMARY KEY,
                returncode INTEGER NOT NULL,
                stdout TEXT NOT NULL,
                stderr TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                ts REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, mlir_text: str, flags: tuple[str, ...]) -> dict | None:
        k = _key(mlir_text, flags)
        cur = self._conn.execute(
            "SELECT returncode, stdout, stderr, duration_ms, ts FROM verify WHERE key = ?",
            (k,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "returncode": row[0],
            "stdout": row[1],
            "stderr": row[2],
            "duration_ms": row[3],
            "ts": row[4],
            "cached": True,
        }

    def put(self, mlir_text: str, flags: tuple[str, ...], result: dict) -> None:
        k = _key(mlir_text, flags)
        self._conn.execute(
            "INSERT OR REPLACE INTO verify (key, returncode, stdout, stderr, duration_ms, ts) VALUES (?, ?, ?, ?, ?, ?)",
            (
                k,
                result["returncode"],
                result["stdout"],
                result["stderr"],
                result["duration_ms"],
                result["ts"],
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def verify(
    mlir_text: str,
    flags: tuple[str, ...] = ("--verify-diagnostics",),
    cache: VerifyCache | None = None,
    timeout: float = 30.0,
) -> dict:
    """Run `mlir-opt <flags>` on `mlir_text`, return structured result.

    Cache hits are O(hash). Cache misses shell into the Docker container.
    """
    own = cache is None
    if own:
        cache = VerifyCache()
    try:
        hit = cache.get(mlir_text, flags)
        if hit is not None:
            return hit
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(MLIR_OPT), *flags],
                input=mlir_text,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            rc = -1
            out = ""
            err = f"TIMEOUT after {timeout}s: {e}"
        duration_ms = int((time.perf_counter() - t0) * 1000)
        result = {
            "returncode": rc,
            "stdout": out,
            "stderr": err,
            "duration_ms": duration_ms,
            "ts": time.time(),
            "cached": False,
        }
        cache.put(mlir_text, flags, result)
        return result
    finally:
        if own:
            cache.close()


def passes(mlir_text: str, cache: VerifyCache | None = None) -> bool:
    """Convenience: True iff `mlir-opt --verify-diagnostics` exits 0."""
    return verify(mlir_text, cache=cache)["returncode"] == 0


if __name__ == "__main__":
    # Smoke test: a trivial MLIR module should verify.
    sample = """module {
  func.func @main() -> i32 {
    %0 = arith.constant 42 : i32
    return %0 : i32
  }
}
"""
    r = verify(sample)
    print(json.dumps(r, indent=2))
