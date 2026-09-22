"""Semantics-carrying reward term: one gold-differential trial per rollout.

`functional_match(text, gold, trial)` reuses the gold-differential harness (`sci.eval.differential`): both the
candidate and its template gold are wrapped in the same driver (seeded random inputs generated from the GOLD
signature), lowered and executed in the pinned LLVM container, and their parsed outputs compared with the
harness's tolerance rule. Both pool dialects go through the same pipeline: `arith+func` scalars and
`linalg` memref kernels (`--convert-linalg-to-loops` is in the harness pipeline and memref arguments are
materialized as `memref.global` data and printed after the call).

Caching (repeated rollouts of the same program are free):
  * gold trial outputs per (sha256(gold), trial), in memory and in sqlite;
  * the full verdict per (sha256(text), sha256(gold), trial), in memory and in sqlite.
The sqlite file is `results/functional_reward_cache.sqlite`; each thread owns its connection (WAL mode,
busy timeout), so `rewards_parallel`'s thread pool can call this concurrently. The function never raises:
anything the harness cannot handle yields status `unsupported` with match False.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

from sci.eval.differential import (
    PIPELINE,
    RUNNER,
    _docker,
    build_driver,
    classify_signature,
    gen_trial,
    outputs_equal,
    parse_first_fn,
    parse_stdout,
    signature_compatible,
    strip_outer_module,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO / "results" / "functional_reward_cache.sqlite"

STATUSES = ("match", "mismatch", "signature_mismatch", "gold_fail", "cand_fail", "unsupported")

_mem_lock = threading.Lock()
_gold_mem: dict[tuple[str, int], dict] = {}          # (gold hash, trial) -> {"ok": bool, "out": parsed | None}
_result_mem: dict[tuple[str, str, int], dict] = {}   # (text hash, gold hash, trial) -> verdict
_tls = threading.local()
_cache_path: Path = DEFAULT_CACHE


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------- sqlite (one connection per thread) ----------------

def set_cache_path(path: Path | None) -> None:
    """Redirect the sqlite cache (tests). Also clears the in-memory caches and per-thread connections."""
    global _cache_path
    _cache_path = Path(path) if path is not None else DEFAULT_CACHE
    clear_memory_cache()
    if getattr(_tls, "conn", None) is not None:
        try:
            _tls.conn.close()
        except Exception:
            pass
    _tls.conn = None; _tls.conn_path = None


def clear_memory_cache() -> None:
    with _mem_lock:
        _gold_mem.clear(); _result_mem.clear()


def _conn() -> sqlite3.Connection | None:
    c = getattr(_tls, "conn", None)
    if c is not None and getattr(_tls, "conn_path", None) == _cache_path:
        return c
    try:
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(_cache_path, timeout=30.0, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE IF NOT EXISTS gold_trial (key TEXT PRIMARY KEY, ok INTEGER NOT NULL, out TEXT NOT NULL, ts REAL NOT NULL)")
        c.execute("CREATE TABLE IF NOT EXISTS verdict (key TEXT PRIMARY KEY, match INTEGER NOT NULL, status TEXT NOT NULL, dt REAL NOT NULL, ts REAL NOT NULL)")
        c.commit()
    except sqlite3.Error:
        c = None
    _tls.conn = c; _tls.conn_path = _cache_path
    return c


def _db_get_gold(key: str) -> dict | None:
    c = _conn()
    if c is None:
        return None
    try:
        row = c.execute("SELECT ok, out FROM gold_trial WHERE key = ?", (key,)).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return {"ok": bool(row[0]), "out": json.loads(row[1]) if row[1] else None}


def _db_put_gold(key: str, rec: dict) -> None:
    c = _conn()
    if c is None:
        return
    try:
        c.execute("INSERT OR REPLACE INTO gold_trial VALUES (?, ?, ?, ?)",
                  (key, int(rec["ok"]), json.dumps(rec["out"]) if rec["out"] is not None else "", time.time()))
        c.commit()
    except sqlite3.Error:
        pass


def _db_get_verdict(key: str) -> dict | None:
    c = _conn()
    if c is None:
        return None
    try:
        row = c.execute("SELECT match, status, dt FROM verdict WHERE key = ?", (key,)).fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else {"match": bool(row[0]), "status": row[1], "dt": float(row[2])}


def _db_put_verdict(key: str, rec: dict) -> None:
    c = _conn()
    if c is None:
        return
    try:
        c.execute("INSERT OR REPLACE INTO verdict VALUES (?, ?, ?, ?, ?)",
                  (key, int(rec["match"]), rec["status"], float(rec["dt"]), time.time()))
        c.commit()
    except sqlite3.Error:
        pass


# ---------------- container lower + run (harness pipeline, bounded timeout) ----------------

def _lower_and_run(driver: str, timeout: float) -> dict:
    """Same pipeline and runner as `sci.eval.differential.lower_and_run`, with a caller-chosen timeout per stage."""
    try:
        r1 = _docker(["mlir-opt"] + PIPELINE, driver, timeout)
    except subprocess.TimeoutExpired:
        return {"exec_ok": False, "stdout": "", "err": "lower_timeout"}
    except OSError as e:
        return {"exec_ok": False, "stdout": "", "err": f"docker:{e}"[:300]}
    if r1.returncode != 0:
        return {"exec_ok": False, "stdout": "", "err": (r1.stderr or "")[:300]}
    try:
        r2 = _docker(RUNNER, r1.stdout, timeout)
    except subprocess.TimeoutExpired:
        return {"exec_ok": False, "stdout": "", "err": "exec_timeout"}
    except OSError as e:
        return {"exec_ok": False, "stdout": "", "err": f"docker:{e}"[:300]}
    if r2.returncode != 0:
        return {"exec_ok": False, "stdout": r2.stdout or "", "err": (r2.stderr or "")[:300]}
    return {"exec_ok": True, "stdout": r2.stdout or "", "err": ""}


def _gold_outputs(gold: str, gold_hash: str, sig: dict, trial: int, timeout: float) -> dict:
    """{"ok": bool, "out": parsed outputs | None} for the gold on trial `trial` (memory → sqlite → container)."""
    mkey = (gold_hash, trial)
    with _mem_lock:
        rec = _gold_mem.get(mkey)
    if rec is not None:
        return rec
    rec = _db_get_gold(f"{gold_hash}:{trial}")
    if rec is None:
        td = gen_trial(sig, gold_hash, trial)
        driver = build_driver(strip_outer_module(gold), sig["name"], sig, td)
        r = _lower_and_run(driver, timeout)
        rec = {"ok": r["exec_ok"], "out": parse_stdout(r["stdout"]) if r["exec_ok"] else None}
        _db_put_gold(f"{gold_hash}:{trial}", rec)
    with _mem_lock:
        _gold_mem[mkey] = rec
    return rec


def _match_uncached(text: str, gold: str, gold_hash: str, trial: int, timeout: float) -> tuple[bool, str]:
    gsig = parse_first_fn(gold)
    reason, _ = classify_signature(gsig)
    if reason:
        return False, "unsupported"
    csig = parse_first_fn(text)
    if csig is None or csig["name"] == "main":
        return False, "signature_mismatch"
    compat, _ = signature_compatible(gsig, csig)
    if not compat:
        return False, "signature_mismatch"
    g = _gold_outputs(gold, gold_hash, gsig, trial, timeout)
    if not g["ok"]:
        return False, "gold_fail"
    td = gen_trial(gsig, gold_hash, trial)
    driver = build_driver(strip_outer_module(text), csig["name"], gsig, td, callee_types=csig["arg_types"])
    r = _lower_and_run(driver, timeout)
    if not r["exec_ok"]:
        return False, "cand_fail"
    ok = outputs_equal(parse_stdout(r["stdout"]), g["out"])
    return ok, ("match" if ok else "mismatch")


def functional_match(text: str, gold: str, trial: int = 0, timeout: float = 20.0) -> dict:
    """One differential trial of `text` against `gold`:
    {"match": bool, "status": match|mismatch|signature_mismatch|gold_fail|cand_fail|unsupported, "dt": seconds}.
    `dt` is the wall time of this call (≈0 on a cache hit). Never raises."""
    t0 = time.perf_counter()
    try:
        th, gh = _sha(text), _sha(gold)
        mkey = (th, gh, trial)
        with _mem_lock:
            hit = _result_mem.get(mkey)
        if hit is None:
            hit = _db_get_verdict(f"{th}:{gh}:{trial}")
            if hit is not None:
                with _mem_lock:
                    _result_mem[mkey] = hit
        if hit is not None:
            return {"match": hit["match"], "status": hit["status"], "dt": time.perf_counter() - t0, "cached": True}
        ok, status = _match_uncached(text, gold, gh, trial, timeout)
        dt = time.perf_counter() - t0
        rec = {"match": bool(ok), "status": status, "dt": dt}
        with _mem_lock:
            _result_mem[mkey] = rec
        _db_put_verdict(f"{th}:{gh}:{trial}", rec)
        return dict(rec)
    except Exception:  # noqa: BLE001 - the reward must never take the training loop down
        return {"match": False, "status": "unsupported", "dt": time.perf_counter() - t0}
