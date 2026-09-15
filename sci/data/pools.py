"""Evaluation prompt pools.

Static pools live in `data/pools/*.jsonl` (one row per prompt:
`{prompt_id, dialect, source, nl, gold_mlir}`; `gold_mlir` is null for mined
rows). They are test-only and never used for training or model selection.

Construction (reproducible from raw sources with `build_pools`):
  * `arith_func_200`: the 150 MLIR-Spec prompts in id order, then mined MLIR
    test-suite cases (`sci/data/mine_tests.py`) shuffled with `random.Random(0)`
    and appended until n=200, using each case's weak natural-language stub.
  * `linalg_125`: the 30 Linalg-Spec prompts, then mined cases filtered to
    memref-only linalg named ops within `TARGET_LINALG_OPS`, same shuffle, until
    n=125.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POOL_DIR = REPO / "data" / "pools"

TARGET_LINALG_OPS = {"matmul", "matvec", "fill", "copy", "transpose", "broadcast",
                     "add", "sub", "mul", "div", "exp", "abs"}


def load_pool(name: str) -> list[dict]:
    path = POOL_DIR / f"{name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _spec_rows(bench_jsonl: Path, dialect: str) -> list[dict]:
    rows = []
    for l in bench_jsonl.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        rows.append({"dialect": dialect, "source": f"spec:{r['id']}", "nl": r["nl"],
                     "gold_mlir": r["mlir"]})
    return rows


def build_pools(spec_arith: Path, spec_linalg: Path, mined_jsonl: Path,
                seed: int = 0, n_arith: int = 200, n_linalg: int = 125) -> dict[str, list[dict]]:
    lines = mined_jsonl.read_text().splitlines()

    arith = _spec_rows(spec_arith, "arith+func")
    rng = random.Random(seed); shuffled = list(lines); rng.shuffle(shuffled)
    for l in shuffled:
        if len(arith) >= n_arith:
            break
        r = json.loads(l); nl = r.get("weak_nl", "").strip()
        if nl:
            arith.append({"dialect": "arith+func", "source": f"mined:{r.get('source_path','')}#{r.get('case_idx','')}",
                          "nl": nl, "gold_mlir": None})

    linalg = _spec_rows(spec_linalg, "linalg")
    rng = random.Random(seed); shuffled = list(lines); rng.shuffle(shuffled)
    for l in shuffled:
        if len(linalg) >= n_linalg:
            break
        r = json.loads(l)
        if "linalg." not in r.get("mlir", ""):
            continue
        ops = set(m.group(1) for m in re.finditer(r"linalg\.(\w+)", r["mlir"]))
        if not ops or not (ops <= TARGET_LINALG_OPS):
            continue
        if "tensor<" in r["mlir"]:
            continue
        nl = r.get("weak_nl", "").strip()
        if nl:
            linalg.append({"dialect": "linalg", "source": f"mined:{r.get('source_path','')}#{r.get('case_idx','')}",
                           "nl": nl, "gold_mlir": None})

    for pool in (arith, linalg):
        for i, row in enumerate(pool):
            row["prompt_id"] = i
    return {"arith_func_200": arith[:n_arith], "linalg_125": linalg[:n_linalg]}


def write_pools(pools: dict[str, list[dict]], out_dir: Path = POOL_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in pools.items():
        (out_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
