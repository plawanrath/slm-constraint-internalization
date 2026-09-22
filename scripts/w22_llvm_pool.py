"""Build an LLVM IR training pool by lowering the MLIR template golds (optional second-target replication).

For every row of data/pools/{train,dev}_v1.jsonl: lower the gold with `mlir-opt --convert-to-llvm | mlir-translate
--mlir-to-llvmir`, keep the function body, rename numbered values (%3 -> %v3) and insert the `entry:` label the
generation grammar expects, and keep the row only if the result replays under `llvm_gen_c1c2`, passes `llvm-as`,
and passes the dominance-aware scope validator. Rows whose lowering uses intrinsics or forms outside the grammar
(e.g. `llvm.smax`) are dropped. The natural-language request is kept with "MLIR" rewritten to "LLVM IR".

  python scripts/w22_llvm_pool.py [--n-train 2000] [--n-dev 100] [--limit N]

Writes data/pools/train_llvm.jsonl, data/pools/dev_llvm.jsonl ({prompt_id, nl, gold, dialect: "llvmir", source})
and results/w22_llvm_pool/report.json (kept/dropped counts by reason).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LOWER = "mlir-opt --convert-to-llvm | mlir-translate --mlir-to-llvmir"


def lower(mlir: str) -> str | None:
    p = subprocess.run(LOWER, shell=True, input=mlir, capture_output=True, text=True)
    if p.returncode != 0 or "define" not in p.stdout:
        return None
    body = [l for l in p.stdout.splitlines() if l.startswith(("define", "  ", "}")) and not l.strip().startswith(";")]
    if not body or not body[0].startswith("define"):
        return None
    ir = re.sub(r"%(\d+)\b", r"%v\1", "\n".join(body))
    lines = ir.splitlines()
    return "\n".join([lines[0], "entry:"] + lines[1:]) + "\n"


def convert(rows: list[dict], n_keep: int, reasons: Counter) -> list[dict]:
    from sci.transfer.llvmir.grammar import is_parse_valid
    from sci.transfer.llvmir.scope import accept_or_reject
    from sci.transfer.llvmir.verify import passes
    out = []
    for r in rows:
        if len(out) >= n_keep:
            break
        ir = lower(r["mlir"])
        if ir is None:
            reasons["lower_fail"] += 1; continue
        if not is_parse_valid(ir, "llvm_gen_c1c2"):
            reasons["grammar"] += 1; continue
        if not accept_or_reject(ir)[0]:
            reasons["scope"] += 1; continue
        if not passes(ir):
            reasons["llvm_as"] += 1; continue
        nl = re.sub(r"\bMLIR\b", "LLVM IR", r["nl"])
        out.append({"prompt_id": r["prompt_id"], "nl": nl, "gold": ir, "dialect": "llvmir",
                    "source": {"pool": "v1", "family": r.get("family"), "dialect": r.get("dialect")}})
        reasons["kept"] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="scan at most this many source rows per split")
    ap.add_argument("--out-dir", default="data/pools")
    args = ap.parse_args()
    out_dir = REPO / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, n in (("train", args.n_train), ("dev", args.n_dev)):
        src = [json.loads(l) for l in (REPO / "data/pools" / f"{split}_v1.jsonl").read_text().splitlines() if l.strip()]
        if args.limit:
            src = src[: args.limit]
        reasons: Counter = Counter()
        rows = convert(src, n, reasons)
        (out_dir / f"{split}_llvm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        report[split] = {"scanned": sum(reasons.values()), "kept": len(rows), "reasons": dict(reasons)}
        print(f"[llvm-pool] {split}: {report[split]}", file=sys.stderr)
    rd = REPO / "results" / "w22_llvm_pool"; rd.mkdir(parents=True, exist_ok=True)
    (rd / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
