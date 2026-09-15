"""Functional (gold-differential) scoring of ladder cells.

For each (model, constraint) in a ladder JSONL, converts the spec-prompt rows (prompt_id < n_spec)
into the differential harness's candidate format and runs it. Writes
results/<out>/<model>/<constraint>/{gold_gate.jsonl,differential.jsonl,summary.json} and an
aggregate results/<out>/functional_summary.json with composed rates
(functional matches / verify-valid, and / all gate-passed prompts).
  python scripts/w02_functional.py --ladder results/w02_table1/ladder.jsonl --out results/w02_functional
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv/bin/python")
POOL_TO_DIALECT = {"arith_func_200": ("arith+func", "arith", 150), "linalg_125": ("linalg", "linalg", 30)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=None, help="comma list of model tags (default all)")
    ap.add_argument("--constraints", default="none,c1,c1_c2,c1_c2_c3")
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.ladder).read_text().splitlines() if l.strip()]
    models = sorted({r["model"] for r in rows}) if not args.models else args.models.split(",")
    out_root = REPO / args.out; out_root.mkdir(parents=True, exist_ok=True)
    agg_path = out_root / "functional_summary.json"
    agg = json.loads(agg_path.read_text()) if agg_path.exists() else {}
    for model in models:
        for constraint in args.constraints.split(","):
            cell_dir = out_root / model / constraint
            if (cell_dir / "summary.json").exists():
                summ = json.loads((cell_dir / "summary.json").read_text())
            else:
                cand = [r for r in rows if r["model"] == model and r["constraint"] == constraint]
                if not cand:
                    continue
                cand_path = cell_dir / "candidates.jsonl"; cell_dir.mkdir(parents=True, exist_ok=True)
                with cand_path.open("w") as f:
                    for r in cand:
                        d, dkey, n_spec = POOL_TO_DIALECT[r["pool"]]
                        if r["prompt_id"] >= n_spec:
                            continue
                        f.write(json.dumps({"model": model, "dialect": d, "seed": r.get("seed", 0), "prompt_id": r["prompt_id"],
                                            "generated": r["generated"], "verify_valid": r["verify_valid"]}) + "\n")
                print(f"[functional] {model}/{constraint}", file=sys.stderr)
                subprocess.run([PY, "-m", "sci.eval.differential", "--out-dir", str(cell_dir), "--candidates", str(cand_path),
                                "--dialects", "arith,linalg"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                summ = json.loads((cell_dir / "summary.json").read_text())
            cell = {}
            for dkey, d in summ.get("dialects", {}).items():
                nvv = sum(1 for r in rows if r["model"] == model and r["constraint"] == constraint
                          and POOL_TO_DIALECT[r["pool"]][1] == dkey and r["prompt_id"] < POOL_TO_DIALECT[r["pool"]][2] and r["verify_valid"])
                cell[dkey] = {"gold_gate_pass": d["gold_gate_pass"], "n_verify_valid": nvv, "n_scored": d["n_scored"],
                              "n_match": d["n_functional_match"], "signature_mismatch": d["cand_status_counts"].get("signature_mismatch", 0),
                              "match_over_verify_valid": round(d["n_functional_match"] / nvv, 4) if nvv else None,
                              "match_over_gate_passed": d.get("match_rate_among_gate_passed"),
                              "ci95_gate_passed": d.get("match_ci95_among_gate_passed")}
            agg.setdefault(model, {})[constraint] = cell
            agg_path.write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=1)[:4000])


if __name__ == "__main__":
    main()
