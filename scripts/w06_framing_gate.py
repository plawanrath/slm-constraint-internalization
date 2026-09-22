"""G2 framing gate: assemble per-layer residuals for every evaluated checkpoint and the base models,
compare against the pre-registered thresholds, and print the decision table.
  python scripts/w06_framing_gate.py [--residuals results/w04_residuals/residuals.json] [--table1 results/w02_table1/ladder.jsonl]
Writes results/w06_gate/gate.json and gate.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sci.eval.residual import residuals_from_jsonl

REPO = Path(__file__).resolve().parents[1]
LAYERS = (("v1", "C1 syntax"), ("v2", "C2 type domain"), ("v3", "C3 scope"), ("v4", "C4 shape/interface"))


def internalized(entry: dict) -> bool:
    lo, hi = entry["ci95"]
    return lo <= 0.0 <= hi and entry["pp"] <= 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--residuals", default="results/w04_residuals/residuals.json")
    ap.add_argument("--table1", default="results/w02_table1/ladder.jsonl")
    ap.add_argument("--out", default="results/w06_gate")
    args = ap.parse_args()
    res = json.loads((REPO / args.residuals).read_text()) if (REPO / args.residuals).exists() else {}
    # base models from Table 1
    t1 = REPO / args.table1
    if t1.exists():
        rows = [json.loads(l) for l in t1.read_text().splitlines() if l.strip()]
        for model in sorted({r["model"] for r in rows}):
            res[f"{model}_base"] = {pool: residuals_from_jsonl(t1, model, pool) for pool in ("arith_func_200", "linalg_125")}
    lines = ["| tag | pool | " + " | ".join(f"free {k} / residual {k} (pp, CI)" for k, _ in LAYERS) + " | free verify | masked verify |", "|---" * 7 + "|"]
    gate = {}
    for tag in sorted(res):
        for pool, r in res[tag].items():
            if "rates" not in r or "none" not in r["rates"]:
                continue
            cells = []
            for k, _ in LAYERS:
                free = r["rates"]["none"][k]; rr = r["residual_pp"].get(k)
                cells.append(f"{100*free:.1f} / {rr['pp']:+.1f} [{rr['ci95'][0]:+.1f},{rr['ci95'][1]:+.1f}]{' ✓' if internalized(rr) else ''}" if rr else f"{100*free:.1f} / –")
            fv = r["rates"]["none"]["verify"]; mv = r["rates"].get("c1_c2_c3", {}).get("verify", float("nan"))
            lines.append(f"| {tag} | {pool} | " + " | ".join(cells) + f" | {100*fv:.1f} | {100*mv:.1f} |")
            gate[f"{tag}|{pool}"] = {k: {"free": r["rates"]["none"][k], "residual": r["residual_pp"].get(k),
                                         "internalized": internalized(r["residual_pp"][k]) if k in r["residual_pp"] else None} for k, _ in LAYERS}
    out = REPO / args.out; out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(gate, indent=2))
    md = "# G2 framing gate\n\nInternalized := residual 95% CI contains 0 and point ≤ 2pp. ✓ marks internalized cells.\n\n" + "\n".join(lines) + "\n"
    (out / "gate.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
