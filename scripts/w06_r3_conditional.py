"""Scope residual conditioned on parse-valid free output (companion to w06_framing_gate.py).

R3 as defined for the residual tables is gated on parsing: a free output that does not parse has v3 = False, so a student
whose free decoding parses less often shows a *lower* free scope-failure rate than its masked self and R3 can be
negative. This script pairs, per prompt, the free and full-stack outputs of every trained checkpoint and reports
v3 free vs v3 under C1+C2+C3 on the prompts whose free output parses, with paired bootstrap CIs.
  python scripts/w06_r3_conditional.py [--ladder results/w04_residuals/ladder.jsonl] [--out results/w06_gate/r3_conditional.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sci.eval.residual import violations
from sci.eval.stats import paired_bootstrap_diff

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="results/w04_residuals/ladder.jsonl")
    ap.add_argument("--out", default="results/w06_gate/r3_conditional.json")
    ap.add_argument("--tags", default=None, help="comma-separated; default: every non-ablation tag")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (REPO / args.ladder).read_text().splitlines() if l.strip()]
    tags = args.tags.split(",") if args.tags else sorted({r["model"] for r in rows if "__" not in r["model"]})
    out = {}
    for t in tags:
        for pool in sorted({r["pool"] for r in rows if r["model"] == t}):
            by: dict[int, dict] = {}
            for r in rows:
                if r["model"] == t and r["pool"] == pool and r["constraint"] in ("none", "c1_c2_c3"):
                    by.setdefault(r["prompt_id"], {})[r["constraint"]] = violations(r["generated"])
            ids = [i for i, v in by.items() if "none" in v and "c1_c2_c3" in v and not v["none"]["v1"]]
            if not ids:
                continue
            a = [int(by[i]["none"]["v3"]) for i in ids]; b = [int(by[i]["c1_c2_c3"]["v3"]) for i in ids]
            d = paired_bootstrap_diff(a, b)
            out[f"{t}|{pool}"] = {"n_parse_free": len(ids), "v3_free": round(sum(a) / len(ids), 4), "v3_full": round(sum(b) / len(ids), 4),
                                  "r3_pp": round(100 * d.point, 1), "ci95": [round(100 * d.ci_low, 1), round(100 * d.ci_high, 1)]}
            e = out[f"{t}|{pool}"]
            print(f"{t:40s} {pool:15s} n={e['n_parse_free']:3d} v3 free={e['v3_free']:.3f} full={e['v3_full']:.3f} R3|parse={e['r3_pp']:+.1f} [{e['ci95'][0]:+.1f},{e['ci95'][1]:+.1f}]")
    (REPO / args.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO / args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
