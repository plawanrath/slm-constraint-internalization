"""C4 (shape / interface) residual per scale: the value-dependent layer has no mask, so the only
measurement is v4 on free decoding before training (base model, Table 1 ladder) vs after training
(trained checkpoints, Week-4 ladder), plus the functional signature-mismatch rate from the functional
summaries. Paired by prompt_id within a pool.

  python scripts/w09_c4_residual.py [--table1 results/w02_table1/ladder.jsonl] [--trained results/w04_residuals/ladder.jsonl]
Writes results/w09_c4_residual/{c4.json, c4.md}.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sci.eval.residual import violations
from sci.eval.stats import bootstrap_ci, paired_bootstrap_diff

REPO = Path(__file__).resolve().parents[1]
POOLS = ("arith_func_200", "linalg_125")
_TAG_RE = re.compile(r"^(?P<model>.+?)_(?P<method>ssd|rft|grpo|sft)_s(?P<seed>\d+)$")


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def _v4_by_pid(rows: list[dict], model: str, pool: str, constraint: str) -> dict[int, int]:
    return {r["prompt_id"]: int(violations(r["generated"])["v4"]) for r in rows
            if r["model"] == model and r["pool"] == pool and r["constraint"] == constraint}


def _functional_sig(summary: dict | None, constraint: str, pool: str) -> dict | None:
    """signature_mismatch count over gold-gate-passed prompts from a functional summary entry."""
    if not summary:
        return None
    key = "arith" if pool.startswith("arith") else "linalg"
    cell = summary.get(constraint, {}).get(key)
    if not cell:
        return None
    n = cell["gold_gate_pass"]; k = cell["signature_mismatch"]
    ci = bootstrap_ci([1] * k + [0] * (n - k))
    return {"n": n, "signature_mismatch": k, "rate": round(k / n, 4), "ci95": [round(ci.ci_low, 4), round(ci.ci_high, 4)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table1", default="results/w02_table1/ladder.jsonl")
    ap.add_argument("--trained", default="results/w04_residuals/ladder.jsonl")
    ap.add_argument("--functional-base", default="results/w02_functional/functional_summary.json")
    ap.add_argument("--functional-trained", default="results/w04_functional/functional_summary.json")
    ap.add_argument("--out", default="results/w09_c4_residual")
    args = ap.parse_args()
    base_rows = _rows(REPO / args.table1); tr_rows = _rows(REPO / args.trained)
    fb = json.loads((REPO / args.functional_base).read_text()) if (REPO / args.functional_base).exists() else {}
    ft = json.loads((REPO / args.functional_trained).read_text()) if (REPO / args.functional_trained).exists() else {}
    tags = sorted({r["model"] for r in tr_rows})
    result = {}
    lines = ["| tag | pool | v4 free base | v4 free trained | Δ pp [CI] | v4 under C1+C2+C3 trained | sig-mismatch base (none / full) | sig-mismatch trained (none / full) |",
             "|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        m = _TAG_RE.match(tag)
        base_model = m.group("model") if m else tag
        for pool in POOLS:
            b = _v4_by_pid(base_rows, base_model, pool, "none"); t = _v4_by_pid(tr_rows, tag, pool, "none")
            t_full = _v4_by_pid(tr_rows, tag, pool, "c1_c2_c3")
            if not b or not t:
                continue
            ids = sorted(set(b) & set(t))
            d = paired_bootstrap_diff([t[i] for i in ids], [b[i] for i in ids])  # trained − base
            entry = {"base_model": base_model, "n": len(ids),
                     "v4_free_base": round(sum(b[i] for i in ids) / len(ids), 4),
                     "v4_free_trained": round(sum(t[i] for i in ids) / len(ids), 4),
                     "v4_full_trained": round(sum(t_full.values()) / len(t_full), 4) if t_full else None,
                     "delta_pp": {"pp": round(100 * d.point, 1), "ci95": [round(100 * d.ci_low, 1), round(100 * d.ci_high, 1)], "p": d.p_value},
                     "functional_signature_mismatch": {
                         "base": {c: _functional_sig(fb.get(base_model), c, pool) for c in ("none", "c1_c2_c3")},
                         "trained": {c: _functional_sig(ft.get(tag), c, pool) for c in ("none", "c1_c2_c3")}}}
            result[f"{tag}|{pool}"] = entry

            def _fs(x):
                return "–" if not x else f"{x['signature_mismatch']}/{x['n']}"
            fsb = entry["functional_signature_mismatch"]["base"]; fst = entry["functional_signature_mismatch"]["trained"]
            v4full = "–" if entry["v4_full_trained"] is None else f"{100 * entry['v4_full_trained']:.1f}"
            lines.append(f"| {tag} | {pool} | {100*entry['v4_free_base']:.1f} | {100*entry['v4_free_trained']:.1f} | "
                         f"{entry['delta_pp']['pp']:+.1f} [{entry['delta_pp']['ci95'][0]:+.1f},{entry['delta_pp']['ci95'][1]:+.1f}] | "
                         f"{v4full} | "
                         f"{_fs(fsb['none'])} / {_fs(fsb['c1_c2_c3'])} | {_fs(fst['none'])} / {_fs(fst['c1_c2_c3'])} |")
    out = REPO / args.out; out.mkdir(parents=True, exist_ok=True)
    (out / "c4.json").write_text(json.dumps(result, indent=2))
    md = ("# C4 residual (shape / interface), free decoding before vs after training\n\n"
          "v4 = mlir-opt shape/interface diagnostic on a parse-valid program. Δ = trained − base, paired by prompt_id, "
          "10k-resample bootstrap. Signature mismatch = verify-valid outputs whose function interface differs from the gold "
          "(functional harness), over gold-gate-passed prompts.\n\n" + "\n".join(lines) + "\n")
    (out / "c4.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
