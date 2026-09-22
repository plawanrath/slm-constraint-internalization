"""Two analyses on top of the pass@k decodes (no new compute).

(a) Subset test: for every trained student with a functional cell, which of its greedy-correct prompts lie inside the
    untrained model's pass@16-correct set on the released arith+func pool. The selection mechanism predicts that
    on-policy students are correct almost only inside that set, and gold-supervised students also outside it.
(b) Sampled pass@1: for every checkpoint with 16 released-pool samples, the mean over prompts of the fraction of
    correct samples, with a bootstrap 95% interval over prompts, next to the greedy full-stack count.

  python scripts/w19_passk_analysis.py   ->  results/w19_passk_analysis/{subset.json,sampled_pass1.json,summary.md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
K_TRIALS = 5


def greedy_correct_set(tag: str, constraint: str = "c1_c2_c3") -> set[int] | None:
    """Prompt ids (idx) the harness scored as correct for results/w04_functional/<tag>/<constraint> (base: results/w02_functional)."""
    for root in ("results/w04_functional", "results/w02_functional"):
        p = REPO / root / tag / constraint / "differential.jsonl"
        if p.exists():
            trials: dict[int, list[bool]] = {}
            for l in p.read_text().splitlines():
                if l.strip():
                    d = json.loads(l)
                    if d.get("dialect") == "arith":
                        trials.setdefault(d["idx"], []).append(bool(d.get("match")))
            return {i for i, t in trials.items() if len(t) == K_TRIALS and all(t)}
    return None


def main() -> None:
    out = REPO / "results" / "w19_passk_analysis"; out.mkdir(parents=True, exist_ok=True)
    base = json.loads((REPO / "results/w16_passk/smollm2-360m-instruct/passk.json").read_text())
    base_set = {int(pid) for pid, m in base["per_prompt"].items() if any(m)}
    base16 = {int(pid) for pid, m in base["per_prompt"].items() if any(m[:16])}
    base4 = {int(pid) for pid, m in base["per_prompt"].items() if any(m[:4])}
    k_max = len(next(iter(base["per_prompt"].values()), []))
    K_GRID = [k for k in (4, 16, 64, 128, 256) if k <= k_max]
    base_at = {k: {int(pid) for pid, m in base["per_prompt"].items() if any(m[:k])} for k in K_GRID}
    students = [("smollm2-360m-instruct", "base, greedy full stack"),
                ("smollm2-360m-instruct_ssd_s0", "SSD"), ("smollm2-360m-instruct_ssd_s1", "SSD s1"), ("smollm2-360m-instruct_ssd_s2", "SSD s2"),
                ("smollm2-360m-instruct_rft_s0", "RFT"), ("smollm2-360m-instruct_rft_s1", "RFT s1"), ("smollm2-360m-instruct_rft_s2", "RFT s2"),
                ("smollm2-360m-instruct_ssd_s0__reward=functional", "SSD + functional reward"),
                ("smollm2-360m-instruct_ssd_s0__pool=v1c", "SSD + coverage pool"),
                ("smollm2-360m-instruct_ssd_s0__c3_mask=1", "SSD + dense scope mask"),
                ("smollm2-360m-instruct_ssd_s0__gold_in_group=1", "SSD + one gold per group"),
                ("smollm2-360m-instruct_ssd_s0__init=sft_s0", "SFT-gold then SSD"),
                ("smollm2-360m-instruct_sft_s0", "SFT-gold"), ("smollm2-360m-instruct_sft_s1", "SFT-gold s1"), ("smollm2-360m-instruct_sft_s2", "SFT-gold s2")]
    subset = {}
    lines = ["# pass@k analyses (arith+func, 143 gold-gate prompts)", "",
             f"Untrained 360M: {len(base4)} prompts correct within 4 masked samples, {len(base16)} within 16.", "",
             "## (a) Subset test: greedy-correct prompts under the full stack vs the base pass@16 set", "",
             "| student | correct | inside base pass@16 | outside | outside as % of correct |", "|---|---|---|---|---|"]
    for tag, label in students:
        s = greedy_correct_set(tag)
        if s is None:
            continue
        inside = len(s & base16); outside = len(s - base16)
        subset[tag] = {"label": label, "n_correct": len(s), "inside_base16": inside, "outside_base16": outside,
                       "outside_ids": sorted(s - base16), "inside_base4": len(s & base4)}
        lines.append(f"| {label} | {len(s)} | {inside} | {outside} | {100 * outside / len(s):.0f}% |" if s else f"| {label} | 0 | 0 | 0 | -- |")
    (out / "subset.json").write_text(json.dumps(subset, indent=2))

    # (a2) the same test against a deeper base budget: does gold supervision stay outside the base's reach as k grows?
    lines += ["", f"## (a2) Solved outside the base pass@k set, k = {K_GRID}", "",
              "| student | correct | " + " | ".join(f"outside @{k}" for k in K_GRID) + " |",
              "|---|---|" + "---|" * len(K_GRID)]
    depth = {"base_pass_at_k": {k: len(base_at[k]) for k in K_GRID}, "n_gate": base.get("n_gate"), "students": {}}
    for tag, label in students:
        st = greedy_correct_set(tag)
        if not st:
            continue
        row = {k: len(st - base_at[k]) for k in K_GRID}
        depth["students"][tag] = {"label": label, "n_correct": len(st), "outside": row}
        lines.append(f"| {label} | {len(st)} | " + " | ".join(str(row[k]) for k in K_GRID) + " |")
    lines.append("")
    lines.append("Base pass@k (of %s gate prompts): " % base.get("n_gate") + ", ".join(f"k={k}: {len(base_at[k])}" for k in K_GRID) + ".")
    (out / "subset_depth.json").write_text(json.dumps(depth, indent=2))

    lines += ["", "## (b) Sampled pass@1 (16 masked samples per prompt, temp 0.8) with bootstrap 95% CI over prompts", "",
              "| checkpoint | greedy full-stack correct | sampled pass@1 (of 143) [CI] | pass@16 |", "|---|---|---|---|"]
    sampled = {}
    rng = np.random.default_rng(0)
    for d in sorted((REPO / "results/w16_passk").glob("*/passk.json")):
        p = json.loads(d.read_text())
        if p.get("pool") == "dev_v1":
            continue
        tag = d.parent.name
        per = np.array([np.mean(m) for m in p["per_prompt"].values()])
        n = len(per)
        boots = [rng.choice(per, n, replace=True).mean() for _ in range(10_000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        gs = greedy_correct_set(tag)
        sampled[tag] = {"n": n, "pass1_mean": float(per.mean()), "pass1_ci95": [float(lo), float(hi)],
                        "pass1_count": float(per.mean() * n), "pass16": p["curve"][-1]["n_pass"], "greedy_full": len(gs) if gs is not None else None}
        lines.append(f"| {tag} | {len(gs) if gs is not None else '--'} | {per.mean() * n:.1f} [{lo * n:.1f}, {hi * n:.1f}] | {p['curve'][-1]['n_pass']} |")
    (out / "sampled_pass1.json").write_text(json.dumps(sampled, indent=2))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
