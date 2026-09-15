"""Constraint ladder runner: model × pool × constraint stack → per-row JSONL + summary.

Resumable: rows already present in the output JSONL are skipped. Protocol matches the
prior study: max_tokens=600, greedy first attempt, C3 retries at temp 0.8 seeded per
(seed, attempt), verification via `mlir-opt --verify-diagnostics` (sqlite-cached).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from sci.constraints.parser import is_parse_valid
from sci.data.pools import load_pool
from sci.eval.generate import Generator, build_prompt
from sci.eval.stats import bootstrap_ci, paired_bootstrap_diff
from sci.eval.verify import verify

CONSTRAINTS = ("none", "c1", "c1_c2", "c1_c2_c3")
POOL_DIALECT = {"arith_func_200": "arith+func", "linalg_125": "linalg"}


def _done_keys(path: Path) -> set[tuple]:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["model"], r["pool"], r["constraint"], r["prompt_id"]))
    return keys


def run_model(model_repo: str, revision: str, model_tag: str, pools: list[str], constraints: tuple[str, ...],
              out_jsonl: Path, max_tokens: int = 600, seed: int = 0, method: str = "none",
              ckpt: str = "base", ckpt_dir: Path | None = None, limit: int | None = None) -> None:
    done = _done_keys(out_jsonl)
    todo = [(p, c) for p in pools for c in constraints]
    if all((model_tag, p, c, i) in done for p, c in todo for i in range(len(load_pool(p))
                                                                           if limit is None else limit)):
        print(f"[ladder] {model_tag}: all cells present, skipping", file=sys.stderr)
        return
    gen = Generator(model_repo, revision=revision)
    if ckpt_dir is not None:
        from sci.train.run import load_ckpt
        load_ckpt(gen, Path(ckpt_dir))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a") as fout:
        for pool_name in pools:
            rows = load_pool(pool_name)
            if limit:
                rows = rows[:limit]
            dialect = POOL_DIALECT[pool_name]
            for constraint in constraints:
                n_ok = n_pv = 0; t_cell = time.perf_counter(); n_done = 0
                for r in rows:
                    key = (model_tag, pool_name, constraint, r["prompt_id"])
                    if key in done:
                        continue
                    prompt = build_prompt(gen.tokenizer, r["nl"], dialect)
                    try:
                        g = gen.generate(prompt, constraint=constraint, max_tokens=max_tokens, seed=seed)
                        text = g.text
                        pv = is_parse_valid(text)
                        vv = bool(pv) and verify(text)["returncode"] == 0
                        row = {"model": model_tag, "method": method, "ckpt": ckpt, "train_seed": None,
                               "constraint": constraint, "pool": pool_name, "seed": seed,
                               "prompt_id": r["prompt_id"], "nl": r["nl"], "generated": text,
                               "parse_valid": pv, "verify_valid": vv, "attempts": g.attempts,
                               "scope_passed_per_try": g.scope_passed_per_try, "finished": g.finished,
                               "n_tokens": len(g.tokens), "dt": round(g.dt, 3)}
                    except Exception as e:  # noqa: BLE001
                        row = {"model": model_tag, "method": method, "ckpt": ckpt, "train_seed": None,
                               "constraint": constraint, "pool": pool_name, "seed": seed,
                               "prompt_id": r["prompt_id"], "nl": r["nl"], "generated": "",
                               "parse_valid": False, "verify_valid": False, "attempts": 0,
                               "scope_passed_per_try": [], "finished": False, "n_tokens": 0, "dt": 0.0,
                               "error": f"{type(e).__name__}: {str(e)[:200]}"}
                    fout.write(json.dumps(row) + "\n"); fout.flush()
                    n_done += 1; n_pv += row["parse_valid"]; n_ok += row["verify_valid"]
                    if n_done % 25 == 0:
                        print(f"  [{model_tag}|{pool_name}|{constraint}] {n_done}/{len(rows)} parse={n_pv} verify={n_ok} "
                              f"{(time.perf_counter()-t_cell)/n_done:.2f}s/gen", file=sys.stderr)
                print(f"[ladder] {model_tag}|{pool_name}|{constraint}: verify {n_ok}/{n_done} "
                      f"in {(time.perf_counter()-t_cell)/60:.1f} min", file=sys.stderr)


def summarize(jsonl: Path, out_json: Path) -> dict:
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    cells: dict[tuple, dict[int, dict]] = {}
    for r in rows:
        cells.setdefault((r["model"], r["pool"], r["constraint"]), {})[r["prompt_id"]] = r
    summary: dict = {}
    for (model, pool, constraint), byid in sorted(cells.items()):
        ids = sorted(byid)
        vv = [int(byid[i]["verify_valid"]) for i in ids]
        pv = [int(byid[i]["parse_valid"]) for i in ids]
        ci = bootstrap_ci(vv)
        summary.setdefault(model, {}).setdefault(pool, {})[constraint] = {
            "n": len(ids), "parse_rate": round(sum(pv) / len(ids), 4), "verify_rate": round(ci.point, 4),
            "verify_ci95": [round(ci.ci_low, 4), round(ci.ci_high, 4)],
            "mean_attempts": round(sum(byid[i]["attempts"] for i in ids) / len(ids), 3),
            "mean_dt": round(sum(byid[i]["dt"] for i in ids) / len(ids), 3),
            "unfinished": sum(1 for i in ids if not byid[i].get("finished", True)),
        }
    # paired rung deltas
    for model in summary:
        for pool in summary[model]:
            deltas = {}
            for a, b in (("c1", "none"), ("c1_c2", "c1"), ("c1_c2_c3", "c1_c2"), ("c1_c2_c3", "none")):
                ka, kb = (model, pool, a), (model, pool, b)
                if ka in cells and kb in cells:
                    ids = sorted(set(cells[ka]) & set(cells[kb]))
                    if ids:
                        d = paired_bootstrap_diff([int(cells[ka][i]["verify_valid"]) for i in ids],
                                                  [int(cells[kb][i]["verify_valid"]) for i in ids])
                        deltas[f"{a}-{b}"] = {"pp": round(100 * d.point, 1), "ci95_pp": [round(100 * d.ci_low, 1), round(100 * d.ci_high, 1)],
                                              "p_one_sided": round(d.p_value, 4), "n_pairs": d.n}
            summary[model][pool]["paired_deltas"] = deltas
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    return summary
