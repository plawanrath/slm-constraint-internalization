"""Scaling fit of the per-layer residual against model size.

For every layer k ∈ {v1, v2, v3, v4}, family (SmolLM, Gemma) and pool (arith_func_200, linalg_125,
all) fits Residual_k(size) = a · size^(−β) in log space, for the base models (Table-1 ladder,
results/w02_table1/ladder.jsonl: the free-vs-masked violation gap before training) and for every
trained method present in results/w04_residuals/ladder.jsonl (seed-averaged, ablation tags with
`__` excluded). Per-prompt violation flags are recomputed with sci.eval.residual.violations (served
by the verify cache) so β gets a paired prompt-resampling bootstrap CI. Residuals ≤ eps are clipped
to eps before the log (count reported). Parameter counts are the exact totals of the pinned HF
checkpoints (embedding + blocks + final norm; tied lm_head not double counted). If
results/w08_synth_law/fit.json exists it is copied in as the synthetic-law fit.
Writes results/w15_scaling/{fit.json, fit.md}. Missing scales are simply absent (n points reported).
  python scripts/w15_scaling_fit.py [--n-boot 2000] [--wait-docker MIN]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sci.constraints.parser import is_parse_valid  # noqa: E402
from sci.eval.residual import violations  # noqa: E402
from sci.eval.verify import VerifyCache  # noqa: E402

# Exact parameter counts from the pinned config.json files (SmolLM3 from its safetensors index).
PARAMS = {"smollm2-135m-instruct": 134_515_008, "smollm2-360m-instruct": 361_821_120, "smollm2-1.7b-instruct": 1_711_376_384,
          "smollm3-3b": 3_075_098_624, "gemma-3-270m-it": 268_098_176, "gemma-3-1b-it": 999_885_952}
HF_REPOS = {"smollm2-135m-instruct": "HuggingFaceTB/SmolLM2-135M-Instruct", "smollm2-360m-instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "smollm2-1.7b-instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
            "gemma-3-270m-it": "google/gemma-3-270m-it", "gemma-3-1b-it": "google/gemma-3-1b-it"}
FAMILY = {m: ("gemma" if m.startswith("gemma") else "smollm") for m in PARAMS}
LAYERS = ("v1", "v2", "v3", "v4")
STACK = {"v1": "c1", "v2": "c1_c2", "v3": "c1_c2_c3", "v4": "c1_c2_c3"}
POOLS = ("arith_func_200", "linalg_125")
FLAGS = ("--verify-diagnostics",)
_TAG_RE = re.compile(r"^(?P<model>.+?)_(?P<method>ssd|rft|grpo|sft)_s(?P<seed>\d+)$")


def params_from_config(cfg: dict) -> int:
    """Dense decoder parameter count from an HF config (llama / gemma3_text style; tied embeddings)."""
    c = cfg.get("text_config", cfg)
    h, L, V, ff = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"], c["intermediate_size"]
    nh, nkv = c["num_attention_heads"], c.get("num_key_value_heads") or c["num_attention_heads"]
    hd = c.get("head_dim") or h // nh
    attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h
    gemma = "gemma" in c.get("model_type", "")
    norms = (4 * h + 2 * hd) if gemma else 2 * h
    return V * h + L * (attn + 3 * h * ff + norms) + h


def recount_params() -> dict[str, int | None]:
    """Recount from the local HF cache via huggingface_hub (no paths hardcoded)."""
    from huggingface_hub import try_to_load_from_cache
    lock = {l.split("@", 1)[0]: l.split("@", 1)[1].strip() for l in (REPO / "scripts/env/models.lock.txt").read_text().splitlines() if "@" in l}
    out = {}
    for tag, repo in HF_REPOS.items():
        idx = try_to_load_from_cache(repo, "model.safetensors.index.json", revision=lock.get(repo))
        cfg = try_to_load_from_cache(repo, "config.json", revision=lock.get(repo))
        n = None
        if isinstance(idx, str):
            n = json.loads(Path(idx).read_text()).get("metadata", {}).get("total_parameters")
        if n is None and isinstance(cfg, str):
            n = params_from_config(json.loads(Path(cfg).read_text()))
        out[tag] = n
    return out


def docker_up(container: str = "sci-llvm") -> bool:
    r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    return r.returncode == 0 and container in r.stdout.split()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


class Flagger:
    """violations() with a text memo; rows whose verify result is uncached are dropped when the container is down."""

    def __init__(self, wait_min: float = 0.0):
        self.cache = VerifyCache(); self.memo: dict[str, dict | None] = {}; self.n_dropped = 0
        self.docker = docker_up()
        t0 = time.time()
        while not self.docker and time.time() - t0 < 60 * wait_min:
            print("[w15] waiting for the sci-llvm container ...", file=sys.stderr); time.sleep(30); self.docker = docker_up()

    def __call__(self, text: str) -> dict | None:
        if text in self.memo:
            return self.memo[text]
        f = None
        if not is_parse_valid(text) or self.docker or self.cache.get(text, FLAGS) is not None:
            f = violations(text)
        else:
            self.n_dropped += 1
        self.memo[text] = f
        return f


def series_arrays(rows: list[dict], model_tag: str, flag) -> dict[str, dict[str, np.ndarray]]:
    """Per pool: {'idx': prompt keys, k: (free − masked) paired per prompt} for every layer k (int arrays)."""
    cells: dict[tuple, dict[int, dict]] = {}
    for r in rows:
        if r["model"] == model_tag and r["pool"] in POOLS:
            f = flag(r["generated"])
            if f is not None:
                cells.setdefault((r["pool"], r["constraint"]), {})[r["prompt_id"]] = f
    out = {}
    for pool in POOLS:
        free = cells.get((pool, "none"))
        if not free:
            continue
        d = {}
        for k in LAYERS:
            m = cells.get((pool, STACK[k]))
            if not m:
                continue
            ids = sorted(set(free) & set(m))
            d[k] = np.array([int(free[i][k]) - int(m[i][k]) for i in ids], dtype=float)
        if d:
            out[pool] = d
    return out


def fit_log(sizes: np.ndarray, res: np.ndarray, eps: float) -> tuple[float, float, float, int]:
    """log r = log a − β log size. Returns (a, beta, r2, n_clipped)."""
    n_clip = int((res <= eps).sum()); r = np.maximum(res, eps)
    X = np.column_stack([np.ones(len(r)), -np.log(sizes)]); y = np.log(r)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef; ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(math.exp(coef[0])), float(coef[1]), (1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0), n_clip


def fit_family(points: dict[str, list[np.ndarray]], sizes: dict[str, float], n_boot: int, eps: float, seed: int) -> dict:
    """points: model → list (one per seed) of per-prompt paired differences, all aligned on the same prompt set.
    Bootstraps prompts (same resample for every model → paired across scales) and refits β."""
    models = sorted(points, key=lambda m: sizes[m])
    n = min(len(a) for m in models for a in points[m])
    sz = np.array([sizes[m] for m in models], dtype=float)
    def resid(idx):
        return np.array([np.mean([a[idx].mean() for a in points[m]]) for m in models])
    full = np.arange(n)
    r0 = resid(full)
    a, beta, r2, n_clip = fit_log(sz, r0, eps)
    rng = np.random.default_rng(seed)
    betas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        betas.append(fit_log(sz, resid(idx), eps)[1])
    betas = np.array(betas)
    lo, hi = (np.quantile(betas, [0.025, 0.975]) if len(betas) else (float("nan"), float("nan")))
    pts = []
    for m, r in zip(models, r0):
        per_seed = [a.mean() for a in points[m]]
        boot = rng.choice(len(points[m][0]), size=(min(n_boot, 2000), len(points[m][0]))) if n_boot else None
        ci = [float(np.quantile(np.mean([a[boot].mean(axis=1) for a in points[m]], axis=0), q)) for q in (0.025, 0.975)] if n_boot else [None, None]
        pts.append({"model": m, "size": sizes[m], "residual": float(r), "residual_pp": round(100 * float(r), 2),
                    "ci95_pp": [round(100 * ci[0], 2), round(100 * ci[1], 2)] if n_boot else None,
                    "n_seeds": len(points[m]), "per_seed": [round(float(x), 4) for x in per_seed]})
    return {"n_points": len(models), "n_prompts": int(n), "a": a, "beta": beta, "beta_ci95": [float(lo), float(hi)],
            "r2": r2, "n_clipped": n_clip, "eps": eps, "n_boot": n_boot, "points": pts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table1", default="results/w02_table1/ladder.jsonl")
    ap.add_argument("--trained", default="results/w04_residuals/ladder.jsonl")
    ap.add_argument("--synth", default="results/w08_synth_law/fit.json")
    ap.add_argument("--out", default="results/w15_scaling")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wait-docker", type=float, default=0.0, help="minutes to wait for sci-llvm if verify results are uncached")
    ap.add_argument("--recount-params", action="store_true", help="recount parameters from the local HF cache and report")
    args = ap.parse_args()
    params = dict(PARAMS)
    if args.recount_params:
        rc = recount_params(); print("[w15] recount:", rc, file=sys.stderr)
        for k, v in rc.items():
            if v is not None and v != params[k]:
                print(f"[w15] parameter count differs for {k}: hardcoded {params[k]} vs cache {v}", file=sys.stderr)
    flag = Flagger(args.wait_docker)
    base_rows = load_rows(REPO / args.table1); tr_rows = load_rows(REPO / args.trained)
    # series: name → model → list of per-seed arrays per pool per layer
    series: dict[str, dict[str, dict[str, dict[str, list[np.ndarray]]]]] = {}
    for m in sorted({r["model"] for r in base_rows} & set(params)):
        arr = series_arrays(base_rows, m, flag)
        for pool, d in arr.items():
            for k, a in d.items():
                series.setdefault("base", {}).setdefault(m, {}).setdefault(pool, {}).setdefault(k, []).append(a)
    for tag in sorted({r["model"] for r in tr_rows}):
        mt = _TAG_RE.match(tag)
        if not mt or mt.group("model") not in params:
            continue
        arr = series_arrays(tr_rows, tag, flag)
        name = f"trained:{mt.group('method')}"
        for pool, d in arr.items():
            for k, a in d.items():
                series.setdefault(name, {}).setdefault(mt.group("model"), {}).setdefault(pool, {}).setdefault(k, []).append(a)
    if flag.n_dropped:
        print(f"[w15] sci-llvm container down: {flag.n_dropped} parse-valid generations without a cached verify result were dropped", file=sys.stderr)
    fits: dict = {}
    md = ["# Scaling fit: Residual_k(size) = a · size^(−β)", "",
          f"Paired prompt-resampling bootstrap on β ({args.n_boot} resamples, seed {args.seed}); residuals ≤ {args.eps} clipped "
          "before the log. Sizes = exact parameter counts of the pinned checkpoints. `base` = free-vs-masked violation gap "
          "of the untrained model (Table 1); `trained:<method>` = residual after training (seed-averaged). A fit is `degenerate` "
          "when all but one point are clipped (the layer has no positive residual at these scales).", "",
          "| series | family | pool | layer | n | β | β 95% CI | a | R² | clipped | points (pp) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, by_model in sorted(series.items()):
        for fam in ("smollm", "gemma", "all"):
            models = [m for m in by_model if fam == "all" or FAMILY[m] == fam]
            for pool in (*POOLS, "all"):
                for k in LAYERS:
                    pts = {}
                    for m in models:
                        if pool == "all":
                            per_seed = [by_model[m][p][k] for p in POOLS if p in by_model[m] and k in by_model[m][p]]
                            if len(per_seed) == len(POOLS) and all(p in by_model[m] for p in POOLS):
                                ns = len(by_model[m][POOLS[0]][k])
                                if all(len(by_model[m][p][k]) == ns for p in POOLS):
                                    pts[m] = [np.concatenate([by_model[m][p][k][i] for p in POOLS]) for i in range(ns)]
                        elif pool in by_model[m] and k in by_model[m][pool]:
                            pts[m] = by_model[m][pool][k]
                    if len(pts) < 2:
                        if pts:
                            m0 = next(iter(pts)); r0 = float(np.mean([a.mean() for a in pts[m0]]))
                            fits.setdefault(name, {}).setdefault(fam, {}).setdefault(pool, {})[k] = {
                                "n_points": 1, "note": "need ≥ 2 scales", "points": [{"model": m0, "size": params[m0], "residual": r0, "residual_pp": round(100 * r0, 2)}]}
                            md.append(f"| {name} | {fam} | {pool} | {k} | 1 | – | – | – | – | – | {m0}={100 * r0:+.1f} (no fit: one scale) |")
                        continue
                    f = fit_family(pts, params, args.n_boot, args.eps, args.seed)
                    fits.setdefault(name, {}).setdefault(fam, {}).setdefault(pool, {})[k] = f
                    degenerate = f["n_clipped"] >= f["n_points"] - 1
                    f["degenerate"] = degenerate
                    md.append(f"| {name} | {fam} | {pool} | {k} | {f['n_points']} | {f['beta']:.3f} | [{f['beta_ci95'][0]:.3f}, {f['beta_ci95'][1]:.3f}] | "
                              f"{f['a']:.3g} | {f['r2']:.2f} | {f['n_clipped']}{' (degenerate)' if degenerate else ''} | " +
                              ", ".join(f"{p['model']}={p['residual_pp']:+.1f}" for p in f["points"]) + " |")
    result = {"params": params, "family": FAMILY, "layers": list(LAYERS), "fits": fits, "n_dropped_rows": flag.n_dropped}
    synth = REPO / args.synth
    if synth.exists():
        result["synthetic"] = json.loads(synth.read_text()); md += ["", f"Synthetic law fit included from `{args.synth}`."]
    else:
        md += ["", f"No synthetic fit (`{args.synth}` missing)."]
    out = REPO / args.out; out.mkdir(parents=True, exist_ok=True)
    (out / "fit.json").write_text(json.dumps(result, indent=2))
    (out / "fit.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
