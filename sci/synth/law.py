"""C3 residual and the scaling-law fit for the synthetic sweep.

  v3(stack)            = fraction of dev prompts whose generation fails C3 under that stack
  Residual_3(D, size)  = v3(free) − v3(masked, C1+C2), paired bootstrap over prompts
  Residual_3           ≈ a · D^α · size^−β, fitted by least squares in log space

`residual_from_rows` and `fit_power_law` are pure numpy so the eval/fit path is testable without
any model; the sweep runner feeds them from per-prompt eval rows.
"""
from __future__ import annotations

import math

import numpy as np

from sci.eval.stats import paired_bootstrap_diff

MODEL_SIZES = {"SmolLM2-135M-Instruct": 135e6, "SmolLM2-360M-Instruct": 360e6, "SmolLM2-1.7B-Instruct": 1.7e9,
               "SmolLM3-3B": 3e9, "gemma-3-270m-it": 270e6, "gemma-3-1b-it": 1e9}


def model_size(name: str) -> float:
    """Parameter count for a model name or HF repo id (`org/name` accepted)."""
    return MODEL_SIZES[name.split("/")[-1]]


def residual_from_rows(rows: list[dict], n_resamples: int = 10_000, seed: int = 0) -> dict:
    """rows: per-prompt dicts with boolean `scope_ok_free` and `scope_ok_c3` (the C1+C2 mask plus the
    scope-validator retry loop; falls back to `scope_ok_masked` for rows that predate the c3 arm).
    Returns the paired residual v3(free) − v3(c3) with a bootstrap CI and the point rates of all arms."""
    if not rows:
        raise ValueError("no eval rows")
    key = "scope_ok_c3" if all("scope_ok_c3" in r for r in rows) else "scope_ok_masked"
    v_free = np.array([0 if r["scope_ok_free"] else 1 for r in rows], dtype=float)
    v_mask = np.array([0 if r["scope_ok_masked"] else 1 for r in rows], dtype=float)
    v_c3 = np.array([0 if r[key] else 1 for r in rows], dtype=float)
    d = paired_bootstrap_diff(v_free, v_c3, n_resamples=n_resamples, seed=seed)
    return {"v3_free": float(v_free.mean()), "v3_masked": float(v_mask.mean()), "v3_c3": float(v_c3.mean()),
            "c3_arm": key, "residual": d.point, "ci_low": d.ci_low, "ci_high": d.ci_high, "p_value": d.p_value, "n": d.n}


def fit_power_law(points: list[dict], eps: float = 1e-3) -> dict:
    """Least-squares fit of log r = log a + α log D − β log size over points {D, size, residual}.
    Residuals ≤ eps are clipped to eps (their count is reported). Returns a, alpha, beta, r2, n."""
    pts = [p for p in points if all(k in p for k in ("D", "size", "residual"))]
    if len(pts) < 3:
        raise ValueError("need at least 3 points to fit a · D^α · size^−β")
    r = np.array([p["residual"] for p in pts], dtype=float)
    n_clipped = int((r <= eps).sum())
    r = np.maximum(r, eps)
    X = np.column_stack([np.ones(len(pts)), np.log([p["D"] for p in pts]), -np.log([p["size"] for p in pts])])
    y = np.log(r)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"a": float(math.exp(coef[0])), "alpha": float(coef[1]), "beta": float(coef[2]), "r2": r2,
            "n": len(pts), "n_clipped": n_clipped, "eps": eps}


def predict(fit: dict, D: float, size: float) -> float:
    return fit["a"] * D ** fit["alpha"] * size ** (-fit["beta"])


def aggregate_seeds(sweep_rows: list[dict]) -> list[dict]:
    """Mean residual per (model, D) over seeds → fit points {model, D, size, residual, n_seeds}."""
    groups: dict[tuple[str, int], list[float]] = {}
    for r in sweep_rows:
        groups.setdefault((r["model"], int(r["D"])), []).append(float(r["residual"]))
    return [{"model": m, "D": D, "size": model_size(m), "residual": float(np.mean(v)), "n_seeds": len(v)}
            for (m, D), v in sorted(groups.items())]
