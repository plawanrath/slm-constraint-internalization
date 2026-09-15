"""Statistics for the main results matrix.

Primary API:
  - pass_rate(passed): point estimate.
  - bootstrap_ci(passed, n_resamples=10_000): percentile CI for a single rate.
  - paired_bootstrap_diff(a, b, n_resamples=10_000): CI + p-value for (a - b)
    on paired binary outcomes (same prompts).
  - min_detectable_effect(n, alpha=0.05, baseline_rate=0.5): power-analysis helper.
  - compare_cells(df_a, df_b): convenience wrapper over pandas rows from harness.jsonl.

All inputs are binary arrays (0/1); all outputs are plain Python dicts so results
can be JSON-serialized directly for figure generation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    n: int
    n_resamples: int

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "n_resamples": self.n_resamples,
        }


@dataclass
class PairedDiffResult:
    point: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    n_resamples: int

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n": self.n,
            "n_resamples": self.n_resamples,
        }


def pass_rate(passed: np.ndarray | list[int]) -> float:
    arr = np.asarray(passed, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def bootstrap_ci(
    passed: np.ndarray | list[int],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile-method bootstrap CI for a binary pass-rate."""
    arr = np.asarray(passed, dtype=float)
    n = arr.size
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0, n_resamples)
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_resamples, n), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        point=float(arr.mean()),
        ci_low=float(low),
        ci_high=float(high),
        n=int(n),
        n_resamples=n_resamples,
    )


def paired_bootstrap_diff(
    a: np.ndarray | list[int],
    b: np.ndarray | list[int],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedDiffResult:
    """Paired bootstrap CI for (rate_a - rate_b) + one-sided p-value that a > b.

    Paired across identical prompt indices — both arrays must be same length
    and aligned by sample index.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"shape mismatch: {a_arr.shape} vs {b_arr.shape}")
    n = a_arr.size
    if n == 0:
        return PairedDiffResult(float("nan"), float("nan"), float("nan"), float("nan"), 0, n_resamples)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = a_arr[idx].mean(axis=1) - b_arr[idx].mean(axis=1)
    observed = float(a_arr.mean() - b_arr.mean())
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # One-sided p-value (test: a > b): fraction of bootstrap diffs ≤ 0.
    p_value = float((diffs <= 0).mean())
    return PairedDiffResult(
        point=observed,
        ci_low=float(low),
        ci_high=float(high),
        p_value=p_value,
        n=int(n),
        n_resamples=n_resamples,
    )


def min_detectable_effect(
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
    baseline_rate: float = 0.5,
) -> float:
    """Normal-approximation MDE for a two-proportion test at sample size n per arm.

    Returned in absolute percentage points (for the power analysis).
    """
    if n <= 0:
        return float("inf")
    from scipy.stats import norm  # type: ignore
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    p = baseline_rate
    # Two-proportion pooled SE approximation.
    se = math.sqrt(2 * p * (1 - p) / n)
    return float((z_a + z_b) * se)


def summarize_cells(rows: list[dict], key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    """Group harness rows by (model, constraint, dialect, seed) and produce
    per-cell bootstrap CIs."""
    grouped: dict[tuple, list[int]] = {}
    for row in rows:
        k = tuple(row[f] for f in key_fields)
        grouped.setdefault(k, []).append(1 if row["passed"] else 0)
    return {k: bootstrap_ci(v).as_dict() for k, v in grouped.items()}
