"""Unit tests for the stats module."""
from __future__ import annotations

import numpy as np

from sci.eval.stats import (
    bootstrap_ci,
    min_detectable_effect,
    paired_bootstrap_diff,
    pass_rate,
)


def test_pass_rate_basic() -> None:
    assert pass_rate([1, 0, 1, 1]) == 0.75
    assert pass_rate([]) != pass_rate([])  # NaN


def test_bootstrap_ci_contains_point() -> None:
    rng = np.random.default_rng(42)
    data = (rng.random(500) < 0.6).astype(int)
    r = bootstrap_ci(data.tolist(), n_resamples=2000, seed=0)
    assert r.ci_low < r.point < r.ci_high
    assert abs(r.point - 0.6) < 0.05
    assert r.n == 500


def test_paired_bootstrap_detects_positive_effect() -> None:
    rng = np.random.default_rng(0)
    n = 500
    a_probs = rng.uniform(0.4, 0.9, n)
    a = (rng.random(n) < a_probs).astype(int)
    # b is strictly worse than a for each prompt with some probability.
    b = (rng.random(n) < a_probs - 0.1).astype(int)
    r = paired_bootstrap_diff(a, b, n_resamples=2000, seed=0)
    assert r.point > 0
    assert r.ci_low > -0.02  # CI should not cross 0 much given effect size
    assert r.p_value < 0.05


def test_mde_shrinks_with_n() -> None:
    mde_100 = min_detectable_effect(n=100)
    mde_1000 = min_detectable_effect(n=1000)
    assert mde_1000 < mde_100
    assert mde_100 > 0.10  # ~0.14 at n=100, p=0.5
    assert mde_1000 < 0.08


def test_paired_requires_equal_length() -> None:
    import pytest
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0, 1], [1, 0])
