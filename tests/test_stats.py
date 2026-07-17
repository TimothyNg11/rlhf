"""Tests for grpo_math.stats: pass@1, bootstrap CIs, paired-delta CIs."""

import numpy as np
import pytest

from grpo_math.stats import (
    bootstrap_ci,
    paired_delta_ci,
    pass_at_1,
    per_problem_means,
)


def test_pass_at_1_grand_mean():
    results = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    assert pass_at_1(results) == pytest.approx(3.0 / 6.0)


def test_pass_at_1_all_correct():
    results = np.ones((5, 4))
    assert pass_at_1(results) == pytest.approx(1.0)


def test_per_problem_means():
    results = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    means = per_problem_means(results)
    np.testing.assert_allclose(means, [0.5, 1.0, 0.0])


def test_bootstrap_ci_deterministic_given_seed():
    per_problem = np.array([0.2, 0.5, 0.8, 1.0, 0.0, 0.4, 0.6, 0.9, 0.3, 0.7])
    lo1, hi1 = bootstrap_ci(per_problem, n_boot=1000, seed=42)
    lo2, hi2 = bootstrap_ci(per_problem, n_boot=1000, seed=42)
    assert lo1 == lo2
    assert hi1 == hi2


def test_bootstrap_ci_different_seed_can_differ():
    per_problem = np.array([0.2, 0.5, 0.8, 1.0, 0.0, 0.4, 0.6, 0.9, 0.3, 0.7])
    ci_a = bootstrap_ci(per_problem, n_boot=1000, seed=0)
    ci_b = bootstrap_ci(per_problem, n_boot=1000, seed=1)
    assert ci_a != ci_b


def test_bootstrap_ci_bounds_sane():
    per_problem = np.array([0.2, 0.5, 0.8, 1.0, 0.0, 0.4, 0.6, 0.9, 0.3, 0.7])
    lo, hi = bootstrap_ci(per_problem, n_boot=2000, alpha=0.05, seed=0)
    assert 0.0 <= lo <= hi <= 1.0
    mean = per_problem.mean()
    assert lo <= mean <= hi


def test_bootstrap_ci_constant_array_is_degenerate():
    per_problem = np.full(10, 0.5)
    lo, hi = bootstrap_ci(per_problem, n_boot=500, seed=0)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_paired_delta_ci_identical_arrays_zero_delta():
    x = np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.6, 0.4])
    delta, lo, hi = paired_delta_ci(x, x, n_boot=1000, seed=0)
    assert delta == pytest.approx(0.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


def test_paired_delta_ci_matches_mean_difference():
    a = np.array([0.8, 0.9, 0.7, 1.0, 0.6])
    b = np.array([0.4, 0.5, 0.3, 0.6, 0.2])
    delta, lo, hi = paired_delta_ci(a, b, n_boot=2000, seed=0)
    assert delta == pytest.approx(a.mean() - b.mean())
    assert lo <= delta <= hi


def test_paired_delta_ci_deterministic_given_seed():
    a = np.array([0.8, 0.9, 0.7, 1.0, 0.6])
    b = np.array([0.4, 0.5, 0.3, 0.6, 0.2])
    r1 = paired_delta_ci(a, b, n_boot=1000, seed=7)
    r2 = paired_delta_ci(a, b, n_boot=1000, seed=7)
    assert r1 == r2


def test_paired_delta_ci_shape_mismatch_raises():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.1, 0.2])
    with pytest.raises(AssertionError):
        paired_delta_ci(a, b)


def test_bootstrap_ci_coverage_simulation():
    """Simulate 200 synthetic benchmarks from known Bernoulli means and check
    that the 95% CI covers the true mean in roughly [0.90, 0.99] of cases.
    Fixed seed throughout so this never flakes.
    """
    rng = np.random.default_rng(12345)
    n_benchmarks = 200
    n_problems = 60
    k_samples = 8
    covered = 0

    for i in range(n_benchmarks):
        true_mean = rng.uniform(0.2, 0.8)
        per_problem_p = np.full(n_problems, true_mean)
        samples = rng.binomial(k_samples, per_problem_p) / k_samples
        lo, hi = bootstrap_ci(samples, n_boot=1000, alpha=0.05, seed=i)
        if lo <= true_mean <= hi:
            covered += 1

    coverage = covered / n_benchmarks
    assert 0.90 <= coverage <= 0.99
