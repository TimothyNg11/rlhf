"""Evaluation statistics: pass@1 and bootstrap confidence intervals.

All functions operate on NumPy arrays of per-sample verdicts (0.0/1.0) or
per-problem means, over a fixed problem set.
"""

from __future__ import annotations

import numpy as np


def pass_at_1(results: np.ndarray) -> float:
    """Grand mean over all (problem, sample) verdicts. ``results`` has shape [n_problems, k]."""
    return float(np.mean(results))


def per_problem_means(results: np.ndarray) -> np.ndarray:
    """Per-problem mean verdict. ``results`` has shape [n_problems, k] -> [n_problems]."""
    return np.mean(results, axis=1)


def bootstrap_ci(
    per_problem: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Problem-level percentile bootstrap CI of the mean.

    Resamples problems (with replacement) ``n_boot`` times and returns the
    ``(alpha/2, 1 - alpha/2)`` percentiles of the resampled means. Deterministic
    given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n = per_problem.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = per_problem[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired problem-level bootstrap CI of the mean difference ``mean(a) - mean(b)``.

    ``a`` and ``b`` must be per-problem means over IDENTICAL problem sets (same
    shape). The same resampled problem indices are applied to both arrays on
    each bootstrap replicate, preserving the pairing.
    """
    assert a.shape == b.shape, f"a and b must have identical shape, got {a.shape} vs {b.shape}"

    delta = float(np.mean(a) - np.mean(b))

    rng = np.random.default_rng(seed)
    n = a.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_delta = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo = float(np.percentile(boot_delta, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_delta, 100 * (1 - alpha / 2)))
    return delta, lo, hi
