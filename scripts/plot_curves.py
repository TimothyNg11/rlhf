#!/usr/bin/env python
"""CLI entry point for the G1 acceptance-artifact plot: an overlay of one or
more training runs' ``metrics.jsonl`` curves as small multiples.

Smoke test (no GPU):
    python scripts/plot_curves.py --runs results/train/g1-seed0/metrics.jsonl \\
        --out results/train/g1-seed0/curves.png

Overlay multiple runs (e.g. our GRPO run vs. the TRL reference run):
    python scripts/plot_curves.py \\
        --runs results/train/g1-seed0/metrics.jsonl results/train/g1_trl-seed0/metrics.jsonl \\
        --labels ours,trl --out results/g1_overlay.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402  (must follow matplotlib.use("Agg"))

from grpo_math.trainer.metrics import read_metrics  # noqa: E402

DEFAULT_METRICS = "reward_mean,kl_mean,entropy_mean,response_len_mean,parse_rate"


def _ema(values: list[float], factor: float) -> list[float]:
    """Exponential moving average: ``ema[0] = values[0]``;
    ``ema[t] = factor*ema[t-1] + (1-factor)*values[t]``."""
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(factor * out[-1] + (1 - factor) * v)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay GRPO training-run curves (G1 acceptance artifact).")
    parser.add_argument("--runs", nargs="+", required=True, help="one or more metrics.jsonl paths")
    parser.add_argument(
        "--labels",
        default=None,
        help="comma-separated labels, same count as --runs (default: each run file's parent dir name)",
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help="comma-separated train-row metric keys to plot, one panel each",
    )
    parser.add_argument(
        "--ema", type=float, default=0.9, help="EMA smoothing factor in [0, 1); 0 disables the overlay"
    )
    parser.add_argument("--out", required=True, help="output PNG path")
    return parser


def _resolve_labels(args: argparse.Namespace) -> list[str]:
    if args.labels is None:
        return [Path(r).parent.name for r in args.runs]
    labels = args.labels.split(",")
    if len(labels) != len(args.runs):
        raise SystemExit(f"--labels count ({len(labels)}) must match --runs count ({len(args.runs)})")
    return labels


def _load_runs(paths: list[str], labels: list[str]) -> list[dict]:
    runs = []
    for path, label in zip(paths, labels):
        rows = read_metrics(path)
        runs.append(
            {
                "label": label,
                "train": [r for r in rows if r.get("kind") == "train"],
                "eval": [r for r in rows if r.get("kind") == "eval"],
            }
        )
    return runs


def _plot_metric_panel(ax, runs: list[dict], key: str, ema_factor: float, colors: list[str]) -> None:
    for j, run in enumerate(runs):
        color = colors[j % len(colors)]
        steps, values = [], []
        for row in run["train"]:
            v = row.get(key)
            if v is None:
                continue
            steps.append(row["step"])
            values.append(v)
        if not values:
            continue
        raw_label = run["label"] if ema_factor <= 0 else None
        ax.plot(steps, values, alpha=0.3, color=color, label=raw_label)
        if ema_factor > 0:
            ax.plot(steps, _ema(values, ema_factor), color=color, label=run["label"])
    ax.set_title(key)
    ax.set_xlabel("step")


def _plot_pass_at_1_panel(ax, runs: list[dict], colors: list[str]) -> None:
    for j, run in enumerate(runs):
        color = colors[j % len(colors)]
        pts = [r for r in run["eval"] if r.get("pass_at_1") is not None]
        if not pts:
            continue
        steps = [r["step"] for r in pts]
        values = [r["pass_at_1"] for r in pts]
        lo = [v - r["ci_lo"] for v, r in zip(values, pts)]
        hi = [r["ci_hi"] - v for v, r in zip(values, pts)]
        ax.errorbar(steps, values, yerr=[lo, hi], fmt="o", color=color, label=run["label"])
    ax.set_title("pass_at_1 (dev)")
    ax.set_xlabel("step")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    labels = _resolve_labels(args)
    metric_keys = [m.strip() for m in args.metrics.split(",") if m.strip()]
    runs = _load_runs(args.runs, labels)

    n_panels = len(metric_keys) + 1  # +1 for the pass_at_1 panel
    n_cols = n_panels if n_panels < 3 else 3
    n_rows = -(-n_panels // n_cols)  # ceil division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, key in enumerate(metric_keys):
        _plot_metric_panel(axes_flat[i], runs, key, args.ema, colors)
    _plot_pass_at_1_panel(axes_flat[len(metric_keys)], runs, colors)

    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # One legend for the whole figure: dedupe labeled handles across panels
    # (a run can be missing from an individual panel if all its values were None).
    handles_by_label: dict[str, object] = {}
    for ax in axes_flat[:n_panels]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            handles_by_label.setdefault(ll, hh)
    if handles_by_label:
        fig.legend(
            list(handles_by_label.values()),
            list(handles_by_label.keys()),
            loc="upper center",
            ncol=len(handles_by_label),
            bbox_to_anchor=(0.5, 1.02),
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
