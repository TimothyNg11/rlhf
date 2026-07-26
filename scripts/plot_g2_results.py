#!/usr/bin/env python
"""G2 headline figure: the accuracy ladder (base -> round 1 -> round 2) and the
paired improvements with 95% CIs, from the bf16 k=8 eval sweeps.

Absolute accuracies + their bootstrap CIs are read from the eval runs'
``summary.json`` files (results/sweep2/, gitignored — run on a machine that has
them). Paired deltas are the ``scripts/paired_delta.py --metric lenient``
outputs recorded in docs/g2_plan.md (paired CIs cannot be recomputed from
summaries alone; rerun paired_delta.py against the sample files to reproduce).

    python scripts/plot_g2_results.py --out report/g2_results.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

SWEEP = Path("results/sweep2")

# (label, eval run dir) in ladder order, base first.
RUNS = [
    ("base\nQwen2.5-0.5B-Instruct", "base05_k8_bf16"),
    ("round 1 · seed 0", "g2b_s0_step100_bf16"),
    ("round 1 · seed 1", "g2b_s1_step100_bf16"),
    ("round 2 · seed 0", "g2round2_lr25_s0_step50_bf16"),
    ("round 2 · seed 1", "g2round2_lr25_s1_step50_bf16"),
]

# Paired lenient delta vs base (pp), 95% CI — from scripts/paired_delta.py,
# recorded in docs/g2_plan.md.
PAIRED_VS_BASE = {
    "round 1 · seed 0": (3.70, 2.67, 4.73),
    "round 1 · seed 1": (4.40, 3.37, 5.40),
    "round 2 · seed 0": (5.47, 4.40, 6.54),
    "round 2 · seed 1": (4.84, 3.77, 5.92),
}

BLUE = "#2a78d6"      # trained models (validated categorical slot 1, light mode)
GRAY = "#52514e"      # base / reference ink
INK = "#0b0b0b"
MUTED = "#8a8985"
SURFACE = "#fcfcfb"


def load_point(run_dir: str) -> tuple[float, float, float]:
    s = json.loads((SWEEP / run_dir / "gsm8k" / "summary.json").read_text(encoding="utf-8"))
    return (
        100 * s["pass_at_1_lenient"],
        100 * s["ci_lo_lenient"],
        100 * s["ci_hi_lenient"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot the G2 headline results figure.")
    parser.add_argument("--out", default="report/g2_results.png")
    args = parser.parse_args(argv)

    points = [(label, *load_point(run_dir)) for label, run_dir in RUNS]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(10.5, 3.6), width_ratios=[1.15, 1.0], facecolor=SURFACE
    )

    # --- Panel A: absolute accuracy ladder --------------------------------
    ys = list(range(len(points)))[::-1]
    for y, (label, val, lo, hi) in zip(ys, points):
        color = GRAY if label.startswith("base") else BLUE
        ax_a.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="butt", zorder=2)
        ax_a.plot([val], [y], "o", color=color, ms=8, zorder=3)
        ax_a.annotate(
            f"{val:.1f}", (val, y), textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=9, color=INK,
        )
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([p[0] for p in points], fontsize=9, color=INK)
    ax_a.set_ylim(-0.6, len(points) - 0.25)
    ax_a.set_xlabel("GSM8K test accuracy — lenient pass@1, k=8 (%)", fontsize=9, color=GRAY)
    ax_a.set_title("Accuracy (95% bootstrap CI)", fontsize=10, color=INK, loc="left")

    # --- Panel B: paired improvement vs base ------------------------------
    trained = [p for p in points if not p[0].startswith("base")]
    ys_b = list(range(len(trained)))[::-1]
    ax_b.axvline(0, color=MUTED, lw=1, ls="--", zorder=1)
    for y, (label, *_rest) in zip(ys_b, trained):
        d, lo, hi = PAIRED_VS_BASE[label]
        ax_b.plot([lo, hi], [y, y], color=BLUE, lw=2, solid_capstyle="butt", zorder=2)
        ax_b.plot([d], [y], "o", color=BLUE, ms=8, zorder=3)
        ax_b.annotate(
            f"+{d:.1f}", (d, y), textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=9, color=INK,
        )
    ax_b.set_yticks(ys_b)
    ax_b.set_yticklabels([p[0] for p in trained], fontsize=9, color=INK)
    ax_b.set_ylim(-0.6, len(trained) - 0.25)
    ax_b.set_xlabel("improvement over base — paired Δ pass@1, pp (95% CI)", fontsize=9, color=GRAY)
    ax_b.set_title("Paired improvement (format-proof)", fontsize=10, color=INK, loc="left")
    ax_b.set_xlim(left=min(-0.4, ax_b.get_xlim()[0]))

    for ax in (ax_a, ax_b):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color="#e8e7e3", lw=0.8, zorder=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=GRAY, labelsize=8)
        for tl in ax.get_yticklabels():
            tl.set_color(INK)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
