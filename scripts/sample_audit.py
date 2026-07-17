#!/usr/bin/env python
"""CLI to build a manual-audit markdown sheet from a ``samples.jsonl.gz`` eval output.

Samples a stratified-by-verdict subset without replacement, aiming for roughly
half correct / half incorrect when both are available (checker false-negative
hunting needs incorrect-labeled samples), and writes markdown with an empty
``auditor_note:`` line per row for manual annotation.

Usage:
    python scripts/sample_audit.py --results results/.../samples.jsonl.gz \\
        --n 50 --seed 0 --out audit.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

_TAIL_CHARS = 400


def load_records(path: str | Path) -> list[dict]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Seeded sample without replacement, stratified by verdict (correct/incorrect),
    aiming for as close to a 50/50 split as the available pool allows -- topping
    up from the larger class when the other is short."""
    correct = [r for r in records if r["verdict"] == 1.0]
    incorrect = [r for r in records if r["verdict"] != 1.0]

    half = n // 2
    n_correct = min(half, len(correct))
    n_incorrect = min(n - n_correct, len(incorrect))
    n_correct = min(n - n_incorrect, len(correct))  # top up if incorrect pool was short

    rng = random.Random(seed)
    sampled = rng.sample(correct, n_correct) + rng.sample(incorrect, n_incorrect)
    rng.shuffle(sampled)
    return sampled


def render_markdown(records: list[dict]) -> str:
    lines = ["# Sample audit\n"]
    for r in records:
        tail = r.get("completion", "")[-_TAIL_CHARS:]
        lines.append(f"## {r['problem_id']} (sample {r.get('sample_idx', '?')})\n")
        lines.append(f"- gold: `{r['gold']}`")
        lines.append(f"- extracted: `{r.get('extracted')}`")
        lines.append(f"- verdict: {r['verdict']}")
        lines.append(f"\n```\n{tail}\n```\n")
        lines.append("auditor_note: \n")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a manual audit sheet from eval samples.")
    parser.add_argument("--results", required=True, help="path to samples.jsonl.gz")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="output markdown path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    records = load_records(args.results)
    sampled = stratified_sample(records, args.n, args.seed)
    Path(args.out).write_text(render_markdown(sampled), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
