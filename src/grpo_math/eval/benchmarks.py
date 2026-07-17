"""Benchmark registry and problem loading for the eval harness.

Two source kinds:
  - ``hf``: loaded via ``datasets.load_dataset`` (lazy import inside each loader --
    ``datasets`` is a ``gpu``-extra dependency, not a core one, and must never be
    imported at module import time).
  - ``local``: a JSONL file of ``{"problem_id", "prompt_raw", "gold", "metadata"}``
    records, used by tests (``tests/fixtures/*.jsonl``) and for custom problem sets.

HF dataset ids/fields below are pinned from public docs; they are validated live
at gate G0 on the GPU box and are NOT exercised by CI (no network / no ``datasets``
package in this dev environment).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# DeepSeek's recommended math prompt, appended to every problem statement (both
# hf and local sources) after the raw problem text.
ANSWER_INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

# mmlu_stem-specific instruction: the final answer is a multiple-choice letter,
# not a computed value, so it gets its own instruction rather than the generic one.
_MMLU_INSTRUCTION = (
    "\n\nPlease reason step by step, and put the letter of your final answer "
    "(A, B, C, or D) within \\boxed{}."
)

_MMLU_STEM_SUBJECTS = [
    "abstract_algebra",
    "college_mathematics",
    "college_physics",
    "college_chemistry",
    "college_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
]
_MMLU_SAMPLE_SIZE = 400
_MMLU_SEED = 0
_MMLU_LETTERS = "ABCD"


@dataclass(frozen=True)
class EvalProblem:
    problem_id: str
    prompt: str  # raw problem statement + answer-format instruction, NOT chat-templated
    gold: str
    metadata: dict


@dataclass(frozen=True)
class BenchmarkSpec:
    """Declarative source + field mapping for one benchmark.

    ``load_benchmark`` dispatches on ``source`` and, for ``hf`` sources, on
    ``kind`` to pick the loader: ``"simple"`` (field-mapping only), ``"gsm8k"``
    (custom gold extraction), or ``"mmlu_stem"`` (filter + sample + MC formatting).
    """

    source: str  # "hf" | "local"
    kind: str = "simple"  # "simple" | "gsm8k" | "mmlu_stem" (hf sources only)
    dataset_id: str | None = None  # hf: e.g. "HuggingFaceH4/aime_2024"
    hf_config: str | None = None  # hf: e.g. "main" (gsm8k), "all" (mmlu)
    split: str | None = None  # hf: e.g. "train", "test"
    prompt_field: str | None = None  # hf: source column -> EvalProblem.prompt
    answer_field: str | None = None  # hf: source column -> EvalProblem.gold
    path: Path | None = None  # local: jsonl file path
    take_last: int | None = None  # keep only the last N loaded problems (applied before `limit`)


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime24": BenchmarkSpec(
        source="hf",
        dataset_id="HuggingFaceH4/aime_2024",
        split="train",
        prompt_field="problem",
        answer_field="answer",
    ),
    "aime25": BenchmarkSpec(
        source="hf",
        dataset_id="yentinglin/aime_2025",
        split="train",
        prompt_field="problem",
        answer_field="answer",
    ),
    "math500": BenchmarkSpec(
        source="hf",
        dataset_id="HuggingFaceH4/MATH-500",
        split="test",
        prompt_field="problem",
        answer_field="answer",
    ),
    "gsm8k": BenchmarkSpec(
        source="hf",
        kind="gsm8k",
        dataset_id="openai/gsm8k",
        hf_config="main",
        split="test",
        prompt_field="question",
        answer_field="answer",
    ),
    "gsm8k_dev": BenchmarkSpec(
        # Same source as "gsm8k" but the train split's last 500 problems, held
        # out as a dev set for in-loop eval (see grpo_math.data.gsm8k, which
        # holds out the same tail-slice of the same split for training data).
        source="hf",
        kind="gsm8k",
        dataset_id="openai/gsm8k",
        hf_config="main",
        split="train",
        prompt_field="question",
        answer_field="answer",
        take_last=500,
    ),
    "amc23": BenchmarkSpec(
        source="hf",
        dataset_id="math-ai/amc23",
        split="test",
        prompt_field="question",
        answer_field="answer",
    ),
    "mmlu_stem": BenchmarkSpec(
        source="hf",
        kind="mmlu_stem",
        dataset_id="cais/mmlu",
        hf_config="all",
        split="test",
    ),
}


def register_local_benchmark(name: str, path: str | Path) -> None:
    """Register a ``local`` JSONL benchmark under ``name`` in the module-level registry.

    Used by tests (see ``tests/fixtures/*.jsonl``) and for custom problem sets.
    This is the registry-extension hook: it's a plain write into the public
    ``BENCHMARKS`` dict rather than a separate plugin system, since one dict
    entry is all a JSONL-backed benchmark needs.
    """
    BENCHMARKS[name] = BenchmarkSpec(source="local", path=Path(path))


def _load_local(name: str, spec: BenchmarkSpec) -> list[EvalProblem]:
    problems = []
    with open(spec.path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problems.append(
                EvalProblem(
                    problem_id=row["problem_id"],
                    prompt=row["prompt_raw"] + ANSWER_INSTRUCTION,
                    gold=row["gold"],
                    metadata=row.get("metadata", {}),
                )
            )
    return problems


def _load_hf_dataset(spec: BenchmarkSpec):
    from datasets import load_dataset  # lazy: datasets is a gpu-extra dep

    if spec.hf_config is not None:
        return load_dataset(spec.dataset_id, spec.hf_config, split=spec.split)
    return load_dataset(spec.dataset_id, split=spec.split)


def _load_hf_simple(name: str, spec: BenchmarkSpec) -> list[EvalProblem]:
    ds = _load_hf_dataset(spec)
    problems = []
    for i, row in enumerate(ds):
        problems.append(
            EvalProblem(
                problem_id=f"{name}_{i}",
                prompt=str(row[spec.prompt_field]) + ANSWER_INSTRUCTION,
                gold=str(row[spec.answer_field]),
                metadata={},
            )
        )
    return problems


def _load_gsm8k(name: str, spec: BenchmarkSpec) -> list[EvalProblem]:
    ds = _load_hf_dataset(spec)
    problems = []
    for i, row in enumerate(ds):
        gold = str(row[spec.answer_field]).split("#### ")[-1].strip().replace(",", "")
        problems.append(
            EvalProblem(
                problem_id=f"{name}_{i}",
                prompt=str(row[spec.prompt_field]) + ANSWER_INSTRUCTION,
                gold=gold,
                metadata={},
            )
        )
    return problems


def _load_mmlu_stem(name: str, spec: BenchmarkSpec) -> list[EvalProblem]:
    ds = _load_hf_dataset(spec)
    rows = [(row["subject"], i, row) for i, row in enumerate(ds) if row["subject"] in _MMLU_STEM_SUBJECTS]
    rows.sort(key=lambda t: (t[0], t[1]))  # stable order before sampling

    if len(rows) > _MMLU_SAMPLE_SIZE:
        rng = np.random.default_rng(_MMLU_SEED)
        chosen = np.sort(rng.choice(len(rows), size=_MMLU_SAMPLE_SIZE, replace=False))
        rows = [rows[i] for i in chosen]

    problems = []
    for subject, orig_idx, row in rows:
        options = "\n".join(f"{_MMLU_LETTERS[j]}. {choice}" for j, choice in enumerate(row["choices"]))
        prompt = f"{row['question']}\n{options}" + _MMLU_INSTRUCTION
        gold = _MMLU_LETTERS[row["answer"]]
        problems.append(
            EvalProblem(
                problem_id=f"{name}_{subject}_{orig_idx}",
                prompt=prompt,
                gold=gold,
                metadata={"subject": subject},
            )
        )
    return problems


def load_benchmark(name: str, *, limit: int | None = None) -> list[EvalProblem]:
    """Load the ordered problem set for benchmark ``name``. If set, ``spec.take_last``
    keeps only the last N problems (applied right after loading, for any source
    kind); ``limit`` then truncates the list AFTER that (for smoke runs)."""
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark {name!r}; registered: {sorted(BENCHMARKS)}")

    spec = BENCHMARKS[name]
    if spec.source == "local":
        problems = _load_local(name, spec)
    elif spec.kind == "gsm8k":
        problems = _load_gsm8k(name, spec)
    elif spec.kind == "mmlu_stem":
        problems = _load_mmlu_stem(name, spec)
    else:
        problems = _load_hf_simple(name, spec)

    if spec.take_last is not None:
        problems = problems[-spec.take_last :]

    if limit is not None:
        problems = problems[:limit]
    return problems
