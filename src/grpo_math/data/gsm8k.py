"""GSM8K train/dev split + prompt-index sampling for GRPO training (docs/PLAN.md).

``load_gsm8k_train`` lazy-imports ``datasets`` (a ``gpu``-extra dependency, not
installed in this dev environment) so this module can be imported anywhere.
"""

from __future__ import annotations

import numpy as np

from grpo_math.eval.benchmarks import ANSWER_INSTRUCTION, EvalProblem, gsm8k_gold


def split_holdout(items: list, dev_holdout: int) -> tuple[list, list]:
    """Tail-slice ``items`` into ``(train, dev)``: the last ``dev_holdout`` items
    become ``dev``, everything before them becomes ``train``. ``dev_holdout``
    must be > 0 and < ``len(items)``."""
    if not 0 < dev_holdout < len(items):
        raise ValueError(
            f"dev_holdout must be > 0 and < len(items) ({len(items)}); got {dev_holdout}"
        )
    return items[:-dev_holdout], items[-dev_holdout:]


def load_gsm8k_train(*, dev_holdout: int = 500) -> tuple[list[EvalProblem], list[EvalProblem]]:
    """Load ``openai/gsm8k`` config ``"main"`` split ``"train"`` and split it into
    ``(train, dev)`` via :func:`split_holdout`.

    Gold extraction and prompt construction match
    ``grpo_math.eval.benchmarks._load_gsm8k`` exactly, so the returned ``dev``
    slice is identical to what benchmark ``"gsm8k_dev"`` (``take_last=500`` on
    the same split) loads.
    """
    from datasets import load_dataset  # lazy: datasets is a gpu-extra dep

    ds = load_dataset("openai/gsm8k", "main", split="train")
    problems = []
    for i, row in enumerate(ds):
        gold = gsm8k_gold(str(row["answer"]))
        problems.append(
            EvalProblem(
                problem_id=f"gsm8k_train_{i}",
                prompt=str(row["question"]) + ANSWER_INSTRUCTION,
                gold=gold,
                metadata={},
            )
        )
    return split_holdout(problems, dev_holdout)


class PromptSampler:
    """Epoch-based, resumable sampler over prompt indices ``0..n_items-1``.

    Each epoch is a fresh permutation of ``range(n_items)``, seeded by
    ``seed * 1000 + epoch`` so distinct epochs (and distinct seeds) never
    reuse the same permutation. ``next_batch`` walks the current epoch's
    permutation ``batch_size`` indices at a time; when fewer than
    ``batch_size`` indices remain, ``drop_last`` decides whether to roll to a
    fresh epoch (dropping the partial tail) or return the smaller remainder.
    """

    def __init__(self, n_items: int, batch_size: int, *, seed: int, drop_last: bool = True):
        self.n_items = n_items
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.batch_idx = 0
        self._permutation = self._make_permutation(self.epoch)

    def _make_permutation(self, epoch: int) -> np.ndarray:
        return np.random.default_rng(self.seed * 1000 + epoch).permutation(self.n_items)

    def next_batch(self) -> list[int]:
        """Return the next batch of indices, rolling to a new epoch as needed."""
        start = self.batch_idx * self.batch_size
        remaining = self.n_items - start
        if remaining <= 0 or (remaining < self.batch_size and self.drop_last):
            self.epoch += 1
            self.batch_idx = 0
            self._permutation = self._make_permutation(self.epoch)
            start = 0
            remaining = self.n_items

        end = min(start + self.batch_size, self.n_items)
        batch = self._permutation[start:end].tolist()
        self.batch_idx += 1
        return batch

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "batch_idx": self.batch_idx}

    def load_state_dict(self, d: dict) -> None:
        """Restore state so that ``next_batch()`` returns exactly what the
        original sampler would have returned next: recompute the epoch's
        permutation from ``seed`` + ``epoch``, then resume at ``batch_idx``."""
        self.epoch = d["epoch"]
        self.batch_idx = d["batch_idx"]
        self._permutation = self._make_permutation(self.epoch)
