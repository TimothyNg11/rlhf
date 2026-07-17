# grpo-math

A **from-scratch GRPO** (Group Relative Policy Optimization) reinforcement-learning pipeline,
implemented line-by-line and validated against TRL's `GRPOTrainer`, that trains
`Qwen/Qwen2.5-0.5B-Instruct` on **GSM8K** with a binary verifiable reward. Deliberately tiny
(~$70 of single-H100 compute, ~4k context, G=4): the point is not a leaderboard number, it's

1. an RL-for-reasoning training loop whose every design decision is understood and tested,
2. a real reward curve with honest analysis (including parse-rate-corrected accuracy, separating
   "learned formatting" from "learned reasoning"), and
3. a miniature but rigorous ablation study — KL penalty on/off and group size G=4 vs G=8
   (compute-matched) — each reported as a paired bootstrap CI against a seed-repeat noise floor.

No competition-math claims: expected outcome is roughly **+8–15 pts GSM8K pass@1 over base**,
with MATH-500 as a transfer check and an MMLU-STEM subset as a no-collapse check.

See [`docs/PLAN.md`](docs/PLAN.md) for the full plan: recipe, budget, schedule, gates, and
ablation methodology.

## Status

- [x] **Week 1 (CPU-side)**: repo scaffold, run configs, reward/verifier module (math-verify +
      52-case golden file), stats module (bootstrap + paired-delta CIs), eval harness
      (benchmark registry, vLLM/fake backends, `run_eval.py` + `sample_audit.py` CLIs).
- [ ] **G0**: base-model baselines on a rented 1×H100 (GSM8K k=4, MATH-500 k=4, MMLU-STEM-400)
      + 50-sample manual grading audit.
- [ ] **G1**: from-scratch GRPO trainer + unit tests; TRL `GRPOTrainer` reference run on the
      identical config; curves must match within noise.
- [ ] **G2**: headline run (400 steps) + seed-repeat.
- [ ] Ablation arms (KL off; G=8 compute-matched) + final eval sweep.
- [ ] Technical report: curves, ablation table with CIs, honest limitations.

## Repo layout

| Path | Contents |
|---|---|
| `configs/` | YAML run configs (`base.yaml` + inheritable overrides per run/ablation) |
| `src/grpo_math/config.py` | YAML config loader with `inherit:`-based deep merge |
| `src/grpo_math/stats.py` | pass@1, bootstrap CIs, paired-delta CIs (NumPy only) |
| `src/grpo_math/rewards/` | `\boxed{}` extraction + math-verify answer checking + reward |
| `src/grpo_math/eval/` | Benchmark registry, generation backends (vLLM/fake), eval runner |
| `src/grpo_math/trainer/` | GRPO trainer (stub — next milestone, see `docs/PLAN.md` W2–3) |
| `src/grpo_math/data/` | Data prep (stub) |
| `scripts/` | `run_eval.py`, `sample_audit.py` |
| `tests/` | pytest suite incl. golden-file checker cases (`tests/golden/answers.jsonl`) |
| `docs/PLAN.md` | Full project plan (verbatim) |
| `report/` | Eval sweeps, ablation tables, final report (populated later) |

## Quickstart

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
pytest
```

## Scope note

This repo's tests are **CPU-only** and run on a Windows dev box (Python 3.13). No `torch`,
`vllm`, `transformers`, or `datasets` are imported at module import time anywhere in `src/` or
`tests/`. Training and evaluation (vLLM rollouts, GRPO updates) target a **single Linux H100**
(A100 also fine) — see the `gpu` extra in `pyproject.toml`. Example G0 eval command on the GPU
box:

```bash
pip install -e ".[gpu]"
python scripts/run_eval.py --config configs/eval.yaml --benchmark all \
    --model Qwen/Qwen2.5-0.5B-Instruct --backend vllm
```
