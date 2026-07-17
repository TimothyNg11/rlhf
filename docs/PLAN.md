# GRPO But Tiny: From-Scratch GRPO on GSM8K at 0.5B — Project Plan v3 (~$70)

## Context

Third scope revision, user-directed: drop the DeepScaleR/AIME reproduction entirely and shrink to the minimal project that still earns the "I implemented and understand GRPO" credential — **Qwen2.5-0.5B-Instruct, GSM8K only, ~4k context, G=4, short training, from-scratch trainer, ~$70 total** (user confirmed the ~$70 tier over ~$40 bare-bones, keeping a miniature ablation study; base model user-confirmed). The resume value now rests on (a) a from-scratch GRPO implementation validated against TRL, (b) a real reward curve with honest analysis, and (c) small but rigorous error-barred ablations — which GSM8K's 1,319-problem test set powers *better* than AIME's 30 problems ever did.

**Everything CPU-side already built survives unchanged**: reward/verifier module (`src/grpo_math/rewards/`), stats module (`src/grpo_math/stats.py`), config loader, eval harness (`src/grpo_math/eval/`, `scripts/run_eval.py` — gsm8k already in the registry, fp8 KV eval path kept), 158 passing tests. The trainer was never built, so nothing is thrown away except config files.

## The recipe (v3 binding values)

| Knob | Value | Note |
|---|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, bf16, full FT | richest public tiny-GRPO precedent; parse rate high from step 0 |
| Hardware | **1×H100** (~$2.2/hr; A100 fine too) | single GPU, single process — no DDP |
| Data | `openai/gsm8k` train (7,473); **last 500 held out as dev** | dev drives in-loop eval + stopping; test touched only at milestones |
| lr | 2e-6 constant AdamW | 2× the 1.5B recipe: smaller model, short run |
| Prompts/step | 64 (mini-batch 32 → 2 updates/iter) | |
| G | 4 | per user; ablation arm tests 8 |
| Rollout sampling | temp 0.9, top_p 1.0 | 0.6 is too cold for exploration at 0.5B (deviation from DeepScaleR, documented) |
| Context | prompt ≤512, response ≤3072 (~4k total) | GSM8K needs far less; cap is generous |
| KL / clip / advantage / loss | 0.001 low-var-KL, clip 0.2, group-norm, token-mean | unchanged from v2 |
| Steps | 400 (stop early if dev plateaus ≥100 steps) | ~45–60 s/step → 5–7 hrs |
| Reward | binary math-verify on `\boxed{}`; truncation → 0 | format bonus ONLY if G0 parse rate <90% |

**Trainer simplification** (this is now a feature): single-process single-GPU loop — vLLM engine and HF train step alternate via sleep/wake on one card. No DDP, no multi-engine weight broadcast. All correctness assertions kept: recomputed old-logprobs (never sampler logprobs), ratio≈1.0 at step 0, weight-hash sync check, token-mean normalization across micro-batches. DDP noted in the report as the documented scale-up path.

## Budget & hours (1×H100 @ ~$2.2/hr)

| Run | Hours | Cost (cap) |
|---|---|---|
| G0: baselines (GSM8K test k=4, MATH-500 k=4, MMLU-STEM-400 k=1) + 50-sample audit + setup | ~1.5 | $5 ($8) |
| G1: own trainer 100 steps + TRL GRPOTrainer reference, same config, curves overlaid | ~3 | $7 ($10) |
| Headline: 400 steps, seed 0 | 5–7 | $13 ($16) |
| Seed-repeat: identical, seed 1 (full-length noise floor) | 5–7 | $13 ($16) |
| Ablation arm: KL 0.001 → 0 | 5–6 | $12 ($15) |
| Ablation arm: G=8 @ 32 prompts (compute-matched: 256 completions/step both ways) | 5–6 | $13 ($16) |
| Final eval sweep: 5 checkpoints × ~20 min | ~1.5 | $4 |
| Contingency | — | $8 |
| **Total** | **~27–33 GPU-hrs** | **≈ $75 worst case** |

All runs fit overnight; arms can run on two cheap pods in parallel. Optional ~$2 curiosity row: AIME/AMC evals on the final model (registry already supports them) — expected ~0%, reported as scale honesty.

## Ablation analysis (miniature but properly powered)

- Arms run **full 400 steps** (no shared-horizon compromise needed at these prices); compared at step 400 plus full-curve overlays.
- Primary metric: GSM8K **test** pass@1 (k=4, temp 0.6) — CI ±~1.3 pts via existing `bootstrap_ci`; paired deltas via `paired_delta_ci`. Effects ≥2–3 pts resolve cleanly.
- Noise floor: headline seed-0 vs seed-1 delta, reported next to every ablation delta.
- Honest-analysis piece unique to 0.5B: **parse-rate-corrected accuracy** (accuracy among parseable answers vs raw) to separate "learned formatting" from "learned reasoning" — reviewers of tiny-model RL always ask this.
- Secondary: MATH-500 transfer, MMLU-STEM no-collapse check, response-length and entropy dynamics, KL drift (for the KL arm).

## Gates

- **G0**: base model parse rate >90%; GSM8K score in sanity band 30–50% (Qwen reports 49.6% with their prompt; ours differs — boxed, temp 0.6); 50-sample manual audit shows no checker false-negatives. Fail → fix harness/prompt before anything else.
- **G1**: own-trainer curve matches TRL GRPOTrainer within noise on the identical config. This is the load-bearing credential — do not proceed on "close enough".
- **G2** (step ~50 of headline): reward slope positive, entropy not collapsing, parse rate stable, no KL blowup. Stop-loss otherwise.

## Schedule (part-time; finishes BEFORE the Sept paper crunch)

| Week | Work | Gate |
|---|---|---|
| W1 (Jul 14) | Config rewrite (below), README reframe; rent 1×H100: G0 | G0 |
| W2–3 | From-scratch single-GPU GRPO trainer + unit tests; TRL reference run | G1 |
| W4 | Headline + seed-repeat (overnight runs) | G2 |
| W5 | Two ablation arms + final eval sweep | arms healthy |
| W6 (~Aug 22) | Report: curves, ablation table with CIs, parse-corrected analysis, limitations | shipped |

## Repo changes to execute on approval

1. **`configs/base.yaml`**: rewrite with the v3 recipe table above (model, lr 2e-6, prompts 64/mini 32, G=4, temp 0.9, prompt 512/response 3072, gsm8k dataset + `dev_holdout: 500`, eval_during_training → gsm8k_dev k=2 every 25 steps).
2. **New**: `configs/headline_gsm8k.yaml` (inherit base, max_steps 400, seeds via CLI/run field), `configs/ablation_kl_off.yaml` (kl_coef 0), `configs/ablation_g8.yaml` (group_size 8, prompts_per_step 32) — both inherit headline.
3. **Delete**: `configs/headline_8k.yaml`, `ablation_baseline.yaml`, `ablation_trunc_mask.yaml`, `phase1_gsm8k.yaml`, and the whole `configs/stretch/` dir (1.5B/DeepScaleR configs; superseded — the v2 recipe values remain recorded in plan history/public sources).
4. **`configs/eval.yaml`**: `benchmarks: [gsm8k, math500, mmlu_stem]`, `k_default: 4`; keep `kv_cache_dtype: fp8`, keep `k_final_aime` (harmless, registry still supports optional AIME rows).
5. **Tests**: `tests/test_config.py` — update `test_load_base_config_no_inherit` assertions (model name → Qwen2.5-0.5B-Instruct, lr → 2e-6); glob-parametrized tests adjust automatically to the new file set. Suite must stay green (count will drop with deleted configs).
6. **`README.md`**: reframe (tiny from-scratch GRPO on GSM8K; honest scale framing; ablations with CIs; no AIME claim).
7. **`docs/PLAN.md`**: sync to this v3.
8. **Memory**: update `grpo-math-project-state.md` (scope v3, ~$70, Qwen2.5-0.5B-Instruct, GSM8K).

Existing code to reuse untouched: `grpo_math.rewards.compute_reward`, `grpo_math.stats.{bootstrap_ci,paired_delta_ci}`, `grpo_math.eval` harness + `scripts/run_eval.py` (gsm8k loader already registered), config loader.

## Verification

- After config/README changes: `./.venv/Scripts/python.exe -m pytest -q` green.
- G0 runbook (GPU box): `pip install -e ".[gpu]"` then `python scripts/run_eval.py --config configs/eval.yaml --benchmark all --model Qwen/Qwen2.5-0.5B-Instruct --backend vllm` + `scripts/sample_audit.py` on the gsm8k samples.
- W2–3 trainer lands with its own unit tests (advantage/clip/KL/masking) and the G1 overlay plot as the acceptance artifact.

## Honest framing & risks

- Expected headline: GSM8K +8–15 pts over base (band is wide — 0.5B RL is seed-sensitive; that's exactly what the seed-repeat quantifies). No AIME/competition claims anywhere.
- Risk: gains dominated by formatting → mitigated by parse-corrected metric and reporting both.
- Risk: temp 0.9 rollouts + tiny model → entropy collapse or reward hacking of the verifier; G2 watches entropy, audit watches the checker.
- Risk: TRL mismatch at G1 exposes a trainer bug late in W3 → that's the point of the gate; budget contingency covers re-runs.
