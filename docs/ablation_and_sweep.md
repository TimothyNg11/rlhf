# Ablation study + final test-set sweep

Qwen2.5-0.5B-Instruct, from-scratch GRPO on GSM8K, single H100. All ablation arms
run at **100 steps** — the reliable stable horizon for this recipe (longer runs hit the
entropy/length runaway diagnosed in `g1_diagnosis.md`). Stack: vLLM 0.25.1, torch 2.11,
transformers 5.14.1. Raw metrics under `results/train/*` and `results/sweep/*` (gitignored).

## 1. Ablation: what each lever buys

Four arms off the stabilized recipe `configs/g1_robust.yaml` (lr 1e-6, KL 0.003,
`truncation_mode: mask`, temp 0.9, G=4). Each arm changes exactly one variable; a second
baseline seed gives a seed-repeat noise floor. `dev` = last-500 GSM8K *train* holdout,
in-loop pass@1 (k=2, temp 0.6). Figure: `report/ablation_overlay.png`.

| arm | change vs baseline | dev@25 | dev@50 | dev@75 | peak entropy | outcome |
|---|---|---|---|---|---|---|
| baseline seed 0 | — (KL-on, G4) | 0.608 | 0.627 | 0.647 | 1.46 | ran 100 |
| baseline seed 1 | — (noise floor) | 0.608 | 0.626 | 0.634 | 0.37 | ran 100 |
| KL-off | KL 0.003 → 0 | 0.613 | 0.621 | 0.648 | 0.84 | ran 100 |
| G=8 | G 4 → 8, 32×8 (compute-matched) | 0.620 | 0.646 | **0.675** | **2.08** | **stop-loss abort @97** |

**Seed noise floor.** Same recipe, two seeds: dev@75 differs by only **1.3 pp** (0.647 vs
0.634), but the *entropy trajectory* differs enormously (peak 1.46 vs 0.37). 0.5B RLVR is
mildly seed-sensitive in accuracy but highly seed-sensitive in stability — any single-run
stability claim is unreliable, which is exactly why the noise floor was run.

**KL regularization (0.003 → 0).** Removing the KL anchor raised peak entropy (0.84 vs a
lucky seed's 0.37) but left dev accuracy unchanged (0.648 ≈ baseline). At lr 1e-6 with
truncation-masking, **KL is a secondary stabilizer** — the learning-rate cut and DAPO-style
overlong masking are the load-bearing levers, not KL.

**Group size (G=4 → G=8, compute-matched at 256 completions/step).** G=8 gave the best dev
accuracy (0.675 @75, **+2.8–4.1 pp over G=4, exceeding the 1.3 pp noise floor**) — bigger
groups yield lower-variance advantage estimates and faster learning. But it was the **least
stable**: entropy crossed the 2.0 abort threshold at step 97 and the stop-loss terminated
the run. Fewer distinct prompts per step (32 vs 64) concentrates the updates and drives
entropy up. Clean accuracy/stability trade-off.

**Entropy stop-loss.** Fired correctly and only on the divergent arm (G=8 @97), bounding
wasted compute; all three stable arms completed untouched.

## 2. Final test-set sweep (held-out, base vs trained)

Best checkpoint = `g1_robust` seed-0 **step-80** (taken at the dev peak, before seed-0's
late entropy creep). `g8` = its step-80 (before the @97 abort), reported as the ablation's
best. Eval k=4, temp 0.6, on the official test sets. Paired 95% CI over the identical
problem set (`scripts/paired_delta.py`).

| benchmark (n) | base pass@1 | headline pass@1 | Δ raw [paired 95% CI] | g8 pass@1 |
|---|---|---|---|---|
| GSM8K **test** (1319) | 0.316 | **0.395** | **+7.9 [+6.3, +9.4]** | 0.398 |
| MATH-500 (500) | 0.209 | 0.232 | +2.3 [+0.1, +4.5] | 0.215 |
| MMLU-STEM (400) | 0.264 | 0.299 | +3.5 [+0.4, +6.6] | 0.327 |

No capability collapse anywhere (MMLU-STEM held or rose). GSM8K test gain is large and
its CI clearly excludes zero; MATH-500 (out-of-distribution) and MMLU are small, CIs barely
off zero.

### The load-bearing caveat: format adherence, not reasoning

Parse rate jumps sharply with training (GSM8K 80.0% → 99.6%, MATH 91.6% → 96.5%, MMLU
80.4% → 95.3%). Correcting for it — solve-rate **among answers that actually emit a boxed
result** (pass@1 / parse_rate):

| benchmark | base | headline | g8 |
|---|---|---|---|
| GSM8K | 0.395 | **0.396** | 0.401 |
| MATH-500 | 0.228 | 0.240 | 0.226 |
| MMLU-STEM | 0.329 | 0.314 | 0.340 |

**The parse-corrected solve rate is flat across all three benchmarks.** The entire raw
pass@1 gain is explained by format compliance: the base model could already solve ~39.5% of
GSM8K but only boxed its answer 80% of the time; RL + format-bonus lifted boxing to 99.6%,
*surfacing latent capability* rather than teaching new reasoning. This is a known RLVR
dynamic on small models — with a format+correctness reward, the cheapest gradient is format
compliance on already-solvable problems. The parse-rate correction is what exposes it.

### Dev overstates absolute performance

Base scores 57.6% on the dev holdout but 31.6% on GSM8K test — a 26-pt gap for the *same
model*, most plausibly GSM8K *train*-set contamination in Qwen2.5's pretraining (dev = held-
out train problems). The held-out **test** number is the honest one; this is precisely why
the final test sweep matters and the dev curve alone would have been misleading.

## 3. Takeaways

- Stabilizer ranking: **lr-cut + truncation-mask ≫ KL** for this recipe; KL is secondary.
- **Larger groups help accuracy but hurt stability** (compute-matched G=8: best dev, earliest
  divergence); the stop-loss caught it.
- **0.5B RLVR stability is strongly seed-dependent** (entropy peak 0.37 vs 1.46, same recipe).
- Headline: **+7.9 pp GSM8K test** (paired CI [+6.3, +9.4]), **no collapse** on MMLU/MATH —
  but **parse-rate correction shows the gain is format adherence, not reasoning**. The
  honest contribution is a correct, stable trainer plus the analysis that distinguishes the
  two.
