# G2 pre-registration: does the model actually get better at math?

Phase goal: a parse-proof demonstration that GRPO on GSM8K improves the policy's
math ability — not its answer formatting. Everything in this file was fixed
BEFORE the first GPU session of the phase; the budget is ~$70 on a ~$2.2/hr H100.

Motivation: the previous phase's +7.9pp GSM8K-test gain was attributed entirely
to format compliance (boxed-parse rate 80.0% → 99.6%, parse-corrected solve rate
flat 0.395 → 0.396; `docs/ablation_and_sweep.md`). Causes addressed here: the
reward's boxed gate + format bonus (cheapest gradient was formatting), a short
effective run, and GSM8K-train contamination producing zero-advantage groups.

## Pre-registered metric

- **Primary:** lenient-extraction pass@1 on the official GSM8K test set (1319),
  paired 95% CI vs the lenient base baseline (`scripts/paired_delta.py --metric
  lenient`). Lenient extraction: last `\boxed{}` → `#### x` → "answer is x" →
  last number (`scripts/regrade_lenient.py` re-grades old runs; the eval runner
  dual-grades new runs). **Success = CI excludes 0 AND Δ ≥ +3.0pp.**
- **Secondary (learning vs sharpening):** Δpass@8 (k=8, any-correct). pass@8 CI > 0
  → "learning" (the distribution genuinely improved); pass@1 up with pass@8 flat →
  "sharpening" (reported as such, still a positive result).
- **Anti-hacking cross-checks (final claim only):** strict-boxed delta alongside
  lenient (lenient ≫ strict = de-boxing signature → investigate before claiming);
  paired delta on the decision-untouched last-819 test problems.

## Pre-registered checkpoint selection

Candidate = saved checkpoint with the maximum in-loop dev pass@1 (ties → later
step). Dev (last-500 train holdout) is contaminated in level (base: 57.6% dev vs
31.6% test) and is used ONLY for stability monitoring and checkpoint selection —
never for go/no-go decisions, which ride exclusively on test-set evals.

## Training changes vs the last phase

| lever | old | new (`configs/g2_main.yaml`) |
|---|---|---|
| reward extraction | boxed-gated | lenient (correct pays regardless of format) |
| format_bonus | 0.2 | 0.0 |
| training data | all 7,473 | difficulty band [1/8, 7/8] of base-model solve rate (k=8 map) |
| group size | 4 | 8 (the one above-noise ablation winner) |
| prompts/step | 64 (G4) / 32 (G8 arm) | 48 — halves the prompt concentration that drove the G=8 entropy runaway |
| steps | 100 (stable horizon) | 300, cosine lr decay (min 0.1x), KL 0.006, save every 50 |

Unchanged: lr 1e-6, temp 0.9/top_p 1.0, truncation-mask, TIS cap 2.0, entropy
stop-loss 2.0, prompt still requests boxing (the reward just stops paying for it).
Tripwire: per-step extraction-method fractions (`extract_frac_*`) — a sustained
shift from `boxed` toward `last_number` triggers a sample audit.

## Gates

- **Gate A ($0, informational):** offline lenient re-grade of the existing sweep
  (base, g1_robust step-80, g8 step-80). Fixes the lenient baseline b0. Never
  stops the phase. **Results: see below — RUN, gates all passed.**
- **Gate B (step 150, ~$15 cum.):** milestone eval, first-500 GSM8K test, k=4.
  Continue iff paired lenient Δ ≥ +2.0pp OR CI excludes 0. Fail → 1.5B branch
  (`configs/g2_15b.yaml`, own baseline + own difficulty map; Gate D = same rule
  at its step 100). Entropy-abort before step 100 AND gate fail → 1.5B immediately.
- **Gate C (final, ~$33 cum.):** full-test k=8 base + candidate. Success per the
  pre-registered metric above. Pass → seed-1 repeat (~$21) or, if marginal,
  400-step extension (~$7). Fail with a healthy run → 1.5B branch if ≥ ~$37
  remains, else honest null write-up with the format/sharpening decomposition.

Budget discipline: every pod session has a stated cost and kill condition; if
actuals exceed plan by >25% at any gate, the seed repeat is dropped first.

## Gate A results (2026-07-24, CPU-only re-grade of the existing sweep)

`scripts/regrade_lenient.py` strict-recompute sanity gate passed on all 9
run×benchmark dirs (recomputed strict pass@1 == stored summaries exactly).

GSM8K test (n=1319), lenient metric, paired vs base:

| checkpoint | lenient pass@1 | Δpass@1 [95% CI] | lenient pass@4-any | Δpass@4 [95% CI] |
|---|---|---|---|---|
| base | 0.3734 | — | 0.6187 | — |
| g1_robust s0 step-80 | 0.3950 | **+2.16 [+0.66, +3.62]** | 0.6406 | +2.20 [-0.38, +4.78] |
| g8 s0 step-80 | 0.3982 | **+2.48 [+1.02, +3.94]** | 0.6475 | **+2.88 [+0.38, +5.38]** |

**Correction to the last phase's conclusion.** The parse-corrected ratio
(pass@1 / parse_rate = 0.395) OVERSTATED the base model's true solve rate: it
assumes unboxed completions are correct at the same rate as boxed ones, but the
lenient re-grade shows they are correct far less often (true base = 0.3734).
Roughly a third of the raw +7.9pp was therefore REAL improvement, not format —
"the entire gain is format adherence" in `ablation_and_sweep.md` was too
pessimistic. Both existing checkpoints show parse-proof gains with CIs
excluding zero, and g8's pass@4-any is also up — consistent with genuine
learning, before any of G2's levers (format-free reward, difficulty filtering,
longer stabilized G=8 run) have even been applied.

Per the pre-registered Gate A rule this raises the prior on the 0.5B branch and
the phase proceeds unchanged: the success bar remains Δ ≥ +3.0pp with CI > 0 on
the lenient metric (g8 step-80 sits at +2.48, below the bar).

Transfer benchmarks (lenient pass@1, unpaired summary): MATH-500 base 0.2175 →
g1_robust 0.2315 / g8 0.2150; MMLU-STEM base 0.2963 → g1_robust 0.3025 / g8
0.3281. Extraction-method histograms show trained checkpoints box ≥99% (lenient
≈ strict for candidates), so the lenient metric's honesty gain is on the base
side — exactly the confound the old analysis missed.

## Gate B + Gate C results (2026-07-25, 0.5B branch)

Training: `g2_main` (lr 1e-6) hit an entropy runaway at step 62 (0.45@51 →
2.69@62; stop-loss abort — the G=8 + difficulty-band + lenient-reward stack
concentrates more gradient per step than any prior recipe). Single-variable
retry `g2_main_b` (lr 5e-7) doubled the stable horizon but aborted at step 135
(entropy 2.14). Dev pass@1 0.576 → 0.647@100, plateau 0.646@125 — the model
converges by ~step 100, then drifts. Best-dev checkpoint (pre-registered rule):
step_0100.

- **Gate B (first-500 test, k=4): PASS** — lenient Δpass@1 +2.60pp
  [+0.20, +5.10] paired vs base.
- **Gate C (full test 1319, k=8): FAIL on magnitude** — lenient Δpass@1
  **+1.34pp [+0.25, +2.41]**: a real, parse-proof gain (CI > 0) but below the
  pre-registered +3.0pp bar. Δpass@8 +1.59 [−0.61, +3.79] (no distribution-level
  claim). Strict Δ +5.34 [+4.23, +6.45] — the strict−lenient gap is format, no
  de-boxing signature. Transfer (candidate, k=4, lenient): MATH-500 0.222 (base
  0.2175), MMLU-STEM 0.281 (base 0.2963) — no collapse, no transfer gain.
- The first-500 milestone (+2.60) vs full-test (+1.34) gap is exactly the
  slice-favorability the untouched-819 robustness row was designed to expose.

**0.5B branch conclusion:** at its ~100-step stable horizon, GRPO gives the
0.5B a consistently real but small parse-proof gain (+1.3 to +2.5pp across
this and the Gate A checkpoints). Below the success bar → per the
pre-registered tree, the phase pivots to Branch B (`configs/g2_15b.yaml`,
Qwen2.5-1.5B-Instruct) with ~$59 of budget remaining.

## fp8 KV-cache eval bug (2026-07-25) — affects all earlier absolute numbers

Setting up the 1.5B branch exposed that `eval.yaml`'s `kv_cache_dtype: fp8`
corrupts Qwen2.5 generation in vLLM 0.25.1: digit tokens are silently dropped
("lay  eggs per day"), with degenerate 32k-token loops. On the 1.5B this
collapsed GSM8K to ~2%; a bf16 sanity re-eval then showed the **0.5B was also
corrupted all along**: base = **0.4585 strict / 0.4642 lenient, 97.3% boxing**
under bf16, vs 0.316 / 0.373 / 80% boxing under fp8. Consequences:

- All absolute test numbers measured before this date (G0, both sweeps, Gate A
  re-grade, Gate B/C above) are fp8-deflated. The "base only boxes 80%" story
  and much of the dev-vs-test contamination gap were artifacts.
- Paired deltas compared same-config runs, so they measure real relative
  improvement *under fp8-corrupted inference*; the clean-inference 0.5B delta
  (base vs g2_main_b step-100, both k=8 bf16) is being re-measured.
- Fix: `configs/eval_bf16.yaml`; every eval from here on uses bf16 KV, and
  paired comparisons must match KV dtype on both sides. Erratum added to
  `docs/ablation_and_sweep.md`.
- 1.5B baseline (bf16, k=8): strict 0.7282 / lenient 0.7346 / pass@8 0.9227,
  parse 98.8% — healthy, consistent with published numbers.

## Clean-inference Gate C (2026-07-25): **PASS**

Re-measuring the pre-registered quantity under bf16 KV (full GSM8K test, k=8,
paired, same config both sides):

| metric | base | g2_main_b step-100 | Δ [95% CI] |
|---|---|---|---|
| **lenient pass@1 (primary)** | 0.4671 | **0.5041** | **+3.70 [+2.67, +4.73]** |
| lenient pass@8 (secondary) | 0.7832 | 0.7930 | +0.99 [−0.91, +2.88] |
| strict pass@1 (cross-check) | 0.4611 | 0.5039 | +4.28 [+3.25, +5.34] |
| lenient pass@1, untouched last-819 | — | — | +3.72 [+2.44, +5.02] |

- **Success bar met**: Δ ≥ +3.0pp AND CI excludes 0. The fp8-corrupted eval had
  been *understating* the gain (+1.34 under fp8).
- Pre-registered classification: **"sharpening"** (pass@1 up, pass@8 flat) — the
  policy reliably lands answers it could previously only sometimes sample, and
  crosses 50% on GSM8K test.
- No reward-hacking signature (strict ≈ lenient; boxing high on both sides).
- The gain reproduces on the 819 test problems no gate ever examined.

Decision per the pre-registered tree: Gate C pass → **seed-1 confirmation run**
(g2_main_b --seed 1, same kill/selection/eval protocol). The 1.5B branch is no
longer required; its baseline + difficulty map (~$7) remain available if the
user wants a second-model result later.

## Final verdict (2026-07-25): SUCCESS, seed-replicated

Seed-1 repeat (`g2_main_b --seed 1`, identical protocol: entropy abort @139 vs
seed-0's @135, dev plateau by ~step 100, best-dev checkpoint = step_0100):

| seed | lenient Δpass@1 (full test, k=8, bf16, paired) | strict Δ | Δpass@8 |
|---|---|---|---|
| 0 | +3.70 [+2.67, +4.73] | +4.28 | +0.99 [−0.91, +2.88] |
| 1 | **+4.40 [+3.37, +5.40]** | +4.98 | +0.00 [−1.90, +1.90] |

**The pre-registered success condition holds in both independent seeds** (seed-1's
CI lower bound alone clears the +3.0 bar). Character: consistent "sharpening" —
pass@1 up ~4pp, pass@8 flat — the policy reliably lands answers it previously
only sometimes sampled. Base 0.4671 → candidates 0.5041 / 0.5111.

What produced the genuine gain vs the last phase's format-only result: lenient
reward (format shortcut removed), difficulty-band training data (nearly every
group carries gradient), G=8, lr 5e-7 — at a reproducible ~135-step stable
horizon with convergence by ~100. Secondary finding with independent value: the
fp8-KV eval bug that had deflated every absolute number in the project and
manufactured the "base can't format" narrative.

Spend: ~$26 of the $70 cap (incl. all detours). Remaining budget untouched;
optional extensions (1.5B arm — baseline + map already banked; transfer sweep
for seed-1; 400-step probe) are user's call.

## G2.1 pre-registration: iterated difficulty re-mapping (round 2)

Hypothesis under test: the round-1 dev plateau at ~step 100 was **gradient
starvation** — the band was computed from the *base* model's solve rates, and
mastered prompts yield all-correct zero-advantage groups. Round 2 re-maps
difficulty from the trained policy (`g2_main_b` seed-1 step_0100, the 51.1%
model), rebuilds the band, and trains a second round from that checkpoint
(`configs/g2_round2.yaml` — same recipe, fresh optimizer/cosine over 150 steps;
only the parent model and the map change).

Pre-registered BEFORE round-2 GPU spend:

- **Primary (incremental):** paired lenient Δpass@1, round-2 best-dev
  checkpoint vs its parent (seed-1 step_0100), full GSM8K test, k=8,
  `eval_bf16.yaml`. **Success = CI excludes 0 AND Δ ≥ +1.0pp.**
- Secondary: cumulative Δ vs base; Δpass@8 class; strict-vs-lenient check.
- Checkpoint selection: unchanged best-dev rule.
- Interpretation: success → gradient-starvation supported; null → the plateau
  is a sharpening ceiling at 0.5B, reported as a finding (round-1 result
  stands either way).
- Confirmation on success only: independent lineage (seed-0 step_0100 parent,
  its own re-map, `--seed 1`; `configs/g2_round2_b.yaml`).
- **Amendment (2026-07-25, before any round-2 GPU spend):** the pod restart
  provisioned a fresh volume; seed-1's checkpoint existed only there and is
  lost. Parent switched to **seed-0 step_0100** (hash-verified local copy,
  re-uploaded). The primary incremental comparison is now vs seed-0's parent
  eval (`results/sweep2/g2b_s0_step100_bf16`, pass@1_lenient 0.5041). The
  confirmation arm downgrades to same-parent/different-RNG (`--seed 1`) —
  guards run-to-run luck, not parent-checkpoint luck; noted as a limitation.
- Predicted map shift (falsifiable): vs the base map (train mean 0.52, band
  kept 4,836) the trained-policy map should show the 8/8 mass swelling and the
  band shrinking. Pre-declared fallback: if band keeps < ~1,500 prompts, widen
  to [0, 7/8] via config.
- Budget: ~$12 base path, ~$22 with confirmation, from the ~$44 remaining.

## Execution status

- P0+P1 (measurement layer + trainer threading + configs) landed at commits
  0c71c0d, d2fb23a, 3fea7d6; suite 344 green on Windows CPU.
- Next: pod session P2 — 5-step smoke of `g2_main` + difficulty map build
  (`scripts/build_difficulty_map.py`, ~$4), then the P4/P5 main run with Gate B
  at step 150.
