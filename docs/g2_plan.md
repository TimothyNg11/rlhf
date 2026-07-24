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

## Execution status

- P0+P1 (measurement layer + trainer threading + configs) landed at commits
  0c71c0d, d2fb23a, 3fea7d6; suite 344 green on Windows CPU.
- Next: pod session P2 — 5-step smoke of `g2_main` + difficulty map build
  (`scripts/build_difficulty_map.py`, ~$4), then the P4/P5 main run with Gate B
  at step 150.
