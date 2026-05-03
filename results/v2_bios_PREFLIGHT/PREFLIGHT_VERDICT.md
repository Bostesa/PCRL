# BIOS Round 2 — CPU pre-flight verdict

**Decision: FAIL — no AWS launch.** Per the Round-2 plan: "If gate fails,
do NOT launch, accept §5.5 negative result framing."

## Run

- Mac, MPS, n_train=5000, 1 epoch, ~5 min wall.
- HEAD: see commit immediately following this file.
- Estimator stack: nHSIC primal + Theil-adjusted held-out R² dual + online
  LEACE refit (buffer=512, refit_every=10, shrinkage=True,
  constrain_cov_trace=True). All three Round-2 components active.

## Gate criteria & result

| criterion | threshold | observed | pass |
|---|---|---|---|
| dev marginal adj-R² | < 0.10 | 0.7073 | **FAIL** |
| dev TPR RMS gap | < 0.15 | 0.0494 | PASS |
| dev TPR max-abs gap | (informational) | 0.1334 | — |
| dev top-10 acc | (informational) | 0.5548 | — |

The gate is conjunctive (AND). adj_R² fails → overall FAIL.

## What worked

- **nHSIC primal signal moves cleanly in [0, 1]** instead of saturating at 1.0.
  In-batch nHSIC window means: step 0 → 0.000, step 50 → 0.030, step 100 →
  0.043, epoch avg → 0.049. Round 1's OLS R² was pinned at 1.0 throughout.
- **Holdout R² descends monotonically** under training: 0.989 → 0.965 → 0.949
  in 100 primal steps. Round 1 was frozen at 0.86–0.88 across 5 epochs.
- **λ does NOT saturate at λ_max=100** (final λ=6.99). Round 1's λ pinned at
  100 from epoch 0 onward. The dual is now actively responding.
- **Online LEACE refit ran 13 times** in 1 epoch (every 10 primal steps).
  No errors; refit cadence sustainable.
- **TPR-gap (BIOS standard fairness metric per De-Arteaga 2019) is competitive.**
  RMS=0.049, max=0.133. Belrose 2023 Table 3 LEACE baselines on BIOS report
  RMS ≈ 0.06–0.10 — the pre-flight is already inside that band after 1
  epoch on 5K samples.

## What failed

- **Adjusted-R² is high (0.707) on dev.** Theil's correction subtracts only
  ~0.007 of bias at d=768, n=31764 (the noise floor is genuinely tiny on
  full dev). The remaining 0.700 is real, model-induced gender info in
  [CLS]. The estimator stack moves it but slowly.

## Why the gate failed despite the favorable trajectory

The user's gate was set on the assumption that 1-epoch on 5K predicts where
12-epochs on 50K would land. The pre-flight evidence is mixed:

- **Trajectory says go:** the dual is responding, holdout R² is moving in
  the right direction at non-trivial speed (4 pp / 100 steps), TPR-gap is
  already competitive.
- **Static gate says stop:** the absolute value of dev adj-R² at the end of
  the pre-flight is 7× the gate threshold. Extrapolating linearly from the
  observed slope (0.04 / 100 steps), reaching 0.10 from 0.71 would take
  ~1500 steps — *might* fit in 12 epochs × ~1500 steps/epoch on n=50K, but
  the slope likely flattens.

## Per-occupation TPR gaps

| occupation | gap (M − F) |
|---|---|
| professor | +0.001 |
| physician | **−0.133** |
| attorney | +0.015 |
| photographer | +0.022 |
| journalist | +0.002 |
| nurse | +0.077 |
| psychologist | 0.000* |
| teacher | 0.000* |
| dentist | 0.000* |
| surgeon | 0.000* |

\* Likely zero because at 1 epoch on 5K samples the model never recovers
   the correct occupation for either gender on these tail classes — not
   evidence of fairness, just under-trained.

## Files

- `summary.json` — full args, gate decision, round2_estimator_stack metadata
- `history.jsonl` — single epoch with all new metric fields
- `monitor.jsonl` — 13 dual-refresh entries with raw + adj R²
- `checkpoint_final.pt` — model + task head (445 MB; can be deleted)
- `preflight.log` — full stdout
