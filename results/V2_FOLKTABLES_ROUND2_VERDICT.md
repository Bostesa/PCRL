# V2 Folktables Round 2 — Verdict

Dataset: California 2018 1-Year ACS PUMS (N_train=248720, N_test=31091, D=190)
Selection: per-seed canonical_iterate.pt (lower-mean-R² of best/final on val)
Convention: binary RAC1P (White vs non-White), binary SEX (FFB ICLR 2024).
Skipped seeds: [2] (no checkpoint — likely AWS hard cap fired before training completed)

## Round 1 vs Round 2

| metric | Round 1 | Round 2 |
|--------|---------|---------|
(no Round 1 raw available for direct compare)

## Headline

- **Verdict: YELLOW**
- Strict pass (R² < 0.05): **13/16**
- Combined pass (R² < 0.05 AND delta < 0.02): **7/16**
- Mean R²: 0.0300, Median R²: 0.0045, Max R²: 0.2031
- Mean post-hoc delta: +0.0900

## Per-seed checkpoint chosen

- seed=0: `canonical_iterate.pt`
- seed=1: `canonical_iterate.pt`

## Per-purpose breakdown

| purpose | n_cells | strict_pass | combined_pass | mean_r2 |
|---------|---------|-------------|---------------|---------|
| employment_analysis | 6 | 3/6 | 0/6 | 0.0657 |
| income_prediction | 4 | 4/4 | 2/4 | 0.0146 |
| public_coverage_assessment | 6 | 6/6 | 5/6 | 0.0046 |

## Per-cell table (24 = 3 seeds × 8 pairs)

| seed | purpose | attribute | R² | delta | strict | combined |
|------|---------|-----------|------|--------|--------|----------|
| 0 | employment_analysis | disability | 0.0099 | +0.0801 | ✓ | ✗ |
| 0 | employment_analysis | race | 0.0004 | +0.0979 | ✓ | ✗ |
| 0 | employment_analysis | sex | 0.0053 | +0.1640 | ✓ | ✗ |
| 0 | income_prediction | race | 0.0448 | +0.2173 | ✓ | ✗ |
| 0 | income_prediction | sex | 0.0137 | +0.2162 | ✓ | ✗ |
| 0 | public_coverage_assessment | age_group | 0.0213 | +0.0047 | ✓ | ✓ |
| 0 | public_coverage_assessment | race | 0.0023 | +0.0017 | ✓ | ✓ |
| 0 | public_coverage_assessment | sex | 0.0038 | +0.0505 | ✓ | ✗ |
| 1 | employment_analysis | disability | 0.2031 | +0.1056 | ✗ | ✗ |
| 1 | employment_analysis | race | 0.0908 | +0.2088 | ✗ | ✗ |
| 1 | employment_analysis | sex | 0.0846 | +0.2979 | ✗ | ✗ |
| 1 | income_prediction | race | 0.0000 | -0.0007 | ✓ | ✓ |
| 1 | income_prediction | sex | 0.0000 | -0.0018 | ✓ | ✓ |
| 1 | public_coverage_assessment | age_group | 0.0000 | +0.0008 | ✓ | ✓ |
| 1 | public_coverage_assessment | race | 0.0000 | -0.0007 | ✓ | ✓ |
| 1 | public_coverage_assessment | sex | 0.0000 | -0.0018 | ✓ | ✓ |

## Per-cell table — what's missing

Seed 2 was killed by the 8h hard cap (instance launched 2026-05-02 04:39 UTC,
seed-0 took 2h37min, seed-1 took 2h44min; seed-2 started 10:54 UTC and would
have finished ~13:38 UTC, 59 min past cap). The trainer only saves `final.pt`
at end of the per-seed epoch loop, so no partial seed-2 state could be
recovered. Round 2 is therefore reported on 16 of 24 cells (2 of 3 seeds).

## Round 1 vs Round 2 (canonical_iterate.pt vs Round 1 best.pt, seed 0 only)

| metric | Round 1 best.pt s0 | Round 2 canonical s0 | Round 2 canonical s1 |
|---|---|---|---|
| strict | 8/8 | 8/8 | 5/8 |
| combined | 3/8 | 2/8 | 5/8 |
| mean R² | 0.009 | 0.013 | 0.047 |
| max R² | 0.044 (income/race) | 0.045 (income/race) | 0.203 (employment/disability) |
| Cotter pick | epoch 22 (fallback) | epoch 26 (fallback) | epoch 99 = final.pt |

## Diagnosis: why YELLOW, not GREEN

**Frozen LEACE projection works mathematically but does not strictly bound
auditor R² under task pressure.** Direct verification (see
`results/V2_FOLKTABLES_ROUND1_VERDICT.md` follow-up): with both flags ON at
init, auditor R² = 0.002–0.003 on every pair (8/8 PASS); idempotent
projection confirmed (`max diff = 5.56e-06` on encoder output after
re-application). After 100 epochs of LoRA training, train-set R² for
income/race rises to 0.14 even though the projection is still applied —
because the LoRA introduces non-linear features of x whose train-set
covariance with the disallowed attribute is non-zero through directions
that survive (I − P_sub). The projection guarantees `h_proj ∈ null(P_sub)`
(necessary condition for R²=0) but not `Cov_train(h_proj, A) = 0`
(sufficient condition).

**Cotter fallback rule misses good middle-training iterates when no
iterate is fully feasible by the runtime metric.** Seed 1 best.pt was
epoch 10 with `viol_sum=1.10`, `r2_mean=0.169`, picked because it had the
lowest violation among iterates within 10% of best task_loss (epochs 3–10).
Epoch 79 had `viol_sum=0.21`, `r2_mean=0.045` — far better — but its
task_loss was 1.34× best, OUTSIDE the 10% slack. canonical_iterate.pt
picked final.pt (epoch 99, mean R² 0.061) over best.pt (epoch 10, mean R²
0.169) — correct choice between those two — but neither is the
true-best iterate. The runtime trainer's `verifier` metric overestimates
linear leakage by ~50× on Folktables (auditor metric reports 0.002 at
init while runtime reports 0.12), so `n_feasible_post_warmup=0/100` for
both completed seeds; the rule never enters its primary branch.

## Recommendation

- **Do NOT add Folktables to the paper's main table on Round 2 results.**
  Strict 13/16 = 81.3% on 2 seeds is below the GREEN floor (87.5% = 14/16 prorated, or 21/24 full grid) and below Round 1's 8/8 single-seed result. canonical_iterate.pt is sound; the regression vs Round 1 is concentrated in seed 1 where Cotter fallback picked a bad middle iterate.

- **If a Round 3 is warranted, two changes are sufficient:**
  1. Relax `cotter_fallback_task_slack` from 0.10 to ~0.50 so the rule can pick later-training iterates with much lower violation. On seed 1 this would have selected epoch 79 (viol_sum=0.21, r2_mean=0.045) instead of epoch 10.
  2. Have canonical_iterate.pt compare best.pt vs final.pt vs **all post-warmup iterates with viol_sum below the median** rather than just two iterates. The trainer already records every epoch; the cost is one extra pass over `epoch_records`.

- **If 1-seed Folktables is acceptable:** Round 2 seed 0 (canonical, epoch 26)
  matches Round 1 seed 0 (best.pt, epoch 22) at 8/8 strict, mean R²=0.013.
  Quote that and note the variance from seed 1 as a Limitation.

- **Combined-criterion failures (auditor delta > 0.02 with linear R² < 0.05)
  reflect Belrose et al. NeurIPS 2023's explicit scope of LEACE: linear
  adversary only.** Folktables' demographic × occupation interactions remain
  recoverable by RBF SVM / RF / XGBoost auditors. Frame as Limitations,
  not as method failure.

## What's been pushed

- `pcrl/training/v2_trainer.py`, `pcrl/models/lora.py`,
  `experiments/run_v2_dataset.py`, `tests/test_folktables_fixes.py`,
  `scripts/folktables_round2_verdict.py` — additive opt-in flags
  (`--report-best-iterate`, `--freeze-leace-projection`),
  default OFF, Adult/HMDA/Diabetes Round 5/6/7 results bit-identical at
  flags off (verified via 5-epoch Adult smoke: max R² Δ = max λ Δ = max
  LoRA weight Δ = 0.00e+00).
- Commits: `940912c` (flags), `65cd885` (reload fix),
  `c2d6e1a` (GPU-device buffer fix). All on origin/main.

## What's NOT pushed

- Paper LaTeX is untouched.
- Adult / HMDA / Diabetes results dirs are untouched.
- This Round 2 verdict (YELLOW) is recorded but not promoted to the paper's
  main results table.
