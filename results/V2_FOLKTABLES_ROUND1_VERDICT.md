# V2 Folktables Round 1 — Verdict

**Status: PARTIAL (1 seed of 3 completed)**

## Run summary

- **Dataset:** California 2018 1-Year ACS PUMS via `folktables` (`pcrl/data/folktables.py`).
  N_train=248,720, N_val=31,090, N_test=31,091, D=190.
- **Purposes:** 3 (income_prediction, employment_analysis, public_coverage_assessment),
  joint cardinality cap 12 (binary RAC1P + binary SEX + DIS + 3-level AGEP_bin).
- **Config:** Round 5 — LoRA rank=8 alpha=16, lambda_min=5, warmup_when_leace_init=False,
  HEAD `4041b53` (Folktables) on top of `f542c06` (per-class OvR for K≥6, no-op here).
- **Instance:** g4dn.xlarge `[redacted-instance-id]`, launched 03:08 UTC.
- **Seed 0:** completed full 200 epochs in 4h 40min (16,817s).
  best_epoch=22, last_epoch=199, cotter=fallback feasible=0/200.
- **Seeds 1, 2:** did NOT complete. The 8h hard cap fired at 11:09 UTC, killing seed 1
  mid-training; seed 2 never started. Per-epoch wall on the instance was ~85s
  (single-threaded data loader, GPU 15% util) — at that pace 200 epochs × 3 seeds ≈ 14h,
  comfortably over the 8h cap. The `num_workers=0` data-loader setting in
  `experiments/run_v2_dataset.py` is the operative bottleneck.

## Headline (held-out test set, generate_report)

| selection | strict (R²<0.05) | combined (R²<0.05 AND delta<0.02) | mean R² |
|---|---|---|---|
| **best.pt** (epoch 22, Cotter task-loss fallback) | **8/8** | **3/8** | **0.009** |
| final.pt (epoch 199) | 5/8 | 0/8 | 0.072 |

best.pt clears the strict R² threshold on every (purpose, attribute) pair on the
held-out test set under the FFB binary-race convention. The combined criterion is
only met for the `public_coverage_assessment` purpose; for `income_prediction` and
`employment_analysis`, the encoder removes linear leakage but the post-hoc auditor
suite (LR + RF + RBF SVM + XGBoost) still recovers the attribute well above the
2-percentage-point delta threshold.

## Per-cell breakdown (seed 0)

### best.pt — Cotter selector pick (epoch 22)

| purpose | attribute | R² | delta | strict | combined |
|---|---|---|---|---|---|
| income_prediction | sex | 0.0272 | +0.2571 | ✓ | ✗ |
| income_prediction | race | 0.0441 | +0.2886 | ✓ | ✗ |
| employment_analysis | sex | 0.0000 | +0.1265 | ✓ | ✗ |
| employment_analysis | race | 0.0001 | +0.0251 | ✓ | ✗ |
| employment_analysis | disability | 0.0000 | +0.0853 | ✓ | ✗ |
| public_coverage_assessment | sex | 0.0003 | +0.0169 | ✓ | **✓** |
| public_coverage_assessment | race | 0.0000 | +0.0022 | ✓ | **✓** |
| public_coverage_assessment | age_group | 0.0023 | +0.0117 | ✓ | **✓** |

### final.pt — last training epoch (epoch 199)

| purpose | attribute | R² | delta | strict | combined |
|---|---|---|---|---|---|
| income_prediction | sex | 0.0694 | +0.0792 | ✗ | ✗ |
| income_prediction | race | 0.1327 | +0.0489 | ✗ | ✗ |
| employment_analysis | sex | 0.0232 | +0.1687 | ✓ | ✗ |
| employment_analysis | race | 0.0054 | +0.1122 | ✓ | ✗ |
| employment_analysis | disability | 0.0063 | +0.0949 | ✓ | ✗ |
| public_coverage_assessment | sex | 0.0019 | +0.0530 | ✓ | ✗ |
| public_coverage_assessment | race | 0.0140 | +0.0406 | ✓ | ✗ |
| public_coverage_assessment | age_group | 0.3201 | +0.2272 | ✗ | ✗ |

## Verdict

Under the spec's 1-seed proportional thresholds (≥75% strict pass = GREEN):

- **best.pt: GREEN (8/8 strict).** Eligible to enter the paper's main results table
  alongside Adult / HMDA / Diabetes under the FFB binary-race convention if a
  1-seed result is acceptable. (Standard paper hygiene asks for ≥3 seeds for
  variance estimation.)
- **final.pt: RED (5/8 strict, 0/8 combined).** Late-epoch drift is significant —
  same Cotter-fallback pattern Adult Round 4 exhibited.

## Diagnosis

1. **The encoder is achieving the constraint at epoch 22, then drifting.**
   `val_r2_mean` trajectory: 0.56 (post-LEACE-after-1-epoch) → **0.04 by epoch 20**
   → 0.05-0.06 (epoch 60-100) → drifts up to **0.11 by epoch 199**. Round 5's
   `lambda_min=5` floor is not strong enough to hold income_prediction at <0.05.

2. **Linear suppression vs auditor delta diverge sharply on this dataset.** On Adult
   Round 4, post-hoc delta was a leading indicator that the encoder still leaked.
   On Folktables, the encoder achieves R²≈0 on all employment/public_coverage
   pairs but the auditor still finds 0.03-0.27 lift over majority. Likely a strong
   *non-linear* axis (occupation × demographics interaction) that linear erasure
   does not touch.

3. **public_coverage_assessment is the easy purpose.** All 3 disallowed attrs
   simultaneously hit combined-pass on best.pt — the only purpose where this
   happens. Income prediction is the hard one (high majority delta because
   demographics are strongly predictive of income > 50k).

## Recommendation for paper integration

- **If 1-seed Folktables is acceptable for the paper's preliminary table:**
  Quote best.pt strict 8/8 (mean R² 0.009) and combined 3/8 with the caveat that
  this represents Cotter task-loss fallback selection (no feasible iterate found
  in 200 epochs). Note that the `combined` failures are post-hoc auditor delta,
  not linear-R², and reflect non-linear demographic correlations the linear
  certificate is not designed to suppress.

- **If the paper requires 3-seed variance:** Re-launch with `num_workers=4`
  in `experiments/run_v2_dataset.py` (currently 0), `batch_size=512` (currently
  256), and `--epochs 100` instead of 200 — the probe and seed-0 history both
  show convergence by epoch ~20 with subsequent drift, so 100 epochs captures the
  signal with much shorter wall. Estimated 3 seeds in ~4-5h on g4dn.xlarge with
  these changes.

- **If improving the *combined* metric matters for paper claims:** This is a
  representational issue, not a tuning issue. Consider adding a non-linear
  independence regularizer per Adult Round 4's recommendation (joint multivariate
  LEACE with HSIC or vCLUB at higher weight). The current vCLUB MI bound
  (`lambda_vclub=1.0`) appears too weak for Folktables' demographic-occupation
  coupling.

## Files

- `checkpoints/v2_folktables_ROUND1_s0/best.pt` — Cotter task-loss fallback (epoch 22)
- `checkpoints/v2_folktables_ROUND1_s0/final.pt` — last epoch (199)
- `/tmp/folktables_round1/orchestrator.log` — orchestration log
- `/tmp/folktables_failures.log` — failure notes (AWS auth expiry, slow data loader)
