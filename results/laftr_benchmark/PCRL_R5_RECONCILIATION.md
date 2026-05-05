# PCRL R5 reconciliation: 56/60 vs 54/60

**TL;DR — your paper memory is right, my Stage 2 number was wrong.** I quoted
the wrong field. The reconciled headline under the canonical (post-hoc, test-set
auditor) criterion is **56/60**, matching your memory. My Stage 2 report's
"54/60 / 21/24" came from a *different* field (`linear_r2` in
`per_seed_results.json`, which is the train-time R² that the verdict commit
explicitly says is **not** comparable to the published numbers).

## 1. Field names actually logged in `per_seed_results.json`

`results/v2_adult_ROUND5/per_seed_results.json` per `attribute_results` entry:

```json
{
  "purpose":             "income_prediction",
  "attribute":           "race",
  "linear_r2":           0.010619,        // train-time, history.r2_per_pair_per_epoch[-1]
  "empirical_best_acc":  0.861222,        // best post-hoc auditor accuracy
  "majority_baseline":   0.861319,
  "delta":               -9.7e-05,        // best_emp_acc − majority
  "adj_pass":            true             // (delta ≤ ε) — empirical-Δ adjusted
}
```

Per `per_seed[i].per_purpose_health[purpose]`:
`per_dim_std_mean`, `per_dim_std_min`, `per_dim_std_max`, `effective_rank`,
`l2_norm_mean`, `l2_norm_std`, `shape`.

`results/v2_*_ROUND*/dominant_axis_audit.json` per `per_seed[s].rows[i]`:
```json
{
  "purpose": "...",  "attribute": "...",  "num_classes": 5,
  "r2_onehot":   0.014233,       // CANONICAL auditor R²_onehot (test-set OLS)
  "r2_da":       0.018604,       // CANONICAL auditor R²_DA   (max OvR R²)
  "mlp_da_delta": 0.000465,      // MLP-DA Δ (post-hoc nonlinear leakage)
  "per_class_r2": [...],  "priors": [...],
  "predicted_r2_onehot_from_convex_combo": ...,
  "is_multiclass": true,
  ...
}
```

**Three R² values exist per cell**, computed at different stages:
1. **`linear_r2`** (per_seed_results.json) — *train-time* R²,
   `history.r2_per_pair_per_epoch[-1]`, batch-averaged during training.
2. **`r2_onehot`** (dominant_axis_audit.json) — *post-hoc auditor R²* on the
   full test set, computed by `LinearComplianceCertificate.check()`. **This is
   the canonical published metric.**
3. **`r2_da`** (dominant_axis_audit.json) — dominant-axis R², `max_k OvR_R²_k`.

Plus an empirical accuracy-based criterion:
- **`adj_pass`** (per_seed_results.json) — `True` if
  `empirical_best_acc − majority_baseline ≤ ε`. Distinct from R²; uses the
  best post-hoc *attack* accuracy from a suite of auditors (linear, MLP, etc.).

## 2-4. Counts under each criterion

| Dataset | N | train-time `linear_r2` ≤ 0.05 | **auditor `r2_onehot` ≤ 0.05** | auditor `r2_da` ≤ 0.05 | empirical `adj_pass` |
|---------|---:|---:|---:|---:|---:|
| Adult     | 24 | 21/24 | **23/24** | 23/24 | 12/24 |
| HMDA      | 18 | 16/18 | **16/18** | 15/18 |  3/18 |
| Diabetes  | 18 | 17/18 | **17/18** | 17/18 | 17/18 |
| **TOTAL** | 60 | 54/60 | **56/60** | 55/60 | 32/60 |

**Direct answer:** 23/24 on Adult comes from the **auditor `r2_onehot`** in
`dominant_axis_audit.json` (the post-hoc test-set OLS). 21/24 comes from the
**train-time `linear_r2`** in `per_seed_results.json` (history's last-batch
average). The two cells that flip between 21 and 23 are both Adult/seed_0/
employment_analysis: `age_group` (train 0.0619 → audit 0.0054) and
`marital_status` (train 0.0806 → audit 0.0033).

## 5. One full cell entry

```json
// per_seed_results.json — Adult seed_0 income_prediction/race
{
  "purpose": "income_prediction",
  "attribute": "race",
  "linear_r2": 0.010619,
  "empirical_best_acc": 0.861222,
  "majority_baseline": 0.861319,
  "delta": -0.000097,
  "adj_pass": true
}

// dominant_axis_audit.json — Adult seed_0 income_prediction/race  (same cell)
{
  "purpose": "income_prediction", "attribute": "race", "num_classes": 5,
  "r2_onehot": 0.014233,                            // canonical auditor R²
  "r2_da": 0.018604, "r2_da_argmax_class": 4,
  "per_class_r2": [0.000999, 0.013014, 0.010505, 0.008740, 0.018604],
  "priors": [0.00989, 0.02709, 0.09369, 0.00810, 0.86122],
  "predicted_r2_onehot_from_convex_combo": 0.014233,
  "convex_combo_residual": 7.06e-06,
  "is_multiclass": true,
  "mlp_da_delta": 0.000465, "mlp_da_argmax_class": 4,
  "mlp_per_class_delta": [0.0, 0.000199, 6.6e-05, -0.000133, 0.000465],
  "mlp_per_class_acc": [0.9901, 0.9731, 0.9064, 0.9918, 0.8617],
  "wall_seconds": {"linear": 0.016, "mlp": 46.17}
}

// per_seed_results.json — Adult seed_0 per_purpose_health.income_prediction
{
  "shape": [15060, 64],
  "per_dim_std_mean":  0.2854,
  "per_dim_std_max":   0.6413,
  "per_dim_std_min":   0.1226,
  "l2_norm_mean":      8.7682,
  "l2_norm_std":       1.0252,
  "effective_rank":    5.2918
}
```

## 6. Was the JSON modified after training?

`per_seed_results.json` mtime: 2026-04-30 01:45:20 (commit `a00d574`)
`dominant_axis_audit.json` mtime: 2026-05-01 23:10:15 (later)
`summary.json` mtime: 2026-04-30 01:45:20 (same as per_seed_results)

**Git history relevant commits:**
- `a00d574` "V2 Round 5 results: PARTIAL" — initial commit, reports
  Adult **22/24** strict R²<0.05 on `final.pt`. JSONs created here.
- `2efd2a3` "V2 Round 5 verdict: replace train-time R² with auditor R² (canonical)" —
  later commit, updates the verdict markdowns to **23/24**, but does **not**
  re-save `per_seed_results.json` with the auditor R². Instead the canonical
  numbers were materialized in a new file: `dominant_axis_audit.json`.

So the JSON wasn't modified — there are simply two files holding two metrics.
**`per_seed_results.json:linear_r2` is *not* the published metric**; it's a
train-time number kept for reference. The published metric is
`dominant_axis_audit.json:r2_onehot`. My Stage 2 report read the wrong file.

The verdict commit message explicitly warns this:
> Earlier draft of V2_ROUND5_VERDICT.md used `history.r2_per_pair_per_epoch[-1]`
> (train-time per-batch averaged R²) which is not comparable to Round 4's
> canonical baseline (test-set auditor R² via `generate_report`). With the
> auditor metric, Round 4 → Round 5 strict R²<0.05: Adult: 20/24 → **23/24**.

## 7. Collapse vs strict-pass: independent measurements

**Yes, they are independent. A cell can pass strict AND be collapsed
simultaneously — and most cells in fact do.**

- Collapse measures the **encoder output distribution** at test time:
  `per_dim_std_mean < 0.5` (most dims have low variance) OR `effective_rank
  < 2.0` (the rank of the test-set covariance is degenerate).
- Strict-pass measures **R² of OLS regression of encoder reps onto attribute
  one-hot labels**.

Critically, **a collapsed encoder is *more likely* to pass strict.** If the
encoder produces a near-constant output, OLS predicting any attribute from
that constant explains almost zero variance → low R² → easy strict pass. The
extreme case is HMDA/underwriting/seed_0 with `per_dim_std_mean = 0.000`
(literally a constant representation): every R² is essentially 0, every cell
passes strict, but the encoder is a degenerate rank-1 matrix and the task head
recovers majority predictor.

**Concrete overlap on the R5/R7 numbers:**

| Bucket (60 (dataset, purpose, seed, attr) cells) | Count |
|---|---:|
| Cells COLLAPSED (per_dim_std<0.5 OR eff_rank<2)               | 53/60 (88%) |
| Cells passing strict on auditor R²_oh ≤ 0.05                  | 56/60 (93%) |
| Cells **BOTH collapsed AND strict-pass** ("compliance via collapse")  | **49/60 (82%)** |
| Cells **strict-pass AND not collapsed** (cleanly compliant)   | **7/60 (12%)** |

The 7 cleanly-compliant cells are concentrated in 3 (dataset, purpose, seed)
groups: Adult/employment_analysis/seed_0 (3 attrs), HMDA/underwriting/seed_2
(2 attrs), Diabetes/billing_audit/seed_2 (2 attrs).

So under the canonical metric, PCRL R5 strict-passes 56/60. **Of those 56
passes, 49 (82%) are accompanied by representation collapse, and only 7 (12%)
are clean.** The headline number is right. The mechanism behind most of those
56 passes is *partial representation collapse*, not utility-preserving
suppression. That distinction is what the §5.3 task-acc analysis was probing
in the prior diagnostic.

## Master reconciliation table

Per-cell columns: `linear_r2_traintime`, `r2_onehot_auditor`, `r2_da_auditor`,
`mlp_da_delta`, `empirical_delta`, `adj_pass`, `majority_baseline`,
`empirical_best_acc`, `per_dim_std_mean`, `effective_rank`, `task_acc`,
`task_acc_unconstrained`.

- CSV: `results/laftr_benchmark/PCRL_R5_RECONCILIATION.csv` (60 rows)
- JSON: `results/laftr_benchmark/PCRL_R5_RECONCILIATION.json`

## What this means for §5.3 and Stage 2

1. **§5.3 strict-pass headline of 56/60 is correct under the canonical
   auditor metric** (`r2_onehot` from `dominant_axis_audit.json`). The
   memory you cited matches the verdict commit. The 54/60 in my Stage 2
   report is the *train-time* metric, which the verdict commit explicitly
   labels "not comparable" to the published numbers.

2. **The collapse concern from the previous diagnosis still stands**, but
   with corrected mechanism wording: of the 56 strict passes, 49 (82%)
   coincide with representation collapse and 7 (12%) are clean.

3. **The §5.3 "task accuracy within 1pp of unconstrained" claim is still
   problematic** for the same reasons in the previous diagnosis (5/9
   purpose-task cells fail 1pp, 2/9 hit 1pp via task-head reverting to
   majority predictor). The reconciliation here doesn't change that.

4. **My Stage 2 LAFTR-vs-PCRL report needs an erratum**: `STAGE2_ADULT.md`
   should quote PCRL **23/24 (auditor R²_oh)**, not 21/24. I'll patch it
   when you confirm next steps.

## Files written

- `results/laftr_benchmark/PCRL_R5_RECONCILIATION.md` (this file)
- `results/laftr_benchmark/PCRL_R5_RECONCILIATION.csv` (60-row master table)
- `results/laftr_benchmark/PCRL_R5_RECONCILIATION.json` (same, JSON)
