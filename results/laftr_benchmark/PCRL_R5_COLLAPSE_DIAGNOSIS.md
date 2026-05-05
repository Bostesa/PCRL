# PCRL R5 collapse diagnosis (read-only, no retraining)

## Q1 — "Stage 2 PCRL evaluation": clarification

Stage 2 did **not** re-evaluate PCRL. Stage 2 trained 9 LAFTR encoders on Adult
and read the PCRL R5 numbers verbatim from the existing
`results/v2_adult_ROUND5/per_seed_results.json`. The collapse note in the
Stage 2 report was lifted from that file's existing `health_notes` field.
There is **one** set of PCRL R5 Adult metrics, produced by the original
ROUND5 training run, and that is the set Stage 2 reported and that this
diagnosis examines.

## Q2 — Are these the same checkpoints / were they reloaded or modified?

The numbers come from the same `per_seed_results.json` produced at training
time by the ROUND5 trainer. Nothing has been re-trained, re-loaded, or
modified since. The .pt checkpoints from ROUND5 are not on disk in this
repo (the repo stores summary JSON only — see `ls results/v2_adult_ROUND5/`),
so any re-evaluation would have to re-train. **No re-evaluation was needed**:
the diagnostics in question were already logged at training time as the
`per_purpose_health` block per (seed, purpose), one tier deeper than the
top-level `health_notes` summary the Stage 2 report cited.

## Q3 — Per-cell `per_dim_std` and `effective_rank` from `per_seed_results.json`

Threshold (matches `summary.json.thresholds`): collapse iff
`per_dim_std_mean < 0.5` **OR** `effective_rank < 2.0`.

### Adult (results/v2_adult_ROUND5/)

| seed | purpose | per_dim_std_mean | per_dim_std_min | effective_rank | flag |
|-----:|---------|-----------------:|----------------:|---------------:|------|
| 0 | income_prediction        | **0.285** | 0.123 | 5.29 | COLLAPSED (std<0.5) |
| 0 | employment_analysis      | 0.534     | 0.273 | 8.82 | ok |
| 0 | education_assessment     | **0.281** | 0.066 | **1.20** | COLLAPSED (std<0.5, rank<2) |
| 1 | income_prediction        | **0.332** | 0.104 | **1.95** | COLLAPSED (std<0.5, rank<2) |
| 1 | employment_analysis      | **0.411** | 0.084 | 2.24 | COLLAPSED (std<0.5) |
| 1 | education_assessment     | **0.403** | 0.356 | **1.69** | COLLAPSED (std<0.5, rank<2) |
| 2 | income_prediction        | **0.290** | 0.117 | 5.42 | COLLAPSED (std<0.5) |
| 2 | employment_analysis      | **0.475** | 0.233 | 2.44 | COLLAPSED (std<0.5) |
| 2 | education_assessment     | **0.297** | 0.044 | **1.06** | COLLAPSED (std<0.5, rank<2) |

**Adult: 8/9 cells collapse.** Only employment_analysis/seed_0 is healthy.

### HMDA (results/v2_hmda_ROUND5/)

| seed | purpose | per_dim_std_mean | per_dim_std_min | effective_rank | flag |
|-----:|---------|-----------------:|----------------:|---------------:|------|
| 0 | underwriting        | **0.000** | 0.000 | **1.00** | COLLAPSED (std<0.5, rank<2) — **point collapse (std=0)** |
| 0 | pricing_analysis    | **0.474** | 0.234 | 8.19 | COLLAPSED (std<0.5) |
| 0 | fair_lending_audit  | **0.171** | 0.030 | **1.14** | COLLAPSED (std<0.5, rank<2) |
| 1 | underwriting        | **0.195** | 0.192 | **1.00** | COLLAPSED (std<0.5, rank<2) |
| 1 | pricing_analysis    | **0.400** | 0.195 | 2.30 | COLLAPSED (std<0.5) |
| 1 | fair_lending_audit  | **0.315** | 0.036 | **1.03** | COLLAPSED (std<0.5, rank<2) |
| 2 | underwriting        | 0.508     | 0.356 | 7.35 | ok |
| 2 | pricing_analysis    | **0.460** | 0.242 | 3.50 | COLLAPSED (std<0.5) |
| 2 | fair_lending_audit  | **0.324** | 0.141 | **1.33** | COLLAPSED (std<0.5, rank<2) |

**HMDA: 8/9 cells collapse.** HMDA underwriting/seed_0 is **point collapse**
(`per_dim_std = 0.0` exactly — every dim is constant across all 15K test points).

### Diabetes (results/v2_diabetes_ROUND7/)

| seed | purpose | per_dim_std_mean | per_dim_std_min | effective_rank | flag |
|-----:|---------|-----------------:|----------------:|---------------:|------|
| 0 | billing_audit             | **0.301** | 0.147 | 3.37 | COLLAPSED (std<0.5) |
| 0 | quality_research          | **0.150** | 0.059 | **1.75** | COLLAPSED (std<0.5, rank<2) |
| 0 | clinical_decision_support | **0.426** | 0.194 | 4.95 | COLLAPSED (std<0.5) |
| 1 | billing_audit             | **0.438** | 0.200 | 4.12 | COLLAPSED (std<0.5) |
| 1 | quality_research          | **0.258** | 0.046 | **1.48** | COLLAPSED (std<0.5, rank<2) |
| 1 | clinical_decision_support | **0.313** | 0.126 | 2.82 | COLLAPSED (std<0.5) |
| 2 | billing_audit             | 0.548     | 0.396 | 6.98 | ok |
| 2 | quality_research          | **0.361** | 0.190 | 7.85 | COLLAPSED (std<0.5) |
| 2 | clinical_decision_support | **0.306** | 0.091 | 2.01 | COLLAPSED (std<0.5) |

**Diabetes: 8/9 cells collapse.**

### Cross-dataset summary

| Dataset | cells collapsed | strict pass (R²≤0.05) |
|---------|-----------------|----------------------|
| Adult    | 8/9 | 21/24 |
| HMDA     | 8/9 | 16/18 |
| Diabetes | 8/9 | 17/18 |
| **TOTAL**| **24/27 (89%)** | **54/60** |

Note: the published headline that came up earlier ("56/60") doesn't match the
linear-R²≤0.05 strict count (54/60). The 56/60 likely uses the
empirical-Δ adjusted-pass criterion or a different per-seed roll-up. The
strict count *I can verify from the JSONs is 54/60.*

## Q4 — Task accuracy on collapsed cells vs unconstrained ("Standard, no privacy")

Unconstrained references: `results/{hmda,diabetes}/standard_ceiling.json` (seed 0,
StandardEncoder MLP[128,128]→64, λ=0). Adult unconstrained for income from
`results/adult/extended_baselines.csv` (single seed). For Adult
occupation_group / education_level, no stored unconstrained number, but those
tasks are deterministic functions of one-hot-encoded feature columns, so the
ceiling is essentially 1.000.

| dataset | purpose | task | PCRL R5 mean acc | per-seed | unconstrained | gap (pp) | within 1pp? |
|---------|---------|------|-----------------:|----------|--------------:|---------:|:-----------:|
| adult    | income_prediction         | income                       | **0.7873** | [0.769, 0.820, 0.773] | 0.8495 | **−6.22** | **NO** |
| adult    | employment_analysis       | occupation_group             | 0.9920 | [0.989, 0.994, 0.993] | ≈1.000 | ~−0.8 | YES (deterministic) |
| adult    | education_assessment      | education_level              | 0.9995 | [1.000, 0.999, 1.000] | ≈1.000 | ~−0.05 | YES (deterministic) |
| hmda     | underwriting              | loan_decision                | **0.8979** | [0.893, 0.893, 0.909] | 0.9123 | **−1.44** | **NO** (and 2/3 seeds = majority 0.8926 exactly) |
| hmda     | pricing_analysis          | loan_amount_band             | **0.4980** | [0.444, 0.504, 0.545] | 0.6136 | **−11.56** | **NO** |
| hmda     | fair_lending_audit        | tract_denial_high            | **0.6341** | [0.634, 0.634, 0.634] | 0.6585 | **−2.44** | **NO**; **all 3 seeds = majority 0.6341 exactly** (encoder ignored) |
| diabetes | billing_audit             | primary_diagnosis_category   | **0.2864** | [0.169, 0.337, 0.353] | 0.4149 | **−12.85** | **NO** |
| diabetes | quality_research          | readmission_outcome          | 0.9119 | [0.912, 0.912, 0.912] | 0.9119 | +0.00 | YES — but **all 3 seeds = majority 0.9119 exactly** (encoder ignored) |
| diabetes | clinical_decision_support | medication_change_outcome    | 0.9992 | [1.000, 0.999, 0.999] | 0.9996 | −0.04 | YES |

**Within-1pp count: 4 of 9 (purpose, dataset) cells.** Of those 4:
- 2 cells hit "within 1pp" *only because the encoder collapsed and the task head
  learned the majority predictor* (HMDA/fair_lending, Diabetes/quality_research —
  every seed's task_acc equals majority to ≤1e-3).
- 2 cells hit "within 1pp" because the task is deterministic from features
  (Adult/occupation_group, Adult/education_level — both methods ≈ 1.000).

**Genuine within-1pp on a non-trivial task (i.e. unconstrained > majority + 1pp
*and* PCRL within 1pp of unconstrained): 1 of 9 cells** —
Diabetes/clinical_decision_support (a 0.55→1.00 task where PCRL gets 0.999).

## Q5 — Honest assessment

**The §5.3 paper claim of "task accuracy within 1pp of unconstrained on every
purpose" does not hold on the existing R5/R7 checkpoints.** Specifically:

1. **5 of 9 (dataset, purpose) task cells fail the 1pp test outright** —
   Adult/income (−6.22pp), HMDA/loan_decision (−1.44pp), HMDA/loan_amount_band
   (−11.56pp), HMDA/tract_denial (−2.44pp), Diabetes/primary_diagnosis (−12.85pp).
   These are the cells where the unconstrained encoder genuinely beats majority
   by a meaningful margin and PCRL drops well below it.

2. **2 of the 4 cells that pass the 1pp test pass it degenerately** —
   HMDA/fair_lending and Diabetes/quality_research return *exactly* the majority
   baseline on all 3 seeds (to 4 decimal places). The encoder has collapsed and
   the task head is predicting majority. They are nominally "within 1pp" only
   because unconstrained also happens to be near majority on those tasks (both
   are themselves close to no-signal).

3. **Collapse is the rule, not the exception.** 24 of 27 (89%) (seed, purpose)
   cells across the three datasets fall below the published collapse thresholds
   (`per_dim_std_mean < 0.5` OR `effective_rank < 2.0`) per the same
   `per_seed_results.json` files the strict-pass headline is derived from. One
   cell (HMDA/underwriting/seed_0) has `per_dim_std = 0.0` exactly — point
   collapse to a constant representation. The strict-pass headline of
   ~54/60 R²≤0.05 reflects the fact that a near-constant encoder cannot leak
   anything.

4. **The "compliance via shared backbone with K LoRAs preserves task acc"
   architectural argument is not supported by the current numbers.** It holds
   only on tasks that are deterministic from features (where any encoder
   trivially solves the task) or where the unconstrained baseline is also at
   majority (where there's no task to lose accuracy on). On the only
   non-deterministic tasks where a real classifier can outperform majority by
   ≥5pp — Adult/income, HMDA/loan_amount_band, Diabetes/primary_diagnosis —
   PCRL takes a 6-13pp hit.

### Recommendation for §5.3

The "within 1pp" claim should be replaced with what the data supports:

- **As written:** "PCRL achieves 54/60 strict compliance (R²_onehot ≤ 0.05)
  with task accuracy within 1pp of unconstrained on every purpose."
- **As honest:** "PCRL achieves 54/60 strict compliance via partial
  representation collapse (per_dim_std<0.5 on 24/27 cells) and partial
  task-head reversion to majority predictor (3/9 task cells exactly = majority
  across all seeds). Where the task is genuinely non-trivial (Adult/income,
  HMDA/loan_amount_band, Diabetes/primary_diagnosis_category), PCRL trails the
  unconstrained ceiling by 6.2–12.9pp. The compliance/utility frontier
  reported by R5/R7 is therefore a *partially-collapsed* compliance, not a
  utility-preserving compliance, on most cells."

This is fixable in two directions:
- **(a) Re-train.** PCRL needs a representation-health regularizer (rank /
  per-dim-std lower bound) added to the proxy-Lagrangian objective and a
  re-run. The Stage 2 / Stage 3 LAFTR comparison in the paper should use the
  re-trained PCRL R6+ checkpoints, not the R5/R7 ones.
- **(b) Re-frame the contribution.** Drop the "task-acc preservation" sub-claim
  and pitch the strict-pass result as "PCRL provides a verifiable compliance
  primitive at the cost of task utility on non-deterministic predictive tasks,
  with quantified per-task degradation in §5.3." The collapse becomes a
  documented limitation, not an accidental result.

Either path needs to land in the paper. The current numbers do not support
the §5.3 wording.

## Per-cell artifacts

- `results/v2_adult_ROUND5/collapse_diagnostic.json` — per-cell std/rank/flags (Adult)
- `results/v2_adult_ROUND5/task_acc_vs_unconstrained.json` — task acc vs unconstrained
- This document: `results/laftr_benchmark/PCRL_R5_COLLAPSE_DIAGNOSIS.md`
