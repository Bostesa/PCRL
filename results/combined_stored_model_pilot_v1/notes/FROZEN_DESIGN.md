# Frozen design for the CELL-A pilot (coordinator, written 2026-10-02 before any real-data fit)

Owned worktree: `/Users/nathansamson/PCRL/.worktrees/combined-stored-model-pilot-v1`. Branch:
`research/combined-stored-model-pilot-v1`, created from preparation commit 031860fbfd910b9aaed19b7600d288d15858dbc5.

**Fixed** (do not change):
- the checkpoint (PCRL Round-4 Adult seed 0, sha256 1cfc2fef…);
- the rows (PCRL Adult test split, 15,060 rows / 15,055 record units);
- the role assignment (`pilot-roles-v1` hash; attacker_fit 7,571 / attacker_val 2,239 / assessment 5,250);
- the 26-unit panel;
- the noise grid;
- the bars (0.52 / 0.55 / 0.60; 0.55 historical);
- the support rule (100 / 30 / 100);
- the exposure status (development data).

Private inputs: `~/PCRL_eval_cache_private/pilot_adult_s0/` (`features.npz`, `labels.npz`, `cache/adult_s0_test.npz`,
`releases/`, `manifest_*.json`). Private outputs go to `~/PCRL_eval_cache_private/pilot_adult_s0/run_v1/` and never
into git.

## Units (exactly 26; the runner must compare unit IDs, not counts)

- **U-untreated (8):** `manifest_<purpose>__<attr>.json` for the pairs income_prediction/{race,sex},
  employment_analysis/{race,age_group,marital_status}, education_assessment/{sex,race,income}.
- **U-noise (18):** `manifest_income_prediction__sex__p0_sigma{0.25,0.5,1,2,4,8}_seed{0,1,2}.json`.
- Release seeds are release draws over the same people, not independent populations or encoder seeds.

## Quantities per unit

All are computed on the same aligned assessment IDs.

### Linear, closed form
- **N0 historical native check (reproduction).** Fixed ridge λ = 1e-6 on the unnormalised centred Gram, one-hot pooled
  R², fit = score on all 15,060 test rows, untreated only. The historical code used a float32 Gram; replicate both
  float32 and float64. Compare with `origin/main:results/v2_adult_ROUND4/dominant_axis_audit.json`
  per_seed["0"].rows[*].r2_onehot.
  - This is the only fitting-row check: the historical check scored the test split in-sample.
  - Category: C1 if the reproduced value is > 0.05; otherwise "historical check passes".
  - Noise units: N0 is not defined historically. Mark NA (no historical check).
- **N1 within-assessment native.** Same estimator, fit = score on assessment rows. A separately named descriptive
  statistic, not a historical reproduction.
- **G1 held-out fixed-penalty native (primary P1).** Same normalisation, centring and λ = 1e-6 as N0 (the native
  quantity), fit on attacker_fit, scored on assessment. SS_tot is taken around the attacker_fit means; R² is not
  clamped. This is "generalisation of the matched native quantity".
- **G2 held-out relative-ridge (R02, secondary).** Scale-invariant probe. Reported separately; a G1–G2 difference is a
  penalty/scale effect, not generalisation.
- **ρ₁² held out (secondary).**
- **Pure metric contrast.** The same G1 fitted linear predictor (the one-hot least-squares predictions) is scored
  both as R² and as macro OvR AUC on assessment, using the predicted one-hot column as the score. This isolates
  R² → AUC for a fixed predictor.

### Attackers (sensitive-label recovery)

Fit on attacker_fit, select on attacker_val only, score on assessment.

- **L (linear):** multinomial logistic regression, grid C ∈ {0.01, 0.1, 1, 10}.
- **NL (nonlinear):**
  - GBT and MLP effective grids exactly as frozen in EFFECTIVE_PROTOCOL.json.
  - **The choice between GBT and MLP is made on attacker_val log-loss.** One "NL-selected" predictor per unit/surface.
  - Assessment never selects.
- **Surfaces:** rep, outputs, rep+outputs.
  - Untreated units: all three.
  - Noise units: rep and rep+outputs. The outputs surface is the clean model output, identical for every noise arm
    and seed, so its fitted attackers and predictions are **reused** from the untreated income_prediction/sex unit
    and marked `reused`, not new evidence.
- **Defense-informed noise attack (noise units only; A2 access).**
  - Gaussian class-conditional LRT on the released rep.
  - Fitted on released attacker_fit rows, knowing σ: class covariances estimated from released data minus σ²I,
    eigenvalues floored at 1e-6·trace/d. Means from released data (noise is mean-zero).
  - No clean vectors. Selection: none (closed form).
- **A4 white-box population LRT (stress test, labelled).**
  - The historical Tier-2 form: class Gaussians from **clean** attacker_fit reps plus known Σ = σ²I, scoring released
    assessment rows.
  - Reported separately; never used to refute a guarantee.
- **Repeated-query extension: STAGED, not run.**
  - The frozen release contract is one release per row per seed, a persistent draw per row. Under that contract N = 1.
  - A fresh-query contract is a different release interface and would need a separate amendment.
  - The runner must carry `ReleaseContract(noise="gaussian", sigma=σ, persistent=True)` from the manifest to the
    attacker access records. A global `noise="none"` is a bug.

### References and utility

- **Label-only reference (LO).** Predict s from the true task label y_task of the unit's purpose: a frequency table
  P(s | y_task) estimated on attacker_fit, scored on assessment (AUC and log-loss).
  - Output leakage beyond label-only = AUC(outputs, NL-selected) − AUC(LO), paired over the same assessment units.
  - The task labels for the three purposes: income_prediction → income; employment_analysis → the PCRL employment
    task label; education_assessment → the PCRL education task label. Take them from the PCRL b96c412 dataset object
    (`task_labels`), add them to `labels.npz` as new keys, and update the manifest hashes in the new lock.
- **Recovery metrics.** AUC (macro OvR over supported classes), log-loss reduction vs the attacker_fit prior
  `1 − LL(model)/LL(prior)`, and Brier skill. All on assessment.
- **U1 frozen-head utility.** The unit's purpose head applied to the released representation. Untreated: the stored
  clean logits. Noise: head(h + noise) by frozen forward, no fitting. Report task accuracy, log-loss and AUC on
  assessment against the untreated reference, with paired differences.
- **U2 independent utility probe.** Logistic regression (C grid as L) predicting the purpose's task label from the
  released representation. Fit on attacker_fit, select on attacker_val, score on assessment. Run for all 26 units.

### Support

Freeze support **before** inference, from class counts in all three roles.
- A class is supported if it has ≥ 100 rows in attacker_fit, ≥ 30 in attacker_val and ≥ 100 in assessment.
- A pair is supported if both its classes are supported.
- Fewer than 2 supported classes makes the unit NE.
- Macro AUC averages supported classes only. Worst-class and worst-pair use supported classes/pairs only.
- Coverage is reported. Unsupported classes keep reason codes.
- NE (not estimable) is distinct from UNRESOLVED (estimable, but the interval straddles the bar or τ).

## Inference

- **Sampling unit:** the assessment record unit (duplicates collapse). Cluster bootstrap over units with fitted
  predictors held fixed (no refitting).
- **Seed aggregation for noise arms:** average the statistic over the three release seeds *within* each bootstrap
  replicate (same resampled units). Report seed spread separately.
- **Exploratory intervals:** two-sided 90 % percentile, B = 2,000, seed 20261002.
- **Primary family:** 16 endpoints, frozen.
  - **P1-<pair> (8):** untreated G1 vs τ = 0.05. "Fails to generalise" if the one-sided lower bound is > 0.05.
    "Generalises" if the upper bound is ≤ 0.05.
  - **P2-<pair> (8):** untreated rep-surface NL-selected macro AUC vs bar 0.55. "Recovery outside scope established"
    if the lower bound is > 0.55. "Below bar established" if the upper bound is < 0.55.
  - **Adjustment:** Bonferroni simultaneous one-sided bounds at α = 0.05/16 = 0.003125 each. Percentile bootstrap
    with B = 20,000 (seed 20261003); tail count 62.5 replicates; resolution 5e-5.
  - Not Holm and not bootstrap p-values. These are simultaneous bounds.
  - **Validity note:** percentile-bootstrap bounds are asymptotic, not exact. This is recorded.
  - NE endpoints stay in the family (family size 16 fixed) and are reported NE.
- **Secondary (exploratory, unadjusted):** everything else, including all noise-arm endpoints, the bars 0.52/0.60,
  worst-class and worst-pair, LO contrasts, utility, the LRT attacks and the decomposition.

## Decomposition table (per untreated unit)

F0 native (N0) → F1 held-out same quantity (G1) → F2 same predictor scored as AUC (pure metric) → F3 logistic
regression AUC (L: linear model change) → F4 NL-selected AUC on rep (attacker family) → F5 outputs surface NL → F6
rep+outputs NL. One factor changes per step.

## Saved predictions (format both the runner and the independent replay rely on)

Per unit, write to `run_v1/units/<unit_id>/`:
- `preds.npz`, keys:
  - `assess_row_id`, `assess_unit`, `y_s`, `y_task`;
  - for every (surface, recipe): `P__<surface>__<recipe>`, the class probability matrix on assessment rows;
  - `G1_pred` and `G2_pred`, the one-hot least-squares predictions on assessment;
  - `G1_prior` and `G2_prior` (attacker_fit means);
  - `LO_P`;
  - `U1_logits`, `U2_P`.
- `fit_records.json`: effective configuration, selection tables (validation scores only), access records, timing.
- `models/`: pickled fitted estimators (joblib) for replay without retraining.
- `supported.json`: the frozen supported classes and pairs with counts per role.

Fitting only through `--execute-scientific-fits`. Thread count OMP_NUM_THREADS=1. One scheduler.

## Addendum D1 (coordinator, 2026-10-02, before any real-data fit; responds to role-1 mapping flags)

**Decisions**

| # | Issue | Decision |
|---|---|---|
| 1 | L grid | Use the **locked** preparation grid C ∈ {0.01, 0.1, 1, 10, 100} (5 values). The 4-value list above was a transcription error. |
| 2 | Log-loss metric | Report both under separate names. `LLR_nats` = LL₀ − LL (protocol/metrics definition). `LL_skill` = 1 − LL/LL₀. LL₀ = attacker_fit class prior. |
| 3 | Release contract | Map to the existing `ReleaseContract(noise="persistent_token", sigma=<manifest σ>)`. This is one persistent draw per row per release seed. The manifest `release` block must be read; the global `noise="none"` stamp is a bug. |
| 4 | A4 attack form | Gaussian class-conditional form: class means and covariances from **clean** attacker_fit rep + σ²I. This matches durable-guarantees `battery.py:60-103` (reconciliation DG-F01). The pinned point-mass mixture `NoiseLRTAttacker` is **not** used for A4. A2 (`noise-informed LRT` from released data, covariance − σ²I, eigen-floor) is new code. |
| 5 | Task labels and inputs | `task_labels_v1.npz` is a separate file. `labels.npz` is untouched, because `np.savez` timestamps would change its hash. v2 manifests for the 26 units declare the clean `rep_p0` (A4) and the hash-pinned checkpoint (U1 head) as inputs. |
| 6 | Saved arrays and quantiles | Held-out ρ₁² saves per-row `RHO_u`, `RHO_v` on assessment. Percentile quantiles use numpy `method="linear"` (Hyndman–Fan type 7) for every bound, primary and exploratory. |
| 7 | N0 naming and precision | N0 = "historical native check: in-sample on the PCRL **test** split". Reproduce the exact mixed precision (float32 centring and Gram, float64 solve, as in `eval_round4_dominant_axis.py:158-172`), plus a full-float64 variant. Noise units: N0 = **unverified** (durable-guarantees' historical approvals exist only on their own rows); not "not defined". |
| 8 | income_prediction/race | Historical N0 = 0.0575 > τ, so this pair is a known C1 from prior exposure. P1 for this pair stays in the 16-endpoint family, but its outcome is **not** read as C2: a failing held-out value adds no generalisation finding when the fitting-sample check already fails. |
| 9a | R11 (projection refit) | Deferred: not authorized. |
| 9b | Attacker refit seeds | Single refit seed (random_state = 0). Refit variance is **not** estimated; this is a stated limitation, and intervals are conditional on the fitted predictors. |
| 9c | 200-permutation null | Dropped for real units: it would require 200 refits per unit, well over the allowance. The synthetic null/positive controls run instead; amendment recorded. |
| 9d | Per-cell real-data controls | Dropped; synthetic controls through the same CLI path replace them. Recorded. |
| 9e | τ sensitivity | G1 decisions are reported at τ ∈ {0.01, 0.02, 0.05, 0.10} from the same exploratory intervals (no extra fits). Only τ = 0.05 is primary. |
| 9f | Task-label class support | Accuracy and log-loss use all assessment rows. Macro-F1 and per-class utility use task classes meeting 100/30/100; occupation class 5 (1 row) is excluded with a reason. |
| 9g | U2 probe | LR only; the MLP probe is dropped (amendment). Report absolute utility, the untreated reference and paired differences. Normalised lift only when clean lift over the constant predictor ≥ 0.03; otherwise flagged. |
| 9h | Private archive index | The coordinator produces it after the run. |
| 10 | Near-ceiling utility | employment_analysis and education_assessment task labels are deterministic recodings of inputs (head accuracy ≈ 0.9995). Their U1/U2/LO results are near ceiling and are reported with that note. |
| 11 | Validation support threshold | attacker_val minimum is **30**, not 100. The pinned admission's 100 is a bug for this rule. |
