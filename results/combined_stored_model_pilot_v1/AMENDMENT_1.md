# Amendment 1: executable-contract repairs before the CELL-A pilot

**Written:** 2026-10-02T17:03Z (UTC). This is **before any real-data attacker or utility-probe fit**: the private run
directory `run_v1/` contained only `PLAN.json`, `DRY_RUN.json` and `inputs/`, and no `units/`.

**Basis:** known development artifacts. The pinned preparation source was read
(`research/combined-evaluation-preparation-v1` @ 031860fbfd910b9aaed19b7600d288d15858dbc5). The original
`PILOT_LOCK.json` (preparation) is preserved unchanged as the original record.

**Fixed, unchanged by this amendment:**
- checkpoint: PCRL Round-4 Adult seed 0, sha256 1cfc2fef…c061;
- rows: PCRL Adult test split, 15,060 rows / 15,055 units;
- role assignment `pilot-roles-v1` (7,571 / 2,239 / 5,250);
- the 26-unit panel;
- noise grid σ ∈ {0.25, 0.5, 1, 2, 4, 8} × release seeds {0, 1, 2};
- bars 0.52 / 0.55 / 0.60;
- support 100 / 30 / 100;
- exposure status (development data);
- the scientific question.

## Has any assessment outcome been seen?

**No new assessment outcome for any of the 26 units has been seen.** The preparation and this repair phase ran
only:
- frozen forward passes;
- admission;
- dry runs;
- synthetic tests;
- recounts of stored historical values;
- a check that the frozen heads reproduce the stored logits (task-label outputs; no sensitive-attribute recovery
  was computed).

**This is not a blinded study.** Prior exposure, already seen and motivating the design:

| Source | What was seen |
|---|---|
| `origin/main:results/v2_adult_ROUND4/dominant_axis_audit.json` | Historical native values on this checkpoint and these rows, e.g. income_prediction/race r2_onehot = 0.0575, above τ |
| `per_seed_results.json` | EmpiricalAudit held-out accuracies on this encoder |
| durable-guarantees' XGB/MLP attack AUCs on this encoder | Including the income/sex noise arms, on other rows (the PCRL train partition) |

The cell, the noise grid and the bars were all chosen knowing these.

## Changes: original versus executed

Full original-side detail is in `notes/mapping/amendment_items.md` (A1–A24). Executed settings are in
`notes/FROZEN_DESIGN.md` (incl. Addendum D1) and in machine-readable form in `EFFECTIVE_PROTOCOL.json`.

| # | Original (preparation) | Executed |
|---|---|---|
| A1 | Shell called `fit-attackers` with the default linear/gbt/mlp slate. The LRT, adaptive and repeated-release classes were unwired. | Per unit: L, GBT and MLP candidates, and NL-selected, on each applicable surface. LRT-A2 (σ-informed, released data only) and LRT-A4 (clean-population stress test) run on the noise units. Repeated-query (A3) is **staged, not run**: under the frozen persistent-draw contract N = 1. |
| A2 | L grid conflict (5 values locked, 4 in the design text) | Locked 5 values, C ∈ {0.01, 0.1, 1, 10, 100}. |
| A3 | GBT read only lr and leaves (9 configs); MLP ignored some declared keys | GBT: 20 frozen configs. MLP: 18 configs. Every key is either consumed or listed as unsupported in EFFECTIVE_PROTOCOL.json. The effective settings and per-unit selection tables are saved. |
| A4 | No GBT/MLP selection rule ("best of", a risk of assessment selection) | NL = the member with lower **attacker_val** log-loss (ties go to GBT). The fitting phase receives no assessment arrays (tested). |
| A5 | Output recipes were GBT-only or nested | NL-selected on outputs and on rep+outputs. |
| A6 | Global `noise="none"` stamped on every access record | The per-unit `persistent_token` contract with σ from the manifest reaches every access record (tested). |
| A7 | A4 LRT was a point-mass mixture | Gaussian class-conditional LRT from clean attacker_fit reps + σ²I. This matches durable-guarantees' Tier-2 form. |
| A8 | No A2 LRT | Implemented: released class covariance − σ²I. The eigen floor is 1e-6 · trace(*released* class covariance)/d, because trace(cov − σ²I) can be negative at large σ. |
| A9 | One "native" R² that was actually fitted and scored on assessment rows | Four separate quantities, never interchanged: **N0** (historical check reproduced: in-sample on all 15,060 test rows, historical mixed precision + float64); **N1** (within-assessment, descriptive); **G1** (held-out fixed ridge 1e-6, same normalisation; primary P1); **G2** (relative ridge, scale-invariant; secondary). |
| A10 | R02 ρ grid ambiguous | G2: ρ ∈ {0, 1e-4, 1e-2}, selected on attacker_val. |
| A11 | Label-only, U1, U2 not implemented | **LO**: P(s \| y_task) on attacker_fit, Laplace α = 1. **U1**: frozen head, hash-checked; noise arms by forward pass. **U2**: logistic regression only (the MLP probe is dropped). Utility metrics: accuracy, log-loss, AUC, macro-F1 over supported task classes; paired differences against untreated. Normalised lift only when the clean lift point is ≥ 0.03. |
| A12 | No task labels in `labels.npz` | Separate `task_labels_v1.npz` (sha 1e17ec9c…) with income, occupation_group and education_level from PCRL b96c412, joined by row_id. `labels.npz` is untouched (sha 367972fb…). 26 new v2 manifests. Occupation class 5 (3 / 1 / 1 rows) is excluded from per-class utility with a reason. |
| A13 | Log-loss reduction defined two ways | Both are reported, named `LLR_nats` = LL₀ − LL and `LL_skill` = 1 − LL/LL₀. Brier skill is also reported. |
| A14 | Holm with bootstrap-fraction p-values over ≤ 3 endpoints | 16 frozen endpoints: P1 (G1 vs τ = 0.05) and P2 (rep NL-selected macro AUC vs 0.55), each × 8 untreated pairs. **Bonferroni simultaneous one-sided percentile bounds** at α = 0.05/16, B = 20,000, seed 20261003, numpy `linear` quantiles; tail count 62.5 replicates. These are asymptotic bounds, not exact p-values. NE endpoints stay in the family. |
| A15 | Exploratory intervals | Unchanged: 90 % two-sided, B = 2,000, seed 20261002. Noise arms are averaged over the three release seeds within each replicate; seed spread is reported separately. |
| A16 | Support checked on assessment only (code), or 100 in every role (admission) | Frozen per unit in `supported.json` before any fit: 100 / 30 / 100. NE is kept distinct from UNRESOLVED. |
| A17 | 8-step decomposition including the adaptive step | F0 N0 → F1 G1 → F2 the same G1 predictor scored as AUC → F3 L AUC → F4 NL-selected rep AUC → F5 outputs NL → F6 rep+outputs NL. The adaptive step is replaced by the separately labelled LRT rows on noise units. |
| A18 | Refit seeds (3), 200-permutation null, per-cell real controls | **Dropped for real units**: single refit seed (0), so intervals are conditional on the fitted predictors and refit variance is not estimated. Synthetic positive and null controls run through the same shell/CLI path instead. |
| A19 | τ grid | G1 decisions are reported at τ ∈ {0.01, 0.02, 0.05, 0.10} from the exploratory intervals. Only 0.05 is primary. |
| A20 | Lock v1 pinned the protocol, config, access table and manifests | Lock v2 (`PILOT_LOCK.json` in this directory) pins: the effective protocol; every consumed `stored_model_eval/*.py` and script; the access table; the exact 26 unit IDs with manifest sha256; labels, task labels, features, forward cache and checkpoint sha256; the primary family; and B / seeds / α. Execution refuses on any mismatch (tested). |
| A21 | Runner hard-coded the preparation worktree | The runner derives its worktree from its own location and asserts the branch `research/combined-stored-model-pilot-v1`. It selects units by explicit ID list (26 of the 152 available), and the lock check runs inside the CLI. |
| A22 | Output locations | `~/PCRL_eval_cache_private/pilot_adult_s0/run_v1/units/<unit_id>/` (private; refuses inside git). |
| A23 | Config text said the role split was stratified | Superseded: the executed split is the unstratified record-key hash `pilot-roles-v1` (unchanged rows). |
| A24 | Noise-unit N0 | "unverified": historical approvals exist only on durable-guarantees' rows. |
| A25 | income_prediction/race | Historical N0 = 0.0575 > τ is a known C1. Its P1 stays in the family, but it is **not** read as C2. |
| A26 | Budget | The executable slate was recalibrated synthetically at the pilot sizes: about 0.26 CPU-h for all 26 units (fits ≈ 642 s, inference ≈ 283 s), against the unchanged 2 CPU-h allowance. **No staged schedule was needed.** No recipe was cut for budget. |

## Stale statements corrected in this branch's copy only

The preparation branch is untouched.

| File | Correction | Original sha256 (at 031860f) |
|---|---|---|
| `results/combined_evaluation_preparation_v1/PILOT_PROTOCOL.md` | "Draft v1" is marked superseded by EXECUTED_PROTOCOL.md. CELL-B "Eligible by rule" is replaced by "Not eligible (fails rule 4: no projection arm); not part of this run". | 095aca43bc51108d17a5d8622a3404046ef0de57f707221b4d4d73335c2c61b5 |
| `results/combined_evaluation_preparation_v1/ADVISOR_BRIEF.md` | CELL-B "eligible" is corrected to "fails rule 4; not part of this run". | 97c99058ba21001a7a862c812d54e2733ee4985630ff99884049d35e333ee395 |

The preparation `PILOT_LOCK.json` is not edited. Its pins refer to the original bytes at 031860f.

## Executable coverage

`EXECUTION_COVERAGE.csv` holds the planned endpoint → recipe → surface → implementation → stored prediction →
inference → report rows (189 planned rows from role 1), with statuses updated after the run.

The pre-fit status of every row is one of:

| Status | Meaning |
|---|---|
| planned | will run |
| reused | outputs surface of the noise units |
| staged | repeated-query, A3 |
| NA | e.g. N0 on noise units |
| unavailable | projection arms: Q never saved; refit not authorized |
