# Gap confirmation against the pinned source (031860f)

Role 1 (protocol-to-code mapping). Written 2026-10-02. I read only source, the frozen design, the preparation
package, manifests and label marginals. I fitted nothing and opened no assessment result.

All file:line references are to `git show 031860fbfd91:<path>`. The worktree copy of `stored_model_eval/` is
already being edited by Role 2 (access, admission, attackers, config, pipeline), so its line numbers differ
from these.

Abbreviations:
- `SH` = `results/combined_evaluation_preparation_v1/notes/evaluator/run_pilot_cell_a.sh`
- `CFG` = `results/combined_evaluation_preparation_v1/notes/methodology/protocol_config.json`
- `FD` = `results/combined_stored_model_pilot_v1/notes/FROZEN_DESIGN.md`

## Verdict summary

| # | Gap | Real at 031860f? | Repair required by FD (or scope amendment) |
|---|---|---|---|
| 1 | Attacker slate not wired; noise contract hard-coded to none | **Yes** | Wire L, NL-selected, A2 LRT and A4 LRT. Stage the repeated-query extension. Carry the manifest contract into the access records. List R11 as deferred. |
| 2 | Held-out R² not carried through save, load, infer and report | **Yes** | Save G1/G2 predictions and priors. Make infer and report consume them. Bootstrap them, primary and exploratory. |
| 3 | No label-only, Brier/log-loss, U1 or U2 rows | **Yes** | Add LO, the LO contrast, LLR and BSS, U1 (with a frozen head for noise units) and U2. |
| 4 | Multiplicity, utility, exposure and decomposition keys not consumed | **Yes** | Add a primary-family module with 16 fixed endpoints and Bonferroni simultaneous bounds (B = 20,000). Consume the rest or mark it unsupported. |
| 5 | The value called "native" is the within-assessment statistic, not the historical check | **Yes** | Rename it N1. Add N0 on all 15,060 rows in float64 and in the historical mixed float32 arithmetic. Noise units: NA (unverified). |
| 6 | R02 is not the native quantity; there is no pure metric contrast | **Yes** | Add G1 (fixed ridge 1e-6, SS_tot around the fit means, unclamped), keep G2 (R02) separate, and add the pure-metric row. |
| 7 | Effective grids differ from the declared ones; no GBT-vs-MLP selection | **Yes** | Freeze EFFECTIVE_PROTOCOL.json, select between GBT and MLP on attacker_val, and decide the L C-grid conflict. |
| 8 | Lock incomplete, worktree path hard-coded, units counted rather than compared | **Yes** | Derive the root, strengthen the lock, compare unit IDs and check every pin in the CLI. |
| 9 | Support checked on assessment only | **Yes** | Freeze supported.json at 100/30/100 before inference, with reason codes and coverage. Keep NE distinct from UNRESOLVED. |

## 1. Attacker slate and noise contract

**Call path:**
- `SH:38` runs `fit-attackers --manifest "$m" $MODE --out-dir` with no `--attackers` argument.
- `cli.py:177` defaults `--attackers` to `linear,gbt,mlp`, and `cli.py:99` splits it.
- `pipeline.py:54` looks each name up in `FAMILIES[a]`.
- `attackers.py:123` defines `FAMILIES` as linear, gbt and mlp only.

**Not referenced anywhere in the pinned package:**
- `NoiseLRTAttacker` (`attackers.py:157-192`)
- `AdaptiveAttacker` (`:195-222`)
- `RepeatedReleaseAttacker` (`:225-248`)
- `ReleaseChannel` (`:131-154`)
- `releases.repeated_releases` (`releases.py:31-42`)

A grep confirms that the only hits are their own definitions.

**Noise contract bug (real):**
- `pipeline.py:58-59` builds `ReleaseContract` from the global `cfg["release_contract"]`, which defaults to
  `{"noise":"none"}` (`config.py:45`) and is never overridden by CFG.
- Every noise unit's `release` block is ignored. That block is `{"kind":"gaussian_noise","sigma_abs":σ,"seed":k,"release_count":"one","outputs_surface":"clean model output"}`.
- `load_admitted` (`admission.py:208-221`) never reads `manifest["release"]`.

**Second defect: the A4 form does not match.** The pinned `NoiseLRTAttacker` is an exact point-mass mixture,
p(r|s) = mean over i in class s of N(r; h_i, σ²I), over clean vectors (`:175-186`). FD and the access table
(R09) define A4 as **class-conditional Gaussians** fitted on clean attacker_fit reps plus σ²I. R10 (the A2
LRT on releases) does not exist in the code.

**API mismatch.** FD writes `ReleaseContract(noise="gaussian", sigma=σ, persistent=True)`. At pin,
`access.py:26-37` accepts only `noise ∈ {none, fresh_per_query, persistent_token}` and has no `persistent`
field. The FD form raises `ValueError`. Either amend the dataclass, or map the manifest to
`ReleaseContract(noise="persistent_token", sigma=sigma_abs)` and record that mapping. The coordinator should
pick one.

**Repair required:**
- Recipes L, CAND-GBT, CAND-MLP and NL-selected on {rep, outputs, rep+outputs} for untreated units; on
  {rep, rep+outputs} for noise units, with outputs reused.
- LRT-A2 (new) and LRT-A4 (new class-Gaussian form, or an amendment if the mixture is kept) on noise rep.
- A contract taken from the manifest for every access record.
- R12 is staged: under the persistent contract N = 1, and a fresh-query contract needs its own amendment.
- R11 (general adaptive) is **not in FD**. It must appear in AMENDMENT_1 and the coverage table as deferred
  or unavailable, not silently omitted.

**Further inputs needed:**
- **A4 needs the clean attacker_fit reps.** Noise manifests list `fwd` as a file but declare only
  `rep` = release as `representations`. Add a declared `clean_representations` array (`fwd:rep_p0`).
- **U1 for noise units needs the checkpoint** (`forward.py:119` `FrozenPCRLv2.head`). Noise manifests do not
  pin it; add it (sha256 1cfc2fef…).

## 2. Held-out R² through save, load, infer and report

**Path:**
- `pipeline.py:46-47` computes `closed_form_linear` (`:66-90`).
- `save_scores` writes it only as scalars inside `fit_records.json` (`:105-107`).
- `load_scores` (`:111-119`) reads only `row_id/y/unit/record_key/P__*`.
- `infer` (`:142-155`) loops over `scores["probs"]` with AUC statistics only.
- `report` (`:194-208`) iterates `AUC_METRICS` (`:24`).
- No predictor, target encoding, prior or sufficient statistic is saved, so R02 cannot get an interval.

**Repair (FD "Saved predictions"):**
- Save `G1_pred`, `G2_pred`, `G1_prior` and `G2_prior` (attacker_fit means), plus `y_s` and `assess_row_id`.
- Write an R² statistic closure `w -> 1 - Σw‖z-ẑ‖² / Σw‖z-z̄_fit‖²`, unclamped, and feed it to
  `cluster_bootstrap`.
- Make report emit R² rows.
- Test the CLI fit → save → load → infer → report round trip.

**Missing from FD's key list:** per-row arrays for held-out ρ₁² (`u = H b`, `v = a[y]`). Without them ρ₁² is
point-only. Add `rho_u`/`rho_v` or declare ρ₁² point-only.

## 3. References and utility

**What exists but is never called:**
- `metrics.label_only_reference_auc` (`:387-397`, Laplace α = 1)
- `prior_from_fit` (`:506`)
- `brier_skill` (`:511`)
- `logloss_reduction` (`:519-525`)

**What does not exist:** U1 or U2 code. Task labels are also absent from `labels.npz`, whose keys are
`row_id, unit, role, record_key, sex, race, age_group, marital_status, income`.

**Repair:**
- **LO and its contrast:** paired over identical assessment IDs.
- **LLR and BSS:** for every recipe.
- **U1:** untreated = stored logits. Noise = `head(h+ε)`.
- **U2:** LR probe.

**Definition conflict:** FD defines log-loss reduction as the ratio `1 − LL/LL₀`. CFG `metrics.logloss_reduction`
and `metrics.py:519-525` define the difference `LL₀ − LL` in nats. Pick one and record it, or report both.

## 4. Unconsumed config sections

`config.py:140-141` marks `{bars, primary_tau, interval, support, roles, attackers}` as consumed. Everything
else goes to `meta.not_consumed_by_evaluator`:

`companion_text, decision_rule, decomposition_factorial, exposure, metrics, multi_purpose, multiplicity, numerical, primary_bar, r2_tau_grid, status, surfaces, utility, version`

I verified this list by running `load_protocol` on CFG: the sha is `6dae304f…`, which matches PILOT_LOCK.

"Consumed" sections also drop most sub-keys silently (see the key table below). No code implements Holm,
bootstrap p-values, a family, or simultaneous bounds. `cluster_bootstrap` (`inference.py:65-103`) gives only
two-sided percentile intervals (`:101`).

**Repair (FD):**
- 16 frozen endpoints `P1-<pair>` and `P2-<pair>`.
- One-sided Bonferroni bounds at α = 0.05/16 = 0.003125 each; percentile bootstrap, B = 20,000, seed
  20261003.
- NE endpoints stay in the family.

**Quantile rule.** The tail count is 62.5, so the quantile interpolation rule must be frozen (for example
`np.quantile(method="lower")` for LCB and `"higher"` for UCB) and recorded beside the resolution of 5e-5.

## 5. The value called "native" is within-assessment

`pipeline.py:74`: `native = r2_onehot_ridge(H[ei], y[ei], …)` fits and scores on assessment rows only, cast
to float64 at `:72`.

**What the historical value is:** `scripts/eval_round4_dominant_axis.py:158-172` (origin/main) computes
`LinearComplianceCertificate(0.05, 1e-6).check(test_reprs, test_labels)` on all **test-split** rows, with
the test split normalised by the train `norm_stats`. The arithmetic (`pcrl/purposes/verification.py:89-103`)
is mixed precision:
- float32 H centring and Gram;
- float64 ridge term, one-hot, solve and prediction;
- clamped at 0.

`metrics.r2_onehot_ridge(dtype=np.float32)` casts *everything* to float32, so it is **not** an exact replica.

**Repair:**
- Rename the pinned value N1.
- Add N0 on all 15,060 rows in float64, plus a faithful mixed-precision replica.
- Compare N0 with `dominant_axis_audit.json` `per_seed["0"].rows[*].r2_onehot` (8 rows).
- Noise units: NA. Card 5 records a dg in-sample 5-draw approval on dg rows, which cannot be reproduced on
  this pool, so mark the check "unverified".

**Wording.** For PCRL v2 the historical native check ran on the **test** split (GUARANTEE_CARDS card 2), not
on the defense-fitting (train) rows. Report N0 as "historical native check as computed (test split,
in-sample)" and never as training-sample compliance.

**Optional.** A train-split N0 is possible fit-free: run a frozen forward pass on the PCRL train split, which
is local (`data/adult/adult.data`). It is not in FD; adding it is the coordinator's call.

## 6. R02 versus the native quantity, and the pure metric contrast

- R02 (`metrics.py:458-480`) has a relative eigen-floor and runs with rho = 0 only. The rho grid
  {0, 1e-4, 1e-2} and its attacker_val selection (CFG R02) are not implemented.
- The held-out branch of `r2_onehot_ridge` (`:131-145`) centres SS_tot on **score-row** means (`:142`) and
  clamps (`:145`). So the G1 required by FD (fixed ridge 1e-6, SS_tot around the attacker_fit means,
  unclamped) does not exist.
- No code scores the least-squares predictor as an AUC.

**Repair:** add G1 and the pure metric row. Keep G2 separate and label any G1−G2 difference as a
penalty/scale effect. **Decide the rho grid** for G2: either implement the grid with val selection or amend
to rho = 0 floor-only.

## 7. Effective grids

Effective configuration at pin (`load_protocol(CFG)` plus `plan.grid_sizes`):

| Recipe | Effective at pin | Dropped from the declaration |
|---|---|---|
| **L** | C ∈ {0.01, 0.1, 1, 10, 100} (5); LR `max_iter=2000` (`attackers.py:90`) | Declared `max_iter` was 5000 |
| **GBT** | lr {0.03, 0.1, 0.3} × leaves {15, 31, 63} = 9 configs; `max_iter=500`; `early_stopping=False` (`:104`) | `min_samples_leaf`, `l2_regularization`, "20 random configs", early stopping, 3 seeds |
| **MLP** | hidden {64², 128², 256²} × alpha {1e-5, 1e-4, 1e-3} = 9; `early_stopping=True`; `max_iter=200` (DEFAULT) | `learning_rate_init`, 3 seeds |

- **R06N** (10 configs) and **R07N** (a nested 38-config slate) are not distinguished.
- **No GBT-versus-MLP selection.** Each family is reported separately (`pipeline.py:52-62`), so a "best of
  R03/R04" taken in report would be chosen on assessment.
- **FD conflict:** FD's L grid {0.01, 0.1, 1, 10} drops C = 100, which CFG, the access table and the
  translated pin all contain.

**Repair:** write EFFECTIVE_PROTOCOL.json with every executed key, choose GBT vs MLP on attacker_val
log-loss, and resolve the C-grid conflict explicitly.

## 8. Lock, path and unit selection

- **Hard-coded path:** `SH:7` sets `WT=/Users/nathansamson/PCRL/.worktrees/combined-evaluation-preparation-v1`,
  the old worktree.
- **What the lock check covers:** `SH:15-25` runs only under `EXECUTE=1`. It checks the CFG sha and the 152
  manifest shas. It does **not** check `protocol_md_sha256`, `access_table_sha256`, `code_commit`, any code
  file, the checkpoint or the data hashes.
- **Admission interpreter:** `SH:33` admits with system `python3`, not `$PY`.
- **Unit selection:** `SH:28-32` selects by glob plus a `case` filter and counts `n` (`:42-44`). It never
  compares unit IDs.
- **The CLI has no lock check.** `fit-attackers --execute-scientific-fits` can be called directly.

**Repair:**
- Derive the root from `git rev-parse --show-toplevel`, or take a validated `--root` that must equal the
  owned worktree on branch `research/combined-stored-model-pilot-v1`.
- New lock covering: effective protocol, consumed code files (sha256), access table, the exact 26 unit IDs,
  `features.npz`, `labels.npz`, the task-label file, the fwd cache, the releases, the checkpoint, and the
  primary family.
- The CLI refuses to run if any of those differ.
- Admission still verifies array hashes.

## 9. Class support across roles

- `per_class_ovr_auc` (`metrics.py:281-298`) and `supported_pairs` (`:344-353`) decide support from
  assessment rows only, at a single `min_support` (CFG `min_class_n` = 100).
- `per_class_ovr_auc` also requires the complement to have ≥ 100 rows.
- `min_class_n_attacker_val` (30), `min_class_n_roles` and `pair_rule` are never translated.
- Admission applies the manifest's 100 to every role (`admission.py:165, 192-200`). Its warnings therefore
  flag race class 1 (59 in val) and age_group class 3 (76 in val) as unsupported, although both pass
  100/30/100.
- `decide` (`inference.py:106-114`) keeps NOT_ESTIMABLE separate from UNRESOLVED. `classify_outcome`
  (`pipeline.py:171-180`) folds both into C5; report the codes separately.

**Admission counts (reused, from `admit` on the existing manifests):**

| Attribute | attacker_fit | attacker_val | assessment | Supported at 100/30/100 |
|---|---|---|---|---|
| sex | 2447/5124 | 722/1517 | 1744/3506 | 2/2 |
| race | 67/221/696/63/6524 | 21/59/210/19/1930 | 61/128/505/40/4516 | {1,2,4}: 3/5 classes, 3/10 pairs |
| age_group | 1411/3914/1999/247 | 405/1151/607/76 | 957/2726/1385/182 | 4/4 |
| marital_status | 3968/3603 | 1171/1068 | 2738/2512 | 2/2 |
| income | 5721/1850 | 1705/534 | 3934/1316 | 2/2 |

**Repair:** freeze `supported.json` per unit from all three roles before any bootstrap, keeping reason codes
and coverage.

## Every scientific key in protocol_config.json

Status at 031860f, and the proposal.

- **C** = consumed and used.
- **T** = translated but unused downstream.
- **I** = silently ignored inside a "consumed" section.
- **U** = listed as unconsumed.

| Key | At pin | Proposal |
|---|---|---|
| version, status, companion_text | U | Metadata. `status` "DRAFT" is stale; correct it in the new copy (the new sha goes in the new lock). |
| bars | C (`config.py:101`; `pipeline.py:153`) | Consumed (exploratory decisions at 0.52/0.55/0.60). |
| primary_bar | U | To be consumed by the primary-family module (P2 bar). |
| r2_tau_grid | U | To be consumed (secondary G1/G2 decisions at 0.01/0.02/0.10), or explicitly unsupported. FD names only τ = 0.05. |
| primary_tau | C → `r2.tau` (`:103`), used only by recount (`cli.py:75`) | To be consumed by P1. |
| interval.method | I | Validate it equals the percentile cluster bootstrap. |
| interval.B, alpha, sidedness | C (`:106-111`; α → 0.10 two-sided) | Consumed for exploratory intervals. The primary family uses a separate B = 20,000 and α = 0.003125 (new keys). |
| interval.unit | T (`bootstrap.unit_by_dataset`, never read) | Amend adult "row" to "record unit (5 duplicates collapse)", as `resolve_units` already does via record_key. |
| interval.attackers_held_fixed_in_bootstrap | I | Consume as an assertion (no refit inside infer). |
| interval.refit_variance (attacker_seeds 3, K 8) | I | Not in FD. Mark explicitly unsupported in the core (amendment), or stage it. K = 8 is NA under the persistent contract. |
| interval.defense_seed_variation | I | NA: one defense seed. Release seeds are handled by the FD seed aggregation. |
| interval.max_statistics | I (but `worst_*_stat`, `inference.py:129-137`, re-max per replicate) | Consume; validate. |
| interval.permutation_null (200, same_max_rule, no max over seeds) | I (`inference.py:145` exists, uncalled) | Not in FD. Decide: consume it as the real-cell null control, or mark it unsupported. |
| interval.bootstrap_seed | C | Consumed (20261002). |
| interval.survey_weights | I | NA for Adult; validate null. |
| support.min_class_n | C (assessment-only) | To be consumed across attacker_fit and assessment. |
| support.min_class_n_roles | I | To be consumed (supported.json). |
| support.min_class_n_attacker_val | I | To be consumed (30). |
| support.min_pair_n | T (`min_pair_support`, never read) | To be consumed. |
| support.pair_rule | I | To be consumed (FD: both classes supported in all roles; equivalent or stricter). |
| support.not_estimable_code, ne_reasons | I | To be consumed in SUPPORT_COVERAGE. |
| support.macro_over | C ("supported") | Consumed. |
| roles.attacker_fit/val/assessment shares | I (used by the prep script) | Validate role counts against the lock. |
| roles.stratify ["s","y"], split_seed 20261002 | I | **Stale.** The §18 decision is a record-key hash ('pilot-roles-v1'), unstratified. Mark superseded in EFFECTIVE_PROTOCOL. |
| roles.pool, group_by, defense_fit | I | Validate (defense_fit role = 0 rows). |
| roles.secondary_overlap_arm | I | Explicitly unsupported: no defense_fit rows in this pool. |
| roles.alignment_assertion | I (admission aligns arrays) | To be consumed: assert identical `assess_row_id` across surfaces, recipes and units. |
| numerical.precision | defaults only | Consumed via the float64 default; validate. |
| numerical.linear_floor (rel 1e-6, also_report_unfloored) | I (hard-coded `metrics.py:458`) | Consume rel. Unfloored variant: to be consumed or unsupported. |
| numerical.historical_native_float32_reported_beside_float64 | U | To be consumed by N0 (mixed-precision replica). |
| metrics[] (9 ids) | U | native → N0/N1; heldout_linear_r2 → G2; brier_skill and logloss_reduction to be consumed (resolve the LLR definition); macro, worst-class and worst-pair consumed; rho1_sq held-out consumed, in-sample unsupported or optional; native_dominant_axis historical only, unsupported. |
| surfaces[] (rep/out/rep+out) | U | Map to rep/outputs/rep+outputs. The declared output object (logit vector per purpose) goes in the lock. |
| attackers R00 | U | → N0/N1. |
| attackers R01 C | C (5 values) | Resolve against the FD 4-value grid. `max_iter` 5000 vs 2000: freeze one. |
| attackers R02 rho grid | I | To be consumed, or amended to rho = 0. |
| attackers R03 lr, leaves, max_iter | C | Other R03 keys I. Freeze in EFFECTIVE_PROTOCOL (20-random / early stopping / min_samples_leaf / l2 consumed or explicitly dropped). |
| attackers R04 hidden, alpha | C | learning_rate_init and 3 seeds I. Freeze likewise. |
| attackers R05 | I | To be consumed as LO (frequency table; FD drops the LR variant). |
| attackers R06L/R06N/R07L/R07N/R08 | I | Consumed via the surface loop. FD replaces R06N (GBT, 10 configs) and the R07N nested slate with NL-selected per surface. Amendment. |
| attackers R09, R10 | I | To be consumed (new A4 class-Gaussian form; new A2). |
| attackers R11 | I | Explicitly unsupported (deferred) per FD. |
| attackers R12 | I | Staged (N = 1 under the persistent contract). |
| attackers R13, R14, R15 | I | Explicitly unsupported (NA). |
| decision_rule (+ category_mapping) | U (`decide` and `classify_outcome` exist; the latter is uncalled) | To be consumed. Report NE and UNRESOLVED separately inside C5. |
| decomposition_factorial F0–F7 | U | To be consumed as the **amended** F0–F6; F7 unsupported. |
| multiplicity.* | U | **Superseded** by the FD primary family (Bonferroni bounds; P3 demoted to secondary; noise arms secondary). |
| utility.rows, kinds U1/U2 | U | To be consumed. U2 MLP dropped by FD (amendment). U3 label if used. U4 unsupported. |
| utility.metrics (+ macro_f1_multiclass) | U | Accuracy, log-loss and AUC consumed. macro-F1: decide. |
| utility.report (paired_difference_bootstrap) | U | To be consumed. |
| utility.normalised_lift_rule, frontier_rule | U | Not in FD. Consume or explicitly unsupported. |
| multi_purpose.* | U | Explicitly unsupported in the pilot, except that `exposure.lock_contents` asks for the permission table in the lock. |
| exposure.data_status, new_data | U | To be consumed as the "development (spent)" label everywhere. |
| exposure.lock_contents | U | To be consumed by the new lock: add the declared output objects and the permission table. |
| exposure.amendments_file ("AMENDMENTS.md") | U | Conflicts with the prompt's `AMENDMENT_1.md`. Name it in the lock. |
