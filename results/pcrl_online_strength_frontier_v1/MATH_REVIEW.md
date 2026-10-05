INTERIM STATUS (train/data): 0 REQUIRED
INTERIM STATUS (select/family/infer): 2 REQUIRED (S1 osf/select.py, S2 osf/infer.py) - repair before SELECTION_AND_AUDIT_LOCK; neither touches training, so neither uses the two-amendment training budget. Patches below; both verified on copies (all fixtures pass with them).

# Mathematics and design review: online strength frontier

**Role:** design / math reviewer, 2026-10-04. Owns this file and `osf/tests/test_math_review.py`. I edited no
lead-owned file and no other agent's file. Mutations were injected only into copies in my private scratch space.

**Scope of this interim (train/data).**
- Prompt sections 3, 7, 8, 9, 10, 12, 16 against the ACTUAL code: `osf/train.py`, `osf/data.py`, and the parts of
  `osf/run.py` that drive them (admit, parity, fidelity, replay, timing, bank).
- Pinned sources read: `rgj/train.py` (J-O / L-O, coefficients, fixed head, Transform, recovery, parameter ordering,
  minibatch RNG), `smf/train.py` (`normalized_direction`, allocation, TASK path), `jcv/train.py` (Model, task_losses,
  flat_grad, assign_add), `smf/data.py`, `rgj/data.py`, `rgj/finalize.py`, `smf/run.py` (how admitted U / RAW units
  were produced), and the predecessor `MATH_REVIEW.md` (R1-R4 regressions checked explicitly).

**Checks run.**
- `osf/tests/test_math_review.py`: **37 tests, all pass, ~5 s** (synthetic CPU fixtures plus two light real-data
  loader checks that read no assessment label and fit nothing).
  `cd <WORKTREE> && OMP_NUM_THREADS=1 PYTHONPATH=. <venv>/python -m pytest osf/tests/test_math_review.py -q`
- Independence: own synthetic data (SEX prior ~0.67, model seed 3), own functional forward (encoders, fixed-head views,
  critics, transforms, recovery surrogate with the constant, task losses) and own update algebra. The engine is used only
  to produce trajectories/snapshots; expected values are recomputed from first principles.
- Mutation receipts: **28 injected defects (22 in a copy of `osf/train.py`, 6 in a copy of `osf/data.py`), 28 caught.**
- One light real-data check (counts/hashes only, no labels of the assessment, no fit): see "Data roles" below.

**Verdict (train/data): 0 REQUIRED, 5 RECOMMENDED (A1-A4, A6) plus one pinning note (A5), 8 NOTES.** The engine implements the registered RAW, NORM and TASK
semantics; nothing found justifies an engineering amendment before TRAINING_PROTOCOL_LOCK. The RECOMMENDED items are
hardening/reporting improvements that change no trained number; the lead may fold any of them in before the lock.

## 1. Gradient semantics: what was verified, and how

| Item (prompt section) | Result | Fixture(s) |
|---|---|---|
| One step equals the registered formula: theta_s - theta_{s-1} = -lr kappa [t_enc + q, t_head], q from beta p_i (RAW) or stop_grad(rho s_i ‖t_i‖/‖p_i‖) p_i (NORM), declared salt-0 minibatch, fixed warm head, aligned critics/transform; first, middle and last step; RAW-J/L, NORM-J/L, a = 0.5/1/2 | PASS (rel. L2 error of the update <= 2e-3, float32 transform noise; defects move it by O(1)) | `test_one_step_reconstruction_from_first_principles` (6 configs x 3 steps) |
| Receipts equal independently recomputed ‖t_i‖, ‖p_i‖, ‖q_i‖, ratio, cos, R per view, kappa, pre-clip total (§12) | PASS | same |
| Normaliser denominator uses encoder blocks only, never task-head gradients (§10) | PASS | same; mutation M01 caught by 9 fixtures |
| Positive scalar: magnitude only, direction never changed (also when capped) (§8, §10) | PASS | `test_cap_changes_magnitude_never_direction`, `test_zero_rule_and_threshold` |
| Float64 norms (float32 overflow/underflow cases) | PASS | `test_float64_norms_in_the_scalar` |
| Cap 100 wired; capped steps report the actual (lower) ratio and a cap flag; realized RMS falls short of rho, never inflated | PASS | `test_cap_changes_magnitude_never_direction` |
| Zero threshold 1e-12 wired to the training call; ‖p‖ <= tol, ‖t‖ = 0 -> q = 0 with reason | PASS | `test_zero_rule_and_threshold`, `test_zero_threshold_is_wired_to_HP` |
| Zero directions contribute q = 0 and a logged reason; the other encoder is NOT scaled up to meet the budget; constant winning on every view -> reason 1 on both encoders and an update bitwise equal to TASK, critics still training (predecessor R1) | PASS | `test_realized_vs_conditional_summary`, `test_all_views_losing_gives_task_update_and_logged_zero` |
| Global clip 5 over [t_enc + q, t_head], applied after q; strength ratios are pre-clip and clip-invariant; pre_total² = Σ pre_enc² + pre_head² | PASS | `test_clip_is_global_and_applied_after_q` |
| Heads receive the plain task gradient in every mode | PASS | one-step reconstruction (head blocks), `test_identical_task_head_and_input_schema_for_every_mode` |

## 2. Fixed allocation and strength statistics (§8, §12)

- s_income = a/√((a²+1)/2), s_occupation = 1/√((a²+1)/2) applied to encoder 0 = income (KS = [2, 6]; TData Y[0] =
  income). Swapping recipients (mutation M02) is caught.
- On an identical step-1 state, changing a leaves ‖t_i‖ and ‖p_i‖ unchanged, gives ‖q_i‖ = rho s_i(a) ‖t_i‖, an
  RMS of the two individual ratios exactly rho, and ratio_1/ratio_2 = a; the proxy coefficients do not depend on a
  (`test_allocation_changes_split_not_budget`). Changing a does not raise the declared RMS budget.
- Combined ratio ‖q_concat‖/‖t_concat‖ = √((r₁²t₁² + r₂²t₂²)/(t₁² + t₂²)) is the task-norm-weighted version and differs
  from rho when a ≠ 1 and t₁ ≠ t₂ (asserted). Both are archived per step (`comb_ratio`, `rms_ratio`).
- RAW receipts' combined ratio equals pinned rgj's logged penalty/task norm ratio at every logged step (rel 1e-5);
  RAW individual ratios are not constant and their RMS differs from the combined ratio, so a combined RAW ratio of
  2.6-3.2 must not be read as two constant individual ratios (§3) (`test_raw_combined_ratio_matches_rgj_logged_norms`).
- Realized vs conditional (§12): `realized_ratio` counts zero steps as 0; `conditional_ratio_nonzero_steps` excludes
  them and is labelled conditional; zero fractions and reasons are reported (`test_realized_vs_conditional_summary`;
  mutations M07, M21 caught). STRENGTH_PROFILES.csv must take its "realized" columns from `realized_ratio` only.

## 3. Raw fidelity, parity and RNG (§7, §10)

- RAW is bitwise equal to pinned rgj J-O/L-O at other betas and seed than the lead's test (beta 0.6 joint, 0.1 local,
  seed 3), including critics, clip counts and coefficients beta/3 (joint) and beta/2 (local)
  (`test_raw_bitwise_rgj_other_betas_seed`; mutations M03, M15, M19 caught).
- Admission premise for reused smf U: smf.train's task line (stage A = salt 0) is bitwise equal to osf TASK
  (`test_smf_task_line_equals_osf_task`). smf RAW controls were run as rgj J-O/L-O stage B (salt 0) for 40 epochs
  from the fresh NEW_DEFENSE_FIT warm start (`smf/run.py` stage_raw / stage_warm), so all admitted units sit on the
  common salt-0 convention; `osf/run.py` stage_replay additionally requires bitwise reproduction of each admitted e20/e40
  checkpoint and critics.
- rho = 0 / beta = 0 equal TASK bitwise with critics training AND captures running (`test_zero_strength_parity_with_captures_running`).
- Receipts, snapshots and the log callback consume no randomness, also with global numpy/torch RNG reseeded
  (`test_receipts_and_snapshots_consume_no_randomness`; mutation M09 caught).
- Task minibatch order is identical across TASK, RAW-L and NORM-J and equals my reconstruction of rng([seed, 0, ep]);
  salt 1 differs (`test_common_minibatch_order_across_modes`; M16 caught).
- Penalty views read the FIXED warm head, never the moving training head (M17 caught by reconstruction at later steps).

## 4. Local isolation, snapshots, absent paths (§7, §10, §12)

- A local arm's encoder trajectory is bitwise invariant to the pair critics' weights (re-initialised x3 + 0.1), while a
  joint arm's is not; the shadow pair bank still trains, with the same critic-update count
  (`test_local_encoder_update_invariant_to_pair_critics`; M04, M13 caught). In a joint arm with dead v1 critics,
  encoder 1 still receives protection through R_pair (`test_joint_pair_term_reaches_encoder_with_dead_local_view`).
- Snapshot alignment: a capture at step s holds theta_{s-1} with the critics after their step-s updates; final
  `theta_T_minus_1` holds theta_{T-1} with the critics aligned to it and differs from `theta_T`; `theta_T` equals the
  epoch-40 checkpoint; Adam states are saved (predecessor R4) (`test_snapshot_alignment_theta_T_minus_1_vs_theta_T`; M14 caught).
- No refit, transport, controller, probe, dual or matched-count code path exists in `osf/train.py`, and `train_run`
  exposes no such argument (`test_no_controller_or_refit_paths`).
- Input schema and parameter set are identical for every mode (83 inputs, two encoders, two affine heads). Deployed
  heads are fitted by the same `rgj.finalize.finalize_model` (fit on OSF_DEFENSE_FIT, C on HEAD_VALIDATION) for every
  bank unit and every admitted unit (`osf/run.py` save_release_unit), with admitted releases checked bitwise on smf rows.

## 5. Frozen-minibatch equivalence (§10)

- Independent algebra on a frozen RAW-J snapshot: with rho_i = r_i = ‖beta p_i‖/‖t_i‖ the norm expression reconstructs
  beta p_i with relative L2 error <= 1e-6; a common rho fails when r₁ ≠ r₂ (rel. error > 1e-3); a cap below beta fails;
  t_1 = 0 (zeroed training head, p_1 ≠ 0) fails (`test_frozen_equivalence_independent_algebra`,
  `test_frozen_equivalence_fails_for_zero_task_gradient`, `test_frozen_equivalence_tolerance_rejects_one_percent_scalar_error`).
- The lead's `frozen_equivalence` reaches the same verdicts in all these cases (M22, a silently per-encoder "common" rho,
  is caught).

## 6. Data roles (§6, §10)

Light real-data check (counts and invariants only; no assessment label read, no fit):
- OSF_DEFENSE_FIT 15,434 rows / 15,428 groups; HEAD_VALIDATION 1,500; AUDIT_FIT 6,065; INNER_SELECTION 2,235;
  CRITIC_FIT 10,764 / CRITIC_VAL 2,343 / DIAGNOSTIC_CALIB 2,327.
- OSF_DEVELOPMENT_ASSESSMENT 13,936 rows / 13,929 groups = ORIG 5,243 + RGJ_DEV 3,397 + SMF_DEV 3,796 + CERT 1,500;
  0 rows lost to group overlap; pool groups sum to the total (no group spans two pools).
- The five roles are group-disjoint; each fitting role and subrole equals smf's row-for-row and value-for-value
  (`test_real_roles_match_smf_fitting_roles_and_union`); the assessment equals my independent re-derivation of
  (four pools) minus (groups touching a fitting role or an exclusion).
- The 5 numeric columns have mean 0 / population sd 1 on OSF_DEFENSE_FIT only (not on fit + assessment); 78 one-hot
  columns binary; 83 columns; sex, race and both task labels are -1 on every assessment row; `labels_for` refuses the
  assessment for training/heads/inner_audit/selection (`test_real_preprocessing_fitted_on_defense_fit_only_and_labels_sealed`).
- Synthetic: exclusion of a pool group sharing a group with an exposure/duplicate exclusion; an ineligible CERT pool leaves
  entirely; a pool group overlapping a fitting role is removed AND the loader refuses (loud, not silent)
  (`test_assign_roles_union_and_exclusion_synthetic`, `test_group_overlapping_a_fitting_role_fails_loudly`).

## RECOMMENDED (train/data; none changes a trained number)

**A1. `frozen_equivalence` tolerance mixes norms.** It accepts when max|q_norm - q_raw| <= 1e-4 ‖q_raw‖₂ (max-abs error
against an L2 norm). On my synthetic snapshot ‖q‖₂/max|q| ≈ 3.7-4.0, so a scalar misstatement up to ≈3.7e-4 relative is
accepted (measured: 3e-4 accepted, 5e-4 rejected), while the honest reconstruction error is <= 1e-6. The registered
failure cases (common rho, cap, t = 0) are still rejected, so this is not a defect. Proposed patch (osf/train.py,
`frozen_equivalence`): replace
`err = float((qi.double() - q_raw[lo:hi].double()).abs().max())` with
`err = float((qi.double() - q_raw[lo:hi].double()).norm())` (relative L2), and keep `EQUIV_RTOL = 1e-4` (or 1e-5).

**A2. `osf/run.py` stage_fidelity records but does not require the expected failures.** `ok` ignores
`expected_failure_common_rho` and `expected_failure_cap`. Proposed: `ok = ... and not fail_cap["equivalent"] and
(not fail_common["equivalent"] or abs(rr[0] - rr[1]) <= 10 * T.EQUIV_RTOL * max(rr))`.

**A3. `capture_steps_for` returns {fraction: step}.** `train_run` tests `step in capture_steps`; passing the dict
itself compares steps with fractions and captures only step 1 (1 == 1.0). `osf/run.py` passes `.values()`, so no run
is affected (`test_capture_steps_for_must_be_passed_as_step_values`). Proposed: in `train_run`,
`capture_steps = set(capture_steps.values()) if isinstance(capture_steps, dict) else set(capture_steps)`.

**A4. RAW receipts at t_i = 0 with q_i ≠ 0 understate strength.** In RAW mode the update applies q_i, but with ‖t_i‖ = 0
the receipt leaves `zero` = 0, `ratio` = NaN and `realized_ratio` = 0, so the step silently counts as zero strength.
It needs an exactly zero task gradient on an encoder (not expected with nonzero warm heads), hence not REQUIRED.
Proposed: in the receipts loop, `if mode == "RAW" and tn == 0 and qn > 0: rec["ratio"][j, i] =
rec["realized_ratio"][j, i] = np.inf` and count it in a `diag["undefined_ratio_steps"]` counter.

**A5. `CERT_ELIGIBLE = True` is a constant (evidence now exists; downgraded to a pinning note).** §6 makes CERT
conditional on custody proving it never entered any eligible current model's fitting/selection. The custody owner's
ROLE_MANIFEST.json (written after my first read) records `cert_eligibility.verdict = ESTABLISHED`, zero fitting rows from
any assessment pool for warm/U/RAW/LEACE/FARE, and `loader_flag_agrees_with_custody = true`. Remaining advice: pin
ROLE_MANIFEST.json's hash in DATA_AND_ENGINEERING_LOCK next to `osf/data.py` (lock.py's DOCS list already names it), so
the constant is locked together with its evidence.

**A6. Unsealing is guarded by the caller, not the loader.** `osf.data.load(unseal=True)` itself verifies nothing;
`osf.assess.load_unsealed` gates it. Proposed: `load(unseal=True)` should require a verified-lock token (e.g. the dict
returned by the assess gate, with the assessment row_id hash) and refuse otherwise.

## NOTES (train/data)

- N1. `labels_for` is documentary for training: TData reads `D["sex"]`/`D["y"]` directly, and subrole names
  (CRITIC_FIT, ...) are refused by `labels_for`. The effective guard is the -1 masking at load (verified).
- N2. `diag["nonfinite"]` counts nonfinite privacy gradients that were zeroed (step applied) as well as skipped steps;
  run.py's half-lr retry triggers on either. This is inherited from rgj/smf and applies identically to every arm.
- N3. Epoch checkpoints store theta_T with the critics/transform last used at step T (aligned to theta_{T-1});
  document this in METHOD_CARD next to the theta_{T-1}/theta_T diagnostic.
- N4. Realized summaries include non-applied steps (nonfinite total norm); such steps carry nonfinite norms, so they
  surface as NaN rather than silently.
- N5. `fp64` minibatch fingerprints are 63-bit truncations of SHA-256: identifiers, not proofs.
- N6. In RAW mode `p_norm` is derived as ‖beta p‖/beta in float32 (matches my recomputation to 1e-4).
- N7. `check_partition` is stricter than the exclusion rule: a pool group overlapping a fitting role is excluded but
  then the loader refuses. On the real data no such overlap exists (0 excluded rows), so this only matters as a loud
  failure mode.
- N8. Admitted smf RAW/U runs lack per-step receipts; `osf/run.py` stage_replay regenerates them by a bitwise replay,
  which is the right design (the admitted checkpoint stays the released model).

## Injected defects (train/data)

Each mutation is applied to a copy of the lead file in private scratch space, the full fixture file is run against the
copy (`OSF_REVIEW_TRAIN` / `OSF_REVIEW_DATA`), and the copy is deleted. **28 / 28 caught.**

| Mutation | Caught by (examples) |
|---|---|
| M01 task-head gradient in the NORM denominator | one-step reconstruction (NORM), allocation, clip |
| M02 allocation swapped between recipients | allocation, one-step reconstruction (a ≠ 1) |
| M03 RAW joint coefficient beta/2 instead of beta/3 | RAW one-step, rgj bitwise, rgj logged ratio, frozen algebra |
| M04 local NORM proxy includes the pair term | pair invariance, one-step (NORM-L), realized summary |
| M05 NORM protection sign flipped | one-step reconstruction, cap, clip |
| M06 clip norm ignores the task heads | one-step reconstruction, clip, cap |
| M07 realized summary silently conditional | realized vs conditional |
| M08 no-gradient steps not logged as zero events | all views losing |
| M09 snapshots consume the critic RNG | receipts consume no randomness |
| M10 float32 norms in the scalar | float64 norms |
| M11 cap 100 not applied | cap |
| M12 zero threshold not wired | zero threshold wiring |
| M13 local shadow pair bank not trained | pair invariance, rgj bitwise (L-O), parity counts |
| M14 capture keyed one step early | snapshot alignment, one-step reconstruction |
| M15 critic minibatch stream differs from rgj | rgj bitwise, rgj logged ratio |
| M16 task minibatch salt off by one | common minibatch order, one-step reconstruction |
| M17 penalty views read the moving training head | one-step reconstruction (later steps), cap |
| M18 combined ratio = mean of individual ratios | allocation, rgj logged ratio |
| M19 recovery surrogate without the constant | all views losing, joint pair term, rgj bitwise |
| M20 head update scaled | one-step reconstruction (heads), clip, cap |
| M21 RMS ratio from conditional ratios | realized vs conditional |
| M22 frozen-equivalence common rho silently per-encoder | frozen algebra |
| D01 numerics standardised on all kept rows | real preprocessing, real role equality |
| D02 race label not sealed | real preprocessing / sealing |
| D03 RGJ_DEV pool dropped from the union | synthetic union, real union |
| D04 group-overlap exclusion not applied | synthetic union, loud failure |
| D05 selection may read sealed assessment labels | real sealing / allowlist |
| D06 ineligible certification pool still admitted | synthetic union |

## 7. Selection, endpoint family and inference (§13, §14) - `osf/select.py`, `osf/family.py`, `osf/infer.py`

Reviewed at commit 9ecdb69 (files as committed by the lead). Fixtures run `osf.select.select_all` end to end on a
synthetic inner world (every path redirected to a temporary folder; a guard asserts the real results folder is
byte-identical afterwards) and `osf.infer.main` end to end on synthetic outer predictions with duplicated exact-record
groups. Nothing real is read except the light loader (for the fitting prior hash and constants).

| Item | Result | Fixture(s) |
|---|---|---|
| One global configuration per family across seeds; task gates G1 Acc >= Acc(U) - 0.01, G2 gain >= 0.8 U-gain, G3 gain >= 0.03 required on EVERY seed (a config failing one gate on one seed is infeasible) | PASS | `test_selection_rules_section13` |
| L*: lowest mean inner coalition AUC among feasible RAW-L/NORM-L; tie lower measured compute, then id | PASS | same |
| C*: all RAW-J/RAW-L, NORM-L, U, E, F, F0; may be RAW-J incl. beta 0.6; a reference's own infeasible status is binding; gate-failing LEACE excluded | PASS | same, `test_selection_C_star_may_be_raw_joint_beta_06` |
| N*: guards L* + 0.005 and C* + 0.005 on both recipients, every seed (a 1e-4 excess on one seed excludes); tie lower rho, then a closest to 1, then id | PASS | `test_selection_rules_section13` |
| R*: guard L* + 0.005; tie lower beta | PASS | same |
| Descriptive fallback: summed positive shortfalls (accuracy units + AUC units) first, then mean coalition AUC; cannot pass a claim and is not the deployable best | PASS | `test_selection_descriptive_fallback_cannot_pass` |
| U without nontrivial utility -> every status INVALID; truthful baseline packaged | PASS (selection) | `test_selection_U_invalid_invalidates_everything` |
| Missing comparator invalidates affected claims | **FAIL -> S1** | `test_REQUIRED_S1_missing_L_star_invalidates_dependent_nominees` |
| Study label INCOMPLETE_OR_INVALID when validity is missing | **FAIL -> S2** | `test_REQUIRED_S2_label_is_invalid_when_U_lacks_utility` |
| 27 slots = 3 x 9 clauses with the registered stats, targets and sides; comparator-minus-nominee coalition; nominee-minus-comparator local; Claim B utility clauses retained as aliases of Claim A; the family never shrinks | PASS | `test_primary_family_27_slots_and_z` |
| z = Phi^-1(1 - 0.05/54) recomputed independently (bisection on erfc, no scipy): 3.1130169; secondary z from its 61-entry ledger | PASS | same |
| Strict thresholds: a bound equal to the target cannot pass | PASS | `test_strict_thresholds_tie_cannot_pass` |
| Gain retention Acc - 0.8 Acc(U) - 0.2 Acc(const) = (Acc - const) - 0.8 (Acc(U) - const) (> 0 iff more than 80% of U's gain is kept) | PASS | `test_gain_retention_statistic_algebra` |
| Inherited weighted bootstrap = explicit multinomial resampling of exact-record GROUPS (rows of a group move together), one draw shared by all statistics | PASS (exact to 1e-12) | `test_group_bootstrap_equals_explicit_group_resampling` |
| `osf.infer` primary points and SEs (P19 coalition, P20 local, P22 accuracy vs U, P24 retention, P26 gain) equal an independent recomputation: mean over attacker refits per seed, per-seed paired differences, mean over seeds, group bootstrap B replicates with seed 20261006, the same draw for every arm and seed, interval point +- z SE | PASS (rel 1e-9) | `test_infer_primary_endpoints_match_independent_group_bootstrap` |

### REQUIRED S1 - a missing guard comparator does not invalidate N* / R* (osf/select.py)

**Defect.** §13: N* must satisfy local AUC <= L* + 0.005 AND <= C* + 0.005 on every seed; R* must satisfy <= L* + 0.005;
"missing comparators make affected claims invalid". When no RAW-L/NORM-L configuration is task-feasible (L* =
NO_FEASIBLE_NOMINEE), `select_all` evaluates N* against the C* guard only and R* against no guard at all, and reports
both as `NOMINEE` (the gap is only listed in `missing_guards`, which neither `family.claim_decision` nor the
deployable-best rule reads). Consequences: Claim B (N* vs C*) can be recorded as PASS for an N* whose nomination rule
could not be evaluated, and an unguarded N* or R* can become the packaged "deployable best". (The overall NORM label
stays protected because Claim A also needs L*; Claim C needs L* as well.)

**Failing reproduction** (`test_REQUIRED_S1_missing_L_star_invalidates_dependent_nominees`): every local configuration
fails G1 (income 0.80 vs U 0.85); NORM-J|r3|a0.5 has pair 0.70 and locals within the C* guard; RAW-J|b0.3 pair 0.75.
Result: L* = NO_FEASIBLE_NOMINEE, N* = NOMINEE NORM-J|r3|a0.5 (`missing_guards = ["L*"]`), R* = NOMINEE RAW-J|b0.3
(no guard), `claim_decision("B", all nine PASS)` = PASS, `deployable_best` = NORM-J|r3|a0.5.

**Proposed patch** (osf/select.py, `select_all`, directly after the two `missing_guards` lines):
```python
        for P in (N, Rs):                    # review S1: a missing guard comparator invalidates the nomination
            if P["missing_guards"] and P["status"] == "NOMINEE":
                P["status"], P["descriptive_config"], P["config"] = "INVALID_MISSING_COMPARATOR", P["config"], None
```
The would-be pick stays visible (and scored descriptively via `eval_lock.resolve`), `claim_decision` then refuses
Claim B, and the deployable best is chosen among valid nominees only. Verified on a copy: all fixtures pass.

### REQUIRED S2 - the study label never becomes INCOMPLETE_OR_INVALID (osf/infer.py)

**Defect.** §13 ("If U fails nontrivial utility ... record the invalid status") and §18 ("INCOMPLETE_OR_INVALID if
required scientific validity/coverage is missing; separate that from a complete negative"). `osf.infer.main` calls
`FAM.overall_label(dec)` with the default `complete=True` and never reads the lock's `U_valid` or INVALID statuses,
so an invalid study is labelled EXPERIMENTAL_NO_ADVANTAGE, i.e. as a complete negative.

**Failing reproduction** (`test_REQUIRED_S2_label_is_invalid_when_U_lacks_utility`): a lock with `U_valid = False`
and every status INVALID (exactly what `select_all` writes in that case); `infer.main` returns label
EXPERIMENTAL_NO_ADVANTAGE.

**Proposed patch** (osf/infer.py, `main`, replacing `out["label"] = FAM.overall_label(dec)`):
```python
    complete = bool(EL.get("U_valid", False)) and not any(str(s.get("status", "")).startswith("INVALID")
                                                         for s in EL["statuses"].values())
    out["label"] = FAM.overall_label(dec, complete=complete)          # review S2
```
(With S1, `INVALID_MISSING_COMPARATOR` also yields INCOMPLETE_OR_INVALID: with L* absent no claim can be validly
evaluated, which is missing coverage, not a complete negative. A NO_FEASIBLE_NOMINEE N* remains a complete negative.)
The lead's synthetic lock in `osf/tests/test_late.py` has no `U_valid` key; add `"U_valid": True` there if its label is
ever asserted. Verified on a copy: all fixtures pass.

### RECOMMENDED / NOTES (selection and inference)

- **A7 (RECOMMENDED).** Per-clause decisions for a descriptive fallback (or an INVALID nominee) are written to
  PRIMARY_ENDPOINTS.csv as PASS / NOT_ESTABLISHED like any other clause. The claim is protected by `claim_decision`, but
  a reader counting clause PASSes could misread a fallback. Proposed: in `infer.main`, set the clause decision to
  `DESCRIPTIVE_ONLY` (keeping point/SE/bounds) whenever the clause's nominee or comparator status is not NOMINEE.
- **N9.** Gate ties are float-fragile in principle (e.g. G2 at an exact rational tie: 117 of 280 exact ties on
  n = 2,235 give a float margin of about -7e-17 and would fail an "at least" gate). It cannot bind here: G1 and G3 ties
  are impossible on 2,235 rows (0.01 x 2235 and 0.03 x 2235 are not integers), and G2 can only bind before G1 when U's gain
  over the constant is below 0.05 (inner-selection constants are 0.762 income / 0.283 occupation; U's historical
  accuracies 0.852 / 0.480 give gains of about 0.09 / 0.20). No change needed; integer-count gates would remove the
  question.
- **N10.** `infer` drops nonfinite bootstrap replicates and records `n_finite_replicates`; for SEX AUC and accuracy on
  about 13.9k rows none are expected. A primary row with fewer than B finite replicates should be flagged in VALIDATION.md.
- **N11.** C* ties fall back to the config id (the prompt names no C* tie rule); L*'s compute tie is in practice always
  equal across configurations (identical epochs and critic steps), so it reduces to the id.
- **N12.** The deployable best is the lowest mean inner coalition AUC among valid N*, R*, C*. This is a reasonable
  reading of §18 ("inner-selected best eligible development model"); record the rule in PROTOCOL.md so it is locked.
- **N13.** The bootstrap resamples exact-record groups; a group's rows enter AUC/accuracy as rows with the group's weight
  (inherited). This matches "a group with replicated rows does not create extra people" at the resampling level.

## 8. Design notes for interpreting the strength frontier (§3, §12)

- **No common rest point.** With uncapped nonzero directions, ‖t_i + q_i‖ >= |rho s_i - 1| ‖t_i‖ (triangle
  inequality, since ‖q_i‖ = rho s_i ‖t_i‖). So for rho s_i ≠ 1 the full-batch NORM field has no stationary point
  except where t_i = 0 (where protection switches off by the zero rule). A fixed-beta RAW run instead has stationary
  points of L_task + beta P, where t_i = -beta p_i and the per-encoder ratio is exactly 1. NORM at rho = r is therefore
  not "RAW with a matched coefficient"; it is a different dynamical system that keeps moving at speed >= lr kappa
  |rho s_i - 1| ‖t_i‖. This is one reason why matched average strength cannot equalize the temporal profile,
  directions or trajectory (§12), and why the frontier comparison is not a causal isolation of magnitude.
- **Combined vs individual ratios.** RAW-J's logged combined ratio is the task-norm-weighted quadratic mean
  √((r₁²t₁² + r₂²t₂²)/(t₁² + t₂²)); it lies between r₁ and r₂ and is pulled toward the encoder with the larger task
  gradient. A combined 2.6-3.2 is compatible with very unequal individual ratios; STRENGTH_PROFILES.csv must report the
  replayed per-encoder realized ratios (zeros counted) before NORM rho 3 is described as "matching" the incumbent.
- **Clip interaction.** Global clipping scales t and q together, so ratios are clip-invariant, but at high rho the clip
  binds more often and shortens the task step itself; clip fractions belong next to every strength comparison.

## 9. Bounded prior-art note (no novelty claim)

- Adversarial representation learning against a sensitive attribute is established; attribute it to Madras, Creager,
  Pitassi and Zemel, "Learning Adversarially Fair and Transferable Representations", ICML 2018, PMLR 80
  (https://proceedings.mlr.press/v80/madras18a.html), as the source studies do. Related lineage, cited from reviewer
  memory and not re-verified in this session: gradient reversal for domain adaptation (Ganin and Lempitsky, ICML 2015),
  censoring representations with an adversary (Edwards and Storkey, ICLR 2016), adversarial debiasing with a projection
  term (Zhang, Lemoine and Mitchell, AIES 2018).
- Rescaling one loss's gradient relative to another's norm, and sweeping fixed coefficient/strength banks, are standard
  multi-objective training devices (e.g. GradNorm, Chen et al., ICML 2018; gradient surgery/PCGrad, Yu et al.,
  NeurIPS 2020; also from memory). A per-recipient fixed RMS allocation of the scalar is a reparameterised
  hyperparameter, not by itself an algorithmic contribution. A winning comparison here would establish behaviour on this
  reused benchmark; novelty would need its own bounded assessment.
- No nonconvex constrained-learning or two-player convergence theorem applies: the encoder update is a clipped,
  stop-gradient-rescaled surrogate direction (not the gradient of any fixed objective, see §8), the surrogate takes a
  minimum over online critics and the constant, and the critics are trained online on a moving target. Do not borrow
  duality-gap, Lagrangian or GDA convergence results for it; R_v is not mutual information and not a secrecy certificate.
