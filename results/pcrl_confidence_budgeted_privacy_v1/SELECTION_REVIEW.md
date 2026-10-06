# Selection review (role C: statistics and selection reviewer)

| Item | Value |
|---|---|
| Study | Confidence-budgeted privacy compression (cbp), branch `research/pcrl-confidence-budgeted-privacy-v1` |
| Scope | Prompt §§9, 11, 12 and 14: ordinary and headroom eligibility, comparators, guards, ordering and ties, fallbacks, diagnostics, the 37-slot family, bootstrap and critical value, the label truth table and classifier, and the deliberate defects within this scope |
| Method | An independent **oracle**: a transcription of the prompt text, written without importing cbp or qpc. It is compared with the lead's executables on synthetic inner-record banks and synthetic saved assessment predictions. Each deliberate defect has a named fixture, and each REQUIRED finding has a failing fixture and a patch |
| Tests | `cbp/review_tests/test_selection_review.py`, owned by role C and never locked: 46 tests |
| Real data | **None read.** No fits, no inner or assessment records, no assessment labels, no private units |
| Resources | Every test run used the shared semaphore (labels `C:*`). Each run took under 15 s of CPU |
| Status at 17:24Z (HEAD 1739cddf) | **Role C verdict: no open REQUIRED finding; no objection to FIT_LOCK / AUDIT_AND_SELECTION_LOCK on the reviewed hashes.** 7 REQUIRED, all resolved and verified by their fixtures. 11 RECOMMENDED: 10 adopted, 1 open (SEL-C11, reporting only). 5 NOTES |

**Finding classes:**
- **REQUIRED:** a demonstrated defect with a failing fixture and a patch. It blocks the Stage 1 freeze until resolved.
- **RECOMMENDED:** a change the lead should make; it does not block the freeze.
- **NOTE:** recorded for the reader; no change required.

## 1. Files reviewed

| File | First read (UTC) | sha256 prefix at the last full run (17:24Z, HEAD 1739cddf) |
|---|---|---|
| `HEADROOM_SELECTION_RULES.json` | 16:58 | ee0f88d6cddc |
| `LABEL_TRUTH_TABLE.json` | 17:00 | e78d408b00b2 |
| `cbp/family.py` | 17:00 | 1431a9873c75 |
| `cbp/select.py` | 17:02 | 527bf2f14c44 |
| `cbp/infer.py` | 17:08 | 514198a436d0 |
| `cbp/eval_lock.py` | 17:09 | ca8b2c12d273 |
| `PRIMARY_FAMILY.json` | 17:13 | 06693acd39bd |
| `PROTOCOL.md` §§8–12 | 17:13 | 1971fffb5d1a |
| `cbp/review_tests/test_selection_review.py` | (author) | 65bceab27c1e |

The lead's commit chain during this review is 21a97bc → 75dbc30 → 807fa89 → 0a385a86 → dfcf4cf → 1739cddf. The last full run, on HEAD 1739cddf at 17:24Z, was **44 passed, 0 failed, 2 xfailed** (SEL-N3 NOTE and SEL-C11 RECOMMENDED).

## 2. What was checked against the prompt, item by item

| Prompt requirement | Where | Result |
|---|---|---|
| Ordinary eligibility per seed and task, with no seed averaging (§9) | `select.candidate_rows` via `qpc.utility.gate_record`; config = all seeds | OK. Fixture `fixture_seed_average`: a code passes on the seed mean (dLL 0.0052) and fails seed 2 (0.0115). It is rejected, while the seed-mean defect would nominate it |
| Headroom ≤ 0.006 nats LL and ≤ 0.0035 Brier, each task and seed, privacy nominees only (§9) | `select.HEAD_LL/HEAD_BR`, recomputed from the excesses (not qpc's 0.0075 flag) | OK. A JOINT code at dLL 0.008 is excluded from P\* and J\* but is the ordinary-winner diagnostic. Brier 0.0034 passes and 0.0036 fails. The "headroom at the ordinary limit" defect is caught |
| The same headroom for LOCAL, SEQ-12, SEQ-21 and JOINT, with no reward for the JOINT name (§9) | `pick(priv, need_headroom=True)`; alias set (SEL-R2) | OK after SEL-R2 |
| Comparators need ordinary eligibility only and are not dropped for failing headroom (§9) | `pick(..., nominee=False)` | OK. The strongest nonjoint control fails only headroom and stays C_rate and C_global |
| T\* closed list: DIRECT, FINE, U, CLASS-ONLY, F0; RAW-J excluded and disclosed (§9) | `T_STAR_CLOSED`; rules `roles.T*` | OK. RAW-J would be T\* under the legacy list; here it stays out of T\* and is C_global |
| C_rate (20 candidates) and C_global (26 candidates, JOINT excluded) (§9) | `NONJOINT_CODES`; the C_global list | OK. Oracle equality on 40 random banks |
| Guards per recipient per seed, ≤ comparator + 0.005, inclusive (§9) | `select.guard_ok` | OK after SEL-R4. Per-seed is binding: a +0.0051 excess on one recipient and seed rejects the code even though its seed mean is +0.0044 |
| Ordering: mean INNER pair AUC, mean summed LL, actual states, config ID; tie tolerances registered; continuous counts +inf for ordering only, with finite JSON (§9) | `select.key`; rules `ordering` | OK after SEL-R1. SELECTION.json and selection.json are strict-finite, and SRC\|U `mean_states` is null |
| Fallback ordering (ordinary, then headroom, then local-guard shortfall) with the exact formulas; decision failure or non-estimability is INVALID; a missing comparator gives no guard rank (§9) | `ordinary_shortfall`, `headroom_shortfall`, `guard_shortfall`, `pick` | OK after SEL-R6 and SEL-C1. All 24 privacy shortfalls equal the oracle to 1e-12, and the oracle checks "shortfall = 0 iff eligible" on 4,000 random draws |
| Prespecified diagnostics: ordinary winner without headroom, each family's headroom winner, source λ 0.1 controls, headroom changes the winner and how much (§9) | `select_all` diagnostics; `HEADROOM_VS_STANDARD_SELECTION.csv` | OK after SEL-C3. The guarded and unguarded readings are both reported, give-up is ≥ 0 by nesting with statuses for absent cases, and the oracle's family winners match |
| One global configuration; no assessment-based choice (§9, §12) | `select` reads `inner__*` only; `eval_lock` reads selection.json only; `infer` takes roles from the lock | OK. Planted `assessment` and `outer` fields change no role (select). An arm with the lowest assessment pair AUC does not displace the locked P\* (infer) |
| 37 slots and clauses, kinds, targets, strict sides (§11) | `family.PRIMARY`, `PRIMARY_FAMILY.json` | OK. The JSON slots equal the executable |
| z = NormalDist().inv_cdf(1 − 0.05/74), verified (§11) | `family.Z_PRIMARY` | OK. Independent erfc bisection gives 3.204845205010585 (difference 2e-14). Wrong values (0.05/37, 0.05/148, 0.05/66, 0.975) are all at least 0.033 away |
| B = 1999, exact-record-group bootstrap, seed 20261008, identical draws, equal-seed aggregation (§11) | `infer.main`, `UnitBootstrap(ctx.units, 1999, 20261008, 250)` | OK. `UnitBootstrap` receives `assess_unit`. P06's SE is reproduced to rel 1e-9 by an independent replay (sequential multinomial draws over sorted unique groups, one stream). A row-level bootstrap gives a different SE, so the wrong-group defect is caught. Points equal equal-weight seed means of per-seed paired differences (AUC third-checked with sklearn) |
| True-label loss, not KL (§14) | `infer.row_losses`; select reads `utility.logloss` | OK. The fixture's KL-to-teacher value differs from the true-label value by more than 1e-4, and infer matches the true-label value |
| Truth table and labels (§11) | `family.claim_status`, `q_status`, `overall_label`, `LABEL_TRUTH_TABLE.json` | OK after SEL-R3. The oracle agrees on every nominee × comparator × control state, every single failed clause in every failure kind, and all 192 A/B/C/Q label combinations |
| The classifier distinguishes fit/admission, ordinary, headroom, local guard, precision, measured violation and complete pass (§11) | `SELECTION_REASONS`; `clause_outcome`; `cause_by_kind` | OK after SEL-R3, SEL-C7 and SEL-C8 |
| Tests for absent nominee, absent comparator, aliases, every single failed clause, nonfinite values, zero-variance identities and mixed claims (§11) | This file, §4 | OK. Nonfinite attacker scores are covered after SEL-R7 |
| Deliberate defects (§14): seed-averaged eligibility; headroom at the ordinary limit; KL; wrong bootstrap group or critical value; absent comparator passing; assessment-rescored nominee | §4 | All six are discriminated by named fixtures, and the lead's code passes each one |

## 3. Findings

### REQUIRED

**SEL-R1. Resolved (807fa89).** The tie-tolerance text contradicted the rounding rule.
- The text said "rounding to 12 decimals (values within 5e-13 are tied)".
- Rounding is a quantisation, not a tolerance. For example, 0.81660000000049 and 0.81660000000051 are 2e-14 apart but round apart.
- The text now states the exact rule: the seed mean (s0 + s1 + s2)/3 in seed order, then `round(x, 12)`; equal rounded values tie.
- Fixture: `test_registered_tie_tolerance_text_matches_the_rounding_rule`.

**SEL-R2. Resolved (807fa89).** The exact-alias tie-break rewarded the JOINT name.
- `"U|JOINT|"` sorts first, and JOINT keeps unchanged same-λ SEQ and LOCAL witnesses. A JOINT unit can therefore be bitwise a SEQ map and win only on the configuration ID.
- The ordering is unchanged, as the prompt prescribes. The label now names `simplest_family` of P\*'s alias set (by deployed-map fingerprint, all seeds). Partial aliases and the ID tie-break are disclosed.
- After SEL-C9 the alias set also flags identity with FINE-TASK or DIRECT-TASK.
- Fixture: `test_unchanged_joint_alias_names_the_simpler_family`.

**SEL-R3. Resolved (807fa89).** `q_status` lacked the measured-violation class and accepted an empty clause set.
- A Q clause with its lower bound above 0.01 was reported as CLAUSE_NOT_ESTABLISHED.
- An empty clause set was reported as a precision failure.
- `q_status` now mirrors `claim_status`: MEASURED_VIOLATION_SUPPORTED_BY_BOUND, NO_CLAUSES and MISSING_SLOTS, returned as a 3-tuple.
- Fixture: `test_q_status_distinguishes_measured_violation_and_refuses_empty`.

**SEL-R4. Resolved (807fa89).** The guard was not inclusive at its registered bound.
- The code decided the guard by `(a − g − 0.005)/0.005 == 0`. For a = g + 0.005 in float, this is a few ulp above zero.
- All 100,000 random exact-boundary cases were rejected. The registered "≤", and qpc, accept them.
- The guard is now decided by the comparison `a <= g + 0.005`, and the shortfall is exactly 0 iff the guard holds.
- Fixture: `test_select_guard_is_inclusive_at_the_registered_bound`.

**SEL-R5. Resolved (807fa89).** In the 21a97bc version, the executable and the text disagreed on a missing guard comparator with no eligible candidate.
- The code said INVALID_NOMINEE; the truth-table text said NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE.
- The lead changed the code to follow the text after batch 2's SEL-N2, before I sent this as a finding. The code is now INVALID only if some candidate is eligible.
- Fixture: `test_missing_guard_status_agrees_with_registered_truth_table`.

**SEL-R6. Resolved (807fa89).** A missing guard comparator was hidden by a zero field in the fallback rank.
- The fallback was ranked with `guard_shortfall or 0.0`, and the gap was labelled `missing_guards_not_needed`. §9 says: "do not use a zero field to hide missing comparator coverage … report that dependency as invalid".
- The fallback is now ranked on the ordinary shortfall, the headroom shortfall and the ordering only, with `fallback_rank_status = INVALID_MISSING_GUARD_COMPARATOR` and `missing_guards`. eval_lock and infer carry these fields.
- Fixture: `test_missing_guard_fallback_rank_is_not_a_zero_field`.

**SEL-R7. Resolved (dfcf4cf; verified 17:24Z).** A nonfinite attacker score yielded a finite, sortable AUC.
- `infer.Ctx.rec` passes the saved `P_auc_*` scores to `stored_model_eval.pilot_infer._auc_prep`. That function ranks scores with `np.unique`, which sorts NaN last, so a NaN row silently becomes the most-SEX=1 row.
- In the fixture, one NaN (J\*, seed 1, attacker seed 1, row 0) leaves P01, P12 and P23 with numeric outcomes (PASS), not INVALID.
- §6 and §11 require nonfinite primary quantities to be INVALID, never sortable successes. Neither cbp.assess nor cbp.infer checks saved scores.
- NaN released probabilities are already handled: log loss and Brier become NaN, so the slot is INVALID (tested). The inner path is safe because `cbp.audit._auc_fixed` uses sklearn, which raises on NaN.
- Patch, given to the lead:
  1. In `Ctx.rec`, register NaN bases when `np.isfinite(P3[..., 1]).all()` fails, so the existing nonfinite-replicate rule makes the slot INVALID.
  2. Add a per-arm finiteness receipt in `inference.json`.
  3. Ask role D to make `cbp.assess` refuse, or count, nonfinite score arrays.
  4. Apply the same in F's replay.
- Fixture: `test_infer_nonfinite_primary_is_invalid_never_dropped`.
- Resolution: `Ctx.rec` registers NaN bases for any nonfinite score column 1, so the slot is INVALID. `inference.json` carries a per-arm `finiteness_receipt` over `P_auc_*`, `P_ce_*`, `prob*` and `hard*`. Role D's `cbp.assess` refuses nonfinite outer records. The fixture now passes: P01, P12 and P23 are INVALID, and A, B and C are INCOMPLETE_OR_INVALID (NONFINITE_PRIMARY_QUANTITY).

### RECOMMENDED

| ID | Finding | Status |
|---|---|---|
| SEL-C1 | A decision-preservation failure of any U-derived candidate is a technical failure: the role is INVALID and the candidate is never ranked. It is not merely ineligible, which is what qpc's `gate_record` alone gives | Adopted; tested |
| SEL-C2 | State Q's consequence when DIRECT-TASK i8o64 is not ordinarily eligible | Adopted: NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE, with P34–P37 DESCRIPTIVE_ONLY |
| SEL-C3 | Define the headroom-vs-standard give-up: sign, per-seed values, a status when either side is absent, and a DESCRIPTIVE flag on fallbacks. Report the unguarded reading as well. JOINT-family winner ≠ J\* | Adopted |
| SEL-C4 | References follow the admission record; a technical failure of an admitted candidate makes its role INVALID | Adopted |
| SEL-C5 | Enforce the registered slot set (no shrunken conjunction) | Adopted as MISSING_SLOTS via `claim=`; see SEL-N3 |
| SEL-C6 | Technical-validity scope per claim: a failure touching one claim must not erase another claim's valid PASS (§11, "affected claim") | Adopted: `failure_scope`, and infer `--failures` routes per-claim failures |
| SEL-C7 | Root cause by clause kind, so a measured pair violation is never read as an established confidence violation (§13, §17) | Adopted: `cause_by_kind` |
| SEL-C8 | Exact technical root-cause codes (MISSING_UNIT, DECISION_PRESERVATION_FAILURE, NON_ESTIMABLE_INNER_METRIC), not FIT_OR_ADMISSION_FAILURE for everything | Adopted; tested |
| SEL-C9 | The alias set should include FINE-TASK and DIRECT-TASK, so a privacy nominee bitwise equal to an untrained code is flagged | Adopted (`identical_to_untrained`) |
| SEL-C11 | `HEADROOM_VS_STANDARD_SELECTION.csv` (a named deliverable) lacks four things that exist only in SELECTION.json diagnostics: the unguarded-reading row; the three source λ 0.1 control rows (JOINT, SEQ-12, SEQ-21); a summary of whether headroom changes the winner and the pair AUC given up (mean and per seed); and a DESCRIPTIVE / `fallback_rank_status` column on fallback rows | **Open** (xfail fixture `test_headroom_vs_standard_csv_carries_every_prespecified_diagnostic`) |
| SEL-C10 | `infer` `clauses_passing` counts numeric PASSes on DESCRIPTIVE_ONLY fallback rows: C = 10 for an ineligible fallback in the fixture. Rename it to `clauses_passing_numeric`, or count only `decision == PASS` | Adopted (dfcf4cf): `clauses_passing_scored` (decision == PASS) and `clauses_passing_numeric_including_descriptive`; the fixture passes |

### NOTES

**SEL-N1. Data-dependent role aliases.**
- In the source, C_rate and C_global were the same configuration (U\|SEQ-21\|i8o64\|l0.1), so claim B's P12–P14 equal claim A's P01–P03.
- P\* = J\* and T\* = Q config are also possible.
- These aliases are now recorded per slot (`alias_of_by_role`), and the slots are kept. Prose must count each comparison once.

**SEL-N2. Asymmetric precedence.**
- A comparator TECHNICAL_FAILURE outranks a nominee NO_ELIGIBLE, but a comparator NO_ELIGIBLE does not.
- This is registered and conservative.

**SEL-N3. `claim=` is optional.**
- `claim_status(claim=None)` still accepts a shrunken conjunction. Infer always passes `claim=`, and test_late covers MISSING_SLOTS through infer.
- Making the argument mandatory would close the gap for future callers (xfail `test_claim_status_requires_claim_argument`).

**SEL-N5. PROTOCOL wording (17:13Z read).**
- §8's fallback paragraph states only the branch "guard comparator missing and some candidate eligible → INVALID (MISSING_GUARD_COMPARATOR)".
- Add the other registered branch: when nothing is eligible, the status is NO_ELIGIBLE. The fallback is then ranked without the guard key and gets `fallback_rank_status = INVALID_MISSING_GUARD_COMPARATOR` (SEL-R6).
- §12 should say that the unguarded "strongest ordinary privacy" diagnostic is inner-only. The scored list carries the T\*-guarded reading.
- The JSON files already say both, so this is wording only.
- Adopted in 1739cddf, together with SEL-N4 of batch 4: `Ctx.lab` maps a resolved config that is missing from the scored labels to the INVALID path.

**SEL-N4. What the 0.006 headroom can and cannot buy (interpretation; not a defect).** The lead takes this into the interpretation. PREDICTIONS.json was registered at 357b300 and is not edited. The source assessment interval for the occupation log-loss excess was 0.008072 with upper bound 0.012122. That implies SE ≈ 0.00126 and z·SE ≈ 0.00405 nats. The headroom therefore leaves almost exactly one z·SE before the 0.01 limit. If a nominee's assessment excess has that SE, the probability that clause 7 passes is approximately:

| True assessment excess (nats) | P(clause 7 passes) |
|---|---|
| 0.004 | 0.94 |
| 0.005 | 0.77 |
| 0.006 | 0.48 |
| 0.007 | 0.20 |

Three effects work against the nominee:
- The inner estimates rest on 2,235 rows, roughly √(13,929/2,235) ≈ 2.5 times the assessment SE (about 0.003).
- The rows are shared across seeds, so the per-seed headroom check does not average this noise away.
- P\* is the most private code that clears the inner limit, so its inner excess is selected to be low (a winner's curse on the constraint).

A nominee near the headroom boundary therefore has roughly even or worse odds on clause 7. The headroom is a design choice, as the prompt says, not a precision guarantee. PREDICTIONS.json and the final interpretation should not treat passing headroom as making clause 7 likely.

## 4. Test inventory (`cbp/review_tests/test_selection_review.py`)

**Oracle self-tests.** These do not import cbp:
- `test_z_is_verified_not_copied`;
- `test_shortfall_formulas_hand_computed`;
- `test_ordinary_shortfall_is_zero_iff_ordinarily_eligible`;
- `test_clause_classifier_distinguishes_precision_point_and_established_violation`;
- `test_label_oracle_mixed_and_absent_cases`;
- `test_bank_dimensions_match_prompt`;
- `test_fixtures_discriminate_the_deliberate_defects`, where each defect variant changes the oracle's answer;
- `test_guard_is_per_seed_per_recipient_inclusive`;
- `test_missing_comparator_gives_no_guard_rank`;
- `test_random_banks_basic_invariants`.

**Registered files:**
- `test_rules_json_registers_prompt_section_9_constants`;
- `test_registered_tie_tolerance_text_matches_the_rounding_rule` (SEL-R1);
- `test_primary_family_json_equals_executable`;
- `test_protocol_selection_text_carries_the_registered_numbers`;
- `test_registered_truth_table_text_agrees_with_executable_reasons`.

**cbp.family:**
- `test_family_slots_constants_and_independent_z`;
- `test_clause_outcome_matches_oracle_off_boundary` (20k intervals; includes the zero-variance identity and NaN/inf/None);
- `test_claim_status_matches_oracle_for_every_role_and_single_clause_failure`;
- `test_q_status_distinguishes_measured_violation_and_refuses_empty` (SEL-R3);
- `test_claim_status_refuses_a_shrunken_conjunction`;
- `test_claim_status_requires_claim_argument` (xfail, SEL-N3);
- `test_overall_label_matches_oracle_exhaustively` (mixed valid/incomplete claims displayed).

**cbp.select.** These use synthetic `inner__*` records, with `cbp.run` record access monkeypatched and outputs written to tmp:
- `test_select_matches_oracle_on_random_banks` (40 banks, nominees and fallbacks of P\* and J\* exercised);
- `test_select_every_seed_not_seed_mean`;
- `test_select_headroom_is_0006_not_ordinary_limit`;
- `test_select_comparators_keep_headroom_failures`;
- `test_select_t_star_closed_list_excludes_rawj`;
- `test_select_guard_is_inclusive_at_the_registered_bound` (SEL-R4);
- `test_select_decision_failure_is_invalid_not_a_shortfall`;
- `test_select_nonfinite_candidate_is_invalid`;
- `test_missing_guard_status_agrees_with_registered_truth_table` (SEL-R5);
- `test_select_ignores_assessment_and_kl_fields`;
- `test_select_json_finite_and_continuous_states_null`;
- `test_unchanged_joint_alias_names_the_simpler_family` (SEL-R2);
- `test_missing_guard_fallback_rank_is_not_a_zero_field` (SEL-R6);
- `test_headroom_vs_standard_csv_carries_every_prespecified_diagnostic` (xfail, SEL-C11).

**cbp.infer.** These run `infer.main` end to end on synthetic preds.npz and EVALUATION_LOCK files written to tmp:
- `test_infer_bootstrap_groups_seed_B_and_z`;
- `test_infer_points_equal_seed_means_with_true_label_loss` (sklearn AUC third check; KL discrimination);
- `test_infer_group_bootstrap_se_reproduced_independently` (wrong-group discrimination);
- `test_infer_zero_variance_identities_pass_and_are_kept`;
- `test_infer_nonfinite_probability_makes_loss_slots_invalid`;
- `test_infer_nonfinite_primary_is_invalid_never_dropped` (SEL-R7);
- `test_infer_absent_comparator_never_passes`;
- `test_infer_fallback_nominee_never_passes`;
- `test_infer_roles_come_only_from_the_lock`;
- `test_infer_clauses_passing_counts_only_scored_decisions` (SEL-C10).

Run:
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m cbp.sema \
    --label C:selection-review -- ~/PCRL/.venv/bin/python -m pytest -q -p no:cacheprovider -rxX \
    cbp/review_tests/test_selection_review.py
```

## 5. Limits of this review

- **Synthetic only.** The oracle checks that the executables implement the registered rules on synthetic banks and predictions. It does not check that cbp.audit's inner records contain the right attacker AUCs or the composed-source closure (role D and the verifier F).
- **Boundary semantics.** These are tested where the registered text fixes them: inclusive eligibility, the inclusive guard, strict clause bounds, and Brier/LL excesses computed as c − u. Exact float boundaries of the excess comparisons (e.g. 0.2285 − 0.225 = 0.003500000000000003) behave identically in the oracle and the code, and are measure-zero on real data.
- **Independence.** The oracle is independent of cbp and qpc code. It was written by the reviewer from the prompt text, so it is not a second scientific replay. F's replay (no runner imports) remains the independent verification required by §14.

## 6. Verdict for the Stage 1 freeze (17:24Z)

**Role C finds no open REQUIRED item on HEAD 1739cddf.**
- The reviewed hashes are those in §1.
- The selection, family, label and inference executables and their registered JSON and PROTOCOL text agree with the independent transcription of prompt §§9, 11 and 14 on every tested case.
- The six deliberate defects in this scope are each caught by a named fixture.

**Still open (RECOMMENDED; does not block):** SEL-C11, the missing rows and columns in `HEADROOM_VS_STANDARD_SELECTION.csv`.

**If a reviewed file changes after this verdict:** the suite should be re-run (command in §4) before the lock, and the new hashes recorded here.
