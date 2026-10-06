INTERIM STATUS: 0 REQUIRED (2026-10-05; R1-R3 applied by the lead and re-verified: 65/65 reviewer tests, 157/157 dpc/tests; 52/53 injected defects caught, the 1 survivor equivalent)

# Math / design review: decision-preserving joint compression

**Role and files.** Math/design reviewer (role 3). This role owns three files: this file,
`PRIOR_ART_AND_BASELINE_GAPS.md`, and `dpc/tests/test_math_review.py` (which also contains the exhaustive-fixture
runner). The reviewer edited no other file, committed nothing and pushed nothing.

**Finding classes.**
- REQUIRED: a demonstrated defect, with a failing fixture and a proposed patch.
- RECOMMENDED: a change the reviewer advises.
- NOTE: a limit or scope statement, retained for the record.

**Commands.**

    cd <worktree> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest dpc/tests/test_math_review.py -q
    cd <worktree> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.tests.test_math_review --exhaustive --out <json>

**Reviewed code.** Everything is as committed at `e3aa42612`. Reviewed sha256 prefixes:

| File | sha256 prefix |
|---|---|
| partition.py | 4b1d0febb751 |
| release.py | 0f4b7e26f70d |
| compress.py | 96988629698b |
| deploy.py | 504eaeac43c3 |
| audit.py | e024235b4854 |
| utility.py | 23a07815c744 |
| assess.py | 434103381684 |
| baselines.py | 2ac7b4b6f130 |
| controls.py | d02fa61a7343 |
| select.py | e51fddb41abf |
| family.py | f718a598f9c1 |
| infer.py | a0a3d3b34496 |

**What the reviewer did and did not touch.**
- Independence: the REFERENCE section of the test file imports nothing from `dpc`. Every checked quantity is
  recomputed from rows with reference code, and `dpc` entry points are used only to produce the objects under test.
- Real Adult data: the only contact was one light, label-free shape check (section 6).

## 0. Pre-lock answer to the lead: audit, utility, assess, baselines and controls

**Result: no REQUIRED finding on these files.**

**Cell-conditional readers: compliant, not a REQUIRED deviation.**
- The readers use (n_t1 + a·pi_1)/(n_t + a) with a in {0.5, 1, 5}. This is a smoothing of strength a, meaning a
  total pseudo-count a placed at the AUDIT_FIT prior.
- At n_t = 0 it reduces to pi_1, which is the registered fallback for unseen tokens. Reader and fallback are
  therefore one coherent rule.
- A symmetric (n_t1 + a/2)/(n_t + a) would shrink toward 0.5 rather than toward the 0.68/0.32 prior.

**Verified by tests.**
- The formula, the prior on unseen tokens and renumbering invariance.
- The exact pair tuple, checked against a brute-force dictionary.
- The fallback rule: the lowest CE on unseen INNER tuples, with ties broken in the order local_1, local_2, prior.
- No `_pair_keys` collisions.
- Decoder collisions: a token whose ID carries S is detected separately by the CC readers and by the slate's one-hot
  columns, while a decoded-q-only reader stays near 0.5.
- XOR coalition and null releases.
- Fixed AUC orientation: no flip on an anti-informative bank, and weighted `class_auc` equals sklearn with
  sample_weight.
- Utility formulas against sklearn, and non-strict per-task gates.

**Scope notes (no change required).**
- Composed source AUC is the maximum of many selection-optimistic inner AUCs. That inflates SRC rows and loosens the
  J*/P* guards whenever C_global or T* is a source. This is what spec section 11 mandates.
- `dpc/controls.py` re-exports `dpc.audit.stage_controls`. The plants (CONF, COLL, XOR) are structurally correct. The
  reviewer has not executed them on real data, by role.

## 1. Test inventory (65 test cases in 34 functions, all passing; about 17 s on one thread)

| Spec item (sections 8, 10, 12-14) | Tests |
|---|---|
| Class preservation: binary and six-class, ties (two-way, K-way, non-zero index), exact 0/1 underflow, unseen predicted classes, empty fitting classes, JSON and npz round trips (bit-exact), decision a function of the token, a wrong supplied decision array refused | `test_smoothing_rule_sum_argmax_ties_underflow`, `test_mean_of_same_class_vectors_keeps_class_in_float64`, `test_class_preservation_pointwise_ties_underflow_unseen_and_roundtrip`, `test_fine_partition_matches_documented_rule_and_row_statistics` |
| No grouping by true Y, SEX, per-row loss or assessment labels; planted true-label partition refused or non-representable, including a single borderline mislabelled row hidden in a cell whose mean keeps its argmax | `test_partition_refuses_label_keyed_groupings`, `test_class_only_aliases_and_label_refusal` |
| KL merge costs and MI changes against brute-force row recomputation; pair table, row totals and local marginals after merges and exchanges on either recipient | `test_kl_zero_terms_and_merge_cost_sufficient_statistics`, `test_plugin_mi_matches_closed_form_and_ignores_empty_cells`, `test_merge_and_exchange_deltas_and_tables_against_rows`, `test_method_objectives_match_row_recomputation` |
| Algorithm identity with an independent implementation of section 9: greedy, lexicographic tie rule, best-move refinement, sweep rules | `test_method_equals_independent_reference_algorithm` (7 settings), `test_greedy_lexicographic_tie_rule`, `test_refinement_takes_the_best_move_wide_fixture` |
| Sequential freezing and the stage-one coefficient | `test_stage_one_coefficient_is_one_point_five_lambda`, `test_sequential_freezing_and_stage_one_coefficient` |
| Nested JOINT witnesses; a dropped strong initialisation is detected | `test_reference_joint_dominates_its_witnesses`, `test_joint_final_not_worse_than_any_witness_or_refined_start` (5 fixtures, each start uniquely best on one) |
| Token identities rather than decoder collisions; a secret S-token is detected | `test_decoder_collision_token_secretly_carrying_S_is_detected` |
| AUC orientation; sparse and unseen token pairs; shared exact-record bootstrap indices | `test_auc_orientation_is_fixed_and_never_flipped`, `test_pair_reader_exact_tuple_fallback_rule_and_no_key_collisions`, `test_cell_reader_formula_unseen_prior_and_renumbering`, `test_infer_end_to_end_against_independent_bootstrap` |
| Coalition-positive and null fixtures (objectives and audit) | `test_coalition_fixture_terms`, `test_coalition_fixture_through_method`, `test_null_fixture_plugin_mi_is_finite_sample_optimism_only`, `test_coalition_and_null_releases_through_finite_readers` |
| Structural statements of section 8 | `test_chain_rule_identities_on_fitting_law`, `test_data_processing_conditional_on_public_state` |
| Selection: per-seed gates, C_match, C_global, T*, J*, P*, tie order, missing comparators | `test_selection_matches_independent_reference` (12 random banks against an independent selector), `test_selection_designed_edge_cases` |
| Family and inference: 33 slots, z, strict thresholds, gain-retention algebra, multinomial group bootstrap with one draw sequence, seed averaging, DESCRIPTIVE_ONLY | `test_family_slots_z_and_claim_rule`, `test_infer_end_to_end_against_independent_bootstrap` |
| Deployment from the 83-column input; protected interface only | `test_deploy_schema_and_interface_refusals`, `test_deploy_requires_the_registered_purpose_shapes` |
| Utility formulas | `test_utility_metric_formulas_against_sklearn` |

**What was verified, in short.**

**Smoothing and class preservation.**
- The prototypes are finite and positive, sum to 1 within 1e-15·K, and have a strict argmax d.
- Their deviation from the barycentre is at most (K+2)·1e-12.

**Fine partitions.**
- `fit_fine` reproduces an independent implementation of the documented k-means rule exactly: the same
  assignments, and means equal to 1e-15 when converged.
- Stored n, S and A equal the row statistics of nearest-centroid deployment on the fitting rows, so the k-means "stale
  last assignment" problem is handled.

**Objectives.**
- Every family's receipted D1, D2, I1, I2, I12 and F values equal recomputation from released rows to 1e-10
  relative.
- Every merge and move increment (dD, dI_own, dI12) and every table after an update equal brute force.

**Algorithm identity.**
- The method's FINE-TASK, LOCAL, SEQ-12, SEQ-21 and JOINT solutions equal the reviewer's reference implementation
  on every tested fixture.
- On the remaining fixture settings the comparison is by objective value, through the exhaustive runner.

**SEQ.**
- The first map is the D + 1.5 lam I solution, and it differs from the 1.0 lam solution on those fixtures.
- The first map is invariant to any replacement of the other recipient's scores.

**JOINT.** The final F_joint is at most every unchanged witness and at most the reviewer's own refinement of every
start.

**Inference.** On a synthetic assessment with duplicate exact-record groups, every primary point, SE (ddof 1), bound
and strict decision equals an independent multinomial-group bootstrap. That bootstrap uses B = 1999, seed 20261007,
one draw sequence for all arms and seeds, and the pairwise-definition AUC.

## 2. Exhaustive finite fixtures (retained gaps; NOT an Adult solver certificate)

**Fixtures.**
- `small_seed0` and `small_seed3`: recipient 1 binary with 4+4 fine cells; recipient 2 three-class with 3+3+3.
- `coalition`: S = A xor B, with 4 cells on each recipient.
- `null`: S independent of the scores.

**Settings.** m in {1,2,3} and lam in {0.1,1,10}. Every class-preserving map with exactly min(m, n_c) cells per class
is enumerated: 49 x 27 pairs at m = 2 and 36 x 1 at m = 3. The "at most m" (cap) space was also enumerated, with up to
196 x 125 pairs.

**Definitions.**
- Gap = family value minus the global minimum of the family's OWN objective over the exactly-m space.
- SEQ arms are measured against F_joint. Their stage-1 gap and conditional stage-2 gap are listed separately.

The method's solutions are identical to the reference implementation in every setting. The run was repeated on the
current code and gave an identical summary.

| Family | Globally best on its own objective | Largest gap | Detail |
|---|---|---|---|
| CLASS-ONLY (m = 1) | 12/12 | 0 | One map |
| FINE-TASK | 24/24 | 0 | Also optimal on the cap space |
| LOCAL | 24/24 | 0 | Cap-space gap up to 0.030 (fixed rate, N1) |
| DIRECT-TASK (F_task) | 12/24 | 0.0081 | Row k-means; inside the fine-state family only on these fixtures (N5) |
| SEQ-12 (F_joint) | 3/24 | 1.72 | Stage-1 gap 0 everywhere; conditional stage-2 gap at most 0.0064 |
| SEQ-21 (F_joint) | 15/24 | 1.72 | Stage-1 gap 0 everywhere; conditional stage-2 gap at most 0.040 |
| JOINT (F_joint) | 20/24 | 0.0398 | Gaps below |

**JOINT gaps.**

| Fixture | m | lam | Gap |
|---|---|---|---|
| small_seed0 | 2 | 10 | 0.0398 |
| small_seed3 | 2 | 1 | 0.00021 |
| coalition | 2 | 1 | 0.0064 |
| coalition | 2 | 10 | 0.0064 |

These are local optima of greedy plus single-cell moves. On the coalition fixture, for example, the optimum needs two
simultaneous moves. They are labelled fixture gaps.

**Not claimed.** No global-optimality claim is made for Adult. Nothing here shows that a lower fitted F_joint gives
lower held-out recovery.

## 3. Findings

### REQUIRED

**None.** No demonstrated defect was found in any reviewed file.

### RECOMMENDED

**R1 (`dpc/select.py`, T\*). REF|F0 is excluded from T\*.**
- REF|F0 is the official no-fairness matched compression, so it is a privacy-untrained score/compression release.
  T* is drawn only from FINE-TASK, DIRECT-TASK, CLASS and SRC.
- Patch: `T_star = pick([rows[c] for c in ids if family(c) in UNTRAINED + ("SRC",) or c == "REF|F0"], none="NO_FEASIBLE_CONTROL")`.
- Alternatively, record the exclusion and its reason in PROTOCOL.md section 8 before SELECTION_AND_AUDIT_LOCK.
- Not a demonstrated defect: section 13's T* list says "including" and is not exhaustive.

**R2 (`dpc/infer.py`, Ctx). Labels and sex are not checked across arms.**
- Rows and groups are asserted equal across arms, but `sex`, `y_income`, `y_occ` and `const_class` are taken from the
  first arm only.
- Patch: add `assert all(np.array_equal(p[x], z0[x]) for x in ("sex", "y_income", "y_occ", "const_class"))` in the
  same loop.

**R3 (`dpc/infer.py`, `complete`). One missing comparator invalidates the whole study.**
- Any INVALID_* status makes the global label INCOMPLETE_OR_INVALID, even when the other claim's nominee and comparator
  are valid. Section 13 says a missing comparator invalidates "the relevant claim".
- The current rule is conservative. State it explicitly in PROTOCOL.md section 9, or record per-claim validity.

### NOTES

**N1 (fixed rate).**
- The families search the exactly-min(m, n_c) space: greedy stops at the cap, and refinement never empties a cell.
- Where n_c = m, no merge is possible. On the 2-cell coalition fixture every family keeps the full coalition leak at
  m = 2.
- Allowing at most m cells would lower the minimum F_joint by up to 1.69 nats at lam = 10 on these fixtures.
- This is the registered fixed-rate design; lower rates are separate bank entries.

**N2 (SEQ stage one).**
- Stage one treats the other recipient as unreleased: D_1 + 1.5 lam I(S;C_1).
- In fact d_2 is always released, so the coalition always holds at least (C_1, d_2).
- This is a scope limit of the "sequential" control, which follows section 9 as written.

**N3 (finite-sample optimism).**
- The plug-in bias is about (|alphabet| - 1)/(2N): roughly 0.0005 nats for I_1 at m = 8 and roughly 0.02 nats for
  I_12 at m = 8, with 16 x 41 used pair cells (see N7) on 15,434 rows.
- lam·I_12 therefore partly penalises alphabet size.
- These are training criteria only. On the null fixture, JOINT's lower fitted I_12 is not a held-out benefit.

**N4 (gain gate).** "Retains at least 80% of U's gain" is an accuracy gain, consistent with clauses 10-11. For U-teacher
codes it holds automatically, because decisions are identical. Confidence quality is guarded only by the log-loss and
Brier gates.

**N5 (DIRECT-TASK containment).** On these fixtures DIRECT-TASK lies inside the fine-state family. On Adult it does
not, and JOINT's dominance over its witnesses never covers DIRECT-TASK (section 9).

**N6 (smoothing).** The AIB identity, merge cost = (n_a+n_b)·JS, holds exactly only for unsmoothed barycentres. The
implementation uses smoothed prototypes consistently in fitting, evaluation and deployment.

**N7 (label-free Adult shape facts, read from the fitted fine-partition receipts and teacher decisions; no labels).**
- **Occupation class 5 is never predicted.** On all 39,170 rows, for every teacher and seed (max p_2[:, 5] about
  4.5e-4), the teacher never predicts class 5. Recipient 2 therefore has 5 effective predicted classes plus a single
  fallback token that is never emitted on admitted rows. Every arm's class-5 recall is 0 by class preservation. This
  is a teacher weak class (section 15), not a code defect.
- **Fine-cell counts.** Every effective predicted class has the full 16 fine cells (classes 3 and 4 included), with
  0-4 sparse cells (n < 5) per partition, so no m in {2,4,8} aliases the fine state.
- **K-means convergence.** Most classes reached the 20-round cap without convergence. The stored mean and the
  assignment centroid then differ by up to 1.1e-2 (`max_abs_mean_vs_centroid`). Deployment uses the recorded
  assignment centroid, so this stays consistent with fitting. Report it as non-convergence, not as a bug.
- **Teacher probabilities.** They are float64, with |row sum - 1| <= 6.7e-16 and no exact top-2 ties, so the 1e-9
  input-sum tolerance and the tie rule never bind on admitted rows.

**N8 (what the reviewer did not execute).** The real-data controls (`run_prelock_controls`), `final_audit` on
assessment rows, `dpc.assess` and the full FINAL slate (MLP, HGB, DA) on study-sized data were reviewed by reading only.
The audit tests use a one-LR slate where a fitted slate is needed.

## 4. Injected defects (copies of `dpc/`, never the originals)

The harness copies the package to a scratch directory, applies one textual defect, and runs this test file against the
copy (`PYTHONPATH=<copy>:<worktree>`). An unmutated copy passes 65/65 first.

**Round 1** (earlier test file, 53 mutants): 48 caught. The 5 survivors were:
- P3: label gate disabled;
- R3: encode skips the argmax check on a supplied d;
- C8: first-improvement refinement;
- C11: greedy tie goes to the last candidate;
- D2: unknown deploy flags tolerated.

**Tests added for the survivors.**
- A single borderline mislabelled row hidden in a mean-dominated cell (P3).
- A wide fixture where best-move and first-move refinement differ (C8).
- An exact-tie fixture with mirror-image vectors (C11).
- Unknown-flag cases for deploy (D2).

**Re-run.** P3, C8, C11 and D2 are now caught. R3 is behaviourally equivalent: a wrong d is still refused, through the
class-preservation AssertionError instead of ValueError. A test for exactly that refusal was added.

**Final run** (current test file, all 53 mutants, unmutated copy 65/65 first): **52 of 53 caught**.
- The only survivor is R3, which is behaviourally equivalent.
- That makes **52 of 52 non-equivalent mutants caught**.
- Per-family counts: partition 5/5, release 2/2 (+1 equivalent), compress 14/14, audit 9/9, select 9/9, family 2/2,
  infer 7/7, deploy 4/4.

| Mutant | File | Result | First failing test |
|---|---|---|---|
| P1 smoothing without eps*e_d tie term | partition.py | CAUGHT | `test_fine_partition_matches_documented_rule_and_row_statistics[2-absent0]` |
| P2 last-index argmax tie rule | partition.py | CAUGHT | `test_fine_partition_matches_documented_rule_and_row_statistics[2-absent0]` |
| P3 label-keyed decision array accepted | partition.py | CAUGHT | `test_class_preservation_pointwise_ties_underflow_unseen_and_roundtrip[2]` |
| P4 deployment uses Euclidean nearest cell | partition.py | CAUGHT | `test_fine_partition_matches_documented_rule_and_row_statistics[6-absent1]` |
| P5 k-means init ascending p_c | partition.py | CAUGHT | `test_fine_partition_matches_documented_rule_and_row_statistics[2-absent0]` |
| R1 prototype = unweighted mean of fine means | release.py | CAUGHT | `test_class_preservation_pointwise_ties_underflow_unseen_and_roundtrip[2]` |
| R2 fallback token decodes toward class 0 | release.py | CAUGHT | `test_class_preservation_pointwise_ties_underflow_unseen_and_roundtrip[2]` |
| R3 encode skips argmax(P) check of supplied d | release.py | SURVIVED | equivalent: wrong d still refused |
| C1 stage-one coefficient 1.0 lam | compress.py | CAUGHT | `test_method_equals_independent_reference_algorithm[1-2-1.0]` |
| C2 JOINT drops refined SEQ-21 start | compress.py | CAUGHT | `test_joint_final_not_worse_than_any_witness_or_refined_start[3-2-3.0]` |
| C3 JOINT drops greedy-joint start | compress.py | CAUGHT | `test_method_equals_independent_reference_algorithm[2-2-1.0]` |
| C4 JOINT drops refined FINE-TASK start | compress.py | CAUGHT | `test_joint_final_not_worse_than_any_witness_or_refined_start[35-2-10.0]` |
| C5 LOCAL weight lam instead of lam/2 | compress.py | CAUGHT | `test_method_equals_independent_reference_algorithm[8-2-10.0]` |
| C6 JOINT pair weight lam/2 | compress.py | CAUGHT | `test_method_equals_independent_reference_algorithm[0-2-1.0]` |
| C7 merge dI12 ignored for recipient-2 merges | compress.py | CAUGHT | `test_merge_and_exchange_deltas_and_tables_against_rows` |
| C8 refinement first-improvement instead of best | compress.py | CAUGHT | `test_refinement_takes_the_best_move_wide_fixture` |
| C9 apply_move forgets pair table on recipient 2 | compress.py | CAUGHT | `test_method_equals_independent_reference_algorithm[1-2-1.0]` |
| C10 SEQ stage 2 revises the first recipient | compress.py | CAUGHT | `test_method_objectives_match_row_recomputation[0-2-1.0]` |
| C11 greedy tie -> last candidate | compress.py | CAUGHT | `test_greedy_lexicographic_tie_rule` |
| C12 MI reported in bits (engine and row check) | compress.py | CAUGHT | `test_method_objectives_match_row_recomputation[0-2-1.0]` |
| C13 refinement may empty a coarse cell | compress.py | CAUGHT | `test_method_objectives_match_row_recomputation[0-2-1.0]` |
| C14 greedy stops one merge early (cap m+1) | compress.py | CAUGHT | `test_method_objectives_match_row_recomputation[0-2-1.0]` |
| A1 cell readers keyed on decoded probabilities | audit.py | CAUGHT | `test_decoder_collision_token_secretly_carrying_S_is_detected` |
| A2 slate view without token one-hot | audit.py | CAUGHT | `test_decoder_collision_token_secretly_carrying_S_is_detected` |
| A3 AUC orientation flipped to max(a, 1-a) | audit.py | CAUGHT | `test_auc_orientation_is_fixed_and_never_flipped` |
| A4 pair keys collide (base = max t2) | audit.py | CAUGHT | `test_pair_reader_exact_tuple_fallback_rule_and_no_key_collisions` |
| A5 unseen token -> 0.5 instead of prior | audit.py | CAUGHT | `test_cell_reader_formula_unseen_prior_and_renumbering` |
| A6 one-hot columns by numeric token id | audit.py | CAUGHT | `test_decoder_collision_token_secretly_carrying_S_is_detected` |
| A7 pair fallback rule = highest CE | audit.py | CAUGHT | `test_pair_reader_exact_tuple_fallback_rule_and_no_key_collisions` |
| A8 decoder check disabled | audit.py | CAUGHT | `test_decoder_collision_token_secretly_carrying_S_is_detected` |
| A9 cell reader drops the prior weight | audit.py | CAUGHT | `test_cell_reader_formula_unseen_prior_and_renumbering` |
| S1 tie order: token states before log loss | select.py | CAUGHT | `test_selection_designed_edge_cases` |
| S2 C_global includes JOINT | select.py | CAUGHT | `test_selection_matches_independent_reference[4]` |
| S3 gates pass if ANY seed passes | select.py | CAUGHT | `test_selection_matches_independent_reference[0]` |
| S4 J* guard drops C_global | select.py | CAUGHT | `test_selection_matches_independent_reference[0]` |
| S5 T* excludes continuous sources | select.py | CAUGHT | `test_selection_matches_independent_reference[0]` |
| S6 guard buffer 0 | select.py | CAUGHT | `test_selection_matches_independent_reference[1]` |
| S7 missing comparator guard silently dropped | select.py | CAUGHT | `test_selection_designed_edge_cases` |
| S8 source composition disabled | select.py | CAUGHT | `test_selection_matches_independent_reference[9]` |
| S9 logloss gate allowance 0.02 | select.py | CAUGHT | `test_selection_matches_independent_reference[0]` |
| F1 Bonferroni divisor 33 instead of 66 | family.py | CAUGHT | `test_family_slots_z_and_claim_rule` |
| F2 claim passes with 10 of 11 clauses | family.py | CAUGHT | `test_family_slots_z_and_claim_rule` |
| I1 non-strict lower bound | infer.py | CAUGHT | `test_infer_end_to_end_against_independent_bootstrap` |
| I2 SE with ddof 0 | infer.py | CAUGHT | `test_infer_end_to_end_against_independent_bootstrap` |
| I3 bootstrap seed shifted | infer.py | CAUGHT | `test_infer_end_to_end_against_independent_bootstrap` |
| I4 retain clause with U and const swapped | infer.py | CAUGHT | `test_infer_end_to_end_against_independent_bootstrap` |
| I5 DESCRIPTIVE_ONLY override removed | infer.py | CAUGHT | `test_infer_end_to_end_against_independent_bootstrap` |
| I6 bootstrap over rows instead of exact-record groups | infer.py | CAUGHT | `test_infer_end_to_end_against_independent_bootstrap` |
| I7 seeds averaged on the comparator only (nominee seed 0) | infer.py | CAUGHT | `test_infer_end_to_end_against_independent_bootstrap` |
| D1 reordered columns accepted | deploy.py | CAUGHT | `test_deploy_schema_and_interface_refusals` |
| D2 unknown flags tolerated | deploy.py | CAUGHT | `test_deploy_schema_and_interface_refusals` |
| D3 extra arrays tolerated in the input npz | deploy.py | CAUGHT | `test_deploy_schema_and_interface_refusals` |
| D4 extra outputs tolerated | deploy.py | CAUGHT | `test_deploy_schema_and_interface_refusals` |


## 5. Prior art and baseline gaps

See `PRIOR_ART_AND_BASELINE_GAPS.md`. In summary:
- Neither the official PURIFIER code nor a Taylor et al. solver can be pinned. The PURIFIER repository named in the
  paper is empty (GitHub API: size 0, created and last pushed 2022-11-29), and no Taylor code exists.
- Any secondary comparison would be a re-implementation with disclosed changes, not the official method.
- This does not invalidate the narrow registered comparisons. It rules out any "beats PURIFIER" or "beats Taylor"
  statement and any novelty claim for agglomeration, privacy-funnel objectives or multi-recipient release.

## 6. Real-data contact by the reviewer

The only contact was a label-free shape check of N7, which read fine-partition receipts and teacher decision and
probability arrays from the private units. No SEX, task label, attacker or assessment row was read.

## 7. Re-verification after the lead applied R1-R3 (2026-10-05, working tree)

| Item | Change by the lead | Reviewer test update |
|---|---|---|
| R1 | T* candidates include REF\|F0 (`dpc/select.py` 17973ea12fd1) | The reference selector now uses the same rule; 12 random banks match again |
| R2 | `infer.Ctx` asserts that sex, y_income, y_occ and const_class are equal across arms (`dpc/infer.py` e3fbc023d6d6) | The end-to-end inference test now also checks that an arm with altered SEX is refused |
| R3 | Validity is per claim (`dec[claim]["valid"]`); `family.overall_label` gives a favourable label only to a PASSing valid claim, gives INCOMPLETE_OR_INVALID if any claim is invalid and none passes, and otherwise gives EXPERIMENTAL_NO_ADVANTAGE (`dpc/family.py` 9c77688e2e49) | The label table covers every combination; the end-to-end run records per-claim validity |

**Result.** 65/65 reviewer tests pass, and the full `dpc/tests` suite passes 157/157.

**R3 as implemented.**
- If claim A PASSes but claim B is invalid, no joint label is given, and the label is INCOMPLETE_OR_INVALID. This is
  correct, because JOINT needs both A and B.
- A valid PASSing claim C keeps its label even when claim A or claim B is invalid.

The R1 to R3 entries in section 3 are now resolved.
