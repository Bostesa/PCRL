INTERIM STATUS (2026-10-06 04:48Z): 0 OPEN REQUIRED. Stage A: 2 REQUIRED found and fixed (section 2). Stage B: 0
REQUIRED (section 3). Selection: SEL-R1 and SEL-R2 found and fixed (sections 4.1-4.2). Attackers/assessment: 0
REQUIRED, 1 RECOMMENDED (section 4.3). Mutation testing follows the locks (section 5). Stage A detail: Two REQUIRED runner-binding defects (SA-R1,
SA-R2, both in `qpc/run.py`) were found and are already resolved in `qpc/run.py` a64bb028ae5b, re-verified by the
failing fixture, which now passes. No REQUIRED finding in the mathematics of `qpc/kmeans.py`, `qpc/stagea.py`,
`qpc/release.py` or `qpc/gate.py`. 4 RECOMMENDED (SA-C1 applied; SA-C2 applied in PROTOCOL section 7), 11 NOTES.
Reviewer tests: 49/49 pass (about 11 s).

# Math, invariants and protocol review: confidence capacity and privacy (qpc)

**Role and files.** Role C. This role owns three files: this file, `PRIOR_ART_AND_BASELINE_GAPS.md` and
`qpc/tests/test_math_review.py`. It edited no other file, committed nothing and pushed nothing.

**Finding classes.**
- REQUIRED: a demonstrated defect, with a failing fixture and a proposed patch. It blocks the governing lock until
  fixed.
- RECOMMENDED: a change the reviewer advises.
- NOTE: a limit or scope statement, kept for the record.

**Command** (light; about 3 s on one thread):

    cd <worktree> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest qpc/tests/test_math_review.py -q

**Independence.** The REFERENCE section of the test file imports nothing from `qpc` or `dpc`. Every checked
quantity is recomputed with reference code (scalar `math.log` KL, own nearest-centroid assignment, own k-means
iteration and own k-means++ draw). `qpc` entry points are called only to produce the objects under test. `dpc` is
imported only as the oracle for the A1 reproduction, because reproducing dpc is the registered requirement.

**Real data.** None. The reviewer ran no real-data code, no reference audit and no fit, and read no label.

## 1. Standard statements and their assumptions

These are standard facts, not new theorems. They are not attribute-specific guarantees, and they are not
training-data privacy.

### 1.1 Decision containment

**Assumptions.**
- Every token belongs to exactly one teacher-predicted class, and the released decision is that class. The code
  enforces this: `token_tables` refuses a token that mixes classes, and `encode` refuses a release whose decision
  differs from argmax p.
- The statement holds for any joint law of (S, p1, p2): a population law, or the empirical law of the fitting rows.

**Statement.**
- d_i is a function of C_i. So I(S; C1, C2) = I(S; d1, d2) + I(S; C1, C2 | d1, d2) >= I(S; d1, d2), and likewise
  for each recipient alone.
- Operationally, any reader of the decisions can be simulated from the release.
- For Bayes-optimal readers, the ROC of the finer view dominates that of the coarser view (Blackwell ordering of
  experiments), so population-optimal AUC on the release is at least population-optimal AUC on the decisions.

**What it does not say.**
- It gives no statement about a fixed, fitted attacker family on finite samples.
- The source's measured decision-only pair AUC (about 0.74) is recovery by the declared attackers. It is not a
  universal population AUC certificate. The structural fact is only that the decision is always recoverable.
- Checked on fitting laws by `test_decision_containment_and_chain_rule_on_fitting_law`.

### 1.2 Data processing (post-processing by a public map)

**Assumptions.**
- C_i = g_i(p_i), where g_i is deterministic, public and frozen after fitting.
- For a row independent of the fitting sample, g_i does not depend on that row's S.
- For a fitting row the map may depend on the sample, including SEX for privacy-trained maps. The chain
  S -> p_i -> C_i then holds only conditional on the fitted map.

**Statement.**
- Conditional on the fitted maps, I(S; C_i) <= I(S; p_i) and I(S; C1, C2) <= I(S; p1, p2).
- Merging tokens never increases plug-in MI on a fixed law.
- Operationally, an attacker who holds p_i can compute g_i(p_i). That is why the U continuous-source attack banks
  must include composition with every fitted code before selection (prompt section 10).

**What it does not say.**
- It bounds the release by its source. It does not make the source private.
- It gives no AUC or attacker-family guarantee.
- Checked by `test_data_processing_for_deterministic_public_maps`.

### 1.3 Class preservation

**Assumptions.**
- Each token's prototype is smooth(mean) of same-predicted-class teacher vectors, with
  smooth(m) = (m + eps 1 + eps e_c)/(1 + (K+1) eps) and eps = 1e-12, in float64.
- Deployment routes a row only among the cells of its own predicted class.

**Statement.**
- Every member has p_c >= p_k (first-index ties), and floating-point summation and division are monotone. So the
  computed mean has m_c >= m_k, and the eps e_c term makes the argmax strict.
- The released decision therefore equals the teacher decision for every row, including rows the fit never saw.
- A class absent from the fitting rows decodes to smooth(uniform, c), whose argmax is c.
- Accuracy, confusion matrices and recalls equal the teacher's, so teacher weak classes (class-5 recall 0) cannot be
  repaired. Log loss and Brier are not preserved and are measured.
- Checked by `test_smoothed_mean_of_same_class_vectors_keeps_the_class` and
  `test_release_class_preservation_ties_underflow_and_absent_class`.

### 1.4 KL k-means monotonicity (prior work: Banerjee et al. 2005)

**Statement.**
- KL(p || q) is a Bregman divergence in q's second-argument position, so the member mean minimises the summed
  divergence.
- With unsmoothed centroids, neither the assignment step nor the update step increases the objective, and the
  iteration reaches an assignment fixed point in finitely many steps.

**Limitation.**
- The implementation decodes smooth(mean), which is not the exact minimiser. Monotonicity therefore holds only up to
  O(eps), and the registered cap plus the best-coherent-iterate rule handle the residual.
- The minimal achievable distortion is non-increasing in the cap. A Lloyd local optimum from a given start need not
  be, so the capacity curve is an empirical description, not a theorem.

### 1.5 Fitted teacher KL versus true-label confidence (standard decomposition)

Let pi(x) be the true conditional class law, p the teacher, and q the decoded prototype. Then:

- **Log loss:** E[log-loss(q) - log-loss(p)] = E[KL(p || q)] + E[sum_y (pi_y - p_y)(log p_y - log q_y)].
- **Brier:** E[Brier(q) - Brier(p)] = E||q - p||^2 + 2 E<q - p, p - pi>.

**Consequences.**
- For a calibrated teacher the second terms vanish, and the fitted D is the expected confidence cost.
- Miscalibration makes the relation inexact, in either direction.
- The inner gate therefore uses true labels on INNER_SELECTION. A low fitted D is not a confidence certificate, and
  passing the inner gate is not a fit-distortion certificate.

### 1.6 Scope notes required by the prompt

**Fitted plug-in MI at larger alphabets is biased and is not protection.**
- Under a null (S independent of the code), plug-in MI is positive and grows roughly like (occupied cells - 1)/(2N).
- On 15,434 rows that is about 0.07 nats at 2,048 occupied cells (`test_plugin_mi_bias_grows_with_alphabet_under_null`).
- Stage B fine partitions (up to 32 x 128 per class pair) can occupy pair alphabets where this bias is comparable to
  the lambda-weighted differences being optimised.
- So lambda I12 partly penalises alphabet size. A fitted I decrease is not protection, not a population bound and
  not a reliable ranking by itself. Held-out attacks govern every claim, and the permutation-null MI at the same
  alphabet (prompt section 9) must be reported beside every fitted MI.

**Unfinished clustering is an optimisation limitation, not proof that finishing it fixes confidence.**
- Most source fine-clustering runs stopped at the 20-round cap.
- A1 isolates optimisation time at a fixed rate. Passes 1-20 of the 200-round run are bitwise the source passes,
  tested.
- Whatever A1 shows (in the prediction file the lead put P(QP1) = 0.07), a lower fitted KL from more rounds says
  nothing by itself about the inner true-label gate (section 1.5).
- The old 20-round code is not incorrect for stopping at its registered cap.

## 2. Stage A review (review 1)

**Reviewed working-tree files** (untracked or uncommitted; sha256 prefixes):

| File | sha256 prefix |
|---|---|
| `qpc/kmeans.py` | f126e6b38706 |
| `qpc/stagea.py` | 2e0ca610d3d0 |
| `qpc/release.py` | 8131d085ff04 |
| `qpc/gate.py` | fa789d6ce8a0 |
| `qpc/utility.py` | afa671c41eda |
| `qpc/run.py` (Stage A and gate stages only) | 5cd5ba2f2ed6 (found SA-R1, SA-R2); re-reviewed at a64bb028ae5b (resolved) |
| `PROTOCOL.md` (sections 5 and 7) | fbddcad82a83 |

The registered k-means rules were first reviewed from the `qpc/kmeans.py` docstring. `METHOD_CARD.md` sections 1-5
and 7, written by role B at about 04:15Z, were then read and agree with the code. That covers the starts, the generator
`default_rng([seed, K, c])`, multiplicity-weighted KL D^2 draws, RTOL 1e-9 over 3 passes followed by a final update
and assignment, the best coherent iterate, the 1e-12 relative start tie, and the initialisation-only KL clip.

### 2.1 What was checked, and how

**KL k-means++ definition and determinism.** Tests: `test_kpp_definition_matches_reference_and_is_deterministic`,
`test_kpp_sampling_law_is_divergence_weighted`, `test_kpp_roundoff_guard_and_degenerate_draws`.
- The chosen distinct vectors equal an independent transcription of the registered rule.
- Repeated calls are identical, and row permutation does not change the starts (distinct vectors with
  multiplicities).
- The two registered seeds give different starts.
- No vector is chosen twice.
- Over 6,000 seeds, the joint law of the first two centres matches the definition. That law is
  P(first = i) = w_i/W and P(second = j | i) proportional to w_j KL(u_j || smooth(u_i)). The chi-square is under its
  threshold, and zero-weight vectors are never drawn.
- The roundoff guard clips [-1e-12, 0) in initialisation weights only and refuses a genuine negative divergence. The
  optimised objective never clips.

**Convergence rule and the best fully coherent iterate.** Test:
`test_qpc_iteration_matches_independent_reference_and_returns_best_coherent_iterate`, for each of the three starts on
three fixtures.
- The fixtures are a binary continuum with slow Lloyd convergence, six-class rows with exact ties and zeros, and
  six-class rows with an absent class.
- An independent transcription of the rule returns the same assignment, pass, stop reason and round count. The rule
  is: assignment fixed point; or three successive relative changes below 1e-9 followed by one final update and
  assignment; or the 200 cap followed by a final update and assignment.
- The returned objective is the trajectory minimum, with later passes winning exact ties.
- Deploying the returned centroids reproduces the returned cells, and the objective equals the row-level KL to the
  decoded prototypes.
- At a returned fixed point the routing centroid equals the decoded prototype bitwise.

**Training statistics equal the deployed assignment.** Test:
`test_training_statistics_equal_deployed_policy_and_release_rows`.
- Independent nearest-centroid deployment on the fitting rows reproduces the stored n and S.
- Release prototypes are smooth(S/n) by the registered formula.
- The row-level mean KL of the released vectors equals the receipt `mean_kl_fit`.
- `kmeans._check_deployment` already enforces this inside every fit.

**Start selection.** Test: `test_start_selection_is_per_class_lowest_fitting_kl_with_fixed_ties`.
- The per-class winner is the lowest class-total fitting KL, recomputed from each start's deployed partition, with
  ties within 1e-12 relative going to the earlier start.
- The recipient total equals the sum of the winners.
- A non-argmax label array is refused, so no label can key the fit.

**Empty-cell handling.** Test: `test_empty_cells_keep_centroid_are_reported_and_removed`.
- Fixtures with empty-cell events reproduce the reference, in which an empty cell keeps its previous centroid.
- Cells that are empty in the returned iterate are removed, and no deployed cell lacks fitting rows.

**Class preservation and smoothing.** Tests: `test_smoothed_mean_of_same_class_vectors_keeps_the_class`,
`test_release_class_preservation_ties_underflow_and_absent_class`.
- Covers ties, exact 0/1 vectors, 1e-300 entries, K-way ties and rows of an absent class (fallback token decoding to
  smooth(uniform, c)).
- A wrong decision array is refused.

**Larger capacity creates new cells.** Test: `test_larger_capacity_creates_new_cells_and_never_pretends`.
- From m2 = 8 to 16, 32 and 64, every rich class gains actual cells (at least 48 of 64 realised), and fitting KL
  falls strictly.
- A class with only 10 distinct vectors yields exactly 10 cells at m2 = 64.
- The absent class keeps one fallback token, and the alphabet equals the sum of actual tokens.

**A1 reproduction semantics.** Tests: `test_a1_source_rule_reproduces_dpc_fit_fine_bitwise`,
`test_a1_source_kmeans_matches_independent_reference`,
`test_a1_unit_parity_with_admitted_source_release_and_alias_of_a2_source_start`.
- Rule "dpc" equals `dpc.partition.fit_fine(max_cells=m, rounds=20)` bitwise, and also equals an independent
  transcription.
- The 200-round run shares passes 1-20 bitwise and is never worse.
- `a1_unit` reproduces a dpc-built admitted release with exact token IDs and bitwise decoded vectors on all rows.
- A1's 200-round partition is A2's source-start partition at (8, 8), with the same fingerprint, so it is an alias and
  not an extra nominal fit.
- A 1e-9 perturbation of the admitted release is flagged as an engineering blocker.

**Release format and per-recipient caps.** Test: `test_asymmetric_caps_save_restore_and_no_hidden_ids`.
- Caps are enforced separately for m1 and m2, and swapped recipients are refused.
- A JSON round trip gives bitwise identical assignment and decoded vectors.
- dpc records are refused as qpc releases.
- The release arrays are exactly row_id, tok, q, hard and alpha.

**Gate and rate selection.** Tests: `test_gate_rate_selection_matches_the_registered_rule` (400 random banks against
a transcription of prompt A4) and `test_gate_eligibility_needs_every_seed_and_every_gate`.
- Eligibility needs every seed, every gate and decision preservation.
- A missing seed fails.
- Headroom needs all seeds.

### 2.2 Findings

#### REQUIRED

**SA-R1 (`qpc/run.py`, `stage_stagea`, the A1 parity check, around line 235). The runner reads a key that the stage
never writes.**
- The runner tests `r.get("source_parity", {}).get("ok", False)`.
- `qpc.stagea.a1_unit` writes the verdict under `parity_with_admitted_release`.
- On a synthetic fixture with exact-ID and bitwise parity, the runner would raise
  `SystemExit("A1 REPRODUCTION MISMATCH ...")` for every seed. This is loud, not silent, but it blocks Stage A as
  locked.
- Failing fixture: `test_runner_reads_the_keys_that_stagea_writes`.
- Patch, either:
  - in `run.py`: `par = r.get("parity_with_admitted_release", {})`, then `if not par.get("ok", False): raise
    SystemExit(... par)`; or
  - in `stagea.a1_unit`: also set `rec["source_parity"] = parity`.

**SA-R2 (`qpc/run.py`, `stage_gate`, the A1 rows, around lines 288-298). CONVERGENCE_DIAGNOSTIC.csv would silently
lose the fit objective, convergence and work.**
- The runner reads `a1.get("versions", {}).get(ver, {}).get(f"r{i}", {})` and the fields `objective`,
  `rounds_max`, `converged_classes`, `classes` and `work_rounds`.
- None of these exists. `a1_unit` writes `a1[ver]["per_recipient"][str(i)]` with `objective_total`, `mean_kl_fit`,
  `all_converged`, `work` and `per_class` (each holding its start's `rounds_used`, `converged` and `stop_reason`).
- Every one of those CSV columns would be None. Prompt A1 requires "fit objective, held-out confidence, convergence
  and elapsed work".
- Failing fixture: `test_runner_reads_the_keys_that_stagea_writes` (second assertion).
- Patch:

      fit = a1[ver]["per_recipient"][str(i)]
      pcs = [pc for pc in fit["per_class"] if not pc["fallback"]]
      st = [pc["starts"][0] for pc in pcs]
      row = {..., "fit_mean_kl": fit["mean_kl_fit"], "fit_objective_total": fit["objective_total"],
             "rounds_cap": a1["rounds"][ver], "classes": len(pcs),
             "converged_classes": sum(s["converged"] for s in st),
             "fixed_point_classes": sum(s["stop_reason"] == "assignment_fixed_point" for s in st),
             "cap_classes": sum(s["stop_reason"] == "cap" for s in st),
             "max_rounds_used": max(s["rounds_used"] for s in st),
             "work_assign_passes": fit["work"]["assign_passes"], ...}

**Resolution (verified 04:21Z).** `qpc/run.py` a64bb028ae5b fixes both defects.
- It reads `r["parity_with_admitted_release"]["ok"]` together with `ENGINEERING_BLOCKER`, and preserves a failed A1
  unit as `__FAILED` before stopping.
- It builds the A1 CSV rows from `a1[ver]["per_recipient"][i]`: fit objective total, mean KL, convergence, classes
  converged, rounds used (maximum and sum) and stop reasons of the winning start.
- On a JSON round-tripped synthetic A1 record the extraction yields real values. `test_runner_reads_the_keys_that_stagea_writes`
  now passes.

#### RECOMMENDED

**SA-C1 (`qpc/run.py`, around line 255). Pass the fitting rows to `pair_unit`.**
- Call `SA.pair_unit(pols[(1, m1)], pols[(2, m2)], T, bind_meta(k, cid, D), fit_rows(D))`.
- Without `tr`, `pair_record` skips its fitting-row token-count check, and the receipts lack the emitted states,
  emitted entropy and singleton counts on fitting rows.
- Prompt section 14 asks for "token cap versus actual emitted states and entropy".
- The per-recipient invariant is still enforced inside `kmeans._check_deployment`, so this is a reporting gap, not a
  correctness defect.
- **Applied** in `qpc/run.py` a64bb028ae5b (`tr=tr`).

**SA-C2 (`PROTOCOL.md` section 7). Register the numeric convergence constants in the locked text.**
- The constants are RTOL = 1e-9 on the relative change of the class-total fitted KL, PATIENCE = 3, cap 200, a
  start-selection tie of 1e-12 relative, and the initialisation-only KL clip guard of 1e-12.
- Either state them in section 7, or name `qpc/kmeans.py` (hash-locked) as the registration. `METHOD_CARD.md`, which
  section 7 cites, was not present at review time. Check before STAGE_A_LOCK.

**SA-C3 (A1 parity strictness). Surface any non-exact A1 parity as a deviation.**
- `release.token_parity.ok` accepts a token bijection with |dq| <= 1e-15.
- A1 src20 uses dpc's own arithmetic, so the expected and observed result on synthetic data is exact
  (`ids_rule = "exact_ids"`, `q_rule = "bitwise"`).
- Surface both fields in CONVERGENCE_DIAGNOSTIC, and treat anything other than exact/bitwise as a disclosed
  deviation, even though `ok` is True.

**SA-C4 (convergence labelling). Do not let "converged" read as "fixed point".**
- `converged = True` is recorded for stop_reason "relative_tolerance" as well as for an assignment fixed point.
- Report the three stop reasons (fixed point, tolerance, cap) separately in CONVERGENCE_DIAGNOSTIC, FIT_MANIFEST and
  the reports.
- Partly applied: the A1 CSV now carries `stop_reasons`. The A2 receipts also hold `stop_reason` per class and start.

#### NOTES

- **SA-N1 (income tie-break).** The prompt's "then lower income log loss" is implemented, and written in PROTOCOL
  section 7, as worst-seed income log loss. That is consistent and locked.
- **SA-N2 (best-shortfall ordering).** The ordering uses only the normalised confidence excess. A
  decision-preservation failure cannot reach the gate on U DIRECT-TASK codes, because `encode` raises first.
- **SA-N3 (tolerance mismatch, inherited from dpc).** Input rows are accepted with |sum - 1| <= 1e-9, but smoothing
  checks prototype sums at 1e-12, and `guarded_kl` refuses KL < -1e-12. A fitting row with a sum error above about
  1e-12 therefore raises a ValueError: loud, not silent. Admitted teachers have |sum - 1| <= 6.7e-16 (source N7), so
  this cannot occur on admitted data.
- **SA-N4 (empty cells).** An empty cell keeps its previous centroid (dpc rule). A stale centroid may stay empty and
  is removed at the end, so actual states can fall below the cap. This is reported per pass
  (`empty_cells_per_pass`, `removed_empty_cells`). It is a registered capacity limitation; there is no re-seeding.
- **SA-N5 (routing centroid versus decoded prototype).** At a cap or tolerance stop, or when an earlier pass is the
  best iterate, the routing centroid can differ from the decoded prototype. The policy is still coherent: routing
  gives cells, cells give statistics, statistics give prototypes, and this is tested. Report
  `max_abs_unsmoothed_vs_smoothed` and the mean-versus-centroid gap as in the source.
- **SA-N6 (A1 r200 stopping before pass 21).** If the tolerance rule fired before pass 21, the 200-round code would
  have run fewer passes than the 20-round code. This was not observed on the fixtures, and the best-iterate rule never
  returns a worse objective than the shared prefix. Report `rounds_used` against 20.
- **SA-N7 (determinism pin).** Determinism depends on numpy's PCG64, SeedSequence([seed, K, c]) and np.unique
  ordering, so the numpy version (2.4.2 here) belongs to the pin.
- **SA-N8 (no approximation factor).** The k-means++ starts carry no approximation guarantee here; Bregman seeding
  bounds need conditions that fail near the simplex boundary.
- **SA-N9 (capacity curve).** A non-monotone capacity-curve point, from a different local optimum, must be reported
  as observed, not smoothed.
- **SA-N10 (the gate decides).** Fitted KL is a proxy for confidence cost (section 1.5), so the true-label inner gate
  is the decision, as registered.
- **SA-N11 (seed identity).** `gate.summarize` checks that the number of seeds is 3, not that they are seeds 0-2. The
  runner always passes 0-2.

### 2.3 Test inventory after review 1

25 tests in `qpc/tests/test_math_review.py`, all passing (about 3 s). `test_runner_reads_the_keys_that_stagea_writes`,
the SA-R1/SA-R2 fixture, failed against `run.py` 5cd5ba2f2ed6 and passes against a64bb028ae5b.

## 3. Stage B review (review 2; 2026-10-06 04:41Z)

**Reviewed working-tree files:**

| File | sha256 prefix |
|---|---|
| `qpc/partition.py` | a4396381979f |
| `qpc/compress.py` | 3495e2a88171 |
| `METHOD_CARD.md` (section 6) | 4bbc2bbff013 |
| `PROTOCOL.md` (section 8) | 89a10e18b98e |

Mutation testing was moved after the locks at the lead's request (section 5).

### 3.1 What was checked, and how

The reference is an independent transcription of the search on label vectors. It uses its own fine-level objective,
and a separate row-level brute force computed from the released tokens and decoded vectors.

**Objectives D, I1, I2 and I12** (`test_stageb_families_equal_independent_reference`).
- Every receipt term equals the row-level brute force within 1e-12 on 8 fixture settings.
- The settings use asymmetric caps (2,2), (2,3), (1,2), (2,1) and (3,2), lambda from 0.01 to 10, and the small, XOR,
  null and six-class fixtures.

**Exact merge and move deltas** (`test_merge_and_move_deltas_and_tables_against_brute_force`).
- At random intermediate states, including virtual labels left by moves, every merge and every single-cell move
  increment of `State` equals the brute-force difference of row-level objectives within 1e-12.
- That covers dD, dI_own, dI12 and the weighted total.
- After every `apply_merge` and `apply_move`, `terms()` and the pair table equal recomputation.

**Search rules.**
- The canonical maps of FINE-TASK, LOCAL, SEQ-12, SEQ-21 and JOINT are identical to the reference on all 8 settings.
- The reference rules are:
  - greedy to the per-recipient caps, smallest increment first, ties within 1e-12 going lexicographically;
  - then objective-improving extra merges, below -1e-12;
  - then up to 5 sweeps, each an exchange pass (best target, ties to the lowest label, no emptied cell) followed by an
    improving-merge pass.

**At most the cap, with objective-improving merges** (`test_at_most_cap_extra_merges_and_local_optimality`).
- No class exceeds its cap.
- At every converged JOINT, no single merge or move improves F_joint by more than TOL.
- A strong-lambda LOCAL fixture ends below its caps through extra merges.

**The sequential correction** (`test_sequential_correction_stage_one_is_F_joint_with_class_only_counterpart`).
- SEQ-12 and SEQ-21 stage one is F_joint with the other recipient's CLASS-ONLY release. The receipt's stage-one
  F_joint and I(S; C_a, d_b) equal row recomputation.
- The corrected stage-one map is never worse on that F_joint than the old D + 1.5 lambda I map.
- On a designed fixture (recipient 2's decision carries S), the correction changes the stage-one map at lambda = 3
  and lowers F_joint by 0.00079.
- The first map is frozen, and stage two equals the reference.

**JOINT starts and witness dominance** (`test_joint_candidates_and_witness_dominance`).
- There are 9 candidates: 5 refined starts and 4 unchanged witnesses.
- The final F_joint, recomputed from rows, is at most every unchanged witness and every refined start.
- Passed-in witnesses give the same pair as recomputed ones.
- `unresolved_local_optima` is reported.

**Fine partitions** (`test_fine_unit_caps_starts_rows_and_deployment`).
- `fine_unit` fits only the fitting rows, with the Stage A starts and rule; its fingerprint equals
  `kmeans.fit_recipient` at caps 32 and 128.
- Caps are realised on rich classes.
- The private all-row assignment equals deployment, and its fitting-row counts equal the stored statistics.
- Support receipts match the stored cells.

**METHOD_CARD section 6 against the code.** Sections 6.1-6.7 describe the code as implemented: objective, n log n
table, cache invalidation, extra merges, sweep definition, families, correction receipts, permutation-null receipts
and the runner unit. Two wording points are given below (SB-N1 and SB-N2).

### 3.2 Optimiser gaps on tiny exhaustive fixtures (NOT an Adult certificate)

**Method.**
- Every class-preserving map with at most m cells per class is enumerated on both recipients, up to 64 x 64 map
  pairs.
- Gap = family value minus the global minimum of the family's OWN objective over that space.
- SEQ arms are measured against F_joint, with their stage-one gap (corrected objective) and conditional stage-two gap
  listed separately.
- Runner: `python -m qpc.tests.test_math_review --exhaustive --out <json>`, about 2 s.

**Results.**

| Fixture | m1, m2 | lambda | FINE-TASK | LOCAL | SEQ-12 F_joint gap (stage 1 / stage 2) | SEQ-21 F_joint gap (stage 1 / stage 2) | JOINT |
|---|---|---|---|---|---|---|---|
| small | 2, 2 | 0.1 | 0 | 0 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| small | 2, 2 | 1 | 0 | 0 | 0.0324 (0 / 0) | 0.0025 (0 / 0) | **0.0025** |
| small | 2, 2 | 10 | 0 | 0 | 0.0145 (0 / 0) | 0 (0 / 0) | 0 |
| small | 1, 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| small | 2, 1 | 10 | 0 | 0 | 0 | 0 | 0 |
| xor | 2, 2 | 0.1 | 0 | 0 | 0.00007 (0 / 0) | 0 | 0 |
| xor | 2, 2 | 1 | 0 | 0 | 0.0058 (0 / 0.0058) | 0.0092 (0 / 0.0061) | **0.0058** |
| xor | 2, 2 | 10 | 0 | 0 | 0.0058 (0 / 0.0058) | 0.0092 (0 / 0.0061) | **0.0058** |
| null | 2, 2 | 1 | 0 | 0 | 0 | 0.00009 (0 / 0) | 0 |
| null | 2, 2 | 10 | 0 | 0 | 0 | 0 | 0 |
| six | 2, 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| six | 3, 1 | 10 | 0 | 0 | 0 | 0 | 0 |

**Reading the table.**
- FINE-TASK and LOCAL attain their own global optimum in 12 of 12 settings.
- The SEQ stage-one gap is 0 everywhere. Their F_joint gaps are structural to sequential design: the first map is
  fitted before the second exists.
- JOINT is globally best in 10 of 12 settings. Its gaps are 0.0025 (small, lambda 1) and 0.0058 (XOR, lambda 1 and
  10).

**XOR coordinated move** (`test_xor_fixture_requires_coordinated_moves`).
- S = A xor B, with A and B confidence sub-level clues on the two recipients; each alone is independent of S.
- JOINT removes the coalition leak by cross-mixing recipient 2 into {0,3}/{1,2}.
- The global optimum is the cheaper pairing {0,2}/{1,3}, which differs by SWAPPING two fine cells of one class.
- The JOINT solution is a strict local optimum for every single-cell move and every merge, so reaching the optimum
  needs a coordinated two-cell move that the registered search does not make.
- The gap is reported in `unresolved_local_optima`. It is the registered optimiser's limitation, not a defect.

### 3.3 Findings

**REQUIRED: none.**

**RECOMMENDED: none blocking.** The two wording points below are NOTES.

**NOTES.**
- **SB-N1 (METHOD_CARD 6.3, step 2).** "For F_task, merges never decrease distortion" holds exactly only for
  unsmoothed barycentres.
  - With eps-smoothing, the first-order eps terms cancel between cells with the same support.
  - When a cell mean has exact zeros, a merge can lower the smoothed distortion by at most about n eps / N.
  - The TOL = 1e-12 guard makes this immaterial except on pathological fixtures. Recommend "up to O(eps)".
- **SB-N2 (labels after moves).** A move can carry a coarse cell's label fine index out of the cell, so internal
  labels become virtual.
  - Merge ties in the merge pass that ends a sweep are then ordered by internal labels, not by lowest member.
  - This is deterministic and transcribed identically in the reference. Exported groupings are canonical.
  - METHOD_CARD's "label = a member fine index" holds at creation only.
- **SB-N3 (scope of dominance).** JOINT dominance holds only within the fine-state family and on the fitting
  objective. It says nothing about held-out recovery, and DIRECT-TASK is not contained (PROTOCOL section 8).
- **SB-N4 (stage-one objective).** The corrected SEQ stage one is
  D_a + lambda(I_a/2 + I(S; C_a | d_b)) + constant. It can keep a clue that the old surrogate removes when that clue
  is redundant with d_b, or the reverse.
- **SB-N5 (fitted MI at large alphabets).** At the selected Stage B rate (income 8, occupation 64, over 32 and 128
  fine cells per class), pair alphabets can reach hundreds of occupied cells.
  - The permutation-null receipts (100 fixed permutations, seed 20261006) must be read beside every fitted MI.
  - lambda I12 partly penalises alphabet size (section 1.6).
- **SB-N6 (tolerances).** `fit_policy_pair` asserts receipt-versus-row agreement at 1e-9 (observed about 1e-15).
  `_optimise` asserts that refinement never raises the stage objective.

## 4. Selection, validity and inference review (review 3)

### 4.1 Preliminary pass on the lead's files (2026-10-06 04:30Z, before AUDIT_AND_SELECTION_LOCK)

**Reviewed working-tree files:**

| File | sha256 prefix |
|---|---|
| `qpc/select.py` | 281b300aee49 |
| `qpc/family.py` | f38b06e71319 |
| `qpc/infer.py` | 6b42a9bf2023 |
| `LABEL_TRUTH_TABLE.json` | bc390107a989 |
| `PROTOCOL.md` (section 11) | 89a10e18b98e |

`qpc/audit.py` (role D) is reviewed in section 4.3, once ready.

**Checked.**

- `test_selection_matches_independent_protocol_transcription`.
  - `qpc.select.select_all` was run on 60 random synthetic inner-record banks. `qpc.run`'s record access was
    monkeypatched, and outputs went to a temp dir.
  - Each bank has 8 Stage A rates, 2 Stage B rates x (FINE-TASK + 4 families x 3 lambdas), CLASS, both sources
    and three references.
  - Q*, C_global, T*, C_rate(J*), J* and P* equal an independent transcription of PROTOCOL section 11.
    Eligibility in the transcription is recomputed from raw metrics, not through `qpc.utility`.
  - The deterministic DESCRIPTIVE_ONLY fallbacks of J* and P* also match.
  - At least 5 banks with a J* nominee and at least 5 with a P* nominee were exercised.
- `test_selection_guards_are_per_seed_per_recipient_and_never_dropped`.
  - A JOINT code exactly within + 0.005 of its guards is nominated.
  - The same code at + 0.006 on one recipient of one seed is not nominated, and gets a descriptive fallback.
- `test_claim_status_and_label_match_protocol_text_exhaustively`. Every combination of claim inputs and of
  overall-label inputs agrees with a transcription of the section 11 status table, precedence and label list.
- `test_family_37_fixed_slots_and_z`.
  - There are 37 fixed slots: P01-P37, split A 11, B 11, C 11, Q 4.
  - z = NormalDist().inv_cdf(1 - 0.05/74) = 3.2048452050105634 exactly.
  - Every clause kind, target and strict side is as registered.
- **Read** (`infer.py`).
  - Strict `lower >` and `upper <` decisions.
  - SE with ddof 1, and a nonfinite replicate makes a primary slot INVALID; nothing is dropped.
  - DESCRIPTIVE_ONLY whenever a role is not NOMINEE.
  - The retain clause uses jcv `lin` = X0 - 0.8 X1 - 0.2 X2 on [acc(N), acc(U), acc(const)].
  - The bootstrap is `UnitBootstrap`: a multinomial over exact-record groups, seed 20261007, with one sequential
    draw stream shared by every statistic.
  - Arms are refused if row IDs, groups, SEX, labels or the constant differ across them.

**Findings.**

**SEL-R1 (REQUIRED; PROTOCOL section 11 and `LABEL_TRUTH_TABLE.json` against `qpc/select.py`). The guard-blocked
nominee rule is executable but not registered.**

What the code does:
- In `select.py`, an ELIGIBLE JOINT code whose guard comparator is not a NOMINEE makes J* INVALID_NOMINEE ("eligible
  JOINT blocked only by a missing guard comparator (guards are never dropped)"). The guard comparator is C_rate at
  its rate or C_global; for example, C_rate may have computed candidates but none eligible.
- Claim A's comparator then becomes INVALID_COMPARATOR ("no J* cell").
- P* is INVALID_NOMINEE whenever T* is not a NOMINEE.

What the registered text says:
- `LABEL_TRUTH_TABLE.json` defines nominee TECHNICAL_FAILURE as "candidate set not computable".
- It maps a comparator with "candidates computed, none eligible" to NO_ELIGIBLE, that is
  NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR.
- PROTOCOL section 11 states neither the guard-blocked rule nor the "no J* cell" consequence.

Why it matters:
- In this case the per-claim status from the executable (A: INVALID_COMPARATOR, B: INVALID_NOMINEE) differs from
  what the registered text gives or leaves ambiguous. A text reading would give A NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR,
  and B NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE or INVALID.
- The overall label is the same (INCOMPLETE_OR_INVALID) under both readings.
- This is exactly the dpc NO_FEASIBLE_CONTROL ambiguity the prompt forbids, and `LABEL_TRUTH_TABLE.json` itself says
  "Any disagreement found later is a defect".
- It is reachable only if a guard comparator fails, which the truth-table notes expect never to happen, because the
  DIRECT-TASK code at each Stage B rate is eligible and U is eligible.

Failing fixture: `test_guard_blocked_nominee_status_is_registered_in_protocol_and_truth_table`.

Patch: register the executable rule, which is the smallest change.
- In PROTOCOL section 11 and `LABEL_TRUTH_TABLE.json` (`claim_inputs.nominee.TECHNICAL_FAILURE`, plus a note), add:
  "TECHNICAL_FAILURE also covers an otherwise eligible nominee blocked only because one of its guard comparators
  (C_rate(J) or C_global for J*; T* for P*) is not a NOMINEE; guards are never dropped. Claim A's C_rate is resolved
  at the J* nominee cell, else at the J* descriptive fallback cell; when J* has neither, C_rate is INVALID_COMPARATOR
  (no J* cell)."
- Alternatively, change `select.py` so that claim A reads NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR when the blocking
  C_rate is NO_ELIGIBLE_COMPARATOR. The text must then say so.

**SEL-C1 (RECOMMENDED). Attribute claim A's failure to the nominee when it originates there.**
- When J* is INVALID_NOMINEE because a JOINT unit failed technically, claim A reads INVALID_COMPARATOR through the
  derived "no J* cell" C_rate, because comparator technical failure takes precedence. That holds even when every
  C_rate candidate is fine.
- Recommend recording that C_rate state as derived from J*, or reporting claim A as INVALID_NOMINEE in that case. The
  label is unaffected.

**SEL-C2 (RECOMMENDED; confirm deliberately). A valid PASS on claim C is suppressed when A or B lacks coverage.**
- `overall_label` returns INCOMPLETE_OR_INVALID whenever any claim lacks coverage. A valid PASS on claim C therefore
  gets no PRIVACY_COMPRESSION label if claim A or B is NOT_APPLICABLE or INVALID.
- PROTOCOL section 11 states this, so text and code agree. It is stricter than dpc's adopted R3 rule ("a valid
  PASSing claim C keeps its label").
- The prompt's "INCOMPLETE_OR_INVALID: required technical validity or comparator coverage missing" can be read either
  way, because claim C does not require A's or B's comparators.
- Confirm the choice before the lock. The per-claim table is published in either case.

**SEL-C3 (RECOMMENDED). Let the reference candidates follow the admission record.**
- `run.scored_ids()` always includes REF|E, REF|F and REF|F0. If a reference's provenance were invalid, its missing
  record would make C_global INVALID_COMPARATOR, and T* too through F0. The prompt says references are used "when
  their source provenance remains valid".
- All three are present and matching now (`SOURCE_ADMISSION.json`), so the point is moot unless admission fails.
  Recommend that `scored_ids` follow the admission record.

**SEL-N1.** T* includes SRC|RAW-J_b0.3, as PROTOCOL section 11 states explicitly. "Privacy-untrained" refers to the
release; the RAW-J teacher itself is privacy-trained. It is expected to be ineligible in any case (+0.013 occupation
nats in dpc).

**SEL-N2.** `infer.label_from` maps a Q* INVALID_NOMINEE (a technical failure among the Stage A rows) to
NOT_APPLICABLE_NO_Q. The missing-item text then reads "no eligible Stage A configuration despite a met gate", which
mislabels the reason. The label (INCOMPLETE_OR_INVALID) is correct.

**SEL-N3.** `pick` rounds the ordering keys to 12 decimals, so exact ties between seed means are resolved by the
next key, as registered.

### 4.2 Resolution of SEL-R1, SEL-R2 and SEL-C1-C3 (verified 04:36Z)

**SEL-R1. Fixed.**
- PROTOCOL section 11 has a new paragraph, "Guard-blocked nominees and the J* cell".
- `LABEL_TRUTH_TABLE.json` `claim_inputs.nominee.TECHNICAL_FAILURE` now covers guard-blocked nominees, and a note
  covers "no J* cell". The fixture passes.

**SEL-C2. Adopted.**
- A validly PASSing claim keeps its favourable label. Other claims' coverage gaps are listed in missing_items.
- With no favourable label, any coverage gap gives INCOMPLETE_OR_INVALID.
- The original rule is kept as `family.overall_label_stage_a_rule`, and inference reports the label under both
  rules (PROTOCOL section 11).

**SEL-R2 (REQUIRED, found while re-testing against the amended text). Fixed by the lead at about 04:40Z.**
- After SEL-C2, PROTOCOL section 11 label item 6 read "... or Q* INVALID". `overall_label` and the truth table also
  treat an unresolved Q* (NOT_APPLICABLE_NO_Q despite a met gate) as a coverage gap.
- The text therefore gave EXPERIMENTAL_NO_ADVANTAGE where the code gives INCOMPLETE_OR_INVALID.
- Item 6 now reads "or Q* INVALID or unresolved (NOT_APPLICABLE_NO_Q despite a met gate)".
- Fixture: `test_claim_status_and_label_match_protocol_text_exhaustively`. It reads item 6 from the text and passes.

**SEL-C1, SEL-C3 and SEL-N2.**
- SEL-C1 is documented in the truth table, which names the root cause in missing_items.
- SEL-C3 is moot, since every reference is admitted.
- The SEL-N2 reason text now reads "Q*: not resolved despite a met gate".

### 4.3 Attackers, baselines and assessment (role D; review 3 proper, 2026-10-06 04:47Z)

**Reviewed working-tree files:**

| File | sha256 prefix |
|---|---|
| `qpc/audit.py` | cac49164e495 |
| `qpc/baselines.py` | 9266e226768f |
| `qpc/assess.py` | bde690bb4c6d |
| `qpc/utility.py` | afa671c41eda |
| `qpc/data.py` (unseal gate) | e0195efce276 |
| `qpc/eval_lock.py` (lock fields read by `assess`) | read only |

Imported unchanged: `dpc/audit.py` f2c5a2699f4f and `smf/audit.py` 4b9ac46ddc9a.

**Checked.**

**Fixed orientation.**
- `dpc.audit.auc1` is the Mann-Whitney AUC of P(S = 1), equal to an independent implementation with ties counted as
  1/2.
- An anti-informative reader stays below 0.5 and is never flipped or clamped.
- `ce1` clips at 1e-12.
- Composed and slate selection take maxima of these unflipped values.
- Test: `test_attacker_auc_orientation_ce_clip_and_null_threshold`.

**Separate AUC and CE selection.**
- `inner_family` selects the AUC winner (highest INNER AUC) and the CE winner (lowest INNER CE) separately on the
  seed-0 bank.
- Only those two attackers are refit at seeds 0-2. Cell readers are deterministic.
- `recovery.auc` is the seed mean of the AUC-selected attacker, and `recovery.ce` that of the CE-selected one.
- D's tests cover this (`test_auc_and_ce_selection_are_separate`, `test_selected_attackers_refit_at_seeds_...`).

**Composed source closure before selection.** Test: `test_composed_source_bank_winner_rule_matches_transcription`.
- On 40 random synthetic banks, the per-family, per-view AUC and CE winners, the reported values and the freeze list
  of `composed_source_bank` equal an independent transcription.
- The banks include exact seed-0 ties, and policies whose seed-0 value beats the source while their seed mean does
  not.
- The transcription selects on seed-0 values with first-bank ties, reports the winner's seed mean, and composes the
  decisions family only with the class-only code.
- A record from another slate is refused.
- Closure refuses missing policy inner units and unexpected release units (D's `test_composed_bank_refuses_without_closure`).
- `select.py` reads the composed SRC|U record, which is written in the `inner_src` stage after every policy inner
  unit, so the comparator and guard values include composition before any selection.

**Unseen-token and unseen-pair fallbacks.**
- An unseen local token gets the AUDIT_FIT SEX prior.
- An unseen tuple gets the rule in {local_1, local_2, prior} with the lowest CE on the INNER rows whose tuple is
  unseen, with ties in that order. It is frozen before scoring.
- This is dpc's reader convention, reviewed with brute-force tests and mutants in the source review. D re-tests it
  (`test_unseen_local_token_uses_fit_prior_and_unseen_pair_rule_is_inner_selected`).
- Fallback use is reported per role by `coverage_receipt`.

**Control pass rules.**
- The null rule is AUC_B <= 0.5 + 3.5 sd0, with sd0 = sqrt((n0 + n1 + 1)/(12 n0 n1)) on held-out half B. This was
  verified independently.
- CONF_r1/r2, COLL_r1/r2 and ROT_r1/r2 must exceed 0.75. ROT must also exceed 0.75 on the pair, with bit-exact
  serialisation.
- XOR must have pair AUC > 0.75 with both local AUCs <= the null threshold.
- The 20% re-drawn noisy-S plant has perfect-reader AUC 0.9, verified.
- The plan is structural and fixed before any control runs. A failed plant is recorded and triggers technical
  review.

**Assessment gate and unseal path.**
- `qpc.data.load(unseal=True)` accepts only the caller `qpc.assess`, and only when EVALUATION_LOCK.json is
  byte-identical on origin.
- `assess.open_assessment` requires the following: a committed, unmodified, pushed lock; every scoring-chain file at
  its locked hash; and every loaded worktree module locked.
- `outer_unit` re-verifies the lock before every unit, and checks the unit file maps when recorded.
- `load_unsealed` checks the assessment role's rows, groups and row-ID hash against the lock. `eval_lock` always
  writes `assessment_role`.
- D's tests cover each refusal (uncommitted, unpushed, modified and misnamed locks; missing chain hashes; unsealing
  outside assess).

**Utility gates as used by select.**
- Every inner record carries `qpc.utility.release_inner_utility` of the release and of U of the same seed.
- `select.py` recomputes `gate_record` against the SRC|U record's utility.
- My selection transcription recomputes eligibility from raw metrics and agrees on 60 banks (section 4.1).

**Findings.**

**REQUIRED: none.**

**AU-C1 (RECOMMENDED). The A1 diagnostic codes sit outside the composed bank.**
- The prompt says "U continuous-source attack banks include composition with EVERY fitted code of this study".
- The composed bank and its closure cover the registered release bank (every `pol__s{k}__U_*` unit in `scored_ids`).
  They do not cover the two A1 diagnostic releases per seed (`a1__s{k}` src20 and r200), which were also fitted in
  this study.
- src20 is the source study's DIRECT-TASK m8 code. r200 is A2's source-start receipt at (8, 8) and is not a
  registered release.
- Either state this scope in PROTOCOL section 10 and the EVALUATION_LOCK description ("every fitted code of the
  registered release bank; A1 diagnostic releases excluded"), or add their inner records to the bank.
- The effect on a maximum over about 40 banks is expected to be negligible. The point is that the closure claim must
  say exactly what it covers.

**NOTES.**
- **AU-N1 (provenance of the reviewed attacker code).** `dpc/audit.py` (f2c5a2699f4f) is the source file after dpc
  AMENDMENT_A1, which changed only the control plant bits. The dpc review's attacker mutants were run on the
  pre-amendment file (e024235b4854). The qpc layer is covered by D's tests and the tests above.
- **AU-N2 (selection optimism).** The composed source AUC is a maximum over the source's own bank plus every code
  bank of the same seed, about 40 banks per family. It is selection-optimistic. That inflates SRC|U as C_global or T*
  and loosens the J*/P* guards, as registered and as in dpc.
- **AU-N3 (fine partitions are public too).** A deployed Stage B policy JSON carries its fine assignment partition,
  the routing centroids, so a holder of the continuous probabilities can compute fine-cell IDs. The composed bank
  adds readers of released codes only. The source's own slate on continuous p covers that information, but no
  categorical fine-ID reader is in the bank. This is a scope statement, not a defect.
- **AU-N4 (ROT plant).** ROT puts a 1e-6-amplitude clue in a Haar-rotated direction. Its detection depends on how the
  slate scales features. A failure is a technical-review trigger by design, never a leakage reading.
- **AU-N5 (null multiplicity).** About 30 view-level null tests at z = 3.5 give roughly a 1% chance of one false
  exceedance in total, all of which are recorded.
- **AU-N6 (unseen tokens at larger alphabets).** With occupation codes of up to 64 states per class, unseen-token
  and unseen-tuple fractions on INNER and assessment rows can be non-trivial. Coverage receipts must accompany the
  nominee AUCs (prompt section 9). Any unestimable metric blocks its dependent claim.

## 5. Prior art

See `PRIOR_ART_AND_BASELINE_GAPS.md`.
- PURIFIER's official repository was still empty on 2026-10-06 (GitHub API: size 0, pushed 2022-11-29; commits
  endpoint "Git Repository is empty").
- No Taylor, Vippathalla and Coon solver code was found.
- Clustering, output compression, privacy-funnel objectives, data processing and sequential collusion constraints
  are prior work. Our sequential arms are matched adaptations, not the Taylor solver. No novelty is claimed.
