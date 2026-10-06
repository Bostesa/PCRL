INTERIM STATUS (Stage A review, 2026-10-06 04:22Z): 0 OPEN REQUIRED. Two REQUIRED runner-binding defects (SA-R1,
SA-R2, both in `qpc/run.py`) were found and are already resolved in `qpc/run.py` a64bb028ae5b, re-verified by the
failing fixture, which now passes. No REQUIRED finding in the mathematics of `qpc/kmeans.py`, `qpc/stagea.py`,
`qpc/release.py` or `qpc/gate.py`. 4 RECOMMENDED (SA-C1 applied), 11 NOTES. 25/25 reviewer tests pass. Stage B and
selection reviews pending.

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

## 3. Stage B review (review 2)

Pending: `qpc/partition.py` and `qpc/compress.py`, with mutation testing.

## 4. Selection, validity and inference review (review 3)

Pending: `qpc/select.py`, `qpc/family.py`, `qpc/infer.py`, `LABEL_TRUTH_TABLE.json`, `qpc/audit.py` and
`qpc/utility.py`.

## 5. Prior art

See `PRIOR_ART_AND_BASELINE_GAPS.md`.
- PURIFIER's official repository was still empty on 2026-10-06 (GitHub API: size 0, pushed 2022-11-29; commits
  endpoint "Git Repository is empty").
- No Taylor, Vippathalla and Coon solver code was found.
- Clustering, output compression, privacy-funnel objectives, data processing and sequential collusion constraints
  are prior work. Our sequential arms are matched adaptations, not the Taylor solver. No novelty is claimed.
