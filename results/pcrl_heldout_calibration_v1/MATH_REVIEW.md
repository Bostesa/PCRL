# MATH_REVIEW — hcal (role B, mathematical reviewer)

Date: 2026-10-07. Worktree `research/pcrl-heldout-calibration-v1` at 53f477958, with role C/D/A files uncommitted. Reviewed against PROTOCOL.md sections 4, 6 and 7. Between my first and last reads the protocol changed only in the section 10 ownership table, so sections 4, 6 and 7 were unchanged.

Files reviewed: hcal/calib.py, hcal/select.py, hcal/ids.py, hcal/family.py, hcal/infer.py, hcal/bank.py, hcal/stages.py,
hcal/eval_lock.py, hcal/deploy.py (decoder reproduction only), lra/decoder.py, qpc/utility.py (gate), the
dpc/qpc check helpers, and tests test_calib.py, test_data_select.py, test_infer.py, test_bank.py.

Scope of what I ran:
- Synthetic data only. No real data, labels or teacher arrays were loaded.
- The test suites pass: calib, infer and select give 65/65, and the bank common-record, union, compose and view tests give 6/6 under `hcal.sema`.
- Reproducer scripts are in the session scratchpad, named `B_token32.py`, `B_temp.py`, `B_temp_one.py`, `B_argmax.py`, `B_repro.py` and `B_repro2.py`.

**Verdict: 0 BLOCKING, 3 SHOULD-FIX, 12 NOTE.** The core mathematics is correct. That covers the H-TOKEN32 objective identity, the temperature solver, the selection rules, the 23-slot family, the inference and the common bank. The SHOULD-FIX items are:
- one registration ambiguity, which can change the Ucal* winner;
- one identity-path robustness defect in U class temperatures;
- one place where the reported sub-criteria and the overall label can disagree.

---

## Findings

### 1. SHOULD-FIX — Ucal* and the "mean summed task NLL" tie key use the CLIPPED inner log loss, but the registered text says "NLL"

**Where.**
- hcal/select.py:14 (docstring), :78 (Ucal*), :139 (tie key) and :292 (SELECTION_RULES text).
- PROTOCOL.md:317 (section 6, Ucal*), together with section 6 "Ordering" ("lowest mean summed task NLL").

**Issue.**
- The code ranks by `inner[task]["logloss"]` from qpc.utility.task_metrics. That is `−log clip(p_y, 1e-12, 1)`, the clipped scoring log loss.
- Section 4 uses "NLL" for the UNCLIPPED calibration objective, and section 6 says only "NLL". The registered rule is therefore ambiguous.
- The ambiguity changes the result. The two definitions can pick different Ucal* families. That family is then locked and becomes the reference for the calibration-matched gate and for P12–P15.

**Reproducer (B_repro2.py).**
- Setup: a calibrated binary teacher on 2,235 inner rows, plus three saturated confident errors with p_y = 1e-30. Compare U identity with U global temperature at α = 1.1.

| Release | Clipped NLL | Unclipped NLL |
|---|---|---|
| identity | 0.556656 | 0.612289 |
| global α = 1.1 | 0.557212 | 0.560921 |

- Clipped, identity wins. Unclipped, global wins.

**Fix.**
- Before SCIENCE_LOCK, register explicitly in PROTOCOL section 6 and SELECTION_RULES.json what both Ucal* and the "mean summed task NLL" tie key use. Suggested wording: "the 1e-12-clipped natural-log inner log loss of qpc.utility.task_metrics (the same quantity as the LL gates)".
- Optionally record the unclipped inner NLL alongside as a receipt.
- No code change is needed if clipped is the intended rule. The code is self-consistent, and the gates also use the clipped LL.

### 2. SHOULD-FIX — U class temperature refuses identity rows that the input check admitted, which contradicts "α = 1 returns p exactly"

**Where.** hcal/calib.py:574–590 (`apply_u`, class branch) → `_check_u_out` (:552–558, |Σq − 1| ≤ 1e-12 on ALL rows).

**Issue.**
- The input P is admitted by `check_probs` with row sums within INPUT_SUM_TOL = 1e-9 (dpc/partition.py:40, :63).
- In the H-CLASS-TEMP path:
  - rows of classes with α_c = 1, which includes every class with fewer than 50 calibration representatives, are copied as P exactly;
  - those rows are then held to the released-table tolerance of 1e-12.
- So H-CLASS-TEMP raises CalibrationError whenever any identity-class row has |ΣP − 1| in (1e-12, 1e-9]. It raises even when every α_c = 1.
- The global identity path, `temp_apply_probs` with α = 1 (:568), does no such check. The two identity paths behave differently.
- `fit_u_temp` calls `apply_u` on the calibration rows, and `stage_calibrate` calls it on all rows, so both hit this path.
- Likelihood in this study is low. The teacher is `predict_proba` cast to float64 (dpc/deploy.py:139–144), so its sums are typically within a few ulps. However:
  - it is a latent refusal at the science stage, where a rerun would consume one of the two post-lock amendments;
  - it contradicts the registered identity rule.

**Reproducer.**
- Setup: `P = dirichlet(1,1,1)` with 400 rows; rows of class 2 are scaled by (1 + 5e-11), giving max |ΣP − 1| = 5.0e-11.
- `apply_u(P, H-CLASS-TEMP, alphas=[1.0, 1.0, 1.0])` raises "not normalised within 1e-12".
- `apply_u(P, H-GLOBAL-TEMP, alpha=1.0)` returns P.

**Fix (either option).**
- (a) Apply the 1e-12 normalisation check only to the rows actually transformed (α_c ≠ 1). Keep the argmax check on all rows.
- (b) Add an explicit, label-free engineering precondition, max over all teacher rows of |ΣP − 1| ≤ 1e-12, recorded in ENGINEERING_CHECKS.json. The study then cannot hit the refusal.

Option (a) matches the registered identity rule most directly.

### 3. SHOULD-FIX — Sub-criteria can read PASS while the label is INCOMPLETE_OR_INVALID; controls do not gate nomination

**Where.**
- hcal/family.py:131–137 (`criteria`) and hcal/infer.py:255.
- hcal/select.py:331–334 (`select_all`).
- PROTOCOL.md:295 ("Controls … must pass before nomination") and section 6 ("A missing comparator, artifact or failed control makes the dependent claim INCOMPLETE_OR_INVALID").

**Issue.**
- `criteria(outcomes, nominee_valid)` ignores technical validity and comparator state.
- `select_all` nominates P* and T* whatever the controls say. It only records `controls_all_ok`.
- The overall label is still correct, because eval_lock folds `controls_all_ok` and replay into `technical_validity.ok`, and `overall_label` then returns INCOMPLETE_OR_INVALID.
- But inference would still emit `OriginalCriterion = PASS` and `CalibrationMatchedCriterion = PASS` next to that label.
- `LABEL_TRUTH_TABLE` already shows this. Its "technical defect" row has label INCOMPLETE_OR_INVALID with `criteria = ('PASS', 'PASS')`, and so does "comparator technical failure".
- By section 7, OriginalCriterion is "PASS iff P01–P11 all PASS **with a valid nomination**". A nomination made with a failed required control is not valid.

**Fix.**
- Give `criteria` the technical validity and comparator state, and return INCOMPLETE_OR_INVALID whenever precedence level 1–2 applies.
- Also do one of the following:
  - mark the selection status when controls fail, for example `P*.status = "NOT_NOMINATED_CONTROLS_FAILED"`, mapped to TECHNICAL_FAILURE in eval_lock.state;
  - or document that nomination is provisional until `controls_all_ok`.
- Add truth-table rows "controls failed, all slots pass" and "comparator technical failure", asserting that the criteria are not PASS.

### 4. NOTE — The claim "U's recovery is never capped below an audited code's recovery" holds for the seed-0 selection value, not the reported mean

**Where.** PROTOCOL.md:263 and the hcal/bank.py docstring (U COMPOSITION); `_union`, bank.py:531.

**Issue.**
- U's bank is a superset of each audited partition's common bank. The legacy banks are inside the lra SRC|U composed bank, and every audited fresh bank is appended.
- So U's **seed-0** selected value is at least every code's seed-0 value.
- The **reported** value is the winner bank's attacker-seed 0–2 mean, and that can be lower.

**Reproducer (B_repro.py (i)).**

| Bank | Seed-0 value | Seeds 0–2 values | Mean |
|---|---|---|---|
| Code p's winner | 0.700 | 0.700, 0.700, 0.700 | 0.700 |
| Another partition's fresh bank | 0.701 | 0.701, 0.680, 0.680 | 0.687 |

- The union takes the second bank for U, so U reports 0.687, below code p's 0.700.

**Fix.**
- Reword to "U's bank contains every audited code bank, so U's seed-0 selection value is never below an audited code's".
- Optionally report both values.
- Selection is unaffected.

### 5. NOTE — T* can be continuous U

**Where.** PROTOCOL.md:329; select.py:242; test_data_select.py:221.

**Issue.**
- The T* pool includes U identity and its two calibrations, exactly as registered.
- As a result, a comparator almost always exists. U identity passes the U0 gate trivially and Ucal* passes the Ucal gate.
- If T* resolves to a U variant, P01–P03 compare P* with the continuous source, not with a task-only code. That weakens the "beats task-only" reading.
- The existing test already accepts either outcome.

**Fix.** Add a disclosure flag to selection.json and the report: "T* is continuous U: comparator is the source interface, not a task-only code".

### 6. NOTE — The alias disclosure compares release bytes, not partitions

**Where.** select.py:219–232 and :270–277.

**Issue.**
- `release_identity` hashes (tok, q, hard).
- Under the protocol's own threat model, a decoder change "changes no information". The faithful alias notion is therefore the token partition up to a bijective relabelling, together with the decisions, with q ignored.
- Two cases would be missed:
  - P* = (task-only partition, re-decoded);
  - a relabelled copy of a task-only partition.
- This is practically moot, because such an alias should fail the 0.02 pair-benefit gate up to attacker noise.

**Fix.** Test whether (tok_P*, tok_T) is a bijection on the permitted rows. Optional.

### 7. NOTE — The audit plan is triggered by MEAN-only eligibility

**Where.** select.py:153.

**Issue.**
- An lra partition whose only eligible non-diagnostic variant is MEAN is audited, even though MEAN is a control and never a nominee.
- This matches the literal protocol ("non-diagnostic variants") and only costs compute.

**Fix.** Optionally record the reason as `ELIGIBLE_CONTROL_ONLY`.

### 8. NOTE — The docstring claims rounding that the gates do not apply

**Where.** select.py:25 says "Means over seeds … compared after round(x, 12)".

**Issue.**
- The guard (:203–204) and the benefit gate (:205–207) compare unrounded floats.
- Only the ordering keys (`_order_key`) are rounded.
- Both approaches are defensible, but the registered text must match the code.

**Fix.** Align the docstring and SELECTION_RULES with the code: inclusive unrounded gates, rounded ordering keys.

### 9. NOTE — The `nll_ok` temperature certificate has about 5× headroom over rounding noise

**Where.** calib.py:109 and :434 (`NLL(α) ≤ NLL(1) + 1e-15·scale`).

**Test (B_temp_one.py).**
- 580 constructions whose exact optimum is α* = 1, built as L' = log softmax(α̂ L).
- K ∈ {2, 6}, T ≤ 300, N ≤ 2000.
- Result: 0 failures, but the worst margin was 0.79–0.89 tolerance units. Rounding noise reaches about 20% of the allowance.

**Assessment.** A failure would be a loud CalibrationError, never silent.

**Fix (optional).** Evaluate the decrease as `math.fsum` of per-row differences, or use 1e-14·scale. Keeping the registered value is acceptable.

### 10. NOTE — U near-ties at 1 ulp raise

**Where.** calib.py:561–571 (`temp_apply_probs`), which uses non-strict argmax == decision as registered.

**Issue.**
- In B_repro.py (iii), with α ∈ {0.25, 0.5, 2, 4}:
  - when the top two teacher probabilities differ by 1 ulp of 0.5 and the larger one has the larger index, the call raises;
  - a gap of 2 or more ulps is preserved.
- This is loud, already covered by `test_argmax_preservation_near_ties`, and has negligible probability for float64 `predict_proba`.

**Tokens are safe.** In B_argmax.py:
- 8,000 smoothed tables with exact ties in u (margin of ε only), tested at 404 values of α, gave 0 failures.
- The minimum relative margin at α = 0.25 is 5.0e-13, about 2,250 ulps.

### 11. NOTE — Precision of the registered z

**Where.** family.py:25–26.

**Values.**

| Source | z |
|---|---|
| Registered, `NormalDist().inv_cdf(1 − 0.05/46)` | 3.0653831516447343 |
| Exact quantile at the float argument (mpmath) | 3.06538315164473396 |
| Exact real quantile | 3.06538315164474749 |
| scipy `norm.isf(0.05/46)` | 3.0653831516447476 |

**Issue.**
- The 1.35e-14 gap comes from rounding 1 − p in float64. It is immaterial to the intervals.
- An independent recomputation by role E with scipy or mpmath would fail the 1e-15 assert in family.py:26.

**Fix.** Tell E to use the registered expression, or a tolerance of at least 1e-13.

### 12. NOTE — The supplementary table duplicates the primary diagnostics

**Where.** family.py:69–73.

**Issue.**
- `SUPPLEMENTARY_PARTITIONS` includes the primary diagnostic partition, JOINT λ0.1. D01–D08 are therefore re-reported at nominal 95%, giving two readings of one estimand.
- Two extra contrasts appear that section 7 does not name: `original_d0_vs_global` and `original_d1_vs_token`. They are descriptive only.

**Fix.** Drop the duplicates for JOINT λ0.1, or label them "same estimand as Dxx (primary, Bonferroni z)".

### 13. NOTE — decoder.json releases calibration label counts

**Where.** deploy.py:76–122.

**Issue.**
- The calibrated decoder record embeds `y_cal` and `n_cal`, the per-token CALIBRATION_HELDOUT task-label counts, so that the H-TOKEN32 table can be reproduced.
- `n_cal` can be 1, so for such a token the record reveals that person's label.
- This adds no row-level SEX information for an assessment row, and the threat model disclaims training-data guarantees.

**Fix.** State explicitly that the deployed decoder discloses calibration label counts, or keep `y_cal` private and bind the table by hash.

### 14. NOTE — For U, `nll_one` is not the NLL of the identity release

**Where.** calib.py:424–425 and :593–625.

**Issue.**
- For U, the solver's `nll_one` is the NLL of P' (P floored at 1e-12 and renormalised), not of the identity release P.
- The identity release is scored separately as `score_ll_one`.
- The calib docstring documents this.

**Fix.** Add one line to CALIBRATION_RULES so readers do not compare `nll_one` with the identity release's NLL.

### 15. NOTE — Pending items and scope limits

- hcal/assess.py appeared during this review and was read. It writes one `oatt__s{k}__<partition or SRC|U>` unit per attack key, shared by every decoder variant. The winner must equal the lock's, and every refit must reproduce its stored inner predictions bitwise. The stored P has columns [1 − p, p], consistent with infer's use of column 1. No issue found.
- COMPLETE_INTERFACE_EQUIVALENCE.json has not been written yet.
- hcal/controls.py is outside this review's scope.

---

## Verified (checked and found correct)

### 1a. H-TOKEN32 / T-TOKEN32 (calib.py:204–355; lra/decoder.py)

- **Objective identity.** Expanding the objective gives

  Σ_i[−log q_{Y_i} + ½‖q − e_{Y_i}‖²] + 32·KL(μ‖q)
  = −Σ_k (y_k + 32μ_k) log q_k + ½·n·‖q‖² − y·q + [½·n + 32·Σ μ log μ].

  This is exactly lra's f(u) with a = y + 32μ (A = Y + 32·PBAR, calib.py:250).
- **The quadratic coefficient is the calibration count.** `nf = n_cal` (calib.py:245, :336).
- **The dropped constant is restored in `obj_full`.**
  - Maximum relative gap between `obj_full` and an independent row-wise evaluation of the protocol objective: 2.9e-15, over 60 random tokens.
  - Those tokens covered K ∈ {2, 3, 6}, μ with exact zeros, n_cal from 1 to 39, and labels all off the decision class.
- **Optimality.** A multi-start SLSQP with class-dominance constraints was never better (0 of 60 cases).
- **n matters, and the code uses n_cal.** Substituting n = n_fit gives a different q in 54 cases.
- **KL direction is forward, KL(μ‖q).** The solver matches the forward-KL reference. A reverse-KL objective gives a visibly different q.
- **Bitwise identity.** The output equals `lra.decoder.solve_batch` when μ = S/n (400 tokens with K = 6, and also in the test suite).
- **The fixed prior is never replaced by a local mean.**
  - μ comes from the bank as `token_S / token_n` on OSF_DEFENSE_FIT (admit.py:229).
  - It is passed unchanged (calib.py:769, :336) and never recomputed from calibration rows.
  - `test_fit_token32_uses_the_fixed_prior_only` covers this.
- **Fallbacks.**
  - Reserved status is decided first, so a reserved token stays reserved even when calibration rows reach it.
  - Tokens with n_cal = 0 return q0 exactly, enforced by `array_equal` (calib.py:343).
- **Certificates.** Fitted tokens carry lra `cert_row` computed on the final released vector, and `summary_violations` runs unchanged.
- **Parameter count.** FITTED tokens × (K − 1).

### 1b. Temperatures (calib.py:358–541)

- **Formulas.**
  - NLL(α) = mean[LSE(αL) − αL_y] is convex.
  - Its derivative is g = mean[Σ_k p_k L_k − L_y].
  - Its curvature is mean Var_p(L).
  - The code matches all three, and a finite-difference test exists.
- **Bounded bisection.**
  - The decision order is: g(0.25) ≥ 0 → low boundary; otherwise g(4) ≤ 0 → high boundary; otherwise bisect.
  - The bisection keeps the invariant g(lo) ≤ 0 < g(hi) and stops when lo and hi are adjacent floats, after about 55 halvings out of the 100 allowed.
  - For a convex objective, the boundary KKT sign conditions are sufficient.
- **Stress test (400 random problems).**
  - Ranges: K ∈ {2, 6}, T up to 200, N up to 2,000, q0 entries down to about 1e-12.
  - 0 certification failures.
  - NLL at the solution was never more than 4.4e-16 above a bounded-Brent reference.
  - Interior |α − α_ref| ≤ 1.4e-7, which is Brent's own tolerance.
  - On steep problems the interior `grad_rel` was at most 5.8e-18, against a tolerance of 1e-9.
- **Identity path.** α == 1.0 returns q0 or P bitwise. A solver result of exactly 1.0 takes this path.
- **No calibrator optimises a clipped objective.**
  - Tokens use log q0, with q0 ≥ ε/Z.
  - U uses the registered input floor, and the solver objective is the unclipped NLL of the vector actually released when α ≠ 1.
- **Clipped scoring is recorded separately.** Recorded fields: `nll_alpha` and `nll_one` (unclipped), `score_ll_alpha` and `score_ll_one` (clipped), and the clipped counts at α and at 1.
- **Class order under positive α.**
  - Tokens are checked with strict argmax against the token class.
  - U is checked with argmax == decision on every row, and d must equal the teacher argmax.
- **Class fallback.** Fewer than 50 calibration representatives, including an absent class, gives α = 1.
- **Scope.** Each recipient (task) of each map and seed gets its own α. Temperature families transform every token, as registered.

### 1c. Continuous U

- The log-input rule is exactly as registered: max(p, 1e-12), renormalise, then softmax(α log p').
- α = 1 returns p exactly, apart from finding 2.
- The fitted temperature is applied to all rows (stages.py:91–92).
- Classes are the teacher decision d.

### 2. Selection (select.py, ids.py) against PROTOCOL section 6

- **Ucal\*.** The rule matches: mean of (s0 + s1 + s2)/3 of the per-seed summed inner LL; ties within 1e-12 go to identity, then global, then class; finite vectors and preserved decisions required; one family across seeds; locked in audit_plan and re-checked in `select_all`. The only open point is the NLL definition (finding 1).
- **U0 gate.** It is `qpc.utility.gate_record`, unchanged: acc, LL, Brier, retention 0.8, gain 0.03, inclusive margins, and exact decision preservation on every row.
- **Ucal gate.** LL and Brier only, in the qpc margin form (Ucal\* + allowance) − value ≥ 0, on each task and seed, for every candidate and the comparator.
- **Audit plan.** It follows the rule: audit if any non-diagnostic variant passes both gates on every task and seed, plus 5 always-audited partitions (the union of the 4 task-only and 4 diagnostic partitions: DIRECT-TASK, FINE-TASK, CLASS, C-TASK and JOINT λ0.1); all others are PREDECLARED_UTILITY_INELIGIBLE.
- **T\* pool.** The 4 task-only partitions with all their variants except T-TOKEN32, plus the 3 U variants.
- **P\* pool.** The 53 privacy-trained partitions (24 legacy, 24 W-family, 5 K-family) with D0 (legacy only), D1 and the H-* decoders. MEAN and T-TOKEN32 are excluded.
- **Guard.** AUC_v(P\*) ≤ AUC_v(T\*) + 0.005, inclusive, for v1 and v2, on every seed.
- **Benefit.** The mean over seeds of [AUC_pair(T\*) − AUC_pair(P\*)] must be at least 0.02, inclusive.
- **Ordering.** Mean pair AUC, then the worst-seed normalised excess against both U0 and Ucal\*, then the mean summed task LL, then the registered ID order.
- **Estimands.** The selection estimand is the inner AUC of the common-record winner, as a seed mean. This matches the inference estimand. No mismatch beyond findings 1, 3 and 5.

### 3. Family and inference (family.py, infer.py)

- **Slots.** All 23 are exactly as in section 7:
  - P01 is lower-bounded with target 0.02.
  - P02 and P03 are upper-bounded with target 0.01, using v1 for income and v2 for occupation.
  - P04 and P05 are lower-bounded with target −0.01.
  - P06 and P07 are upper-bounded with target 0.01, and P08 and P09 with target 0.005.
  - P10 and P11 test acc − 0.8·acc(U0) − 0.2·acc(const), lower-bounded with target 0.
  - P12 and P13 are upper-bounded with target 0.01 against Ucal\*, and P14 and P15 with target 0.005.
- **Diagnostic signs.**
  - D01–D04 are T-TOKEN32 minus H-TOKEN32; a positive value means held-out fitting helped.
  - D05–D08 are H-TOKEN32 minus H-GLOBAL-TEMP; a positive value means sharing helped.
  - Order within each group: income LL, occupation LL, income Brier, occupation Brier.
  - Partition: JOINT λ0.1.
- **Constants.** B = 1,999, bootstrap seed 20261011, and z as in finding 11.
- **Estimate.** For each seed, the paired statistic is computed first. The endpoint is the equal-weight mean over the three seeds.
- **Interval.** SE is the standard deviation (ddof 1) over the replicates, and the interval is point ± z·SE.
- **Bootstrap.**
  - It is a multinomial draw over exact-record groups (`assess_unit`).
  - One RNG stream produces one weight matrix per replicate, shared by every statistic, arm and seed.
  - Arms are checked for row, label and unit alignment.
- **Classifiers.**
  - Clause outcomes: PASS, NOT_ESTABLISHED_PRECISION, NOT_ESTABLISHED_POINT, MEASURED_VIOLATION or INVALID, with strict inequalities.
  - Diagnostic outcomes: SUPPORTS_POSITIVE, SUPPORTS_NEGATIVE or UNRESOLVED.
- **Label precedence.** `overall_label` matches section 7. I checked it exhaustively over all 72 combinations of (engineering, inputs, technical validity, nominee state, comparator state): 0 mismatches.
- **Missing comparator.** A missing comparator maps to INCOMPLETE_OR_INVALID, per section 6.

### 4. Interface equivalence (bank.py, stages.py, infer.py)

- **Information argument.** For any public, token-level table v and decision map c, the release (t, v(t), c(t)) generates the same σ-algebra as t. So no decoder variant adds or removes information; only finite attackers can differ.
- **The fresh view.** It contains:
  - the one-hot token;
  - every registered table, in registered order, including T-TOKEN32 on the diagnostic partitions;
  - the one-hot decision.

  `fresh_views` enforces decision == token class and validates every table.
- **One shared record.** A single `common_record` per (partition, seed) is assigned to every variant (`assign_variants`) and checked with `check_common`.
- **Shared assessment predictions.** hcal/assess.py writes one `oatt__` unit per partition key (or SRC|U), and infer keys the assessment attack predictions by partition. Every variant of a partition therefore receives identical predictions.
- **Union rule.** First strict improvement greater than 1e-12 on the seed-0 value, with ties keeping the earlier bank.
- **Bank order.** Legacy banks come first. U's composition puts the admitted SRC|U record first, then the audited fresh banks in registered order.
- **Tests.** test_bank's common-record, union and compose_u tests pass.
