# Statistical and readiness review of PROTOCOL.md (before any reader fit)

Statistical/readiness reviewer (separate agent), 2026-10-17, before `READER_BANK_LOCK.json` and before any reader fit.

**Read:**
- `PROTOCOL.md`, `FROZEN_CANDIDATE_MANIFEST.json` and `SOURCE_ADMISSION.json` (414 copies, 0 problems);
- the source `ANALYSIS_LOCK.json`, `report/DEV_REPORT_RESULTS.json` (published values only), `RESEARCH_DECISION.md`, `PROTOCOL_POSTRUN_CLARIFICATIONS.md` and `READERS_SUMMARY.json`;
- `amd/report.py`, `amd/readers.py`, `scm_bench/readers.py`, `task_conditioned_representation/attacks.py` and `task_headroom/inference.py`;
- the installed scikit-learn 1.8.0 sources for SplineTransformer and HistGradientBoosting.

**Not done:** no model was fitted, no SEX or task label was read, and neither DEV_REPORT truth nor the sealed assessment was touched.

## Findings by question (short)

1. **Reader bank (section 7): fair in design.**
   - Every release gets the same recipes, model seed, effort and candidate set. Single views see only their own output. Identical occupation wires share one fit, and the finalist's internal score d is never a feature.
   - Three ambiguities would let implementation choices be made late: candidate order and tie handling (A1), recipe construction (A2), and CATI details (A3). The reader-seed diagnostic also misclassifies HGB as stochastic (A4).
   - **CATI regime by wire** (source coverage, ATTACK_FIT distinct income vectors):

     | Regime | Releases |
     |---|---|
     | Categorical | FINALIST, TASKONLY, B2TWIN (6); PUB_IBSI (exactly 32 on both seeds, the threshold itself) |
     | Continuous | NEURAL_RAW, H_LJ (> 7,000) |

   - What CATI does to each claim:
     - **B:** whatever extra power CATI gains on categorical codes acts against FINALIST and not against its continuous references, so it is *conservative* for claim B.
     - **M and J:** symmetric, because TASKONLY and B2TWIN are 6-level codes too.
2. **Section 7.3: AGREED_WITH_CHANGES** (below). The changes complete the specification; the model class is unchanged.
3. **Section 7.4: deterministic and ATTACK_VALIDATION-only, but the stated tie property fails.**
   - With the registered pair order (16 pair readers, then 16 income-view, then 16 occupation-view), a new pair recipe sits before the source ignore-one alternatives. It would win a tie against a source ignore-one winner, so "a tie with a source reader keeps the source reader" is not delivered.
   - Fix in A1.
4. **Section 8: coherent as a conjunctive robustness re-read.** It keeps the same keys, signs, margins, conjunctions, critical values and shared weights.
   - Requiring ORIGINAL ∧ ENHANCED is an intersection-union test: no α inflation, and B, M and J keep α/3 each.
   - **Caveat 1.** "Larger reader class" is not "stronger reader on DEV_REPORT". Each ENHANCED winner is selected on ATTACK_VALIDATION among 48 candidates. The AUC SE on 3,000 people is about 0.010, so the enhanced winner can read *lower* on DEV_REPORT than the original.
   - **Caveat 2.** The interpretation table is incomplete. It has no case for "ORIGINAL fails, ENHANCED holds", which matters for J (source `JOINT_ADVANTAGE_NOT_ESTABLISHED`, s1 clause bound −0.0025) and for the nested 0.020 test. It also has no case for control failure or reproduction discrepancy.
   - Fix in B1–B2. Both seeds share people, so the per-seed clauses are positively dependent. The IUT is valid under any dependence, but "both seeds" is robustness to the release seed, not replication.
5. **Section 10: acceptable as a first-order approximation.**
   - Problems in the plan: D1 is mis-specified (it omits claim B's 8 recipient clause bounds); "the two binding income clauses" is ambiguous; the search and Monte Carlo are unspecified; Σ is singular; and the estimand is not stated (frozen readers or refit).
   - **Readiness fact, knowable now from published reader-free values:** D2 is governed by the income utility clauses against Ucal, whose plug-in slack is about 0.03–0.04 SE. This does not depend on the reader bank (Notes N3).

---

## Decision on 7.3: AGREED_WITH_CHANGES

**Why the construction is agreed in substance.**
- It is label-free and fixed in advance, with every transform fitted on ATTACK_FIT, and applied by the same rule to every wire.
- **Income view, categorical case.** Indicators under L2 give a shrunken per-level SEX rate, close to LOOKUP; harmless.
- **Income view, continuous case.** A flexible one-dimensional smoother of P(income = 1). That is the right reader class, because a single income view is a function of that scalar.
- **Occupation view.** An additive spline in the log-contrasts plus bilinear terms and the decision: a reasonable non-tree reader of the 4-simplex.
- The single-view constructions feed the ignore-one alternatives, which FINALIST, H_LJ and B2TWIN share.

**Required changes** (put verbatim in `READER_BANK_LOCK.json`):

**C1. Formulas.**
- logit P(income = 1) = log max(p1, 1e-300) − log max(p0, 1e-300).
- Occupation log-contrasts: c_j = log max(p_j, 1e-300) − log max(p_0, 1e-300), j = 1–4, with class 0 the first stored column.
- "p" and "log p" for income mean both columns, (p0, p1) and (log max(p0, 1e-300), log max(p1, 1e-300)), as in §7.2.

**C2. Spline knots.**
- Problem: scikit-learn 1.8.0 checks for duplicate knots only when knots are given as an explicit array (`_polynomial.py` line 894), not for `knots="quantile"` (lines 768–776). Tied outputs can therefore give repeated knots and, if the first two coincide, a zero-length boundary extension. PUB_IBSI occupation has 471 distinct vectors, a median of 13 people per vector.
- **Rule:**
  1. base = np.quantile(x_ATTACK_FIT, linspace(0, 1, n_knots)), with numpy's default linear method (what `knots="quantile"` computes);
  2. base = np.unique(base);
  3. if at least 2 knots remain, use SplineTransformer(knots=base[:, None], degree=3, extrapolation="constant", include_bias=True);
  4. otherwise omit that block.
- Where no ties exist this equals the registered construction. Record every knot vector per wire and seed in the lock.

**C3. Income quantile groups (continuous case).**
- cuts = np.quantile(P1_ATTACK_FIT, [1/6, 2/6, 3/6, 4/6, 5/6]) (linear method).
- group = np.searchsorted(cuts, P1, side="left"), so a value equal to a cut takes the lower group.
- Empty groups are allowed: their columns are constant, and FiniteScaler gives them scale 1.

**C4. Categorical income.**
- Categories are the distinct exact float64 byte strings of the received 2-vector on ATTACK_FIT, sorted by (P(income = 1), bytes).
- An unseen vector at scoring gets all indicators zero, as registered.
- Report unseen counts per wire, role and seed. Expected: 0 for FINALIST (the manifest shows the same 6 package vectors bitwise on every role).
- Record the regime of each income wire in the lock before any SEX-labelled fit. PUB_IBSI is categorical under "≤ 32".

**C5. Columns and order (fixed).**
- **Income view, categorical:** [I_1..I_m].
- **Income view, continuous:** [G_0..G_5, p0, p1, log p0, log p1, 8 spline columns of the logit].
- **Occupation view:** [p0..p4, c1..c4, spline(c1), spline(c2), spline(c3), spline(c4) at 7 columns each, c1c2, c1c3, c1c4, c2c3, c2c4, c3c4, D0..D4]. That is 48 columns, or fewer if C2 drops knots.
- **Pair:** [income block, p0..p4, c1..c4, interactions in income-indicator-major order].
- Products and interactions are formed on raw values; FiniteScaler is then fitted on the whole design on ATTACK_FIT.
- The decision one-hot uses np.argmax (first index on ties), as in the source `kinds()`.

**C6. Convergence.**
- Record `n_iter_` and every ConvergenceWarning for each CATI (lbfgs, max_iter=5000) and MLP fit. The source code suppresses warnings; record them here instead.
- A non-converged fit is used as fitted and flagged. It is never refitted with other settings.

---

## Must-fix before fitting

### A. Blocks `READER_BANK_LOCK.json` and reader fits

**A1. Candidate order, values and ties (§7.4).**
- **Single view (16):** the 8 source readers in source order (OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB, LOOKUP), then ET5, ET25, HGB15, HGB31, MLP64, MLP6432, CATI01, CATI1.
- **Pair (48), source-first:**
  - the 24 source candidates in source order: 8 source pair readers, 8 source IGNORE_occupation, 8 source IGNORE_income;
  - then 8 new pair readers, 8 new IGNORE_occupation, 8 new IGNORE_income.
- **Values for source candidates.**
  - They enter with their *stored* oriented ATTACK_VALIDATION AUCs from the admitted source `READERS_SEED.json` (bitwise), not refit values.
  - A selected source candidate is scored with its stored DEV_REPORT score.
  - A §7.1 refit that is not bitwise is a reported discrepancy, never a substitute.
- **Ties** are exact float equality with the earliest candidate winning (source `pick()`: key (auc, −index)).
- **Result:** the ENHANCED winner equals the ORIGINAL winner unless a new candidate is *strictly* higher on ATTACK_VALIDATION.

**A2. Recipe construction.**
- Do **not** route new recipes through `task_conditioned_representation.attacks.fit_reader`. Its name-substring dispatch would silently give:
  - HGB15 and HGB31 the source HGB settings;
  - MLP64 and MLP6432 the source (64, 32) network with alpha 1e-4;
  - CATI01 an LR with C = 1 and max_iter = 500.
- Lock each new estimator's full `get_params()` and the sklearn, numpy and scipy versions (1.8.0 / 2.4.2 / 1.17.1 here), and assert them at fit time.
- Defaults to state explicitly:

  | Model | Defaults |
  |---|---|
  | MLP | adam, relu, shuffle=True, tol=1e-4, n_iter_no_change=10, learning_rate_init=1e-3. The training-loss stop is not early stopping. |
  | HGB | max_bins=255, max_features=1.0 |
  | ET | bootstrap=False, gini, max_depth=None |
  | LR | l2, fit_intercept=True, tol=1e-4 |

**A3.** Adopt C1–C6.

**A4. §7.6 seed classification.**
- HistGradientBoosting here is **seed-invariant**: early_stopping=False, max_features=1.0, and n = 8,001 < 200,000 (the binning subsample). random_state only drives the early-stopping split, binning subsample and feature subsampling (`gradient_boosting.py` lines 603–611 and 715–721; `splitting.pyx` lines 502–511).
- Replace "stochastic (ET, HGB, MLP)" with "seed-dependent (ET, MLP)".
- For seed-invariant winners (HGB, LR, CATI, LOOKUP), report "seed-invariant", not "measured zero reader variance".
- State that §7.6 measures only the RNG component, not the auxiliary-sample component (which ATTACK_FIT people were drawn). §10's "measured" reader term is therefore a lower bound.

**A5. Control failures (§7.5).**
- No recipe is dropped, replaced or retuned after any control is seen.
- An ATTACK_VALIDATION control outside its band is a logged deviation before the lock. Any repair applies to every release and view.
- ENHANCED CV requires all 5 recipes × 3 views × 2 seeds, on both ATTACK_VALIDATION and DEV_REPORT.
- The recomputed OUT_HGB controls are compared bitwise with the source's stored controls.

### B. Freeze before `REPORTING_LOCK` (does not block reader fits)

**B1. Complete the interpretation table** for each of B, the nested 0.020 test, each M step and the M label, and J.

| ORIGINAL | ENHANCED | Label |
|---|---|---|
| holds | holds | `SURVIVED_THIS_READER_BANK` |
| holds | fails (CV valid) | `READER_SENSITIVE`: first failing condition in source condition order, plus the side that moved it (below) |
| fails | holds | `ENHANCED_ONLY_PASS`, descriptive. The source label stands: a second look at the same people cannot rescue J or the 0.020 target. |
| fails | fails | `NOT_ESTABLISHED_EITHER_BANK`; the source label stands |
| any | ENHANCED CV fails | `ENHANCED_NOT_INTERPRETABLE` (ENHANCED claim labels are `CONTROL_FAILURE`) |
| does not reproduce the source | any | `DISCREPANCY`; interpretation suspended for that claim until resolved |

- **Side attribution for `READER_SENSITIVE`,** from the paired reader-class changes:
  - `FINALIST_READ_HIGHER`: the FINALIST winner reads higher;
  - `COMPARATOR_READ_LOWER`: the comparator winner reads lower;
  - `BOTH`.
- `COMPARATOR_READ_LOWER` must be described as ATTACK_VALIDATION selection instability, not added finalist exposure.

**B2. ENHANCED E is re-evaluated.**
- Its 8 recipient point gates are reader-dependent and use the ENHANCED single-view winners.
- `claim_ref E` inside ENHANCED B, M and J means ENHANCED E.
- M_SS uses B's H_LJ keys, so its ENHANCED status follows from them.
- IDENTICAL_RELEASE occupation clauses stay exact zeros, because the winner is shared by construction.

**B3. Inference constants.**
- Keep K_P = 52, K_S = 20 and the four critical values exactly, even though the H_JJ and T_TREE keys are dropped. The source code asserts these values.
- The ORIGINAL scorecard uses the stored source scores.

**B4. Freeze the §10 specification before the DEV_REPORT re-read**, because the draws come from it. Items F1–F6 under "Notes for the feasibility stage" below.

**B5. Lock contents.** The REPORTING_LOCK also hashes:
- the §7.6 refit models and their DEV_REPORT scores;
- the CATI knot vectors and categories;
- the frozen candidate order;
- the feasibility specification.

---

## Optional

- **O1.** Clip c_j and the income logit to their ATTACK_FIT [min, max] before the linear and product terms. This matches extrapolation="constant" and keeps extreme scoring values from driving products.
- **O2. Symmetric continuous counterpart in the pair CATI.**
  - Interact the income logit (or its spline) with the 9 occupation features for continuous income wires.
  - Or use decision-aligned groups: tertiles of P(income = 1) within each income decision, mirroring the finalist's 3 + 3 levels.
  - This lowers the chance that `READER_SENSITIVE` reflects reader-design asymmetry. The current asymmetry is conservative for B.
- **O3. Reader-matched descriptive benefits:** the same recipe on FINALIST and on each comparator, for all 16 pair recipes. Mark it DEV_REPORT-descriptive and never used for selection. It separates the effect of the reader class from selection noise.
- **O4. Near-tie flag.** Report each winner's ATTACK_VALIDATION gap to the runner-up and to the original winner. Flag `AV_NEAR_TIE` when the gap is below one AUC SE (≈ 0.010).
- **O5. Auxiliary-sample component.** If budget allows, refit the FINALIST, NEURAL_RAW and H_LJ pair winners on 4 household-bootstrap resamples of ATTACK_FIT.
- **O6.** Centred log-ratio coordinates in the occupation view, instead of class-0 contrasts, to remove the arbitrary reference.

---

## Notes for the feasibility stage

### Specification to freeze (B4)

**F1. D1 must include claim B's 8 recipient clause bounds** (U1 ≤ 0.010). As written, D1 has only point gates plus the benefit bounds, so it is not the source scorecard.
- Two of the eight are IDENTICAL_RELEASE and pass deterministically.
- State that CV and the availability conditions are assumed to pass.

**F2. Name the scenario-3 clauses by key.** The published values give three near-binding income quantities:

| Endpoint | Slack | SE | Slack / SE |
|---|---|---|---|
| `UTILITY\|income\|Ucal\|brier_excess\|s0` | 8.1e-5 | 0.00193 | 0.042 |
| `UTILITY\|income\|Ucal\|brier_excess\|s1` | 5.6e-5 | 0.00205 | 0.027 |
| `UTILITY\|income\|{U0,Ucal}\|accuracy_loss\|s1` | 4.1e-4 | 0.00372 | 0.110 |

- The U0 and Ucal accuracy endpoints are numerically identical, because calibration keeps the decision. Move them together.
- "Fully to cap" makes the limits as n → ∞ equal to 0.5 per clause for D1 and Φ(−c1) = 0.0083 for D2. Report that as *unreachable*, not as "beyond n_max".

**F3. Search.**
- A fixed log-spaced grid from 3,955 to at least 10^8 households.
- At least 20,000 Monte Carlo draws per grid point with a fixed seed; report the MC SE (≤ 0.0036).
- Required n is the smallest grid n reaching the target; report the full curve and the adjacent-grid bracket.
- Report the analytic n → ∞ limit per definition. Joint power can be bounded or non-monotone when a point gate sits at or above its cap. Keep "not reached by n_max" separate from "limit below target".

**F4. Singular Σ.**
- IDENTICAL_RELEASE endpoints have zero variance; treat them as deterministic.
- Duplicated U0/Ucal accuracy keys are perfectly collinear.
- Factor Σ by eigendecomposition with negative eigenvalues clipped to 0, not Cholesky.

**F5. Which scorecard governs.** Run on both the ORIGINAL and ENHANCED plug-ins. The less favourable one governs §13.

**F6. Estimand.**
- **Primary:** frozen releases, frozen fitted readers and frozen winners, applied to new independent households from the same population. No reader or release variance enters, but the confirmation is then "against these fitted readers".
- **Scenario 4** applies only if a prospective protocol refits readers or retrains releases.
  - The measured reader term is the RNG component only, a lower bound (A4).
  - The release term from two seeds has 1 degree of freedom. It is confounded with the reader seed (MODEL_SEED_BASE + s) and with same-people noise, so it stays hypothetical.
- **Re-selection.** If winners are re-selected prospectively, the endpoint is a max over candidates and not smooth; MVN does not cover it.

### Justification and assumptions to state

**N1. The scaling rule Σ(n) = Σ_DEV × 3,955 / n is valid to first order.**
- Every endpoint is a household-clustered ratio mean or a two-sample U-statistic (AUC). Its Hájek projection has variance proportional to 1/G, and the remainder is O(1/G²).
- Assumptions:
  - households are drawn iid from the same population: same SEX mix, household-size distribution (1.503 persons per household) and output distribution;
  - the analysis is unweighted (no PWGTP or replicate weights);
  - no transport: another state-year changes both μ and Σ.
- B = 2048 adds about 1.6% relative Monte Carlo error to each SE; negligible.

### Near-zero utility slack, and what it implies (knowable now; reader-free, so identical in both scorecards)

**N2. A point gate with true slack s > 0 passes with probability Φ(s·√(n / 3955) / SE_DEV).** That tends to 1, but slowly. For 80% marginal power on one clause:

| Clause | D1 point gate | D2 one-sided c1 bound |
|---|---|---|
| Ucal Brier s1 | ≈ 3.7 million households | ≈ 55 million |
| Ucal Brier s0 | ≈ 1.6 million | ≈ 23 million |
| accuracy_loss s1 | ≈ 0.23 million | ≈ 3.4 million |

- At n = 100,000, the Ucal Brier s1 clause alone gives D2 a probability of about Φ(0.138 − 2.395) ≈ 0.012.

**N3. The plug-in slack is only 0.03–0.11 SE, so the data do not show that the true slack is positive.** Plug-in power is an assumption here, not evidence. Report two things:
- **(a)** power curves over an assumed true-slack grid, for example s ∈ {−0.001, 0, plug-in, 0.0005, 0.001, 0.002};
- **(b)** assurance: power averaged over μ ~ N(μ̂, Σ_DEV). Under (b), each near-binding clause is capped near Φ(slack / SE_DEV) ≈ 0.51–0.54 as n → ∞, so D1 and D2 assurance stay at or below about 0.5 for every n.

**N4. D3 and the shrinkage scenarios.**
- Under the ORIGINAL plug-in, D3's limit is 0: the seed-1 benefit against NEURAL_RAW is 0.0199, below 0.020. At 75% and 50% shrinkage, every benefit sits below or near 0.020.
- Report this with the registered wording ("not reliably establishable at that target as n grows").
- The plug-in benefits come from the result that motivated this follow-up, so they are favourably selected. Consider presenting the 75% scenario first.

### Other notes

- **N5. α.** D2 uses c1 (α/3 across B, M and J). A B-only prospective intersection-union test at one-sided 0.025 would use c = 1.960. That may be shown as a sensitivity, never as the registered criterion.
- **N6. Seeds.** Σ_DEV carries the covariance across seeds. Do not treat the two seeds as independent replications.
- **N7. Extra variance.** Add reader or release variance as an unscaled Σ_extra on AUC-type endpoints only, using paired refit differences for benefits.
- **N8. Optional non-normal check:** μ + √(3955 / n) × (bootstrap replicate deviation).
