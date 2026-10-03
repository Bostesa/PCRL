# Review of the statistical plan in PROTOCOL.md (draft)

**Reviewer.** The read-only mathematics and protocol reviewer, 2026-10-03.

**What was reviewed.** `PROTOCOL.md` (md5 687146d287ba7d3071fa3c595f289478, with z already edited to 3.1440). I also
checked it against:
- the assignment, `PCRL_Overnight_Output_Leak_Diagnosis_And_Replication_Prompt.txt`;
- `COVERAGE_AND_SUPPORT.csv`;
- the estimator code in `stored_model_eval/metrics.py` and `bench_infer.py`;
- the prior study's PROTOCOL.md and PRIMARY_ENDPOINTS.csv.

Synthetic checks are in `score_math_checks.py` and `score_math_checks.out`.

Items are tagged **[REQUIRED]** (fix before LOCK.json) or **[OPTIONAL]**.

---

## 1. Primary family count

**Arithmetic.** 4 + 4 + 2·(1 + 10) = 30, which is exact.
- 4 usefulness endpoints (U-frozen and U-refit × 2 cells).
- 4 macro recovery endpoints (FC and CH × 2 cells).
- 22 pair slots, which are 2 contrasts × (1 Adult sex pair + C(5,2) = 10 HMDA race pairs).

This matches the assignment's literal family.

**Expected NOT_ESTIMABLE slots.** These follow from role counts only, not outcomes; source `COVERAGE_AND_SUPPORT.csv`.
- HMDA race classes 3 and 4 fail support:
  - class 3: attacker_fit 95 < 100 (assessment 93 < 100);
  - class 4: attacker_fit 53, attacker_val 12, assessment 46.
- So 7 of the 10 HMDA pairs are NOT_ESTIMABLE: every pair containing 3 or 4.
- That gives **14 of 30 slots NOT_ESTIMABLE (7 pairs × FC and CH) and 16 estimable**.
- Keeping all 30 in the Bonferroni denominator is what the assignment mandates ("do not shrink"). It is conservative
  and correct.
- **[REQUIRED]** Write the expected 14 NOT_ESTIMABLE slot IDs into the lock now, derived from the repaired roles. This
  shows they were fixed from role counts before fitting.

**Exact aliases.** For binary Adult sex:
- The benchmark estimator `macro_ovr_auc` returns the AUC of column P₁.
- The pair score is P₁/(P₀ + P₁), which equals P₁ whenever P₀ + P₁ = 1.

So **FC-adult-pair(0,1) ≡ FC-adult and CH-adult-pair(0,1) ≡ CH-adult**, up to float rounding of P₀ + P₁. The family
therefore contains **14 distinct estimable endpoints**.
- **[REQUIRED]** Declare these two slots as known aliases in the lock. Keep them in the count, as the prompt
  requires. Have the verifier check equality to about 1e−12. Report them once in prose, so the duplicated PASS is not
  counted as independent support.

**Missing status.** The protocol defines only PASS, NOT_ESTABLISHED and NOT_ESTIMABLE.
- **[REQUIRED]** Add **UNAVAILABLE** (or UNRESOLVED, as the prior study did) for estimable slots lost to a missing
  input, a technical failure or the budget. These slots stay in the denominator. NOT_ESTIMABLE should mean
  class-support failure only.

## 2. Critical value and the normal approximation

**Critical value.** z = Φ⁻¹(1 − 0.05/60) = **3.143980287069** (scipy `norm.isf(0.05/60)`).
- The draft's earlier 2.9352 was Φ⁻¹(1 − 0.05/30), the value for a one-sided Bonferroni at 0.05/30 rather than the
  two-sided simultaneous interval the prompt asks for. The current 3.1440 is correct.
- **[REQUIRED]** Compute z in code (`norm.isf(alpha/(2m))`) with an assertion, and record it to 6 decimals in
  LOCK.json.

**Interpretation.** The intervals are two-sided simultaneous intervals, but decisions are one-sided (lower > target).
So the family-wise rate of false PASS claims is ≤ 0.025, not 0.05.
- **[OPTIONAL]** State this in one sentence. It is conservative and needs no change.

**Is a normal interval with a bootstrap SE acceptable at n ≈ 5,000?** Yes, for AUC and accuracy differences in the
range this study expects.
- AUCs are U-statistics, and differences of paired AUCs on the same rows are asymptotically normal (DeLong). Accuracy
  differences are means of paired {−1, 0, 1} variables.
- **Synthetic check** at n = 5,000, prevalence 1/3, ΔAUC ≈ 0.04, fixed scorers:
  - the bootstrap SE (B = 1,999) was 0.0061–0.0062, against a true sampling SD of 0.0058 over 400 fresh samples (a
    slight overestimate, which is conservative);
  - replicate skewness was |γ₁| ≤ 0.07 and excess kurtosis ≤ 0.12;
  - the 0.5 % and 99.5 % bootstrap quantiles matched the normal ones to within 0.0007.
- **Smallest supported HMDA pair** (3,083 vs 302 rows): SE 0.011, skew +0.03.
- **Monte-Carlo error** of the SE at B = 1,999 is about 1/√(2(B − 1)) = 1.6 %. That moves a bound by about
  0.05·SE, which is negligible except at a knife edge.
- **Why not a percentile bound?** At tail 0.05/60 with B = 1,999, only 1.67 replicates lie in the tail, so a
  percentile bound is not estimable. The normal-SE construction is the only feasible one at this B. The prompt
  forbids "exact tail coverage from a few extreme bootstrap quantiles", so this is consistent.

**Where the approximation is unsuitable, and how to report it** [REQUIRED: lock these rules]:

| Situation | Where it can occur | Rule to lock |
|---|---|---|
| **Difference identically 0 by construction**, so SE = 0 | FC when the full bank selects the same fitted ignore-offset unit used for R(centred) on every encoder seed. Any S4 contrast whose pair bank selects that single-recipient attacker. U-gain if the head equals the constant on every row. | Report est = 0, SE = 0, status NOT_ESTABLISHED, sub-label `IDENTICAL_BY_SELECTION` (or `IDENTICAL_BY_CONSTRUCTION`). Never divide by SE. This is not a statistical non-finding. |
| **Degenerate or near-degenerate discrete SE** | An accuracy gain driven by very few discordant rows (for example, a nearly constant head). In the synthetic case of 6 discordant rows, the replicate skew is +0.31. | Always report the discordant counts n₁₀ and n₀₁ per seed. If min(n₁₀ + n₀₁) < 30 on any seed, flag `NORMAL_APPROX_WEAK`. Add an exact McNemar/binomial bound as a sensitivity check, without changing the locked decision. At target 0.01 such a head cannot PASS anyway. |
| **Bounded statistic near 1** | Not expected in the primary family: AUCs are about 0.5–0.8; the HMDA underwriting constant accuracy is 4,223/4,764 = 0.886, so frozen-head accuracy will sit near 0.9. Possible in S3 if an attribute is almost determined by a multiclass output (for example Adult education → income), and in the planted-leak controls. | If either AUC in a contrast is > 0.98, or either accuracy is > 0.99, flag `NEAR_BOUND`. Also report the 2.5/97.5 % percentile interval and the replicate skewness. The decision stays as locked. |
| **Selection differs across seeds** | The bank picks offset-using candidates on some encoder seeds and ignore-offset candidates on others. | Report the per-seed selected candidate in PRIMARY_ENDPOINTS.csv. The mean-over-seeds estimand is unchanged. |

**Other specification gaps.**
- **[REQUIRED]** State that est is the original-sample point estimate, not the replicate mean, and that the SE uses
  ddof = 1.
- **[REQUIRED]** State that a replicate's statistic is recomputed exactly as the point estimate: weighted AUC per
  attacker seed, then mean over attacker seeds, then mean over encoder seeds, with the same weight vector across
  seeds, views and surfaces.
- **[OPTIONAL]** Report an `n_ne_replicates` column as the prior study did. It should be 0 here.

**Resampling unit.**
- Multinomial counts over canon_key groups: Adult assessment has 5,243 rows in 5,243 groups; HMDA has 4,764 rows in
  4,762 groups.
- Weights are constant within a group, and the same draws are used for every view, seed and surface. This is a valid
  paired construction.
- Its stated limitation is correct: canon_key groups exact duplicate records, not households or applicants. Intervals
  are conditional on the fitted attackers and heads.

## 3. Pair AUC definition

The score is s = P_j / (P_i + P_j), computed on records with S ∈ {i, j}, with j as the positive class. Orientation is
fixed in advance and there is no max(AUC, 1 − AUC).

- **Is it well defined?** Mathematically yes, because softmax, logistic, GBT and Dirichlet-smoothed probabilities are
  strictly positive. Numerically, both can underflow to 0, for example an MLP softmax where a third class dominates by
  more than about 745 in logit.
- **What the code does.** `stored_model_eval.metrics.pair_score` and `bench_infer.pair_auc` already map
  den = 0 → 0.5, which counts as a tie (½) in the Mann–Whitney sum. This is orientation-neutral and acceptable.
  - **[REQUIRED]** Write this rule into the protocol, and report the count of zero-denominator rows per slot. If the
    count is > 0, add a sensitivity AUC that excludes those rows.
- **Precision.** For 0 < den in the subnormal range, the ratio loses precision. A monotone equivalent is
  σ(log P_j − log P_i), from `predict_log_proba` where available. It has identical ranks wherever both logs are
  finite, so the AUC is unchanged.
  - **[OPTIONAL]** Compute the score in the log domain.
- **Which probabilities.** The score uses the 5-class attacker's probabilities, including the columns of unsupported
  classes 3 and 4. That is fine, because the ratio removes the normaliser. The protocol should say it is the same
  fitted multiclass attacker as the macro endpoint, not a separately fitted pair attacker. This is implied but not
  stated.
- **[REQUIRED]** Add the worst-supported-pair summary that the prompt requires ("an average must not hide a
  recoverable group"). Report it per surface, as the maximum over supported pairs of the absolute pair AUC, labelled
  descriptive, with its argmax.

## 4. The full bank and the FC contrast

FC = R(full bank) − R(centred).

**Why FC can be negative or exactly zero.**
- **(a) Selection uses a different criterion.** The bank is chosen on attacker_val log loss but scored on assessment
  macro AUC. A candidate with better validation log loss can have lower assessment AUC.
- **(b) Selection noise.** At about 2,100 validation rows, log-loss differences between candidates can be within
  noise.
- **(c) Exactly zero.** If the selected candidate is the same fitted `centred` unit that defines R(centred), every
  replicate gives exactly 0 (§2 table).
- **(d) No forced ordering.** Fitted-attacker AUC is not monotone under garbling (SCORE_MATH §5b). So even a
  "superset" input does not force R(full bank) ≥ R(centred).

**What a PASS means.** Conditional on the fitted attackers and this reused assessment pool, the validation-selected
full-output attacker achieves macro AUC more than 0.02 above the centred-logit attacker, with simultaneous 95 %
confidence. It does *not* mean any of the following:
- that c by itself is informative (c can be informative only jointly with d);
- that the information gap equals 0.02 AUC;
- that centring protects against other attackers.

**Two logical gaps:**

1. **[REQUIRED] The comparator is weaker than the bank.**
   - The full bank includes the `prob` unit as an ignore-offset candidate, but R(centred) uses only the centred
     slate.
   - In the primary binary cells, prob and centred are bijective on every row (SCORE_MATH §7). Anyone holding a
     centred release can compute the probabilities, and the prompt says "the richer view can use the smaller view"
     and "include margin-derived probabilities ... in the attacker bank".
   - If the `prob` attacker beats the `centred` attacker, FC would PASS without the offset playing any role. It
     would be a parameterisation artefact, misattributed to the offset.
   - **Fix:** define the centred-release attacker as an **ignore-offset bank**: NL on centred, NL on prob, and NL on d
     as a single non-redundant column (K − 1 differences for multiclass). Select by validation log loss with the same
     tie rule. Then use FC = R(full bank) − R(ignore-offset bank).
   - The full bank must contain every ignore-offset-bank candidate, which makes the two banks nested. Use the same
     ignore-offset bank as the left term of CH.
   - Keep the single-slate `centred` and `prob` rows as matched canonical ablations, as the draft already does.
2. **[REQUIRED] Selection granularity and candidate list are ambiguous.**
   - "Slate on (l0, l1)" could mean the 43 configurations or the NL unit.
   - The lock must state the exact candidates. The suggestion is {NL(l0, l1), NL(d, c), NL(centred), NL(prob),
     NL(d)}, each with GBT-vs-MLP selected as in the benchmark.
   - It must also state whether L enters, and that bank selection happens **once per encoder seed**, using
     attacker-seed-0 validation log loss (or the mean over attacker seeds; pick one). The chosen candidate is then
     refit and scored at attacker seeds 0, 1, 2.
   - Report the selected candidate per seed, and whether it uses c.

**Interpretation rule to lock.** FC PASS is described as "offset-attributable" only if the bank selected an
offset-using candidate ((l0, l1) or (d, c)) on every encoder seed. Otherwise, report the PASS with the selection
pattern and without offset attribution.

**[OPTIONAL]** Report the telescoped sum FC + CH = R(full bank) − R(hard). It is exact per replicate. It is the
analogue of the prior P1 (+0.263), so readers can see how the earlier gap splits. Label it descriptive.

## 5. Secondary families

**S3 (34): count correct.**
- Adult has 8 disallowed pairs: income {race, sex}; employment {race, age_group, marital_status}; education
  {sex, race, income}. HMDA has 6: underwriting {race, ethnicity}; pricing {race, sex}; fair_lending {race, sex}.
- That is 14 pairs × 2 contrasts = 28, plus 6 purposes = 34. All 14 cells have ≥ 2 supported classes, so 0
  NOT_ESTIMABLE are expected.
- z = Φ⁻¹(1 − 0.05/68) = 3.180426.
- **[REQUIRED]** Write each secondary family's z in the lock.
- **[REQUIRED]** S3 contains the two primary cells' FC and CH, and U-frozen for income_prediction and underwriting.
  State that these are re-tested inside S3 (overlapping families), or exclude them and say so. Either is fine if
  locked.
- **[REQUIRED]** Define `dc` for K > 2. The draft defines only binary (d, c). The suggestion is (l̃ or K − 1
  differences, c).
- **[REQUIRED]** Adult education_assessment seeds 1 and 2 have float64 softmax saturation on about 21–30 % of rows
  (SCORE_MATH §7). There, prob is a strict garbling of centred. Report saturation counts per cell and seed, and do not
  describe R(centred) − R(prob) as parameterisation for those cells.
- **[REQUIRED]** The Stage 1 log loss must be computed as −log_softmax(l)[y] in float64. Clipping softmax gives
  ε-dependent or infinite values on saturated misclassified rows. The frozen-head logit ranges reach 4,304.
- **[OPTIONAL]** Adult employment task class 5 has 3 / 1 / 1 rows. The multiclass task ROC-AUC and balanced accuracy
  should state their class coverage, using the same support thresholds and macro over supported task classes.

**S4 (6): count correct.** Each of 3 contracts (full, centred, hard) has 2 contrasts, R(pair) − R(income) and
R(pair) − R(employment). z = Φ⁻¹(1 − 0.05/12) = 2.638257.
- Race is the only attribute disallowed for both income_prediction and employment_analysis (PERMISSION_TABLE). Its
  supported classes are {1, 2, 4}.
- **[REQUIRED]** State the conjunction: a combination finding for a contract requires *both* contrasts to PASS. This
  is an intersection–union test, so no extra multiplicity is needed.
- **[REQUIRED]** The "centred" pair contract needs the same ignore-offset-bank treatment as §4. If the pair bank
  selects a single-recipient attacker, that contrast is identically 0 (§2 rule).

**S5 (4): count correct.** z = Φ⁻¹(1 − 0.05/8) = 2.497705. The selection and nominee rules have gaps:

1. **[REQUIRED] Untreated gain ≤ 0 or near 0.**
   - The nominee rule uses (U2_val(F) − k)/(U2_val(A) − k) ≥ 0.80.
   - If the denominator is 0, the ratio is undefined.
   - If it is negative, the inequality reverses. For example, an untreated gain of −0.01 and a FARE gain of −0.02 give
     a ratio of 2.0, which would "pass".
   - The screen uses the *frozen-head* gain, not U2, so it does not protect the denominator.
   - **Fix:** add to the screen that the untreated U2 attacker_val gain is ≥ 0.03 on average and > 0 on every seed.
     If that fails, the cell is ineligible, recorded as screen failure. Write the nominee and endpoint rule in linear
     form: (A_F − k) − 0.8·(A_A − k) ≥ 0. When A_A − k > 0 this is equivalent to the ratio, and it is always defined.
2. **[REQUIRED] Replace the delta-method ratio endpoint.**
   - Use the linear contrast L = mean over seeds of [A_F − 0.8·A_A − 0.2·k], with PASS iff lower(L) > 0, using the
     same bootstrap.
   - A delta method on a ratio whose denominator is about 0.03 ± 0.006 is unreliable. The prior HMDA gain was about
     0.017.
   - Report the ratio itself as descriptive. Specify per-seed or pooled. The linear form removes the ambiguity.
3. **[REQUIRED] Seeds without a nominee.** If any encoder seed is NO_FEASIBLE_NOMINEE (or the official FARE
   admission fails for the class counts), the 4 slots are reported as NOT_ESTABLISHED (or UNAVAILABLE). No fallback
   configuration may PASS. This mirrors the prior "every encoder seed has an admissible nominee" rule.
4. **[REQUIRED] Ambiguous eligibility wording.** "Every sensitive class with ≥ 100 defense_fit rows plus ≥ 2
   supported classes" can be read two ways. Reword as "every declared class has ≥ 100 defense_fit rows, and ≥ 2
   classes meet the 100/30/100 attacker support".
5. **[REQUIRED] Name the utility measure.** Acc in "Acc(FARE) − Acc(A)" must be named: the U2 probe, as in prior P3,
   or the own-head release accuracy.
6. **[OPTIONAL] Ties.** The screen statistic depends only on the purpose, so every attribute of a purpose ties. The
   attribute is then chosen by the lexicographic tie-break. Say so explicitly, and give the exact string order
   (`age_group` < `ethnicity` < `income` < `marital_status` < `race` < `sex`).
7. **[OPTIONAL] Trivial tasks.** Record whether the selected task label is a deterministic recoding of an input
   feature, as earlier project notes found for Adult education_level and occupation_group. A trivially recoverable task
   makes "retains a useful task" weak evidence.

## 6. Consistency with the prior study

| | Prior output-aware study (f7425b15) | This study |
|---|---|---|
| Family | 12 rows | 30 primary, plus S3 = 34, S4 = 6, S5 = 4 |
| Bound | One-sided lower **percentile** bound at α = 0.05/12 = 0.004167 each | Two-sided simultaneous Bonferroni **normal** interval est ± z·SE, z = Φ⁻¹(1 − 0.05/60) = 3.14398; decision on the lower end |
| Bootstrap | B = 20,000, seed 20261011, numpy "linear" quantile; secondary 90 % at B = 2,000, seed 20261013 | B = 1,999, seed 20261021, SE = replicate sd |
| Shared | Group bootstrap over assessment canon_key groups, predictors fixed, paired | Same |

**[REQUIRED]** Add a short "Relation to prior inference" paragraph covering three points:
- Both constructions are stated as above.
- The new one is prompt-mandated.
- Decisions are not comparable across studies. The new per-endpoint one-sided level, 0.05/60, is about 5× stricter
  than 0.05/12.

Any replay of a prior endpoint for custody uses the prior estimator. New endpoints use the new one.

## 7. Other protocol items

- **[REQUIRED]** Correct the saturation sentence: "|d| ≥ 36.7 in float64 gives p1 ∈ {0, 1}" is false for d < 0
  (SCORE_MATH §7). Replace it with:
  - p1 = 1.0 for d ≥ 36.7368;
  - p1 = 0.0 only for d ≤ −709.78 (scipy `expit`) or d ≤ −745.13 (stable softmax);
  - also count |d| ≥ 16.6355, the float32 threshold. In the primary data that is 92 + 11 Adult rows.

  Also require that all probability surfaces be computed and stored in float64.
- **[REQUIRED]** Make the targets consistent. The table says "≥ 0.01" and "≥ 0.02", but the decision rule says
  "lower > target". Use one (the prior study used strict >).
- **[OPTIONAL]** Define the centred binary surface as (−d/2, d/2) computed from d, so it is bit-exactly a function of
  d. It is already bit-equal to l − c on the stored float32-upcast logits.
- **[OPTIONAL]** Add a GBT parity diagnostic. HistGradientBoosting bins by quantiles, so it is nearly invariant to the
  monotone map d ↦ σ(d). NL(centred) and NL(prob) should therefore agree closely whenever both select GBT, and any
  gap then comes from the MLP. Reporting the GBT-vs-MLP choice per surface makes the centred-vs-prob difference
  interpretable.
