# EDG1 statistical review, round 1: the frozen design

Statistical reviewer, campaign `pcrl_estimator_diagnostic_v1` (EDG1). Written 2026-10-25 from 17:43Z (`date -u`). Round 1
reviews the design only. I have seen no simulation outcome.

## Scope

**Read.**
- EDG1:
  - PROTOCOL.md as registered at c169705c2. The amendments appended since (A1-A3) are tooling only, and
    `git diff c169705c2 HEAD` shows no change to `analyze.py`, `estimators.py`, `simulate.py`, `laws.py` or `bank.py`.
  - GRID.json, LAWS.json and BENCH.json (timing only).
  - `code/edg/{analyze,estimators,simulate,laws,bank,selftest,source_prc}.py`, and the vendored LOOKUP, orientation, AUC and
    selection helpers.
- DRF1:
  - RESEARCH_DECISION.md;
  - `drf/dr.py` (`nested`), `drf/rules.py` (`useful_optimistic`), `drf/solver.py`, `drf/finite_law.py`;
  - two published aggregates, read only to identify which law the twin's numbers use: `results/dr/FIT_V0.json`
    (`path_full[0].insample`) and `results/dr/OPTIMISTIC_V0.json` (`nested.entries.IDENTITY`).

**Not opened.** The contents of `out/` (I listed file names only), SUMMARY.json, TABLES.md, `figures/`, any draft report and the
verifier's outputs. I opened no `.private` folder, cache or other worktree.

**Computed** in the scratchpad, on one thread, with bytecode writing off:
1. a population-only tie check of the 16 frozen laws (`laws.build`; no sampling, no estimator; 0.5 CPU-s);
2. a small simulation on an invented independence law, using my own code and not a grid law (2.4 CPU-s).

## Verdict

The design is statistically sound for its stated purpose:
- the estimators reproduce DRF1's computations;
- the laws satisfy their stated relations;
- nothing in `analyze.py`, `estimators.py` or `simulate.py` mis-computes a declared number.

There are **five REQUIRED items** (a2, a3, a4, e2, f1). They concern what reported quantities mean (the estimand and its labels)
and which claims the numbers can carry. None requires re-running the grid. The RECOMMENDED items add post hoc, labelled
summaries that the questions need.

## (a) The decision rule R2SE

**a1 NOTE (verified).** The SE code is the standard DeLong (Hajek) linearization.
- Each person's influence value is (V10 - AUC) / n1 for positives and (V01 - AUC) / n0 for negatives, with ties counted as 1/2.
  These are the influence values of the AUC as a functional of the joint (S, score) law, so the random class counts are accounted
  for.
- The values sum to zero, so the unit-sum sandwich needs no centring. The G / (G - 1) factor is negligible here (G >= 600).
- The selftest checks the person-unit case against DeLong.
- In the clustered cell, summing over households is the usual clustered U-statistic linearization. Within-household
  opposite-class pairs (about 6,000 of about 10^8) do not move expectations.

R2SE is therefore a correctly specified two-sided rule (nominal 4.55%, 2.28% per direction) in two settings:
- B_same and B_indep, whose scores are known and fixed;
- D_bank, conditional on FIT and VALIDATION, because SCORE units are disjoint from them (by household in the clustered cell).

It is correctly labelled naive for C_same and C_nested: the ranking is chosen on the same sample it is scored on.

**a2 REQUIRED. For D_bank, R2SE targets the fitted readers, not dA.**
- Conditional on the oriented LOOKUP tables, E[d_bank] = dpop = pop(T_x) - pop(T_y).
- dpop differs from dA wherever the two views' LOOKUP estimation losses differ: the REF levels, RED pair - income, and every
  non-null RED and COMP comparison.
- A resolved d_bank can then be a correct statement about the readers that still counts as a false positive or a wrong direction
  against dA. For a fixed dpop != 0, such counts rise as SCORE grows instead of falling to the nominal level.

The report must:
1. describe D_bank's dA-relative rates as "how often a valid fitted-reader conclusion misstates population information", never as
   miscalibration of R2SE;
2. base any calibration statement on the rate measured against dpop. That rate can be computed from the recorded per-repetition
   `D_bank_pop` values and must be labelled post hoc.

The grid contains exact calibration checks. In IND (all three comparisons) and in COMP occupation - income, every score of the
wire has population AUC exactly 1/2. There dpop = dA = 0 identically, and D_bank's false-positive rate should be about 4.55%.

**a3 REQUIRED. Label R2SE for D_nested as naive as well (the protocol labels only C).**
- The pooled cross-fit AUC mixes five tables. Each person's label enters the four tables that score the other folds' members of the
  same wire.
- "Rankings held fixed" therefore ignores both the variability of the tables and this cross-fold label dependence. Item e2 shows
  that the dependence biases the pooled value.
- As in a2, the target is the fitted readers' recovery, not A.

D_nested's resolved rates describe a naive practice, as C's do.

**a4 REQUIRED. The B null comparisons cannot show that R2SE is calibrated.** I checked the frozen laws (population only). In
every comparison with dA = 0, the two known-posterior scores fall into one of two cases.
- **Identical rankings or constant scores.** Then d = se = 0 exactly, and nothing can be resolved. This covers:
  - IND at pi = 0.5, and IND|0.2|uniform: one posterior value per view;
  - the uniform REF levels: 8 values per level, forming the same 8 groups at every level;
  - RED uniform, pair against income.
- **Rankings that differ only through float64 rounding of the merged posterior:**
  - IND|0.2|skewed: 2 values per view where the law has 1 (max |post - pi| = 2.8e-17);
  - REF|0.5|skewed: 15 / 20 / 24 / 18 values at L16 / L64 / L256 / L1024 where the law has 8 (REF|0.2|skewed: 16 / 31 / 30 / 23);
  - RED skewed: 14 or 17 pair values against 11 or 14 income values;
  - COMP single views: 1-5 values where the law has 1.

Splitting ties between wires with equal likelihood ratios leaves the population AUC unchanged (Delta = 0), so B stays exactly
unbiased and A is unaffected. But B's null false-positive rates and P(d = 0) measure float tie-breaking, not R2SE.

Any claim that R2SE is calibrated must therefore rest on two post hoc checks, each labelled as such:
1. the coverage of dA by d +- 2 se for B_indep and B_same in the non-null comparisons (RED income - occupation, RED pair -
   occupation, COMP pair - income, COMP pair - occupation), where E[d] = dA exactly;
2. D_bank in IND and in COMP occupation - income (a2).

Also report B_indep's bias against A, which should be 0 within MCSE, as a check of the simulation.

**a5 NOTE.**
- At 500 repetitions the MCSE of a rate near 4.55% is about 0.93 points.
- `_rate` gives an MCSE of 0 for a zero count. Quote an upper bound instead (about 3 / R = 0.6%).
- The rule resolves any case with se = 0 and d != 0. Count such cases; they should be rare and arise only from degenerate scores.
- B_indep is scored on n people and D_bank on 0.3 n, so their power and false-positive rates are not at equal effective size.

## (b) Error definitions

**b1 NOTE (sound).** The definitions are relative to population information, which is the right reference for the study's
questions, subject to a2 and a3:
- **False positive:** |dA| <= 1e-12 and any resolved conclusion, reported in total and by direction.
- **Power and wrong direction:** defined when dA != 0.
- **Descriptive sign error:** sign(d) != sign(dA). This counts d = 0 as an error; the reported P(d = 0) makes that visible.

**b2 RECOMMENDED. Separate theorems from findings.**
- **Nested partitions.** These are every REF comparison; IND pair - income and pair - occupation; RED pair - income and pair -
  occupation; COMP pair - income and pair - occupation. Here C_same and C_nested have d >= 0 in every sample, because any ranking
  of the coarse cells is a ranking of the fine cells with the same sample AUC. So P(d < 0) = 0, and "C favours the coarser output"
  holds with certainty. The findings are its size and its resolved rate.
- **Non-nested comparisons.** C's direction is an empirical result only in IND occupation - income, RED income - occupation and
  COMP occupation - income.
- **Bias signs.** E[C_same] >= A, because C_same >= B_same pointwise and E[B_same] = A. E[D_bank] <= A, because
  E[D_bank | tables] = pop(tables) <= A.

The report should state these and present the simulation as measuring magnitudes.

**b3 RECOMMENDED. Report M20 by direction.**
- M20 is two-sided (|d| >= 0.020). DRF1's only 0.020 rule applied to the exact estimator is directional: `rules.useful_optimistic`
  requires AUC_pair(bracket) - AUC_pair(X) >= 0.020, on point values with no SE.
- Report P(d >= +0.020) and P(d <= -0.020) separately. C's upper tail favours coarser releases; a lower tail for D favours finer
  ones.
- For dA != 0, add P(d - dA >= 0.020) and P(d - dA <= -0.020), post hoc. Sign errors are informative for RED's small dA; magnitude
  errors matter for COMP.

**b4 RECOMMENDED. C's power is bias-inflated where the finer view is also the more informative one** (RED pair - occupation, COMP
pair - single views). Report it beside the bias of d (mean d - dA, already computed), and do not cite it as an advantage of C.

## (c) Law families, refinement, clustered scenario, truth

**c1 NOTE (verified).** The stated relations hold in `laws.py`, and every LAWS.json check passes at 1e-12:
- REF children inherit their parent's q, with child shares independent of S.
- RED occupation = (g // 4, z), with z independent of (S, g).
- COMP's u and v are weight-centred, so each single view's posterior is pi.
- COMP occupation cells 14 and 15 merge, giving occupation support 15 and pair support 240. The internal-id value is 0.5844 against
  0.5829 merged (pi 0.5, uniform).
- Every estimator uses merged wire ids, and A is computed on merged masses.

**c2 NOTE (verified). The clustered cell's person-level marginal equals the REF law exactly.**
- P(S = 1) = 1/2 at every household position. For the second member, 0.8 x 1/2 + 0.2 x 1/2 = 1/2.
- g | S is the inverse CDF of w q / sum(w q), applied to a marginally uniform copula variable.
- Each child given its parent is the inverse CDF of the shares, applied to copula variables independent of S.
- Since sum(w q) = pi = 1/2, P(S = 1, g) = w_g q_g / (2 sum(w q)) = w_g q_g.

So the person-weighted marginal AUC is the right truth for person-level recovery estimators. The selftest's check (maximum
deviation below 0.003 on 300,000 people) is weak for cells of mass 1e-3 or less, but the analytic argument suffices.

**c3 NOTE (adequacy and its limits).** The families answer Q1-Q3 for these laws:
- **Nulls at fixed A:** support ratios of 4, 16 and 64 (REF, IND, RED pair - income), and one near-equal-support null (COMP
  occupation - income, 15 vs 16).
- **Finer view more informative:** RED pair - occupation; COMP pair - each single view.
- **Finer view less informative:** RED income - occupation.

The report should state these limits:
1. **Non-null support stops at 256.** The non-null comparisons reach only 256 (RED pair) and 240 (COMP pair) wires, while DRF1's
   twin pair had 727 occupied pair cells at n = 20,000 (FIT_V0.json). Wrong-direction rates at DRF1-like pair support are not
   simulated; only the null bias at 1,024 is.
2. **Small true differences in RED.** dA = 0.0081-0.0119, both for income - occupation and for pair - occupation, which is below
   DRF1's 0.020 scale. COMP's dA are 0.063-0.104. No family has dA != 0 at equal support.
3. **One signal strength and a reader-friendly emission.**
   - Only beta = 1 is used (A 0.63-0.68).
   - The emitted vectors carry the SEX logit in their first coordinate by construction, and refinement children lie within
     N(0, 0.15^2) of their parent.
   - This favours parametric bank readers and makes them nearly blind to refinement, in a way DRF1's task-posterior vectors need not
     be. Bank and class-limit findings are specific to this emission.
4. **One stylized clustered cell.** Second members have the opposite S 80% of the time, and outputs are positively correlated
   (rho 0.5). A household's influence values therefore tend to have opposite signs, so the household SE can come out smaller than
   the person SE. The cell cannot show how much ignoring clustering mis-states SEs in DRF1's households.

**c4 RECOMMENDED.** C's bias is driven by occupied cells and per-cell counts. Figure 1 plots nominal support, and in skewed sparse
cells this differs sharply from occupied support. Annotate or plot against the recorded `occupied_mean`.

## (d) Bank decomposition and class limits

**d1 NOTE (verified). The components add up and are well defined.**
- A - winner SCORE = (A - best pop) + (best pop - winner pop) - (winner SCORE - winner pop).
- The first two terms are >= 0 by construction. Every recipe score is a function of the wire, and the appended support rows enter no
  fit, orientation, selection or NEW8 design (`design_for` uses FIT rows only).
- Test variation has conditional mean 0, because SCORE is independent of FIT and VALIDATION.
- The winner's curse (VALIDATION orientation, maximum selection and noise) is a separate diagnostic, not a term of the sum.
- When the pair winner is an IGNORE candidate, its population AUC is correctly taken from the single-view law.

**d2 RECOMMENDED. Precision of the bank results.**
- BENCH.json projects 25.4 CPU-s per subset repetition, so the rule gives min(10, max(3, 59)) = 10 repetitions. The cap binds: any
  projection up to 150 CPU-s gives the same 10, so the timing could not steer the count.
- With 10 repetitions, report an MCSE (SD / sqrt 10) for every component. The bank table prints only means, plus an SD for test
  variation. Treat components within 2 MCSE of 0 as unresolved.
- No error-rate statement about the bank is possible. Bank readers store no influence values, so R2SE cannot be applied, and 10
  repetitions could not estimate a rate anyway.

**d3 RECOMMENDED. Class limits.**
1. **Saturated recipes.** LOOKUP, CATI01 and CATI1 (one-hot logistic) are saturated models of the wire, so their class limit is A
   by construction: the pseudo-count and penalty vanish relative to the data.
   - Their 30,000-person values measure residual finite-sample loss at 30,000, not a class limitation.
   - That loss is not trivial for sparse views: L256 skewed has cells of mass 9.1e-5, about 2.7 expected people.
   - Label these values "large-sample (30,000) value".
2. **Genuine class limits.** Only OUT_LR, LOG_LR and DEC_LR have one.
3. **No Monte Carlo error.** There is one draw per view.
4. **Recipe level only.** The refit splits recipe-level loss for these six recipes. It cannot split the bank-level A - best pop
   unless the best recipe is one of them.
5. **Split not computed.** `analyze.py` stores the large-fit values but computes no split. The report should give, per recipe:
   - class loss = A - pop_large;
   - finite-training loss = pop_large - mean pop at FIT 4,000, with an MCSE.

**d4 RECOMMENDED. Paired bank differences.**
- Q5 concerns bank comparisons between releases of different support. The subset has paired views on the same people: REF L256
  against L16, RED pair against income, IND pair against each single view. `analyze.py` summarizes views separately.
- Add, post hoc and labelled, the per-repetition paired differences of the winner's SCORE AUC and population AUC, as a mean +- MCSE
  over the 10 repetitions.
- Assert that the bank's LOOKUP recipe reproduces the grid's D_bank population and VALIDATION AUCs for the same cell and
  repetition. The people, split, keys and orientation are the same, so they should agree exactly.

## (e) Analysis code

**e1 NOTE. No defect found that mis-computes a declared number.**
- **`analyze.py`:**
  - `_stats`, `_rate`, the [d, se_unit, se_person] triplets and the 1e-12 null tolerance;
  - the false-positive, power, wrong-direction, sign-error and M20 logic, and None-to-NaN handling;
  - the person SE kept only for the clustered cell;
  - the table columns, and the Figure 2 components (bar 1 - bar 2 - bar 3 equals the gap).
- **`estimators.py` and `simulate.py`:**
  - the unit set of each estimator (study, SCORE or test sample; households in cell 48);
  - C_nested's people-weighted influence values;
  - D_nested's pooled law over wire x fold;
  - D_bank's oriented table, with unseen wires at the FIT rate;
  - the household sampler, and the folds and split drawn by unit.

**e2 REQUIRED. Figure 2's title, and "test-sample variation" for D_nested.** The title states a result written before any outcome:
"the gap = optimism + reader loss (+ zero-mean test variation)". For D_nested, value - pop_pooled is not zero-mean. Two mechanisms
push the pooled cross-fit AUC below its readers' population AUC:
- a person's label enters the tables that score the same wire's members in other folds. A positive raises, and a negative
  lowers, the score of the opposite-class people it is compared with, so within-wire pairs are misordered more often than half;
- with shrinkage, and with unseen wires scored at the training rate, folds richer in positives receive lower scores.

This is the known pessimism of pooled cross-validated AUC. I checked it on an invented independence law (my own code; 400
repetitions; not a grid law). There pop_pooled = 1/2 exactly.

| cells K | n | pooled - 1/2 (MCSE) |
|---|---|---|
| 16 | 2,000 | -0.0064 (0.0009) |
| 16 | 8,000 | -0.0031 (0.0005) |
| 256 and 1,024 | 2,000 and 8,000 | -0.0008 to -0.0039 |

Fold-wise AUCs were unbiased, all within 1.1 MCSE of 0. The term is small next to a gap of about 0.1. But at 500 repetitions the
MCSE is about 0.0005, so it will be detectable. The protocol's "test-sample variation = sample value - pop" is exact only for
D_bank.

Required:
1. Either regenerate Figure 2 with a neutral title, as a named post-results correction under PROTOCOL 8 that keeps the original, or
   quote the title in the report as a pre-registered expectation and show, per view, whether the measured mean is within 2 MCSE
   of 0.
2. In the report, describe `D_nested_pop_*__test_variation_value_minus_pop` as "cross-fit dependence + test variation". The bar
   legend already reads "test variation + cross-fit pooling".

**e3 RECOMMENDED.** Figure 1's suptitle also asserts results: the bias "grows with support and shrinks with n" and "LOOKUP is
biased low". The support and LOOKUP parts are theorems (b2), but the n-dependence is empirical. Use a neutral title, or check the
title against the results explicitly.

**e4 RECOMMENDED. Integrity guards.**
- `_load` concatenates every `out/sim_*.jsonl.gz` and `out/bank_*.jsonl.gz` without checking that each (cell, rep) or
  (bank_cell, rep) occurs once.
- `bank_class_limits` takes the lexicographically last `bank_*.run.json`.
- `out/` holds one label of each today (`sim_all`, `bank_all`), so the frozen run is unaffected. A re-run or partial file, however,
  would be double-counted silently.

Before reporting, assert unique keys and the declared counts: 500 repetitions in each of 49 grid cells and 10 in each of 5 bank
cells. The ordering of the bank 'winners' dict follows the string hash seed (A3); that is cosmetic.

**e5 RECOMMENDED. Compute and label these derived quantities in the report.** They are declared or implied, but the code leaves
them to the report:
- **D's RMSE against its own reader's population AUC.** It can be derived from the stored mean and SD of the test variation.
- **Cell 48 against cell 17.** Give the mean household/person SE ratio, the replicate-SD ratio, and the bias difference with
  combined MCSE.
- **The three-way LOOKUP split:**
  - A - pop_fold_mean: per-fold estimation loss;
  - pop_fold_mean - pop_pooled: fold-mixing calibration loss;
  - pop_pooled - D_nested: see e2.
- **The class-limit split:** see d3.

The Q3 decomposition is specific to DRF1's pairing: exact AUC* on folds of 4,000 against LOOKUP trained on 16,000. Do not present
its shares as properties of the estimators at a common size.

## (f) What the design can and cannot support

**f1 REQUIRED. Tie every DRF1-facing statement to how DRF1 actually used each estimator.**
1. **The twin's numbers map exactly onto EDG1's estimators,** for a fixed release at n = 20,000.
   - 0.6812 is the in-sample value on the true-SEX empirical MODEL_FIT law (`FIT_V0.json` `path_full[0].insample.empirical`,
     727 pair cells): C_same.
   - 0.7475 is the IDENTITY entry's nested held-out exact mean: C_nested.
   - 0.6303 is that entry's nested LOOKUP: D_nested.

   EDG1 can show how this pattern arises under known laws. It cannot say how much of the 0.117 gap is bias.
2. **DRF1 used the nested exact AUC* only in point comparisons.**
   - It entered `rules.useful_optimistic`'s nested mode (`held_out_survives`) and the unregistered stage-sensitivity note.
   - Both use a 0.020 point margin and no SE: "a diagnostic, not inference"; "reported, never a gate".
   - DRF1 never attached a fixed-ranking SE to an exact AUC*, so EDG1's R2SE rates for C describe a hypothetical practice. For
     DRF1 the relevant metrics are the mean d and the directional M20 (b3).
   - DRF1 already wrote that the nested comparison favours coarser releases. EDG1 can corroborate that caveat and give it a size;
     it does not discover it.
3. **DRF1's search objective is not simulated.**
   - DR_AUC and DR_MI optimized the propensity-estimated law. For the twin, its in-sample pair AUC* was 0.6505, against 0.6812 for
     the empirical law.
   - Its masses carry propensity error instead of Bernoulli label noise, so EDG1's C results do not transfer to it.
   - EDG1 cannot say whether the search was steered toward coarser or finer releases.
4. **DRF1's decision rests on the MODEL_SELECT reader bank.**
   - D2, D3, D4 and "no method claim" were judged with se_agg (a household bootstrap over SCORE with the readers fixed, plus the
     version and seed spread) and are stated as fitted-reader recovery.
   - EDG1 implements neither se_agg nor bank error rates.
   - D_bank and E can show whether fitted-reader loss depends on support. That bears, for example, on the resolved +0.034 against
     the coarsening bracket, a fitted-reader comparison between releases of different support.
   - They cannot re-decide any DRF1 label, and nothing in EDG1 can establish a numerical error in DRF1's recorded decision. The
     no-method conclusion therefore stands by construction.
5. **Release optimization is outside EDG1's scope.**
   - EDG1's laws are fixed, whereas DRF1's releases were optimized on MODEL_FIT and refitted per outer fold.
   - Optimizing on an in-sample objective biases that objective downward, opposite to C's upward support bias. OPT_L's in-sample
     "useful" result carries both biases; EDG1 measures only the second.

**f2 NOTE. What the design can and cannot support.**

Can support, for these 16 laws, three sizes and one clustered law:
- the sign of C's support bias (by theorem on nested partitions) and its size (by simulation);
- naive-R2SE resolved rates for C and D_nested;
- the split of C_nested - D_nested into optimism, LOOKUP loss and a cross-fit/test term;
- the bank's loss components at n = 8,000;
- a sized version of DRF1's existing caveat.

Cannot support:
- DRF1's actual bias, whose law is unknown;
- the propensity-law objective;
- the optimizer's curse on releases;
- DRF1's se_agg calibration;
- real household structure;
- non-null behaviour at supports above 256;
- any bias-corrected estimator (none is simulated);
- any population guarantee.

On naming: for a fixed release nothing is held out in C_nested. It is the empirical optimum on a fifth of the sample, so its bias
exceeds C_same's. "Held-out" refers only to DRF1 refitting the release on the other folds. B_indep needs the known law, so it is a
benchmark, not a usable estimator.

**f3 RECOMMENDED. Test whether DRF1's agreement was expected.** DRF1 recorded one agreement: MODEL_FIT nested LOOKUP 0.6028 /
0.6085 (trained on about 16,000) against the MODEL_SELECT bank 0.6049 / 0.6104 (FIT about 4,000). EDG1 has a direct analogue:
compare D_nested at n = 20,000 with the bank winner's SCORE AUC at n = 8,000 for the same law. The grid cells are 5 / 17 / 29 /
38 / 35 against bank cells 4 / 16 / 28 / 37 / 34. This shows whether close agreement is expected or coincidental. Label it post
hoc.

## Classification summary

| class | items |
|---|---|
| REQUIRED | a2, a3, a4, e2, f1 |
| RECOMMENDED | b2, b3, b4, c4, d2, d3, d4, e3, e4, e5, f3 |
| NOTE | a1, a5, b1, c1, c2, c3, d1, e1, f2 |
