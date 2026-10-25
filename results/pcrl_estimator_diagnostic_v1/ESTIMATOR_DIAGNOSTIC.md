# EDG1: what PCRL's recovery estimators measure (synthetic diagnostic)

Campaign `pcrl_estimator_diagnostic_v1`, 2026-10-25. Registration c169705c2 (17:18:28Z, before any outcome); outcomes 88283f433.
This is a measurement diagnostic on synthetic known laws, not a release method.

**Bottom line.**
- **Same-sample empirical optimal AUC (DRF1's exact AUC* on a sample) is biased upward.** The bias grows with the number of
  distinct released outputs and shrinks with sample size. DRF1's nested held-out version (five folds of about 4,000) is 2.0–4.1
  times more biased at n = 20,000 (views with a same-sample bias above 0.0005).
- **Held-out LOOKUP is biased downward.**
  - Sparse cells cause estimation loss.
  - When cross-fitted and pooled, there is also a cross-fit dependence term. Under independence that term is the whole of the
    downward bias.
- **Comparisons across output supports fail in opposite directions.** When true recovery is identical:
  - the empirical optimum declares the finer output more recoverable in every repetition;
  - the held-out LOOKUP declares it less recoverable in 14–100% of repetitions, more often the finer the refinement. For the single
    held-out split, the rate rises with n.
- **Calibration.**
  - For population information, only the known-law score, evaluated on independent data, behaves nominally.
  - The single-split held-out readers are calibrated for their own fitted target, not for population information.
- **DRF1.** EDG1 did not recompute DRF1's decision and could not establish any numerical error in it; the no-method conclusion is
  unchanged. One unregistered DRF1 reading should be re-read: its "not useful on the nested estimator" verdict for OPT_L is
  confounded against OPT_L, so it is not evidence against it (answer 5).
- **No new mechanism is identified.**

## Answers

### 1. Does same-sample empirical optimal AUC favour coarser outputs when population recovery is unchanged?

**Yes, systematically.**
- **Irrelevant refinements** (REF: each output category split into children independently of S, so the population optimum A is
  identical at every level).
  - In all 12 independent REF cells and all three refinement comparisons, both C_same and C_nested scored the refined output
    higher in 100% of the 500 repetitions.
  - The declared rule R2SE (d ± 2·se) called that spurious difference resolved in 99.6–100% (C_same; 99.2% in the clustered cell)
    and 100% (C_nested).
- **Linking an output that adds nothing** (RED: S is independent of the occupation view given income).
  - C_same reported pair > income by +0.011 to +0.111 and C_nested by +0.045 to +0.229.
  - R2SE false-positive rate: 100% in every RED cell.
- **Under independence** (IND), C_same reported pair > income by +0.093 to +0.355, with R2SE false positives of 100%.
- **Where the bias cancels.** It is a function of support and sample, not of information. The one null comparison between outputs
  of equal support (COMP occupation (15) vs income (16), both A = 1/2) gave C_same false positives of 0–1.0%.
- **Held-out LOOKUP fails the other way.** On the same REF comparisons D_nested and D_bank scored the refined output lower:
  P(d > 0) at most 0.29 and 0.11; R2SE false positives 14–100% and 26–100%. All but at most 0.8 percentage points of them were in
  the "refined is less recoverable" direction.
  - For D_bank these rates measure how often a valid conclusion about the fitted readers misstates population information.
  - They are not a miscalibration of R2SE. Judged against its own target, the paired difference of the two fitted readers'
    population AUCs, R2SE covered that target in 94.8–95.8% of repetitions in every family and comparison (post hoc,
    `POST_HOC.json`).
  - In the exact nulls, where every score has population AUC 1/2 (IND, and COMP occupation − income), D_bank resolved 4.2–5.2%
    pooled over cells (2.4–7.2% per cell).
- **R2SE is naive for D_nested too.** The pooled cross-fitted AUC mixes five tables, and each person's label helped fit the tables
  that score the others. Its "rankings held fixed" SE ignores both, giving 13–20% resolved under the exact nulls.
- **The known-law score is calibrated.** B_indep's bias was within ±0.0014 of A everywhere. Post hoc, d ± 2·se covered the true
  difference in 95.2–95.8% of repetitions of the non-null comparisons, for both B_indep and B_same.
  - B's null false-positive rates (0–7.0%) cannot show calibration: in those comparisons the two known-score rankings are identical,
    or differ only by float rounding of tied posteriors (review item a4).
- **Theorem versus finding.**
  - On nested partitions (every REF comparison, and every pair-versus-single comparison), C's difference is ≥ 0 in every sample by
    construction: a ranking of the coarse cells is a ranking of the fine cells with the same sample AUC.
  - E[C_same] ≥ A, because C_same ≥ B_same on every sample and E[B_same] = A.
  - E[D_bank | fitted table] ≤ A.
  - The simulation measures the sizes, not these signs.

### 2. How large can that effect be under these laws and sample sizes?

**Large relative to every DRF1 decision margin.** Bias of the empirical optimum against A (mean over 500 repetitions, MCSE at most
0.0005; `TABLES.md` 1):

| setting | support 16 | 64 | 256 | 1,024 |
|---|---|---|---|---|
| C_same, n = 20,000 (REF / IND, pi = 0.5, skewed) | +0.0008 / +0.0136 | +0.0038 / +0.0281 | +0.0116 / – | +0.0334 / +0.1067 |
| C_nested, folds of about 4,000 (same cells) | +0.0026 / +0.0308 | +0.0130 / +0.0629 | +0.0412 / – | +0.0985 / +0.2177 |
| C_same, n = 2,000 (same cells) | +0.0040 / +0.0437 | +0.0224 / +0.0888 | +0.0675 / – | +0.1412 / +0.2789 |

- **Paired coarse-versus-refined error with identical true recovery** (REF L1024 − L16, all four pi × occupancy cells):
  - C_same +0.033 to +0.056 at n = 20,000; +0.063 to +0.117 at n = 8,000; +0.137 to +0.244 at n = 2,000.
  - C_nested +0.096 to +0.182 at n = 20,000.
  - For comparison, DRF1's material target was 0.020 and its decision differences were 0.001–0.04.
- **The largest values are near the ceiling.** Independence with 1,024 outputs at n = 2,000 gave C_same up to 0.91 and C_nested
  up to 0.99, against a true value of 0.5.
- **Imbalanced class frequency** (pi = 0.2) raised C_same's bias by 9–85% (median 26%). That is across the family × occupancy × n
  × view pairs whose pi = 0.5 bias exceeds 0.002 (statistical review b3).
- **Skewed occupancy** lowered it at the largest support, because fewer cells are occupied.

### 3. How much of the gap to LOOKUP comes from the estimator, and how much from the fitted reader?

At DRF1-like sizes (n = 20,000; nested folds of about 4,000; pi = 0.5, skewed occupancy; Figure 2) the mean gap
`C_nested − D_nested` splits exactly into three parts:
- the nested estimator's optimism, C_nested − A;
- the fitted LOOKUP's estimation loss, A − pop, where pop is the exact population AUC of each fitted ranking;
- test-sample variation plus cross-fit pooling, D_nested − pop.

| law, view (support) | gap | optimism | LOOKUP estimation loss | test variation + pooling |
|---|---|---|---|---|
| IND pair (1,024) | 0.218 | 0.218 (100%) | 0.000 | −0.000 |
| REF L1024 (1,024) | 0.122 | 0.099 (81%) | 0.023 (19%) | −0.001 |
| REF L256 (256) | 0.050 | 0.041 (82%) | 0.008 (16%) | −0.001 |
| RED pair (256) | 0.061 | 0.050 (82%) | 0.010 (17%) | −0.001 |
| COMP pair (240) | 0.091 | 0.074 (81%) | 0.016 (18%) | −0.001 |

- **Large supports (240–1,024).** About four fifths of the gap is optimism of the nested exact estimator, and about one fifth is the
  fitted reader's estimation loss.
- **Support 16.** The gap is small and the cross-fit term is close to half of it. REF L16: optimism 0.0026, loss 0.0002, cross-fit
  0.0023, gap 0.0051. RED income: 0.0031, 0.0001, 0.0023, gap 0.0055.
- **Test variation is mean-zero for a single held-out split.** D_bank's mean was −0.0003 to +0.0005 in these cells and −0.0029 to
  +0.0027 over all cells. It was beyond 2 MCSE in 7 of 160 cell-views, about what chance gives.
- **D_nested's third term is not zero-mean.** It is cross-fit dependence plus test variation: a held-out person's label helped fit
  the tables that score the people they are compared with (the known pessimism of pooled cross-validated AUC).
  - It was negative in all 160 cell-views (−0.00004 to −0.011).
  - Post hoc, it was more than 2 MCSE below zero in 12 of the 13 Figure-2 views (−0.0005 to −0.0027; IND pair −0.0002 ± 0.0003).
  - Figure 2's title ("+ zero-mean test variation") was written before the outcomes and is wrong for D_nested. The figure is kept
    as registered and its legend names the pooling.
- Heterogeneity between the fold rules is negligible here. Their pooled and per-fold population AUCs agree within 0.0001.
- **The split is specific to DRF1's pairing**: exact AUC* on folds of 4,000 against LOOKUP trained on 16,000. Its shares are not
  properties of the estimators at a common size.
- DRF1's twin gap (0.7475 − 0.6303 = 0.117, 727 occupied pair cells, folds of about 4,000) lies inside the synthetic range above.
  That is a consistency observation only. It does not measure DRF1's own split, because its population law is unknown (section 6).

### 4. Which estimator is appropriate for which future claim?

| future claim | appropriate measurement in these settings | not appropriate |
|---|---|---|
| "this fitted reader recovers S at AUC x on new people" | a reader fitted on FIT, oriented / selected on VALIDATION, scored once on separate SCORE data (D_bank, E): unbiased for that reader (test variation mean about 0); R2SE covered the fitted readers' paired population difference in 94.8–95.8% (post hoc) | pooled cross-fitted AUC with a naive SE (D_nested resolved 13–20% under the exact nulls) |
| "the release leaks at least x" (a recoverability lower bound) | the same held-out readers. Their population AUC cannot exceed A (a theorem: pop ≤ A for any score of the wire; bank winners were 0.000–0.015 below A). A single SCORE value can exceed A through test variation (SD about 0.009 at SCORE 2,400), so an "at least" statement needs the lower confidence limit (value − 2·se) | the empirical optimum (biased upward) |
| "the release's optimal recovery is about x" | no estimator here is unbiased. The empirical optimum is an upward-biased value whose bias depends on support and n; report it only with its support, n and a same-size null calibration | treating C as a population value or guarantee |
| "release X recovers less than release Y" when their observable supports differ | neither C differences (favour the coarser) nor held-out reader differences (favour the finer) are valid alone. A direction is credible here only if it is shown by the estimator whose bias works against it, and survives an irrelevant-refinement null at the same sizes | C or nested C as an objective or selector across supports |
| equal-support comparisons | in the one matched-occupancy null (COMP occupation (15) against income (16)), C's support bias cancelled: false positives 0–1.4%, below nominal. Equal support is not shown to be sufficient in general (the bias depends on cell counts and signal) | — |

**Reader bank (E; 6 scenario cells × 10 repetitions at n = 8,000).** Rows marked † are post hoc (`POST_HOC.json`); the others
come from the frozen analysis.

| finding | value |
|---|---|
| A − best population AUC in the bank | 0.0012–0.0121; resolved at 2 MCSE wherever A > 1/2 † |
| selection loss (best pop − winner pop) | 0.0004–0.0034; resolved in 3 of 9 informative views † |
| winner's curse (VALIDATION − winner pop) | up to +0.029 (independence, pair view; resolved in the null and pi = 0.2 views) |
| test variation, mean ± MCSE | −0.0064 ± 0.0044 to +0.0035 ± 0.0043; none resolved |
| finer output at identical recovery (REF, paired over 10 repetitions) † | winner pop L256 − L16 = −0.0083 ± 0.0013; winner SCORE −0.0074 ± 0.0023 (A identical) |
| genuine class limits on COMP pair (OUT_LR, LOG_LR, DEC_LR refitted on 30,000) † | 0.4999–0.5049, i.e. class loss 0.078–0.083 |
| saturated recipes on COMP pair (LOOKUP / CATI01 at 30,000; large-sample values, not class limits) † | 0.573 / 0.577 |
| COMP pair, MLP recipes on 4,000 | 0.567–0.570 |

- **Saturated recipes.** For LOOKUP and CATI (saturated in the wire) the 30,000-person refit measures residual finite-sample loss,
  not a class limit. LOOKUP's finite-training loss from 4,000 to 30,000 people was 0.021–0.030 on the sparse REF L256, RED pair and
  COMP pair views.
- **Model class.** On COMP's interaction law the LR-type recipes cannot see the pair signal at all (A = 0.583). A lower reader AUC
  was a model-class limitation, not less information.
- **The bank LOOKUP recipe reproduced the grid's D_bank exactly** (population and VALIDATION AUC) in all 140 shared views (post
  hoc). That cross-checks the two runs.
- **Ignore-one candidates protect only partly.**
  - In RED at pi = 0.5 the pair winner was an income-only ignore-one candidate in 10 of 10 repetitions. That is the protection the
    bank's ignore-one candidates give against pair readers' extra estimation loss.
  - At pi = 0.2 the pair winners were an income-only candidate in 4 of 10, pair readers in 5 and an occupation-only candidate in 1.
    Pair selection loss there was 0.0034 ± 0.0016, resolved at 2 MCSE.

### 5. Does this diagnostic justify reconsidering any interpretation of DRF1?

**One unregistered reading should be re-read: DRF1's nested-mode "not useful" verdict for OPT_L is confounded against OPT_L by
support-dependent optimism, so it is not evidence against OPT_L.** This sizes and sharpens a caveat DRF1 already wrote.
- EDG1 did not recompute DRF1's decision and could not establish any numerical error in it.
- It implements neither DRF1's se_agg nor its bank error rates, so it cannot re-decide any DRF1 label.
- The no-method conclusion, the higher task cost, the unmatched comparisons and the failed strong-reference conditions stand as
  separate evidence.

**How DRF1 used each estimator.** For a fixed release at n = 20,000, the twin's three numbers are exactly EDG1's estimators:

| twin value | source | EDG1 estimator |
|---|---|---|
| 0.6812 | in-sample exact pair AUC* of the true-SEX MODEL_FIT law (727 occupied pair cells) | C_same |
| 0.7475 | the nested held-out exact mean | C_nested |
| 0.6303 | the nested LOOKUP | D_nested |

- EDG1 shows how that ordering arises under known laws (C_nested > C_same > A > D_nested). It cannot say how much of the 0.117 gap
  is bias.
- DRF1 used the exact estimator only in point comparisons with a 0.020 margin and no SE: `rules.useful_optimistic` and the
  stage-sensitivity note.
- EDG1's R2SE rates for C therefore describe a hypothetical practice. For DRF1 the relevant numbers are the mean difference and the
  directional 0.020 margin (post hoc).
  - In the null comparisons, C_same's difference reached +0.020 in 20% (L64 − L16), 61% (L256 − L16) and 100% (L1024 − L16) of
    all REF repetitions.
  - C_nested reached +0.020 in 60%, 100% and 100%. It was never at or below −0.020 on a nested partition.

**The unregistered nested-mode reading** (DRF1 answer 2, "not useful on the nested estimator", and its stage-sensitivity note).
- **Supports.** DRF1's published records (`dr/FIT_V{0,1}.json`, `dr/OPTIMISTIC_V{0,1}.json`) give both supports. The statistical
  review found this; the coordinator re-checked it.
  - The matched coarsening point is the nested path step whose spend first reaches OPT_L's (step 67 / 68). It has 40 / 35 occupied
    pair cells in-sample (5 income and 8 / 7 occupation atoms).
  - OPT_L had 374 / 398.
- **Size of the confound.** The comparison set a release with roughly ten times the pair support against the bracket point. In
  EDG1's four independent REF cells at n = 20,000 (folds of 4,000), C_nested's optimism was:
  - 0.003–0.020 at 16–64 wires;
  - 0.04–0.19 at 256–1,024 wires.
- **Reading.** The confound runs against OPT_L, by an amount of the same order as the 0.044 / 0.041 shortfall. The nested "not
  useful" reading is therefore not evidence against OPT_L.
- **What EDG1 cannot say.** Whether OPT_L would be useful under an unbiased held-out measure, or how large DRF1's own bias was.
- OPT_L also carries a second bias EDG1 does not simulate. A release chosen by minimizing an in-sample value has that value biased
  low.
- Its nested spends of 2.6–4.0 times the budgets remain separate, unaffected evidence that it was not a usable point.

**The registered decisions** (D2, D3, D4) used bank readers fitted on FIT, selected on VALIDATION and scored once on SCORE.
- They are stated as fitted-reader recovery, and EDG1 found that kind of evaluation unbiased for the fitted reader.
- Those readers carry a support-dependent estimation loss. In EDG1's paired REF subset, the bank winner's population AUC was
  0.0083 ± 0.0013 lower at 256 than at 16 cells for identical A (post hoc). D_bank's mean difference for L256 − L16 at n = 8,000 was
  −0.021 (π = 0.5, skewed).
- **The resolved +0.034 against the SEX-blind coarsening bracket** is a fitted-reader comparison between releases of different
  support.
  - DR_AUC_R1 had 456 / 564 occupied pair cells in-sample; the upper bracket DR_CTRL_HI_R1 had 70 / 68.
  - EDG1's fitted-reader loss grew with support at identical A, so this confound plausibly favours the finer candidate, DR.
  - The two releases are not nested partitions, so this direction is an empirical tendency, not a theorem, and its size is not
    quantified.
  - The DRF1 labels are unchanged. D2 was NOT_SEPARATED because of DR_MI, and the bracket separation was already "primary cost only".
- **D2 (DR_AUC_R1 against DR_MI_R1).** The pair supports were 456 against 445 in version 0 but 564 against 485 in version 1 (16%
  apart). The support effect is not quantified for DRF1's law in either version.

**The R1 agreement.** DRF1's nested LOOKUP prediction agreed with the MODEL_SELECT bank within 0.002.
- In EDG1's analogue (post hoc), the mean D_nested at n = 20,000 and the mean bank winner's SCORE AUC at n = 8,000 differed by
  −0.007 to +0.005 across the 14 subset views.
- A single realization of that difference has an SD of about 0.010. Agreement within 0.002 is therefore consistent with these laws
  but not expected to be that close. It is not a validation of either number.

**Not simulated.** DRF1's search objective was the exact AUC* of the propensity-estimated law: 0.6505 for the twin in-sample,
against 0.6812 for the empirical law. Its masses carry propensity error, not label noise. EDG1 cannot say whether that objective
steered the search toward coarser or finer releases.

### Implication for method design or selection

**A same-sample exact-AUC objective or selector rewards merging output categories even when merging removes no information.**
- In these laws, merging 1,024 irrelevant sub-categories into 16 "reduced" same-sample recovery by 0.033–0.244 and nested recovery
  by 0.096–0.286 with no true change.
- In settings like these laws and sizes, any future recovery objective or selection rule should therefore pass an
  irrelevant-refinement null calibration at the sample sizes and supports in use before its improvements are credited. Its claimed
  improvements should also be confirmed by independently scored readers whose support bias runs against the claim.
- No new release mechanism was identified. No method campaign is started.

## What was measured

| label | quantity | DRF1 counterpart |
|---|---|---|
| A | population optimal-reader AUC of the known law (likelihood-ratio ranking over merged observable wires) | none (unknown for real data) |
| B_indep | known population-optimal score, sample AUC on an independent test sample | none |
| C_same | empirical optimal AUC: cell posteriors ranked and evaluated on the same sample (`drf.solver.auc_star`) | in-sample exact AUC* (n = 20,000) |
| C_nested | each of 5 folds' own empirical optimal AUC, people-weighted mean | nested held-out exact AUC* (`heldout_exact_auc_mean`) |
| D_nested | 5-fold cross-fitted LOOKUP (Beta shrinkage a = 1, unseen → training rate), AUC pooled over people | nested held-out LOOKUP |
| D_bank | bank LOOKUP recipe (exact wire bytes, pseudo-count 2), FIT / VALIDATION orientation / SCORE once | bank LOOKUP on MODEL_SELECT |
| E | 16-recipe bank, VALIDATION selection, SCORE once (bounded subset) | the DRF1 reader grid |

For every fitted LOOKUP and bank recipe the exact population AUC of its fitted ranking under the known law is computed. That
separates estimation loss (A − pop) from the rest (sample − pop). For D_bank the rest is pure test variation; for D_nested it is
cross-fit dependence plus test variation (answer 3). B_same, the known score on the study sample, is a diagnostic: C_same −
B_same ≥ 0 on every sample.

**Design** (frozen in `PROTOCOL.md`, `GRID.json`, `LAWS.json`).
- **Families.**
  - IND: S independent of the outputs.
  - REF: one view refined 16 → 64 → 256 → 1,024, children independent of S given the parent.
  - RED: occupation adds nothing given income.
  - COMP: each view alone has A = 1/2; the pair is informative; two occupation cells emit identical wires and are merged.
  - All population relations are verified in LAWS.json.
- **Factors.**
  - Class frequency 0.5 and 0.2; ties are built in (8 posterior levels shared by 16 atoms).
  - Uniform and Zipf-skewed occupancy.
  - n = 2,000, 8,000 (MODEL_SELECT-sized) and 20,000 (MODEL_FIT-sized).
- **Repetitions.** 500 independently seeded repetitions per cell; the same people for every view in a repetition.
- **Clustered cell.** One cell with explicit synthetic households (mean size 1.52 against DRF1's 1.50) whose person-level marginal
  law equals the independent one. The full reader bank ran on 6 prospectively fixed scenario cells × 10 repetitions.
- **Computed AUCs.** None was undefined: all 24,500 repetitions had both classes in every scoring sample, with 0 undefined values
  recorded.

## Further results

**Known-score estimator.** B_indep's bias was within ±0.0014 of A in every cell and view. The largest was 2.66 MCSE; 4 of 160
views were beyond 2 MCSE, about what chance gives.

**Sample size and class frequency.**
- The empirical optimum's bias falls roughly as 1/sqrt(n). IND income C_same was +0.044 / +0.022 / +0.014 at n = 2,000 / 8,000 /
  20,000.
- It is larger at pi = 0.2: REF L1024 at n = 20,000, skewed, +0.049 against +0.033.

**Unseen cells.** They drive LOOKUP's loss at small n. Up to 37.7% of SCORE people had a wire unseen in FIT (IND pair, n = 2,000,
uniform), and all of them received the training rate.

**Wrong directional conclusions when the true difference is not zero** (RED, true income − occupation = +0.008 to +0.012).

| estimator | result |
|---|---|
| C_nested | wrong direction, resolved, in 96.6–100% of repetitions at every n (finer occupation output looks more recoverable) |
| C_same | wrong in up to 78.2% (n = 2,000); right but rarely resolved at n = 20,000 |
| D_nested and D_bank | wrong in at most 0.2% on this comparison |
| B_indep | never wrong; power 0.32–1.00 |

For RED pair − occupation (true +0.008 to +0.012) the held-out readers erred instead: D_nested was wrong in 2.8–96.4% and D_bank in
6.4–88.8%. In COMP, every estimator's mean difference was positive for the pair's real extra information. D_bank's point estimate
nonetheless had the wrong sign in up to 42.8% of repetitions at n = 2,000. Its power was only 5–9% at n = 2,000 and 26–71%
at n = 8,000.

**Clustered households** (cell 48 against matched cell 17).
- Biases agree within 0.0003 and replicate SDs within 0.0005.
- R2SE rates with household-clustered and person-level SEs agree within 0.006. For example, B_indep L64 − L16 false positives were
  0.040 against 0.036.
- With this household structure, clustering did not change any conclusion. Larger households or stronger within-household
  dependence were not tested.

**Implementation notes** (disclosed, no result recomputed).
- **Study defect D1: B splits exact ties.** Found by the coordinator at about 17:26Z, then independently confirmed and quantified by
  the verifier (`roles/verifier/HANDBACK_1.md`). It is not corrected, and no result was recomputed.
  - **Cause.** B scores people by the law's float posterior. Where posteriors are equal in exact arithmetic they can differ by at
    most 2.2e-16, so B orders those wires by rounding noise instead of scoring them 1/2.
  - **Affected laws.** Every skewed law, IND|0.2, and every COMP law, whose single-view posteriors all equal pi up to rounding.
  - **Affected comparisons.** In 15 of the 32 null comparisons, B's paired difference is therefore not identically 0, as it would be
    with exact ties. B's null false-positive and M20 rates there (M20 up to 30% for B_indep, in IND|0.2|skewed and the COMP single
    views at n = 2,000) are rounding artifacts. In those views B is effectively an arbitrary fixed ranking.
  - **Why B stays valid.** Any order of population-tied wires is population-optimal (to about 1e-15). So B remains unbiased for A,
    its bias column is valid, and its non-null coverage evidence (95.2–95.8%) stands: E[d] = dA, and the SE is that of the ranking
    actually scored.
  - **Not affected.** C and D, which use integer counts and therefore exact ties, and A, which is exact to about 1e-15.
- Figure 2's title says "zero-mean test variation". Its grey bars include the cross-fit pooling offset described in answer 3.

## Recommended measurement procedure (within these settings only)

1. Report the empirical optimal AUC only with its observable support and n, next to a same-size irrelevant-refinement null (this
   study's REF law resized). Do not use it, or its nested version, to compare releases with different supports, or as an objective
   or selector.
2. For attack evidence, use readers fitted on FIT, oriented and selected on VALIDATION, and scored once on separate SCORE data. Keep
   the ignore-one candidates. Read the result as that reader's recovery, a lower-biased value for the release.
3. For release comparisons, require the claimed direction from the estimator whose support bias opposes it:
   - a coarser release claimed less recoverable must be shown by held-out readers;
   - a finer release claimed less recoverable must be shown by the empirical optimum.

   **Why.** False confirmation is rare under these nulls. Held-out readers resolved "refined more recoverable" in at most 0.8%
   (D_nested) and 0.4% (D_bank) of REF null repetitions, and C cannot resolve "refined less recoverable" on nested partitions (a
   theorem).

   **Cost.**
   - Power is low. On RED income − occupation, C_nested had 0% power and C_same 3–21% at n = 20,000. On COMP at n = 2,000, D_bank
     had 5–9%.
   - C's opposing bias is guaranteed only on nested partitions.
4. Where a ranking is fitted on the evaluated data, SEs need calibration (for example by repetition). This was not tested here.
   Pooled cross-fitted AUC with a naive SE resolved 13–20% under the exact nulls.

None of these is an unbiased estimator or a guarantee. They are conservative rules supported by these laws and sizes.

## Limitations

- **Scope.** Synthetic finite laws only (four families, supports 15–1,024, n 2,000–20,000, pi 0.5 / 0.2).
- **Bank subset.** 10 repetitions per cell, so its decomposition means carry MCSEs of about 0.0001–0.006. No error-rate statement
  about the bank is possible.
- **Non-null comparisons stop at support 256.** They reach only 256 (RED pair) and 240 (COMP pair) wires, against the twin's 727
  pair cells. Wrong-direction rates at DRF1-like pair support are not simulated; only the null bias at 1,024 is.
- **Small or large true differences only.** RED's true differences (0.008–0.012) are below DRF1's 0.020 scale, and COMP's
  (0.063–0.104) are well above it. No family has a true difference at equal support.
- **C's power is inflated by its bias** wherever the finer view is also the more informative one (RED pair − occupation, COMP pair −
  single views). Its 100% power there is not an advantage.
- **One signal strength (beta = 1) and a reader-friendly emission.** The emitted vectors carry the S logit in their first
  coordinate, and refinement children sit within N(0, 0.15²) of their parent. That favours parametric readers and makes them nearly
  blind to refinement, so the bank and class findings are specific to this emission.
- **One stylized clustered cell.** It cannot show how much ignoring clustering mis-states SEs in DRF1's households.
- **Class limits.** Genuine class limits were separated only for OUT_LR, LOG_LR and DEC_LR (one 30,000-person refit per view, no
  Monte Carlo error). LOOKUP and CATI are saturated, so their refit measures residual finite-sample loss.
- **Support axis.** Figure 1 plots nominal support. In sparse skewed cells the occupied support is lower (for example 495 of 1,024
  wires occupied at n = 2,000), and the empirical optimum's bias follows occupied cells and per-cell counts.
- **Figure 1's title** was written before the outcomes.
  - "Grows with support" is a theorem for nested partitions only; the left panel also joins non-nested IND views.
  - "LOOKUP is biased low" is a theorem for D_nested's population part (pop ≤ A), and for D_bank in expectation. D_nested's
    cross-fit part is an empirical finding.
  - "Shrinks with n" is an empirical finding that the grid confirms.
- **Not simulated.**
  - DRF1's propensity-weighted objective, and its optimizer's selection on noise. A release chosen by minimizing a same-sample value
    has that value additionally biased low.
  - Households larger than DRF1's, and real census structure.
  - Any bias-corrected estimator. No population guarantee follows from any result here.

## Verification, review and amendments

**Independent verifier** (`roles/verifier/`; own exact-rational, brute-force AUC code; own laws built from the specification).
- **Round 1.** Six of seven checks pass:
  - independence gives exactly 1/2;
  - the known-score AUC equals pair enumeration;
  - merging and A's tie handling are correct;
  - irrelevant refinement preserves A exactly;
  - the RED / COMP relations hold;
  - no test label reaches a fit, orientation or selection (static review plus label-perturbation tests);
  - provenance hashes match.
- **Study defect D1** (B's rounding ties) was confirmed and quantified. It is disclosed above and not corrected.
- **The verifier's own two defects** were kept with corrected runs: an over-strict exact-rational idealization of the DRF1 LOOKUP,
  and a summary range estimated by eye.
- **Round 2** (reproducing summaries from the saved outputs and re-running repetitions from their seeds), with its own code:
  - **Reproduced.** 14 view entries, 10 comparisons, 4 bank entries and 2 post hoc entries, all to 1e-12.
  - **Re-runs.** Three repetitions re-run from their seeds were byte-identical to the saved rows.
  - **Report numbers.** 62 of 64 items (159 numbers) in the report at 523f0bec6 matched. The two last-digit rounding slips
    (95.9% → 95.8%; COMP gap and optimism 0.092 / 0.075 → 0.091 / 0.074) are corrected.
  - **Its own defect.** A precision criterion in its first number check was kept, with a corrected run beside it.
  - **Later additions not re-checked by the verifier.** Numbers added after 523f0bec6 in response to review round 2 were not
    re-checked by it:
    - the DRF1 support counts, which the coordinator re-checked from DRF1's published records;
    - the pi = 0.2 ratio range and the nested-to-same-sample ratio, which the coordinator re-computed from SUMMARY.json;
    - the support-16 decomposition and D_bank's 42.8% sign errors, taken from SUMMARY.json.

**Statistical reviewer** (`roles/statistical_reviewer/`).
- **Round 1, before seeing outcomes.** The design was judged sound, with no analysis defect, and 5 REQUIRED items:
  - a2: D_bank's estimand is the fitted readers;
  - a3: D_nested's R2SE is naive;
  - a4: B's null rates cannot show calibration;
  - e2: D_nested's third term is not zero-mean;
  - f1: DRF1-facing statements are tied to DRF1's actual use of each estimator.
- All five are addressed in this report, with labelled post hoc summaries (`POST_HOC.json`).
- **Round 2** (the draft against the results): all five round-1 items resolved; more than 45 numbers checked.
  - **Three new REQUIRED corrections**, all applied:
    - c1: OPT_L's matched bracket support is published (40 / 35 pair cells against 374 / 398), so the confound runs against OPT_L;
    - b1: the single-split LOOKUP false-positive rate rises with n;
    - d1: the RED ignore-one result holds at pi = 0.5 only.
  - **14 RECOMMENDED items and the notes**, all applied: scoping, post hoc labels, version-specific DRF1 supports, the
    opposing-bias rule's support and cost.
  - Neither round changed DRF1's no-method conclusion or required a re-run.

**Amendments** (`PROTOCOL.md` section 9, append-only).

| amendment | content |
|---|---|
| A1 | custody tool, added before outcomes |
| A2 | publication tool, added before outcomes |
| A3 | tooling fixes after custody rehearsal 1 failed (failure record kept) |
| A4 | the post hoc summaries |

No registered estimator, law, seed, grid or analysis file changed after registration.

## Resources, data access and custody

**Time.** Start 2026-10-25T16:55:33Z; budget 2 hours. Registration at 17:18:28Z (23 min); runs finished at 17:31:10Z. The final
time is in the closing message and `custody/` records.

**CPU** (budget 2 CPU-hours; no paid compute; one numerical thread per process; at most two heavy processes, the grid and the bank,
both run once).

| item | CPU-s | basis |
|---|---|---|
| grid, 49 cells × 500 | 743.5 | logged |
| bank subset with class-limit refits | 348.7 | logged |
| verifier rounds 1–2 | about 56 | logged in its outputs |
| statistical reviewer | under 13 | its own report |
| coordinator: selftests, smoke tests, timing benchmark, analysis, post hoc, custody rehearsals, helper scripts | about 200 | estimated; not all logged |
| **total** | **about 1,360 (about 0.38 CPU-hours)** | |

**Agents.** Three live agents including the coordinator: the independent verifier and the statistical reviewer. There was no
recursive spawning.

**Data access.**
- **Read.** Only synthetic data generated here, plus:
  - the published DRF1 code at c4896479e;
  - the DRF1 reader recipes and the nested estimator's helpers, copied from 913eb5c91 and hash-checked against DRF1's code lock;
  - published aggregate records: RESEARCH_DECISION.md, PROTOCOL.md counts, `dr/FIT*_V*.json`, `dr/OPTIMISTIC_V*.json` and
    DEVGRID_CODE_LOCK.json.
- **Not read.** No census record, label, model, private array, row-level release or cache; nothing from MODEL_FIT, MODEL_SELECT,
  CALIBRATION, ATTACK, DEV_REPORT, sealed or capped-out data. No historical release, claim label or protocol was changed.

**Custody.**
- **Backup.** The research branch is backed up as a git bundle with a SHA-256 manifest in a sibling folder,
  `pcrl-estimator-diagnostic-v1-backup`.
- **Restore check.** A restore from the bundle alone checks:
  - the manifest and the tip;
  - the self-test;
  - 7 saved repetitions, recomputed from their seeds, which must equal the saved rows exactly;
  - the frozen analysis re-run on the restored outputs, which must give a value-equal SUMMARY.json and a byte-equal TABLES.md.
- **Rehearsals.** Rehearsal 1 failed on tooling (record kept, amendment A3); rehearsal 2 passed.
- **Final records.** The final backup's restore record and any off-device copy are in `custody/`. A publication receipt, if any,
  is kept on the research branch only.
- **Known gap.** Records written after the final backup are not inside it.

## How to reproduce

From the repository root of this branch, with the PCRL venv interpreter (`../../../.venv/bin/python` from the worktree root;
numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.8):

```
PY=../../../.venv/bin/python
$PY results/pcrl_estimator_diagnostic_v1/edg_run.py laws                 # LAWS.json: known laws, population values, analytic checks
$PY results/pcrl_estimator_diagnostic_v1/edg_run.py selftest             # tiny-law, tie, merging, source-equality and sampler checks
$PY results/pcrl_estimator_diagnostic_v1/edg_run.py simulate --cells 0-48 --reps 500 --label all   # about 12.5 CPU-min
$PY results/pcrl_estimator_diagnostic_v1/edg_run.py bench                # timing-only bank microbenchmark -> BENCH.json
$PY results/pcrl_estimator_diagnostic_v1/edg_run.py bank --label all     # reader-bank subset, about 6 CPU-min
$PY results/pcrl_estimator_diagnostic_v1/edg_run.py analyze              # SUMMARY.json, TABLES.md, figures/
$PY results/pcrl_estimator_diagnostic_v1/edg_run.py posthoc              # POST_HOC.json (labelled post hoc summaries)
$PY results/pcrl_estimator_diagnostic_v1/edg_run.py custody backup DIR   # branch bundle + manifest; `custody restore DIR/<snapshot>` checks it
```
Every process uses one numerical thread. Outputs are deterministic given the seeds.
