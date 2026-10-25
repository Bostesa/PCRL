# EDG1 statistical review, round 2: draft report against the results

Statistical reviewer, EDG1. Written 2026-10-25 from 17:57Z (`date -u`). This review checks the draft `ESTIMATOR_DIAGNOSTIC.md`
at tip ad0e19dee against SUMMARY.json, TABLES.md, POST_HOC.json and both figures.

## Scope

- **Read.**
  - EDG1: the files above.
  - DRF1's published aggregate records, to check the report's DRF1 numbers:
    - `dr/OPTIMISTIC_V{0,1}.json`;
    - `dr/FIT_V{0,1}.json`;
    - `dr/FIT_R1_V{0,1}.json`.
- **Computed.** Three read-only extraction scripts, run in the scratchpad on one thread, under 10 CPU-s in total.
- **Not touched.** Nothing outside the worktree, and no git change.

## Verdict

All five round-1 REQUIRED items are resolved in substance.
- Spot checks: more than 45 numbers checked against SUMMARY.json, POST_HOC.json or DRF1's records. All match except the 9 in
  section 3, which differ by scope, sign of a trend or rounding; three of those are REQUIRED.
- New REQUIRED items: three.
  1. **Answer 5's main new reading rests on a factual error.** It says the matched coarsening point's support is unknown, but DRF1's
     published records give it.
  2. **The bottom line misstates LOOKUP's dependence on sample size.**
  3. **One bank count is mis-scoped.**
- None of the three changes DRF1's no-method conclusion or requires re-running anything.

## (1) Round-1 REQUIRED items

| item | status | where and how |
|---|---|---|
| a2 D_bank targets the fitted readers | **Resolved** | Answer 1 and the answer-4 table: dA-relative rates are described as "a valid fitted-reader conclusion misstating population information". Coverage of the fitted readers' own difference is 94.8–95.9%, and the rate under the exact nulls 4.2–5.2% (both post hoc). Residual: the 4.2–5.2% is pooled over 12 cells; the per-cell registered range is 2.4–7.2% (a note, below). |
| a3 D_nested's R2SE is naive | **Resolved** | Answer 1 explains the mechanism and gives 13–20% resolved under the exact nulls; procedure step 4. |
| a4 B's null comparisons are degenerate | **Resolved** | Answer 1: B's null false-positive rates are said not to show calibration. Non-null coverage is 95.2–95.8% for B_indep and B_same (post hoc), with an implementation note on float ties. |
| e2 Figure 2's "zero-mean" title | **Resolved** | Answer 3 calls the third term "cross-fit dependence plus test variation": negative in all 160 cell-views, and more than 2 MCSE below 0 in 12 of 13 Figure-2 views (post hoc). It states that the registered title is wrong for D_nested; the figure is kept as registered. Residuals: b6 and the range error in section 3. |
| f1 DRF1 mapping | **Resolved, with a new error** | Answer 5 covers the exact mapping of the twin's numbers, DRF1's point-only use with no SE, the propensity-law objective, the decision resting on the bank, and the release-optimization bias. Its new "unregistered reading" paragraph contains the error in c1. |

## (2) Claims in the answers, implication and procedure

### Answer 5 (DRF1)

**c1 REQUIRED. The matched coarsening point's support is in DRF1's published records, and it fixes the direction of the
confound.** The report says the matched coarsening point's support "is not in the published decision", and that "which way the
confound ran depends on the matched point's support, which EDG1 does not know". But DRF1's published FIT records identify it.

DRF1's `rules.useful_optimistic` nested-mode bracket B is the cheapest nested-path point whose nested total-NLL spend is at
least OPT_L's.

| version | OPT_L nested spend | B (step) | B's nested pair AUC* | OPT_L's nested pair AUC* | shortfall |
|---|---|---|---|---|---|
| V0 (`FIT_V0.json`) | 0.0380 | 67 | 0.6425 | 0.6868 | 0.044 |
| V1 (`FIT_V1.json`) | 0.0378 | 68 | 0.6384 | 0.6794 | 0.041 |

These reproduce DRF1's 0.044 / 0.041 exactly. Both points' supports are recorded:
- **B.** Step 67 / 68 leaves 13 / 12 atoms (one merge per step from 80). The in-sample path reports 5 income and 8 / 7
  occupation atoms, giving 40 / 35 pair wires. The fold refits have the same total atom count at the same step, so at most
  42 / 36 pair cells.
- **OPT_L.** In-sample it had 374 / 398 pair wires (`OPTIMISTIC_V{0,1}.json`); the report quotes only V0.

So the comparison set a release with roughly ten times the pair support against B. In EDG1's four independent REF cells at
n = 20,000 (folds of 4,000), C_nested optimism is 0.003–0.020 at 16–64 wires and 0.04–0.19 at 256–1,024. The confound therefore
runs against OPT_L, by an amount of the same order as the 0.044 / 0.041 shortfall.

Required rewording: the nested-mode "not useful" reading is not evidence against OPT_L. It is confounded against OPT_L by
support-dependent optimism of that order. EDG1 still cannot say whether OPT_L would be useful under an unbiased held-out measure,
or how large DRF1's own bias was.
- Drop "in either direction" and "EDG1 does not know".
- Keep the separate evidence: the nested spends were 2.6–4.0 times the budgets.

**c2 RECOMMENDED. The +0.034 against the coarsening bracket: give the direction the published supports imply.** The records
give both supports:
- DR_AUC_R1: 456 / 564 pair wires in-sample (`FIT_R1_V{0,1}.json`);
- DR_CTRL_HI_R1, the upper bracket at steps 62 / 63: 70 / 68 pair wires; the lower bracket has 86 / 83.

In EDG1's laws, fitted-reader loss grows with support at identical A. The bank winner was 0.0083 ± 0.0013 lower at 256 than at
16 wires (post hoc), and D_bank was 0.021–0.046 lower for L256 − L16 at n = 8,000. So the confound plausibly favours the finer
candidate.

State the direction and leave the size unquantified. Note that the two releases are not nested partitions, so the direction is an
empirical tendency, not a theorem. The DRF1 label is unchanged: D2 was NOT_SEPARATED because of DR_MI, and the bracket separation
was already "primary cost only".

**c3 RECOMMENDED. Two version-scoped numbers.**
- "DR_AUC_R1 and DR_MI_R1 had nearly equal in-sample pair supports (456 and 445)" holds for version 0 only. In version 1 they are
  564 and 485, which differ by 16%. Report both versions.
- OPT_L's 374 is V0; V1 is 398.

**c4 RECOMMENDED. The R1-agreement wording overstates.**
- The post hoc analogue compares means: 500 repetitions of D_nested against 10 of the bank winner. The expected difference is
  −0.007 to +0.005.
- A single realization differs with an SD of about 0.010. The REF L256 example shows this: bank winner SD about 0.009 (MCSE
  0.0029 × √10) and D_nested SD about 0.004 (0.00019 × √500).
- Agreement within 0.002, as DRF1 found, is consistent with these laws but not "expected" to be that close. Say so; the existing
  "not a validation" sentence can stay.

**c5 NOTE.**
- These parts of answer 5 match their sources and stay within the design: the twin mapping (0.6812 / 0.7475 / 0.6303, 727
  cells), DRF1's point-only use, the post hoc M20 values (20 / 61 / 100% and 60 / 100 / 100%), the optimism differences (0.03–0.05
  and 0.06–0.12), D_bank L256 − L16 = −0.021, the propensity-law statement and the decision paragraph.
- "No numerical error in DRF1's recorded decision was found" should read that EDG1 did not recompute DRF1's decision and could not
  establish an error.

### Answer 4 (which estimator for which claim)

**d1 REQUIRED. Ignore-one winners in RED.** "In RED the pair winner was an income-only ignore-one candidate in 10 of 10
repetitions" holds only at pi = 0.5. At pi = 0.2 (RED|0.2|skewed) the pair winners were:
- an income-only ignore-one candidate in 4 of 10;
- pair readers in 5;
- an occupation-only ignore-one candidate in 1.

Pair selection loss there was 0.0034 ± 0.0016, resolved at 2 MCSE. Scope the sentence to pi = 0.5 and give the pi = 0.2 result;
the ignore-one protection was partial there.

**d2 RECOMMENDED. The "leaks at least x" row.**
- "Their population AUC never exceeded A" is a theorem (pop <= A for any score of the wire); label it so.
- A single SCORE value can exceed A through test variation: the SD at SCORE 2,400 is about 0.009. It is a lower bound only in
  expectation. An "at least x" statement needs the lower confidence limit (value − 2 se).

**d3 RECOMMENDED. The equal-support row** rests on one null: COMP occupation (15) against income (16), with matched occupancy.
C_same's false-positive rate was 0–1.0% and C_nested's 0–1.4%, both below nominal. The bias depends on cell counts and signal,
not on the support count alone. Say "in this one matched-occupancy null"; equal support is not shown to be sufficient in general.

**d4 RECOMMENDED. Label the post hoc rows** of the bank table, and the same values where answer 5 repeats them. They come from
POST_HOC.json, not from the frozen analysis:
- "resolved at 2 MCSE";
- the 3 of 9 count;
- the REF paired differences;
- the class-limit and saturated rows;
- the 140 / 140 LOOKUP cross-check.

The table also repeats its header row.

### Bottom line, answers 1–3, implication, procedure

**b1 REQUIRED. LOOKUP and sample size.** The bottom line says held-out LOOKUP declares the finer output less recoverable "more
often the finer the refinement and the smaller the sample". The sample-size half is contradicted by the registered rates (REF
cells; range over pi and occupancy):

| estimator | comparison | n = 2,000 | n = 8,000 | n = 20,000 |
|---|---|---|---|---|
| D_bank | L256 − L16 | 0.57–0.89 | 0.89–0.99 | 0.93–1.00 |
| D_bank | L1024 − L16 | 0.75–0.99 | 1.00 | 1.00 |
| D_bank | L64 − L16 | 0.28–0.44 | 0.34–0.52 | 0.26–0.54 (no trend) |
| D_nested | L64 − L16 | 0.46–0.65 | 0.25–0.70 | 0.14–0.47 |
| D_nested | L256 − L16 | 0.89–1.00 | 0.98–1.00 | 0.96–1.00 |

D_bank's rate rises with n. D_nested's falls only for L64 − L16; the finer comparisons are flat at 0.89–1.00 or higher.

This is the a2 mechanism: the fitted readers' true difference shrinks more slowly than the SCORE SE. Required rewording: "more
often the finer the refinement; for the single held-out split the rate rises with n".

**b2 RECOMMENDED.** "Only the known-law score ... behaves nominally" is true for population information only. D_bank is nominal
for its own target (coverage 94.8–95.9%, post hoc). Add that qualifier, which answer 1 already states.

**b3 RECOMMENDED. The pi = 0.2 effect.** "Raised the bias by roughly 20–60%" understates the range. Across all family × occupancy
× n × view pairs with a pi = 0.5 bias above 0.002, the ratio was 1.09–1.85 (median 1.26). Give the range and the subset used.

**b4 RECOMMENDED. "About four fifths ... optimism" holds for supports 240–1,024 only** (the rows shown). At support 16 the
cross-fit term is close to half of a small gap:
- REF L16: 0.0026 optimism, 0.0002 loss, 0.0023 cross-fit, gap 0.0051;
- RED income: 0.0031, 0.0001, 0.0023, gap 0.0055.

Qualify the sentence.

**b5 RECOMMENDED. COMP.** "All estimators found the pair's real extra information" is too strong for D_bank. Its mean difference
was positive (0.009–0.074), but at n = 2,000 its point estimate had the wrong sign in up to 43% of repetitions, and its power was
5–9%. Say "every estimator's mean difference was positive", and give D_bank's sign-error rate.

**b6 RECOMMENDED. "What was measured" contradicts answer 3.** It still says the population AUCs separate "estimation loss
(A − pop) from test variation (sample − pop)". For D_nested, sample − pop is cross-fit dependence plus test variation (answer 3).
Fix the sentence.

**b7 RECOMMENDED. Figure 1's title note.** It says the title's "support and LOOKUP statements are theorems on nested
partitions". The plotted LOOKUP is D_nested. Its population part (pop_pooled <= A) is a theorem, but its cross-fit part is an
empirical finding (negative in all 160 cell-views). Only E[D_bank] <= A is fully a theorem. The left panel also joins non-nested
IND views (income and occupation).

**b8 RECOMMENDED. Procedure step 3, the opposing-bias rule.** Support it with the numbers that justify it, and state its cost:
- **False confirmation is rare under the nulls.**
  - Held-out readers resolved "refined more recoverable" in at most 0.8% (D_nested) and 0.4% (D_bank) of REF null repetitions.
  - C cannot resolve "refined less recoverable" on nested partitions (theorem).
- **Power is low.** On RED income − occupation, C_nested has 0% power and C_same 3–21% at n = 20,000. On COMP at n = 2,000 the
  held-out D_bank has 5–9% power.
- **Scope of C's direction.** C's opposing bias is guaranteed only on nested partitions.

**b9 RECOMMENDED. Wording of the implication and procedure.**
- Step 4's "calibrate SEs by repetition" was not tested here; say so.
- "Under independence" should read "under the exact nulls" (the 13–20% includes COMP occupation − income).
- The implication's "Any future recovery objective ... has to pass" should carry the same "in these laws and sizes" qualifier as
  the procedure.

## (3) Number spot checks (SUMMARY.json and POST_HOC.json)

**Matched.**
- **Answer 1:**
  - C_same and C_nested P(d > 0) = 1.000 in all REF comparisons; C_same false positives 99.6–100%, 99.2% clustered; C_nested
    100%;
  - RED pair − income: C_same +0.011 to +0.111, C_nested +0.045 to +0.229, false positives 100%;
  - IND pair − income +0.093 to +0.355;
  - COMP occupation − income: C_same 0–1.0%, C_nested 0–1.4%;
  - D_nested / D_bank on REF: P(d > 0) at most 0.286 / 0.106; false positives 14–100% / 25.6–100%; at most 0.8 / 0.4 points in
    the other direction;
  - D_nested under the exact nulls 12.8–19.6%;
  - D_bank target coverage 94.8–95.9%;
  - B coverage 95.15–95.77%;
  - B_indep max |bias| 0.00138;
  - B null false positives 0–7.0% (B_indep; maximum in cell 37).
- **Answer 2:** all 18 table entries, including C_same REF L1024 at n = 20,000 +0.0334 and IND pair +0.1067, and C_nested
  +0.0985 / +0.2177. Also:
  - MCSE at most 0.00051;
  - L1024 − L16 ranges 0.033–0.056, 0.063–0.117, 0.137–0.244 and C_nested 0.096–0.182;
  - IND at n = 2,000: 0.915 / 0.991;
  - skewed below uniform at the largest support in all 12 pairs.
- **Answer 3:**
  - gaps 0.218 / 0.122 / 0.050 / 0.061 / 0.092, and optimism and loss with their shares;
  - D_bank test-variation range;
  - cross-fit term negative in all 160 cell-views (−0.00004 to −0.0112);
  - fold rules agree within 0.0001 (Figure-2 cells; at most 0.00012 over all cells).
- **Answer 4 and the bank:**
  - A − best 0.0012–0.0121;
  - selection 0.0004–0.0034, 3 of 9;
  - curse at most 0.0286;
  - test variation from −0.0064 ± 0.0044 to +0.0035 ± 0.0043;
  - L256 − L16 −0.0083 ± 0.0013 and −0.0074 ± 0.0023;
  - COMP class limits 0.4999–0.5049, class loss 0.078–0.083;
  - saturated values 0.573 / 0.577; MLPs 0.567–0.570;
  - LOOKUP finite-training loss 0.021–0.030;
  - LOOKUP cross-check 140 / 140.
- **Answer 5:** the twin mapping (0.6812 is C_same in `FIT_V0.json`); M20 values 19.97 / 60.55 / 100% and 60.06 / 100 / 100%;
  optimism differences 0.028–0.049 and 0.057–0.118; D_bank −0.0213; agreement −0.0068 to +0.0052.
- **Further results:**
  - IND income 0.0437 / 0.0219 / 0.0136;
  - REF L1024 at pi = 0.2, 0.0489 against 0.0334;
  - unseen share 37.7%; occupied 495;
  - the wrong-direction table: C_nested 96.6–100%, C_same up to 78.2%, D at most 0.2%, B_indep power 0.322–1.00;
  - RED pair − occupation: D_nested 2.8–96.4%, D_bank 6.4–88.8%;
  - D_bank COMP power 4.6–9.0% and 25.8–71.4%;
  - clustered: biases within 0.00033, SDs within 0.00051, R2SE within 0.006, B_indep 0.040 / 0.036.

**Differ (9).**

| report | data | class |
|---|---|---|
| matched bracket support "not known" | 40 / 35 pair wires (published) | REQUIRED (c1) |
| LOOKUP rate "more often ... smaller sample" | D_bank rises with n | REQUIRED (b1) |
| RED ignore-one "10 of 10" | 10 / 10 at pi = 0.5; 4 / 10 at pi = 0.2 | REQUIRED (d1) |
| pi = 0.2 "roughly 20–60%" | 9–85% (median 26%) | RECOMMENDED (b3) |
| supports "456 and 445" | V0 only; V1 564 / 485 | RECOMMENDED (c3) |
| cross-fit range "−0.0006 to −0.0027" | −0.0005 (RED pair −0.00053) to −0.0027 | NOTE |
| answer-3 table, RED pair third term "−0.000" | −0.00053, which rounds to −0.001 | NOTE |
| B_indep "largest 2.7 MCSE, one of 160" | largest 2.66; 4 of 160 views beyond 2 MCSE (about 7 expected by chance) | NOTE |
| D_bank exact-null "4.2–5.2%" | pooled over 12 cells; per-cell 2.4–7.2% | NOTE |

## (4) Other notes

- **h1 NOTE.** "Several times more biased" (C_nested against C_same) is 2.0–3.6 times at n = 20,000; give the range.
- **h2 NOTE.** LOOKUP's downward bias is not always "for the same reason: sparse cells". Under IND, A − pop = 0, and the only
  downward term is the cross-fit dependence.
- **h3 NOTE.** The implementation note's "only its sample variance is slightly larger" (B with float-broken ties) was not checked
  by anyone. Drop it, or support it.
- **h4 NOTE.** D_bank's test variation was beyond 2 MCSE in 7 of 160 cell-views, as expected by chance. Prefer that to "within
  about 2 MCSE".
- **h5 NOTE. The theorem and finding labels are otherwise correct:**
  - C's d >= 0 on nested partitions;
  - E[C_same] >= A;
  - E[D_bank | table] <= A;
  - D_bank's dA-relative rates are presented as estimand mismatch, not miscalibration.

## Classification summary

| class | items |
|---|---|
| REQUIRED | c1 (answer 5, OPT_L bracket support and direction), b1 (bottom line, LOOKUP against n), d1 (RED ignore-one scope) |
| RECOMMENDED | b2, b3, b4, b5, b6, b7, b8, b9, c2, c3, c4, d2, d3, d4 |
| NOTE | c5, h1–h5, and the four NOTE rows of the mismatch table |

Round-1 REQUIRED items a2, a3, a4, e2 and f1 are resolved, with c1 a new defect inside the answer that resolves f1. DRF1's
no-method conclusion is unaffected by every item here.
