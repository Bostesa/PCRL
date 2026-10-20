# Challenger, closing report (saved verbatim with attribution)

Author: the challenger of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 16:35Z on 2026-10-20, in answer
to the coordinator's 16:33Z request. The challenger read no MODEL_SELECT output itself beyond the numbers the coordinator sent. Saved
verbatim; nothing below the line was edited.

Coordinator note (not part of the report): the phrase "about 2.7–3.0× the occupation NLL, Brier and men's-guard caps" repeats an
imprecise summary in the coordinator's own 16:33Z message. The record shows that CTL_REM_1p00 fails the occupation NLL clauses at
1.79-1.87 (version 0) and 2.20-2.21 (version 1) times the cap and the men's guard at 2.67 / 2.97 times the cap; its occupation Brier
clauses pass (0.55-0.87). Engineering finding DOC2-01; ADJUDICATIONS.md A6.

---

CHALLENGER CLOSING REPORT (16:35Z)

1. What the CTL path shows

CTL_REM keeps every decision and top probability, and changes only the occupation remainder.

**s = 0.50: invertible reshaping removed nothing.**
- At this strength the remainder becomes (1−w)·p_rem + w·(1−p_top)·R[j].
- Given the released top class and top probability, that map is affine and invertible wherever the cap does not bind.
- As the invertibility argument predicted (catalogue C3), it removed nothing. CTL_REM_0p50 read 0.6596 / 0.5569 / 0.6419 (pair / income / occupation), against FB2_BASE's 0.6605 / 0.5569 / 0.6428.
- It still spent more than FB2_SEX_S3.

**s = 1.00: many-to-one collapse was too expensive.**
- At full strength the remainder is replaced by fixed shares. The occupation vector then depends only on the decision and the top probability.
- The occupation view fell by 0.083 (to 0.5603), but the linked pair fell only 0.0148 (to 0.6457).
- It was task-infeasible, at about 2.7–3.0× the occupation NLL, Brier and men's-guard caps.

**Three structural readings** (development evidence; MODEL_SELECT was used adaptively):
- (a) The occupation view's SEX signal sits mostly in the secondary-class coordinates. Those coordinates are also task-useful, so the information readers exploit is the information the task clauses pay for.
- (b) After the collapse the linked pair still reads about 0.646. The income vector, the occupation decision and its confidence carry SEX jointly, so the pair floor for this base is set by what the tasks themselves transmit.
- (c) FB2's SEX-targeted channel was NOT_SEPARATED from this SEX-blind, deterministic path at both strengths: +0.0059 (se 0.0080) and −0.0029 (se 0.0084). There is no evidence that targeting or randomness beat SEX-blind removal at matched or higher spend. "Not separated" does not mean "equivalent": these SEs are about 40% of the 0.020 target.

2. Update of "operations not found in the records"

**Now tested:**
- **Remainder flattening** (item 2): CTL_REM, above.
- **Randomized releases attacked on ACS FL** (item 4): the FB channel trained through the noise and FB2's post-hoc Gaussian logit channels. They bought about 0.01 of pair AUC at the affordable spend, with no targeting advantage at matched spend (S5 vs MINF_S5: +0.0019, se 0.0089). F1 holds.
- **Coupled randomness across recipients** (item 5): FB2's shared-ε rank-1 coupling was tested as built, but never against a decoupled channel.
- **A SEX-blind path bracketing a candidate's spend** (item 1): done for FB2, where both MINF and CTL_REM bracketed. Generic cell coarsening (C2) was never fitted.

**Still untested:**
- π-routed finite cells. FA-PRC closed at the capacity gate before any protected fit; HQ, ORD and SHUF never ran.
- Continuous-mixture and temperature-matched controls.
- Randomized response on the decisions.
- Input-side recoding or erasure of the six fields or the 145-column encoding.
- Training-time SEX-balanced reweighting.
- MMP (not authorised).
- An isolated coupled-vs-decoupled contrast.

3. A4

I do not object, for three reasons:
- **The gap is in the predictor's transmitted information.** The gap to NEURAL_RAW and H_LJ is 0.024–0.028 in pair AUC. The strongest SEX-blind intervention we built bought only 0.0148, at about 3× the caps, and still read above both target references.
- **The untested routes start from the same weakness.** Input recoding and reweighting act on the same task information, and randomized response acts on decisions. Every earlier family in those places paid utility first.
- **Any result would be weak evidence.** A new family would be read on the same, adaptively used MODEL_SELECT people, and a 0.020 benefit cannot be confirmed on about 17,500 households.

Two requests:
- (a) The closeout should list the untested routes above explicitly, so "no replacement" is not read as "all routes exhausted".
- (b) The CTL result should be worded as "not separated", never as "SEX targeting equals SEX-blind removal".

4. CPU

51.2 CPU-s in total:
- 1.23 CPU-s in four queue units (CTL_REM_PLAN_V0/V1, CTL_REM_FIT_V0/V1);
- 50.0 CPU-s of out-of-queue charges under the label 'challenger', all upper estimates (selftests, scale check, choose step, sanity and parity check).

The CTL evaluations and readers ran in the coordinator's units. My wave-1 extraction sub-agents did no numerical work.
