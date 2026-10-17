# Research decision: PCRL autonomous method-development campaign

**Scope.** This is exploratory method development on public-use U.S. Census Bureau American Community Survey (ACS) 2019 1-Year PUMS person records for Florida. Earlier campaigns already used these records. Nothing here is confirmation.

**The service.** For each person it predicts:
- an income class, 2 classes: PINCP × ADJINC / 10^6 above $50,000, in 2019 dollars;
- an occupation group, 5 classes: OCCP mapped to the five civilian broad groups.

The predictions use only six raw fields: AGEP, WKHP, COW, SCHL, MAR and POBP. The cohort is the replication's: housing-unit residents aged 25–64, civilian employed (ESR 1 or 2), WKHP 1–99. Each recipient receives only its own probability vector and argmax decision. SEX is never an input.

**What is measured.** Evaluation readers infer the dataset's recorded SEX column from each output alone and from the two outputs aligned to the same record (the linked pair). Pairs of numbers are seed 0 / seed 1. Both seeds cover the same 5,943 people in 3,955 households, so they are not separate populations.

**What the intervals cover.** DEV_REPORT intervals are household-bootstrap Wald intervals (B = 2048), conditional on the fitted releases and readers. They are not ACS design-based: no PWGTP or replicate weights are used.

They exclude the adaptive development search:
- 22 MODEL_SELECT evaluations of new configurations, plus 22 of frozen references, all on one fixed household split;
- three families, their revisions and controls;
- the choice of α = 0.5 and m = 6 on CALIBRATION screens with a MODEL_SELECT-derived offset;
- the promotion and its adjudication;
- earlier campaigns' use of MODEL_SELECT and DEV_REPORT.

DEV_REPORT is not historically unseen: the finalist's occupation output and the references' DEV results were published beforehand. The sealed assessment stayed closed. No novelty is claimed.

## 1. Bottom line

The finalist is `B3CTL_UNCOUPLED_MIX50_C6A`, labelled `EDGE_SELECTED`. It recombines components the protected-tree replication had already trained:
- **Occupation output.** The replication's stored, pair-aware JOINT_TREE output, bitwise identical to the occupation output of the frozen reference H_LJ.
- **Income output.** A 6-level, decision-aligned code of the average of the stored LOCAL_TREE and JOINT_TREE income log-odds.
  - Each level is calibrated on CALIBRATION income labels, by L2 logistic regression on the one-hot level.
  - The income output therefore takes at most 6 values; 6 distinct values are observed on ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT for both seeds.
  - It is not computed from the occupation output.

Results of the one logged DEV_REPORT evaluation:

| Claim | Label | What it rests on |
|---|---|---|
| E, eligibility (point gate) | `DEV_POINT_ELIGIBLE` | 24/24 utility points, 4/4 within-SEX guards and 8/8 recipient points hold, at the edge: income summed-Brier excess against Ucal is +0.0049 / +0.0049 (cap 0.005) and seed-1 income accuracy loss is 0.0096 (cap 0.010). Simultaneous utility confidence is `CONFIDENCE_UNRESOLVED` (3 PASS, 25 UNRESOLVED of 28). A point pass is not established population preservation. |
| B, competitive against every eligible reference | `SUPERIOR_TO_ALL_ELIGIBLE` | Linked-pair benefit (reference AUC − finalist AUC): against NEURAL_RAW +0.027 / +0.0199, clause bounds 0.0135 / 0.0059; against H_LJ +0.028 / +0.025, clause bounds 0.0191 / 0.0129. All 8 recipient non-worsening clauses pass at the clause bound. The 2 occupation clauses against H_LJ are IDENTICAL_RELEASE (point = SE = 0) and the 2 against NEURAL_RAW are REPLAY_OF_PUBLISHED, so the 4 income clauses are the only new recipient evidence. Both references meet all 12 DEV utility clauses at the point on both seeds. The occupation component was chosen with knowledge of its published DEV_REPORT results. The refitted NEURAL_RAW and H_LJ pair readers reproduce the published reader scores bitwise, 16/16 per seed. **Positive, below the original 0.020 target**: `MATERIAL_SUPERIORITY_ESTABLISHED` is not reached. Seed 1 against NEURAL_RAW is a POINT_SHORTFALL (0.0199); the other three are UNRESOLVED at 0.020. |
| M, mechanism attribution (fixed sequence) | `MECHANISM_ATTRIBUTION_ESTABLISHED` | On DEV_REPORT, on both seeds, the declared readers recover SEX from the finalist's linked pair less well than from two comparators. **Step 1:** the matched task-only control, which applies the same coarse-code construction to the replication's privacy-off TASK_REFIT outputs: +0.052 / +0.047, clause bounds 0.0391 / 0.0336. **Step 2:** H_LJ, the card's simple control: +0.028 / +0.025, read from claim B's keys (alias), so step 2 adds no comparison beyond B. The task-only control differs in both outputs: its TASK_REFIT occupation output alone reads 0.650 / 0.645, against the finalist's inherited 0.560 / 0.583. Step 1 therefore credits the inherited privacy-trained components as a whole; the new income code's contribution is carried by step 2. **Qualifiers:** against the task-only control the finalist spends +0.019 / +0.015 nats more occupation NLL (marginal 95% intervals exclude 0) and +0.0050 / +0.0039 more income NLL (UNRESOLVED). Against H_LJ it spends +0.0083 / +0.0076 more income NLL (ABOVE_ZERO) and identical occupation NLL. Its privacy-trained components are the replication's stored Stage B corrections. |
| J, joint-design component | `JOINT_ADVANTAGE_NOT_ESTABLISHED` | Against the twin with the same coarsening, calibration and occupation output but a LOCAL-only statistic: +0.016 (clause bound 0.0099) / +0.010 (clause bound −0.0025, UNRESOLVED; the first failing condition). **Mandatory qualifier (cost of the joint component):** the finalist spends +0.0027 / +0.0027 more income summed Brier than the twin (marginal 95% [0.0002, 0.0051] / [0.0003, 0.0051]) and +0.0032 / +0.0036 more income NLL (UNRESOLVED). |
| N, novelty | `NOT_CLAIMED` | Coarsening and histogram calibration of a privacy-trained score, adversarial training and marginalization are known techniques. The privacy-trained parts were inherited. |
| IBSI | descriptive only | PUB_IBSI − finalist pair: +0.017 / +0.005, UNRESOLVED. PUB_IBSI was designated ineligible at the freeze on MODEL_SELECT (seed-1 men's guard 0.00501; it fails the occupation recipient screen against H_LJ). On DEV its men's guard is 0.0051 / 0.0091 and it fails seed-1 utility. Its discrete pair (about 470 distinct vectors) also faces a different reader regime from the finalist's continuous pair. |

**Conjunctions.** These are frozen in `ANALYSIS_LOCK.json`, and each claim is a separate conclusion:
- **E** = all 24 utility points ≤ cap ∧ all 4 guard points ≤ 0.005 ∧ all 8 recipient points ≤ 0.010. E spends no α.
- **B** = CV ∧ E ∧ pair-benefit clause bound > 0 against NEURAL_RAW and H_LJ on both seeds ∧ all 8 recipient clause bounds ≤ 0.010. The 0.020 target is the same four bounds > 0.020 and spends no extra α.
- **M** = step 1, then step 2:
  - step 1: CV ∧ E ∧ the task-only control banked ∧ its pair-benefit clause bound > 0 on both seeds ∧ its 4 recipient clause bounds ≤ 0.010;
  - step 2: the same against H_LJ, read from B's keys.
- **J** = CV ∧ E ∧ the LOCAL-only twin banked and MODEL_SELECT-eligible ∧ its pair-benefit clause bound > 0 on both seeds ∧ its 4 recipient clause bounds ≤ 0.010.
- **α and critical values.** B, M and J each spend one-sided α = 0.025/3, with clause critical value 2.395. Family intervals use 3.304 (52 primary keys) and 3.025 (20 secondary keys).

**Independent verification.** A separate agent with its own code passed all 8 checks.
- Every one of the 72 endpoints matches the record, with a maximum point difference of 0.0.
- All 14 reader refits and the household weights are bitwise identical.
- Selection, labels and deployment parity reproduce.
- The verifier made one logged DEV_REPORT replay, at 17:56:47Z.
- Its discrepancies D1–D3 are procedural or informational and change no number or label. `PROTOCOL_POSTRUN_CLARIFICATIONS.md` documents them.
- **D1 is clarified, not resolved.** The claims draft was written at about 17:37Z. That is after the reviewer's adjudication (about 17:34Z) and after the promotion record (17:35:23Z), so PROTOCOL §9 ("before a finalist is selected") was not met. The record shows only that the conjunctions were hashed and committed before any final reader fit (17:43:02Z), and published before DEV_REPORT was read (pushed 17:45:22Z; read 17:45:30Z).

Planted-SEX and shuffled-label controls stayed inside the frozen bands on every view and seed: planted 1.0; shuffled 0.499–0.535 on ATTACK_VALIDATION and DEV_REPORT.

**Strongest supported conclusion.** On these development records, the declared readers recover SEX less well from this recombined release's linked pair than from the linked pair of either eligible reference (NEURAL_RAW, H_LJ). This holds on both model seeds, fitted to the same people. The readers are the 8-reader slate with the 8,001-person ATTACK_FIT budget. From the occupation output alone, recovery is identical to H_LJ's, because the output is identical. The release meets every original utility, subgroup and recipient condition at the point.

This is a positive, below-target improvement, for four reasons:
- the 0.020 material improvement is not established;
- the income utility budget is almost fully spent;
- the occupation side replays a published output;
- the selection was adaptive and edge-seeking.

It is evidence against the declared readers, not a privacy guarantee. SEX remains readable from the pair (AUC ≈ 0.61–0.62).

## 2. Which ideas were developed, and what each revision taught

All numbers in this section are MODEL_SELECT development proxies (household halves). They are adaptive selection feedback, never effect sizes.
- **Pair bar:** NEURAL_RAW 0.6238 / 0.6269, H_LJ 0.6224 / 0.6233.
- **Recipient bars:** income ≤ 0.5500 / 0.5476; occupation ≤ 0.5824 / 0.5953.

The coordinator proposed all three families, writing each mechanism card before its family's first fit. The coordinator implemented A; separate investigator agents implemented B and C.

### Family A: constrained coalition-aware leaf correction (coordinator)

The family uses the JOINT_TREE critic objective on the frozen CatBoost teacher routing, retrained on 80% of MODEL_FIT. A new per-epoch utility controller runs on a household-separated 20% holdout: it rolls back steps and throttles privacy pressure against per-task and per-SEX budgets.

- **A1.**
  - Income Brier clause fails (0.0062); pair 0.6235.
  - Diagnosis: an in-sample per-SEX budget throttled the occupation recipient.
  - Diagnosis: holdout income costs under-stated MODEL_SELECT by about 0.004, because the teacher is in-sample on MODEL_FIT.
- **A2** (budgets from the teacher's MODEL_SELECT headroom):
  - All clauses pass with margin; pair 0.6103 / 0.6296. The seed-1 pair is above both references.
  - Occupation proxy 0.597 / 0.598, above the occupation bar. Not promotable.
- **A3** (occupation loss weighted by SEX, men 2 : women 1):
  - Calibration moved toward men, the women's budget bound, and the controller removed the occupation pressure (pair 0.639).
  - A fixed SEX weighting moves calibration across groups; it does not reallocate the privacy burden.
- **Lesson.**
  - In family A no configuration met the occupation recipient bar at a feasible men's guard: A2 read 0.597 / 0.598 against bars of 0.582 / 0.595.
  - An internal holdout drawn from the teacher's own fitting people is not held out.

### Family B: coarse / coupled income code beside the JOINT_TREE occupation output (investigator B)

- **B1, pure coupling** (income computed from the occupation vector): pair 0.5666, but income NLL excess against U0 rises to +0.053.
  - A simple calibrated map of the occupation vector (L2 logistic regression on its 4 log-ratios) does not recover income within the caps.
  - The occupation output is essentially one-to-one in the inputs, so this is a limit of the simple map, not an absence of information.
- **B2** (adds a 6-level LOCAL_TREE income code): eligible on both seeds, pair 0.6131 / 0.6132.
  - The uncoupled twin reads 0.6153 / 0.6154, with better income utility.
  - Both pairs are functions of the same occupation output, which is essentially one-to-one in the inputs on the observed people. Neither carries information the other lacks there.
  - The 0.002 proxy gap is a reader effect, and the reader classes disagree on its sign. Coupling is not the active ingredient.
- **B3** (code built from 0.5 LOCAL + 0.5 JOINT income log-odds):
  - The coupled version fails income Brier on seed 0.
  - Its uncoupled twin, the finalist, reads 0.6000 / 0.6001, with income Brier at 0.87–0.88 of the cap.
- **Controls:**
  - matched task-only, the same construction on TASK_REFIT: 0.6455 / 0.6468;
  - teacher-base coupled control: 0.6454 / 0.6463.
- **Not run:** a continuous-mixture control (the same 0.5 / 0.5 statistic released without coarsening) and the coupled B3 on seed 1.
- **Lesson** (MODEL_SELECT selection feedback, not effect sizes):
  - Next to the privacy-trained JOINT_TREE occupation output, the LOCAL-only coarse code read 0.007 / 0.008 below H_LJ.
  - The partly pair-aware code read a further 0.015 / 0.015 below.
  - Without the continuous-mixture control, coarsening and mixing are not separated at the mixed statistic.
  - On DEV_REPORT the pair-aware half (claim J) is not established.

### Family C: SEX-mixture marginalization (investigator C)

The release is the population-share mixture over SEX of CatBoost task models trained with SEX as a training-time input. Without SEX, the same recipe reproduces T_TREE bitwise.

- **C1:** pair 0.6263 / 0.6368.
  - Against T_TREE, its exact SEX-blind control, every proxy reader on the pair and occupation views drops, by 0.011–0.033.
  - On the income view, 5 of 6 readers drop; the seed-1 income OUT_HGB reader rises by 0.011.
  - A matched-cost temperature control moves every reader proxy by at most ±0.0025.
  - The outputs stay essentially one-to-one on the observed inputs, so this is lower reader recovery, not information removal.
  - The reduction is small: occupation recovery falls by 0.022 / 0.011, to 0.621 / 0.629. That is about NEURAL_RAW's level and 0.039 / 0.033 above the occupation bar.
  - The cost falls on men in male-dominated occupation groups (men's guard 0.0044 / 0.0064; seed 1 fails).
- **C2** (marginalized income + JOINT_TREE occupation): pair 0.6277. The family stopped here.

### Selection

PROTOCOL §6 was applied to every fitted configuration (17 trials considered), as adjudicated by the statistical reviewer. Three configurations were promotable: B3CTL, B2 and B2CTL_UNCO.
- All three are EDGE through the shared seed-0 men's guard of 0.0044 (0.88 of the cap), inherited from the JOINT_TREE occupation output.
- B3CTL is also EDGE through its own income Brier on both seeds (0.87–0.88 of the cap).
- With no ROBUST candidate, the lowest mean pair proxy selects B3CTL.
- Had the shared guard been below 0.8 × cap, B2 and B2CTL_UNCO would have been ROBUST, and the rule would have selected B2.

**Disclosure.**
- **The finalist was control-designated.** It was fitted at 17:24:25Z and 17:24:52Z (ledger 41–42) as the coarse-only twin of the coupled B3, after B3 failed the seed-0 income Brier clause.
- **Its α was edge-seeking.** α = 0.5 was the largest value inside the caps on a CALIBRATION screen.
- **An interim promotion record named B2** (17:23:54Z).
  - It applied an exclusion of control-designated configurations. The coordinator wrote that exclusion at 17:23:49Z, after B2's controls were scored and before B3CTL was fitted. It would have excluded the eventual finalist.
  - It also used an incomplete trial set.
  - It was never committed, was deleted at 17:33Z, and is reproduced in `PROMOTION.json` as superseded.
- **The adjudication changed the finalist.** With B3CTL's MODEL_SELECT results known (ledger 1–42), the statistical reviewer ruled that the registered §6 text covers every fitted configuration. This changed the finalist from B2 to B3CTL.

**Forecasts.** These were registered before any final reader fit (`FORECASTS*.json`):

| Forecast | Probability given |
|---|---|
| E | 0.50 |
| B | 0.25–0.30 |
| M_TO | 0.45 |
| J | 0.30 |

The mean Brier score is 0.167 (`FORECAST_SCORES.json`). The outcome exceeded the coordinator's own expectation for B and M.

## 3. Which mechanism changed the released outputs in the intended way

- **Family B: coarsening the income output.** It replaces H_LJ's continuous LOCAL_TREE income output with a 6-level code.
  - On DEV_REPORT the declared readers recover less from the resulting pair: the LOCAL-only coarse twin reads 0.629 / 0.628 against H_LJ's 0.641 / 0.643 (descriptive, not an endpoint), and the finalist reads 0.613 / 0.618.
  - The declared readers still recover more from the finalist's pair than from its occupation output alone: +0.053 / +0.035, ABOVE_ZERO.
  - That gap is reader-relative, not added information. On every observed role and seed, the pair has exactly as many distinct output vectors as the occupation output:

    | Role | Pair (seed 0 / seed 1) | Occupation output (seed 0 / seed 1) |
    |---|---|---|
    | DEV_REPORT | 5,619 / 5,619 | 5,619 / 5,619 |
    | ATTACK_VALIDATION | 2,902 / 2,902 | 2,902 / 2,902 |
    | ATTACK_FIT | 7,530 / 7,526 | 7,530 / 7,526 |

    So on these people the 6-level income output is a function of the occupation vector. The same holds for H_LJ's continuous income, which has one extra distinct pair on ATTACK_FIT.
  - The benefit over H_LJ and over the LOCAL-only twin, which share this occupation output, is therefore a difference in what the declared finite-sample readers extract, not in the information the releases carry on these people.
- **Family C** removed, by construction, the SEX-propensity component of the predictor. The proxy readers' recovery fell by a small amount on MODEL_SELECT, and the outputs stayed essentially one-to-one.
- **Family A's controller** worked as designed: rollback and throttling held the holdout budgets. But at feasible budgets it could not push occupation protection to the bar.

## 4. Did any candidate satisfy the original requirements?

**At the point on DEV_REPORT, yes: the finalist met all 24 utility clauses, the 4 within-SEX guards and the 8 recipient non-worsening conditions.** The utility margins are thin:

| Income clause | Seed 0 | Seed 1 | Cap |
|---|---|---|---|
| Brier excess vs Ucal | 0.0049 | 0.0049 | 0.005 |
| NLL excess vs Ucal | 0.0063 | 0.0060 | 0.010 |
| Accuracy loss | 0.0040 | 0.0096 | 0.010 |

- Simultaneous confidence is unresolved for 25 of the 28 utility and subgroup endpoints. A point pass is not established population preservation.
- The occupation side's utility and guards (men 0.0032 / 0.0040, women −0.0009 / −0.0023) replay H_LJ's published DEV values, because the output is identical.

**Bounded diagnostics** (descriptive; not protocol clauses):
- **Within-SEX income NLL excess against U0.** Men +0.0038 / −0.0005, women +0.0022 / +0.0087. On seed 1 the income cost falls on women.
- **Men's occupation NLL excess.** +0.016 / +0.012, inherited from the JOINT_TREE occupation output and identical for H_LJ.
- **Overall income summed Brier against H_LJ.** The finalist spends +0.0065 / +0.0060 more (marginal 95% intervals exclude 0).

## 5. Against task-only, IBSI, neural and other strong controls

DEV_REPORT pair AUCs, fresh readers:

| Release | Pair AUC | DEV utility, 12 clauses at the point | Finalist's benefit (comparator − finalist) |
|---|---|---|---|
| finalist | 0.613 / 0.618 | pass / pass | — |
| NEURAL_RAW (eligible reference) | 0.640 / 0.638 | pass / pass | +0.027 / +0.0199; clause > 0 on both seeds (seed 1 below 0.020 at the point) |
| H_LJ (eligible reference; same occupation output) | 0.641 / 0.643 | pass / pass | +0.028 / +0.025; clause > 0 on both seeds |
| matched task-only control (TASK_REFIT, same construction) | 0.665 / 0.664 | pass / pass | +0.052 / +0.047; clause > 0 on both seeds |
| LOCAL-only coarse twin | 0.629 / 0.628 | pass / pass | +0.016 / +0.010; seed 1 UNRESOLVED |
| H_JJ (the JOINT_TREE parent) | 0.603 / 0.606 | fails / fails | −0.009 (UNRESOLVED) / −0.011 (BELOW_ZERO); secondary family, COMPARATOR_INELIGIBLE. H_JJ reads lower at the point but is utility-ineligible. |
| PUB_IBSI (adapted) | 0.630 / 0.623 | pass / fails | +0.017 / +0.005, UNRESOLVED; descriptive. PUB_IBSI also fails the men's guard (0.0051 / 0.0091). |
| T_TREE (teacher) | 0.664 / 0.664 | pass / pass | +0.051 / +0.046 (point difference, descriptive) |

**Utility spent against the references (secondary family).**
- Income NLL: the finalist spends +0.0083 / +0.0076 nats more than H_LJ (ABOVE_ZERO) and +0.0008 / +0.0010 more than NEURAL_RAW (UNRESOLVED).
- Occupation NLL: +0.0070 / +0.0028 against NEURAL_RAW (UNRESOLVED); identical to H_LJ.

So the benefit over H_LJ is partly bought with income loss, inside every cap at the point.

## 6. Competitive advantage: established, promising, or not supported?

**Established at the registered clause level against both eligible references on both seeds, for the declared readers. It is positive and below the original 0.020 target.**

The result is conditional on the declared readers and the fitted releases. It is fragile in four ways:
- the utility is at the edge;
- the release was edge-selected and control-designated;
- the occupation side is a published replay;
- the development search and the promotion were adaptive.

The two seeds share people and bootstrap draws, so "both seeds" means robustness to the model seed, not replication. A fresh population would be needed before calling the release a competitive method. This campaign does not open the sealed assessment.

## 7. Evidence for a new ingredient

**Limited.** The only element new to this campaign is the coarse, CALIBRATION-calibrated income code: an equal mix of the stored LOCAL_TREE and JOINT_TREE income scores, paired with the stored JOINT_TREE occupation output. It is a post-processing recombination.
- **Benefit over H_LJ**, which has the same occupation output: positive at the clause level on both seeds and below the original 0.020 target, for the declared readers.
- **Its two parts are not separated.** Coarsening and the JOINT half of the statistic cannot be told apart: claim J is not established, and no continuous-mixture control was run.
- **Coupling is not the active ingredient.** The finalist is uncoupled. On MODEL_SELECT the coupled B2 and its uncoupled twin differ by 0.002, and the reader classes disagree on the sign.
- **The privacy-trained parts were inherited.** Novelty is not claimed.

## 8. The precise bottleneck that remains

1. **Income utility.** The protection is bought with essentially the whole income Brier budget against Ucal: 0.0049 / 0.0049 of 0.005, and seed-1 accuracy loss 0.0096 of 0.010.
   - The JOINT-only code (α = 1) already exceeded the income Brier cap on the CALIBRATION screen and was not run.
   - The coupled B3 exceeded the cap on MODEL_SELECT seed 0, and its pair proxy was 0.0018 above the finalist's, not below it.
2. **Residual reader-level synergy.** The declared readers recover 0.035–0.053 more AUC from the linked pair than from the occupation output alone. Yet on no observed role does the income output add a distinct split of people beyond the occupation output (§3).
3. **Inherited occupation burden.** The men's occupation Brier is 0.0032 / 0.0040 against 0.005, unresolved. It bound families A and C.
4. **Precision.** Paired SEs of 0.004–0.006 put an established 0.020 margin out of reach at the observed effect of 0.020–0.028.

## 9. Not claimed

- Reader results are evidence against the declared readers and their auxiliary sample budget only. They are not guarantees.
- Deterministic outputs are not assumed one-to-one. The one-to-one statements above are observations on the evaluated people, not properties of the construction.
- UNRESOLVED is not equivalence.
- The lookup reader is not population-optimal: it is degenerate on continuous pairs and strong on the 6-level income output.
- There is no information-removal or "added information" claim for the linked pair.
- There is no fairness or subgroup claim beyond the registered guard.
- This campaign did not design the privacy-trained components it reuses.
