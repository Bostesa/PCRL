# Research decision: direct-recovery frontier campaign (`pcrl_direct_recovery_frontier_v1`, DRF1)

First written 2026-10-25 from 04:36Z (`date -u`). Revised 05:04-05:06Z (`date -u` 05:04:03Z before, 05:05:50Z after) following the statistical reviewer's claims review
(`roles/statistical_reviewer/CLAIMS_REVIEW.md`: 11 required corrections, all applied; 11 recommended changes, all adopted).

All results are development evidence on the Florida ACS 2019 development cohort (admitted public-use PUMS records). SEX is the
dataset's recorded demographic variable; "recovery" means how well fitted readers infer it from what the two recipients receive.
Every recovery number is read on adaptively reused MODEL_SELECT people for the declared reader bank.

Conventions and sources:
- Benefit = comparator pair AUC - candidate pair AUC; positive favours the candidate (lower recovery). `se` is the rules' se_agg.
  "Resolved" means x - 2 se > 0 and both version means > 0 (for the 0.020 target, the same with 0.020 in place of 0).
- The two model versions are procedure repeats that share every training row and the MODEL_SELECT people; they are not replications.
  Repeated reader splits and reader seeds on the same people are not replications either. Intervals are conditional development
  summaries and do not correct the adaptive search.
- Records: `ENDPOINTS.json` and `FRONTIER.json` / `FRONTIER.csv` (machine-readable), `TABLES.md`, `claims/*/DECISIONS.json`,
  `claims/*/D2R.json` and `claims/CAMPAIGN_SUMMARY.json`. The D2R records were written after the write-once decision records, which
  still read "D2R NOT_RUN".

## Decision

**No method claim.** The registered headline reading (DRF_R1, card DR_DRF_R1) stops at **failure stage 3: the legal mapping keeps the
original usefulness conditions but ties its controls** (`STAGE_3_KEEPS_UTILITY_TIES_CONTROLS`, class `FEASIBLE_NO_RELIABLE_GAIN`).
The adaptive revision R2 reads the same. D2 (a component gain beyond both family controls), D3 (superiority to the strong references)
and D4 (the 0.020 material target) are met in no reading.

| Reading | Card (status) | D1 usefulness | D2: CTRL frontier / OBJ_MI | FC vs SUCC | D3 passed | D4 | Stage | D2R |
|---|---|---|---|---|---|---|---|---|
| DRF_V1 | DR_DRF_v1 (registered; superseded by the pre-outcome R1 card) | CAND_INFEASIBLE | not assessed | not assessed | not assessed | not assessed | 2: optimistic succeeds, legal mapping fails | not run |
| DRF_R1 | DR_DRF_R1 (registered headline; MODEL_FIT trigger) | TASK_FEASIBLE (point) | primary cost only / NOT_SEPARATED | NOT_RESOLVED | NEURAL_RAW | NOT_MET | 3 | ROBUSTNESS_NOT_CONFIRMED |
| DRF_R2 | DR_DRF_R2 (adaptive; MODEL_SELECT trigger) | TASK_FEASIBLE (point) | primary cost only / NOT_SEPARATED | NOT_RESOLVED | NEURAL_RAW, TWJ | NOT_MET | 3 | ROBUSTNESS_NOT_CONFIRMED |

Both v1 and R1 are registered: their cards were committed at 03:14:30Z and 03:27:33Z, before the first MODEL_SELECT read
(03:34:32Z). R1 is the headline, and v1 is superseded (`claims/CAMPAIGN_SUMMARY.json`, flag `MULTIPLE_REGISTERED_READINGS`).

**Stage sensitivity (not registered).**
- The registered rule judges the optimistic point in-sample. On the nested mode no optimistic point is useful, and under that
  reading v1 and R1 would both be stage 1.
- The nested label uses the held-out exact AUC*, not LOOKUP. That measure is strongly optimistic for fine partitions: for the twin
  it is 0.7475 nested exact, against 0.6303 held-out LOOKUP and 0.6812 in-sample. So the nested comparison favours coarser
  releases.

Recovery on the registered cells (pair / income view / occupation view, versions 0 / 1; `TABLES.md` 2):

| Release | Pair | Income view | Occupation view |
|---|---|---|---|
| DR_AUC_R1 (candidate, registered) | 0.6049 / 0.6104 | 0.5276 / 0.5394 | 0.5779 / 0.5803 |
| DR_MI_R1 (previous objective, same family) | 0.6045 / 0.6259 | 0.5308 / 0.5418 | 0.5779 / 0.5879 |
| DR_CTRL_HI_R1 (SEX-blind coarsening bracket) | 0.6424 / 0.6409 | 0.5550 / 0.5518 | 0.6043 / 0.6039 |
| DR_AUC_R2 (candidate, adaptive) | 0.6069 / 0.6134 | 0.5360 / 0.5380 | 0.5692 / 0.5851 |
| DR_MI_R2 (adaptive) | 0.6098 / 0.6183 | 0.5326 / 0.5364 | 0.5829 / 0.5868 |
| TWB_C64 (the family origin, a task-only twin) | 0.6476 / 0.6496 | 0.5525 / 0.5526 | 0.6340 / 0.6335 |
| SUCC (the unchanged strong start) | 0.6047 / 0.6122 | 0.5251 / 0.5265 | 0.5612 / 0.5791 |
| RETIRED_B3CTL | 0.6025 / 0.6053 | 0.5345 / 0.5277 | 0.5612 / 0.5791 |
| H_LJ | 0.6193 / 0.6231 | 0.5433 / 0.5455 | 0.5612 / 0.5791 |
| NEURAL_RAW | 0.6262 / 0.6356 | 0.5422 / 0.5387 | 0.6260 / 0.6338 |

## Answers

### 1. What changed operationally from the closed finite-routing work?

FRC1's PRC re-routed people among the twin's finite cells by tilting toward a pi-weighted plug-in dependence objective at a swept
strength, and screened usefulness afterwards. FSA1 then showed that a better search lowered that surrogate without lowering held-out
recovery. DRF1 changed three operations and kept the family otherwise fixed:
- **Objective.** The exact optimal-reader AUC of the finite law of (emitted wire, SEX), for the linked pair plus each view
  (likelihood-ratio order, ties 1/2, identical emitted vectors merged), under the propensity-estimated law. This is the reported
  recovery quantity itself, not a dependence proxy. The derivation and enumeration checks on small laws (ties, unequal class
  frequencies, merged outputs, linked pairs) are in `roles/investigator_a/`.
- **Constraints inside the search.** Eight explicit spends over the identity (income and occupation NLL, summed Brier and accuracy
  loss, and the men's and women's occupation Brier). Each is capped at 0.6 times the twin's published MODEL_SELECT slack to the
  original cap (`dr/BUDGETS.json`, frozen before any fit). There is no strength sweep and no post hoc utility screen.
- **Mapping class.** A deterministic many-to-one lookup from 64 / 256 legal fine cells (twin codeword x propensity quartile, functions
  of the six fields only) to at most 16 / 64 atoms, emitting teacher-mean (B1) tables. The emitted vectors are the ones the task costs
  are computed on. The identity is the admitted twin TWB_C64 and is always the start.

The isolating comparison is the same family, cells, moves, constraints, budgets, data exposure and search budget with the previous
dependence objective (DR_MI: PRC's pi-weighted plug-in MI, pair + income + occupation). Beside it:
- an untargeted SEX-blind coarsening path of the twin (greedy least-total-NLL atom merges), bracketed at the candidate's nested
  MODEL_FIT total-NLL spend before any MODEL_SELECT read;
- the unchanged strong start SUCC (the family change FC, reported separately);
- the twin and the nine strong references.

No residual, Fisher penalty, guard estimator or strength sweep was used. The design overlaps privacy funnels and the earlier SCM /
FRC1 finite routing; no novelty is claimed (D6 NOT_CLAIMED).

### 2. Did the optimistic construction find a useful region, and under which assumptions?

**Yes in-sample, under favourable assumptions; not on the nested held-out estimator.**
- **OPT_L** used the legal fine cells. It was fitted on the true-SEX law of the MODEL_FIT people themselves, in-sample, with the same
  frozen budgets estimated from realized MODEL_FIT labels; the mapping is deterministic.
  - In-sample exact pair AUC* was 0.6213 / 0.6222 within budget: 0.0243 / 0.0226 below the SEX-blind coarsening path at matched spend
    on the same law and mode, with single views within 0.010 (`USEFUL_OPTIMISTIC_POINT`).
  - On the nested MODEL_FIT estimator (five household folds, the same procedure refitted per fold), OPT_L was not useful.
    - Its held-out exact pair AUC* (0.6868 / 0.6794) was 0.044 / 0.041 above the SEX-blind coarsening path's at matched nested
      total-NLL spend (`held_out_survives: []`). Its held-out LOOKUP pair AUC was 0.6116 / 0.5979.
    - Its nested spends were also 2.6-4.0 times the budgets: occupation NLL 0.0277 / 0.0262 against 0.0097 / 0.0100, men's
      occupation Brier 0.0119 / 0.0114 against 0.0035 / 0.0029. So no held-out point is within budget either; the rule's nested
      label does not itself test the budgets.
- **OPT_S** routed by the true SEX, an oracle that is not deployable. In-sample it reached pair AUC* 0.6400 / 0.6344 and was not
  useful.
  - Its search space contains OPT_L's, so this is an optimizer shortfall (disclosed by the verifier), not evidence about the
    objective.
- Both are MODEL_FIT-only diagnostics. They gate nothing and are not candidates.

### 3. Could a legal mapping reproduce it on separated development households?

**The first registered mapping (v1) did not keep usefulness. The registered headline R1, a revision decided on MODEL_FIT evidence
before any MODEL_SELECT outcome, kept all 28 conditions at the point. It did not reach below the strong start, and it is not a
reproduction of the optimistic point.**
- **v1 (DR_AUC and DR_MI, realized-label constraints).** The MODEL_FIT nested estimator showed that the in-sample constraints did not
  transfer: nested occupation NLL spend about +0.028 against a 0.0097 budget, men's occupation Brier +0.012 against 0.0035.
  - On MODEL_SELECT both failed the original conditions: the men's occupation guard was +0.0080 / +0.0111 against the 0.005 cap, and
    occupation NLL also failed (occupation Brier too, in some versions).
  - The reader stage was stopped and the releases kept (failure stage 2).
- **R1 (expected-cost constraints under the frozen out-of-fold teachers).** It was decided on that MODEL_FIT evidence, before any
  MODEL_SELECT read.
  - DR_AUC_R1's nested spends were 1.04-1.12 times the budgets on its exhausted metrics: occupation NLL 1.11, men's guard 1.12,
    income Brier 1.04 (income NLL 0.99). DR_MI_R1 reached up to 1.21.
  - On MODEL_SELECT, DR_AUC_R1 met all 28 original conditions at the point in both versions (worst value / cap 0.564 / 0.692; men's
    guard +0.0028 / +0.0035). That is point eligibility only: 20 of 28 conditions pass the confidence conjunction, none is violated,
    and it is not reserve-qualified.
  - The pass carries the disclosure that its budgets were tuned on the twin's published MODEL_SELECT aggregates, so it is not an
    independent test of constraint transfer.
  - The MODEL_FIT nested LOOKUP prediction (0.6028 / 0.6085) was within 0.0021 / 0.0019 of the MODEL_SELECT reader result
    (0.6049 / 0.6104). That is one check of the nested estimator, not a validation.
  - On MODEL_SELECT it is resolved less recoverable than the SEX-blind coarsening bracket at matched total NLL (+0.0340, se 0.0096),
    but only on that primary cost (answer 4).
  - This is not a reproduction of the optimistic point, which was useful in-sample only and not on the nested estimator.
  - It did not reach below SUCC, the unchanged strong start: +0.0008 (se 0.0120). Both single-view recipient conditions fail against
    SUCC (income version 1 +0.013, occupation version 0 +0.017), and it spent 0.0097 (0.0033) more total NLL than SUCC, resolved.

### 4. Did direct recovery optimization beat the old objective and untargeted controls?

**Not the old objective; the untargeted control only on its primary cost.**
- **Against the old dependence objective** (DR_MI, same family, constraints, exposure and search), it was not separated anywhere:
  - R1 +0.0076 (se 0.0117; version means -0.0004 / +0.0156);
  - R2 (search tripled, adaptive) +0.0039 (0.0082; +0.0029 / +0.0049);
  - the R1 robustness cells +0.0054 (0.0110).
  - **Pair AUC* on MODEL_FIT.** The two objectives reach almost the same value. In R2 the in-sample estimated-law values are
    0.6212 vs 0.6216 and 0.6205 vs 0.6258.
  - **Where the objective difference comes from.** In version 0 most of the J difference comes from the occupation view (0.0030 of
    0.0040). In version 1 most comes from the pair (0.0053 of 0.0092; occupation 0.0034).
  - **Budgets.** Both objectives exhaust the same three budgets: in every R1 / R2 fit, occupation NLL (0.999-1.000 of budget), the
    men's occupation Brier (0.984-1.000) and income Brier (0.990-0.999) are at or near their limits. The constraint with the highest
    ratio differs by objective in version 1 (R1: DR_AUC the men's guard, DR_MI occupation NLL; R2: the reverse).
- **Against the SEX-blind coarsening bracket** (DR_CTRL_HI) the pair benefit is resolved: R1 +0.0340 (0.0096), R2 +0.0322 (0.0120).
  On the R1 robustness cells it was +0.0308 (0.0110), `ROBUSTNESS_CONFIRMED` for that comparator. But it is separated on the
  primary cost only (`SEPARATED_ON_PRIMARY_COST_ONLY`):
  - DR spends resolved more income NLL (+0.0061, se 0.0018) and income Brier (+0.0038, 0.0013) than the bracket, and resolved less
    occupation NLL and men's occupation Brier.
  - The coarsening points themselves fail the men's guard on MODEL_SELECT.
  - The pair benefit is therefore not shown at equal cost on every usefulness metric; part of it may reflect a re-allocation of
    spend from occupation to income (not tested).
- **D2 is `NOT_SEPARATED` in both readings.**
- **Beside D2, never sufficient.** Against the twin TWB_C64 the candidate is far less recoverable: +0.0410 in R1 and +0.0385 in
  R2, resolved, and `ROBUSTNESS_CONFIRMED` on the robustness cells. That is an attribution against a weak task-only twin.

### 5. Did any release beat the strong references under the unchanged conditions?

**No.** D3 is `NOT_SUPERIOR` in both readings.
- **DR_AUC_R1** is resolved below NEURAL_RAW only: +0.0232 (0.0110), recipient gates pass.
  - H_LJ +0.0135 (0.0124) and B2TWIN +0.0076 (0.0110) are point only, with the occupation recipient gate failing.
  - SUCC +0.0008 (0.0120) and RETIRED_B3CTL -0.0038 (0.0140).
  - Against TWJ, W, TR100 and TT100 it is resolved but fails a recipient gate.
- **DR_AUC_R2 (adaptive)** passes NEURAL_RAW +0.0207 (0.0100) and TWJ +0.0238 (0.0101).
  - H_LJ +0.0110 (0.0097) is point only.
  - SUCC -0.0017 (0.0103) and RETIRED_B3CTL -0.0063 (0.0110).
- **D4**, a resolved 0.020 against both NEURAL_RAW and H_LJ, is `NOT_MET`: NEURAL_RAW point only, H_LJ not met.
  - At this design's se_agg of about 0.010, the registered rule needs a point benefit near 0.040 against H_LJ. That is a consequence
    of this resolution, not a universal threshold.

### 6. What survived reader refitting?

- **Reader refits (computational reproducibility: the same cells, seeds and data, not robustness evidence).**
  - The verifier's independent refits of six winning readers (HGB15, CATI01 twice, LOG_LR, ET25, MLP6432) reproduced their SCORE and
    validation AUCs exactly, with scores bitwise equal to the cache. Six more R2 refits were also exact.
  - Six custody readers (`custody/READER_CUSTODY_*.json`: pair winners of DR_AUC_R1 in two cells, DR_MI_R1 and TWB_C64, plus
    DR_AUC_R1's occupation and income winners) were refitted exactly as the grid fits them. Their oriented SCORE scores equal the
    grid cache, and they re-score bitwise after reloading.
- **R1 on the robustness splits R3 and R4** (24 robustness cells per comparator; reader seeds k2 / k3 beside k0 / k1; release and
  every selection unchanged; `claims/DRF_R1/D2R.json`): `ROBUSTNESS_NOT_CONFIRMED`.
  - **Confirmed (pair benefit and robustness recipient gates):** over the SEX-blind coarsening bracket (+0.0308, 0.0110; still
    separated on total NLL only) and over the twin (+0.0388, 0.0084; never sufficient).
  - **Not confirmed:** the edges over DR_MI_R1 (+0.0054, 0.0110), NEURAL_RAW (+0.0197, 0.0103; no longer resolved) and H_LJ (+0.0123,
    0.0126).
  - **Sign change:** against SUCC (+0.0008 to -0.0002).
  - **Recipient failures:** a single-view recipient condition fails on the robustness cells against SUCC and RETIRED_B3CTL.
  - The refit effectiveness reads `REFIT_GENUINELY_EFFECTIVE`.
  - The record lists SUCC twice in `beside`, so its flags and caveat text repeat SUCC. This is a cosmetic defect of the frozen rules
    code (`PROTOCOL_DEVIATIONS.md` item 4); the labels and sets are correct (verifier, hand-backs 2 and 3).
- **R2 (adaptive) on the same robustness splits** (`claims/DRF_R2/D2R.json`; bank written after the R2 decisions and before any
  robustness output): `ROBUSTNESS_NOT_CONFIRMED`.
  - **Confirmed (pair benefit and robustness recipient gates):** over the CTRL bracket (+0.0302, 0.0118; still separated on total
    NLL only) and over the twin (+0.0381, 0.0102; never sufficient).
  - **Not confirmed:** DR_MI_R2 (+0.0040, 0.0092), NEURAL_RAW (+0.0190, 0.0110), H_LJ (+0.0116, 0.0121) and RETIRED_B3CTL
    (-0.0115).
  - **SUCC:** -0.0009 (0.0108), with the income recipient condition failing.

### 7. What precise limitation now prevents a method claim?

Within this bounded finite family, at the unchanged usefulness limits, three findings block a method claim. The constraints at
budget are occupation NLL, the men's occupation Brier guard and income Brier (in-sample MODEL_FIT ratios 0.98-1.00 in every R1 / R2
fit).
- **No objective effect is resolved.**
  - DR_AUC vs DR_MI: +0.0076 (se 0.0117; version means -0.0004 / +0.0156) in R1, +0.0039 (0.0082) in the adaptive R2, and +0.0054
    (0.0110) on the R1 robustness cells.
  - At R1's se_agg a true 0.010 benefit would be resolved about 13% of the time (about 22% at R2's), so a small effect is neither
    shown nor excluded.
  - Both objectives exhaust the same three budgets (occupation NLL, the men's occupation Brier, income Brier).
- **The separation from SEX-blind coarsening holds on total NLL only.**
  - At the bracket DR spends resolved more income NLL (+0.0061, se 0.0018) and income Brier (+0.0038, 0.0013), and resolved less
    occupation NLL and men's occupation Brier (`SEPARATED_ON_PRIMARY_COST_ONLY`).
  - So the +0.034 pair benefit is not shown at equal cost on every metric.
  - The coarsening points themselves fail the men's guard on MODEL_SELECT.
- **The family's MODEL_SELECT recovery is not below the strong start's, at a higher cost.**
  - DR's pair AUC is 0.605-0.613, against SUCC 0.605 / 0.612 and RETIRED_B3CTL 0.603 / 0.605: not below either. It spends resolved
    more total NLL than SUCC.
  - Against SUCC its occupation view is 0.017 more recoverable in version 0 and its income view 0.013 more in version 1, both beyond
    the 0.010 allowance (the other versions are within it).
  - In-sample, the search stopped with occupation NLL, the men's guard and income Brier at budget: within this family, these
    budgets and this local search the views were not coarsened further. Whether another mapping in the family could coarsen them
    within budget was not tested.

**What a method claim would need.** A method claim (METHOD_CANDIDATE) would need D2 and D3.
- **D2:** a resolved pair benefit with the recipient gates beyond both family controls, with no secondary metric resolved higher.
- **D3:** a resolved benefit with the recipient gates against every point-eligible reference of the nine, SUCC, RETIRED_B3CTL and
  H_LJ included.
- **D4,** the material target (a resolved 0.020 against NEURAL_RAW and H_LJ), is reported separately and is not implied by either.
- Any of these would then need confirmation on households not used here.

## Uncertainty: what each statement covers (RULES.md 10)

- **Household sampling.**
  - The reader-cell SEs are household bootstrap SEs over SCORE households, with the release, readers and selections fixed. Their
    weighted RMS (se_hh) is the household part of se_agg, an upper bound under correlated cells.
  - The usefulness SEs are household-clustered over MODEL_SELECT.
  - Both are conditional on everything fitted, and neither covers refitting the release or the readers.
- **Fitting variation.**
  - se_rs, the spread of the four version x repetition block means (version, reader split and reader seed, about one degree of
    freedom each), is the second part of se_agg = max(sqrt(se_hh^2 + se_rs^2 + se_mc^2), 0.001), with se_mc = 0 here.
  - The two versions are procedure repeats on the same rows, so "both versions" is a robustness requirement, never replication.
  - The nested MODEL_FIT estimator describes the procedure at about 80% of MODEL_FIT, not the deployed single fit. Its fold spread
    mixes overlapping training sets and is never used as an SE.
  - Optimizer-seed dependence is not covered by any SE. The change from R1 to R2 was only a larger search over a superset of seeds,
    yet it moved MODEL_SELECT pair AUC by +0.0020 / +0.0030 for DR_AUC and +0.0053 / -0.0076 for DR_MI. That variation is of the same
    order as the objective comparison itself (+0.004 to +0.008). The R1-to-R2 change also includes the readers refitted on the
    changed releases.
- **Numerical randomness.**
  - The optimizer is deterministic given its seeds, so there is no Monte Carlo term.
  - Reader seeds enter through the block means.
  - Seed equalities on few-valued views are explained and counted (`grid_validity`: 82 explained equalities per reading; the
    robustness refits read `REFIT_GENUINELY_EFFECTIVE`).

## Registered expectations against outcomes (PROTOCOL 12; card predictions)

- **PROTOCOL 12 (registered, before any label read).**
  - P(DR_AUC G1-feasible in both versions) about 0.6: v1 no, R1 yes.
  - P(DR_AUC's in-sample objective below DR_MI's in both versions) about 0.8: v1 no (version-1 optimizer shortfall), R1 and R2 yes.
    The v1 card words this as the in-sample exact pair AUC* rather than the objective; the outcome is the same under both wordings
    (v1 no, R1 and R2 yes).
  - P(D2) about 0.10, P(D3) about 0.03, P(D4) about 0.02: none met.
  - "Most likely: the men's occupation guard binds first and DR_AUC lowers the twin's grid pair AUC by 0.01-0.03, mostly not
    separated from the coarsening control or not reaching SUCC":
    - the constraint with the highest ratio for DR_AUC was occupation NLL in four of its six fits, the men's guard in one (R1
      version 1) and income Brier in one (v1 version 1);
    - the reduction from the twin was larger (0.041 / 0.039);
    - it did not reach below SUCC;
    - the coarsening control was separated on its primary cost only.
- **R1 card predictions (registered; card committed before any MODEL_SELECT outcome).**
  - "DR_AUC_R1's MODEL_FIT exact pair AUC* lies above DR_AUC's, and its nested held-out spends lie within about 1.5 x its budgets
    (probability 0.6)": yes (0.6193 / 0.6259 against 0.6089 / 0.6209; nested spends at most 1.12 x).
  - P(DR_AUC_R1 meets all 28 conditions on MODEL_SELECT in both versions) 0.5: yes.
  - P(D2) 0.08, P(D3) 0.02, P(D4) 0.01: none met.
- **R2 card predictions (adaptive: written after the R1 MODEL_SELECT outcomes).**
  - "R2 lowers J_auc below R1 by 0.005-0.02": version 0 by 0.0021 (below the range), version 1 by 0.0114.
  - The same for DR_MI_R2 and J_mi: lowered by 0.0015 / 0.0022.
  - "In-sample pair AUC* changes by at most 0.005": DR_AUC +0.0019 / -0.0054 (version 1 exceeds it slightly); DR_MI_R2 +0.0017 /
    -0.0031.
  - "Both releases keep G1 (probability 0.75)": yes.
  - "The DR_AUC vs DR_MI difference stays unresolved (0.75)": yes.
  - "No change of D3 or D4": D4 unchanged; D3 passed TWJ in addition, still `NOT_SUPERIOR`.

## Not claimed

- Protection against arbitrary readers, or a population privacy guarantee from exact finite-law calculations.
- Confirmation, or replication from versions, repetitions or reader seeds.
- Superiority from a narrower comparison, or a mechanism contribution from an unmatched comparison.
- Competitiveness from a gain against the weak twin.
- Novelty, or anything about the closed roles.
- Anonymity: public account and commit metadata may identify the owner.
