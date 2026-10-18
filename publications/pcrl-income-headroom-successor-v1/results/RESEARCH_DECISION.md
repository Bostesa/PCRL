# Decision: PCRL income-headroom successor development campaign

**Scope.** Statistical machine-learning research on public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for
Florida, in the existing development roles only. The service predicts an income class (2 classes: PINCP x ADJINC / 10^6 above
$50,000) and an occupation group (5 classes) from AGEP, WKHP, COW, SCHL, MAR and POBP; each recipient receives its own float64
probability vector and argmax decision. SEX is the dataset's recorded demographic column; separate evaluation readers estimate it
from the released predictions (each output alone and the linked pair). SEX is never a service input.

**Status of the evidence.** Exploratory development on previously used records, not confirmation. The successor was selected
adaptively on MODEL_SELECT (an adaptively used role) and evaluated once on DEV_REPORT, which has now been read 14 times in the
lineage (this campaign: one coordinator evaluation at 03:08:07Z and one separately logged verifier replay at 03:17:54Z). Seeds 0
and 1 are two fits on the same people, not replications. Conditional intervals do not cover the adaptive search. Every
recovery statement is evidence against the declared readers and auxiliary budget only.

## Answer in one paragraph

The campaign found a successor, **H3_STRONG_LJDIR_PB**, that keeps income summed-Brier and NLL headroom on DEV_REPORT (income
summed-Brier margins of 3.8-5.0 SE and NLL margins of 5.8-7.0 SE, where the retired candidate's binding Brier clause had
0.03-0.04 SE; household-bootstrap SEs, the reserve rule's max(successor, H_LJ) SE) while, against the declared readers, its
linked pair reads lower than both eligible references at the point (DEV benefits +0.014 / +0.017 against H_LJ, both benefit
clauses passing at c1 as condition values of claim P; +0.013 / +0.012 against NEURAL_RAW, with the seed-1 clause unresolved). It
is **not a competitive successor** under the frozen conjunction: claim P fails on the NEURAL_RAW seed-1 bound and, under the
enhanced readers only, on the seed-1 income recipient clause against H_LJ (reader-sensitive); the registered income reserve
erodes on the seed-1 accuracy clause (0.83 SE; claim R2 `RESERVE_ERODED`, R3 `INCOME_CONFIDENCE_UNRESOLVED (4 of 12)`).
Reported beside the composite, not part of it: the inherited JOINT_TREE occupation constraints still bind (claim O). It is a
**useful piece (income Brier and NLL headroom kept on DEV_REPORT)**, not a robust development candidate; "headroom kept" is a
descriptive statement about 8 of the 12 income clauses on already-used records, not a registered label. At the plug-in, the
governing (ENHANCED) planning requirement for claim P's conjunction (D1) fell from about 4.6 million to about 129,000 households,
still far above the about 17,510 that exist for a same-population test. As for the retired candidate, the requirement is not
identified by these data: the four binding clauses sit 0.40-0.87 SE from their caps, and at a true slack of 0 D1 never reaches
80% power (limit 0.16).

## What was explored, and why

The validation of the retired candidate showed that a coarse income output lowered linked-pair recovery but left income Brier
0.03-0.04 SE from its cap. The registered question was whether a different income learner or output construction could keep lower
recovery with real quality margin. Before any fit, the inherited occupation output was quantified (`INHERITED_OCCUPATION_MARGINS.json`):
its men's guard sat 0.32 / 0.62 SE from its cap on MODEL_SELECT, so an occupation-side family was added as a recorded expansion.

| Family | Investigator | Direction | Configurations | Outcome |
|---|---|---|---|---|
| H | separate agent | a stronger income task learner (HGB stack) with held-out checks (no gain over T_TREE on CALIBRATION, +0.0003 worse; 0.0016 / 0.0013 on MODEL_SELECT), then a budgeted transplant of the stored LOCAL / JOINT_TREE income corrections | 9 | H3 nominated |
| P | separate agent | income-only corrections trained against linked-pair critics with the occupation output frozen, then an explicit label-free distortion budget | 14 (allotment raised from 12 by the coordinator) | P3 qualified, not chosen |
| O | separate agent | SEX-balanced, information-preserving recalibration of the inherited occupation output | 7 (OC_BFIT added after the O1 fits, disclosed in O's card) | raises the men's guard margin on MODEL_SELECT (0.32 / 0.62 → 1.83 / 1.62 SE; below its 2-SE target); UTILITY_HEADROOM_ONLY alone |
| coordinator | main session | P income + O occupation combinations | 4 | qualified, not chosen (lower minimum income-reserve ratio than H3) |

34 of 36 seed-specific configurations; every configuration, control, failure and decision is in `EXPERIMENT_LEDGER.csv`. Family
H's registered Step-1 consequence (no genuine headroom → Step 2 as an attribution test only; H's CARD.md) was triggered on
CALIBRATION but not on MODEL_SELECT.

## Which revisions taught something

- **P1 → P2 → P3.** On seed 0 (P1 was fitted on one seed only), P1 (strong critic pressure with a per-epoch projection) was beaten
  at the same budget by simply scaling the stored JOINT_TREE income corrections (PC_JSCALE); investigator P's interpretation, not
  tested further, is that the training dynamics injected in-sample, critic-chasing corrections. P2 (JOINT_TREE strength, one end
  projection) qualified; P3 (same training, larger budget) read 0.6072 / 0.6113 on the development pair proxy with income Brier
  still at or below Ucal's. Matched-cost temperature shrinkage of the twin did nothing on seed 0 (0.6362 at P2's cost, 0.6363 at
  P3's; one-seed controls).
- **H1 → H2 → H3.** H1's registered budget rule was stopped by a noisy accuracy term (β = 0; seed 0 only). H2 (decision-preserving
  residual; seed 0 only) reached a low pair proxy (0.604) but came within 0.00003 of the income Brier cap (+0.00497 against 0.005;
  reserve not met) and pushed the income view above the recipient screen (+0.0113 against NEURAL_RAW): abandoned. H3
  (precision-aware budget) spent part of the headroom: 0.6101 / 0.6136 at income Brier −0.0023 / −0.0033 vs Ucal.
- **O1.** Ordinary calibration worsens the men's guard (OC_TCAL 0.00513 / 0.00509, failing); a bias fitted on MODEL_FIT helps on
  seed 0 only; the SEX-balanced objective adds a stable transfer of guard slack from women to men (men 0.00166 / 0.00230, 1.83 /
  1.62 SE) without raising occupation-view recovery. It stops short of 2 SE because the in-sample gap differs from MODEL_SELECT's.

## The registered selection

Under the frozen nomination rule (`roles/statistical_reviewer/NOMINATION_RULE.md`; applied in `NOMINATION.json`), six
configurations qualified (point eligibility, two-seed reserve, every-cell superiority over both references and B2TWIN, recipient
screens). Four were within 0.005 of the best development pair score (CMB_P3JR_O1SEXBAL 0.60862, P3_JR_B60 0.60923, PC_JSCALE_B60
0.61012, H3_STRONG_LJDIR_PB 0.61185); the rule then prefers the largest minimum income-reserve ratio, which was H3's (2.862
against 2.677, 2.677 and 2.212). So the nominee reads 0.0032 above the best development score in exchange for more income reserve
("robust reserve before point AUC"). The occupation-repaired combination CMB_P3JR_O1SEXBAL had the best development score but lost
to H3 on the minimum income-reserve ratio (2.677 < 2.862); its equal ratio and larger fit count only ordered it behind P3_JR_B60.
The reviewer's freeze check confirmed the choice (`roles/statistical_reviewer/FREEZE_CHECK.md`). Matched task-only twin:
HC_STRONG; declared component control: HC_TTREE_LJDIR_PB (the same step on the old T_TREE base). The nominee's MODEL_SELECT flags,
carried into FINALIST_LOCK: READER_UNSTABLE; its seed-1 margin over B2TWIN was thin (minimum cell +0.0012); simultaneous utility
confidence unresolved for 8 of 28 conditions.

## DEV_REPORT results (one logged evaluation; ENHANCED readers decide; ORIGINAL beside them)

| Release | Pair-view SEX AUC s0 / s1 (ENHANCED) | Income-view SEX AUC s0 / s1 (ENHANCED) | Income Brier vs Ucal | Income accuracy loss |
|---|---|---|---|---|
| **SUCCESSOR (H3)** | **0.6272 / 0.6263** | 0.5419 / 0.5438 | −0.00164 / −0.00124 | +0.00017 / +0.00707 |
| H_LJ | 0.6414 / 0.6428 | 0.5496 / 0.5378 | −0.00154 / −0.00109 | +0.00135 / +0.00892 |
| NEURAL_RAW | 0.6400 / 0.6386 | 0.5465 / 0.5509 | +0.00361 / +0.00335 | 0.00000 / +0.00454 |
| twin HC_STRONG | 0.6557 / 0.6578 | 0.5626 / 0.5682 | −0.00546 / −0.00441 | +0.00118 / +0.00791 |
| component control HC_TTREE_LJDIR_PB | 0.6293 / 0.6236 | 0.5420 / 0.5442 | +0.00001 / +0.00020 | +0.00236 / +0.00808 |
| B2TWIN | 0.6314 / 0.6278 | 0.5531 / 0.5518 | +0.00226 / +0.00223 | +0.00135 / +0.00892 |
| RETIRED_B3CTL | 0.6130 / 0.6168 | 0.5413 / 0.5454 | +0.00492 / +0.00494 | +0.00404 / +0.00959 |

Claims (the frozen conjunctions of `roles/statistical_reviewer/CLAIMS_DRAFT.md`, encoded in `ihs/claims.py`; c1 = 2.3949991):

| Claim | ENHANCED | ORIGINAL | What it rests on |
|---|---|---|---|
| E, development point eligibility | DEV_POINT_ELIGIBLE | same | all 24 utility, 4 guard and 8 recipient points pass; simultaneous confidence unresolved for 17 of 28 |
| R, utility reserve on DEV | RESERVE_ERODED; income confidence unresolved 4 of 12 | same | 10 of 12 income margins meet 1.7267 SE; seed-1 accuracy 0.83 SE |
| P, pair superiority over both references | NOT_SUPERIOR_TO_ALL_ELIGIBLE [READER_SENSITIVE] | NOT_SUPERIOR_TO_ALL_ELIGIBLE | H_LJ: +0.0141 (bound +0.0092) / +0.0166 (+0.0057); NEURAL_RAW: +0.0128 (+0.0004) / +0.0123 (−0.0015); the per-reference legs are condition values, not separate conclusions |
| N, recipient non-worsening | NON_WORSENING_UNRESOLVED | NON_WORSENING_ESTABLISHED | seed-1 income vs H_LJ: ENHANCED +0.0061, bound 0.0217 (side BOTH); ORIGINAL −0.0070, bound 0.0013 |
| M, against the matched twin | TASK_ONLY_ATTRIBUTION_ONLY | same | twin benefit +0.0285 (bound +0.0220) / +0.0316 (+0.0209); step 2 is P |
| C, against the component control | NOT_ESTABLISHED | same | +0.0020 / −0.0026 (ENHANCED); clause intervals −0.0026 to +0.0066 and −0.0134 to +0.0082 |
| T, the 0.020 target | TARGET_NOT_ESTABLISHED | same | all four reference benefits 0.012-0.017; this does not show the target false |
| O, inherited occupation constraints (beside the composite) | INHERITED_CONSTRAINTS_BIND | same | 6 conditions below the R2 rule, in DEV SE units: men's guard 0.86 / 0.55; occupation NLL vs Ucal 0.40 / 1.48; vs U0 0.87 / 1.53 |
| composite (E ∧ R2 ∧ P) | NOT_COMPETITIVE | NOT_COMPETITIVE | P fails; R2 eroded |

Controls: every planted-SEX and shuffled-label control was inside its band on ATTACK_VALIDATION and DEV_REPORT (CV holds).
Descriptive: the successor reads 0.0042 / 0.0015 below B2TWIN (marginal intervals include 0; `NO_DESCRIPTIVE_GAIN_OVER_B2TWIN`)
and 0.0142 / 0.0094 above the retired candidate, with 0.0066 / 0.0062 less income Brier than the retired candidate. Every
ENHANCED winner sits within 0.010 of its ATTACK_VALIDATION runner-up (near ties are the rule with 16 recipes and their ignore-one
alternatives), and 27 of the 48 winners differ between the banks (`report/DEV_REPORT_RESULTS.json`).

## The questions of the brief

1. **Did income quality headroom improve?** Yes, for Brier and NLL, and it survived on DEV_REPORT: income Brier vs Ucal −0.0016 /
   −0.0012 (3.8-5.0 SE of margin), NLL 5.8-7.0 SE. Seed-1 income accuracy (0.0071) did not keep the reserve (0.83 SE); the
   tree-based seed-1 releases in the bank show comparable accuracy losses against the seed-1 U0 on DEV (H_LJ and B2TWIN +0.0089,
   twin +0.0079, retired +0.0096), NEURAL_RAW a smaller one (+0.0045).
2. **Did lower recovery survive without spending that reserve?** Partly, and only against the declared readers. Both pair-benefit
   clauses against H_LJ pass at c1 (bounds +0.0092 / +0.0057; condition values inside P, not separate conclusions), but under the
   ENHANCED readers the seed-1 income recipient clause against H_LJ is unresolved (+0.0061, bound 0.0217), and with matched
   household-bootstrap reader refits the seed-1 benefit against H_LJ was negative in 3 of 4 refits. Against NEURAL_RAW the seed-0
   clause passes (bound +0.0004) and seed 1 holds only at the point (bound −0.0015). The income reserve stayed except on the
   seed-1 accuracy clause. The benefits are about half the retired candidate's (+0.022 to +0.028), which bought them with the
   income reserve.
3. **Which effects came from what?**
   - *Task learning:* the STRONG learner adds utility (−0.0014 / −0.0011 income Brier at the matched step on MODEL_SELECT) and a
     little recovery reduction there (0.0016 / 0.0032); on DEV the component claim is not established (+0.0020 / −0.0026). Its
     headroom over T_TREE was visible on MODEL_SELECT but not on CALIBRATION (T_TREE was not in the DEV bank); on DEV the twin
     (the STRONG learner alone) has income Brier −0.0055 / −0.0044 vs Ucal, against H_LJ's LOCAL income −0.0015 / −0.0011.
   - *The recovery-targeting step:* the successor differs from its matched task-only twin only by the transplanted stored
     critic-trained LOCAL / JOINT_TREE income corrections at a budgeted strength. Claim M, step 1: the declared readers recover SEX
     from the successor's linked pair less well than from its matched task-only twin's (+0.0285 / +0.0316; clause bounds +0.0220 /
     +0.0209); step 2 (claim P) fails, so the label is `TASK_ONLY_ATTRIBUTION_ONLY`. Mandatory qualifiers: the successor spends
     +0.0049 / +0.0042 nats of income NLL and +0.0038 / +0.0032 of income Brier more than the twin on DEV; the twin (label
     UTILITY_HEADROOM_ONLY on MODEL_SELECT) is point-eligible with RESERVE_MET (minimum ratio 3.30) but reads higher than both
     references on the pair (DEV benefits of the twin −0.0143 / −0.0150 against H_LJ and −0.0157 / −0.0192 against NEURAL_RAW),
     so the contrast is a mechanism contrast, not a competitiveness result. Family P showed the same pattern on MODEL_SELECT: a
     pair-trained correction direction plus an explicit budget; training against the frozen occupation output added only about
     0.001.
   - *Component control (claim C):* against HC_TTREE_LJDIR_PB the pair difference is +0.0020 / −0.0026 (not established); the
     successor spends −0.00235 / −0.00193 nats of income NLL and −0.00164 / −0.00144 of income Brier relative to it (c1
     intervals exclude 0 on both seeds), i.e. the stronger learner buys income utility; its effect on pair recovery is not resolved
     (clause intervals −0.0026 to +0.0066 and −0.0134 to +0.0082). An unresolved clause is not equivalence.
   - *Ordinary compression / calibration:* no matched temperature control exists for the nominee (its income output is continuous
     and Platt-mapped). In family P, temperature shrinkage of the twin at P2's and P3's label-free cost read 0.6362 / 0.6363 on
     seed 0 (one-seed controls), far above P2 / P3, so ordinary shrinkage did not explain family P's development-proxy reduction.
     Quantization (B2TWIN) is beaten on development proxies but not resolved on DEV (+0.0042 / +0.0015;
     `NO_DESCRIPTIVE_GAIN_OVER_B2TWIN`).
   - *Proposed components:* no family's specific step is established as the source of the gain on DEV; no novelty is claimed.
4. **Do the inherited occupation constraints still bind?** Yes (claim O): the men's guard (0.86 / 0.55 SE) and occupation NLL
   (0.40 SE on seed 0) remain, unchanged by construction. Family O's recalibration raises the men's guard margin on MODEL_SELECT
   from 0.32 / 0.62 to 1.83 / 1.62 SE (short of the family's 2-SE target) and coexists with the income-side gain
   (CMB_P3JR_O1SEXBAL), but it was not the nominee and has no DEV evidence.
5. **Robust candidate or useful piece?** A useful piece: income Brier and NLL headroom kept at DEV scale beside pair recovery
   below both references at the point (against the declared readers); not a robust development candidate (the registered
   conjunction fails, one clause is reader-sensitive, the registered reserve erodes on one clause, and rough reader-refit SDs of
   0.007-0.009 are of the order of the benefits of 0.012-0.017).
6. **What supports the next decision?** `CONFIRMATION_FEASIBILITY.md`: at the plug-in, claim P's conjunction needs about 129,000
   (D1) to 292,000 (D2, with confirmatory utility bounds) households, against about 17,510 available, and the requirement is not
   identified by these data. Three of the four binding utility clauses belong to the inherited occupation output and the fourth is
   seed-1 income accuracy; the governing ENHANCED D1 requirement is set first by the reader-sensitive seed-1 income recipient
   clause against H_LJ (0.60 SE from its 0.010 limit), then by the occupation NLL clause.

## Recommendation

**Retain H3_STRONG_LJDIR_PB as the campaign's income-side result and runnable package, described as "income Brier and NLL
headroom kept on DEV_REPORT (R2: RESERVE_ERODED on the seed-1 accuracy clause); not competitive under the registered
conjunction". Do not request a confirmation population for it.** Any next step needs a new authorization and a frozen protocol,
and should start from the measured bottlenecks, not from a new population:
- the occupation side (Family O's recalibration, or an occupation output with real margin) must come with the income-side output,
  because the inherited occupation NLL clause sets the D2 requirement and the ORIGINAL D1 requirement and is second only to the
  reader-sensitive income recipient clause in the governing ENHANCED D1 requirement;
- the seed-1 income accuracy clause and the reader-sensitive income recipient clause against H_LJ are the remaining income-side
  risks;
- the benefits are reader-relative: under the four matched household-bootstrap refits of each ENHANCED pair winner
  (`per_seed.*.refit_roster.aux_sample`; descriptive, not covered by the conditional intervals) the benefit against H_LJ was
  +0.0075 to +0.0216 on seed 0 but −0.0082 to +0.0073 on seed 1 (3 of 4 negative), and against NEURAL_RAW −0.0010 to +0.0152 on
  seed 0 (2 of 4 negative) and +0.0129 to +0.0179 on seed 1; every release reads lower under these refits (the successor's seed-0
  pair AUC 0.627 → 0.594-0.604), so this term mixes a resampling shift with variance.

## Verification and custody

Independent verification with separate code (`INDEPENDENT_VERIFICATION.json`, `roles/verifier/`): checks 1-8 PASS, with every
number reproduced (124 admitted copies re-hashed; 42 MODEL_SELECT evaluations and 840 development-grid cells bitwise; 21
nomination labels; 96 final reader refits and 96 winners bitwise; the separately logged DEV_REPORT replay at 03:17:54Z reproduced
all 130 endpoints and every claim label in both scorecards). Check 9 PASS on the numbers; check 10 PARTIAL on wording. Its
documentation items and the engineering and claims review's must-fix items (`roles/claims_reviewer/CLAIMS_REVIEW.md`, disposition
in `CLAIMS_REVIEW_DISPOSITION.md`) were corrected before publication; `PROTOCOL_DEVIATIONS.md` items 6-8 list the engineering
items. Deployment from the six raw fields: `DEPLOYMENT_CHECKS.json` (6 packages bitwise through the API, the CSV wrapper and the
underlying function; recorded refusals for SEX, extra, reordered and missing fields and a hidden-output request). Backup and
custody: `BACKUP_VERIFICATION.json` (written at closeout).

## Not claimed

No arbitrary-reader protection, privacy guarantee, information removal or population-optimal lookup. No novelty, no joint-design
advantage, no fairness claim beyond the registered guard. Deterministic outputs are not assumed injective; observed mostly unique
vectors do not show information equality. An unresolved clause is not equivalence, and a point pass is not population
preservation.
