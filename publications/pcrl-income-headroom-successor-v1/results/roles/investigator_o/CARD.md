# Family O card: occupation-side margin repair (investigator O)

Exploratory development on previously used records (PCRL income-headroom successor campaign). Not confirmation. Public-use
U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida. The service predicts an income class (2 classes) and an
occupation group (5 classes) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding `X`). Each
recipient receives only its own float64 probability vector and argmax decision. SEX is the dataset's recorded demographic column
(sex0 = SEX code 1, men; sex1 = SEX code 2, women). Readers are separate evaluation classifiers that estimate SEX from the released
probability vectors.

Written 2026-10-18 ~02:25Z, before any fit of this family and before the registration commit. Family O is the coordinator's
recorded expansion (brief section 4; PROTOCOL.md section 8: an occupation-side family keeps a fixed-occupation control).

## Measured bottleneck

The inherited JOINT_TREE E20 occupation output (the occupation output of H_LJ) binds on its own; no income-side mechanism can
change it (`INHERITED_OCCUPATION_MARGINS.json`, `REFERENCE_MS_TABLE.csv`, MODEL_SELECT, seed 0 / seed 1):

- men's within-SEX occupation summed-Brier excess vs U0: +0.004417 (SE 0.001814; margin 0.32 SE) / +0.003941 (SE 0.001697;
  0.62 SE) against the 0.005 guard; below 2 SE at every projected household scale;
- women's: -0.004938 (SE 0.001698) / -0.003039 (SE 0.001710): about 0.008-0.010 of slack (4.7-5.9 SE);
- occupation NLL excess vs Ucal +0.003406 (2.30 SE of margin) / +0.002947 (2.58 SE); on DEV_REPORT, seed 0, it was +0.008665,
  0.40 SE from the 0.010 cap;
- the same pattern in the task-only teacher: T_TREE men -0.00103 / +0.00059, women -0.00931 / -0.00788. The JOINT_TREE privacy
  pressure costs both groups about 0.004-0.005 of Brier, but men start with no slack against U0.

Switching to the LOCAL_TREE occupation output (H_LL: men 0.00178 / 0.00288) raises the occupation-view proxy to 0.5874 / 0.5931
(H_LJ 0.5694 / 0.5800), which breaks the occupation recipient bar (H_LJ + 0.010).

## Exact operation that changes the release

Income is H_LJ's own income output (LOCAL_TREE E20, `ihs.bases`), unchanged, so the only difference from H_LJ is the occupation
output. Occupation:

    z  = centred raw score of the JOINT_TREE E20 occupation output (= centred log-ratio of its released probability vector),
         computed from X through the frozen T_TREE routing plus the stored stage-B correction theta;
    p' = softmax(z / T + b),   T > 0 (one temperature), b a centred per-class bias (4 free values).

For T > 0 the map p -> p' is a bijection of the open simplex: every distinct occupation vector stays distinct, p is recoverable
exactly from p' given (T, b), and the information available to any reader of the occupation output or of the linked pair is
unchanged. Only the calibration (and therefore the utility and the within-SEX Brier) and possibly the finite-sample behaviour of
the fitted readers change. A revision may replace z by a blend z_w = (1 - w) z_JOINT + w z_TASKONLY (LOCAL_TREE or TASK_REFIT
E20 corrections on the same routing), which is NOT information-preserving; it would be recorded as such.

Initial configurations (seed 0 first):

| Config | T fitted on | b fitted on | SEX used | Role |
|---|---|---|---|---|
| OC_TCAL | CALIBRATION, overall occupation NLL | none (b = 0) | no | matched control named by the brief: ordinary global temperature |
| OC_TBCAL | CALIBRATION, overall occupation NLL (jointly with b) | CALIBRATION, overall NLL | no | control: ordinary temperature + per-class bias calibration |
| O1_SEXBAL | as OC_TCAL (CALIBRATION) | MODEL_FIT, SEX-balanced minimax (below) | training-time label on MODEL_FIT only | proposed component |

O1_SEXBAL bias: with T fixed at the OC_TCAL value, b(lambda) = argmin_b [lambda * mean_men Brier(p') + (1 - lambda) *
mean_women Brier(p')] on MODEL_FIT, and lambda in [0, 1] chosen by bisection so that the two groups' occupation summed-Brier
excess against U0 (the stored U0 MODEL_FIT output, `ihs.bases.stored_reference`) is equal on MODEL_FIT (minimax: minimize the
larger of the two excesses; lambda* = 1 or 0 if equality cannot be reached). This equalizes the two guard margins (both guards
have the same cap and similar SEs) instead of choosing a SEX weight by hand (source family A3 used a fixed 2 : 1 weight). SEX is a
training-time label only; the runtime release uses X only.

Why T comes from CALIBRATION and b from MODEL_FIT: CALIBRATION people are not in the teacher routing's or the corrections' training
set, so an NLL temperature fitted there is not biased by in-sample sharpness (the replication's Ucal used the same role); but
CALIBRATION has no SEX. MODEL_FIT has SEX but is in-sample for the teacher routing, the JOINT_TREE corrections and U0, so the
MODEL_FIT excesses are in-sample values and are never described as held-out guard estimates. The minimax criterion depends on the
DIFFERENCE between the groups' excesses, so a common in-sample offset does not move it; a group-specific in-sample offset would.
The MODEL_FIT values of H_LJ's own guards are recorded beside the MODEL_SELECT values to show how far they differ.

## Why it could improve the observed trade-off

1. If the privacy pressure pulled each group's occupation predictions toward the pooled distribution, men's and women's mean
   residuals point in roughly opposite directions; a global per-class bias then moves Brier from men to women at a first-order
   rate of about pi_men / pi_women (~1.1) while the overall loss changes only at second order. Women have about 0.008-0.010 of
   guard slack to give.
2. A held-out temperature can lower occupation NLL (the other condition that came close to its cap on DEV_REPORT) for everyone.
3. Because the map is a bijection, occupation-view and pair recovery should move little (the source's matched temperature control
   moved every reader proxy by at most +-0.0025, `MECHANISM_CARDS.md` C1).

Limits stated now: the bias can only transfer Brier between groups along the direction of the groups' mean residuals; if the
privacy cost is mostly lost resolution (sharpness), no recalibration can recover it. Readers are finite-sample learners, so an
information-preserving map can still move a fitted reader's AUC slightly in either direction.

## Simple control that could explain away the benefit

- H_LJ itself (the unmodified JOINT_TREE occupation output; already evaluated).
- OC_TCAL: ordinary global temperature only, no SEX (named by the coordinator's brief).
- OC_TBCAL: ordinary temperature + per-class bias fitted for overall NLL on CALIBRATION, no SEX. If OC_TBCAL moves the men's guard
  as much as O1_SEXBAL, the SEX-balanced fitting is not what helps; the gain is ordinary calibration.

## Observation that triggers revision or abandonment (fixed now)

Measurement: occupation output paired with H_LJ's own income output; report the men's and women's guards (metric and SE units),
the 6 occupation clauses with SEs, the occupation-view and pair proxies (mean and cell range), against H_LJ and the controls.

- Target (success for this family): on both seeds, men's guard margin >= 2 SE (value <= 0.005 - 2 SE), women's guard and all 6
  occupation clauses passing with >= 2 SE of margin, occupation-view proxy no higher than H_LJ's + 0.010 in every cell
  (preferably not higher at all: mean increase <= 0.003), pair proxy not higher than H_LJ's + 0.003 (mean).
- Revise if, on seed 0, O1_SEXBAL improves the men's guard by less than 1 SE (about 0.0018) over H_LJ, or if it achieves that only
  by bringing the women's guard or an occupation clause within 1 SE of its cap, or if the occupation-view proxy mean rises by more
  than 0.005 over H_LJ. Candidate revisions, in order: (a) a richer information-preserving calibrator (per-class temperatures or a
  full affine map of the centred log-ratio with an invertible matrix) under the same minimax criterion; (b) a controlled blend of
  the JOINT_TREE corrections toward the LOCAL_TREE corrections (weight w fixed before scoring, small enough that the
  occupation-view proxy is expected to stay within the bar), with the same recalibration.
- Abandon the family if, after at most two revisions, no configuration reaches a men's guard margin of 1 SE on seed 0 while
  keeping the occupation-view proxy within H_LJ + 0.010 in every cell and every other occupation condition passing; report a
  negative.
- If a matched control (OC_TCAL or OC_TBCAL) explains the result, the control is reported as the finding and no SEX-specific
  component is claimed.

## Closest known methods

- Temperature scaling and vector/matrix scaling of logits (Guo et al. 2017); Platt scaling.
- Minimax group-balanced fitting (group DRO, Sagawa et al. 2020; minimax group fairness, Diana et al. 2021; minimax Pareto
  fairness, Martinez et al. 2020), applied here to a SEX-free post-hoc calibrator with SEX used only as a training-time label.
- In this lineage: family A3 (SEX-weighted occupation task loss in the full correction fit moved calibration strongly between the
  groups: men's guard -0.0082, women's +0.0020), family C's matched temperature control, and the replication's Ucal (U0 with a
  CALIBRATION temperature).

No novelty or theorem is claimed.

## Allotment and records

At most 8 seed-specific fitted configurations (controls included), initial version + at most 2 revisions, about 0.8 CPU-hours
including scoring. Unit ids IHS_O_<config>_S<s>_V<n> (fit) and IHS_O_<config>_S<s>_SCORE_V<n> (development evaluation through
`ihs.devrun score`). Invented-data check of the code: `roles/investigator_o/INVENTED_DATA_CHECK.json` (unit IHS_O_SELFTEST_V1).

## Revisions (appended with time, reason and result)

- **02:24Z, invented-data check** (unit IHS_O_SELFTEST_V1, `INVENTED_DATA_CHECK.json`): 8/8 checks pass (bijection and exact
  recovery, probability contract under extreme scores, analytic gradients vs central differences, loss = evaluation metrics,
  temperature and temperature + bias recovery, minimax equalization with a brute-force perturbation check, JSON round trip).
- **02:35-02:39Z, initial version on seed 0 as planned** (registration commit 74376912c at 02:35Z). OC_TCAL: T = 0.97747; men's
  guard 0.005126 (fails), women's -0.005629. O1_SEXBAL: lambda* = 0.53256, men's guard 0.004417 -> 0.001659 (SE 0.001826, 1.83
  SE), women's -0.002185, every occupation clause equal or better than H_LJ, occupation proxy 0.5682 (H_LJ 0.5694), reserve
  RESERVE_MET. OC_TBCAL: men's 0.005729 (fails). No revision trigger met (improvement 1.52 SE >= 1 SE; women's guard and every
  occupation clause >= 2 SE; occupation proxy not higher), so O1_SEXBAL went to seed 1.
- **02:41Z, control added (not a revision; disclosed adaptive step).** The O1 fits gave lambda* = 0.53256 (seed 0) and 0.53375
  (seed 1), close to each other, which raised the possibility that the "SEX-balanced" bias is just the ordinary in-sample bias.
  Added OC_BFIT before scoring it: identical to O1_SEXBAL except that the bias minimizes the unweighted Brier on MODEL_FIT (no SEX).
  The MODEL_FIT men's share is 0.5103, so lambda* is a men's relative weight of 1.093 (seed 0) / 1.099 (seed 1), not the
  unweighted value. Result: OC_BFIT men's guard 0.003264 / 0.003950; the SEX weighting accounts for -0.00160 / -0.00165 of the men's
  guard (and +0.00181 / +0.00185 of the women's) on both seeds.
- **02:40-02:49Z, seed 1:** O1_SEXBAL men's 0.003941 -> 0.002299 (SE 0.001672, 1.62 SE), women's -0.002759, NLL vs Ucal 0.002947
  -> 0.001642, RESERVE_MET; OC_TCAL men's 0.005085 (fails); OC_BFIT men's 0.003950.
- **02:51Z, decision (ledger entry 56): stop the family without a revision.** Reason: no revision trigger was met, and the
  remaining shortfall to the 2-SE target (0.00031 / 0.00069 of men's margin) comes from an in-sample offset. On both seeds the
  MODEL_FIT men-minus-women excess gap understates the MODEL_SELECT gap by about 0.0037; equalizing in-sample therefore
  under-corrects. Correcting that offset would need out-of-sample SEX labels, which here means fitting on MODEL_SELECT (not
  allowed). The other listed revisions were not pursued: a richer calibrator would face the same offset, and a blend toward
  LOCAL_TREE would raise occupation-view recovery. 7 of 8 configurations used.
- **~03:20Z, platform block (rule 9 of the brief):** writing `roles/investigator_o/REPORT.md` was refused by the harness with the
  message "Subagents should return findings as text, not write report files. Include this content in your final response
  instead." The action was stopped. The full report text was handed to the coordinator in investigator O's final message. The
  full-precision aggregates are in `RESULTS_SUMMARY.json`.
