# Research decision: correction transfer, genuine refits and a stable method comparison (PCRL, 2026-10-18)

**Scope.**
- This is exploratory development on previously used public-use ACS 2019 Florida PUMS records. It used the MODEL_FIT, CALIBRATION
  and MODEL_SELECT development roles; every earlier campaign used MODEL_SELECT adaptively.
- Income and occupation are the useful tasks. The evaluation classifiers ("development readers") estimate the recorded SEX column
  from released probability vectors. "Recovery" always means recovery by these declared readers, never protection against every
  reader.
- The ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels stayed closed for the whole campaign.
- Nothing here is confirmation. The occupation-headroom campaign's NO_QUALIFYING_CONFIGURATION stands unchanged.
- Every number below comes from the records listed in START_HERE.md, the coordinator's findings notes (F1-F12) or the private run
  summaries they cite. "x 1e-3" means the numbers are in thousandths.
- The independent claims review (`roles/claims_review/REPORT.md`) was applied to this text.

## Decision in brief

- **Classification: a replicated, independently replayed diagnosis plus a negative method result (development evidence).** No
  competitive candidate, no useful protection component, nothing to confirm.

- **Question A, cost estimation on excluded households: no.**
  - Neither the source's out-of-fold estimate nor an excluded-household GUARD estimate predicts the subgroup costs a correction
    adds well enough to choose a useful strength.
  - The out-of-fold estimate over-predicts NLL, most likely because of non-nested cross-fitting (checked on one partition). Its
    men's-cost error changes sign between training partitions.
  - The GUARD estimate is unbiased within its noise, but that noise is as large as the slack it has to protect.
  - With REFIT routing, neither the coalition-aware direction (PAIR) nor the occupation-only direction (LOC) predicts better under
    the registered bootstrap on any partition. No direction predicts better on all four partitions under either scheme.
  - Two R1 labels (REFIT_LOC full, REFIT_PAIR_TR full) are SCHEME_DEPENDENT. An improvement appeared only under the superseded
    frozen implementation; it is not established under the registered scheme, which adjudication A5 makes primary. The
    correction's trigger used D-stage inputs only (verifier P4 V2 PASS), and the correction was committed after the repair panels
    had been read with the frozen code (deviations 16 and 18).

- **Question B, method stability: no method result.**
  - Honest refitting left the occupation witness with almost no reserve slack on the binding men's guard.
  - Recipe selection gave NO_RECIPE before any repair panel was opened.
  - In the full deployment every frozen rule chose identity or at most 0.13. In the ensemble deployment SIMPLE chose 0.2-0.6, and
    every such selection crossed the reserve.
  - The coalition-aware correction lowers the declared development readers' recovery repeatably beyond the zero-critic
    continuation (s >= 0.3) and beyond the unchanged witness (s = 1.0 only). It does so only at strengths whose men's cost is many
    times the available slack.

- **Question C, comparison design: it matters a great deal for interpretation.**
  - Under simulation law L1 with two household repetitions, the old per-cell recipient gate passes only 19-31% of true-zero
    differences on the income view. That is a property of that law, not a universal error rate.
  - Every H3-anchored release fails the gate through H3's income view alone.
  - Every required comparator fails the two-SE reserve that the candidates must meet.
  - Under one equal reserve, no release that lowers the readers' recovery qualifies.

- **A new, disclosed adaptive finding.** The source witness's margin is consistent with a training-set-size effect. Occupation
  NLL improves monotonically from 42% to 53% to 70% of MODEL_FIT, toward the source fit at 100%. The men's guard improves from 53%
  to 70% but not from 42% to 53%. A fit-specific selection component at 100% is not excluded.

## 1. Why the old cost estimate missed

**Short answer.** It over-predicted NLL, most likely because of non-nested cross-fitting. Its men's-cost error changes sign and
size from one training partition to the next by more than the reserve it was spending.

The source chose its correction strength from **out-of-fold (OOF) costs** inside the training set. We refitted the same pipeline
on four different, overlapping training partitions of MODEL_FIT (D0, D1, R0, R1) and measured realized costs on excluded
households: the CHECK panels and MODEL_SELECT. Three separate errors appear. Each figure is the mean over the 10 nonzero ladder
strengths, realized minus predicted, for the REFIT-routed directions, x 1e-3 (`diagnosis/TRANSFER_FACTORS.json`).

1. **A systematic over-prediction; nesting reduced it on one partition (D0).**
   - The OOF estimate for PAIR over-predicts the NLL increase on MODEL_SELECT on all four partitions: -1.9 (D0, z -1.1, not
     resolved alone), -4.2 (D1), -3.2 (R0), -4.5 (R1).
   - For the zero-critic continuation CONT it over-predicts its own small costs much more strongly (MODEL_SELECT z from -7.1 to
     -10.3).
   - On D0, the only partition where it was run, nesting every upstream layer inside each outer fold lowered PAIR's predicted NLL
     increase by 4.8 (z -3.1).
     - Against the CHECK panel this removes the over-prediction (CHECK minus nested +0.97, z +0.5).
     - Against MODEL_SELECT it over-corrects (+2.86, z +2.0).
     - For LOC it under-predicts on both panels (+2.96 and +2.43).
     - Nesting was not run for CONT.
   - Non-nested cross-fitting is therefore the likely main source of the bias. One partition cannot establish it.
2. **A partition-specific men's-cost error.**
   - For PAIR, CHECK minus OOF on the men's summed Brier is +0.03 (D0), -5.14 (D1, z -4.0), +2.09 (R0, z +1.5) and -2.76 (R1,
     z -2.2). Its sign and size change with the training partition.
   - On D0, where the OOF men's error was +0.03, the nested estimate errs by +2.70 (z +2.0). Nesting does not remove this kind of
     error and can introduce it.
   - The source showed the same signature on MODEL_SELECT minus OOF at B80: +2.0 / +3.0 on seed 0 and -1.2 / -0.9 on seed 1
     (`preflight/SOURCE_REPLAY.json`).
3. **Sampling noise of a SEX-group increment that is as large as the slack.**
   - The household-clustered SE of a men's or women's increment at s = 1 is 1.5-2.2e-3 on a GUARD or CHECK panel and 1.0-1.3e-3
     on MODEL_SELECT.
   - The men's reserve slack of the refitted witnesses is +0.49, -0.78, +0.14 and -0.31 (x 1e-3).
   - The realized men's cost at s = 1 on MODEL_SELECT is 3.9-10.5e-3.

## 2. Did honest refitting change that conclusion?

**Short answer.** It confirmed and sharpened it. Refitting did not remove the transfer error, the GUARD estimate did not predict
better, and refitting on part of the data removed most of the slack the source had spent.

- **The error is not an artifact of frozen routing.** Routing the correction through the frozen admitted teacher (FROZEN) or
  through a teacher refitted on FIT alone (REFIT) gives the same pattern on D0 and D1 (D1 men's -3.97 vs -5.14). FROZEN routing
  was run on the diagnostic partitions only.
- **A GUARD estimate is unbiased within noise, but it does not predict better.**
  - On excluded GUARD households, every CHECK minus GUARD and MS minus GUARD error lies within |z| <= 2.9. Nine of 180 exceed 2,
    about the number expected by chance, and their signs vary between partitions.
  - The registered question-A comparison (RULES.md section 6; one household resample per set, B = 2000;
    `diagnosis/QUESTION_A_LABELS.csv`) finds GUARD better than OOF in two places:
    - for the control CONT on D1 and R0;
    - for the FROZEN-routed PAIR on D1 (ensemble). Its GUARD and CHECK households were seen by the frozen routing
      (A1_NOT_SUPPORTED).
  - For the REFIT-routed PAIR and LOC, no difference is resolved. The point estimate favours GUARD on D1 and R1 and OOF on D0 and
    R0 (PAIR full: D = +0.16, +0.38, -0.51, -2.21); q05 is negative in every PAIR and LOC row.
  - Three labels depend on the bootstrap scheme and are flagged SCHEME_DEPENDENT: D0 CONT ensemble, R1 LOC full and R1 Track R
    PAIR full. Under the registered scheme they are NOT_IMPROVED; under the superseded frozen implementation they read
    PREDICTION_IMPROVED.
    - Adjudications A2 (D) and A5 (R) make the registered scheme primary.
    - The shift comes from the superseded scheme's 20-25% smaller SE, not from heavy tails. The top 10 households carry about 5% of
      the variance, and the label stays NOT_IMPROVED after dropping 1, 5 or 10 of them (verifier C3 V3).
    - Disclosed window (verifier P4b): its R1 replays wrote these two labels to the verifier's CHECKS.json 29 s before the
      correction commit, and no coordinator read is logged in that window (deviations 16 and 18).
  - No direction predicts better on all four partitions under either scheme, so the claim ladder is unaffected.
- **Honest refitting costs most of the margin the source spent.** This is the larger change.
  - The source witness was fitted on all 20,000 MODEL_FIT people. Its men's-guard excess on MODEL_SELECT was -3.9 / -2.3
    (x 1e-3).
  - Refits of the same recipe on 42-53% of MODEL_FIT give +0.65 (D0), +2.15 (D1), +1.11 (R0) and +1.89 (R1). They pass every
    point limit but leave reserve slack of only +0.49 / -0.78 / +0.14 / -0.31.
  - An adaptive diagnostic refitted the recipe on 70% of MODEL_FIT (amendment 9; findings F7; the P70 evaluation, a private run
    summary). It restores a positive men's reserve slack (+3.2 / +1.9) and occupation NLL excess of -11.3 / -7.9 (vs U0), closing
    about half of the gap to the source witness.
  - This is at or slightly beyond the registered log-linear size prediction. By the registered reading it is consistent with a
    size effect being sufficient; a fit-specific selection component at 100% is not excluded.
  - The FIT-only dictionary collapse explains little (P70A minus P70: NLL -0.7, SE 0.4; `diagnosis/TRAINING_SIZE_CURVE.json`).

## 3. Was usefulness kept at a nonzero correction strength?

**Short answer.**
- At the point, yes, up to s = 0.5.
- With the two-SE reserve, at no reader-evaluated nonzero strength.
- The only non-crossing nonzero rule choice was 0.017 on R0, which was not reader-evaluated. On R1 the witness itself fails the
  reserve.

- **Rules on the reserved repair panels** (`repair/RULE_OUTCOMES_REPAIR.csv`; rule-selected strengths were not reader-evaluated,
  deviation 12):
  - **R1:** every rule chose identity for every direction. The unchanged witness itself already fails the two-SE reserve on the
    men's guard (ratio 1.82).
  - **R0, the coalition-aware direction (full deployment):**
    - OLD-B50 chose 0.017. It did not cross on the C0 panel (slack use 0.94) or on MODEL_SELECT (0.60).
    - OLD-B80 chose 0.027 and crossed on C0 (1.50).
    - SIMPLE and UNCERTAINTY chose identity.
  - **Ensemble deployment.** SIMPLE chose 0.5 and crossed by more than tenfold (13.8 on C0, 10.5 on MODEL_SELECT). Ensemble slack
    use is measured as the increment over the uncorrected ensemble e_0. That ensemble already uses 4.0 times the witness's men's
    slack on C0 (deviation 6; verifier ATTENTION), so every ensemble release crosses the witness's reserve.
  - The GUARD-based rules never improved on the source rule on both R partitions (`compare_rules`). This was close to forced: on
    R1 every rule chose identity, and compare_rules never counts identity as an improvement, even where it avoided OLD-B80's C0
    crossing (1.50).
- **Complete releases at common strengths** (`repair/RESERVE_QUALIFIED_FRONTIER.csv`):
  - REFIT_PAIR passes the 24 clauses and 4 guards at the point for s <= 0.5 and fails them at s = 1.0.
  - Its minimum reserve ratio falls from 1.78 / 1.73 at s = 0.1 to 0.06 / 0.86 at s = 0.5. The threshold is 2.
  - Realized MODEL_SELECT utilization on R0 is u = 3.8 at s = 0.1 and 26.6 at s = 0.5. On R1 it is infinite at every strength,
    because the slack there is negative.
- **What binds.** The men's occupation guard binds at every reader-evaluated point on both repair partitions
  (`repair/FRONTIER_DRIVERS.csv`).

## 4. Recovery benefit beyond the simplest matched controls

**Short answer.** For the declared development readers: yes against the zero-critic continuation (s >= 0.3) and against the
witness (s = 1.0 only), at large strengths only. Not against the occupation-only critic.

Pair benefit is the comparator's SCORE AUC minus the candidate's; positive means less recovery. Stability uses 8 cells: two
partitions x two household repetitions x two reader seeds (`repair/MATCHED_CONTROLS.csv`; D stage in
`tables/READER_STABILITY_D_FRONTIER*`).

| Coalition-aware correction vs | R stage, s = 1.0 | R stage, s = 0.5 | R stage, s = 0.3 | D stage, s = 1.0 |
|---|---|---|---|---|
| unchanged witness W | +0.041 STABLE_POSITIVE | +0.028 positive, not stable | +0.009 positive, not stable | +0.041 STABLE_POSITIVE |
| zero-critic continuation CONT (same strength) | +0.042 STABLE_POSITIVE | +0.029 STABLE_POSITIVE | +0.011 STABLE_POSITIVE | +0.034 STABLE_POSITIVE |
| occupation-only critic LOC (same strength) | +0.010 sign changes | +0.013 positive, not stable | +0.001 sign changes | +0.013 positive, not stable |
| label-free compression twin (R0 only, 4 cells; not cost-matched) | +0.017 (not labelled) | +0.018 | +0.004 | n/a |

- **The critic, not the extra training, lowers the readers' recovery.** On the repair stage CONT alone reads like the witness:
  -0.0006 to -0.0028 vs W, never positive. On the diagnostic stage it is +0.0063 at s = 1.0, with sign changes.
- **LOC carries most of the effect.** The occupation-only critic LOC alone gives +0.032 vs W at s = 1.0 (positive, not stable).
  The coalition-aware part adds about +0.010 on top of it, and that addition is not stable.
- **The compression comparison is descriptive.**
  - The compression twins exist only on R0, because R1's negative slack leaves the matched target undefined (PROTOCOL_DEVIATIONS
    item 13). The 8-cell rule therefore cannot label them.
  - The twins are not cost-matched either. The source's K search assumes a monotone utilization path, and on R0 the selected codes
    reach only 6-55% of the target (item 17).
- **The cost of the stable benefit.** Where the benefit over CONT is stable (s >= 0.3), the realized men's cost is 14-70 times
  the R0 slack (70 times at s = 1.0). On R1 the slack is negative, so the cost is unbounded relative to it.

## 5. Did it survive across pipelines?

**Short answer.** The lower recovery for these development readers survives. Utility does not.

- **Across training partitions:** the same ordering appears on the four overlapping partitions D0, D1, R0 and R1 (sections 4 and
  6).
- **Routing (FROZEN vs REFIT) at equal strength:** no consistent difference (D stage: -0.0023 at s = 0.5, +0.0016 at s = 1.0).
- **Track R, the whole-service refit** (R0 and R1 only; the income lineage refitted without GUARD or CHECK households):
  - Each partition's Track R occupation teacher is bitwise identical to its Track F routing teacher.
  - The correction trained against the refitted income wire lowers the readers' recovery against its own witness by +0.047 at
    s = 1.0 and +0.023 at s = 0.5 (both STABLE_POSITIVE).
  - Against Track F at equal strength the difference is +0.006 at s = 1.0 (sign changes) and -0.005 at s = 0.5 (not positive;
    `tables/READER_STABILITY_R_TRACKR*`).
  - The refitted income does not keep H3's headroom (section 7). The complete Track R releases fail the reserve: R0 on income Ucal
    Brier (ratio 1.92), R1 on the men's guard (1.82).

## 6. Frontier versus benchmark

**Short answer.** Neither comparison favours the correction. Nothing is competitive on the original benchmark, and no
reserve-qualified point lowers the readers' recovery.

The two comparisons are reported together and never merged (RULES.md section 11; `repair/ORIGINAL_BENCHMARK.json`,
`repair/RESERVE_QUALIFIED_FRONTIER.json`, `figures/FRONTIER_R.*`).

- **Original benchmark** (references exempt from the reserve, as in the source):
  - REFIT_PAIR at s <= 0.5 passes the 24 clauses and 4 guards at the point.
  - It fails the old recipient gate at every strength. The income view fails through H3 (against H_LJ and TWJ) at every strength.
    At s <= 0.5 the occupation view also fails: against all six comparators at s = 0.1 and 0.2, and against five (all except
    NEURAL_RAW) at s = 0.3 and 0.5.
  - It is not STABLE_POSITIVE against all six required comparators at any strength:
    - at s = 1.0: SUCC +0.008, NEURAL_RAW +0.024 and H_LJ +0.026 (all positive, not stable); B2TWIN +0.014 and RETIRED_B3CTL
      +0.001 (sign changes); TWJ +0.038 (stable).
    - at s = 0.5: SUCC -0.005 and RETIRED_B3CTL -0.012.
  - The separate 0.020 target against NEURAL_RAW and H_LJ is reached only as a point mean at s = 1.0
    (DEV_GRID_TARGET_POINT_ONLY), and that release fails utility at the point.
- **Reserve-qualified frontier** (one reserve rule for every point: 28 conditions, k = 2, H_LJ floor):
  - No candidate qualifies.
  - The refitted witness fails (1.82 on R1), and every required reference fails (source records 0.32-1.95).
  - The only qualified points are the zero-critic continuation at s >= 0.3. The record's `qualified_protected_points` field lists
    them because it holds every qualified non-reference point.
  - CONT raises the men's-guard ratio to 2.04-2.24, but its pair AUC (0.634-0.636) matches the witness's (0.634). None of these
    points lowers the readers' recovery relative to W, so no reserve-qualified point lowers recovery.

## 7. The income-side limitation

**Short answer.** H3's frozen income view alone fails both recipient screens. Its Brier / NLL headroom did not survive a refit of
its recipe on about half of MODEL_FIT.

- **Track F changes occupation only.** H3's frozen income view is carried unchanged, and it alone fails both recipient screens.
  - Old gate: maximum cell +0.0309 vs H_LJ and +0.0147 vs TWJ.
  - AGG1: U = 0.0227 vs H_LJ and 0.0159 vs RETIRED_B3CTL.
  - This was recorded before any outcome (`screen/PRE_OUTCOME_H3_INCOME_VIEW.json`).
- **H3's income headroom does not reproduce when its recipe is refitted** (findings F11; the Track R evaluations, a private run
  summary).
  - The refit used the R partitions' FIT sets (75% of the non-panel pool, about 53% of MODEL_FIT) plus CALIBRATION, without the
    GUARD or CHECK households.
  - Income Brier excess vs U0 rises from -2.8 / -3.3 (frozen H3) to +1.6 / +1.1 (refit), and NLL from -4.7 / -5.2 to +1.0 / +0.6
    (x 1e-3).
  - Income accuracy is slightly better (loss -1.7 / +1.4 against +0.1 / +1.7).
  - There was one refit per partition, at about half H3's training size, so training size and recipe are not separated. It is a
    recipe test on different training subsets, not a test of the frozen H3 package.
- So the income side is both a fixed liability (the recipient screen) and an asset that did not survive a refit on about half of
  MODEL_FIT (Brier / NLL headroom). Any successor that keeps H3's income inherits both.

## 8. Classification

**A replicated, independently replayed diagnosis and a negative method result (development evidence).**

- **The diagnosis.** The transfer error was replicated on four overlapping partitions and numerically replayed with independent
  code; the torch-trained correction directions were not refitted by the verifier.
  - There is a systematic over-prediction of NLL on MODEL_SELECT, and a larger one of CONT's own small costs. On D0, nesting
    removed PAIR's NLL over-prediction against CHECK but over-corrected against MODEL_SELECT. Non-nested cross-fitting is the
    likely source. Nesting was not run for CONT.
  - The men's-cost error is partition-specific: up to 5e-3, and about 0 on D0. On D0 the nested estimate itself errs by +2.70
    (z +2.0).
  - The GUARD estimate is unbiased within noise, and that noise is as large as the slack.
  - Honest refitting costs the witness most of its margin. This is consistent with a training-size effect (an adaptive
    diagnostic; a selection component at 100% is not excluded).
- **The negative method result.**
  - NO_RECIPE: all 32 recipes fail G1 on D1, where every rule selects identity.
  - The GUARD-based rules never improve on the source rule on the repair panels. This was close to forced, since identity never
    counts as an improvement.
  - No nonzero strength keeps the reserve.
  - No reserve-qualified point lowers the readers' recovery.
- **Not a competitive candidate and not a useful component.** For the declared development readers, the coalition-aware
  correction's recovery benefit beyond the witness and the continuation is repeatable, but only at utility costs far outside the
  budget. It is not stable beyond the occupation-only critic.
- **Claim ladder.** The highest claimable rung is 0 (`repair/CLAIM_LADDER.json`). JOINT_DESIGN_ADVANTAGE is not claimable and
  NOVELTY is not claimed.
- No impossibility claim follows from this finite experiment.

## The fallback programme (PROTOCOL section 11), item by item

| Item | Result | Record |
|---|---|---|
| Replicate the diagnosis across partitions | Done on D0, D1, R0 and R1 for PAIR, LOC and CONT; both routings on D0 and D1, REFIT only on R0 and R1 | `diagnosis/TRANSFER_FACTORS.json`, `COST_TRANSFER.csv` |
| Systematic or too uncertain? | NLL and CONT over-prediction: systematic on MODEL_SELECT on every partition (the R0 CHECK panel shows none for PAIR or LOC). Men's-cost error: partition-specific, sign-changing, and too uncertain to correct at these sample sizes | section 1 |
| Nested vs non-nested at the same magnitude | D0 only. Nesting removes PAIR's NLL over-prediction against CHECK but over-corrects against MODEL_SELECT (+2.9e-3, z 2.0) and for LOC. It under-predicts the men's cost (+2.7e-3, z 2.0). Not run for CONT. In question A, GUARD was closer to CHECK than the nested estimate (D +0.42 / +0.20, unresolved) | `diagnosis/TRANSFER_FACTORS.json` (NESTED_OOF), `diagnosis/QUESTION_A_LABELS.csv` |
| Does a simple rule explain the uncertainty-aware rule's benefit? | There is no benefit to explain for the correction directions. UNCERTAINTY chose identity in the full deployment on all four partitions and 0.1 in the ensemble on D0 only. SIMPLE chose 0.1 for LOC (full) on D0, larger ensemble strengths that crossed, and identity elsewhere. The control CONT received nonzero, non-crossing strengths from both rules. Neither rule improves on OLD on the repair partitions, which was close to forced | `repair/RULE_COMPARISON.json`, `diagnosis/RULE_OUTCOMES.csv` |
| What drives the frontier | The men's occupation guard binds at every reader-evaluated point. The readers' recovery falls only where that cost is far over budget | `repair/FRONTIER_DRIVERS.csv` |
| Complete the screen study | Done before any new candidate was scored | `screen/SCREEN_STUDY.md` |
| Model identities and the effect of training-household changes | Replayed with separate code: bitwise refits of the witnesses, routing teachers and Track R teachers (directions not refitted). Training-size diagnostic (adaptive) | `roles/verifier/REPORT.md`, findings F7 |
| Package the strongest honest task witness | P70 (whole non-panel pool; an adaptive fit under amendment 9, never a selected release) packaged from the six raw fields; bitwise, every refusal | `deploy/DEPLOYMENT_CHECKS_P70.json` |

## What this suggests (not tested here)

- **Nested cross-fitting on the full training set.** Costs could be estimated this way instead of holding out a GUARD subset (20%
  of the pool on D, with another 20% for CHECK; 25% on R).
  - This remedy is untested. On D0 the nested estimate was not more accurate than GUARD, and it under-predicted the men's cost and
    the MODEL_SELECT NLL.
  - The training-size loss is large, and the GUARD estimate did not predict better.
- **Sample size for a SEX-group cost.** Protecting a reserve of about 1e-3 with a two-SE margin needs far more excluded households
  than one GUARD or CHECK panel offers. At about 2e-3 SE per 1,400 people per group, roughly 8-16 times as many people would be
  needed, which is beyond the development roles.
- **Judging a protection component.** Use one reserve rule for candidates and references alike, and a recipient screen whose
  zero-difference pass rate is known (`screen/SCREEN_STUDY.md`).

## Failed attempts, corrections and disclosures (all preserved)

All 19 items are in `PROTOCOL_DEVIATIONS.md`. The adjudications A1-A6 and the disposition of the claims review are in
`roles/coordinator/ADJUDICATIONS.md`. The principal ones:

- **Before any fit.**
  - Routing recipe: the first REFIT routing used a reduced CatBoost seed and a one-task support repair. The refit investigator
    found and fixed it at 19:07.
  - AGG1 phi_prior: the formula text (net) and the published value (gross) disagreed. The gross construction was chosen for all
    views before registration.
- **Engineering patches before any correction outcome (deviation 6).** This includes a registered computation changed for the
  ensemble gates. The ensemble's identity is the uncorrected ensemble e_0, which itself crosses the witness's reserve.
- **Question-A bootstrap.** The implementation departed from RULES.md section 6. The corrected run gives the primary labels and the
  first labels are kept as superseded (A2; deviations 10 and 11).
  - The R labels were corrected after the repair panels had been read with the frozen code. The statistical reviewer ruled the
    registered scheme primary under PROTOCOL section 8, and the verifier confirmed the trigger was D-stage only (A5; deviations
    16 and 18).
  - Disclosed window (P4b): the verifier's R1 replays wrote the two SCHEME_DEPENDENT labels to its CHECKS.json 29 s before the
    correction commit, and no coordinator read is logged in that window.
- **The verifier's genuine FAILs.**
  - F3 V2: frozen rule code changed after the freeze (above).
  - C2.1: P70A's witness record states the wrong fold tags (ERRATA E1; values unaffected).
  - Verifier check bugs were corrected under A1, A3, A4 and A6. No criterion was widened.
- **Not reader-evaluated or not matched.** Rule-selected strengths were not reader-evaluated (deviation 12). The compression twins
  exist on R0 only (deviation 13) and are not cost-matched (deviation 17).
- **Time stamps.**
  - The coordinator wrote estimated times instead of actual ones (deviations 8, 11 and 19).
  - The A2 heading says "about 20:35Z"; the file was created at 20:30:14Z.
  - The A5 heading says "about 20:50Z"; the file was written at 21:04:46Z.
  - The verifier's message headed "20:29Z" was sent between 20:27:28Z and 20:28:04Z.
- **Code and queue records.**
  - Code was committed while units waited for a slot, and a few units launched with uncommitted code. The launch audit shows the
    launched code equals committed content or is not imported (deviations 3, 7 and 14).
  - The queue's record of uncommitted paths truncates a label (deviation 15).
- **Units.**
  - 1 failed queue unit: the verifier's VER_QA_INFLUENCE_R1_V1, refused by the access layer before any label read and rerun.
  - 1 verifier unit was registered but never launched (VER_STRUCT_V1).
  - No campaign fit failed.
- **Record errors in the coordinator's notes.**
  - F8 holds the superseded question-A labels; F9 corrects them.
  - F11 says "75% of MODEL_FIT" for the Track R training sets; it should be 75% of the non-panel pool, about 53% of MODEL_FIT
    (ERRATA E2).

## Not claimed

- No population guarantee, no confirmation, and no statement about final attack or DEV_REPORT outcomes; those roles stayed closed.
- No reinterpretation of the source's NO_QUALIFYING_CONFIGURATION. The AGG1 what-if leaves it unchanged and is descriptive only.
- No claim that nesting removes the bias, that training size causes the margin loss, or that the reserve-qualified continuation
  points are protective.
