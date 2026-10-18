# Coordinator findings notes (append-only, times UTC; development evidence on previously used records)

## F1. Source replay (19:13)
- The source P2_PAIR / P2A_PAIR projection decisions (B80, B50; both seeds) reproduce exactly from the stored out-of-fold arrays
  (`preflight/SOURCE_REPLAY.json`, maximum scale difference 0.0).
- Realized MODEL_SELECT minus out-of-fold incremental cost at the selected releases (the source transfer error):
  - men's summed-Brier increase: +0.0020 (P2_PAIR B80 s0), +0.0030 (P2A_PAIR B80 s0), -0.0012 / -0.0009 (seed 1);
  - women's: -0.0030 / -0.0018 (seed 0), -0.0010 / -0.0020 (seed 1): the source projection over-predicted the women's cost on both
    seeds while it under-predicted the men's cost on seed 0.
- The installed CatBoost 1.2.8 refits the admitted seed-0 occupation teacher exactly (identical teacher SHA-256).

## F2. Honest refitting costs most of the witness margin (19:15)
- A1_PO_16 recipe refitted on FIT only (D0 8,419 people; D1 8,390; 42% of MODEL_FIT), evaluated on MODEL_SELECT with H3 income:
  occupation NLL excess vs U0 -0.0058 (D0) / -0.0005 (D1), against -0.0161 / -0.0128 for the source witness fitted on all 20,000;
  men's guard +0.0007 / +0.0022 (source -0.0039 / -0.0023); women's -0.0059 / -0.0020.
- Reserve slack (margin - 2 x se_used) of the binding men's guard: +0.0005 (D0) and -0.0008 (D1). The witnesses still pass every
  point limit; D1 fails the two-SE reserve on the men's guard before any correction.
- The two D partitions differ by 0.0053 nats in occupation NLL excess: training-partition variation is of the same order as the
  margins the rules spend.
- Consequence: under the frozen rules, any useful correction strength needs a positive reserve slack; on D1 every rule must select
  identity unless a correction lowers the men's cost.

## F3. Recipient screens before any new outcome (19:14; reader investigator, label-free)
- H3's frozen income view fails the old per-cell gate (vs H_LJ, TWJ) and AGG1 (vs H_LJ, RETIRED_B3CTL). Every Track F candidate
  that keeps H3's income therefore fails both recipient screens through its income view, whatever its occupation output.
- Five of the six required comparators share one occupation wire (JOINT); the occupation-view screen has two distinct comparator
  wires.

## F4. Cost transfer on the diagnostic partitions, REFIT_PAIR (first look, 19:57; ladders x 1e-3, household-clustered SEs)
- Men's summed-Brier increase at s = 1: D0 OOF 11.85 (1.29), GUARD 8.73 (2.01), CHECK 11.53 (2.19), MODEL_SELECT 10.52 (1.23), NESTED
  7.14 (1.35); D1 OOF 11.58 (1.28), GUARD 6.12 (1.97), CHECK 1.90 (1.88), MODEL_SELECT 3.86 (1.12). The source estimator is close on D0
  and over-predicts about threefold on D1: its error is not stable across training partitions, as in the source (seed 0 vs seed 1).
- NLL increase at s = 1: OOF over-predicts on both partitions (D0 22.5 vs CHECK 15.0, GUARD 14.0, MS 18.5; D1 22.9 vs 10.3, 14.8, 13.9).
  The nested estimator on D0 (14.2) sits on the excluded-household values.
- Women's increase: OOF over-predicts on D0 (6.19 vs CHECK -0.29, MS 3.67).
- Sampling: the household-clustered SE of a SEX-group increment at s = 1 is about 2e-3 on GUARD / CHECK (about 1,400 people per
  group) and 1.1-1.2e-3 on MODEL_SELECT, against men's reserve slack of +0.5e-3 (D0) and -0.8e-3 (D1). At useful strengths the
  realized subgroup cost of a fresh sample cannot be predicted to better than its own sampling noise, which is as large as the slack.

## F5. Transfer-error factors on D0 / D1 (20:15; diagnosis/TRANSFER_FACTORS.json; mean over the 10 nonzero ladder strengths, x 1e-3,
## realized minus predicted, z = mean over strengths of error / sqrt(se_a^2 + se_b^2), treating the two estimates as independent)
- Source estimator (non-nested OOF) against excluded households, PAIR direction, REFIT routing: men's CHECK-OOF +0.03 (z +0.1) on D0 but
  -5.14 (z -4.0) on D1; NLL -3.82 (z -1.6) / -6.25 (z -3.0); women's -3.39 (z -2.6) / -0.23. FROZEN routing: the same pattern (D1 men
  -3.97, z -2.9; NLL -6.98, z -3.2). Zero-critic continuation CONT: OOF over-predicts its own small costs strongly (MS-OOF NLL z -7.1 on
  D0, z -10.3 on D1). The OOF estimator over-predicts on average, by an amount that changes with the training partition.
- Nested OOF (every upstream layer refitted per outer fold; D0): NLL CHECK-NESTED +0.97 (z +0.5) for PAIR, i.e. nesting removes most of
  the NLL over-prediction; men's CHECK-NESTED +2.70 (z +2.0), i.e. the nested estimate under-predicts the men's cost on D0.
- GUARD (excluded households): every CHECK-GUARD and MS-GUARD error lies within |z| <= 2.2 (PAIR: |z| <= 1.6), with signs that differ
  between partitions: consistent with sampling noise, not a systematic bias.
- Routing (FROZEN vs REFIT) changes little: the over-prediction appears with both. H3 anchor: LOC (no H3) shows the same OOF pattern as
  PAIR (D1 men -3.34, z -3.3), so the exposed income anchor is not needed to produce it.
- Deployment: CHECK - CHECK_ENS for PAIR men +1.65 / -2.14 (z +0.9 / -1.5): the deployed full correction and the ensemble differ by
  amounts within noise.
- Caveat (engineering review ENG-08): z-scores of same-people pairs (NESTED vs OOF; CHECK vs CHECK_ENS) treat dependent estimates as
  independent and are only indicative.

## F6. D-stage matched frontier (20:18; reader grid D_FRONTIER on MODEL_SELECT, D0 -> seed-0 cells, D1 -> seed-1 cells; pair benefit =
## comparator SCORE AUC minus candidate's, 4 cells per partition; tables/READER_STABILITY_D_FRONTIER_KEY.*)
- REFIT_PAIR at s = 1.0: vs witness +0.0405 STABLE_POSITIVE (+0.046 / +0.035); vs matched continuation REFIT_CONT at s = 1.0 +0.0342
  STABLE_POSITIVE; vs REFIT_LOC at s = 1.0 +0.0133 POSITIVE_NOT_STABLE; vs SUCC +0.0086 POSITIVE_NOT_STABLE (+0.013 / +0.0045); vs
  H_LJ +0.026, NEURAL_RAW +0.025 (both POSITIVE_NOT_STABLE); vs RETIRED_B3CTL +0.0017 SIGN_CHANGES.
- REFIT_PAIR at s = 0.5: vs witness +0.0237 STABLE_POSITIVE; vs CONT at 0.5 +0.0187 STABLE_POSITIVE; vs LOC at 0.5 +0.0059 SIGN_CHANGES.
  At s = 0.2: vs witness +0.0085 POSITIVE_NOT_STABLE.
- Continuation alone (CONT at 1.0 vs witness) +0.0063 SIGN_CHANGES: the critic, not extra training, lowers recovery.
- FROZEN vs REFIT routing at equal strength: -0.0023 (0.5) / +0.0016 (1.0), not consistent.
- The refitted witness alone reads 0.032 higher than SUCC (W vs SUCC -0.0319).
- Cost at these strengths (MODEL_SELECT, D0 / D1, x 1e-3): men's +10.5 / +3.9 at s = 1.0, +4.1 / +0.9 at 0.5, +1.35 / +0.11 at 0.2, against
  men's reserve slack +0.5 / -0.8: the strengths that lower recovery stably are far outside the refitted witnesses' occupation-side
  budget.

## F7. Training-size diagnostic (ADAPTIVE, amendment 9; 20:24; diagnosis/TRAINING_SIZE_CURVE.json, read with TRAINING_SIZE_BET.json)
- P70 (A1_PO_16 recipe on the whole non-panel pool, 14,022 people, 70% of MODEL_FIT), MODEL_SELECT with H3 income: men's guard -0.00179
  (s0) / -0.00026 (s1); occupation NLL excess vs U0 -0.0113 / -0.0079; men's reserve slack +0.0032 / +0.0019 (positive again).
- Registered predictions: size alone (log-linear between 53% and 100%) men's -0.0011 / 0.0000, NLL -0.0105 / -0.0069; flat beyond 53%
  +0.0011 / +0.0019, -0.0060 / -0.0021. Observed: at or slightly beyond the log-linear prediction, within the fit-to-fit spread at a fixed
  size (paired MODEL_SELECT differences D1 - D0: NLL +0.0019 (SE 0.0019), men's -0.0000 (0.0012); R1 - R0: +0.0006 (0.0016), -0.0008
  (0.0010)). Registered reading: consistent with a training-size effect being sufficient; a fit-specific selection component at 100%
  is not excluded.
- Paired differences on the same MODEL_SELECT people: P70 - R0 NLL -0.0053 (0.0013), men's Brier -0.0029 (0.0008); source s0 - P70 NLL
  -0.0049 (0.0013), men's -0.0021 (0.0008); P70A (admitted dictionary) - P70 NLL -0.0007 (0.0004), men's -0.0005 (0.0002): the FIT-only
  dictionary collapse explains little (share of MODEL_SELECT people with a collapsed POBP level: 2.1-2.5% at 42%, 1.1-1.3% at 53%, 0.3%
  at 70%).
- Consequence for the design (not a claim of this campaign): the honest repair design's 42-53% training sets cost most of the witness
  margin that the source spent; a design that keeps a larger training share (or estimates costs by properly nested cross-fitting on
  the full training set) would keep more slack. Nothing here changes NO_RECIPE.

## F8. Question A on the diagnostic partitions (20:27; cts.questiona, RULES.md section 6; D = MAE_rel(OOF) - MAE_rel(GUARD), relative to
## the statistic's reserve slack, paired household bootstrap B = 2000; PREDICTION_IMPROVED iff D > 0 and the 5% quantile > 0)
- REFIT_PAIR full: D0 D -0.51 (q05 -1.28) NOT_IMPROVED (MAE_rel GUARD 0.88 vs OOF 0.37); D1 D +0.16 (q05 -0.22) NOT_IMPROVED (0.66 vs 0.82).
- REFIT_PAIR ens: D0 -0.38, D1 +0.44 (q05 -0.13): NOT_IMPROVED. REFIT_LOC full / ens: NOT_IMPROVED on both partitions.
- REFIT_CONT: PREDICTION_IMPROVED on D1 (full and ens) and D0 ens; NOT_IMPROVED D0 full.
- FROZEN_PAIR (A1_NOT_SUPPORTED): D1 ens PREDICTION_IMPROVED; others NOT_IMPROVED.
- Nested estimator in place of OOF (D0): REFIT_PAIR D +0.42 (q05 -0.63), REFIT_LOC +0.20: GUARD closer than the nested estimator,
  not resolved.
- Reading: for the correction directions, an excluded-household GUARD estimate is not reliably better than the source's out-of-fold
  estimate at predicting realized CHECK costs with these sample sizes: GUARD wins where the out-of-fold estimate is biased (D1) and
  loses where its own sampling noise dominates (D0). Either estimator's average error is 0.4-1.3 times the available slack.

## F9. Question A under the registered bootstrap (20:36; diagnosis/QUESTION_A_LABELS.csv; PROTOCOL_DEVIATIONS items 10-11)
- The verifier found that the first implementation resampled households separately for each statistic. RULES.md section 6 registers
  one resample per set. The corrected run (one shared resample per set, B = 2000, registered tag) gives the primary labels; the first
  labels are kept as superseded. D, the MAE terms and the point estimates are unchanged. The bootstrap SE of D is 0.99-1.37 times the
  superseded one (median about 1.1).
- Primary PREDICTION_IMPROVED (5 of 34 units): D1 FROZEN_PAIR ens (q05 +0.014; A1_NOT_SUPPORTED), D1 REFIT_CONT full (+0.380) and
  ens (+0.458), R0 REFIT_CONT full (+1.376) and ens (+2.017).
- Superseded to NOT_IMPROVED: D0 REFIT_CONT ens (q05 +0.010 -> -0.059), R1 REFIT_LOC full (+0.105 -> -0.020), R1 REFIT_PAIR_TR full
  (+0.050 -> -0.020). This matches the verifier's five-seed replays.
- No PAIR or LOC arm is PREDICTION_IMPROVED on any of the four partitions. GUARD beats the source's out-of-fold estimate only for
  the zero-critic continuation CONT, on D1 and R0, where the out-of-fold estimate badly over-predicts CONT's own small costs.
  The nested estimator (D0) does not resolve PAIR or LOC either.

## F10. Rules on the repair panels (20:48; repair/RULE_OUTCOMES_REPAIR.csv, repair/RULE_COMPARISON.json)
- R1 (men's reserve slack -0.00031): every rule selects identity for every direction and deployment. The identity outcome is itself
  a reserve crossing (the witness alone fails the two-SE reserve; verifier ATTENTION, wording only).
- R0 (men's slack +0.00014): REFIT_PAIR full: OLD-B80 0.027 crosses on the C0 panel (max slack use 1.50; MODEL_SELECT 0.96); OLD-B50
  0.017 does not (0.94 / 0.60); SIMPLE and UNCERTAINTY select identity. REFIT_PAIR ens: SIMPLE 0.5 crosses on both (13.8 / 10.5).
  REFIT_LOC full: both OLD levels cross on C0 (1.05, 1.67). REFIT_PAIR_TR full: OLD 0.014 / 0.023 do not cross (0.60 / 0.96);
  SIMPLE ens 0.2 crosses (4.8 / 3.9).
- compare_rules: SIMPLE and UNCERTAINTY never improve on OLD on both R partitions. In the full deployment they are identity
  (A_IDENTITY_NO_IMPROVEMENT). In the ensemble deployment SIMPLE crosses (A_CROSSES). On R1 every rule ties at identity.
- Realized MODEL_SELECT utilization of REFIT_PAIR on R0 at the reader-evaluated strengths: u = 3.8 (s = 0.1), 8.4 (0.2), 13.7 (0.3),
  26.6 (0.5), 70.1 (1.0). On R1 u is infinite for every strength: the slack is negative, so any non-negative men's cost exceeds it.
  The continuation CONT has u <= 0.3 at s >= 0.3 on R1, because it lowers the men's Brier.

## F11. Track R whole-service refit (20:48; TRACKR_EVAL_MS per R partition, TRACK_R/<P>/CROSSCHECK.json)
- Each R partition's Track R occupation teacher is bitwise identical to its Track F REFIT routing teacher (R0 682 trees, R1 791).
  Track R adds only the refitted income lineage.
- H3R income (the H3 recipe refitted on FIT and CALIBRATION, without GUARD / CHECK households) loses H3's Brier / NLL headroom. Its
  income Brier excess vs U0 is +0.0016 (R0) and +0.0011 (R1), against H3 frozen -0.0028 / -0.0033; NLL is +0.0010 / +0.0006 against
  -0.0047 / -0.0052. Its income accuracy is slightly better: loss -0.0017 / +0.0014 against +0.0001 / +0.0017.
- Complete Track R releases (with the refitted witness) pass the 12 clauses and 2 guards at the point. They fail the reserve: R0 on
  income Ucal Brier (ratio 1.92), R1 on the men's guard (1.82, inherited from the witness). The task-only twin HCR keeps income
  headroom on R0 (ratio 3.46) but fails on R1 through the witness's men's guard.
- So the frozen H3 package's income margin does not survive a refit of its recipe on 75% of MODEL_FIT. This is the income-side
  limitation, and Track R cannot rescue the occupation side either: its occupation witness is the same refitted witness.

## F12. Repair-stage readers, reserve-qualified frontier and original benchmark (20:58; repair/, tables/READER_STABILITY_R_*)
- R_FRONTIER (R0 -> seed-0 cells, R1 -> seed-1 cells; 8 cells per comparison). REFIT_PAIR at s = 1.0:
  - vs W +0.0412 and vs CONT at 1.0 +0.0417, both STABLE_POSITIVE;
  - vs LOC at 1.0 +0.0096, SIGN_CHANGES;
  - vs SUCC +0.0083, NEURAL_RAW +0.0242 and H_LJ +0.0257, all POSITIVE_NOT_STABLE;
  - vs B2TWIN +0.0143 and RETIRED_B3CTL +0.0014, both SIGN_CHANGES; vs TWJ +0.0382, STABLE_POSITIVE.
- At s = 0.5: vs CONT +0.0285 STABLE_POSITIVE, vs W +0.0279 and vs LOC +0.0131 POSITIVE_NOT_STABLE. At s = 0.3: vs CONT +0.0113
  STABLE_POSITIVE.
- Controls against W: CONT -0.0006 to -0.0028 (never positive); LOC +0.0315 at s = 1.0 (POSITIVE_NOT_STABLE). The occupation-only
  critic carries most of the effect.
- These repeat the D-stage ordering (F6).
- Compression twins (R0 only, 4 cells): REFIT_PAIR beats its GUARD-utilization-matched twin by +0.004 to +0.018 at every strength.
  The 8-cell rule cannot label this.
- Track R: the refitted-income correction vs its own witness gives +0.047 (s = 1.0) and +0.023 (0.5), both STABLE_POSITIVE. Against
  Track F at equal strength it gives +0.006 / -0.005. H3R:W vs W gives -0.0001: the refitted income lineage does not change
  recovery.
- Reserve-qualified frontier (same rule for every point): no candidate qualifies; the witness fails (1.82 on R1); every required
  reference fails (source records). The only qualified points are CONT at s >= 0.3 (ratios 2.04-2.24), with pair AUC 0.634, the
  same as the witness. No reserve-qualified protected control exists.
- Original benchmark: REFIT_PAIR passes 24 + 4 at the point for s <= 0.5 and fails at s = 1.0. The old recipient gate fails at
  every strength (H3's income view). There is no superiority over all six required comparators at any strength. The 0.020 target
  reaches POINT_ONLY at s = 1.0 only, where utility fails.
- The men's guard binds every reader-evaluated point on both R partitions (repair/FRONTIER_DRIVERS.csv).
