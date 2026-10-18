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
