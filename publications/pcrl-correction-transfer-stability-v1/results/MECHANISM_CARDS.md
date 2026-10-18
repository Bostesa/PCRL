# Mechanism cards (correction-transfer campaign)

The operation, controls and checks of every mechanism below were registered in `PROTOCOL.md` sections 6-10, `PARTITIONS.json`
and the statistical reviewer's `RULES.md` (registration commit 39a27cc1e, public receipt 2ad31d2af, 19:13:08Z), before the
first fit of this campaign (19:13:16Z). This file collects them as cards (compiled 19:25Z, during the first fits; nothing below
changes a registered definition). Results are appended under each card with their time; earlier text is not rewritten. A direction
redesign, if any, gets its own card committed before its first fit.

## Card 1. Refitted occupation witness (A1_PO_16 recipe on FIT)
- Operation: scikit-learn HistGradientBoosting multinomial on compact_po coding (AGEP, WKHP numeric; COW, MAR native categories;
  SCHL ordinal; POBP ordinal place-of-birth code) of the admitted encoding re-collapsed with a FIT-only dictionary; 16 leaves, min
  leaf 100, lr 0.025, l2 3; T* <= 1000 = first minimum of the pooled household 5-fold inner out-of-fold NLL on FIT; refit on FIT.
  Household 5-fold cross-fitting on FIT (nested inner CV) gives the out-of-fold base and five fold pipelines. No SEX, no calibrator.
- Failure addressed: the source witnesses were fitted on all MODEL_FIT, so every MODEL_FIT household was in-sample and the source's
  two seeds were near aliases (identical boosting path truncated at a different tree count).
- Controls and checks: four genuinely different FIT sets (D0, D1, R0, R1); model identity audited by the verifier; MODEL_SELECT
  utility against U0 / Ucal reported per partition.
- Closest established method: gradient-boosted trees with ordinal coding of a high-cardinality categorical (established practice).

## Card 2. Routing for the leaf-score correction (REFIT vs FROZEN)
- Operation: REFIT = the replication's T_TREE occupation recipe (CatBoost 1.2.8, depth 4, lr 0.05, <= 800 trees, l2 3; tree count on
  the 15% household internal split; refit) on FIT; FROZEN = the admitted T_TREE of the associated seed (all MODEL_FIT).
- Failure addressed: frozen routing is one of the disclosed possible explanations of the source transfer error.
- Check: routing reproduction of the admitted seed-0 teacher (exact), CatBoost-to-numpy parity on every role.

## Card 3. Correction directions PAIR, LOC and the continuation control CONT
- Operation: softmax(log p_W(x) + s * Delta(x)), Delta = theta[t, leaf_t(x)] summed / T on the routing; theta trained with the source
  P2 recipe (`ohn.protect.train_correction`, guard 'none'): task cross-entropy from the cross-fitted witness logits plus lambda x
  critic risk; PAIR = online SEX critics on the occupation wire and on the linked pair with H3's frozen income wire (coalition-aware
  occupation correction; not a joint two-output design); LOC = occupation-wire critics; CONT = lambda 0 with identical effort and
  random streams (reads FIT SEX for inactive critics, so it is not called SEX-blind). Five fold pipelines + the full-FIT pipeline.
- Deployments: full (full-FIT theta on the deployed witness) and ensemble (mean of the five fold pipelines).
- Closest established methods: conditional adversarial debiasing (Zhang, Lemoine & Mitchell 2018); adversarial gradient tree
  boosting (Grari et al. 2019); the source's evidence table.

## Card 4. Strength rules OLD, SIMPLE, UNCERTAINTY (statistical reviewer; RULES.md sections 3-5)
- OLD: the source projection on out-of-fold FIT costs, budget f x reserve slack (B80 primary, B50 secondary).
- SIMPLE: GUARD point costs <= 0.5 x reserve slack per condition, ascending fixed sequence on the ladder 0, 0.1, ..., 1.0.
- UNCERTAINTY: GUARD costs with household-clustered SEs, a predictive SE for a MODEL_SELECT-sized sample, the baseline SE, one-sided
  z at alpha 0.05, intersection-union over the 8 conditions, fixed sequence; asymptotic and model-based (no finite-sample
  certificate).
- Identity is a valid outcome and earns no protection claim. Rules are compared on the same fitted directions.

## Card 5. Nested out-of-fold estimator (diagnosis; D0)
- Operation: every upstream layer (dictionary, witness and its own cross-fitting, REFIT routing, correction) refitted inside each
  outer fold of FIT; nested out-of-fold costs on the ladder compared with the non-nested source estimator and with GUARD / CHECK.
- Failure addressed: non-nested upstream cross-fitting (a disclosed possible explanation).

## Card 6. Compression control at matched GUARD cost
- Operation: label-free K-means code of the cross-fitted FIT witness vectors (source COMP recipe); released vector = nearest centroid
  of the deployed witness vector; K = smallest ladder value whose GUARD utilization does not exceed the candidate's.

## Card 7. Sensitivity witness (B2_SINGLE recipe on D0's households, partition alias D0B)
- Operation: the source B2_SINGLE recipe (better of TREE_V2 and ADD_V2 by temperature-scaled inner out-of-fold NLL, with that
  temperature; full nested cross-fitting) refitted on D0's FIT with FIT-only preprocessing; the same directions and ladder.
- Role: bounded sensitivity check of the witness recipe; never enters recipe selection.

## Card 8. Track R whole-service refit (R0, R1)
- Operation: T_TREE income and occupation, Stage B LOCAL and JOINT corrections (SEX critics on FIT), H3's HGB income learner,
  CALIBRATION stack, beta budget and Platt map, all refitted without GUARD / CHECK households; then the PAIR direction with the
  refitted income wire. A recipe test on different training subsets, not confirmation of the frozen H3 package.

## Results (appended 21:00; development evidence; numbers x 1e-3 unless stated; records in START_HERE.md)

- **Card 1 (refitted witness).**
  - Four genuinely different fits: D0 and D1 on 42% of MODEL_FIT, R0 and R1 on 53%.
  - MODEL_SELECT men's-guard excess: +0.65 / +2.15 / +1.11 / +1.89, against -3.9 / -2.3 for the source witness on all 20,000.
  - Men's reserve slack: +0.49 / -0.78 / +0.14 / -0.31.
  - Adaptive training-size diagnostic: at 70% (P70) slack recovers to +3.2 / +1.9 and occupation NLL excess vs U0 to -11.3 / -7.9.
  - Packaged with the six raw fields (R0, R1 and P70; bitwise and every refusal).
- **Card 2 (routing).**
  - The REFIT teachers equal the Track R occupation teachers bitwise.
  - FROZEN and REFIT show the same transfer error (D1 men's CHECK minus OOF -3.97 vs -5.14) and no consistent recovery difference.
- **Card 3 (directions).**
  - PAIR lowers recovery STABLE_POSITIVE vs W and vs CONT at s = 1.0 on both stages: R +0.041 / +0.042, D +0.041 / +0.034.
  - It also beats CONT at s = 0.5 and 0.3 on R. It is not stable vs LOC (+0.010 at 1.0), and LOC alone gives +0.032 vs W.
  - CONT reads like the witness (-0.0006 to -0.0028) and raises the men's-guard ratio slightly.
  - The men's guard binds everywhere: R0 u = 3.8 at s = 0.1 and 70 at s = 1.0; on R1 u is infinite.
- **Card 4 (rules).**
  - D stage: all 32 recipes fail G1 on D1 (identity), giving NO_RECIPE.
  - Repair panels: R1 identity for every rule. On R0, OLD-B50 0.017 keeps C0 (0.94) and OLD-B80 0.027 crosses it (1.50); SIMPLE
    and UNCERTAINTY choose identity in the full deployment, and SIMPLE ens 0.5 crosses tenfold.
  - SIMPLE and UNCERTAINTY never improve on OLD on both partitions.
  - Question A under the registered bootstrap: GUARD never predicts better than OOF for PAIR or LOC on any partition.
- **Card 5 (nested estimator, D0).** Removes most of the NLL over-prediction (CHECK minus nested +0.97, z +0.5). Under-predicts the
  men's cost (+2.70, z +2.0). Question A with the nested estimator in place of OOF gives NOT_IMPROVED for PAIR and LOC.
- **Card 6 (compression).**
  - R0 only: R1's negative slack leaves the matched target undefined.
  - PAIR beats its matched twin by +0.004 to +0.018 (4 cells, not labelled).
  - At s >= 0.5 the twins themselves fail the point utility (ratio -0.19 and -1.76 on seed 0).
- **Card 7 (sensitivity witness D0B, B2_SINGLE).** Men's reserve slack -2.02: an even smaller margin than A1_PO_16 on the same
  households. It never entered selection.
- **Card 8 (Track R).**
  - The refitted H3-recipe income loses H3's Brier / NLL headroom (+1.6 / +1.1 Brier excess vs U0, against -2.8 / -3.3).
  - Complete releases fail the reserve: R0 income Ucal Brier ratio 1.92, R1 men's guard 1.82.
  - The correction trained against the refitted income lowers recovery like Track F (+0.047 vs its own witness at s = 1.0,
    STABLE_POSITIVE; +0.006 vs Track F).
- **Corrections to the 21:00 results (21:08; verifier report).**
  - Card 4: the question-A statement holds for the REFIT routing under the registered scheme. Two R1 labels (LOC full and Track R
    PAIR full) were corrected after the repair panels had been read with the frozen code; under that code they are
    PREDICTION_IMPROVED (deviation 16, A5). The FROZEN PAIR on D1 (ensemble) is PREDICTION_IMPROVED with A1_NOT_SUPPORTED.
  - Card 6: the twins are not cost-matched. The K search assumes a monotone path, so the selected codes reach 6-55% of the target
    utilization (deviation 17). The +0.004 to +0.018 differences compare PAIR with cheaper codes and are descriptive only.
