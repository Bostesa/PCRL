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
