# Mechanism cards (index; exploratory development on previously used records)

Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only. The service predicts an
income class (2) and an occupation group (5) from AGEP, WKHP, COW, SCHL, MAR and POBP; each recipient receives its own float64
probability vector and argmax decision. SEX is the dataset's recorded demographic column; readers estimate it from the released
vectors. Each family's full card was written by its investigator **before that family's first fit** and revisions were appended
as they happened: `roles/investigator_h/CARD.md`, `roles/investigator_p/CARD.md`, `roles/investigator_o/CARD.md`. Numbers below
are MODEL_SELECT development values (seed 0 / seed 1; pair = development pair proxy, two-seed grid mean), from the reports and
`EXPERIMENT_LEDGER.csv`. No novelty is claimed for any family.

## Family H: headroom-first income learner (investigator H)

- **Operation.** Step 1: a stronger income task learner on MODEL_FIT (HistGradientBoosting chosen by household-separated 5-fold
  CV), stacked with the T_TREE income score on CALIBRATION ("STRONG"). Step 2: a budgeted transplant of the replication's stored
  critic-trained income corrections as a residual: d = a·(s_STRONG + D_LOCAL + β·D_JOINT) + b, β set on CALIBRATION by a cost
  budget, then a CALIBRATION Platt map. Occupation output: the frozen JOINT_TREE output.
- **Bottleneck addressed.** The retired code started from LOCAL/JOINT tree income scores with little headroom and spent ~0.0075
  of summed Brier, ending 0.35-0.40 SE from the cap.
- **Simple controls.** HC_STRONG (the learner alone; matched task-only twin); HC_TTREE_LJDIR(_PB) (the same step on the old
  T_TREE base: learner-isolation control).
- **Revise / abandon if.** No genuine held-out headroom over T_TREE (< 0.0005 Brier), or the step on the old base does as well.
- **Closest known.** Stacking / calibrated gradient boosting; the lineage's LOCAL/JOINT mixtures (H3 is ≈ s_L + 0.2-0.25·D_J on
  a better base).
- **Record.** H1 (β set by the registered rule) released β = 0: the noisy CALIBRATION accuracy term stopped it; pair ≈ H_LJ.
  H2 (decision-preserving residual, β 0.85) reached pair 0.6042 but income Brier +0.00497 (reserve not met) and a readable
  income view; abandoned. H3 (precision-aware rule; β 0.25 / 0.20): pair 0.6101 / 0.6136 at income Brier vs Ucal −0.0023 /
  −0.0033, RESERVE_MET (min ratio 2.862). STRONG's headroom over T_TREE (0.0016 / 0.0013 Brier) is visible on MODEL_SELECT but
  not on CALIBRATION (+0.0003); the CALIBRATION budget under-predicted the MODEL_SELECT Brier cost 6-9-fold.

## Family P: pair-aware learned income output, occupation frozen (investigator P)

- **Operation.** Income-only leaf-score corrections on the frozen T_TREE income routing, trained on MODEL_FIT against online SEX
  critics on the income wire and the linked pair wire (the accepted JOINT_TREE E20 occupation output, never trained), then one
  projection toward the teacher to an explicit label-free distortion budget (Brier / accuracy / KL against the teacher).
- **Bottleneck addressed.** Crude quantization (6 levels) pays Brier everywhere; JOINT_TREE's income part has no cost control.
- **Simple controls.** PC_TWIN_B40 (identical loop, critics off: matched twin); PC_TEMP (matched-cost temperature shrinkage);
  PC_JSCALE (the stored JOINT_TREE income corrections scaled toward the teacher to the same budget: no training).
- **Revise / abandon if.** A control explains the reduction at matched cost, or no budget inside the reserve reads below H_LJ.
- **Closest known.** JOINT_TREE / LOCAL_TREE (the transported L_TREE recipe), the source campaign's CCLC, adversarial debiasing,
  FAGTB, distortion-constrained privacy-utility formulations.
- **Record.** P1 (rho 3, per-epoch projection) was dominated by PC_JSCALE at the same budget (revise). P2 (rho 1, one end
  projection): pair 0.6133 / 0.6157, qualified with a thin seed-1 margin over B2TWIN. P3 (same training, larger budget): pair
  0.6072 / 0.6113 at income Brier vs Ucal −0.0001 / −0.0014, RESERVE_MET (2.677). Temperature at P3's cost reads 0.6363 (no
  effect). PC_JSCALE_B60 reads 0.6083 / 0.6120: the trained step adds only ~0.001 of pair over scaling the stored JOINT
  correction, plus some income utility.

## Family O: occupation-side margin repair (investigator O; the recorded expansion, PROTOCOL.md §5 and §8)

- **Operation.** An information-preserving recalibration of the JOINT_TREE occupation output, p' = softmax(clr(p)/T + b): T by
  occupation NLL on CALIBRATION; b on MODEL_FIT to equalize the two SEX groups' occupation Brier excess against U0 (SEX as a
  training-time label only). A bijection of the occupation vector.
- **Bottleneck addressed.** The inherited men's occupation guard (0.00442 / 0.00394 vs 0.005; 0.32 / 0.62 SE).
- **Simple controls.** OC_TCAL (temperature only, the brief's control), OC_TBCAL (temperature + bias by NLL on CALIBRATION),
  OC_BFIT (bias on MODEL_FIT without SEX).
- **Revise / abandon if.** The men's guard improves by < 1 SE, or only by pushing another clause within 1 SE, or occupation
  recovery rises by > 0.005.
- **Closest known.** Temperature / vector scaling; minimax group calibration.
- **Record.** O1_SEXBAL: men's guard 0.00166 / 0.00230 (1.83 / 1.62 SE), occupation NLL improved, occupation-view recovery not
  raised; ordinary calibration worsens the guard (OC_TCAL 0.00513 / 0.00509). No revision triggered; the 2-SE target on the
  men's guard is not met (an in-sample to MODEL_SELECT offset of about 0.0037).

## Coordinator combinations (reserve configurations)

CMB_P2JR_O1SEXBAL and CMB_P3JR_O1SEXBAL: a family-P income output beside O1's occupation output (no new fit). The occupation repair
coexists with the income-side gain (CMB_P3: pair 0.6079 / 0.6093, men's guard 0.00166 / 0.00230); they qualify but lose the
registered tie-break (equal income-reserve ratio, more model fits).
