# Exposure / refit investigator: report

Saved by the coordinator at 20:09Z from the refit investigator's returned final text (subagents return findings as text).

## Files
- `EXPOSURE_MAP.md` (registration file; written from source code by 19:02Z, registered at 39a27cc1e, receipt verified 19:13Z; its
  SHA-256 is pinned by the receipt, so corrections go in separate amendments).
- `cts/track_r.py` (committed at d95ea17fc before its first fit; selftest 11 / 12 / 11 checks pass in the torch, CatBoost and python3
  interpreters, including a check that all 22 fitted artifacts stay bitwise identical when GUARD / CHECK labels are permuted and
  their features altered).
- Outputs: `.private/<campaign>/runs/TRACK_R/{R0,R1}/{TEACHERS,STAGE_B,INCOME}/` and `CROSSCHECK.json` per partition.

## Exposure findings (from the code)
1. The source protection cross-fitting is not nested: fold pipeline k trained on base logits from witness fold models j != k, which
   were trained on fold k; the FROZEN T_TREE routing and H3's in-sample income wire (PAIR) also saw fold k.
2. Train / deploy mismatch: the out-of-fold costs that set the source scale use fold thetas on cross-fitted witness logits; the
   release uses the full-data theta on the deployed witness.
3. H3 reaches the occupation pipeline only through PAIR's pair-view critic; LOC and CONT do not depend on H3 (CONT and TASK_REFIT read
   SEX with zero weight).
4. H3 saw every MODEL_FIT household's income, occupation and SEX (its transplanted JOINT income correction carries occupation labels
   and SEX), so every GUARD, CHECK, C0 and C1 household.
5. MODEL_SELECT quantities enter several inputs: H3's B1 budget values, the reserve SE floor (H_LJ), the source projection budget and
   the choice of NEURAL_RAW rho 1 / E2; MODEL_SELECT costs of a strength chosen with a MODEL_SELECT budget are not independent of it.
6. The replication's occupation T_TREE tree count sits at the 800 cap; the internal split's support repair reads labels (it moved no
   household); the predecessor table's "no CatBoost installed" is out of date.

## Track R results (R0 / R1; source seed 0 / seed 1 in brackets)
- Teachers on FIT: income 487 / 460 trees [587 / 529]; occupation 682 / 791 [800 / 799]; CatBoost parity exact on all six roles.
- Stage B (E20, 800 updates per recipient; epoch coverage 0.973 / 0.975 of FIT): lambda_J [1.141, 0.660] / [0.962, 0.680]
  [[1.087, 0.686] / [1.273, 0.570]]; zero-correction identity exact across interpreters.
- H3-recipe income (H3R): HGB one-hot, 16 leaves, min leaf 20, 201 / 140 trees [compact coding: 16 / 20 / 94 and 8 / 80 / 185]; stack
  [w0, w_HGB, w_TTREE] [0.041, 0.068, 0.915] / [0.040, 0.202, 0.771] [[0.043, 0.384, 0.615] / [0.042, 0.304, 0.688]]: the refitted
  HGB carries far less weight and the refitted T_TREE income dominates; beta 0.25 / 0.30 [0.25 / 0.20]; Platt [-0.006, 0.967] /
  [-0.022, 0.957]. HCR = same stack, beta 0, identity Platt. Deployment parity bitwise on all six roles, both releases, both
  partitions.
- Crosscheck: the Track R occupation teacher is bitwise identical to Track F's REFIT routing (R0 594f956b..., 682 trees; R1
  494a9005..., 791 trees) and the FIT dictionaries match.
- CPU: teachers 10.5 / 11.9 s, Stage B 83.9 / 91.3 s, income 45.4 / 43.5 s (286.5 queue CPU-s against a 484 s estimate) plus 150
  CPU-s charged out of queue (REFIT_INVESTIGATOR_OUT_OF_QUEUE_V1).
- Deviations recorded in INCOME.json: FIT-only dictionary (z-map kept, tree-invariant); internal split, Step-1 folds and probe rows
  formed within FIT; Stage B epoch coverage about 97% of FIT (about 51% of MODEL_FIT in the replication, fixed steps); TASK_REFIT not
  run; Step-1 ExtraTrees and the CALIBRATION comparison table not refitted (not in the release); B1 kept (source values set from
  MODEL_SELECT); CALIBRATION shared with the frozen H3; U0 / Ucal not refitted. The investigator read no MODEL_SELECT label.
