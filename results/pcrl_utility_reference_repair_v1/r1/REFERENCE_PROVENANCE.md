# R1 reference and learner provenance audit

This is a development comparison on previously reused Adult rows. It records
membership and implementation facts known before the new R1 task fits.

## Confirmed training and evaluation membership

- Frozen U0 in seeds 0 and 1 is the `smf` task-only `tl__s{k}__e40`
  continuation: a fresh task-only 20-epoch warm start plus 40 task-only
  continuation epochs on **all 15,434 OSF_DEFENSE_FIT rows**. The two source
  encoders use the 83 permitted predictors. The deployed StandardScaler and
  logistic task heads were fitted on that same role; their C grid was selected
  using historical HEAD_VALIDATION labels. This study does not reopen those
  labels. The admitted U model/head files and record are hash-bound in
  `hcal/admitted/rel__s{k}__U`; the original all-role `release.npz` is excluded
  from this study's restore path.
- Ucal is the same frozen U0 probabilities transformed by one H-GLOBAL-TEMP
  inverse temperature per task/seed. Temperatures were fitted on the 2,000
  CALIBRATION_HELDOUT representatives. The **calibration family** was chosen
  historically using INNER_SELECTION task outcomes. INNER has zero direct
  FIT/CAL row and group overlap, yet that family choice gives it prior
  selection exposure. New INNER intervals remain exploratory.
- In the preceding focused study, `m2.fit_pipeline.run_fit_pipeline` passed
  only the outer FIT training indices into each validation `fit_package`.
  `training.fit_task_head` refit on **all of that passed subset**, which was
  11,574 rows in seed 0 and 11,575 in seed 1. F and every mask were scored on
  the excluded 3,860 or 3,859 FIT validation rows. FIT receipt hashes bind
  these counts and group-disjoint split identities. U0's encoder and deployed
  heads had fitted on those scored rows. This confirms a labeled training
  exposure asymmetry in the old F versus U0 diagnostic; it does not quantify
  how much of the 0.0448–0.0603 occupation/income NLL gap it caused.
- Both old F and U0 used the same inherited 83-column input preprocessing:
  five numeric columns refit using OSF_DEFENSE_FIT and 78 fixed one-hot
  columns. Thus preprocessing saw all FIT feature values, while the old F
  learner did not see outer validation task labels. No SEX, race, income,
  occupation or fnlwgt predictor is in the schema.
- The prospective common comparison is the entire INNER_SELECTION role,
  2,235 rows in 2,234 groups. The inherited role manifest shows zero direct
  row/group overlap against FIT and CAL. This is a new prospective evaluation
  convention, with the old focused study's values and status preserved.

## Implementation differences relevant to repair

| Component | Frozen U0 | Focused-study F (recipe A) | Repair recipe B |
|---|---|---|---|
| Task acquisition | Two 83→64→64→16 encoders and affine training heads, task-only warm20 + continuation40 | Separate 83→64→64→task-class MLPs, Adam and FIT-internal epoch selection, max30 | Fresh source encoder/task architecture and schedule; no U0 weight copy |
| Deployed task head | StandardScaler + logistic on 16D encoder output, C grid chosen on historical HEAD_VALIDATION | Direct MLP softmax | StandardScaler + logistic on 16D, C grid selected on FIT-only nested fold |
| Input role for final fit | All FIT | Previous diagnostic: outer-FIT subset | New refit: all FIT |
| Score calibration | Identity for U0 | Held-out H-GLOBAL-TEMP | Same H-GLOBAL-TEMP, fitted only on admitted CAL |

The task names are income (two ordered classes) and occupation_group (six
ordered classes). Both old and new scoring use argmax accuracy, natural-log
NLL with fixed clipping, and summed multiclass Brier. R3 independently
recomputes these values in the new common-role gate.

## Source anchors and unresolved interpretation

The source protocol, `smf/run.py` stage_warm/stage_taskline,
`jcv/train.py` Model/warm_start, `smf/train.py` task-only continuation,
`rgj/finalize.py` deployed heads, `hcal/stages.py` Ucal selection, prior
`m2/fit_pipeline.py` outer split and `teams/c/training.py` final refit are
the operative evidence. The old full-input gap combines different training
membership and learner/head recipes. A fair-role diagnostic of the original
saved F on INNER may separate the evaluation-population effect, but cannot
by itself isolate which training component produced a new improvement.
