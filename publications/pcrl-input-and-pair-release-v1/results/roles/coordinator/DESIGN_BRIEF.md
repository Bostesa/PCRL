# Design brief (coordinator, campaign `pcrl_input_and_pair_release_v1`, code package `ipr/`)

Written 2026-10-22 from 01:30Z (`date -u`). Campaign start 01:19:06Z; the research window ends 09:19:06Z, closeout 09:49:06Z.
Branch `research/pcrl-input-and-pair-release-v1` (from c577424c0, the dual-design release tip) in the worktree
`.claude/worktrees/pcrl-correction-transfer-stability-v1`.

## Purpose (keep it explicit)

A statistical machine-learning benchmark on already admitted public-use U.S. Census ACS 2019 1-Year PUMS person records for
Florida. A service predicts an income class (2 classes) and an occupation group (5 classes) from six raw fields. The privacy
evaluation measures whether classifiers can infer the dataset's recorded SEX column (0 = men, 1 = women) from the probability
vectors released for those tasks. Goal: a **materially different release method** that keeps useful predictions while reducing
SEX inference from each recipient's output and from the linked pair. A successful method is not assumed; a well-run negative result
is acceptable.

## Fixed setting (unchanged from the previous campaigns)

- Runtime inputs: AGEP, WKHP, COW, SCHL, MAR, POBP only (admitted 145-column encoding X; compact 6-column coding
  `ohn.fam_a.compact_po(X)` = Z with COW, MAR categorical). SEX is never a runtime input or output.
- Released: each recipient (income, occupation) receives only its own float64 probability vector and its argmax decision. Readers
  may link the two outputs of a person (the pair view). A randomized release is allowed (one emitted draw per person), but see the
  closed list.
- Roles: MODEL_FIT (20,000 people, 13,337 households) for all fitting, with income, occupation and SEX labels available as training
  labels under a committed mechanism card; CALIBRATION (2,982 people; income / occupation labels only, never SEX); MODEL_SELECT
  (8,001 people, 5,306 households) for evaluation and readers only, never a fitting input. Cross-fitting uses household folds.
  Everything else (ATTACK_*, DEV_REPORT labels, sealed assessment, capped-out records) is closed. No new records.
- You may retrain both task models and redesign the release construction. Exact source task definitions and household grouping
  are kept.
- Utility (original limits, MODEL_SELECT, two versions v = 0, 1): accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier
  excess <= 0.005 against U0 and Ucal for both tasks (24 clauses) plus men's and women's occupation summed-Brier excess vs U0 <=
  0.005 (4 guards). Recipient non-worsening: neither recipient's own view may become more SEX-revealing than the comparator's (limit
  0.010). The two-SE reserve is a separate heuristic. The 0.020 pair-AUC margin over NEURAL_RAW and H_LJ is the material criterion,
  kept separate from smaller gains.
- Budget for the whole campaign: 48 new release configurations (controls included), 12 CPU-hours, two heavy jobs at once with one
  thread each. Fits must be cheap: sklearn HGB on 20,000 rows is about 5-30 CPU-s per fit.

## What is already known (read these; do not re-derive them)

- `results/pcrl_dual_design_release_v1/FAILURE_MAP.md`: rows 1-13 of the ledger of earlier mechanism families and their dominant
  failures (rows 1-10 in detail in `results/pcrl_new_release_mechanisms_v1/FAILURE_MAP.md`).
- `results/pcrl_dual_design_release_v1/RESEARCH_DECISION.md`, `ADVISOR_BRIEF.md` and `TABLES.md` (row 14 below).
- `results/pcrl_finite_routing_comparison_v1/RESEARCH_DECISION.md`, `results/pcrl_finite_solver_adequacy_v1/RESEARCH_DECISION.md`.

**Row 14 (dual-design release, publication e30cf1c6b).** Two families, judged one idea family by its challenger: both subtract the
propensity-proportional part of the released posterior. DA (SEX-standardized posteriors STD; revision IPW-STD) and DB
(propensity-tilt cancellation PTC; revision R1 with split strengths 0.35 / 0.65). Registered grid (8 cells, 16-recipe bank):

| Release | Pair | Income view | Occupation view | Total NLL excess vs U0 (v0 / v1) |
|---|---|---|---|---|
| DA_TWIN (cached HGB task teachers; the shared twin) | 0.6506 | 0.5606 | 0.6450 | -0.0247 / -0.0209 |
| DA_STD (registered) | 0.6331 | 0.5454 | 0.6206 | -0.0167 / -0.0128 |
| DB_PTC_R1 (adaptive revision; strongest) | 0.6267 | 0.5333 | 0.6166 | -0.0149 / -0.0110 |
| DB_ZTILT_R1 (release-only tilt: amplitude P(SEX given the twin's release); overspent) | 0.6395 | 0.5329 | 0.6177 | -0.0078 / -0.0041 |
| DA_SPD_HI (SEX-blind dithered quantization of the twin, DQ) | 0.6468 | 0.5569 | 0.6411 | -0.0124 / -0.0023 |
| U0 (utility reference) | 0.6498 | 0.5486 | 0.6506 | 0 |
| NEURAL_RAW (target reference) | 0.6377 | 0.5348 | 0.6370 | +0.0017 / +0.0044 |
| H_LJ (target reference; LOCAL income + JOINT-tree occupation) | 0.6209 | 0.5464 | 0.5600 | -0.0033 / -0.0032 |
| TWJ (twin income + JOINT-tree occupation) | 0.6476 | 0.5606 | 0.5600 | -0.0058 / -0.0055 |
| SUCC (H3 income + JOINT-tree occupation) | 0.6145 | 0.5310 | 0.5600 | -0.0022 / -0.0023 |
| RETIRED_B3CTL (6-level income code + JOINT occupation) | 0.6000 | 0.5329 | 0.5600 | +0.0081 / +0.0093 |

Lessons: D2 (beyond matched controls) was met in 0 of 4 looks. DB_PTC_R1: +0.0239 (se 0.0101) vs the twin, +0.0245 (0.0122) vs
DQ, +0.0265 (0.0139) vs its SEX-independent-attribute control, and only +0.0128 (0.0081) vs the release-only tilt, which needs no
propensity at all. Realized se_agg was 0.009-0.018, so effects below about 0.02 pair AUC do not resolve on these 8,001 people.
Note TWJ: the JOINT-tree occupation output is weak alone (0.560) but the linked pair with an accurate income output is back at 0.648.

Measured failures any new method must face: (1) wherever recovery fell materially, task spending explained it (a SEX-blind control
at matched spend did about as well); (2) the linked pair is strongly super-additive (TWJ above); (3) leakage moves between
recipients; (4) readers are data-limited and finite releases favour table (LOOKUP) readers; (5) protecting income cost income
accuracy; (6) the strongest tilt was not separated from a control that uses only what can be estimated from the twin's own release;
(7) a better optimum of a plug-in dependence surrogate did not transfer to held-out readers.

A useful way to see (6): if the twin's released vectors were a sufficient statistic of x for both tasks, any SEX-targeted change of
the release could be imitated by a function of the twin's release fitted with SEX labels. A new method should say why its gain would
not be reproduced at equal spend by (i) SEX-blind spending (DQ), (ii) the same operation with a SEX-independent attribute, and
(iii) a release-only transformation of the twin's vectors (for example the release-only tilt).

## Closed: do not continue or re-tune

Propensity-routed finite cells and their solvers (PRC, HQ, SHUF, UNT, ORD routing; SV1); Gaussian logit channels including the
coupled (joint-direction) SEX-targeted channel FB2 and its SEX-blind twin; learned decoders and calibrators as mechanisms;
controllers (AUC / strength controllers, refreshed critics); leaf-score corrections (PAIR / LOC / CONT); **and the dual-design idea
family: SEX-standardized posteriors, inverse-propensity reweighting of the task models, linear propensity tilts and release-only
tilts (STD, IPW-STD, PTC, ZTILT and re-weightings or re-parametrizations of them).** Re-labelling these is not a new method.
Reusable infrastructure (not mechanisms): the cached task and SEX-side HGB teachers, the shared twin DA_TWIN, the matched-control
kit (`ddr/ctlkit.py`: the SEX-independent attribute PSEUDO2 and the dithered-quantization spend path DQ), the evaluation, reader and
rules code.

## Your design space

- **Designer A: before the release.** What the task models see and learn from: the six inputs (generalization, suppression,
  smoothing or re-encoding of fields, field-structured models), the training rows or targets (other than SEX standardization or
  inverse-propensity reweighting), or the task models' training objective (other than the closed critics, penalties and leaf
  corrections of rows 1-3 and 10). The released vectors are the trained models' outputs, perhaps with a fixed simple map.
- **Designer B: after the task models, jointly for the linked pair.** How the two released vectors are formed from fitted
  statistics, designed for the pair (the super-additive view) rather than recipient by recipient: for example nonlinear or
  conditional maps where one recipient's release depends on what the other recipient receives, or other joint constructions with
  continuous or finite outputs. Not linear tilts, routing to finite cells by propensity, logit noise, decoders or controllers.

Do not read the other designer's directory. Known published components are acceptable; claim no novelty you cannot support.

## Reusable inputs for label-free design work (no label file may be opened before registration)

- `.private/pcrl_new_release_mechanisms_v1/fa_cache/V<v>/TASK.pkl`: dict with 'models' {'inc','occ'} (sklearn HGB, A1_PO_16
  settings, fitted on MODEL_FIT), 'oof' {'inc': (20000, 2), 'occ': (20000, 5)} 5-fold household out-of-fold predictions on
  MODEL_FIT, 'T', 'feature_names', 'row_id_sha256'.
- `.private/pcrl_finite_routing_comparison_v1/fa_cache/V<v>/SEXSIDE.pkl`: 'models' {'sex', 'occ_s'} and 'oof' {'sex': (20000, 2)
  = P(SEX | x), 'occ_s': {0: (20000, 5), 1: (20000, 5)} = P(occupation | x, SEX = s)}; fitted with MODEL_FIT SEX as a training
  label.
- Teacher code: `nrm/fa/prc.py` (fit_teachers, `_prep`, CAT_Z, HGB settings). Compact coding: `ohn/fam_a.py`. Admitted features
  (label-free): `ddr.common.features('MODEL_FIT')` returns X, feature_names, row_id, unit.
- You may compute label-free feasibility estimates from these cached out-of-fold predictions and the admitted features (for example
  the expected KL from the task posterior that your operation would spend, or a model-based expected SEX AUC using the cached
  propensity). Do not open any label file, any `admitted/exports/*/{income,occupation,sex}.npz`, anything of MODEL_SELECT, or any
  reader cache. Use one numerical thread (OMP_NUM_THREADS=1 etc.), keep it under 400 CPU-seconds, and report the CPU seconds used.

## Your deliverable

Write `results/pcrl_input_and_pair_release_v1/roles/designer_<a|b>/PROPOSAL.md` (about 2 pages, plain prose and tables) with:

1. **The exact operation** from the permitted inputs to the released probabilities: fitted statistics, formulas, which role and which
   labels fit each part, hyperparameters, the strength path (a small predeclared grid) and how a strength is selected using
   MODEL_FIT (or CALIBRATION task labels) only.
2. **Closest previous implementation** (repository path and campaign) and **what is substantively different**.
3. **Which measured failure it addresses**, citing the numbers above or in the records.
4. **Why it might preserve task usefulness**, with a label-free feasibility estimate where possible.
5. **The equally trained task-only twin** and **the simplest matched control** (the simplest alternative explanation of any gain:
   same spend, no SEX-specific information), plus the release-only and SEX-independent-attribute controls needed for a fair reading.
6. **A bounded revision plan**: at most two substantive revisions, each triggered by a named observable failure.
7. **Registered predictions** (pair / per-view AUC against the twin and each control; utility), the number of configurations
   (twin, strengths, controls, ablation; the campaign allows about 16 per family across both versions), and a CPU estimate.
8. **Novelty status**: the closest published method(s) you know; claim nothing you cannot support.

Also write `results/pcrl_input_and_pair_release_v1/roles/designer_<a|b>/mech_sketch.py`: a short, self-contained, label-free
reference implementation of the operation's fit and release functions on plain numpy arrays (signature
`fit(Z, y_inc, y_occ, sex, unit, version, strength, **kw) -> state` and `release(state, Z) -> (P_inc (n,2), P_occ (n,5))`; for a
release-side operation the twin's vectors may be passed in), plus `selftest()` on invented data only. Keep it simple; the coordinator
integrates it into the campaign package and the queue.

Prefer a method that is simple, cheap to fit, and has a clear reason to beat its matched controls at equal task spend by at least
about 0.02 pair AUC (smaller effects will not resolve). Return a 10-line summary when done. Please aim to finish within about
40 minutes; the coordinator integrates both proposals and a challenger reviews them before any card is committed.

## Correction (coordinator, after about 01:40Z)

In the table above, TWJ is a strong task-only income output (HC_STRONG, the earlier campaigns' "TW" anchor; not this campaign's
DA_TWIN income, although both read 0.5606 on that grid) combined with the JOINT-tree occupation output (`ohn/devrun.py`). The
lesson is unchanged: an accurate income output linked with the JOINT-tree occupation output brings the pair back to 0.648.
