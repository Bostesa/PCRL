# Design brief (coordinator, campaign `pcrl_dual_design_release_v1`, code package `ddr/`)

Written 2026-10-21 16:45Z (`date -u`). Campaign start 16:37:33Z; research window ends 00:37:33Z on 2026-10-22.
Branch `research/pcrl-dual-design-release-v1` in the worktree `.claude/worktrees/pcrl-correction-transfer-stability-v1`.

## Purpose (keep it explicit)

A statistical machine-learning benchmark on already admitted public-use U.S. Census ACS 2019 1-Year PUMS person records for
Florida. A service predicts an income class (2 classes) and an occupation group (5 classes) from six raw fields. The privacy
evaluation measures whether classifiers can infer the dataset's recorded SEX column (0 = men, 1 = women) from the probability
vectors released for those tasks. Goal of this campaign: a **materially different release method** that keeps useful
predictions while reducing SEX inference from the individual outputs and from the linked pair. A successful method is not
assumed; a well-run negative result is acceptable.

## Fixed setting

- Runtime inputs: AGEP, WKHP, COW, SCHL, MAR, POBP only (admitted 145-column encoding X; compact 6-column coding
  `ohn.fam_a.compact_po(X)` = Z with COW, MAR categorical). SEX is never a runtime input or output.
- Released: each recipient (income, occupation) receives only its own float64 probability vector and its argmax decision.
  Readers may link the two outputs of a person (the pair view). A randomized release is allowed (one emitted draw per person).
- Roles: MODEL_FIT (20,000 people, 13,337 households) for all fitting, with income, occupation and SEX labels available as
  training labels under a committed mechanism card; CALIBRATION (income/occupation labels only, no SEX); MODEL_SELECT (8,001
  people, 5,306 households) for evaluation and readers only, never a fitting input. Any cross-fitting uses household folds.
  Everything else (ATTACK_*, DEV_REPORT labels, sealed assessment, capped-out records) is closed. No new records.
- You may retrain both task models and redesign the release construction. Exact source task definitions and household grouping
  are kept.
- Utility (original limits, MODEL_SELECT, two versions v = 0, 1): accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed
  Brier excess <= 0.005 against U0 and Ucal for both tasks (24 clauses) plus men's and women's occupation summed-Brier excess vs
  U0 <= 0.005 (4 guards). Recipient non-worsening: neither recipient's own view may become more SEX-revealing than the
  comparator's (nrm.rules recipient gate, limit 0.010). The two-SE reserve is a separate heuristic. The 0.020 pair-AUC margin
  over NEURAL_RAW and H_LJ is the material criterion, kept separate from smaller gains.
- Budget for the whole campaign: 48 new release configurations (controls included), 12 CPU-hours, two heavy jobs at once with
  one thread each, about 7.5 research hours left. Fits must be cheap: sklearn HGB on 20,000 rows is about 5-30 CPU-s per fit.

## What is already known (read these; do not re-derive them)

- `results/pcrl_new_release_mechanisms_v1/FAILURE_MAP.md`: the ledger of ten earlier mechanism families and their dominant
  failures, plus the list of operations never tried on ACS FL.
- `results/pcrl_finite_routing_comparison_v1/RESEARCH_DECISION.md` and `TABLES.md`: propensity-routed finite cells (PRC) fitted
  and not separated from task-score-only grouping (HQ), untargeted moves (UNT) or SEX-blind coarsening (ORD).
- `results/pcrl_finite_solver_adequacy_v1/RESEARCH_DECISION.md`: a better solver of PRC's objective (SV1) improves the surrogate
  only.
- `results/pcrl_new_release_mechanisms_v1/RESEARCH_DECISION.md`: SEX-targeted Gaussian logit noise (FB2) not separated from
  SEX-blind noise of the same strength; flattening the occupation remainder (CTL_REM) did nothing until it broke utility.

Registered-grid numbers (finite-routing campaign, 8 cells, 16-recipe bank; pair / income view / occupation view AUC):

| Release | Pair | Income | Occupation | Total NLL excess vs U0 (v0 / v1) |
|---|---|---|---|---|
| TWIN (16 x 64 finite, B1 tables) | 0.6555 | 0.5681 | 0.6379 | -0.0147 / -0.0143 |
| U0 (utility reference) | 0.6497 | 0.5566 | 0.6443 | 0 |
| W (A1_PO_16 occupation + H3 income) | 0.6506 | 0.5414 | 0.6480 | -0.0208 / -0.0180 |
| TR100 / TT100 | 0.6595 / 0.6550 | 0.5414 | 0.6508 / 0.6536 | about -0.017 |
| NEURAL_RAW (target reference) | 0.6309 | 0.5406 | 0.6329 | +0.0017 / +0.0044 |
| H_LJ (target reference; LOCAL income + JOINT-tree occupation) | 0.6252 | 0.5504 | 0.5727 | -0.0033 / -0.0032 |
| SUCC (H3 income + JOINT-tree occupation) | 0.6140 | 0.5414 | 0.5727 | -0.0022 |
| RETIRED_B3CTL (6-level income code + JOINT occupation) | 0.6027 | 0.5376 | 0.5727 | +0.0081 / +0.0093 |

Note the super-additivity of the linked pair (H_LJ: 0.5504 and 0.5727 alone, 0.6252 linked) and that the strongest task-only
bases have about 0.02 nats of total NLL headroom against U0 (FB2_BASE: -0.0245 / -0.0212).

Measured failures any new method must face: (1) wherever recovery fell materially, it was explained by task spending (a SEX-blind
control at matched spend did as well); (2) the linked pair leaks far more than either view; (3) leakage moves between recipients;
(4) finite releases favour table (LOOKUP) readers and readers are data-limited (AUC rises with reader data); (5) protecting income
cost income accuracy (H_JJ).

## Closed: do not continue or re-tune

Propensity-routed finite cells and their solvers (nrm/fa, frc/fa, fsa SV1; PRC, HQ, SHUF, UNT, ORD routing), Gaussian logit
channels (FB and FB2, SEX-targeted or blind), learned decoders and calibrators as mechanisms, controllers (AUC / strength
controllers, refreshed critics), leaf-score corrections (PAIR / LOC / CONT on HGB or CatBoost leaves). Re-labelling these is not
a new method. Reusable infrastructure (not mechanisms): the B1 teacher-mean finite twin, the cached task and SEX-side HGB teachers
(below), the evaluation, reader and rules code.

## Reusable inputs for label-free design work (no label file may be opened before registration)

- `.private/pcrl_new_release_mechanisms_v1/fa_cache/V<v>/TASK.pkl`: dict with 'models' {'inc','occ'} (sklearn HGB, A1_PO_16
  settings, fitted on MODEL_FIT), 'oof' {'inc': (20000, 2), 'occ': (20000, 5)} 5-fold household out-of-fold predictions on
  MODEL_FIT, 'T', 'feature_names'.
- `.private/pcrl_finite_routing_comparison_v1/fa_cache/V<v>/SEXSIDE.pkl`: 'models' {'sex', 'occ_s'} and 'oof' {'sex': (20000, 2)
  = P(SEX | x), 'occ_s': {0: (20000, 5), 1: (20000, 5)} = P(occupation | x, SEX = s)}; fitted with MODEL_FIT SEX as a training
  label.
- Teacher code: `nrm/fa/prc.py` (fit_teachers, embed, embed_oof, CAT_Z, HGB settings). Compact coding: `ohn/fam_a.py`.
- You may compute label-free feasibility estimates from these cached out-of-fold predictions (for example the expected KL from
  the task posterior that your operation would spend, or a fitted plug-in dependence). Do not open any label file, any
  `admitted/exports/*/{income,occupation,sex}.npz`, anything of MODEL_SELECT, or any reader cache. Use one numerical thread
  (OMP_NUM_THREADS=1 etc.), keep it under 300 CPU-seconds, and report the CPU seconds you used.

## Your deliverable

Write `results/pcrl_dual_design_release_v1/roles/designer_<a|b>/PROPOSAL.md` (about 2 pages, plain prose and tables) with:

1. **The exact operation** from the permitted inputs to the released probabilities: fitted statistics, formulas, which role and
   which labels fit each part, hyperparameters, the strength path (a small predeclared grid) and how a strength is selected
   using MODEL_FIT (or CALIBRATION task labels) only.
2. **Closest previous implementation** (repository path and campaign) and **what is substantively different**.
3. **Which measured failure it addresses**, citing the numbers above or in the records.
4. **Why it might preserve task usefulness**, with a label-free feasibility estimate where possible.
5. **The equally trained task-only twin** and **the simplest matched control** (the simplest alternative explanation of any
   gain: same spend, no SEX-specific information), plus any further control needed for a fair reading.
6. **A bounded revision plan**: at most two substantive revisions, each triggered by a named observable failure.
7. **Registered predictions** (pair / per-view AUC against the twin and the control; utility), the number of configurations
   (twin, strengths, controls, ablation), and a CPU estimate.
8. **Novelty status**: the closest published method(s) you know; claim nothing you cannot support. Known published components
   are acceptable.

Also write `results/pcrl_dual_design_release_v1/roles/designer_<a|b>/mech_sketch.py`: a short, self-contained, label-free
reference implementation of the operation's fit and release functions on plain numpy arrays (signature
`fit(Z, y_inc, y_occ, sex, unit, version, strength, **kw) -> state` and `release(state, Z) -> (P_inc (n,2), P_occ (n,5))`, plus
`selftest()` on invented data only). Keep it simple; the coordinator integrates it into the campaign package and the queue.

Prefer a method that is simple, cheap to fit, and has a clear reason to beat its matched SEX-blind control at equal task spend.
Do not read the other designer's directory. Return a 10-line summary when done.
