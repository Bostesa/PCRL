# PCRL income-headroom successor development campaign: registration of boundaries, data access, development evaluation, reserve and nomination rules

**Status.** Registered before any new release or task-model fit in this campaign. The commit that adds this file, together
with `roles/statistical_reviewer/UTILITY_RESERVE_RULE.md`, `RESERVE_RULE.json`, `NOMINATION_RULE.md`, `DEV_GRID_REVIEW.md`,
`CLAIMS_DRAFT.md` and `ihs/reserve.py`, is the registration; a sanitized receipt is pushed to the publication branch before the
first fit. This is an exploratory development campaign on previously used Florida ACS 2019 development records. It is not
confirmation. Mechanism designs and revisions are exploratory decisions made inside the boundaries below and recorded as they
occur (`MECHANISM_CARDS.md`, `EXPERIMENT_LEDGER.csv`); they are not individually pre-registered.

**Disclosure.** The design was informed by the published lineage on these same records, in particular the source development
campaign (public `5901f7c73`) and its validation (`2abfaefdc`). DEV_REPORT is not historically unseen: the lineage access log
records 12 DEV_REPORT events before this campaign (`SOURCE_ADMISSION.json`). MODEL_SELECT has been used for selection by
earlier campaigns. No `AGENTS.md` exists in this repository (tracked files and all-branch history searched at 01:51Z); no
earlier successor branch or run existed, so nothing was resumed.

## 1. Census context and service (unchanged)

- Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida: housing-unit residents aged 25-64, civilian
  employed. Household grouping, eligibility and role assignment are those of the protected-tree replication.
- Tasks: income class (2 classes: PINCP x ADJINC / 10^6 above $50,000) and occupation group (5 classes: OCCP mapped to the five
  civilian broad groups), with the replication's definitions.
- Permitted runtime inputs: the six raw fields AGEP, WKHP, COW, SCHL, MAR, POBP, with the admitted input encoding (145 columns).
- Service contract: each recipient receives only its own float64 probability vector and argmax decision. The evaluation
  considers each output alone and the two outputs aligned to the same record (the linked pair).
- SEX is the dataset's recorded demographic column. Readers are separate evaluation classifiers that estimate SEX from the
  released probability vectors. SEX is never a runtime input or an output; it may be a training-time label on MODEL_FIT.
- References U0 and Ucal: the replication's neural task model and its CALIBRATION-temperature version (frozen outputs).
- The retired candidate `B3CTL_UNCOUPLED_MIX50_C6A` and all its verdicts are preserved unchanged. It is a diagnostic baseline
  (RETIRED_B3CTL), not a competitive nominee.

A new mechanism may train models, representations, output tables and calibrators within its authorized roles. It must compute
releases from the six permitted runtime fields.

## 2. Boundaries and resources

| Item | Limit |
|---|---|
| Elapsed time | 6 hours from 2026-10-18T01:51:35Z (first action); hard stop 07:51:35Z |
| CPU | 10 CPU-hours in total, counting failed and stopped units, verification, reader fits and agent work (upper estimates), through `ihs.jobqueue` |
| Concurrency | at most 2 heavy numerical jobs at once (two queue slots), one numerical thread per job, one shared queue |
| New configurations | at most 36 new seed-specific release or task-model configurations, including newly fitted controls (`EXPERIMENT_LEDGER.csv` `counts_toward_36`); individual model fits reported separately (`n_model_fits`); reader fits reported separately (`reader_fits`) |
| Families | at most 3 materially different mechanism families; each with an initial version and at most 2 justified revisions |
| Compute | no paid or cloud compute; no new population; no new microdata download |

Branch `research/pcrl-income-headroom-successor-v1` in its own worktree, based on the validation tip `0be8c6a27`. Main,
manuscript branches, historical locks, protocols and published results are not modified. Person-level arrays, fitted models,
reader artifacts and private paths stay under `.private/` and are never published. Interpreters (the queue ledger records
the interpreter of every unit):
- development evaluation (`ihs.evaluate`, `ihs.devrun`, the reference units): `python3`, Homebrew Python 3.14.3 with numpy
  2.4.4, scikit-learn 1.8.0, scipy 1.17.1, fingerprinted in `DEV_EVALUATOR_FINGERPRINT.json` (with the SHA-256 of the reader
  code); the grid refuses to run under any other interpreter or reader code (reviewer request S2);
- final readers and the final DEV_REPORT evaluation (only for a promoted finalist): the project `.venv` Python 3.13.7 with
  numpy 2.4.2, scikit-learn 1.8.0, scipy 1.17.1, which are exactly the versions of the validation's READER_BANK_LOCK (the
  validation ran its readers under that interpreter);
- release fits: `python3`, or the `.venv` interpreter when torch is needed; each release records its interpreter.

## 3. Data roles and access (enforced by `ihs.common.load_role`)

| Role | Fields | Use in this campaign |
|---|---|---|
| MODEL_FIT (20,000 people, 13,337 households) | features, raw inputs, income, occupation, SEX | fitting release mechanisms and task models; household-separated internal folds for diagnosis. A teacher-trained record is in-sample for the frozen teacher routing: an internal fold of MODEL_FIT is never described as unseen calibration or guard data |
| CALIBRATION (2,982 / 1,974) | features, income, occupation | calibrators only (or a household-separated internal split of a calibrator); no SEX |
| MODEL_SELECT (8,001 / 5,306) | features, income, occupation, SEX | development evaluation, the reader-proxy grid, adaptive selection and revision feedback. Every label read is logged (`MODEL_SELECT_USE.jsonl`). Never gradient-fitting labels, never calibration labels, never used to fit release parameters |
| ATTACK_FIT (8,001 / 5,319), ATTACK_VALIDATION (3,000 / 1,974) | features, SEX | final readers only, and only after this campaign's committed FINALIST_LOCK (the loader refuses earlier). Never used to train or select a release |
| DEV_REPORT (5,943 / 3,955) | features, raw inputs (label-free) | deployment parity. Truth only through `ihs.common.dev_report_truth` after this campaign's committed and published REPORTING_LOCK, under two NEW follow-up events (one coordinator evaluation, one independent-verifier replay). Closed during exploration. If no finalist is promoted, DEV_REPORT labels are not read |
| Sealed assessment | none | closed; every loader refuses it |

Internal cross-fitting within MODEL_FIT and new splits of used people are development diagnostics, never fresh confirmation.

## 4. Acceptance requirements (fixed; never relaxed)

- **U (utility, 24 clauses).** Both tasks, both references (U0, Ucal), both seeds: accuracy loss <= 0.010, NLL excess <= 0.010
  nats, summed Brier excess <= 0.005.
- **G (subgroup guard).** Within-SEX occupation summed-Brier excess against U0 <= 0.005 for each SEX group and seed.
- **R (recipient non-worsening, binding source margin).** For each eligible reference (NEURAL_RAW, H_LJ) and seed: the
  candidate's single-output recovery minus the reference's <= 0.010 AUC, for the income output and for the occupation output.
- **P (pair benefit).** Reference pair recovery minus candidate pair recovery > 0. The original 0.020 material-improvement
  target is reported separately.
- Point feasibility is distinguished from simultaneous confidence. The 0.001-nat matching requirement of an earlier campaign
  is not used. No equivalence tolerance is introduced or widened after outcomes are seen.

## 5. Inherited constraints, checked once before any fit (brief section 4)

`INHERITED_OCCUPATION_MARGINS.json` (unit IHS_INHERITED_V1, from existing artifacts only: the validation's published DEV_REPORT
endpoints and this campaign's pre-registration MODEL_SELECT reference evaluation) and the decision recorded in §8.

The accepted JOINT_TREE occupation output (H_LJ's, released bitwise by the retired candidate) binds on its own:

| Condition (JOINT_TREE occupation output) | MODEL_SELECT value (SE; margin in SE) | DEV_REPORT value (SE; margin in SE) |
|---|---|---|
| men's within-SEX occupation Brier excess vs U0, seed 0 | 0.00442 (0.00181; 0.32) | 0.00324 (0.00205; 0.86) |
| same, seed 1 | 0.00394 (0.00170; 0.62) | 0.00395 (0.00191; 0.55) |
| occupation NLL excess vs Ucal, seed 0 | 0.00341 (0.00287; 2.30) | 0.00867 (0.00336; 0.40) |
| occupation NLL excess vs U0, seed 0 | 0.00253 (0.00301; 2.48) | 0.00695 (0.00350; 0.87) |

Projected to the roughly 17,510 Florida 2019 households that could test a successor in a separately authorized study (SE x
sqrt(G_measured / 17,510)), the men's guard stays below 2 SE on both seeds from either role's SEs, and occupation NLL (seed 0)
stays below 2 SE from the DEV_REPORT SEs. Changing income cannot improve these conditions.

## 6. Development evaluation (`ihs.evaluate`, `ihs.devrun`; frozen before any new fit)

- Utility: the 12 clauses per seed (scm_bench definitions) with household-clustered paired SEs (5,306 MODEL_SELECT households),
  margins in metric and SE units; the two within-SEX guards with linearized clustered SEs; per-SEX task diagnostics.
- Reader proxies: the fixed grid of two household-split repetitions of MODEL_SELECT (A: the source campaign's halves; B: new
  halves, tag `IHS1_dev_proxy_rep_B`) x two reader initialization seeds (MODEL_SEED_BASE + 100 + k, k = 0, 1), shared by every
  method and reference. Each reader is fitted on one half (about 4,000 people / 2,650 households; smaller than the final
  readers' 8,001-person ATTACK_FIT) and scored on the other, in both directions. Slate: OUT_LR, OUT_HGB, LOG_MLP (source
  constructors), ET25, HGB31, CATI1 (validation recipes), LOOKUP. Views: income, occupation, pair; the pair maximum includes
  the single-view readers. Readers see only released probability vectors and decisions. ATTACK_VALIDATION is never used for
  release selection.
- References evaluated on the same grid before any new fit (units IHS_REFERENCES_S0_V1 / S1_V1): U0, Ucal, T_TREE,
  TASK_REFIT, NEURAL_RAW, H_LJ, H_JJ, H_LL, RETIRED_B3CTL, B3_TASKONLY, B2TWIN, PUB_IBSI (proxies for all but U0 and Ucal).
- Reported per configuration: reader variability across the four cells, per-reader values, and near ties.

## 7. Utility-reserve screen

Defined by the baseline and statistical reviewer before any new fit: `roles/statistical_reviewer/UTILITY_RESERVE_RULE.md` and
`RESERVE_RULE.json`, implemented in `ihs/reserve.py` (invented-data tests `ihs/test_reserve.py`). It is a development selection
heuristic on adaptively used data, not a confidence guarantee and not a changed cap.

## 8. Families, mechanism cards and staging

- Up to three materially different families, each with a mechanism card written before its first fit (`MECHANISM_CARDS.md`):
  the exact operation that changes the release; why it could improve the observed trade-off; the measured bottleneck it
  addresses; the simple control that could explain away its benefit; the revise/abandon observation; the closest known method.
- Staging per family: implementation and invented-data correctness checks; a cheap first-seed fit; the full MODEL_SELECT utility
  vector against both references with SEs; income evaluated alongside the frozen occupation output; a matched task-only control;
  the development reader grid on the transmitted outputs; diagnosis; a recorded revision or a move to another family; promising
  configurations on the second source seed.
- Occupation output: initially the accepted H_LJ / JOINT_TREE occupation output, frozen. Any occupation-side or joint-task family
  is recorded here (with its justification from §5) before its first fit, and keeps a fixed-occupation control.
- Every fit, evaluation, failure and decision is a ledger row.

**Family allocation (recorded before any family's first fit, 2026-10-18 about 02:20Z).**

| Family | Investigator | Direction | Configurations allotted |
|---|---|---|---|
| H, headroom-first income learner | investigator H (separate agent) | a stronger income task learner on MODEL_FIT with honest held-out quality, then a privacy step that spends only part of the measured headroom; occupation output frozen at JOINT_TREE | 10 |
| P, pair-aware learned income output | investigator P (separate agent) | the released income probabilities learned directly against the linked-pair threat with the occupation output frozen at JOINT_TREE (optionally an occupation-informed residual design); matched task-only twin | 12 |
| O, occupation-side margin repair (EXPANSION) | investigator O (separate agent) | information-preserving recalibration (and if needed a controlled blend) of the JOINT_TREE occupation output to restore the men's guard and occupation NLL margins without raising occupation-view recovery; measured with H_LJ's own income output so the occupation contribution is isolated | 8 |
| reserve | coordinator | combinations of the best income and occupation outputs, with a fixed-occupation control and a fixed-income control; final matched controls | 6 |

The occupation expansion is justified by §5: the frozen occupation component itself blocks a robust successor. It is a new
family for a successor; it does not revise the retired candidate or reinterpret its result. Any successor that changes the
occupation output keeps a fixed-occupation control (the same income output beside the unmodified JOINT_TREE occupation output),
so the income contribution remains separately measurable, and a fixed-income control, so the occupation contribution does.

## 9. Finalist nomination (MODEL_SELECT and the development record only)

`roles/statistical_reviewer/NOMINATION_RULE.md`, frozen before any new fit. At most one finalist. If none qualifies, no finalist
is frozen, DEV_REPORT is not read, and the campaign delivers the measured frontier and the best diagnostic packages.

## 10. Final evaluation (only for a promoted finalist)

- Frozen before any final reader fit (`FINALIST_LOCK.json`): package hashes for both source seeds; the selection history and
  every attempted revision; the strongest eligible comparators (NEURAL_RAW, H_LJ); the matched component controls; the exact
  output contract; the final reader recipes and selection rules; endpoint definitions, margins, signs and multiplicity.
- A compact bank of at most 8 effective releases per seed, exact views deduplicated.
- Final readers: the validation's enhanced common slate (`READER_BANK_LOCK.json` of the validation: the source 8 plus ET5,
  ET25, HGB15, HGB31, MLP64, MLP6432, CATI01, CATI1, with single-view alternatives for every pair reader), fitted on ATTACK_FIT
  and selected and oriented on ATTACK_VALIDATION with the original auxiliary budgets; the original bank's winners reported
  separately for historical comparison; real-data planted-SEX and shuffled-label controls; a predefined independent
  winner-refit roster covering the successor and the strongest references.
- A sanitized reporting receipt committed and pushed before one logged coordinator DEV_REPORT evaluation, followed by one
  separately logged independent-verifier replay. After that evaluation the campaign does not return to development.
- Inference: household bootstrap (B = 2048) with Wald intervals, conditional on the fitted releases and readers. Intervals do
  not cover the adaptive search. The two seeds are the same people.

## 11. Claims

Written before nomination with generic method roles (`roles/statistical_reviewer/CLAIMS_DRAFT.md`). Novelty, mechanism
attribution, joint-design advantage and competitiveness are separate conclusions. Reader results are evidence against the
declared readers and auxiliary budget only. Deterministic outputs are not assumed injective; observed mostly-unique vectors do
not show population information equality; counterexamples to a global output mapping do not quantify extra SEX information.

## 12. Feasibility, verification, custody

Feasibility reassessment from existing development residuals only (no new population). Independent verification with separate
code. Deployment from the six raw fields through package APIs and wrappers, and from backup alone. Same-device backup;
off-device custody only if a writable drive is mounted (never forced); warm read-back is not described as cold.

## 13. Roles

Coordinator and integration owner (main session); up to three mechanism investigators (separate agents, in waves); baseline and
statistical reviewer; independent verifier with separate code; engineering and claims reviewer. Assignments: `roles/` and
`HANDOFF.json`.
