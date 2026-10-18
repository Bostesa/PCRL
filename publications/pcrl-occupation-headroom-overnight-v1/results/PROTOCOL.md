# PCRL overnight occupation-headroom and robust-release campaign: registration

**Status.** Registered before any new task-model, release or reader fit of this campaign. The registration commit adds this file,
`DEV_GRID.json`, `EXPOSURE_TABLE.md`, `SOURCE_ADMISSION.json` and the statistical reviewer's `roles/statistical_reviewer/`
(`RULES.json`, `SELECTION_RULE.md`, `CLAIMS_DRAFT.md`) with `ohn/rules.py`. Label reads are refused by `ohn.common.load_role`
until the registration is committed AND its sanitized public receipt has been verified on the GitHub remote with `git ls-remote`
(`ohn.receipt verify`), so no fit can precede the public receipt. Mechanism designs and revisions are exploratory decisions made
inside these boundaries and recorded when they occur (mechanism cards, `EXPERIMENT_LEDGER.csv`); they are not individually
pre-registered. This is development on previously used records, not confirmation.

**Disclosure.** The design was informed by the published lineage on these same records, in particular the source campaign
(income-headroom successor, research tip `2af3c9db2`, public package `89fd2f6e6`) and its DEV_REPORT result. DEV_REPORT has been
read 14 times in the lineage. MODEL_SELECT has been used adaptively by every earlier campaign. No `AGENTS.md` exists in this
repository (checked 04:36Z); the only occupation-named branch (`research/pcrl-occupation-bottleneck-v1`) belongs to the earlier
Adult lineage and is not resumed or reused. The old claim that one observed output globally determines the other was disproved
in the lineage and is not revived.

## 1. Question, data and service (unchanged definitions)

Can a stronger occupation predictor create enough task-quality margin to support a useful reduction in SEX recovery from both
released outputs, while preserving the income improvement of the source component H3 (`H3_STRONG_LJDIR_PB`)? Better occupation
utility is not treated as evidence that recovery falls, and combining two useful components is not assumed to add their benefits;
both are tested.

- Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida: housing-unit residents aged 25-64, civilian
  employed; the replication's cohort, household grouping and role assignment (`SOURCE_ADMISSION.json` pins every admitted file by
  hash against the source's committed records; 201 copies, 0 problems).
- Tasks (verified in the replication's POPULATION_AND_SCHEMA.md and OCCUPATION_MAPPING.json): income class (2: PINCP x ADJINC above
  $50,000 x 10^6 in exact integer arithmetic) and occupation group (FIVE official broad civilian groups: management/business/science/
  arts; service; sales/office; natural resources/construction/maintenance; production/transportation/material moving). No
  six-class definition is imported.
- Runtime inputs: AGEP, WKHP, COW, SCHL, MAR, POBP only (the admitted 145-column encoding, or a deterministic re-coding of it).
- Output contract: each recipient receives only its own float64 probability vector and its argmax decision. The evaluation asks
  what readers can estimate from each actual output separately and when the two outputs are aligned to the same person. Linkage,
  auxiliary labelled data (the original budgets) and one release per person are explicit assumptions: a standard recipient-access
  adversary, not a new threat model.
- SEX is the dataset's recorded demographic column; it may be a training-time label on MODEL_FIT and an evaluation label; it is
  never a runtime input or output.
- Utility references U0 (the replication's neural task model) and Ucal (U0 with a CALIBRATION temperature), frozen.
- Exposure of every frozen component: `EXPOSURE_TABLE.md`.

## 2. Boundaries and resources

| Item | Limit |
|---|---|
| Elapsed time | research window 8 hours from 2026-10-18T04:36:11Z (first action): ends 12:36:11Z; closeout allowance 30 minutes (ends 13:06:11Z); enforced by `ohn.jobqueue` (after 12:36Z only custody / closeout / publication / deployment-check stages) |
| CPU | 14 CPU-hours in total, including failed attempts, retries, refits, reader fits, reviewers' numerical work and agents' out-of-queue upper estimates (`ohn.jobqueue charge`) |
| Concurrency | one shared queue; at most two heavy numerical jobs at once; one numerical thread each; registered before launch; elapsed time, CPU, memory, failures and cache hits recorded separately |
| Configurations | at most 80 new seed-specific fitted release configurations: <= 36 occupation task-model (OCC36; 12 per family), <= 24 privacy-stage (PRIV24), <= 8 conditional income branch (INC8), <= 12 matched-control or stability (CTRL12). Ceilings, not quotas. Cross-fitting submodels, calibrators and readers are counted as individual fits; copied or combined releases are not new fits |
| Families | three occupation-learning families (A stronger categorical boosting; B complementary models + household-cross-fitted stacking; C residual task learning), each an initial design and at most two justified revisions, at most two predefined capacity settings per revision |
| Compute and data | no paid or cloud compute; no new population, microdata download, sealed assessment, capped-out population or previously unauthorized labels; public literature and software documentation may be read |
| Agents | native agents in waves, at most six live including the coordinator; more agents never authorize more heavy processes |

Branch `research/pcrl-occupation-headroom-overnight-v1` in its own worktree, based on the source research tip `2af3c9db2`. Main,
manuscripts, source branches, historical locks and published conclusions are not modified. Person-level arrays, fitted models,
reader artifacts and private paths stay under `.private/` and are never published. Interpreters: `python3` (Homebrew 3.14.3,
numpy 2.4.4, scikit-learn 1.8.0) and the project `.venv` (3.13.7, numpy 2.4.2, scikit-learn 1.8.0, torch 2.10.0, xgboost 3.2.0;
the validated reader versions); every release and reader record states its interpreter.

## 3. Data roles (enforced by `ohn.common.load_role`)

| Role | Use in this campaign |
|---|---|
| MODEL_FIT | fitting release parameters and task models; household-separated internal folds; the household cross-fitting checks (`ohn.common.crossfit_folds`, 5 folds, tag OHN1_crossfit). SEX only as a training-time label |
| CALIBRATION | calibrators only (or household-separated internal splits of a calibrator); no SEX |
| MODEL_SELECT | logged development selection and diagnosis (`ohn.devrun score`) and the development reader grid (`ohn.devreaders`: household-disjoint FIT / VALIDATION / SCORE reader roles per repetition, `DEV_GRID.json`). Never a gradient-fitting or calibration set |
| ATTACK_FIT, ATTACK_VALIDATION | final readers only, after this campaign's committed FINALIST_LOCK (fitting on the 8,001 / selection on the 3,000 people recorded in the manifests); never used to train, select, revise or replace a release |
| DEV_REPORT | label-free deployment parity; truth only through `ohn.common.dev_report_truth` after the committed REPORTING_LOCK and its verified public receipt, under two NEW events (one coordinator evaluation, one separately logged verifier replay); closed unless a finalist is promoted |
| Sealed assessment, capped-out records | closed; every loader refuses them |

No release parameter is fitted on ATTACK_FIT, ATTACK_VALIDATION or DEV_REPORT labels. A rotated split of previously used people
is never described as fresh confirmation.

## 4. Acceptance limits (original; never relaxed) and the development reserve

- U: both tasks x U0 and Ucal x both seeds = 24 clauses: accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed multiclass
  Brier excess <= 0.005.
- G: the men's and the women's within-SEX occupation summed-Brier excess against U0, each <= 0.005, per seed.
- R: recipient non-worsening for each required reference and seed: candidate single-output recovery minus the reference's
  <= 0.010 AUC (income view, occupation view).
- P: positive pair benefit; the original 0.020 material-improvement target is reported separately.
- The development reserve (at least two paired household standard errors of slack below each relevant cap, with explicit
  zero / tiny-SE handling), the whole-release ranking, the witness rule, the stability screen, the income-branch trigger, the
  comparator set, the nomination rule and the generic claims are the statistical reviewer's (`roles/statistical_reviewer/
  RULES.json`, `SELECTION_RULE.md`, `CLAIMS_DRAFT.md`, implemented in `ohn/rules.py`), frozen in this registration. The reserve is a
  development heuristic on adaptively used data: it changes no cap and proves no population guarantee. Point eligibility,
  reserve, conditional confidence, recipient protection, reader stability and material benefit are recorded as distinct outcomes.

## 5. Staging

1. **Admission, exposure, rules, receipt** (this registration).
2. **Occupation task learning (families A, B, C; task-only, no SEX).** Each family writes a mechanism card before its first fit
   (operation and deployed output; the measured failure addressed; why it might create occupation headroom; closest established
   method; matched task-only and calibration controls; primary correctness check; revise / replace / abandon triggers). Fit on
   MODEL_FIT, calibrators on CALIBRATION or internal folds, cheap first-seed fits, diagnosis, at most two justified revisions,
   promising designs on the second seed. Every task-model configuration produces the household cross-fitting check
   (cross-fitted MODEL_FIT predictions and the fold-average on MODEL_SELECT), and the scorer reports accuracy, NLL and summed
   Brier against U0 and Ucal, both SEX-group guards, per-class decompositions, comparisons with the equally exposed task-only
   references (T_TREE, TASK_REFIT), cross-fitted utility and the cross-fitted-versus-deployed difference. A stronger aggregate
   NLL cannot compensate for a failed subgroup guard.
3. **Task-only witnesses.** At most two, selected on utility evidence alone by the frozen rule (original point feasibility on
   both seeds; genuine reserve on the occupation constraints; the full constraint vector reported). Without a qualifying witness,
   the fallback programme (section 9) runs and no recovery-penalty bank is trained on task models that fail usefulness.
4. **Matched protection experiment (privacy stage, PRIV24).** For each qualifying witness, with the task learner and output
   contract matched: the unchanged witness; task-only continuation with the same additional training effort; a local occupation
   recovery penalty; a linked-pair recovery penalty with H3's income frozen (a coalition-aware occupation correction, not a
   two-output joint design); the simplest appropriate compression or calibration control; and the strongest eligible frozen
   neural, tree and code references. The expanded reader bank is the development standard; training-critic strength is recorded;
   fresh readers are fitted on the frozen transmitted outputs. A bounded initial attempt and at most two evidence-driven
   revisions. The old controller, temperature or strength lines are not relaunched without a stated new fact. The full utility
   frontier is preserved with uncertainty in cost differences; the earlier 0.001-nat exact-match rule is not used.
5. **Conditional income branch (INC8).** Only when the frozen trigger (`RULES.json`) fires: a promising occupation witness exists
   but income accuracy prevents a complete candidate from meeting the predeclared reserve. Permitted: one bounded income-task repair
   (a justified task-model correction or ensemble addressing task quality or decision errors while preserving Brier / NLL headroom),
   with a matched task-only twin, the probability-plus-argmax contract, every parameter fitted on MODEL_FIT / CALIBRATION, and the
   fixed-H3 comparisons kept so the occupation contribution stays identifiable. Not permitted: another sweep over H3's privacy
   strength, six-level bins or mixture weights. If it fails, the occupation witness is packaged as a component; the reserve is
   never relaxed.
6. **Reader robustness and selection.** The development reader grid (`DEV_GRID.json`): two fixed household-split repetitions of
   MODEL_SELECT, each with household-disjoint FIT (50% of households) / VALIDATION (20%) / SCORE (30%) reader roles, x two reader
   seeds, shared by all methods; readers = the validated enhanced slate (source 8 + ET5, ET25, HGB15, HGB31, MLP64, MLP6432,
   CATI01, CATI1) with ignore-income / ignore-occupation alternatives for the pair; comparator readers refitted independently in
   every cell; recipes and winners chosen on VALIDATION, every benefit, recipient condition, stability and learning curve
   computed on SCORE; validation near ties and winner changes recorded; paired benefit distributions with household-bootstrap SEs
   on SCORE; nested learning curves at 1/4, 1/2 and the full development FIT size (diagnostics only). The reader-grid code is
   frozen (hash in a dated lock) before the first candidate is scored with readers. People and reader fits are counted separately.
   Selection on the complete release by the frozen rule: at most one whole-release finalist; a control-labelled configuration stays
   eligible; final ATTACK_VALIDATION outcomes never guide release revisions or nomination.
7. **One frozen final evaluation, only if a finalist is promoted.** Before any final reader fit: candidate and comparator package
   hashes, actual outputs and release schemas, the full selection history, reader slate, budgets, deterministic tie rules, refit
   roster, endpoints, signs, reference choices, multiplicity and generic claims (0.020 separately from positive benefit). A
   compact bank (normally <= 8 effective releases per seed, exact aliases deduplicated transparently). Final readers fitted on
   ATTACK_FIT and selected on ATTACK_VALIDATION with declared positive and null controls; these outcomes cannot trigger a revision
   or a substitute nominee. A reporting receipt is published and verified on the GitHub remote before one logged coordinator
   DEV_REPORT evaluation and one separately logged verifier replay. No second nominee, alternate checkpoint search or repair
   follows. Household-grouped conditional intervals with reader-refit sensitivity beside them; they do not cover the adaptive
   development history.

## 6. Agents (waves; at most six live including the coordinator)

Coordinator and integration owner (main session); investigators A, B, C (one occupation family each); statistical and baseline
reviewer; closest-work reviewer (primary sources, evidence table with links); then, in later waves, a protection investigator, a
reader-robustness investigator, an independent verifier with separate numerical code (starting as artifacts become available),
an engineering / deployment reviewer and a claims reviewer. Roles may be combined. Subagent report files may be refused by the
platform; reports are then returned as text and saved by the coordinator with a disclosure.

## 7. Verification, deployment and custody

An independent verifier with separate code checks source and household-role integrity, exposure, utility and subgroup metrics,
reserve calculations and selection, actual transmitted outputs, development stability summaries and any final reader comparisons,
household-aware intervals and the exact claim conjunctions, representative fresh reader refits (including strong controls) and
raw-field deployment from restored artifacts; numerical PASS is separated from documentation review. Packages: the strongest
occupation witness, any selected whole release, the unmodified anchor and the decisive matched control, with raw-field wrappers that
reject missing, extra, reordered, target and SEX fields at the documented entry point and the underlying deployment function, no
hidden or intermediate outputs, and recorded CPU, dependencies and source hashes. Backup with the existing framework and a restore
check from the backup alone; off-device custody only on an already attached writable drive (never forced), otherwise a verified
same-device snapshot with the exact pending command.

## 8. Publication

Remote: `origin` = https://github.com/Bostesa/PCRL.git (verified 04:36Z; not a local mirror). Sanitized receipts and the final
package go to the dedicated branch `publication/pcrl-occupation-headroom-overnight-v1` (git plumbing; no working tree switched; no
history rewritten; main not merged; earlier branches untouched). The research branch stays local. Remote SHAs and public entry
points are checked. If network access prevents a required receipt, the prepared work is preserved and the dependent stage waits;
nothing is backdated. Only file-based handoffs go to the manuscript and claims owners.

## 9. Fallback programme (if no witness or nominee qualifies)

Compare the three task learners' capacity, exposure and calibration behaviour at matched training access; decompose occupation NLL
and Brier costs by class and SEX group with paired household uncertainty; identify whether error comes from teacher quality,
fitting / calibration transfer, runtime ensemble mismatch or recovery pressure; check whether a simple task-only predictor already
explains an apparent recovery change; characterize reader sample-size dependence for a small frozen development bank (never using
final attack-validation data for release selection); replay failed utility and stability gates with independent metric code; build
an occupation task-only witness package if one exists; write a concrete failed-mechanism explanation with its closest existing
method. Exhausted lines are stopped honestly; no impossibility result is claimed from this finite search.
