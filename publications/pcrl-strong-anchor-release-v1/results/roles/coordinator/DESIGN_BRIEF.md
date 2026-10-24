# Design brief for the two mechanism investigators (coordinator; 2026-10-24, from 04:20Z)

Campaign `pcrl_strong_anchor_release_v1`, package `sar/`, branch `research/pcrl-strong-anchor-release-v1`, worktree
`<repository>/.claude/worktrees/pcrl-strong-anchor-release-v1` (all paths below are relative to it).

This is statistical privacy research on an existing census benchmark (public-use ACS 2019 PUMS, Florida). SEX is the dataset's
recorded demographic column. A service predicts an income class (2) and an occupation group (5) from six fields; readers try to
infer SEX from what the recipients receive. The goal is a release that keeps the original usefulness and recipient conditions and
lowers that inference, especially for the linked pair, beyond its own start, matched controls and the strong references.

## 1. What changes in this campaign

Recent campaigns started from task-only services (their twin CTL_TWIN: pair AUC 0.648) that were far weaker than the strong frozen
releases (RETIRED_B3CTL 0.596, B2TWIN 0.606, SUCC 0.607, H_LJ 0.619, NEURAL_RAW 0.625) and improved those weak starts by 0.002-0.024
without ever becoming competitive. This campaign starts from the strong releases themselves. Read first:
`results/pcrl_strong_anchor_release_v1/anchors/ANCHOR_SELECTION.md` (frozen) and `ANCHOR_TABLE.md` (one reader bank, IPR1_MAIN).

## 2. Your investigation

- **Investigator A: starting release SUCC** (H3 income + JOINT_TREE E20 occupation). Question: can a strong protected release
  recover usefulness margin while retaining its recovery profile (and then turn that margin into lower recovery)? Its binding
  condition is the men's within-SEX occupation Brier guard (+0.0044 / +0.0039 against the 0.005 cap; reserve 0.32 / 0.62 SE), while
  women's has -0.0049 / -0.0030. Against its own task-only routing (TASK_REFIT, the same leaves with lambda 0) the JOINT correction
  costs men +0.0056 / +0.0043 and women +0.0053 / +0.0052 summed Brier, occupation NLL about +0.016 nats, and lowers the occupation
  view 0.648 -> 0.573: the guard binds because the tree base leaves men little slack against U0 (TASK_REFIT men -0.0012, women
  -0.0102). Second fact: the pair 0.607 sits 0.034 above the occupation view; H3 income was never protected jointly with JOINT
  occupation (JOINT_TREE's own income, trained with the pair critic, fails income utility).
- **Investigator B: starting release NEURAL_RAW** (U0 network trained from U0 with the joint objective; rho 1, E2). Question: a
  different construction of the two task outputs that targets recovery directly (above all the linked pair) while keeping useful
  probabilities. It has point margin on every condition (max 0.34 / 0.57 of cap), but mild protection: pair 0.6253 = occupation
  view 0.6253, income 0.5336; U0 pair 0.6383. Stronger points of its own path (rho 3, 10; E5-E20; checkpoints stored) were not
  chosen by the replication rule.

You may redesign both task models, learn a new mapping from the six permitted inputs, or use admitted teachers (U0, T_TREE,
TASK_REFIT, the HGB task teachers, JOINT/LOCAL leaf corrections, H3, NEURAL_RAW checkpoints) as training signals. You need not
post-process frozen score vectors. You need not show novelty or a synthetic gain before a direct exploratory fit.

## 3. Boundaries (unchanged service and data)

- Deployment input = the six fields only (admitted 145-column encoding X of AGEP, WKHP, COW, SCHL, MAR, POBP). Never SEX, a true
  task label, undisclosed recipient information or person-level state that is not available at deployment. Each recipient receives
  only its own probability vector (float64, sums to 1) and its argmax decision; one released instance per person (a randomized
  release draws once per person). No repeated-query, model-access or insider claim.
- Exact tasks, classes, occupation grouping, preprocessing and output semantics of the admitted source. Training exposure: the
  20,000 MODEL_FIT people (CALIBRATION task labels, never SEX, only if your card says so, and then for your controls too).
- MODEL_FIT task and SEX labels may fit task learners and release mechanisms (calibration maps on separated MODEL_FIT household folds,
  or reuse frozen admitted calibrators). MODEL_SELECT is evaluation only (the coordinator's utility and reader units); never a fitting
  input of any parameter, strength, control point or selection rule. ATTACK_*, DEV_REPORT labels, the sealed assessment and the
  capped-out records stay closed. Synthetic fixtures for engineering rehearsals.
- **Until the registration is committed and your card is committed, you work label-free**: you may read every committed record under
  `results/`, the code, the label-free admitted features, and frozen release arrays (they are outputs, not labels). Do not open any
  label file (income, occupation, sex) of any role, and do not compute anything new from MODEL_SELECT. The coordinator runs every
  label-reading fit through the shared queue after your card is committed.

## 4. Closed work (do not rename it; a related idea must show its structural change before it is fitted)

Scalar leaf-score corrections (on any routing; this is how JOINT, LOCAL, the occupation-headroom P1-P3 and Track F corrections were
made), decoder rescue, learned decoders and calibrators, another controller, selective flattening and coordinated keyed collapse (IB_CSF),
another ordinary cell merge or propensity-routed cells and their solvers, input-side field generalization inside the S3 budget
(IA_SFG, IA_OSG), Gaussian and coupled logit channels or a slightly different noise strength, SEX standardization, inverse-propensity
reweighting, linear propensity tilts and release-only tilts as mechanisms (dual-design family), the 6-level income codes of the
autonomous-method-development search, temperature or per-class bias rescue (O1_SEXBAL). The previously declined cross-recipient
swap (FS_SWAP) needs a new, substantive rationale. Evidence pointers: `results/pcrl_input_and_pair_release_v1/RESEARCH_DECISION.md`
and `FAILURE_MAP.md`, `results/pcrl_occupation_headroom_overnight_v1/` (witness base + T_TREE-routed corrections, guards
'equal'/'static'), `results/pcrl_income_headroom_successor_v1/` (H3), `results/pcrl_acs_protected_tree_replication_v1/` and
`results/pcrl_correction_transfer_stability_v1/EXPOSURE_MAP.md` (how JOINT_TREE, LOCAL_TREE, NEURAL_RAW, H3 were made),
`results/pcrl_frozen_finalist_validation_v1/` (B3CTL), `results/pcrl_dual_design_release_v1/`, `results/pcrl_refreshed_guarded_joint_v1/`,
`results/pcrl_joint_complete_view_method_v1/`.

## 5. What you deliver (write only under `results/pcrl_strong_anchor_release_v1/roles/<your role>/` and your family directory)

1. `PROPOSAL.md`: the starting release and the binding condition you address; the precise training or release operation; the closest
   existing implementation and the operational difference; what evidence would contradict its rationale; its simplest informative
   matched control (matched to the mechanism, at comparable task cost; say how cost is matched, e.g. a prospectively specified bracket
   on MODEL_FIT out-of-fold paired NLL); an equally exposed task-only or zero-protection continuation if attribution needs it; the
   bounded choices (strength grid, MODEL_FIT-only selection rule, component budgets chosen from the actual anchor margins, not an
   allowance so strict it guarantees near-identity updates); the measurements; how a failure would be interpreted; registered
   predictions with probabilities; the specification count; CPU estimate per fit.
2. `CARD_DRAFT.md` following `results/pcrl_input_and_pair_release_v1/cards/README.md` (the coordinator commits it as
   `cards/<FAMILY>_<name>_v1.md` before the first fit; once committed it is never edited).
3. The implementation: `sar/fa/<name>.py` (A) or `sar/fb/<name>.py` (B), modelled on `ipr/ia/osg.py`, `ipr/ib/csf.py`,
   `ipr/ctl/twin.py` and using `sar/famkit.py`, `sar/famrun.py`, `sar/release.py`, `sar/common.py` (`load` is the only label path; it
   refuses label reads before registration, so test with invented data). It must save a raw-input deployable model (famkit
   `write_model` / `release_all`), the release on every role from admitted features, the controls named in the card, and a
   `selftest` on invented data that passes. `sar/` is a mechanical rename of `ipr/` (the infrastructure is identical); anything you
   need changed in the shared modules, ask the coordinator in your hand-back instead of editing them.
4. A short hand-back message (what you built, what you predict, what you are unsure of).

## 6. Budget

- Specifications (seed-specific fitted releases, each version counts): 36 for the campaign; your family's initial comparison should
  use at most 6 (the candidate and its simplest matched control in both versions, plus a zero-protection continuation if needed);
  up to two revisions per family later (each with its control), and at least 12 of the 36 stay reserved for controls and follow-through.
- CPU: 12 CPU-hours for the whole campaign including readers (readers take most). A fit should cost minutes, not hours, on one
  thread (set OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS = 1 and torch.set_num_threads(1)). Your own label-free design
  work: at most 600 CPU-s, recorded in your hand-back.
- Interpreters: `python3` (3.14, numpy 2.4.4, scikit-learn 1.8, scipy 1.17; no torch), the torch venv
  `<repository>/.venv/bin/python` (torch 2.10, numpy 2.4.2, scikit-learn 1.8), and CatBoost 1.2.8 at
  `.private/pcrl_correction_transfer_stability_v1/venv_catboost/bin/python`.
- Do not spawn agents, run git, push, or touch anything outside your directories. Do not kill processes.
