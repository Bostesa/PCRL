# Agent context: campaign `pcrl_finite_solver_adequacy_v1` (read this first)

Coordinator's shared brief for every agent of this campaign. Written 2026-10-21 from 02:27Z (`date -u`).

## The campaign

- **Question.** Is the propensity-routed finite-cell optimizer (`nrm/fa/prc.py`, imported unchanged by `frc/fa/route.py`) adequate
  for its own stated objective? If a prospectively defined advancement condition is met, a development pilot follows.
- **Repository.** Worktree `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1`, branch
  `research/pcrl-finite-solver-adequacy-v1`. Campaign start 02:22:58Z.
- **Code and records.** New campaign code lives in package `fsa/`, a copy of `frc/` with this campaign's constants. Records live in
  `results/pcrl_finite_solver_adequacy_v1/`. Private arrays live under `.private/pcrl_finite_solver_adequacy_v1/` and are never
  committed.
- **Sources (read-only, immutable).** The previous campaign `pcrl_finite_routing_comparison_v1` (code `frc/`, records
  `results/pcrl_finite_routing_comparison_v1/`, publication 04ac9161f) and its source `pcrl_new_release_mechanisms_v1` (code `nrm/`,
  publication 682068d52). Never edit `frc/`, `nrm/` or anything under their `results/` or `.private/` folders.

## Three questions, kept separate

- **A.** Does the optimizer find good solutions to its stated objective?
- **B.** Does that objective improve actual task usefulness and attribute recovery?
- **C.** Does person-level propensity add anything beyond an equally optimized HQ (task-score-only routing) control?

## Established findings to preserve

These come from 04ac9161f and must not be relitigated:
- teacher-mean (B1) tables repaired avoidable task error on fixed partitions;
- the repaired finite twin TWB_C64 is useful (D1);
- protected routing PRC_S2 (strength 8, one partition per version) has pair AUC 0.6485, against 0.6489 for HQ_S2 and 0.6555 for the
  twin; it showed no benefit beyond matched controls and stays above the strong references;
- the frozen validity rule gives INCOMPLETE, and ENGINEERING_REPAIR holds only under post hoc ER1. Both readings stand.

## Optimizer facts to check

They are documented in card `results/pcrl_new_release_mechanisms_v1/cards/FA_PRC_v1.md` section 1, which is the specification.
- `lloyd` tries seven routing tilts (lambda, lambda/2, ..., lambda/64) and accepts the first proposal that improves the exact
  objective. Raising lambda raises the smallest tilt.
- Routing uses cell SEX-rate tables shrunk with 20 pseudo-counts, while acceptance uses the unshrunk pi-weighted plug-in
  information.
- Codewords are refreshed as cell means of the teacher posteriors, although the objective's per-SEX guard also depends on the
  occupation codewords.
- The emitted B1 tables (out-of-fold teacher means of the final cells) can differ from the codewords used during the search.

These are heuristic limitations to investigate. None is established as a bug.

## Hard constraints, for every agent

- **Data.** Only the admitted Florida ACS 2019 cohort, roles MODEL_FIT, CALIBRATION and MODEL_SELECT, through `fsa.common`
  (`features` is label-free; `load` is logged and gated).
  - ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT labels, the sealed assessment and capped-out people stay closed.
  - No new population or download.
  - No label is read before the registration receipt is verified on GitHub (`fsa.common.registered()`).
  - MODEL_SELECT is never used to fit release parameters.
- **Runtime release.** The release takes six fields (AGEP, WKHP, COW, SCHL, MAR, POBP). Each recipient receives only its own
  probability vector and its argmax decision. Nothing else is transmitted (no cell or row identifiers).
- **Compute.**
  - Every numerical job runs through the queue: `python3 -m fsa.jobqueue run --unit FSA_<ROLE>_<WHAT>_V<n> --stage <stage>
    --rss-gib <GiB> -- <command>`. Two heavy jobs at once, one thread each, zero paid compute.
  - Unit ids are append-only: a rerun uses a new `_V<n>`.
  - Charge out-of-queue numerical work (selftests, scratch checks) with
    `python3 -m fsa.jobqueue charge --label <role> --cpu-s <upper estimate> --note "<what>"`.
- **Agents.** Do not create sub-agents or helpers.
- **Files and commits.**
  - Commit only files in your own role folder (`results/pcrl_finite_solver_adequacy_v1/roles/<role>/`, including its `code/`) unless
    the coordinator assigns another path.
  - Commit messages are plain: no trailers or attribution lines. Never push.
  - Records are append-only. An amendment is a new dated file, never an edit of a committed rule or card.
- **Reporting.**
  - Take every reported time from `date -u`, and never round one ahead.
  - Report what you ran, what failed and what you did not check.
  - Do not describe cached label-derived statistics as untouched data.
