# Brief for the protection investigator (privacy stage; coordinator, written before any witness exists)

Read `AGENT_BRIEF_COMMON.md` first. This stage uses the occupation headroom of a task-only WITNESS (selected by the frozen rule in
`roles/statistical_reviewer/RULES.json`; the coordinator names it) in a MATCHED protection experiment. Only the occupation output
changes; the income output is H3's, frozen. The pair term is therefore a coalition-aware occupation correction (a critic that sees
the linked pair while only the occupation output is trained), never a two-output joint design.

## Arms (task learner and output contract matched; one configuration id per arm, seed and revision)

1. `W` the unchanged task-only witness (already fitted; no new configuration).
2. `CONT` task-only continuation: the identical correction model, optimiser, epochs, batches and initialisation as the penalty arms,
   with the recovery term switched off (the matched twin; separates protection from ordinary task continuation). CTRL12.
3. `LOC` local occupation recovery penalty: a correction trained against online SEX critics that see the released occupation vector
   and decision only. PRIV24.
4. `PAIR` linked-pair recovery penalty with H3's income frozen: critics see (H3 income vector, occupation vector) and both single
   views. PRIV24.
5. `COMP` the simplest appropriate compression or calibration control at a matched (development-measured) cost: e.g. shrinkage of
   the witness's probabilities toward the class prior or a temperature, or a coarse code of the occupation vector, with any
   parameter fitted label-free or on CALIBRATION. CTRL12.
6. Strongest eligible frozen neural, tree and code references (NEURAL_RAW, H_LJ, B2TWIN / RETIRED_B3CTL) and the matched-income
   source successor SUCC (H3 + JOINT_TREE occupation): no fit; they enter the reader grid as comparators.

## Requirements

- Fit every release parameter on MODEL_FIT (SEX only as a training-time label); calibrators on CALIBRATION or household-separated
  internal folds of MODEL_FIT. MODEL_SELECT only through the scorer and the reader grid. Never ATTACK / DEV_REPORT labels.
- Exposure: the witness's deployed MODEL_FIT outputs are in-sample for the witness; its cross-fitted (out-of-fold) MODEL_FIT outputs
  exist in `CROSSFIT.npz`. H3's income output on MODEL_FIT is in-sample for H3 (fitted on MODEL_FIT and CALIBRATION). State which
  base the correction is trained against (deployed in-sample vs cross-fitted), why, and check the transfer to MODEL_SELECT.
- Explain exactly what changes the transmitted probabilities, and what evidence would distinguish protection from ordinary task
  continuation (the CONT twin) and from ordinary compression (COMP). Record training-critic strength, critic recovery during
  training, and the task cost on MODEL_FIT (in-sample) and on internal folds.
- The utility reserve is whole-release and includes BOTH SEX-group guards: the inherited JOINT_TREE output failed mostly on the
  men's guard (it worsened men's occupation Brier by about 0.0055 relative to the teacher). If a guard binds, a guard-aware
  constraint is a legitimate design element; say what it is before fitting.
- Budget: a bounded initial attempt (seed 0 first, then seed 1 for a promising arm) and at most two evidence-driven revisions; do
  not relaunch the old controller, temperature or strength lines without stating the new fact that makes the attempt different.
  Keep the full utility frontier (every arm and strength), with uncertainty in cost differences; no exact-match rule.
- Code: `ohn/protect.py` (selftest on invented data; fit; predict; verify bitwise on all six roles); releases through
  `ohn.devrun.save_occupation(family='PROT', ...)` with kind 'privacy' or 'control'; scoring through `ohn.devrun score` with
  `--kind privacy|control --budget-class PRIV24|CTRL12`; the reader grid is run by the coordinator (bank spec) unless delegated.
