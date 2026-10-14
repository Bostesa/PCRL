# PCRL minimal task-wire adapter (exploratory development study)

**Result:** POINT_FEASIBLE_NO_COMPETITIVE_CRITERION. Execution and independent verification are complete (43/43 checks).

A 325-parameter residual in the occupation head's row space, over a frozen RAW_JOINT trunk and with a teacher-function KL ≤ 0.001 anchor, kept all 24 original point utility clauses on FIT2 and INNER in both seeds. Its privacy term verifiably changed the transmitted occupation predictions. Fresh readers recovered SEX from occupation alone 0.010 AUC less than from RAW_PROJ (established), but from the aligned income + occupation pair only 0.005 less: an established shortfall against the required 0.020. The trust constraint and the coalition coupling produced no measurable recovery difference. The earlier guard checkpoints did lower fresh recovery, but they fail utility.

Start with `results/START_HERE.md`, then `results/RESEARCH_DECISION.md`.

- `code/minimal_task_wire_adapter/`: campaign code (adapter, solver, trainer, readers, reporting, deployment, custody).
- `code/minimal_task_wire_adapter_independent/`: the verifier's separately written code.
- `code/inherited/`: unchanged inherited modules used by the campaign.

People-level arrays, checkpoints, reader predictions and private bindings are not published. Locks use `$PRIV` tokens.
