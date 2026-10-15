# Quickstart

All commands run from the research worktree root, with `PYTHONPATH=.`. Private artifacts live under `.private/` and are not in Git.

## Power and feasibility (public inputs only)
```bash
PYTHONPATH=. python -m tree_completion.power \
  --inputs results/pcrl_frozen_tree_control_completion_v1/power/FEASIBILITY_INPUTS.json \
  --out /tmp/power_out --draws 20000
```
- **Inputs:** aggregate per-group influence-function moments from the already-used INNER role, with no person-level data.
- **Outputs:** `JOINT_POWER_TABLES.csv` (N80/N90 and joint power by scenario and design), `MARGINAL_POWER.csv`, `BINDING_CLAUSES.csv`, `PRIVACY_ONLY_AND_BEYOND_CAP.csv` and `POWER_META.json`.
- **Runtime:** a few seconds.

## Confirmation runner
```bash
P=.private/tree_completion/confirmation/CONFIRMATION_PROTOCOL.json
PYTHONPATH=. python -m tree_completion.confirmation_runner --mode dry-run \
  --protocol $P --protocol-sha256 $(shasum -a 256 $P | cut -c1-64)        # verifies every binding; loads no data
PYTHONPATH=. python -m tree_completion.confirmation_runner --mode execute-fresh \
  --protocol $P --protocol-sha256 <sha>                                    # always REFUSED (exit 2) in this build
```
- **Rehearsals:** they run only through the shared queue.
  - `synthetic-rehearsal` uses synthetic fixtures with the real frozen packages.
  - `development-rehearsal` uses attacker = ATT and assessment = INNER (already used).
  - Their specs are in `.private/tree_completion/root/specs_rehearsal.py`, and their outputs in `confirmation/`.
- **What the development rehearsal reproduced:** all 30 original clause statistics within 1e-12 (28 bitwise), including refit readers. RAW_PROJ and H_SEQ21_CODE were rerun from their frozen models and reproduce their stored INNER releases bitwise.

## Frozen candidate deployment (unchanged from the source campaign)
```bash
PYTHONPATH=. python -m task_headroom.deploy --package <private L_TREE_S0 package> \
  --manifest-sha256 <see FROZEN_CANDIDATE_MANIFEST.json> --input X.npz --input-sha256 <sha> --output out.npz
```
