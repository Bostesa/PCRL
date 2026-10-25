# Quickstart: direct-recovery frontier campaign (`pcrl_direct_recovery_frontier_v1`, DRF1)

Commands run from the repository root of the research worktree (branch `research/pcrl-direct-recovery-frontier-v1`). Every
real-data command goes through the shared queue (`python3 -m drf.jobqueue run ...` or `python3 -m drf.lane LANE.json`), which sets one
numerical thread per job, logs CPU and refuses work outside the time window. Private arrays and models live under
`.private/pcrl_direct_recovery_frontier_v1/` and are never published. Interpreters: `python3` (3.14.3, numpy 2.4.4; fits,
evaluation, deployment) and the torch venv `.venv/bin/python` at the root of the main checkout (3.13.7; the reader grid and its custody
readers, as in every earlier grid).

## 1. Read the results (label-free)

```
python3 -m drf.drtables write        # TABLES.md / TABLES.json from the saved records
python3 -m drf.frontier write        # FRONTIER.json / FRONTIER.csv (machine-readable frontier)
python3 -m drf.frontier endpoints    # ENDPOINTS.json (per-reading labels and outcomes)
```

## 2. Selftests (invented data only)

```
python3 -m drf.finite_law selftest   # exact optimal-reader AUC vs enumeration (investigator A)
python3 -m drf.tiny run              # tiny exact instances: solver vs exhaustive constrained optimum
python3 -m drf.solver                # objective, constraints, incremental moves, merges, expected-cost estimator
python3 -m drf.dr selftest           # family: nested estimator, brackets, emission
python3 -m drf.test_rules            # the frozen decision rules (statistical reviewer)
```

## 3. Use a deployment package (six raw fields in, each recipient's probability vector and decision out)

```
python3 -c "
from drf import deploy as D
rows = [{'AGEP': '41', 'WKHP': '40', 'COW': '1', 'SCHL': '21', 'MAR': '1', 'POBP': '012'}]   # a synthetic row in the admitted string coding
print(D.release('.private/pcrl_direct_recovery_frontier_v1/deploy/DR_AUC_R1_V0', rows))
"
python3 -m drf.deploy check --packages .private/pcrl_direct_recovery_frontier_v1/deploy/DR_AUC_R1_V0 --report /tmp/check.json
```

A package refuses SEX, task targets, identifiers, extra or missing fields and any output other than the probability vector and the
argmax decision; it binds every code module and model file its deployment function reads (a changed module makes it stale: rebuild
with `python3 -m drf.deploy store --name DR/<config> --version v` then `build`). The release is deterministic: per-request keys are
refused. The value coding of the raw fields is the admitted encoder's (see the package's PACKAGE.json `encoder`).

## 4. Re-check custody

```
python3 -m drf.custody backup <parent folder>                      # same-device backup + restore check from the backup alone
python3 -m drf.custody offdevice <backup folder> <existing folder on an attached drive>   # never mounts or forces anything
```

A check right after copying to a still-mounted drive is WARM. For a COLD check: eject the drive normally, re-attach it, then run
`python3 -c "from pathlib import Path; from drf import custody as K; print(K.restore_check(Path('<copy on the drive>'), 'COLD: after an ordinary remount'))"`.

## 5. Re-run a stage (only inside the queue; the campaign's queue is closed at closeout)

Lanes are in `results/pcrl_direct_recovery_frontier_v1/lanes/` (FIT_*, OPT_*, EVAL_*, GRID_*, CLAIMS_*, DEPLOY, READER_CUSTODY). A
closed queue refuses every new unit; a re-run needs a new campaign queue.
