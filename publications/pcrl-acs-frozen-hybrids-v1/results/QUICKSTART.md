# Quickstart

All commands run from the research worktree root with the campaign Python (`~/PCRL/.venv/bin/python`, written `$PY` below). Private artifacts live under `.private/acs_frozen_hybrids/`, which is git-ignored. The source's private artifacts are read from the sibling worktree `pcrl-acs-protected-tree-replication-v1`. Every numerical unit runs through the shared queue (`acs_hybrids.jobqueue`), which enforces 2 slots, 1 thread per job and a 4 GiB RSS cap, and keeps an append-only ledger.

## Pipeline (in order)
```bash
$PY -m acs_hybrids.jobqueue init                                                # campaign start fixed at 2026-10-16T02:10:34Z
J="$PY -m acs_hybrids.jobqueue run --stage <stage> --rss-gib 2.0 --unit <UNIT_ID> --"
$J $PY -m acs_hybrids.restore_admit --out results/pcrl_acs_frozen_hybrids_v1   # restore + admit the frozen source
$J $PY -m acs_hybrids.bank --out results/pcrl_acs_frozen_hybrids_v1            # build + hash the four-combination bank
$J $PY -m acs_hybrids.hybrid_deploy build-verify --out results/pcrl_acs_frozen_hybrids_v1   # DEPLOY_V1, before the lock; packages go to .private/acs_frozen_hybrids/deploy/
$PY -m acs_hybrids.lock analysis --review results/pcrl_acs_frozen_hybrids_v1/STATISTICAL_REVIEW_PRELOCK.md   # then commit
$J $PY -m acs_hybrids.readers --seed 0 --out .private/acs_frozen_hybrids/runs/READERS_0   # needs the committed ANALYSIS_LOCK
$J $PY -m acs_hybrids.readers --seed 1 --out .private/acs_frozen_hybrids/runs/READERS_1
$J $PY -m acs_hybrids.lock reporting        # pre-truth checks + Stage-1 controls; then commit
$J $PY -m acs_hybrids.report --out results/pcrl_acs_frozen_hybrids_v1           # logged DEV_REPORT truth re-read
$J $PY -m acs_hybrids.figure --results results/pcrl_acs_frozen_hybrids_v1/DEV_REPORT_RESULTS.json --out results/pcrl_acs_frozen_hybrids_v1/figures/RECOVERY_VS_UTILITY.png
$PY -m acs_hybrids.closeout all   # FORECAST_SCORES, READER_ACCOUNTING, RESOURCE_INVENTORY
```
- **Write-once outputs.** restore_admit, bank, lock, readers and report refuse to overwrite their outputs. The figure, closeout and deployment build-verify steps overwrite (FIGURE_V2 replaced V1 in place; presentation only).
- **Truth access.** The report reads DEV_REPORT truth only through `acs_hybrids.common.dev_report_truth`. That loader requires this campaign's committed REPORTING_LOCK and logs every re-read in two places: `.private/acs_frozen_hybrids/access/` and the source access log.
- **Sealed role.** The sealed assessment has no export, and every loader refuses it.

## Deploy H_LJ (raw permitted fields → each recipient's probabilities and decision)
```bash
$PY -m acs_hybrids.hybrid_deploy release --package .private/acs_frozen_hybrids/deploy/H_LJ_S0 \
    --input raw.csv --output out.npz [--tasks income|occupation|income,occupation]
```
- **Input header.** `raw.csv` must have exactly the header `AGEP,WKHP,COW,SCHL,MAR,POBP`.
- **Component isolation.** Each recipient reads only its own component: `LOCAL_TREE__E20.npz` and `TREE_TEACHER_income.json` for income, `JOINT_TREE__E20.npz` and `TREE_TEACHER_occupation.json` for occupation (plus `PACKAGE.json`).
- **Refusals.** Other columns, a different column order, and hidden outputs (leaf ids, raw scores, teacher outputs, the other recipient's output) are refused.
- **Build and check.** `build-verify` rebuilds both packages and checks bitwise parity, recipient isolation and the refusals.

## Custody
```bash
$PY -m acs_hybrids.custody snapshot --name SNAPSHOT_V<n>        # ~/PCRL_eval_cache_private/acsh_v1/backups/
$PY -m acs_hybrids.custody restore-check --snapshot <snapshot dir> --restore <fresh dir>
```
`restore-check` re-deploys H_LJ and re-scores the winning pair readers from the backup alone.
