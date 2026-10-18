# Quickstart: running the successor and reproducing the campaign

Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only. The service predicts an
income class (2) and an occupation group (5) from AGEP, WKHP, COW, SCHL, MAR, POBP; each recipient receives its own probability
vector and argmax decision. The private tree (`.private/pcrl_income_headroom_successor_v1/`: admitted exports, fitted parameters,
releases, readers) is never published; the commands below need it (or a restored backup, `BACKUP_VERIFICATION.json`).

## Interpreters

- `python3` = Homebrew Python 3.14.3, numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1: development evaluation (fingerprinted in
  `DEV_EVALUATOR_FINGERPRINT.json`), the successor's deployment (its STRONG income learner is a pickled scikit-learn 1.8.0
  HistGradientBoosting model, `runs/H/STEP1/S<s>/HGB.pkl`, hash in `STEP1.json`), nomination, locks, tables.
- `/Users/<you>/PCRL/.venv/bin/python` = Python 3.13.7, numpy 2.4.2, scikit-learn 1.8.0, scipy 1.17.1, with torch: family P's
  training and the final readers and DEV report (the validation READER_BANK_LOCK versions, asserted by `READER_PLAN.json`).
- Every numerical command runs through the shared queue: `python3 -m ihs.jobqueue run --unit <ID> --stage <stage> --rss-gib <GiB> -- <command>`.

## Deploy the successor from the six raw fields

```python
from ihs import deploy
rows = [{'AGEP': '44', 'WKHP': '48', 'COW': '1', 'SCHL': '16', 'MAR': '5', 'POBP': '012'}]   # exactly these fields, in this order
out = deploy.release('.private/pcrl_income_headroom_successor_v1/runs/DEPLOY/H3_STRONG_LJDIR_PB_S0', rows)
out['income']['probability'], out['income']['decision'], out['occupation']['probability'], out['occupation']['decision']
deploy.release_csv('<package dir>', 'rows.csv')    # CSV with exactly the header AGEP,WKHP,COW,SCHL,MAR,POBP
```

- The wrapper (`release`, `release_csv`) refuses forbidden fields (the replication's schema list, which includes SEX, the task
  labels and identifiers), extra, missing or reordered fields, and any request other than probability and decision. The recorded
  refusal tests (`DEPLOYMENT_CHECKS.json`) cover a SEX field, an extra field, a missing field, reordered fields and a hidden-output
  request. The underlying `ihs.fam_h.predict(seed, config, X, names)` takes the admitted
  145-column encoding and does no schema check; use the wrapper for raw rows.
- The package checks the SHA-256 of every file it depends on (`PACKAGE.json`) before running.
- `python3 -m ihs.deploy check --packages <dirs> --report <json>` reproduces each package's evaluated DEV_REPORT release bitwise
  through the API, the CSV wrapper and the underlying function (label-free) and records the refusals (`DEPLOYMENT_CHECKS.json`:
  6 packages, all bitwise; requests of 7 rows give bit-identical vectors).

## Reproduce the stages (order and gates)

1. `python3 -m ihs.admit` (hash admission) → `python3 -m ihs.references --seed s` (references on the development grid; MODEL_SELECT
   labels are logged) → `python3 -m ihs.inherited`.
2. Family fits: `python3 -m ihs.fam_h fit --seed s --config <ID>` (likewise `fam_o`; `fam_p` under the `.venv` interpreter), then
   `python3 -m ihs.devrun score --family F --config <ID> --seed s ...` (the only path that reads MODEL_SELECT labels; it refuses
   to run under any interpreter or reader code other than the fingerprinted ones). Combinations: `python3 -m ihs.combine`.
3. `python3 -m ihs.nominate` (the frozen rule) → `python3 -m ihs.locks freeze --twin H:HC_STRONG --cc H:HC_TTREE_LJDIR_PB`.
   The loader refuses ATTACK_FIT / ATTACK_VALIDATION SEX until FINALIST_LOCK is committed.
4. `.venv/bin/python -m ihs.final_readers --seed s --out .private/.../runs/READERS_S<s>` → `.venv/bin/python -m ihs.dryrun_report`
   (invented labels) → `python3 -m ihs.locks reporting`.
5. DEV_REPORT truth is read only by `ihs.common.dev_report_truth` after the committed REPORTING_LOCK, once per registered event.
   Both events of this campaign are used (coordinator 03:08:07Z; verifier replay 03:17:54Z); a further call is refused.
6. `python3 -m ihs.feasibility`, `python3 -m ihs.tables`, `python3 -m ihs.custody backup <parent>`, `python3 -m ihs.closeout`.

The two seeds are fits on the same people, not replications; all DEV_REPORT intervals are conditional on the fitted releases and
readers and do not cover the adaptive development search.
