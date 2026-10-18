# Quickstart: running the packages and reproducing the overnight occupation-headroom campaign

Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64, civilian
employed), development roles only. The service predicts an income class (2) and an occupation group (5 official broad civilian
groups) from AGEP, WKHP, COW, SCHL, MAR, POBP; each recipient receives its own float64 probability vector and argmax decision. SEX
is the dataset's recorded demographic column; readers estimate it from the released vectors. The private tree
(`.private/pcrl_occupation_headroom_overnight_v1/` and the verified source mirror `.private/pcrl_income_headroom_successor_v1/`) is
never published; the commands below need it (or a restored backup, `BACKUP_VERIFICATION.json`).

## Interpreters

- `python3` = Homebrew Python 3.14.3 (numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1): scorer, ledger, rules, witness selection,
  nomination, tables, families A (HistGradientBoosting pickles) and B, the income branch (pickled scikit-learn 1.8.0 models), and
  the deployment wrapper.
- `/Users/<you>/PCRL/.venv/bin/python` = Python 3.13.7 (numpy 2.4.2, scikit-learn 1.8.0, torch 2.10.0, xgboost 3.2.0): the
  development reader grid (`ohn.devreaders`, versions locked in `DEVGRID_CODE_LOCK.json`), family C (xgboost) and the protection
  stage (`ohn.protect`, torch critics). Every release record (`OCC.json` / `RELEASE.json`) states its interpreter.
- Every numerical command runs through the shared queue: `python3 -m ohn.jobqueue run --unit <ID> --stage <stage> --rss-gib <GiB>
  -- <command>` (two heavy jobs at once, one thread each, FIFO; 14 CPU-hours; research window 8 h).

## Deploy a complete release from the six raw fields

```python
from ohn import deploy
rows = [{'AGEP': '44', 'WKHP': '48', 'COW': '1', 'SCHL': '16', 'MAR': '5', 'POBP': '012'}]   # exactly these fields, in this order
out = deploy.release('<package dir>', rows)          # {'income': {'probability', 'decision'}, 'occupation': {...}}
deploy.release_csv('<package dir>', 'rows.csv')      # CSV with exactly the header AGEP,WKHP,COW,SCHL,MAR,POBP
```

- Packages are built with `python3 -m ohn.deploy build --anchor H3 --occ <FAMILY>:<CONFIG> --seed <s> --out <dir>` (income anchor H3
  = the source's H3_STRONG_LJDIR_PB income computed by its own unmodified `ihs.fam_h.predict`; occupation by the family module's
  `predict`), `--occ SOURCE:JOINT` for the unmodified source successor, or `--anchor COMPLETE --occ INC:<CONFIG>` for an
  income-branch complete release.
- Both the documented wrapper (`release`, `release_csv`) and the underlying function (`underlying(pkg, X, names)`) refuse missing,
  extra, reordered, target (PINCP, OCCP, ...) and SEX fields; only 'probability' and 'decision' may be requested; every dependency's
  SHA-256 is checked before running (`PACKAGE.json`).
- `python3 -m ohn.deploy check --packages <dirs> --report <json>` reproduces each package's evaluated DEV_REPORT release bitwise
  through the API, the CSV wrapper and the underlying function from DEV_REPORT raw inputs (label-free) and records every refusal
  (18 cases per package, including CSV-path, PINCP, OCCP and SEX fields). The raw-input CSV it writes stays under `.private/` and is
  removed after the check.
- Every candidate at once: `python3 -m ohn.deploy_batch --candidates <FAMILY>/<CONFIG>@<ANCHOR> ... --report
  results/.../deploy/DEPLOYMENT_CHECKS_<tag>.json`. It builds both seeds as `runs/DEPLOY/<FAMILY>__<CONFIG>__<ANCHOR>_S<s>` and checks
  them. The final set is batch 3 (38 packages, `deploy/DEPLOYMENT_CHECKS_BATCH3.json`), with these named packages:
  - the strongest witnesses, `B__B2_SINGLE__H3_S*` and `A__A1_PO_16__H3_S*`;
  - the unmodified anchor, `SOURCE__JOINT__H3_S*`;
  - the protection points and their decisive matched controls, `PROT__P2A_PAIR_B80__H3_S*`, `PROT__P2A_CONT__H3_S*` and
    `PROT__P2A_COMP_B80__H3_S*`.

  Each PACKAGE.json records the hash of every file read at runtime (run directories, base learners, witness chain, admitted mirror
  files, admission records), the family modules, the build environment and a code-closure hash.
- Either interpreter serves every package: prediction uses numpy and scikit-learn only; torch is needed only to train protection
  corrections.

## Reproduce the stages (order and gates)

1. `python3 -m ohn.admit` (hash admission into the mirror) -> `/Users/<you>/PCRL/.venv/bin/python -m ohn.devreaders manifest`
   (DEV_GRID.json) -> `... -m ohn.devreaders selftest` -> `... -m ohn.devreaders lock`. Registration commit, then
   `python3 -m ohn.receipt verify --kind registration --commit <publication commit> --research-commit <research commit>`: label reads
   are refused until this verified receipt exists.
2. References: `python3 -m ohn.devrun references --seed s`; reference grid `... -m ohn.devreaders run --spec
   results/.../grid_specs/REF_BANK.json --seed s --out .private/.../runs/GRID/REF_BANK/S<s>`.
3. Families: `python3 -m ohn.fam_a fit --seed s --config <ID>` (likewise `fam_b`; `fam_c` under the `.venv` interpreter), then
   `python3 -m ohn.devrun score --family F --config <ID> --seed s --hypothesis ... --change ... --training ... --selection ...` (the
   only route that reads MODEL_SELECT labels for utility; logged).
4. Witness selection by the frozen rule: `python3 -m ohn.witness select --families A B C` (WITNESS_SELECTION.json, write-once; the
   income-branch trigger is evaluated there).
5. Protection stage (`.venv` interpreter): `-m ohn.protect fit --witness <F>/<C> --arm CONT|LOC|PAIR|COMP --seed s --config <ID>
   --levels 0.5,0.8,m0.9 [--guard equal|none] [--match-stem ...] [--twin ...]`, `-m ohn.protect verify --seed s --config <ID>`,
   `-m ohn.protect gridspec ...`; family B uses `python3 -m ohn.fam_b bases ...` before `fit`; income branch: `python3 -m ohn.fam_inc
   step1 --seed s`, `fit --seed s --config <ID>`, then `python3 -m ohn.devrun score-complete ...`.
6. Grid banks for candidates (`ohn.devreaders run --spec ...`), then `python3 -m ohn.grid_summary --bank <BANK> ... --out ...` (frozen
   stability labels), `python3 -m ohn.nominate --banks ... --candidates ...` (frozen nomination; NOMINATION.json write-once),
   `python3 -m ohn.tables` (UTILITY_FRONTIER.csv, READER_STABILITY.csv, GRID_RECOVERY.csv), `python3 -m ohn.decompose ...`, and, when
   nothing qualifies, `python3 -m ohn.frontier` (NOMINATION_FRONTIER.csv: score against the minimum whole-release ratio).
7. ATTACK_FIT / ATTACK_VALIDATION SEX are refused unless a FINALIST_LOCK is committed; DEV_REPORT truth is refused unless a
   REPORTING_LOCK is committed and its public receipt verified. Neither was opened in this campaign (no finalist; see
   RESEARCH_DECISION.md).
8. Closeout and custody:
   - `python3 -m ohn.closeout` writes RESOURCE_ACCOUNTING.json, ACCESS_RECEIPT.json and HANDOFF.json (no label read).
   - `python3 -m ohn.custody backup <parent dir>` makes a same-device backup: a branch bundle, a code archive, a private-tree
     archive including the verified source mirror, and a manifest. It then restores into a fresh directory and runs
     `ohn.restore_check` there with the restored code: admitted copies, release records and every package redeployed bitwise.
   - `python3 -m ohn.custody offdevice <backup dir> <dest>` repeats the check on an already attached drive only; nothing is
     mounted or forced.
   - Publication: `python3 -m ohn.publish_build ...` builds the sanitized publication commit with git plumbing; then
     `python3 -m ohn.receipt verify ...` checks the remote SHA.

The two seeds are fits on the same people, not replications; the 8 development grid cells contain two household splits shared by
both seeds; every development result is adaptive evidence on previously used records.
