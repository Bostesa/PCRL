# Quickstart: frozen-finalist validation (PCRL, Florida ACS 2019 development roles)

**Data.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only. The service predicts income class (2) and occupation group (5) from the six fields AGEP, WKHP, COW, SCHL, MAR and POBP. SEX is the dataset's recorded demographic column; separate evaluation readers try to infer it from the released predictions. It is never a service input.

Everything runs from the worktree root, with the project virtual environment:

```
PY=/Users/<you>/PCRL/.venv/bin/python
```

**Private data.** The admitted copies, fitted readers, scores and private draws live under `.private/pcrl_frozen_finalist_validation_v1/`. That directory is never published, and the commands below need it.

**The queue.** Every numerical step runs through the shared queue. It allows at most 2 heavy jobs at once and forces one numerical thread per job:

```
python3 -m ffv.jobqueue run --unit <ID> --stage <stage> --rss-gib 2 -- $PY -m <module> ...
python3 -m ffv.jobqueue summary
```

## 1. The frozen candidate (nothing here changes it)

`B3CTL_UNCOUPLED_MIX50_C6A` (`FROZEN_CANDIDATE_MANIFEST.json`):
- **Occupation.** The stored JOINT_TREE output, bitwise H_LJ's.
- **Income.** One of 6 decision-aligned levels of d = 0.5 × LOCAL_TREE + 0.5 × JOINT_TREE E20 income log-odds:
  - level = `searchsorted(thresholds, d, side="left")`, with frozen label-free MODEL_FIT thresholds;
  - each level's probability vector is fitted on CALIBRATION income labels.
- **Inputs.** AGEP, WKHP, COW, SCHL, MAR and POBP only. SEX is never an input.

## 2. Stages, in the order they ran

| Step | Command | Notes |
|---|---|---|
| Admission by hash | `$PY -m ffv.admit` | writes `SOURCE_ADMISSION.json` (414 copies) |
| Candidate manifest | `$PY -m ffv.manifest` | |
| Output-mapping check | `$PY -m ffv.mapping` | label-free; `OUTPUT_MAPPING_RESULTS.json` |
| Deployment checks | `$PY -m ffv.deploy_check` | label-free; `DEPLOYMENT_CHECKS.json` |
| Deployment from the source backup alone | `FFV_PYTHON=$PY FFV_PYLIB=<private CatBoost lib> bash ffv/backup_deploy_check.sh <backup dir> <out.json>` | research branch only; not in the public package. Uses the backup's own code and data; the interpreter and libraries are local |
| Reader dry run on invented arrays | `$PY -m ffv.readers --seed 0 --out <dir> --invented` | |
| Reader lock | `$PY -m ffv.reader_lock <dry-run dir> <review path>` | |
| Enhanced readers | `$PY -m ffv.readers --seed s --out .private/.../runs/READERS_ENH_S{s}` | needs the committed locks |
| Report dry run on invented labels | `$PY -m ffv.dryrun_report` | |
| Reporting lock | `$PY -m ffv.locks` | then commit and publish the receipt |
| The one logged evaluation | `$PY -m ffv.report --out results/.../report` | refuses a second run; event `DEV_REPORT_TRUTH_REREAD_BY_FFV_COORDINATOR` |
| Confirmation feasibility | `$PY -m ffv.feasibility` | label-free; reads the private draws |
| Public tables | `$PY -m ffv.tables` | |
| Sanitized publication | `$PY -m ffv.publish_build build DEST` then `... commit DEST BRANCH MESSAGE [PARENT]` | |

The DEV_REPORT truth is reachable only through `ffv.common.dev_report_truth`, under two registered events, each usable once: the coordinator's evaluation and the verifier's replay. The source campaign's counters are untouched.

## 3. Using the frozen deployment packages

The packages are the admitted copies under `.private/.../admitted/amd/deploy/{FINALIST,TASKONLY}_S{0,1}`, `NEURAL_RAW_S0` and `H_LJ_S{0,1}`. There is no NEURAL_RAW seed-1 package; that is a coverage gap.

**Two entry points with different strictness** (`DEPLOYMENT_CHECKS.json`; spot-checked in `roles/claims_reviewer/CLAIMS_REVIEW.md`, Appendix A):
- **The documented wrapper** is strict about field names and order. Rows must have exactly AGEP, WKHP, COW, SCHL, MAR and POBP, in that order. SEX, other forbidden fields, extra or reordered keys, and requests for hidden quantities are refused.
  - **Values are not checked** against the cohort or the encoder vocabulary. Out-of-cohort ages and hours (for example AGEP 17 or 200, WKHP 0) are accepted and extrapolated, and unknown category codes are encoded as `__OTHER__`. Only non-numeric or non-finite AGEP/WKHP fail.
  - The valid input domain of `OUTPUT_MAPPING_CHECK.md` is not enforced at deployment.
  - For FINALIST and TASKONLY: `amd.deploy.release(pkg_dir, rows)` and `amd.deploy.release_csv(pkg_dir, path)`.
  - For the frozen references: the same `check_rows` and allowed-output gate around their own code.
- **The underlying source functions** (`amd.fam_b.package_release`, `acs_hybrids.hybrid_deploy.release` and `acs_replication.deploy.release`) refuse hidden-output requests. But they read the row fields by name:
  - **extra keys, a SEX key and reordered keys are silently accepted and ignored**, with bitwise-identical outputs;
  - among key problems, only a missing field fails (KeyError). Non-numeric or non-finite AGEP/WKHP also raise.

  The strict API schema rejection comes from the wrapper, not from the underlying functions. The CSV path is strict in both, because the source CSV reader checks the header.
- **Batch size and the last bit.** The finalist's income probability can differ in the last bit (at most 1.1e-16) depending on how many rows a request contains. This affects 2 of the 6 levels per seed: 2 and 4 on seed 0, 3 and 4 on seed 1. Decisions are unaffected.
  - Full-role batches reproduce the evaluated outputs bitwise, and so did a 600-row batch in the claims review. 1-, 6- and 8-row requests do not.
  - The batch size at which the evaluated bits return was not determined.
  - TASKONLY uses the same arithmetic and was not tested. H_LJ was tested on seed 0 and was unaffected.
- **Threshold ties** (source D3).
  - 2 (seed 0) and 4 (seed 1) MODEL_FIT people sit exactly on a frozen threshold. On seed 1, 2 of them would move one level under an exactly rounded or reversed summation.
  - In every other role, the closest person is 4.5e-6 from a threshold (ATTACK_FIT, seed 0). Neither seed-1 move crosses the income decision (levels 0→1 and 1→2).
  - The thresholds are frozen. A midpoint-threshold variant would be a different release needing its own evaluation.

## 4. Custody

`BACKUP_VERIFICATION.json` records:
- the same-device backup: bundle, code archive and private archive, with a restore-and-deploy check from the backup alone;
- the off-device status. The external drive was not mounted during this campaign, so no off-device copy was made and no cold check was possible. Nothing was forced.
