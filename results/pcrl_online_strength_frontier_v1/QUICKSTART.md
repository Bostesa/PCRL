# Quickstart (commands run in this study, 2026-10-05)

Run everything from `<WORKTREE>` on `research/pcrl-online-strength-frontier-v1`. Set up first:

```
export OMP_NUM_THREADS=1 PYTHONPATH=.
PY=~/PCRL/.venv/bin/python; P=results/pcrl_online_strength_frontier_v1
```

**Where the private data live.**
- Units, checkpoints, critics, step receipts and per-row predictions: `<PRIVATE_CACHE>/osf_v1/run/units/`.
- Admitted smf copies: `<PRIVATE_CACHE>/osf_v1/admitted/` (`ADMISSION.json`).
- Input: `<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz` (sha256 `e0d9e54a…2f12`).

**Behaviour you can rely on.**
- Every stage skips complete, hash-verified units, so rerunning is a resume or a no-op.
- `osf.run` refuses a stage unless:
  - the latest named lock and its dated amendments verify;
  - the lock and amendments are byte-identical on `origin`;
  - the stage's governing lock has been reached (`osf.lock.STAGE_MIN_LOCK`).

| Step | Command | Result in this study |
|---|---|---|
| Tests | `$PY -m pytest -q osf/tests` | 109 pass: pipeline, runner, late, admission, closeout, audit, math review (53 fixtures, 53/53 injected defects caught) |
| Lock check | `$PY -m osf.lock verify $P/<LOCK>.json --stage <stage>` | `ok: true`, with `pushed` true for the lock and amendments |
| Timed worker | `<PRIVATE_CACHE>/osf_v1/run/work.sh <LOCK> <stage> [i/n]` | Wraps `$PY -m osf.run --lock $P/<LOCK>.json --stage <stage> --shard i/n` in `/usr/bin/time -l` |
| Admission | `work.sh DATA_AND_ENGINEERING_LOCK admit` | 3 warm starts replayed bitwise; 15 admitted releases rebuilt, bitwise on the smf rows |
| Parity | `work.sh DATA_AND_ENGINEERING_LOCK parity 0/2` (and `1/2`) | 12/12 bitwise |
| Fidelity | `work.sh DATA_AND_ENGINEERING_LOCK fidelity 0/2` (and `1/2`) | 6/6 pass, under AMENDMENT_A1/A2 |
| Replay | `work.sh DATA_AND_ENGINEERING_LOCK replay 0/2` (and `1/2`) | 15/15 bitwise with the admitted checkpoints |
| Timing | `work.sh DATA_AND_ENGINEERING_LOCK timing` | 0.73 s per NORM epoch, so the full bank was chosen |
| Bank | `work.sh TRAINING_PROTOCOL_LOCK bank 0/2` (and `1/2`) | 48 new 40-epoch fits; 0 nonfinite, 0 rescued |
| References | `work.sh SELECTION_AND_AUDIT_LOCK references 0/2` (and `1/2`) | LEACE and FARE admitted, all osf rows encoded; equal to the quarantined early run |
| Inner audits | `work.sh SELECTION_AND_AUDIT_LOCK inner 0/2` (and `1/2`) | 63 `inner__rel__*` units |
| Selection | `work.sh SELECTION_AND_AUDIT_LOCK select` | `SELECTION.json`, `SELECTION_TABLE.csv`, `INNER_FRONTIERS.csv` |
| Tracking | `work.sh SELECTION_AND_AUDIT_LOCK tracking 0/2` (and `1/2`) | 12 `track__*` units |
| Evaluation lock | `$PY -m osf.eval_lock $P/EVALUATION_LOCK.json`, then commit, push and `git fetch` | The assessment refuses without it on origin |
| Assessment | `$PY -m osf.assess --evaluation-lock $P/EVALUATION_LOCK.json --shard 0/2` (and `1/2`) | 72 `outer__*` units (24 labels × 3 seeds) |
| Inference | `$PY -m osf.infer --evaluation-lock $P/EVALUATION_LOCK.json` | `PRIMARY_ENDPOINTS.csv`, `SECONDARY_ENDPOINTS.csv`, `ALL_LEVELS.csv` |
| Reports | `$PY -m osf.report --part all` | Strength, fidelity, tracking, utility and linear tables; frontier figure |
| **Deploy** | `$PY -m osf.deploy --unit <rel unit> --X <permitted_inputs.npy> --out release.npz` | Tested on `rel__s1__RAW-J_b0.3`, `rel__s1__NORM-J_r3_a1` and `rel__s1__U`: bitwise equal to the stored releases; refuses 84 columns; needs no labels |
| Backup and restore | `$PY -m osf.closeout backup --dest <DRIVE_ROOT>`; with the drive absent, `--local-copy` | See `BACKUP_VERIFICATION.json` |
| Predecessor custody | `$PY -m osf.closeout predecessor --restore`, with the drive mounted | `PREDECESSOR_CUSTODY_REPAIR.json` |
| Independent replay | `$PY $P/verification/replay_osf.py` | `INDEPENDENT_VERIFICATION.json` |

**Input contract for deployment.**
- Exactly the 83 permitted columns in the admitted order, as a float matrix.
- The 5 numeric columns standardised with the OSF_DEFENSE_FIT statistics in `ROLE_MANIFEST.json` (`numeric_refit`).
- The 78 one-hot columns over the loader's fixed category sets; an unseen category is an all-zero block.
- No SEX, race, income, occupation, fnlwgt, identifier or label is accepted.

**Released units.** `MODEL_MANIFEST.json` lists every unit with its role and status, including the deployable best development model.
