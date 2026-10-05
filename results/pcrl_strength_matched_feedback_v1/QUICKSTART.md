# Quickstart (commands run in this study, 2026-10-05)

Run from `<WORKTREE>` on `research/pcrl-strength-matched-feedback-v1`. Set up first:

```
export OMP_NUM_THREADS=1 PYTHONPATH=.
PY=~/PCRL/.venv/bin/python; P=results/pcrl_strength_matched_feedback_v1; L=$P/PHASE_B_PROTOCOL_LOCK.json; E=$P/EVALUATION_LOCK.json
```

**Where the private data live.**
- Units, checkpoints, critics, probes and per-row predictions: `<PRIVATE_CACHE>/smf_v1/`. The drive copy is `<DRIVE_ROOT>/private_smf_v1_20261005/` (`RESTORE_INDEX.json`).
- Inputs: `<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz` (sha256 `e0d9e54a…`).

**Behaviour you can rely on.**
- Every stage skips complete, hash-verified units, so rerunning is a resume or a no-op.
- Each stage verifies the latest named lock plus amendments, and refuses otherwise.
- Phase A stages ran under `$P/PHASE_A_PROTOCOL_LOCK.json`, and the data stages under `DATA_AND_ENGINEERING_LOCK.json`.

| Step | Command | Result in this study |
|---|---|---|
| Tests | `$PY -m pytest -q smf/tests` | 108 pass: pipeline 14, math review 63, audit 29, late 2 |
| Lock check | `$PY -m smf.lock verify $L --stage phaseB` | `ok: true` |
| Warm starts, parity | `--stage warm`; `--stage parity --shard 0/2` (and `1/2`) | 3 fresh warm starts; 36/36 bitwise parity receipts |
| U and raw controls | `--stage taskline`; `--stage raw` (shards) | U e20/e40; RAW-J and RAW-L, β {0.1, 0.3}, e20/e40 |
| Phase A | `--stage phaseA` (shards; REFRESHED runs first), `--stage inner`, `--stage selectA` | 54 runs; `PHASE_A_SELECTION.json`; schedule REFRESHED |
| Controller | `--stage calibrate`; `--stage preflight` | `calib__s{k}`; `PREFLIGHT.json` |
| Phase B | `--stage phaseB` (shards), `--stage baselines`, `--stage inner`, `--stage selectB`, `--stage tracking` | 36 runs; LEACE and FARE refits; `SEED_STATUS.json`, `SELECTION_TABLE.csv` |
| Evaluation lock | `$PY -m smf.eval_lock $E`, then commit, push and `git fetch` | The assessment refuses without it on origin |
| Assessment | `$PY -m smf.assess --evaluation-lock $E --labels …` | 51 `outer__*` units |
| Controls | `$PY -m smf.assess --evaluation-lock $E --controls-only --seeds 0 --controls-labels J-F L-F U F` | Inner roles only; see `AUDIT_CONTROLS.json` |
| Inference | `$PY -m smf.infer --evaluation-lock $E` | `PRIMARY_ENDPOINTS.csv`, `SECONDARY_ENDPOINTS.csv`, `RAW_LEVELS.csv` |
| Reports | `$PY -m smf.report` | Gradient, controller, tracking, utility and linear tables; figure |
| **Deploy** | `$PY -m smf.deploy --unit <unit> --X <permitted_inputs.npy> --out release.npz` | Tested on `B__s1__J-F__r0.25__e20` (descriptive J-F) and `raw__s1__RAW-J__b0.3__e40` (C\*): bitwise equal to the stored releases; refuses 84 columns; no labels needed |
| Backup / restore | `$PY -m smf.closeout --dest <DRIVE_ROOT>` | See `BACKUP_VERIFICATION.json` |
| Independent replay | `$PY $P/verification/replay_smf.py` | `INDEPENDENT_VERIFICATION.json` |

**Input contract for deployment.** The 83 permitted columns in the admitted order. The 5 numeric columns must be standardised with the NEW_DEFENSE_FIT statistics recorded in `ROLE_MANIFEST.json` (`numeric_refit`); one-hot columns use the fixed category sets.

**Units per seed.** `MODEL_MANIFEST.json` lists every packaged unit with its status, including the strongest development baseline (C\*).
