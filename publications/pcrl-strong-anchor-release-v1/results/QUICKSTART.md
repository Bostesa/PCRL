# Quickstart: reading, re-running and deploying (campaign `pcrl_strong_anchor_release_v1`)

Run every command from the repository root on branch `research/pcrl-strong-anchor-release-v1`.
- **Interpreters.** Two are used:
  - `python3`: 3.14 with numpy 2.4, scikit-learn 1.8 and scipy 1.17. It serves records, rules, claims, tables and the python3
    restore pass.
  - The torch environment: `$TORCH_PY`, python 3.13.7 with torch 2.10. It runs the reader grid, the family fits, deployment and reader
    custody.
- **Private files.** They live under `.private/` and are never published. A fresh clone needs them restored from a custody backup (see
  4 below).
- **Label reads.** Anything that reads a task or SEX label runs only through the queue (`python3 -m sar.jobqueue run ...`), which logs
  each read to `.private/pcrl_strong_anchor_release_v1/access/LABEL_ACCESS.jsonl`. The queue is closed at the end of this campaign,
  so a new label computation needs a new campaign's registration.

## 1. Read the results (label-free)

```
python3 -m sar.summarize decisions --reading FA_R2 --full     # labels and details of one reading (FA_REG FB_REG FB_R1 FA_R1 FS_CPI FA_R2)
python3 -m sar.summarize bank --bank SAR1_FA_R2               # registered-cell pair / income / occupation AUC per member
python3 -m sar.summarize utility --release FA/FA_TFRB_R2      # the 28 usefulness conditions on MODEL_SELECT
python3 -m sar.report tables --main SAR1_REFS --extra-bank SAR1_FB_R1 --extra-bank SAR1_FA_R1 --extra-bank SAR1_FA_R2   # TABLES.md / .json
python3 -m sar.test_rules                                     # the reviewer's rule tests (25 checks, invented data)
```

`ENDPOINTS.json` is the compact machine-readable endpoint table. It is rebuilt from the records by `python3
results/pcrl_strong_anchor_release_v1/roles/coordinator/scripts/write_endpoints.py`. The scripts that wrote the deployment validity
and tolerance records and the custody scope trace are in the same folder.
- It has one entry per reading: labels, the three outcomes, recovery per view and version, D2 / D3 / D4 comparisons, usefulness and
  the D2R robustness record.
- The SAR1_REFS reference levels sit beside the readings.

## 2. Use a deployment package (raw input to both recipients' outputs)

Thirteen packages are under `.private/pcrl_strong_anchor_release_v1/deploy/<CONFIG>_V<v>/PACKAGE.json`:
- the point-eligible releases FA_TFRB_R2 V0 / V1 and FB_PFO V0 / V1 (no method claim);
- their informative controls: FA_TFRB_R2_TASK_LO V0 / V1, FA_TFRB_R2_TASK_HI V0, FA_TFRO_R1_CONT V0 / V1, FB_PFO_DQ_HI V0 / V1 and
  FB_PFO_RAWJ_HI V0 / V1.

Each package pins the source encoder, the model files, the frozen inputs and every code module its deployment function loads, all by
SHA-256. A changed dependency is refused. Run it under `$TORCH_PY`; all 13 releases were fitted there.

Equivalence to the evaluated output:
- Bitwise when the request is the full evaluated batch (all 5,943 DEV_REPORT rows).
- For other request batch sizes, within 6e-16 per probability with identical decisions. The cause is the summation order in batched
  matrix products (`deploy/DEPLOYMENT_TOLERANCE.json`; the largest observed difference is 5.6e-16).

```
$TORCH_PY -c "from sar import deploy as DP; out = DP.release_csv('.private/pcrl_strong_anchor_release_v1/deploy/FA_TFRB_R2_V1', 'rows.csv'); print({t: {k: v.shape for k, v in o.items()} for t, o in out.items()})"
```

`rows.csv` must have exactly the columns AGEP, WKHP, COW, SCHL, MAR, POBP, in that order.
- SEX, task targets, identifiers, extra or reordered columns and hidden-output requests are refused.
- The output per recipient is its float64 probability vector and argmax decision; nothing else is exported.

Re-check every package (bitwise parity with the evaluated output on the DEV_REPORT raw inputs, plus every refusal):

```
$TORCH_PY -m sar.deploy check --packages .private/pcrl_strong_anchor_release_v1/deploy/* --report /tmp/sar_deploy_check.json
```

The campaign's own run is `deploy/DEPLOYMENT_CHECK.json`: all bitwise on the full batch and all refused. Probability validity is in
`deploy/DEPLOYMENT_VALIDITY.json`: the worst row-sum error is 4.4e-16, with no forbidden encoded feature.

## 3. Inspect the reader custody (no label needed)

The five fitted readers in `.private/pcrl_strong_anchor_release_v1/runs/READER_CUSTODY/*.pkl` each hold their SCORE design rows and
raw scores.

```
$TORCH_PY -c "import pickle,glob,numpy as np; [print(p.split('/')[-1], (lambda o: bool(np.array_equal(o['predict'](o['model'], o['X_score']), o['score_raw'])))(pickle.load(open(p,'rb')))) for p in sorted(glob.glob('.private/pcrl_strong_anchor_release_v1/runs/READER_CUSTODY/*.pkl'))]"
```

## 4. Restore from a custody backup and check it

A backup directory holds:
- `TIP_COMMIT`;
- `branch_since_base.bundle` (base 9c9f95385);
- `code.tar` (git archive of the code and the public records);
- `private.tar.gz` (the private scope);
- `MANIFEST.sha256` and `RESTORE_CHECK.json`.

To re-check a backup with the restored code only:

```
python3 -c "from pathlib import Path; from sar import custody as K; print(K.restore_check(Path('<backup directory>'), 'manual re-check'))"
```

This re-hashes the manifest and verifies the bundle. It then extracts both archives into a fresh temporary directory and runs
`sar.restore_check` there under both interpreters. That re-check covers:
- the admitted mirror;
- the source pins;
- every bank member's wire;
- the model files;
- the packaged releases' predictions;
- the custody readers;
- the deployment packages.

It overwrites `RESTORE_CHECK.json` in that backup directory. The custody records are `custody/BACKUP_VERIFICATION.json`
(same device) and `custody/OFFDEVICE_COPY.json` (YOTUO, warm).

## 5. The verifier's code

`roles/verifier/code/` recomputes the metrics, SEs, grid winners, aggregates and decision rules with its own functions (`vlib.py` and
the `v_*.py` / `v2_*.py` scripts).
- It never calls `sar.rules`, `sar.claims`, `nrm.rules` or `sar.devreaders`.
- It uses campaign code only as loaders: the access layer `sar.common.load(..., 'verify', ...)`, `sar.release.load` and the frozen
  reference loaders.
- Its reader refits use the locked recipe modules (`scm_bench.readers`, `task_conditioned_representation.attacks`, `ffv.readers`),
  called through its own dispatch.

`roles/verifier/VERIFICATION_REPORT.md` lists the units it ran. Its label reads went through the queue with operation `verify`.
