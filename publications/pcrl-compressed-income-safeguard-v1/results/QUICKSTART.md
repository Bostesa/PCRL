# Runnable artifacts and actual commands

## Standalone deployment

The standalone runtime is compressed_income_safeguard/deployment.py. Its only nonstandard dependencies are NumPy 2.4.2 and Torch 2.10.0; the trial used Python 3.13.7. Private packages live under r5/packages/SERVICE_Sseed and contain DEPLOYMENT.json, model_state.pt and geometry.json. MODEL_MANIFEST.json binds all 18 packages and their source models. An input NPZ contains only X and the exact ordered feature_names. Outputs contain the literal purpose features, own logits/probabilities/decisions; no label or clean teacher is needed.

From the research worktree, using the existing environment:

```sh
PY=../../.venv/bin/python
PACKAGE=.private/compressed_income_safeguard/r5/packages/CONSTRAINED_S0
INPUT=.private/compressed_income_safeguard/r5/inputs/CONSTRAINED_S0.npz
"$PY" -m compressed_income_safeguard.deployment \
  --package "$PACKAGE" \
  --manifest-sha256 54f9559e5a6950ffb297bfea3803f33fba01d3d699c76f790a94278a9fd1e583 \
  --input "$INPUT" \
  --input-sha256 49fb9c70ace65443ec37fe64bd039ba166c2b8f159c504f7cc31ae665ad27cba \
  --output new_private_release.npz
```

That exact model/input invocation was executed inside R5_DEPLOY_CONSTRAINED_S0_V1; it produced output SHA cbb26ecc296d4c0c200ee5215820686e96486402bd6a10438419de0a47c721d7 (also recorded in the authoritative receipt). Use a fresh private output path. Scientific/verification workers must use the durable queue; the deployment CLI is standalone for authorized provider inputs. In the clean publication package set PYTHONPATH=code and supply privately restored package assets and input.

## Commands actually used for the study

Each worker ran through an immutable spec with source/code/config/checkpoint/geometry pins, finite CPU/wall/RSS bounds and explicit one-thread environment. Representative authenticated commands:

```sh
"$PY" -m compressed_income_safeguard.host_queue --count 1
"$PY" -m compressed_income_safeguard.pilot_worker --spec PRIVATE_CONFIG
"$PY" -m compressed_income_safeguard.evaluation_worker --phase fit --spec PRIVATE_CONFIG
"$PY" -m compressed_income_safeguard.trial_reporting --phase trajectory --spec PRIVATE_CONFIG
"$PY" -m compressed_income_safeguard.trial_reporting --phase nomination --spec PRIVATE_CONFIG
"$PY" -m compressed_income_safeguard.trial_reporting --phase report --spec PRIVATE_CONFIG
"$PY" -m compressed_income_safeguard.trial_reporting --phase inference --spec PRIVATE_CONFIG
"$PY" -m compressed_income_safeguard_independent.refit_worker --spec PRIVATE_CONFIG
"$PY" -m compressed_income_safeguard_independent.replay --spec PRIVATE_CONFIG
```

PRIVATE_CONFIG denotes the exact private path registered in each unit receipt, not an unguarded direct-fit command. Unit/config indexes and queue receipts are preserved privately. Source files use only explicit role exports. The closed campaign queue must refuse new scientific work; do not rerun or nominate after INNER exposure.

## Recovery

RESTORE_INDEX.json and BACKUP_VERIFICATION.json identify the versioned snapshot, full uncached byte checks and backup-only replays. Restored code/packages/models/readers alone supplied the numerical restores; original numerical files were not fallback inputs. Private manifest/config absolute paths stay private. Off-device custody remains pending because no writable external filesystem was mounted; the exact unexecuted preparation command and fresh custody-window requirement are in the private owner index. Do not call a same-device restore an off-device backup.
