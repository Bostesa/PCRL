# Independent backup-only verifier

ROOT launches the restored code as a `custody` unit from the restored `files/code` directory:

```sh
python -m occupation_bottleneck_independent.restore_worker --spec ROOT_PRIVATE_CONFIG.json
```

The worker source SHA256 is `3529a71eeb036898e8db5010f7ae58c731069a0ce222d35bde9037f1db8a6d1d`. The own functional helper `occupation_bottleneck_independent/replay.py` remains pinned to `5a845905f1f7d29c950b66cf1fb05f49d4b08e66b8b4f3db852e3c5086d9a242`. Both code files must execute from the restored tree and appear as exact sources in the queue manifest. No numerical execution was performed by R7.

Config keys are `unit_id`, `restore_root`, `public_output`, and exactly two `representatives`: U0 seed0 full, and RAW_JOINT_R03_E5 seed0 full. Each representative supplies:

- `arm_id`, `seed`, `release_variant`.
- `deployment`: a `{path,sha256}` pin to restored `DEPLOYMENT.json`.
- `model_state`: a pin to restored `model_state.pt` (weights-only plain tensor dict).
- `input`: a pin to the restored feature-only inference archive with exact fields `X`, `feature_names`.
- `expected_wire`: a pin to the restored archive with exact fields `r1`, `r2`.
- `readers`: at least one record with `reader`, `view`, `orientation`, `model` pin, and `expected` pin. The independent refit model is the plain dict `model`, `mean`, `scale`. The expected independent score archive has exact fields `row_id`, `unit`, `score`.

Use the original independent refit records: `U0_S0_PAIR` is `REP_LR_C1`, pair, orientation1; `RAW_JOINT_R03_E5_S0_PAIR` is `REP_MLP_64x32`, pair, orientation1. Their model/score SHA values are copied from `r4/INDEPENDENT_ANCHOR_REFITS.json`; they are not recomputed by metadata work. RAW full is required when comparing to that original full-wire refit's scores.

All model, input, expected output, reader and own scientific code paths must resolve inside `restore_root` and must be exactly registered in the active queue unit. Symlink sources, original source-tree fallbacks, forbidden assessment paths, and label paths are rejected before hashing. ROOT's config/output and common queue metadata may remain outside the restored tree. Numerical libraries use the existing environment.

The verifier independently checks the raw83 schema, class order, encoder/head tensor shapes and precision, own-head contract, actual functional model output, frozen independent reader scaling and ATT orientation, and exact scores. It loads no source model, teacher, role loader, true labels, or scientific runner routine. Optional head-row-space packages can be reconstructed from their stored public B/P matrices via the factorized projection arithmetic, with the `projection` pin supplied, but a full-trained reader needs expected scores for that actual projected contract.

Success produces `OB1-independent-backup-only-replay-v1` with two records, exact zero wire and reader differences, restored-only source flags, no labels read or hashed, and no new fit. Only a successful ROOT-owned numerical receipt establishes PASS.
