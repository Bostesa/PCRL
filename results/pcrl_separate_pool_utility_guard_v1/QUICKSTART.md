# Quickstart: standalone raw83 deployment

Packages are private, so model weights and person-level wires are not published. The runtime is `separate_pool_utility_guard/deployment.py`. It needs only an exported package directory (`DEPLOYMENT.json`, `model_state.pt`, `geometry.json`) and a feature-only raw83 `.npz` file with exactly `X` (float, n×83) and `feature_names`. It reads no labels, ids, teacher outputs or source loaders.

```bash
python -m separate_pool_utility_guard.deployment \
  --package <private package dir> --manifest-sha256 <DEPLOYMENT.json sha256 from MODEL_MANIFEST.json> \
  --input features.npz --input-sha256 <sha256> --output wires.npz
```

`wires.npz` holds two arrays:
- `r1`: the income recipient wire, 21 wide. It is the rank-1 projected z, then its own centered logits, probabilities and decision.
- `r2`: the occupation recipient wire, 29 wide. It is the actual z16, then its own centered logits, probabilities and decision.

The aligned pair view is the 50-wide concatenation of the two.

Packaged services, for each of seeds 0 and 1:
- `RAW_HYBRID`: the unchanged start. The epoch-0 selections of SEPARATE_GUARD, GLOBAL_UPDATE_GUARD, OLD_BATCH_GUARD and ORDINARY are bitwise tensor aliases of it.
- `RAW_PROJ`: the strongest known compression anchor.
- `TASK_ONLY` epoch 5: the selected TASK_ONLY state.
- `SEPARATE_GUARD` epoch 5: the trained final state. It was **not selected** because it failed the VALIDATION proxies, and it is kept as a diagnostic.

Inherited rejection checks run on every package and all fired:
- reordered, missing or forbidden columns
- nonfinite or integer input
- requested extra output channels
- a wrong manifest hash
- tampered weights

Standalone and CLI outputs equal the canonical trial wires exactly (max abs difference 0) on the VALIDATION and GUARD feature exports. See `deployment/DEPLOYMENT_RESULTS.json`.
