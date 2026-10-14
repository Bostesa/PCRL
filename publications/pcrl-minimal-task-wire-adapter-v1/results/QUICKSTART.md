# Quickstart: raw83 deployment of the adapter services

The packages are private: they contain model weights. Each package directory holds `model_state.pt` (frozen source encoders and folded own heads), `adapter.pt` (R, b_res, s) or `ownproj.json`, `geometry.json` (the frozen bases) and `DEPLOYMENT.json` (all hashes plus a scientific-status string).

```bash
PYTHONPATH=. python -m minimal_task_wire_adapter.deployment \
  --package $PRIV/deployment/J_TRUST_E10_S0 \
  --manifest-sha256 <DEPLOYMENT.json sha256> \
  --input  features.npz  --input-sha256 <sha256> \
  --output wires.npz
```

- `features.npz` must contain exactly `X` (n×83 float) and `feature_names` (the 83 ordered source names).
- The output contains exactly `r1` (n×21 income wire) and `r2` (n×29 occupation wire): projected representation, own centered logits, own probabilities and own decision.
- The command refuses:
  - reordered, missing or extra columns;
  - non-finite or integer input;
  - forbidden column names (sex, race, ids, labels, teacher scores);
  - extra fields in the input file (labels, `teacher_score`, `row_id`);
  - unknown flags;
  - a wrong manifest hash, or a manifest whose source checkpoint does not match (the test also changes the seed);
  - tampered weights;
  - any extra output-channel request.
- These refusals are tested in `DEPLOYMENT_RESULTS.json`.

Packaged services, both seeds:
- RAW_PROJ (reference);
- J_TRUST_E10 (the nominee; the privacy criterion is not established);
- TASK_TRUST_E10;
- L_TRUST_E10 (local control);
- PRIOR_GLOBAL_UPDATE_GUARD_E5_OWNPROJ (historical diagnostic; fails utility).

Successful engineering is not a privacy assertion. Each package's `scientific_status` field states exactly which criterion is unestablished.
