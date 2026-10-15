# Deployment quickstart (raw83 → income21 / occupation29)

The packages are private (they contain source weights). They are listed in `MODEL_MANIFEST.json` → `deployment_packages` with their manifest SHA-256. From the research worktree:

```
PYTHONPATH=. python -m two_recipient_adapter.deployment \
  --package $PRIV/deployment/J_BOTH_S0 \
  --manifest-sha256 fea595e58530502c79038447e311856bdd0f1280fde96211238e908dda5a1358 \
  --input X.npz --input-sha256 <sha256 of X.npz> --output wires.npz
```

- **Input:** `X.npz` must contain exactly `X` (float, n × 83, in the package's ordered schema) and `feature_names`. Any id, label or teacher field is refused.
- **Output:** `wires.npz` contains `r1` (n × 21) and `r2` (n × 29) only.
- **Refusals:** the runtime refuses reordered, missing or extra columns; non-finite or integer input; forbidden feature names; extra output-channel or unprotected-output requests; unknown flags; and a wrong manifest pin, a mismatched checkpoint manifest or tampered weights. All 15 refusal checks passed for all 16 packages (`DEPLOYMENT_RESULTS.json`).
- **Packages:** RAW_PROJ, J_BOTH, T_BOTH, L_BOTH, S12, S21, I_ONLY and O_PREVIOUS, for both seeds. Each reproduces its stored INNER release exactly (max abs difference 0), and it did so again after a restore from the same-host snapshot alone. The off-device copy is pending.
- **Status field:** the J_BOTH packages carry `POINT_FEASIBLE_NO_COMPETITIVE_CRITERION`. The control packages say they are matched controls and not a competitive privacy result, and RAW_PROJ and O_PREVIOUS say they are references.
