# Prefit helper correction C — 2026-10-12 UTC

The source-manifest inventory helper previously embedded a machine-specific calibration directory in public code. `make_manifest` now requires the caller's `calibration_units_root`; its CLI requires `--calibration-units-root`. The helper derives each `calU__s0` / `calU__s1` directory from that explicit root.

This correction changes only the helper signature, calibration-directory construction and CLI wiring. The role loader, embargo, useful head algebra, prepared source manifest and original science lock retain their existing contents. No person or label array was opened or hashed, and no numerical job ran for this correction. The existing prepared manifest still has SHA256 `0aeab04be697c5b52d533874013e0ea40af0e9594b11a2988dd89562a4eb8cb1`.

For future inventory construction, use newly allocated output destinations:

```sh
python -m task_conditioned_representation.source \
  --prior-manifest "$TCR_PRIOR_MANIFEST" \
  --calibration-units-root "$TCR_CALIBRATION_UNITS_ROOT" \
  --out "$TCR_NEW_PRIVATE_MANIFEST" \
  --public "$TCR_NEW_PUBLIC_INVENTORY"
```

This generic invocation is documentation, not an executed regeneration. The owner-only correction receipt records the concrete invocation and source hashes. The existing prepared inventory remains authoritative for the current campaign.
