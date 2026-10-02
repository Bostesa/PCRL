# Preserved laptop-only results (copies)

These files were **untracked** (not git-ignored) in the primary PCRL checkout `/Users/nathansamson/PCRL`
(branch ablations-facct-2026-07-24 at the time). They were on no Git branch and in no external-drive
archive, so the laptop held the only copy (evidence inventory, flags 5 and 7).

**Copying:** byte-identical copies were made 2026-10-01 for preservation only. The originals were left in
place, and the hashes are in `SHA256SUMS`.

**Contents:** per-seed and summary metrics only. There are no checkpoints, representations or
per-person data.

| Directory | Run |
|---|---|
| `results/rebuttal/erase_layer_pilot_aws/` | Erase-layer pilot (May 2026) |
| `results/rebuttal/cross_purpose/` | Cross-purpose rebuttal comparison |
| `results/v2_{adult,hmda}_CROSS_PURPOSE_AB/`, `results/v2_diabetes_CROSS_PURPOSE_DIABETES/` | Cross-purpose constrained retrains |

**Not copied:** the `infra/` launch scripts. They may contain cloud configuration, so they are listed in
MISSING_FILES_EXACT.md for the author to preserve privately.
