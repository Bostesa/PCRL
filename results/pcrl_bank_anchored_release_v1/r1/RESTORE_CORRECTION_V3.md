# Second restored runtime dependency correction, 2026-10-10

The U-sealed successor `R1_RESTORE_B0_J_READER_V2` failed after 2.474
seconds at its first B0 recipient CLI, before any wire or reader output.
Its owner-only nested traceback identifies a lazy import in decoder
reconstruction: `hcal.deploy.reproduce_table` imports `hcal.calib`, which
imports `lra.decoder`. The V2 runtime source tar omitted `lra`. The V1 and
V2 failed queue receipts, private diagnostics and separate same-device
snapshots/restores remain preserved.

The proposed V3 source tar uses the same pinned source commit. All 450
members of the V2 tar are byte-identical, and exactly 33 tracked members
under `lra/` are added. The 50 deployment files and four reader files keep
their previous hashes. In an isolated extracted source tree, imports of
`bank_anchored.r1_deploy`, `hcal.calib` and `lra.deploy` pass. A synthetic
H-GLOBAL-TEMP `hcal.deploy.reproduce_table` call also passes. This check
reaches the lazy decoder route that earlier import and monkeypatched bank
tests did not exercise.

This changes runtime packaging only. It adds no defense fit, attack fit,
release draw, label access, comparator change or new population. A new
versioned snapshot and queued numerical replay require prospective dated
amendments and independent pre-execution review. Off-device custody remains
pending.
