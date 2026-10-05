# Method delta versus the predecessor (smf, pinned a9951ed)

## Removed

| Item | Reason |
|---|---|
| AUC-driven local controller | Active but no established benefit |
| REFRESHED and ONLINE_MATCHED schedules | Stalled at ρ ≥ 0.75; no schedule effect at matched strength |
| Selected local midpoint initialisation of Phase B | Removed the trajectory difference against raw runs: every arm now starts from the warm state and runs 40 epochs directly |
| Salt split | smf used salt 0 for Phase A/U/raw and salt 1 for Phase B; every arm now uses the explicit numeric salt 0 |

## Kept unchanged

- Encoders, heads, surrogate, critic views and fixed critic-view head;
- ONLINE critic schedule;
- normalized-direction primitive (`smf.train.normalized_direction`): float64, cap 100, zero tolerance 1e-12;
- raw-penalty arm (pinned `rgj.train` operations, bitwise);
- fitting and selection roles;
- head procedure;
- task gates;
- strong attacker slate.

## New

- **Bank.** Strength extends to ρ 3 and 5 (smf stopped at 1.5) and β 0.6. Fixed allocations a ∈ {0.5, 2} at ρ 3 and 5, in both joint and local treatments, replace the dynamic controller.
- **Instrumentation.** Per-step receipts for both encoder blocks (t, p, q norms; individual, combined and RMS ratios; cos; caps; zero reasons; clip factors; fingerprints). The admitted raw runs are replayed with receipts and must reproduce their checkpoints bitwise.
- **R\* nominee.** Ordinary raw joint training is a predeclared benchmark nominee (claim C), not only a control.
- **Consolidated assessment.** A fixed union of four previously used pools (13,936 rows, 13,929 groups), for precision, explicitly not fresh data.
- **Inference.** 27 primary slots (z = 3.113017) and bootstrap seed 20261006.
