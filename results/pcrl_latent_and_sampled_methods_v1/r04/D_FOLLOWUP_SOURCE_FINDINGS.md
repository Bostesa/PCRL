# D seven-arm release and task replay

R04 independently restored the second seed's T and D-K releases, complete
attack views and frozen reader predictions. The seed-1 paired receipt is
`FIRST_PAIR_S1_INDEPENDENT_REPLAY.json` (SHA-256
`64e8d7cadd68cc73bae543c5d9c598e20081797de9e601b60d410d985e580bec`).
Its technical status is `TECHNICAL_REPLAY_PASS`. D-K reduces coalition
SEX AUC against T by `0.042181434` on seed 1, above the point target of
`0.020`. Seed 0's reduction is `0.026676127`. Both seeds' T and D-K arms
fail the original two-task utility point gates, so neither is an eligible
ten-bit winner on this evidence. These are two point screens, not simultaneous
confidence results.

R04 then restored all five followup releases on ATT and INNER from admitted
83-column inputs, frozen U checkpoints, FIT-only scalers, and the saved
calibrated heads. `FOLLOWUP_S0_INDEPENDENT_REPLAY.json` (SHA-256
`a3ffef4a9e33ee0e3e0bcd5b2c8afed21a26d74f30fc6010f14aa4f05f996a6c`)
and `FOLLOWUP_S1_INDEPENDENT_REPLAY.json` (SHA-256
`8d753141489dd404e38a1eeca36cc3d05cfebdc548d242fed61ed1051c4bf15a`)
have `FOLLOWUP_RELEASE_REPLAY_PASS` status. F's actual transmitted 16-value
scaled latent and its task probabilities reproduce from frozen tensors. F
passes every INNER U0/Ucal task point cap in both seeds; L, S12, S21 and C
fail, chiefly on occupation. F is the disclosed continuous diagnostic, not
a ten-bit protection arm. Its complete-view recovery and any strongest
control selection remain pending the fresh followup reader attacks and R04
attack replay. The source receipts establish release/task arithmetic only.

The first attempt to queue the five-arm source verifier with a 4 GiB memory
reservation was declined by R03 before child launch because of host headroom.
Immutable `-002` units used a 3 GiB reservation and completed. The original
`-001` records were preserved and superseded; no source or utility rule
changed in the resource-only replacement.
