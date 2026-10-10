# Attack-stage prefit engineering note

Date: 2026-10-10. This note follows `CAMPAIGN_LOCK.json`; the original
`ATTACK_CONTRACT.md` and `ATTACK_UNIT_FORECAST.json` remain byte unchanged.
No new R4 attacker has fit or read an Adult SEX label at this point. R0's
dated attack-stage amendment must pin final R4 source hashes and eligible
units before an attacker starts.

## Fixed public side information

The R1 selected deployment CLI emits recipient NPZ fields exactly
`code,prob,decision`, and fixed stdout keys exactly
`schema,arm_id,seed,purpose,rows,policy_receipt_sha256,public_table_sha256,release_sha256`.
R4's arm/seed/realization-specific readers receive every varying emitted
field. The new R4 public receipt will also state the fixed stdout values for
both admitted roles and recipients. The `release_sha256` is computed from
the cached role arrays serialized in memory with the same ordered
`np.savez_compressed(code,prob,decision)` call as R1's deployment CLI. This
hash describes that exact cached wire; it is not a hash of R2's private
code-ID archive. B0/W_Q use their own source service metadata and decoder,
and P* uses its distinct source wire. The constants are public side
information, but cannot change a within-unit AUC. R5 independently reviewed
this choice and found no pre-attack blocker. A pooled multi-arm reader would
need these constants encoded as features and is not this sealed slate.

## Corrected resource interpretation

`ATTACK_UNIT_FORECAST.json` retains a deliberately conservative, now
superseded 2.54 CPU-hour ceiling estimate. That calculation divided 2,660
CPU seconds by 384 fits and treated the result as seconds per fit. The closer
actual source `results/pcrl_latent_and_sampled_methods_v1/r03/A_FIXED_ATTACK_COMPLETION_RECEIPT.json`
records **18 real complete-view 33-fit instances**, 4.141 to 10.410 charged
CPU seconds each (median 8.956), 146.193 seconds total, and maximum child
RSS 624,312,320 bytes. Scaling the maximum observed instance charge over
the 44-instance all-arm ceiling gives 458.04 CPU seconds before new
orchestration, inference, replay and possible bank-specific slowdown.
That is a forecast, not a reservation or a reason to weaken the fixed
30-fit slate. R0 should use actual attack smoke timing for queue reserves
and enforce the campaign's 12 CPU-hour and 10 elapsed-hour bounds.

Data-free checks after the side-information implementation: 17/17 pass.
