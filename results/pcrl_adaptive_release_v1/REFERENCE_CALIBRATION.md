# Reference calibration and finite-bank feasibility

The earlier task-aligned-cuts P1 programme selected a reference attacker on a different pool, then imposed that fitted loss as a floor on coefficient rows. An H-only attack has the same loss for every token map on those rows. When its coefficient-row loss fell below the selected reference floor, no channel could satisfy that cut. This was a cross-pool reference contradiction, not a difficult LP or a reason to reinterpret the closed result.

The hash-pinned [historical replay](agents/reference_a/HISTORICAL_REPLAY.json) used stored 2018 coefficient arrays and D17 maps, with no person-row exploration or new fitting:

| Anchor | Old maximum D17 cut violation | Unavoidable old H-only gap | Cuts | Rebased maximum D17 violation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.0095631768 nats | 0.0080164577 nats | 312 | 0 |
| 2 | 0.0074400880 nats | 0.0056375454 nats | 312 | 0 |

The replay file SHA-256 is `4feab5c26ddd3e58c08ba01fc2a780e6f4bc0f0b6102f690e774b3f77cb25a9a`. Its closed-bank hashes are `4df5fec82007687e213c9d0a921701f3d2fe6cee9a6997f1b31baf14a47415b3` and `86f74d9e8b00856e03665e1fec59803789b1c69bca10dd92bedd9b8ee3fb7274` for anchors 1 and 2. These diagnostics do not change those historical banks or outcomes.

The new [registered protocol](PROTOCOL.md) fixes the reference on the *same coefficient people* as every cut. For a role and weighting, let `L_a(Q)=Σ_t,z A_a[t,z]Q[t,z]`, where each attack coefficient already includes normalized original-person mass. With frozen D17 map `Q_ref` and retained attack bank `B`, set `rho(B)=min_{a∈B} L_a(Q_ref)` and require `L_a(Q)≥rho(B)−delta` for every `a`, with registered `delta∈{0,.001,.003}`. Therefore `L_a(Q_ref)≥rho(B)≥rho(B)−delta` for every nonnegative delta. The D17 witness is constructive even when the best or worst cut ignores the token. The implementation checks common coefficient-row, class-order and weighting provenance, independently replays the witness, and recomputes all floors when the bank changes. It does not substitute a separately selected validation risk for `rho`.

This calibration establishes feasibility of the *fixed fitted bank* under the stated rows and reference. It does not show that the candidate improves privacy, that the bank approximates a Bayes attacker, or that any population conditional-information bound holds. Recalibration after adding an attack can lower `rho`, so bank expansion plus rebasing is not necessarily a monotone tightening of the feasible set. A paired per-cut floor `L_a(Q_ref)−delta` would also preserve D17, but is a different, stricter target for non-minimizing attacks and was not silently used here.

In the new run, the anchor-0 first checkpoint archived 156 fitted attack models, 312 cuts, zero replayed reference and candidate cut violations, and a numerical fixed-bank LP gap of approximately `1.1e−16` nats; its receipt SHA-256 is `7e4e0db6445137b6dd0dd6de35c9758d603765e9300a97fb56ccd4fe8bc758d5`. All three A center fits and same-bank controls completed and were read back from private archives. The separately registered privacy-first fixed-bank LP reported `tau` of `0.008931350045` (anchor 0), `0` (anchor 1), and `0.000248163006` nats (anchor 2). These are solver/fitted-bank quantities, not independent sensitive-recovery gains. The archived completed-unit receipt hashes and statuses are recorded in [RUN_STATE.json](RUN_STATE.json).

The reference proof and numerical replay support one claim: the old infeasibility was repaired for this *new 2018 development study* without changing the locked historical protocol. The primary task/privacy endpoint still requires independent selected predictors, paired household uncertainty, and all registered comparators. [INFERENCE.json](INFERENCE.json) reports those separately; its primary conjunction does not pass for either nominated route.
