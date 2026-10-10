# Research decision draft: latent repair and sampled channel

Status: `R19_TEMPLATE_D_EVIDENCE_FILLED_A_FIT_CLOSEOUT_FIXED_DOWNSTREAM_PENDING`. R00 alone writes the effective shared decision. No confirmation population was opened.

## D: verified exploratory result

- Tested arms: T, D-K, L, S12, S21, C on the ten-bit feature-code wire; F on a 16-value-per-recipient continuous latent wire. Both paired model seeds use the same 2,235 INNER people.
- R04 independently replayed task endpoints and fresh complete-view SEX attacks for all seven arms, then all 567 registered simultaneous raw contrasts. Its interval receipt SHA-256 is `3c8ad07fbedcc62044c538451f2ffd4292f0b95088120fc9c26206c02c7d5efa`.
- D-K versus T coalition AUC reduction: seed 0 `0.026676`, seed 1 `0.042181`, equal-seed mean `0.034429`, simultaneous interval `[0.013719, 0.055138]`. The 0.020 benefit target is **UNRESOLVED**. Mean D-K occupation NLL excess over U0 has lower bound `0.010368 > 0.010`; mean Brier excess over Ucal has lower bound `0.006994 > 0.005`. These are **VIOLATIONS**. All six ten-bit arms fail point utility and the matched eligible set is empty.
- Strongest favorable descriptive recovery contrast: D-K versus Ucal score-wire anchor, mean AUC reduction `0.056925` with interval `[0.034748, 0.079102]`; this exceeds 0.020 but D-K fails utility and the service is an anchor, not an eligible ten-bit matched control. Strongest adverse cross-capacity contrast: F coalition AUC exceeds D-K by `0.065924`, interval `[0.043268, 0.088580]`. F passes 24/24 task point clauses, while nine of its 36 seed/mean simultaneous utility clauses remain unresolved.
- D conclusion: a real lower-recovery hard-code tradeoff, **no competitive method pass, no kernel component gain and no joint-design gain**. R18's real one-row D-K deployment and data-free negatives passed (receipt SHA-256 `66aeaf924d030618d67cf60f4a7b151322d3165e48d599a1ab39ce91b4667c81`). R18's restored numerical one-row replay also passed (public SHA-256 `280f062cf67d36e85dbe54c53db463ae2a50e60e8d2ba16e00eb81e6a92c34eb`); R04 verified its linkage (SHA-256 `e3ead5ac4d61049594e43c3c55c333279d50492e5d23e7b82262760cc5fe22de`) without performing a second numerical forward. Full branch bundle and off-device custody remain pending.

## A: bounded FIT closeout; downstream comparison incomplete

- Prospective family: seven references, ten registered A arms, three releases per stochastic arm and K=6,885 ordered clauses. Manifest SHA-256 `f54765a485297cfa40a10e178586b51230679cdefcca9c05b9be421c9a3c1d5b`; R04 prospective order audit SHA-256 `36f03cede6c511a61d999af03c2340600cfe064c2c0f6e5b7f3cef7ccfcd5654`.
- R03's public all-16 FIT closeout (SHA-256 `beb24a699c696adf0c0f7b483c7a1c9e074510393907e05a710d1fac892a9d39`) reports 14 rc0 `COMPLETE_NO_FEASIBLE_POLICY` fits and two `A_SEQ21` rc1 `INCOMPLETE_RESOURCE` fits. The latter stopped after infeasible first-stage searches; stage 2 never ran, so they are technically incomplete rather than completed nulls. Zero of 16 selected a policy. The FIT jobs charged 519.545 seconds combined. R04 independently replayed the first `A_JOINT_STOCH` seed-0 saved search (SHA-256 `5a1cd08d978cc0d2dd47de592e8fcf65512415f77d7c200275912e74a82305ec`): 0/26 feasible checkpoints; this does not prove global channel infeasibility. A full all-16 independent outcome replay remains pending.
- R04 also audited the seed-0 SEQ21 stop (SHA-256 `9c9591197c8d2f71464caf8bcb75fd72aeb9ce749074b1a2b428204f441ffc84`): no feasible stage-1 checkpoint, no stage 2, no actual resource cutoff. This supports the stage-1 interpretation for that seed; it does not independently close seed 1 or the full two-stage comparison.
- R04's prefit source replay showed the frozen identity action fails occupation U0 FIT NLL allowance in both seeds (excesses 0.014312 and 0.015421 versus 0.010) and U0 Brier in seed 1 (0.005348 versus 0.005). It was not a feasible FIT witness. Separate optimistic label-aware bank minima are necessary bounds only.
- The 220 sealed A producer units in R19's saved snapshot account for 14 completed FIT jobs, two failed FIT jobs, 116 selected-policy descendants `NOT_TRIGGERED_PARENT_NO_FEASIBLE_POLICY`, 20 SEQ21 descendants `BLOCKED_PARENT_TECHNICAL_INCOMPLETE`, 23 queue-done fixed-reference/diagnostic release jobs, and 45 fixed units not yet enqueued. No selected-policy releases or attacks ran. Fixed historical references and identity/class diagnostics have separate sources; their readers and endpoint results still need receipts. The old failed prefit attempt and setup charges are outside this 220-unit graph and still count in resources.
- The downstream source sidecar was locally sealed (manifest SHA-256 `2634bfa069741b3589bb5c789cdc26ebd17d802b0081830ab1682351bfadac85`) and registered by R00 (registration SHA-256 `a3fbd98ad1a267402076ebdf233de72c77a4d73e07881259a70901abb26da0d4`). Its `scientific_scope_unchanged=true` is a source/dependency statement, not proof of any fixed control's output or privacy recovery.
- A competitive, joint-design and randomness result is **not established**. The selected-policy comparison lacks actual sampled outputs and attacks. Preserve unrun K6,885 clause IDs; do not substitute the superseded K4,455 draft or shrink the denominator after missing results.

## Campaign-level decision fields for R00

| Field | Draft value or required fill |
| --- | --- |
| Strongest complete method | `NONE_ESTABLISHED_FROM_D`; update only from eligible A evidence. |
| Joint-design contribution | `NOT_EARNED_IN_D`; A pending. |
| Randomness contribution | `NOT_EARNED`; no selected stochastic or same-bank deterministic release. |
| Reused-data status | `EXPLORATORY_DEVELOPMENT_ONLY`. |
| Shared CPU/unit accounting | Join all 220 sealed A producer IDs to R03 receipts: 14 done FIT, two failed FIT, 116 null descendants, 20 technically blocked descendants, 23 queue-done fixed releases and 45 fixed units not yet enqueued at R19's saved snapshot. Include historical failed attempt and separate charges outside the graph. |
| Custody | Same-device D byte restore and numerical one-row replay passed; off-device pending with no authorized writable drive. |
| One next step | Finish or explicitly close fixed A historical/diagnostic downstream work, then use the observed FIT infeasibility to design a new prospective development method rather than claiming confirmation. |

R00 must reconcile [R19's endpoint rows](../PRIMARY_ENDPOINTS_D_UNIFIED_PROPOSAL.csv), R04 independent interpretation, R18 deployment scope, current A receipts and R03 resource totals before converting this draft into a final decision.
