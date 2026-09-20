# TIER_STATUS — `pcrl_utility_extension_v1`

| tier | status | utc | elapsed_h | est. cost | evidence |
|---|---|---|---|---|---|
| T0 infrastructure, archive, restore | **PASS** | 2026-09-20 06:45 | 0.37 | $0.16 | `_scheduler/gates/T0.json`, `_scheduler/tier0_smoke.json` |
| T1 evidence reuse + audit calibration | **PASS** | 2026-09-20 06:51 | 0.48 | $0.20 | `_scheduler/gates/T1.json`, `TIER1_REANALYSIS.md/json`, `CALIBRATION.json` |
| T2 pilot (42 units) | **FAIL** | 2026-09-20 07:12 | 0.82 | $0.35 | `_scheduler/gates/T2.json`, `PILOT_SCREEN.json`, `PILOT_GATE.md` |
| T3 expansion (135 slots) | **NOT STARTED** | — | — | — | forbidden by the T2 gate rule; no partial grid was fitted |
| T4 nominee comparison + 2017 transport | **NOT STARTED** | — | — | — | nothing was nominated |

An earlier T1 attempt FAILED at 06:21 on a cross-architecture assertion (incident 1, `RUN_STATUS.md`);
it was re-run after Amendment 1 and passed. The gate files record inputs, counts, elapsed time and the
next action for every decision.

## Counts

| quantity | value |
|---|---|
| nominal pilot units | 42 (12 protected configurations + 2 controls, x 3 anchors) |
| fitted | 42 |
| audited | 42 (41 distinct + 1 exact duplicate, `X_r2_C1_b030` seed 0) |
| reference audits | 9 (`ref_J`, `ref_A0`, `ref_leace_A0` x 3 anchors) |
| quarantined / probability-integrity failures | 0 / 0 |
| calibration fits | 8 recipes x 3 anchors x 4 endpoints x 2 views |
| degenerate residual targets | 0 (var_T / var(Z_A0) = 0.094 at seed 0) |
| planned but not run | 93 Tier-3 slots, all Tier-4 comparisons, 2017 transport |

## Resume

The run is closed, not paused. Restarting the same program would re-audit the same releases; the
pilot verdict stands on the evidence above. To carry the frontier further, a NEW prospective protocol
is required (`RESEARCH_DECISION.md` section 6) — reusing this one with a larger beta would be an
outcome-driven change. Completed units are reusable by hash from the archive:

    python infra/pcrl_utility_extension_v1/restore.py --bucket <bucket> --prefix pcrl_utility_extension_v1 \
        --chunks results_run__0000 results_run__0001 results_run__0002 --roots roots.json --verify-dir /tmp/v
