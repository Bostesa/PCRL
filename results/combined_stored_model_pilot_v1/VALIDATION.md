# Validation

## Before any real-data fit

| Check | Result |
|---|---|
| Test suite (31 original + 21 new) | **52 passed** |
| End-to-end synthetic run through the same shell/CLI path and effective configuration | Direct-signal control established above both bars. Independent-label null never established above. |
| Native / fixed-ridge / relative-ridge quantities kept distinct; held-out R² survives save → load → infer → report, gets an interval, and negative values are retained | Tested (test_16/17) |
| Support enforced in every role (100/30/100); NE distinct from UNRESOLVED | Tested |
| Label-only and utility rows exist; family membership and Bonferroni adjustment reconstruct | Tested |
| Assessment selection prohibited: the fit phase never receives assessment arrays | Tested |
| Per-unit noise contract reaches attacker access records (no global `noise=none`) | Tested |
| A changed consumed code file or manifest blocks execution | Tested. Also checked by the coordinator directly: the verifier passes on a pristine copy and refuses a one-line change to `metrics.py`. |
| Runner selects exactly the 26 registered unit IDs from the 152 available manifests (compared by ID) | `plan OK` |
| Frozen heads reproduce the stored logits for all 3 purposes | max \|Δ\| = 0 |
| Synthetic budget calibration | ≈ 0.26 CPU-h estimated against a 2 CPU-h allowance; no staging needed |

## After the run

| Check | Result |
|---|---|
| Execution | 26/26 units, exit 0, no technical failures, 2,902 model fits |
| Historical native check reproduced (N0) | 8/8 untreated pairs match `dominant_axis_audit.json` within float32 rounding (max \|Δ\| 4.3e-6) |
| N0 anomaly | The historical mixed-precision value is **negative** for the 3 education pairs (−0.09 to −0.41), stored as clamped 0.0. In-sample ridge R² cannot be negative, so the historical float32 computation failed numerically. In float64 the values are 0.006–0.010, so they still pass τ. |
| Independent replay (separate code; saved predictions only; no retraining) | 19,466 items. **FAIL 0**; PASS 19,358; MC_BORDERLINE 21 (secondary decisions within Monte Carlo noise of a bar); NOTE 1; INFO 86. Primary scope: 118/118 PASS. IDs and roles: 213 PASS. Support: 381 PASS. |
| Replay corrections before the final pass | Worst-class and worst-pair = maximum over supported classes/pairs (locked definition). Log-loss clip = 1e-12 (locked). Leakage beyond label-only mapped. Pair-score orientation as specified. Binary U1 macro OvR as specified. Each was a locked definition or a mapping gap; none was changed to match the runner. |
| Coordinator spot checks (sklearn) | P2 income/sex 0.683557, P2 employment/age_group 0.795239, P1 employment/age_group G1 0.008579: all equal to the runner |
| Replay determinism | Re-run by the coordinator: identical counts |
| Resume | 26/26 hash-verified skips, 0 fits |
| Private drive backup | 612/612 files sha256-verified with uncached reads |

## Category labelling note (replay NOTE)

For P2-income_prediction__race, the runner labels the unit **C1**: its historical native check fails, N0 = 0.0575 >
τ. The replay listed C1 + C3.

**Coordinator decision:** report C1. Outside-scope recovery on that unit is measured and shown (P2 established
above), but it is not counted as a C3 finding, because C3 presupposes a passing fitting-sample check. This follows
the same logic as Addendum D1 #8 for P1.
