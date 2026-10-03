# Amendment S1: stage-5 scoring repair, 2026-10-03 (post-result implementation repair)

**Defect.** Found by the independent replay. `report/infer_s5.py` scored each features-plus-own-head release with that
unit's own GBT/MLP attacker. It ignored the registered plus-surface rule: when validation selects the ignore-rep or
ignore-out candidate, the release is scored by that component unit. The rule is declared in the output-aware protocol
and reused unchanged here.

**Affected selections:**
- A on seed 1;
- F on seeds 0 and 1 (ignore-out, so scored by the nominee's FARE-cell unit);
- FZ on seeds 0 and 2 (ignore-rep, so scored by O_headFZ).
- B was unaffected.

**Repair.** Apply the declared alias rule. The estimand, thresholds and selection are unchanged. The original table is
kept privately as `run/s5/S5_ENDPOINTS_before_repair_S1.csv`.

| Endpoint | Before: point / lower | After: point / lower | Decision |
|---|---|---|---|
| R(LEACE) − R(FARE) | 0.1531 / 0.1392 | 0.1512 / 0.1373 | PASS (unchanged) |
| R(FZ) − R(FARE) | 0.0076 / 0.0001 | 0.0074 / 0.0019 | NOT_ESTABLISHED (unchanged) |
| Acc and retention rows | — | identical | PASS (unchanged) |

**Verification.** The independent replay reproduced the original table under the "own attacker" variant to 2e-16, and
the repaired values under the registered rule.

**Also clarified.**
- The S3 NEAR_BOUND percentile columns are Bonferroni-tail quantiles (0.05/68), which are descriptive.
- The stage-5 certificate refusal counts are distinct shared vectors, not rows; see FARE_USEFUL_TASK_RESULTS.md.
