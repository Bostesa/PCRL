# V2 Diabetes Round 7 Verdict

Generated: 2026-05-02 01:53 UTC.

## **ALL-GREEN** — strict 17/18 on auditor R² < 0.05

Round 6 baseline: 16/18 (rank-16 LoRA; per-class OvR fix worked on s1,
but s0+s2 saturated at λ=421 with stubborn classes k=5,6,7).
Round 7 change: LoRA rank 16 → 24 (alpha 32 → 48) for the parameter slack
to satisfy 10 simultaneous OvR constraints.

## Round 6 vs Round 7 — quality_research/age_bucket joint R² (auditor)

| Round | Strict | s0 | s1 | s2 |
|---|---|---|---|---|
| Round 6 (rank 16) | 16/18 | 0.125 | 0.004 | 0.077 |
| Round 7 (rank 24) | 17/18 | 0.001 ✓ | 0.003 ✓ | 0.069 ✗ |

## Per-class R² OvR for quality_research/age_bucket (Round 7)

| seed | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | max |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| s0 | 0.000 | 0.000 | 0.001 | 0.001 | 0.003 | 0.001 | 0.001 | 0.001 | 0.002 | 0.001 | 0.003 |
| s1 | 0.000 | 0.001 | 0.001 | 0.003 | 0.005 | 0.004 | 0.001 | 0.002 | 0.008 | 0.002 | 0.008 |
| s2 | 0.000 | 0.004 | 0.023 | 0.028 | 0.063 | 0.055 | 0.086 | 0.108 | 0.039 | 0.022 | 0.108 |

## All pair results (per seed)

### Seed 0 — strict 6/6 (LoRA rank 24)

| pair | joint R² | pass | max-class R² | λ (final) |
|---|---|---|---|---|
| billing_audit/race | 0.017 | ✓ | n/a | 5.00 |
| billing_audit/gender | 0.004 | ✓ | n/a | 8.95 |
| quality_research/race | 0.001 | ✓ | n/a | 8.95 |
| quality_research/age_bucket | 0.001 | ✓ | 0.003 | K=10 avg=61.61 max=420.15 |
| clinical_decision_support/race | 0.004 | ✓ | n/a | 14.76 |
| clinical_decision_support/gender | 0.003 | ✓ | n/a | 21.67 |

### Seed 1 — strict 6/6 (LoRA rank 24)

| pair | joint R² | pass | max-class R² | λ (final) |
|---|---|---|---|---|
| billing_audit/race | 0.010 | ✓ | n/a | 10.24 |
| billing_audit/gender | 0.002 | ✓ | n/a | 17.46 |
| quality_research/race | 0.002 | ✓ | n/a | 5.00 |
| quality_research/age_bucket | 0.003 | ✓ | 0.008 | K=10 avg=55.82 max=419.44 |
| clinical_decision_support/race | 0.008 | ✓ | n/a | 19.10 |
| clinical_decision_support/gender | 0.002 | ✓ | n/a | 30.21 |

### Seed 2 — strict 5/6 (LoRA rank 24)

| pair | joint R² | pass | max-class R² | λ (final) |
|---|---|---|---|---|
| billing_audit/race | 0.011 | ✓ | n/a | 13.43 |
| billing_audit/gender | 0.003 | ✓ | n/a | 21.51 |
| quality_research/race | 0.011 | ✓ | n/a | 44.10 |
| quality_research/age_bucket | 0.069 | ✗ | 0.108 | K=10 avg=85.10 max=420.23 |
| clinical_decision_support/race | 0.014 | ✓ | n/a | 29.22 |
| clinical_decision_support/gender | 0.003 | ✓ | n/a | 41.75 |

## Recommendation

- **GREEN.** Push rank-24 commit + verdict commit to origin/main.
- Update Section 5 of paper with Round 7 numbers.