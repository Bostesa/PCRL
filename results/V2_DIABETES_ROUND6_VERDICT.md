# V2 Diabetes Round 6 Verdict

Generated: 2026-05-01 05:58 UTC.

## **PARTIAL** — strict 16/18 on auditor R² < 0.05

Round 5 baseline: 16/18 (Diabetes/quality_research/age_bucket failed on all 3 seeds).

## Strict pass counts (auditor joint R² < 0.05 per pair, all seeds)

| Round | Strict | quality_research/age_bucket s0 | s1 | s2 |
|---|---|---|---|---|
| Round 5 | 16/18 | 0.061 | 0.100 | 0.090 |
| Round 6 | 16/18 | 0.125 ✗ | 0.004 ✓ | 0.077 ✗ |

## Per-class R² OvR for quality_research/age_bucket (Round 6)

| seed | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | max |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| s0 | 0.001 | 0.003 | 0.030 | 0.058 | 0.098 | 0.131 | 0.145 | 0.177 | 0.086 | 0.045 | 0.177 |
| s1 | 0.000 | 0.000 | 0.000 | 0.001 | 0.003 | 0.001 | 0.002 | 0.012 | 0.003 | 0.001 | 0.012 |
| s2 | 0.000 | 0.002 | 0.031 | 0.026 | 0.075 | 0.097 | 0.074 | 0.110 | 0.050 | 0.010 | 0.110 |

## All pair results (per seed)

### Seed 0 — strict 5/6

| pair | joint R² | pass | max-class R² | λ (final) |
|---|---|---|---|---|
| billing_audit/race | 0.010 | ✓ | n/a | 8.98 |
| billing_audit/gender | 0.004 | ✓ | n/a | 15.58 |
| quality_research/race | 0.012 | ✓ | n/a | 54.81 |
| quality_research/age_bucket | 0.125 | ✗ | 0.177 | K=10 avg=91.01 max=421.06 |
| clinical_decision_support/race | 0.005 | ✓ | n/a | 20.15 |
| clinical_decision_support/gender | 0.004 | ✓ | n/a | 30.46 |

### Seed 1 — strict 6/6

| pair | joint R² | pass | max-class R² | λ (final) |
|---|---|---|---|---|
| billing_audit/race | 0.005 | ✓ | n/a | 28.88 |
| billing_audit/gender | 0.006 | ✓ | n/a | 48.72 |
| quality_research/race | 0.002 | ✓ | n/a | 5.00 |
| quality_research/age_bucket | 0.004 | ✓ | 0.012 | K=10 avg=54.18 max=421.46 |
| clinical_decision_support/race | 0.009 | ✓ | n/a | 37.18 |
| clinical_decision_support/gender | 0.001 | ✓ | n/a | 52.00 |

### Seed 2 — strict 5/6

| pair | joint R² | pass | max-class R² | λ (final) |
|---|---|---|---|---|
| billing_audit/race | 0.000 | ✓ | n/a | 20.09 |
| billing_audit/gender | 0.001 | ✓ | n/a | 35.07 |
| quality_research/race | 0.011 | ✓ | n/a | 36.10 |
| quality_research/age_bucket | 0.077 | ✗ | 0.110 | K=10 avg=76.84 max=422.61 |
| clinical_decision_support/race | 0.006 | ✓ | n/a | 36.75 |
| clinical_decision_support/gender | 0.002 | ✓ | n/a | 55.10 |

## Recommendation

- Per-class fix did not move the needle. **Do NOT push.**
- Try fallback: bump LoRA rank 16 → 24 (more representational headroom for K=10 erasure).