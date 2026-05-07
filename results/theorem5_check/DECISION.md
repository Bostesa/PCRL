# Theorem 5 decision gate — verdict: **FALLBACK_E4**

Computed λ*(C), the smallest non-zero eigenvalue of the block correlation matrix of whitened per-purpose PCRL representations, on the canonical PCRL checkpoint (seed 0) for each dataset.

| Dataset | λ*(C) | max_i ε_i | k·ε/λ* (Thm 5 bound) | empirical max R²(h_concat, A) |
|---------|------:|----------:|---------------------:|------------------------------:|
| adult | 0.0000 | 0.0475 | 65505.6894 | 0.2519 |
| hmda | 0.0002 | 0.1178 | 1479.7009 | 0.7037 |
| diabetes | 0.0204 | 0.0167 | 2.4513 | 0.1077 |

## Decision policy

- Datasets with λ*(C) ≥ 0.3: **0/3**
- Datasets with λ*(C) ≥ 0.1: **0/3**
- Bound is non-vacuous (k·ε/λ* < 1) on every dataset: **False**

**FALLBACK_E4** — λ*(C) is small enough that the bound k·ε/λ* exceeds 1 on most datasets, i.e. it is vacuous. Do not introduce Theorem 5; instead, honestly reframe Theorem 2 as a label-shift bound that does not predict the cross-purpose leakage observed in §5 (consistent with the γ_min < 0.1 finding from Tier-1 Task A).

## Why λ*(C) ≈ 0 (structural, not noise)

The block correlation matrix collapses for two compounding reasons. First, the canonical PCRL checkpoints used here are the same ones summary.json flags as ``STATUS: COLLAPSED``: several per-purpose representations are essentially low-rank. Second, the three purposes share a frozen backbone and differ only by a rank-r LoRA perturbation, so the concatenation $\hat h_{\text{concat}} = [\hat h_1, \hat h_2, \hat h_3]$ lives on a subspace of dimension well below $k \cdot d_{\text{repr}} = 192$ even when each $h_p$ is individually full-rank.

Per-purpose effective rank and concat rank deficit:

| Dataset | per-purpose eff. rank (1e-3) | concat numerical rank | concat rank deficit (out of 192) |
|---------|------------------------------|----------------------:|---------------------------------:|
| adult | [64, 64, 5] | 182 | 10 |
| hmda | [1, 64, 4] | 126 | 66 |
| diabetes | [12, 5, 10] | 144 | 48 |

## Per-dataset detail

### adult
- N (val rows) = 6017, concat dim = 192 (per-purpose dims = [64, 64, 64])
- per-purpose effective rank (>1e-3 of top SVD): [64, 64, 5]
- per-purpose top SVD: [133.623, 192.768, 186.271]
- concat numerical rank (>1e-8 of top): 182 of 192
- C top-5 eigenvalues: ['1.497', '1.521', '1.568', '1.604', '1.799']
- C bottom-5 eigenvalues: ['2.667e-12', '3.281e-12', '3.717e-12', '4.048e-12', '4.728e-12']
- numerical zeros (<1e-06): 58
- per-purpose max ε_i: {'income_prediction': 0.04753233970872661, 'employment_analysis': 0.0053834835265620384, 'education_assessment': 0.025858257995044376}
- empirical R²(h_concat, A): {'sex': 0.2519, 'race': 0.0667, 'age_group': 0.1229, 'marital_status': 0.22, 'income': 0.2244}

### hmda
- N (val rows) = 13660, concat dim = 192 (per-purpose dims = [64, 64, 64])
- per-purpose effective rank (>1e-3 of top SVD): [1, 64, 4]
- per-purpose top SVD: [0.018, 265.173, 180.154]
- concat numerical rank (>1e-8 of top): 126 of 192
- C top-5 eigenvalues: ['1.000', '1.000', '1.552', '1.621', '1.743']
- C bottom-5 eigenvalues: ['-8.667e-16', '-7.954e-16', '-7.276e-16', '-6.534e-16', '-5.898e-16']
- numerical zeros (<1e-06): 123
- per-purpose max ε_i: {'underwriting': 0.014428244136183932, 'pricing_analysis': 0.1177903434594687, 'fair_lending_audit': 0.03360732159984092}
- empirical R²(h_concat, A): {'race': 0.2148, 'ethnicity': 0.7037, 'sex': 0.2114}

### diabetes
- N (val rows) = 10725, concat dim = 192 (per-purpose dims = [64, 64, 64])
- per-purpose effective rank (>1e-3 of top SVD): [12, 5, 10]
- per-purpose top SVD: [201.222, 122.744, 223.206]
- concat numerical rank (>1e-8 of top): 144 of 192
- C top-5 eigenvalues: ['1.126', '1.151', '1.208', '1.241', '1.375']
- C bottom-5 eigenvalues: ['2.226e-13', '2.540e-13', '3.420e-13', '3.571e-13', '3.978e-13']
- numerical zeros (<1e-06): 165
- per-purpose max ε_i: {'billing_audit': 0.016670656012304663, 'quality_research': 0.0012187525660823928, 'clinical_decision_support': 0.004286350039360176}
- empirical R²(h_concat, A): {'race': 0.0336, 'gender': 0.0163, 'age_bucket': 0.1077}

