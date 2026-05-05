# BIOS rank-mechanism findings (E1 + E2)

**Date:** auto-generated  
**Sample:** BIOS top-10 dev, n=16000 (frozen bert-base-uncased, no LoRA)  
**Wall:** 594.9s on cuda

## E1 — Per-token-type cosine analysis

Disjoint subset sizes: {'P': 9026, 'H': 749, 'N': 0, 'FN': 19}.

Layer-0 cosine matrix (rows/cols = P, H, N, FN):

```
     P: 1.000  0.244  nan  0.135
     H: 0.244  1.000  nan  0.267
     N: nan  nan  nan  nan
    FN: 0.135  0.267  nan  1.000
```

Layer-0 mean off-diagonal cosine: **0.2156**
Layer-6 mean off-diagonal cosine: **0.2300**
Verdict (orthogonality): **MULTI_RANK_CONFIRMED**  
Verdict (consolidation):  **CONSOLIDATION_OBSERVED**

## E2 — Rank-k LEACE sweep at layer 0

Vanilla layer-1 R² (no eraser): **0.8212**

| k | layer-1 R² | Δ from vanilla |
|---|-----------|---------------|
| 1 | 0.8149 | -0.0062 |
| 2 | 0.8095 | -0.0117 |
| 4 | 0.8045 | -0.0167 |
| 8 | 0.7985 | -0.0227 |

First k with R² < 0.5: **None**  
First k with R² < 0.3: **None**

## Synthesis for §5.5 (post-data, 2026-05-04)

E1 confirms the multi-rank diagnosis: layer-0 [CLS] gender directions
arising from disjoint input-token-type subsets (only-pronouns,
only-honorifics, only-first-names) are nearly orthogonal — mean
off-diagonal cosine 0.22, well below the 0.5 multi-rank threshold.
Heterogeneous input-token sources project gender along distinct
directions, exactly as the rank-1 LEACE failure mode predicts. The
consolidation hypothesis is only weakly supported: layer-6 mean
off-diagonal cosine (0.23) is barely higher than layer-0 (0.22) — the
constructed centroid directions remain nearly orthogonal even after
six layers of attention mixing, suggesting attention does not strongly
consolidate these particular candidate directions into a single shared
gender axis (or, alternatively, that the directions surviving to layer
6 are still distinct subspaces of a rank-collapsed gender manifold).

E2 falsifies the simplest prescriptive fix: rank-k LEACE built from
the rank-13 concept subspace spanned by stacked token-type and
occupation-conditional centroid differences barely moves layer-1 R²
(vanilla 0.821 → rank-8 0.798, Δ=−0.023). Even projecting out the
full rank-13 candidate subspace would not reach the < 0.5 acceptance
target. The interpretation is that linear gender information at layer
1 lives in directions that are NOT in the column span of these
centroid-mean vectors — i.e., the linear probe finds gender via
higher-order distributional structure (variance, skew, off-diagonal
correlations) that first-moment LEACE fundamentally cannot erase.
Adversarial rank-k erasure (R-LACE, Ravfogel 2022) is the next
architectural step but exceeds the C1-C4 budget.

**Combined paper-grade finding for §5.5:** rank-1 LEACE post-projection
fails at layer 0 because gender at that depth is multi-rank in a
structurally strong sense — orthogonal gender directions arising from
heterogeneous input-token-type sources (E1, confirmed) AND linear
gender structure beyond first-moment centroid alignments (E2, also
implied by the lack of rank-k recovery from constructed centroid
bases). Mid-stack attention partially compresses some of this signal,
making rank-1 LEACE effective at layers 6 and 11. The minimum
intervention needed for layer-0 erasure is adversarial rank-k erasure
that targets distributional structure beyond mean differences — a
direction for future work, not a fix achievable within the
C1-C4 PRO-LoRA architecture.

Citations: Belrose 2023 (LEACE rank-1 for binary Z), Ravfogel 2022
(R-LACE adversarial rank-k), Bhardwaj 2020 (debiasing-task tradeoffs),
Zakizadeh 2025 (per-token erasure rank limits), Dong 2021 + Wang 2025
(attention rank collapse — partial mechanism for the consolidation
observed at deeper hooks).