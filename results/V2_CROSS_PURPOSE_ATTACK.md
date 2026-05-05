# V2 Cross-Purpose Concatenation Attack

Auditors trained on the concatenation of all three purpose-specific
representations (`h_concat = [h_p1 | h_p2 | h_p3]`) versus the best
single-purpose baseline.  Numbers are mean ± std across 3 PCRL seeds.
Each cell reports the **best of 3 auditor seeds** (worst case for the
defender). Δ values are in percentage points over the test-split
majority baseline. **Verdict** = PASS if `concat − best_single ≤ 1 pp`,
FLAG otherwise.

Canonical checkpoints:
- **adult**: `checkpoints/v2_adult_ROUND5_s0/final.pt` (+ s1, s2 in same family)
- **diabetes**: `checkpoints/v2_diabetes_ROUND7_s0/final.pt` (+ s1, s2 in same family)
- **hmda**: `checkpoints/v2_hmda_ROUND5_s0/final.pt` (+ s1, s2 in same family)

## Headline findings

- **22/33 (dataset, attribute, auditor) triples flag** under the 1 pp gain threshold.
- **Largest concat gain**: `diabetes/age_bucket/XGB` at **+11.96 pp** (concat Δ +34.37 pp vs single Δ +22.41 pp).
- **Worst-case gain per (dataset, attribute)** (max across LR/MLP/XGB):
  - `adult/age_group` — gain **+9.28 ± 2.62 pp** (XGB) — **FLAG**
  - `adult/income` — gain **+0.30 ± 0.54 pp** (XGB) — **PASS**
  - `adult/marital_status` — gain **+10.87 ± 1.12 pp** (XGB) — **FLAG**
  - `adult/race` — gain **+2.79 ± 0.87 pp** (XGB) — **FLAG**
  - `adult/sex` — gain **+6.70 ± 1.57 pp** (XGB) — **FLAG**
  - `diabetes/age_bucket` — gain **+11.96 ± 2.60 pp** (XGB) — **FLAG**
  - `diabetes/gender` — gain **+0.31 ± 0.08 pp** (MLP) — **PASS**
  - `diabetes/race` — gain **+0.00 ± 0.00 pp** (MLP) — **PASS**
  - `hmda/ethnicity` — gain **+6.45 ± 5.34 pp** (XGB) — **FLAG**
  - `hmda/race` — gain **+5.55 ± 0.90 pp** (MLP) — **FLAG**
  - `hmda/sex` — gain **+5.24 ± 2.46 pp** (MLP) — **FLAG**

## ADULT

| attribute | arch | majority | best_single (Δ pp) | concat (Δ pp) | gain (pp) | verdict |
|-----------|------|----------|--------------------|---------------|-----------|---------|
| age_group | LR | 51.73% | +3.48 ± 0.59 | +5.09 ± 0.25 | +1.62 ± 0.59 | FLAG |
| age_group | MLP | 51.73% | +7.72 ± 2.31 | +15.37 ± 3.95 | +7.66 ± 3.48 | FLAG |
| age_group | XGB | 51.73% | +10.72 ± 2.04 | +20.00 ± 3.18 | +9.28 ± 2.62 | FLAG |
| income | LR | 75.43% | +6.50 ± 2.04 | +6.45 ± 1.85 | -0.05 ± 0.22 | PASS |
| income | MLP | 75.43% | +6.53 ± 1.96 | +6.66 ± 1.86 | +0.14 ± 0.13 | PASS |
| income | XGB | 75.43% | +6.22 ± 1.85 | +6.52 ± 1.31 | +0.30 ± 0.54 | PASS |
| marital_status | LR | 52.30% | +16.72 ± 9.62 | +20.80 ± 8.31 | +4.08 ± 1.98 | FLAG |
| marital_status | MLP | 52.30% | +25.83 ± 5.86 | +32.71 ± 6.64 | +6.87 ± 1.77 | FLAG |
| marital_status | XGB | 52.30% | +27.35 ± 4.31 | +38.22 ± 3.20 | +10.87 ± 1.12 | FLAG |
| race | LR | 86.12% | +0.01 ± 0.01 | -0.08 ± 0.11 | -0.09 ± 0.10 | PASS |
| race | MLP | 86.12% | +2.74 ± 1.67 | +4.03 ± 1.75 | +1.30 ± 1.52 | FLAG |
| race | XGB | 86.12% | +3.41 ± 1.13 | +6.21 ± 0.55 | +2.79 ± 0.87 | FLAG |
| sex | LR | 67.38% | +1.17 ± 0.93 | +1.55 ± 1.48 | +0.38 ± 0.62 | PASS |
| sex | MLP | 67.38% | +14.72 ± 4.34 | +18.81 ± 4.67 | +4.09 ± 1.87 | FLAG |
| sex | XGB | 67.38% | +15.61 ± 3.36 | +22.31 ± 2.99 | +6.70 ± 1.57 | FLAG |

### Worst-case per attribute

| attribute | worst arch | worst gain (pp) | verdict |
|-----------|-----------|-----------------|---------|
| age_group | XGB | +9.28 ± 2.62 | FLAG |
| income | XGB | +0.30 ± 0.54 | PASS |
| marital_status | XGB | +10.87 ± 1.12 | FLAG |
| race | XGB | +2.79 ± 0.87 | FLAG |
| sex | XGB | +6.70 ± 1.57 | FLAG |

## DIABETES

| attribute | arch | majority | best_single (Δ pp) | concat (Δ pp) | gain (pp) | verdict |
|-----------|------|----------|--------------------|---------------|-----------|---------|
| age_bucket | LR | 25.77% | +9.38 ± 3.68 | +19.32 ± 9.55 | +9.94 ± 5.88 | FLAG |
| age_bucket | MLP | 25.77% | +24.33 ± 7.35 | +33.86 ± 10.05 | +9.53 ± 2.87 | FLAG |
| age_bucket | XGB | 25.77% | +22.41 ± 6.36 | +34.37 ± 8.95 | +11.96 ± 2.60 | FLAG |
| gender | LR | 53.50% | +0.35 ± 0.27 | +0.55 ± 0.58 | +0.20 ± 0.34 | PASS |
| gender | MLP | 53.50% | +0.64 ± 0.18 | +0.95 ± 0.15 | +0.31 ± 0.08 | PASS |
| gender | XGB | 53.50% | -0.04 ± 0.14 | -0.55 ± 0.31 | -0.50 ± 0.26 | PASS |
| race | LR | 75.13% | +0.00 ± 0.00 | -0.01 ± 0.02 | -0.01 ± 0.02 | PASS |
| race | MLP | 75.13% | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 | PASS |
| race | XGB | 75.13% | -0.11 ± 0.09 | -0.36 ± 0.11 | -0.25 ± 0.08 | PASS |

### Worst-case per attribute

| attribute | worst arch | worst gain (pp) | verdict |
|-----------|-----------|-----------------|---------|
| age_bucket | XGB | +11.96 ± 2.60 | FLAG |
| gender | MLP | +0.31 ± 0.08 | PASS |
| race | MLP | +0.00 ± 0.00 | PASS |

## HMDA

| attribute | arch | majority | best_single (Δ pp) | concat (Δ pp) | gain (pp) | verdict |
|-----------|------|----------|--------------------|---------------|-----------|---------|
| ethnicity | LR | 72.96% | +11.13 ± 6.79 | +13.63 ± 6.06 | +2.51 ± 1.65 | FLAG |
| ethnicity | MLP | 72.96% | +18.62 ± 7.21 | +24.66 ± 1.04 | +6.04 ± 6.29 | FLAG |
| ethnicity | XGB | 72.96% | +18.87 ± 6.50 | +25.32 ± 1.21 | +6.45 ± 5.34 | FLAG |
| race | LR | 64.84% | +0.52 ± 0.46 | +4.32 ± 5.01 | +3.79 ± 5.04 | FLAG |
| race | MLP | 64.84% | +26.62 ± 2.98 | +32.17 ± 2.08 | +5.55 ± 0.90 | FLAG |
| race | XGB | 64.84% | +28.21 ± 2.28 | +33.35 ± 0.97 | +5.14 ± 1.32 | FLAG |
| sex | LR | 61.50% | +7.01 ± 5.05 | +11.78 ± 9.37 | +4.78 ± 7.21 | FLAG |
| sex | MLP | 61.50% | +30.34 ± 3.14 | +35.58 ± 0.84 | +5.24 ± 2.46 | FLAG |
| sex | XGB | 61.50% | +31.15 ± 3.01 | +36.22 ± 0.66 | +5.07 ± 2.47 | FLAG |

### Worst-case per attribute

| attribute | worst arch | worst gain (pp) | verdict |
|-----------|-----------|-----------------|---------|
| ethnicity | XGB | +6.45 ± 5.34 | FLAG |
| race | MLP | +5.55 ± 0.90 | FLAG |
| sex | MLP | +5.24 ± 2.46 | FLAG |

## Verdict summary

- Total (dataset, attribute, arch) triples evaluated: **33**
- PASS: **11**
- FLAG (concat gain > 1 pp over best single purpose): **22**

Flagged rows (sorted by gain):
- `diabetes/age_bucket/XGB` — concat Δ=+34.37 pp, single Δ=+22.41 pp, gain +11.96 ± 2.60 pp
- `adult/marital_status/XGB` — concat Δ=+38.22 pp, single Δ=+27.35 pp, gain +10.87 ± 1.12 pp
- `diabetes/age_bucket/LR` — concat Δ=+19.32 pp, single Δ=+9.38 pp, gain +9.94 ± 5.88 pp
- `diabetes/age_bucket/MLP` — concat Δ=+33.86 pp, single Δ=+24.33 pp, gain +9.53 ± 2.87 pp
- `adult/age_group/XGB` — concat Δ=+20.00 pp, single Δ=+10.72 pp, gain +9.28 ± 2.62 pp
- `adult/age_group/MLP` — concat Δ=+15.37 pp, single Δ=+7.72 pp, gain +7.66 ± 3.48 pp
- `adult/marital_status/MLP` — concat Δ=+32.71 pp, single Δ=+25.83 pp, gain +6.87 ± 1.77 pp
- `adult/sex/XGB` — concat Δ=+22.31 pp, single Δ=+15.61 pp, gain +6.70 ± 1.57 pp
- `hmda/ethnicity/XGB` — concat Δ=+25.32 pp, single Δ=+18.87 pp, gain +6.45 ± 5.34 pp
- `hmda/ethnicity/MLP` — concat Δ=+24.66 pp, single Δ=+18.62 pp, gain +6.04 ± 6.29 pp
- `hmda/race/MLP` — concat Δ=+32.17 pp, single Δ=+26.62 pp, gain +5.55 ± 0.90 pp
- `hmda/sex/MLP` — concat Δ=+35.58 pp, single Δ=+30.34 pp, gain +5.24 ± 2.46 pp
- `hmda/race/XGB` — concat Δ=+33.35 pp, single Δ=+28.21 pp, gain +5.14 ± 1.32 pp
- `hmda/sex/XGB` — concat Δ=+36.22 pp, single Δ=+31.15 pp, gain +5.07 ± 2.47 pp
- `hmda/sex/LR` — concat Δ=+11.78 pp, single Δ=+7.01 pp, gain +4.78 ± 7.21 pp
- `adult/sex/MLP` — concat Δ=+18.81 pp, single Δ=+14.72 pp, gain +4.09 ± 1.87 pp
- `adult/marital_status/LR` — concat Δ=+20.80 pp, single Δ=+16.72 pp, gain +4.08 ± 1.98 pp
- `hmda/race/LR` — concat Δ=+4.32 pp, single Δ=+0.52 pp, gain +3.79 ± 5.04 pp
- `adult/race/XGB` — concat Δ=+6.21 pp, single Δ=+3.41 pp, gain +2.79 ± 0.87 pp
- `hmda/ethnicity/LR` — concat Δ=+13.63 pp, single Δ=+11.13 pp, gain +2.51 ± 1.65 pp
- `adult/age_group/LR` — concat Δ=+5.09 pp, single Δ=+3.48 pp, gain +1.62 ± 0.59 pp
- `adult/race/MLP` — concat Δ=+4.03 pp, single Δ=+2.74 pp, gain +1.30 ± 1.52 pp
