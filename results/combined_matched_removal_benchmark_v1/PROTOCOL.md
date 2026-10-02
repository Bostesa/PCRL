# Protocol: matched attribute-removal benchmark on frozen Adult and HMDA encoders

**Frozen:** 2026-10-02, before any benchmark fit.

**Authoritative sources, in order:**

| Source | Content |
|---|---|
| `EFFECTIVE_PROTOCOL.json` | Machine-readable, pinned by `LOCK.json` |
| `notes/BENCH_DESIGN.md` | Coordinator design |
| This file | Summary and coordinator decisions |
| `PANEL.csv` | 882 registered units: 126 Tier 1, 756 Tier 2 |
| `PRIMARY_FAMILY.json` | 24 primary endpoints |
| `PERMISSION_TABLE.csv` | Purpose permissions |
| `EFFECTIVE_COVERAGE.csv` | Statistic → fit → saved prediction → inference → report |
| `GUARANTEE_CARDS.md` | Per-method guarantees |
| `RELEASE_ACCESS_TABLE.csv` | Release access per recipe |

Lineage: base pilot 661db8dcbba3abd29bd42d6674c2493c77eb47d8 → reconciliation 07a9ca3 → preparation 031860f.
durable-guarantees: 956f5c88.

## Question

Across removal methods, what does each method's own protection check predict about what recipients can recover,
and what task capability is kept?

The five outcome questions are reported separately:
1. Does the method pass its own check on the stated fitting rows?
2. Does the corresponding quantity generalise under the declared held-out protocol?
3. What do attackers outside that scope recover?
4. Do outputs preserve recovery when representations are protected?
5. What task capability is retained at the declared diagnostic protection level?

A nonlinear attack on LEACE does not refute its linear theorem. Failing to find a C2 does not prove protection
against every predictor.

## Panel

**Tier 1:**
- Cells:
  - Adult `income_prediction` (task `income`) × `sex`;
  - HMDA `underwriting` (task `loan_decision`) × `race`.
- Encoders: PCRL Round-4 `final.pt` seeds {0, 1, 2} per dataset. All 6 were admitted with verified lineage: epoch
  204 and saved λ equal to `lambdas_final`.

**Arms:**

| Arm | Definition |
|---|---|
| A | Untreated |
| B | Official LEACE on the target one-hot |
| C | Official LEACE on the concatenated marginal one-hots of the purpose's disallowed set: Adult {race, sex}, HMDA {race, ethnicity}. Not a claim about intersections. |
| D | Gaussian noise, σ_abs ∈ {0.25, 0.5, 1, 2, 4, 8} × release seeds {0, 1, 2}. One persistent draw per person. |

**References:**
- label-only;
- constant;
- clean outputs;
- untreated.

**Tier 2** is registered now: E1 untreated + LEACE-B on the remaining disallowed pairs; E2 LEACE-C; E3 noise on the
E1 pairs.
- Triggered only by Tier-1 technical validity: all units complete, controls valid, and arm-A N0 reproduction within
  1e-4 per seed.
- Run in a fixed order (dataset → pair → seed → σ → release seed).
- **Stop rule:** before each Tier-2 unit, stop if spent CPU + that unit's projected CPU + the inference reserve
  exceeds 8 CPU-h. Units are never skipped ahead.
- Round-5/7 headline checkpoints are inventoried by hash only, as a separate labelled extension that is not run.

**Calibration (synthetic, at the real sizes):**

| Stage | Projected CPU-h | Cumulative |
|---|---|---|
| Tier 1 | 1.57 | 1.57 |
| E1 | 0.96 | 2.53 |
| E2 | 0.37 | 2.90 |
| E3 | 8.07 | 10.97 |

E3 is expected to be cut by the stop rule. That is decided by budget, not by outcomes.

## Roles and exposure

**Adult:**
- Attacker roles are the pilot's `pilot-roles-v1` on the PCRL test split: 7,571 / 2,239 / 5,250, verified row by row.
- defense_fit = 24,127 historical encoder-training rows. 18 rows duplicating test records are excluded.

**HMDA:**
- Roles are `bench-roles-hmda-v1` on the test split: 6,794 / 2,089 / 4,778.
- defense_fit = 63,704 training rows. 43 duplicate rows are excluded.

**Exposure, disclosed:**
- defense_fit rows are the encoders' own training rows.
- 17 Adult and 42 HMDA test-role rows have an identical record in the training rows. This encoder exposure is
  unresolved; it is not removed.
- HMDA has no applicant ID: equal-key grouping is conservative.
- Everything here is development data that has already been used.

**Support:** 100 per class in defense_fit for eraser concepts, attacker_fit and assessment; 30 in attacker_val.

| Cell | Supported classes | NE |
|---|---|---|
| HMDA race | {0, 1, 2} (3 pairs) | classes 3 and 4 |
| Adult sex | both | — |

LEACE concepts use the full declared one-hot whenever every class has ≥ 100 defense_fit rows. This holds for HMDA
race (smallest class 504).

## Coordinator decisions recorded before fitting

1. **Official LEACE defaults.** concept-erasure 0.2.4 (tag v0.2.4 = 9b18b3d; installed tree sha256 fffac29d…), with
   `svd_tol` 0.01 and default shrinkage.
   - The B/C native status is the official implementation bound: whitened residual ≤ svd_tol. A failure there is C1.
   - Exact-zero cross-covariance is a separate descriptive column. Truncation within tolerance is not C1.
2. **Rank-deficient defense_fit covariances** (Adult s1 rank 29/64; HMDA s0 and s2 16/64). The official behaviour,
   identity outside the fit support, is kept. The label-free out-of-support component norm is reported per (dataset,
   seed) as a transfer limitation.
3. **U1.** The frozen Linear∘ReLU∘Linear head is outside LEACE's scope; it is a compatibility diagnostic only.
4. **Pilot lock.** The pilot `PILOT_LOCK_v2.json` does not verify on this branch, because new code files were added.
   Pilot replay uses commit 661db8d.

## Contracts, attackers, inference

These are as frozen in `notes/BENCH_DESIGN.md` and `EFFECTIVE_PROTOCOL.json`.

**Contracts:** C_rep; C_rep_plus_clean_out; outputs-only; label-only; constant.
- U1 = frozen head (diagnostic).
- U2 = common LR probe.

**Attackers:** L, GBT and MLP grids. Family and hyper-parameters are selected on attacker_val, and the selected recipe
is retrained at attacker seeds {0, 1, 2}. The rep+outputs slate includes ignore-rep and ignore-outputs candidates.
Noise arms add LRT-A2 and LRT-A4.

**Inference:**
- Cluster bootstrap over assessment units, with fitted predictors fixed.
- Seeds (encoder, release, attacker) are averaged within each replicate; they are not units.
- Exploratory: 90 %, B = 2,000.
- Primary: Bonferroni α = 0.05/24, B = 20,000.
- Worst-class and worst-pair bounds are derived from simultaneous per-class/pair bounds.
- σ\* is chosen from attacker_val only.
- U2 non-inferiority margin: −1 percentage point.
