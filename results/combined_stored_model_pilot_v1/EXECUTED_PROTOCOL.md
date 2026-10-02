# Executed protocol: CELL-A stored-model pilot

**Run:** 2026-10-02. Locked at commit 4f3af91 (`PILOT_LOCK.json` is a byte-identical copy of `PILOT_LOCK_v2.json`).
Executed under that lock. The preparation lock `results/combined_evaluation_preparation_v1/PILOT_LOCK.json` is the
original record and is unchanged.

**Authoritative sources, in order:**
1. `EFFECTIVE_PROTOCOL.json`: machine-readable, pinned by the lock.
2. `notes/FROZEN_DESIGN.md` incl. Addendum D1.
3. `AMENDMENT_1.md`: original vs executed settings, written before any fit.

This page summarises them. Nothing here was changed after outcomes were seen.

## Question

When a release passes the method's own check, what can recipients recover, through which surface and with what
access, and how much task performance remains?

The pilot separates:
- (i) implementation compliance (the historical native check reproduced);
- (ii) generalisation of the matched native quantity to held-out rows;
- (iii) recovery outside the guarantee's scope (nonlinear attackers, output surfaces).

## Cell, rows and units

| Item | Value |
|---|---|
| Encoder | PCRL Round-4 Adult seed-0 `final.pt` (sha256 1cfc2fef…c061); frozen, no defense training or refitting |
| Rows | PCRL Adult test split: 15,060 rows, 15,055 record units (exact duplicates collapse) |
| Exposure | Development data, previously used; not a confirmation |
| Roles | `pilot-roles-v1` record-key hash: attacker_fit 7,571 / attacker_val 2,239 / assessment 5,250 |

**26 units:**
- 8 untreated disallowed pairs:
  - income_prediction/{race, sex};
  - employment_analysis/{race, age_group, marital_status};
  - education_assessment/{sex, race, income}.
- 18 noise units: income_prediction/sex × σ_abs ∈ {0.25, 0.5, 1, 2, 4, 8} × release seeds {0, 1, 2}.
  - The release contract is one persistent Gaussian draw per row per seed.
  - Release seeds are draws over the same people, not independent populations.

## Quantities

**Linear (closed form):**

| Name | Definition | Role |
|---|---|---|
| N0 | Historical native check reproduced: fixed ridge 1e-6, in-sample on all 15,060 test rows. Historical mixed precision + float64. | Untreated only |
| N1 | The same estimator, within assessment rows | Descriptive |
| G1 | Held-out native quantity: fit on attacker_fit, score on assessment, SS_tot around the attacker_fit means, unclamped | Primary P1 |
| G2 | Scale-invariant relative-ridge version, ρ selected on attacker_val | Secondary |
| ρ₁² | Held-out top canonical correlation, the contrast diagnostic | Secondary |
| Pure metric | The G1 predictor scored as macro AUC | Secondary |

**Attackers:**
- Linear: L = multinomial logistic regression, C ∈ {0.01, 0.1, 1, 10, 100}.
- Nonlinear: NL = GBT (20 configs) vs MLP (18 configs), selected on attacker_val log-loss.
- Assessment never selects.
- Surfaces:
  - Untreated units: rep, outputs, rep+outputs.
  - Noise units: rep and rep+outputs. The outputs-surface attackers are reused from income_prediction/sex and
    are not counted as independent evidence.
- Noise units also get:
  - LRT-A2: σ-informed, fitted on released data only;
  - LRT-A4: clean-population stress test, labelled as such.
- Repeated-query attack (A3): staged, not run (N = 1 under the persistent contract).

**References and utility:**
- Label-only reference: P(s | y_task), Laplace α = 1.
- U1: frozen purpose head; for noise units, a forward pass on the noisy representation.
- U2: logistic regression probe, selected on attacker_val.
- Reported against the constant predictor, with paired differences against untreated on identical assessment IDs.

## Support

- Thresholds: 100 in attacker_fit, 30 in attacker_val, 100 in assessment. Frozen in `supported.json` before any
  fit.
- Not estimable at the class level:
  - race classes 0 and 3 (fewer than 100 in attacker_fit);
  - occupation task class 5.
- No unit is NE as a whole.

## Inference

- Cluster bootstrap over assessment units, with fitted predictors held fixed.
- **Exploratory:** 90 % two-sided, B = 2,000, seed 20261002. Noise arms are seed-averaged within each replicate.
- **Primary family:** 16 endpoints.
  - P1 (8): G1 vs τ = 0.05.
  - P2 (8): rep NL-selected macro AUC vs 0.55.
  - Bonferroni simultaneous one-sided percentile bounds, α = 0.05/16, B = 20,000, seed 20261003, numpy `linear`
    quantiles, tail count 62.5.
  - These bounds are asymptotic, not exact.
- **Decisions:**

  | Decision | Rule |
  |---|---|
  | ESTABLISHED_ABOVE | lower bound > bar |
  | ESTABLISHED_BELOW | upper bound < bar (≤ τ for P1) |
  | UNRESOLVED | otherwise |
  | NE | not estimable |

- **Limitation:** the intervals are conditional on single fitted predictors (refit seed 0). Refit variance is not
  estimated.

## Not done (by design or authorization)

- Projection arms: Q was never saved, and refitting is not authorized.
- Permutation nulls and per-cell real-data controls: replaced by synthetic controls (Amendment A18).
- Repeated-query attack: staged.
- CELL-B, other encoders, fresh data.
