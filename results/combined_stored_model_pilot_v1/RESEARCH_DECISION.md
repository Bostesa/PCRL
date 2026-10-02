# Research decision: CELL-A stored-model pilot

**Status:** development evidence, 2026-10-02. It covers one frozen encoder (PCRL Round-4 Adult seed 0), the
already-used Adult test split, and 26 release/pair units.

**Independent replay:** 0 FAIL across 19,466 checks.

None of the conclusions below is a new algorithm, a population privacy result, or a fresh confirmation.

## What the pilot answers

### 1. Implementation compliance: does the method pass its own check?

The historical native check was reproduced: in-sample one-hot ridge R² on all 15,060 test rows.

- **8/8 untreated pairs reproduce** the stored values within float32 rounding.
- **income_prediction/race fails its own check** (N0 = 0.0575 > τ = 0.05). It is a known C1, already seen before
  this pilot.
- **The 3 education_assessment pairs** carry a historical value that is a clamped 0.0 hiding a *negative* raw
  in-sample R² (−0.09 to −0.41).
  - That is impossible for a correct ridge fit, so the historical float32 check was numerically invalid there.
  - In float64 the values are 0.006–0.010. These pairs still pass, but the stored "pass" was produced by a broken
    computation.

### 2. Generalisation of the matched native quantity (primary P1)

The same fixed-penalty quantity is fit on attacker_fit rows and scored on held-out assessment rows. Decisions use
Bonferroni simultaneous bounds over 16 endpoints.

| Pairs | Result |
|---|---|
| employment × 3, income/sex | **Established below τ**, held-out R² 0.009–0.033 |
| education × 3 | Formally **below τ**, but G1 is strongly negative (−0.29, −0.60, −4.42): the fixed 1e-6 penalty is ill-conditioned on these representations. The scale-invariant G2 (exploratory) gives 0.004–0.008, which independently supports "below τ". |
| income/race | **Unresolved** (0.052; bounds 0.036–0.069). This is the known C1 pair. |

**No C2 was established.** On this encoder, where the linear check passes it also holds on held-out rows.

### 3. Recovery outside the guarantee's scope (primary P2 and exploratory)

**Nonlinear attacker on the representation:** recovery is **established above 0.55 for 8/8 pairs**.
- Macro AUC ranges from 0.68 (income/sex, education × 3) to 0.82 (employment × 3).
- That is 7 C3 findings, plus income/race, which is labelled C1 because its own check already fails.

**Decomposition (per untreated pair; one factor changed at a time):**

| Step | What changes | What happens |
|---|---|---|
| F1 → F2 | The same held-out linear predictor scored as AUC instead of R², a pure metric change | AUC 0.52–0.76. A small R² can coexist with linear AUC above 0.55 (income/race 0.757; employment 0.57–0.61; education/income 0.589). |
| F3 → F4 | Logistic regression to the NL-selected attacker | The dominant effect: **+0.14 to +0.24 AUC** on 7 pairs. Only +0.001 for income/race, which a linear model already recovers at about 0.75. |
| F4 → F5 / F6 | Representation to task outputs, or outputs added | Output-only recovery is about the same as representation recovery (Δ −0.04 to +0.002). Adding outputs to the representation changes little. |

**Hidden linear contrast:** for income/race the held-out contrast diagnostic is ρ₁² = 0.14, while pooled one-hot R²
is 0.05. The pooled check averages away a recoverable class contrast.

**Output leakage beyond the label-only reference** (outputs-surface NL AUC − P(s \| y_task) AUC, paired):
**established above 0 for 7/8 pairs** (+0.08 to +0.23); education/income is unresolved.

### Noise channel on income/sex

σ ∈ {0.25 … 8}, one persistent draw per row, 3 release seeds.

| Measure | Untreated | σ = 0.25 | σ = 0.5–1 | σ = 2 | σ = 4–8 |
|---|---|---|---|---|---|
| Representation-only NL AUC | 0.684 | 0.619 | ≈ 0.55 (unresolved) | 0.533 (below 0.55) | 0.50–0.51 |
| Representation + clean outputs NL AUC | 0.681 | 0.673 | 0.671 | 0.672 | 0.669–0.672 |
| U1: frozen head accuracy (constant predictor 0.749) | 0.786 | 0.773 | 0.741–0.690 | 0.637 | 0.607–0.590 |
| U2: refitted probe accuracy | 0.800 | 0.798 | 0.797–0.790 | 0.770 | 0.752–0.749 |

The σ-informed LRT (A2) and the clean-population stress test (A4) did not exceed the NL attacker at any σ.

**What the noise results show:**
- Noise protects the **representation surface only**. If task outputs are also released, recovery stays at about
  0.67 at every σ.
- The frozen head loses all of its lift over the constant predictor by σ = 0.5.
- A refitted probe keeps about 40 % of the clean lift at σ = 2, the first σ where representation-only recovery is
  established below 0.55.

### Utility, untreated (assessment rows; constant predictor in brackets)

| Purpose | U1 frozen head | U2 probe | Constant | Note |
|---|---|---|---|---|
| income_prediction | 0.786 | 0.800 | 0.749 | |
| employment_analysis | 0.998 | 0.998 | 0.277 | Near ceiling: the label is a recoding of input columns |
| education_assessment | 0.904 | 0.931 | 0.335 | |

## What remains unanswered

- **Scope.** One encoder, one dataset, one training seed. The intervals are conditional on single attacker fits
  (refit variance is not estimated). This is development data that has already been seen.
- **No matched projection comparison.** The historical projection matrices were never saved, and refitting was
  not authorized.
- **No repeated-query evaluation.** It is staged; under the persistent-draw contract N = 1.
- **No multi-purpose coalition analysis.** It was not part of this pilot.
- **Release object.** The outputs surface is the purpose head's full logit vector, not a thresholded label ŷ;
  label-only outputs would leak less.
- **Nonlinear generalisation.** It was not tested beyond the single held-out assessment split.

## Decision

The repaired evaluator works end to end on real stored artifacts, and an independent replay confirms it.

The pilot separates, for one encoder, three things the earlier papers conflated:
- the linear check holds out of sample (no C2);
- nonlinear recipients recover every attribute that check covers (C3 on 7 pairs);
- task outputs carry most of that signal, beyond the label itself.

A **larger stored-artifact benchmark is justified, but only where frozen interfaces exist**:
- Round-4 seeds 1–2 and HMDA Round-4, from the drive;
- the NeurIPS Round-5/7 headline checkpoints;
- the noise arms on the other pairs, with manifests already admitted.

It would need no defense training. Projection arms and the AAAI-trained channels would require a separate,
explicit defense-refit authorization. Whether to grant it is the next decision.
