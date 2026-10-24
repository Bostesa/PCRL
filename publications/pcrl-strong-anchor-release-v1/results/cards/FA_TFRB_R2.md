# FA_TFRB_R2: the cross-fit bag of the occupation-only SUCC-anchored residual (revision R2; card draft, to be committed as `cards/FA_TFRB_R2.md` before its fit)

> **SAR1 declarations (coordinator; part of this card):** family FA, revision 2 (reading FA_R2; ADAPTIVE_R2: designed after the
> MODEL_SELECT reading of FA_R1, whose aggregate facts it uses); START = SUCC; `cont_declared` = true (CONT = FA_TFRO_R1_CONT, reused,
> no new specification); `ctrl_origin` = START; CTRL = R1's occupation-only lambda = 0 interpolation path bracketed against this
> revision's CAND on CALIBRATION total NLL (points that coincide with START, FA_TFRO_R1_TASK_LO or FA_TFRO_R1_CONT are reused, not saved
> again; new points are saved as FA_TFRB_R2_TASK_LO / _HI); `alt_declared` = false; `selection_data` CTRL = CALIBRATION;
> `start_in_sample_on_model_fit` = true; joint design not claimed; novelty not claimed; no mechanism-specific reader declared; at most 6
> new specifications. No parameter of this card is chosen on MODEL_SELECT; the budgets are SUCC's cap margins from the frozen anchor
> table, as in R1. The investigator's text follows unchanged.


Investigator A, 2026-10-24 06:05Z (`date -u`).

- **Status: ADAPTIVE_R2.** This revision was designed after the FA_TFRO_R1 MODEL_SELECT reading. The MODEL_SELECT facts it uses are the
  coordinator's adjudication aggregates and the committed `claims/FA_R1/DECISIONS.json`; they are listed in section 0. Nothing new was
  computed from MODEL_SELECT, and no label file was opened.
- **Code:** `sar/fa/tfrb.py`, a new module. It reuses `sar/fa/tfr.py` and `sar/fa/tfro.py` unchanged.
- **Deployment:** `sar/fa/__init__.py` routes `FA_TFRB_R2*` ids to `sar.fa.tfrb`. Earlier ids keep their modules.
- **Interpreter:** the torch venv, one thread.
- **Evidence status:** development evidence on previously used records.

## 0. Diagnosis

Figures are v0 / v1. "MS" marks the adaptive facts, used as disclosed.

| | Facts | Reading |
|---|---|---|
| **D1 (MS)** | v1 failed G1 on the men's guard: +0.0099 against 0.005. Over SUCC, men +0.0060 and women −0.0085 (men-minus-women contrast +0.0145); the occupation summed Brier against U0 was −0.00068 against SUCC's +0.00049 (overall change about −0.0012). v0 passed: men +0.0007 (−0.0037 over SUCC), women −0.0034 (+0.0015); contrast −0.0052, overall about −0.0011 | Which group pays changed sign between versions |
| **D2 (MF / CAL records `fa/FIT_R1_V<v>.json`)** | At ρ\* (1.0 / 0.6) the guard estimate's level L, the deployed full fit's CALIBRATION change, was −0.0025 / −0.0039 (se 0.0017), against MS about −0.0011 / −0.0012: within 0.8 / 1.6 SE. Its contrast C, the MODEL_FIT out-of-fold value, was −0.0017 / −0.0001 (se 0.0014), against MS −0.0052 / +0.0145: a 10-SE miss in v1 | **The failure is in the contrast, not the level** |
| **D3** | C was measured on the five fold models, but a sixth run, the full fit, was deployed. The fold-averaged contrasts were near zero at every ρ: v1 +0.0016, −0.0001, +0.0003; v0 +0.0003, −0.0007, −0.0017, −0.0005. Yet the deployed runs' MS contrasts have opposite signs (−0.0052 / +0.0145). FA_TFR_v1's fold-averaged contrasts also had opposite signs across versions (−0.0072 / +0.0042 at ρ 0.3) | Consistent with a **run-specific** asymmetry: which SEX group pays the critic cost differs between training runs. A fold average dilutes it; one deployed run carries it in full. An alternative, that SUCC's in-sample protection on MODEL_FIT changes how the residual acts on new people, does not by itself explain opposite signs with near-zero fold contrasts. It is not excluded |
| **D4 (label-free; `r2_check_lambda0_labelfree.json`, `r2_check_reuse_weights.json`)** | λ = 0 runs are near-unique: the bag of the five fold fits differs from the full fit by a mean KL of 4.2-6.8e-5, against a movement of 0.022-0.027 from SUCC. R1's λ = 0 full fit is reproduced bitwise, compared on the trained weights against both saved models (FA_TFRO_R1_CONT, FA_TFRO_R1_TASK_LO) in both versions | The run-to-run variability belongs to the critic-trained runs. R1's λ = 0 path stays a matched control for a bagged candidate |
| **D5 (MS)** | In v0, R1 did what FA asked: the men's margin was recovered (+0.0007 against SUCC's +0.0044) and the pair was 0.015 lower than SUCC's (0.6027 against 0.6178) | R2 keeps the operation. It changes **what is deployed and how the guard's uncertainty is formed** |

## 1. Operation (mechanism change: deploy the bag of the runs whose guard is measured)

**Runtime inputs:** the six permitted fields only. SEX is never an input or output.

**Release:**
- **income** = SUCC's H3 income vector, bitwise;
- **occupation** = softmax(centre(z_S + (1/5) Σ_k r_k(φ))).
  - r_k is the R1 occupation-only residual (29 → 64 → 64 → 5, zero-initialized output layer).
  - It is fitted on MODEL_FIT minus household fold k (SAR_xf_FA_S<v>, the v1 / R1 folds) with R1's recipe and R1's fold seed key `rho<ρ>|f<k>`.
  - Each fold model keeps its own feature standardization.
- **No separate full fit is made at λ > 0.** The bag is the deployed release.

**Training of each r_k** is R1's, unchanged: distillation of the occupation vector toward the HGB occupation teacher's out-of-fold
probabilities (no task label), plus λ times the joint critic risk (MODEL_FIT SEX). Same critics, calibration, optimizer and schedule.

**Why this is structural and addresses D3.**
- The deployed object is the average of exactly the five models whose out-of-fold contrast is measured. Each MODEL_FIT person is scored
  by the one model that did not see them.
- The out-of-fold contrast is therefore a first-order unbiased estimate of the deployed bag's contrast. The Brier change is linear in a
  small residual up to a quadratic term, and that term is smaller for the bag.
- Averaging also shrinks run-specific asymmetries: by about 1/√5 in SD if the runs are independent.

## 2. Closest previous implementation and the difference

- **FA_TFRO_R1:** the same residual, recipe and fold fits; R1 deployed a separate full fit. R2 deploys the bag and adds run variance to
  the guard SE.
- **Bagging and cross-fit ensembles** (Breiman 1996; stacking practice) are known; no novelty is claimed.

## 3. Measured failure addressed

**R1 v1's men's occupation guard:** +0.0099 against 0.005 on MODEL_SELECT. It is traced, from the records, to the guard contrast being
estimated on runs other than the deployed one (D2, D3).

## 4. Why usefulness may be preserved

- **Occupation:** the fold models are R1's, and R1's CALIBRATION records show resolved occupation gains at ρ 0.3-1.
- **Income:** SUCC's bitwise.
- **The usefulness result is MODEL_SELECT** with the unchanged caps (24 clauses, 4 guards).

## 5. CONT

**FA_TFRO_R1_CONT**, reused (λ = 0, s = 1, gating). No new specification; its trained weights are reproduced bitwise (D4).

## 6. Strength grid and selection rule (prospective; MODEL_FIT and CALIBRATION only; nothing chosen on MODEL_SELECT)

**Grid:** R1's, ρ ∈ {0.3, 0.6, 1, 2, 3, 6, 10}, scanned in increasing order. Each ρ costs five fold fits, and the scan stops at the first
failing ρ. ρ\* is the largest ρ such that it and every smaller ρ pass both parts below. If ρ = 0.3 fails, nothing is released
(NO_FEASIBLE_STRENGTH).

**Budgets:** SUCC's own margins to the caps, unchanged from R1 and read from the committed anchor table (`tfro.succ_margins`).

| Component | v0 | v1 |
|---|---|---|
| occupation NLL | 0.006594 | 0.007053 |
| occupation summed Brier | 0.004877 | 0.004506 |
| occupation accuracy loss | 0.012000 | 0.014000 |
| men's guard | 0.000583 | 0.001059 |
| women's guard | 0.009938 | 0.008039 |

**Part (a):** the deployed bag's CALIBRATION paired occupation spends over SUCC, each at estimate + 2 household-clustered SE, are within
the margins. The income spends are exactly 0, and the code asserts it.

**Part (b):**
- Δ_men = L + (1 − p)·C and Δ_women = L − p·C, where L = the bag's CALIBRATION occupation Brier change, C = the out-of-fold contrast of the
  bag's own fold models, and p = the MODEL_FIT share of men.
- **The changed SE:** SE_C = max(the household-clustered SE, the SD of the five per-fold contrasts / √5).
- SE² = SE_L² + weight²·SE_C².
- Pass if estimate + 2 SE ≤ the men's and the women's margins.

**Why the new SE measures the cap-relevant uncertainty.**
- The bag's contrast on new people is the mean of five models' contrasts, so its run-to-run SD is σ_model/√5.
- Each per-fold contrast is one model's contrast measured on people it did not see. Their spread therefore estimates σ_model plus
  within-fold sampling noise, and SD/√5 estimates the bag-contrast uncertainty including run variance.
- The household SE, which the v1 and R1 rules used, ignores run variance entirely.

**Biases that remain (stated before the fit):**
1. Any SEX-asymmetric part of SUCC's in-sample MODEL_FIT term is still in C. In R1 v0 the MODEL_FIT contrast had the right sign but a
   third of the MODEL_SELECT magnitude, so C may understate magnitudes.
2. Five folds give the between-fold SD only 4 degrees of freedom, so the SE is noisy. It is used at the SAR1 2-SE convention, not at a
   t quantile.
3. First-order algebra only: single-model out-of-fold contrasts stand in for the five-model average on new people. The bag's quadratic
   Brier term is smaller, which is conservative.
4. CALIBRATION's SEX mix is assumed equal to MODEL_FIT's.
5. MODEL_SELECT evaluation noise of the guard (SE about 0.002) is not covered.

**Label use:**
- CALIBRATION income and occupation (never SEX): only part (a), L and the CTRL bracket; nothing is fitted on CALIBRATION, and the controls'
  exposure is identical.
- MODEL_FIT SEX: the critics and the per-fold contrasts. MODEL_FIT task labels: the contrasts only.

## 7. Matched controls (frozen SAR1 rules)

**CTRL:** R1's λ = 0 path, with origin START (s = 0 is SUCC exactly).
- **Path:** the full λ = 0 fit with R1's key `rho0|full`, released at scale s ∈ S_PATH. It is checked bitwise against R1's saved models on
  the trained weights **before any save**. If the check fails, the stop is REUSE_NOT_BITWISE and nothing is saved.
- **Bracket:** `tfr.bracket`, unchanged, on the declared selection data (CALIBRATION paired total-NLL spend over SUCC, which equals the
  occupation NLL spend) against R2's CAND:
  - LO = the largest spend ≤ CAND's;
  - HI = the smallest spend ≥ CAND's + 2 SE;
  - flags CONTROL_PATH_MAY_NOT_BRACKET and CONTROL_PATH_NO_LO_POINT.
- **Reuse at no new specification:**
  - a point at s = 0 is START;
  - a point at s = 0.2 (v0) or 0.3 (v1) is FA_TFRO_R1_TASK_LO;
  - a point at s = 1 is FA_TFRO_R1_CONT.
  - Any other bracket point is saved as FA_TFRB_R2_TASK_LO / _HI: R1's λ = 0 fit at that s, income SUCC's bitwise.
- **Why a full-fit λ = 0 path is matched to a bagged CAND:** λ = 0 runs are near-unique, so the λ = 0 bag and the full fit coincide up to
  a mean KL of 5e-5 (D4).

**CONT:** FA_TFRO_R1_CONT (reused, gating).

**Not built:** SBC, DQ and the release-only tilt. D5 is not claimed.

## 8. Revisions

This is FA's second and last revision slot. None is declared after it.

## 9. Registered predictions, specification count, CPU, fit record

**Predictions** (per version unless stated; registered SAR1 cells):

| Quantity | Prediction |
|---|---|
| ρ\* | P(0.3) 0.40, P(0.6) 0.20, P(1) 0.10, P(≥ 2) 0.05, P(None) 0.25; in at least one version the scan stops on part (b) through SE_C_between: 0.60 |
| SE_C_between > SE_C_household at ρ\*, both versions | 0.75 |
| MS men's guard ≤ 0.005, both versions (G1 on the guard) | 0.75; below SUCC's in both versions: 0.65 |
| all 28 conditions at the point, both versions | 0.70 |
| MS men-minus-women occupation Brier contrast over SUCC within ±0.006, both versions | 0.65 |
| pair AUC (SAR1 cells) | 0.615 (80%: 0.600-0.630); occupation view 0.590 (0.575-0.610) |
| pair benefit over SUCC | −0.002 (80%: −0.015 to +0.012); P(> 0) 0.40; P(resolved) 0.05 |
| pair benefit over CTRL LO / HI | +0.006 / +0.002; resolved over both 0.08 |
| CONTROL_PATH_MAY_NOT_BRACKET in at least one version | 0.35; at least one CTRL point an alias: 0.55 |
| D3 / D4 | 0.02 / 0.01 |

The expectation is a trade: R2 buys robustness of the men's guard with the run-specific part of R1's protection, which the bag averages
away.

**How a failure would be read:**
- **G1 fails on the men's guard again:** the asymmetry is systematic (bias 1), not run-specific. No MODEL_FIT / CALIBRATION estimate of
  the SEX contrast is then reliable for a SUCC-anchored release, and the line stops here.
- **Feasible, but the pair is not below SUCC:** the protection R1 showed in v0 was run-specific.
- **NO_FEASIBLE_STRENGTH:** run variance is too large for the men's margin even at ρ 0.3.

**Specifications:** at most 6 = CAND 2 (FA_TFRB_R2 × 2 versions) + at most 4 new CTRL points (FA_TFRB_R2_TASK_LO / _HI × 2). There are
fewer when bracket points are reused aliases. CONT is reused: 0 new.

**CPU** (the critic-fit cost is measured; the rest estimated): per ρ, 5 critic fold fits at about 9.5 CPU-s; about 2-3 ρ values are
expected before the stop. With the λ = 0 full fit, releases (CAND + at most 2 CTRL × 6 roles) and the MODEL_FIT readers, that is about
3-5 CPU-min per version.

**Fit record** (`results/pcrl_strong_anchor_release_v1/fa/FIT_R2_V<v>.json`):
- CAND: its config, ρ\*, status (FITTED / NO_FEASIBLE_STRENGTH / REUSE_NOT_BITWISE) and the bag's CALIBRATION spend over SUCC;
- per ρ: the part (a) spends and flags, the guard estimate (L, C, the five per-fold contrasts, SE_C_household, SE_C_between, SE_C, Δ_men,
  Δ_women with SEs) and flags, λ per fold, and the biased MODEL_FIT spends;
- CTRL: the origin, the selection data, every S_PATH point's spend and SE against CAND, LO / HI with config ids or aliases, the flags,
  the reuse check and R1's points;
- CONT: reused and gating;
- the frozen-file and TASK.pkl hashes;
- the descriptive MODEL_FIT readers.

## 10. Novelty

Not claimed. The closest published methods are bagging and cross-fit ensembles (Breiman 1996), adversarial debiasing (Zhang, Lemoine &
Mitchell 2018) and knowledge distillation (Hinton, Vinyals & Dean 2015). No literature search was run.
