# Card DA_STD_v1: SEX-standardized task posteriors (family DA, Designer A)

Committed before the first fit it governs (coordinator, from `roles/designer_a/PROPOSAL.md`, the challenger's review
`roles/challenger/REVIEW_DESIGNS.md` and `PROTOCOL_AMENDMENT_1.md`). Code: `ddr/da/std.py` (entry `python3 -m ddr.da.std fit
--version V --card <this file>`), shared controls `ddr/ctlkit.py`, data-rich readers `ddr/richreaders.py`, deployment `ddr.da.predict`.

## 1. Operation

Z = `ohn.fam_a.compact_po(X)` (six fields; COW, MAR categorical). s = 1 for women.

| Part | Definition | Fitted on |
|---|---|---|
| Twin tI, tO | the cached task teachers (HGB, A1_PO_16 settings, inner household-CV T*, full refit; `TASK.pkl`, pinned by hash in SOURCE_ADMISSION.json) | MODEL_FIT income, occupation (inherited fit) |
| qO_s(x) = P(occ \| x, s) | the cached SEX-side teacher `occ_s` (HGB on [Z, SEX]; `SEXSIDE.pkl`, pinned) | MODEL_FIT occupation; SEX as a training input (inherited) |
| qI_s(x) = P(inc \| x, s) | NEW: HGB on [Z, SEX], the same settings, inner household-CV T* (tag `OHN_A_cv_S<v>`), full refit, out-of-fold predictions on the 5 household folds `NRM_FA_xf_S<v>` at s = 0 and 1 | MODEL_FIT income; SEX as a training input |
| w | the value on 0.30, 0.31, ..., 0.60 minimizing the larger of the men's and women's MODEL_FIT out-of-fold occupation summed-Brier excess of the lam = 1 release over the twin (ties to the value nearest the SEX share) | MODEL_FIT occupation and SEX labels |

**Release** (deterministic float64; decisions = argmax, first class on ties; clipped at 1e-6 and renormalized):
P_inc = (1 - lam) tI + lam [(1 - w) qI_0 + w qI_1];  P_occ = (1 - lam) tO + lam [(1 - w) qO_0 + w qO_1].
Identity: sum_s pi_s(x) q_s(x) = m_w(x) + (pi(x) - w) Delta(x), Delta = q_1 - q_0. At lam = 1 the person's SEX propensity no longer
enters through the SEX mixture. No propensity model is deployed (except in the COMP ablation). lam = 0 is the twin.

**Strength and selection (MODEL_FIT only, per version):** grid lam in {0.5, 0.75, 1.0}; out-of-fold releases on `NRM_FA_xf_S<v>`
(the inherited teachers' out-of-fold predictions; T fixed at the full-pool T*, non-nested, as in the caches). Six spend clauses over
the twin (occupation NLL, income NLL, occupation accuracy loss, income accuracy loss, men's and women's occupation summed Brier),
estimate + 2 household-clustered SE <= budget **S3** (0.012, 0.006, 0.008, 0.006, 0.0045, 0.0045; chosen because the twin's published
MODEL_SELECT headroom against U0 is about 0.013-0.016 nats of occupation NLL and 0.01 of occupation accuracy). lam* = the largest grid
value such that it and every smaller grid value meet the budget; CAND_STRENGTH_DIFFERS_BY_VERSION recorded if the versions differ. None
qualifying: NO_FEASIBLE_STRENGTH (no fallback).

## 2. Closest previous implementation and what differs

The MMP mixture over SEX values (`results/pcrl_mechanism_design_v1/roles/designer_1/PROPOSAL.md`; data-free, never fitted) is the
lam = 1 device. New here: fitted on the real teachers; a strength path with a MODEL_FIT rule; a guard-balanced weight; the matched
control paths below; the COMP and STD_OCC ablations. Not a closed recipe (no cells or routing, no noise channel, no decoder or
calibrator, no controller, no leaf correction). Designer B's PTC tests the same idea family with a deployed propensity (challenger
finding 1): a joint negative closes one idea, not two.

## 3. Measured failure addressed

(1) Spend explained every earlier drop (FB2 vs SEX-blind noise +0.0027 / +0.0019; PRC_S2 vs HQ_S2 +0.0003). STD's price falls on the
propensity term only (Brier price E[(pi - w)^2 \|\|Delta\|\|^2]). (2) Pair super-additivity: one w removes the shared latent pi(x) from
both recipients at once.

## 4. Why usefulness may be preserved (label-free, disclosed; not outcomes)

Designer A's plug-in at lam = 1, w 0.45: occupation spend 0.0083 nats (person-level SE 0.0009), accuracy loss 0.0039 (0.0016), men's
Brier +0.0014. Coordinator's proxy (`roles/coordinator/labelfree/MIXTURE_PROXY.json`): KL from the twin 0.0084-0.0087 at full strength.

## 5. Twin, controls and ablations (both versions; one fit unit per version)

| Configuration | Kind | Definition |
|---|---|---|
| DA_TWIN | twin | lam = 0: the cached task teachers (never reads SEX). Also DB's twin (one shared twin, challenger finding 4) |
| DA_STD | candidate | lam*, w as above |
| DA_SBC_LO, DA_SBC_HI | control (SBC path) | the same operation with SEX replaced by a SEX-independent attribute A: first a fixed-seed permutation of the MODEL_FIT SEX column (expected to fall short of CAND's spend: a noise column gives q_0 = q_1); if its lam = 1 out-of-fold total-NLL spend does not reach CAND's + 2 SE, PSEUDO2 (`ddr.ctlkit`; PROTOCOL_AMENDMENT_1) with gamma the smallest of (1, 2, 4) that reaches it (else 4, flagged CONTROL_PATH_MAY_NOT_BRACKET). A-aware teachers at the candidate's components' iteration counts (inc_s T*, occ_s T*); w and the lam grid by the same rules. Two bracket points (PROTOCOL 4) on the lam grid; profile (b) of A recorded beside SEX's |
| DA_SPD_LO, DA_SPD_HI | control (SPEND path) | DQ (`ddr.ctlkit`): subtractive-dithered quantization of the twin's standardized principal-component scores, steps (0.35, 0.7, 1.4); two bracket points |
| DA_COMP | ablation | the same SEX-aware parts mixed with the person's fitted propensity pi(x) (cached SEX-side teacher) at lam*: isolates fixing the weights |
| DA_STD_OCC | ablation | occupation STD at lam*, income = twin: per-recipient attribution and leakage transfer |

D2 reads DA_STD against TWIN, the SBC path [TWIN, DA_SBC_LO, DA_SBC_HI] and the SPEND path [TWIN, DA_SPD_LO, DA_SPD_HI] (RULES.md 3).
Descriptive only (never a decision input and never a stop): UND (the twin's boosting stopped early at 0.5x / 1x / 2x STD's per-task
out-of-fold NLL spend) and household cross-fitted MODEL_FIT readers on the out-of-fold releases (`da/FIT_V<v>.json`); the data-rich
reader check (`ddr.richreaders`, PROTOCOL_AMENDMENT_1) with the label READER_BUDGET_DEPENDENT for a D2 pass whose data-rich benefit over
a bracketing control is <= 0 in either version mean.

## 6. Revisions (at most two; each a new card committed before its fit)

- **R1, IPW-STD (estimator).** Trigger (either): (a) at lam*, the OCCUPATION out-of-fold NLL spend exceeds twice Designer A's plug-in
  value (0.0083 nats at lam = 1, scaled by lam^2) (MODEL_FIT); (b) the registered comparison is NOT_SEPARATED and, for either
  recipient, the out-of-fold linear R^2 of logit pi on the lam = 1 release's log-probabilities exceeds half the twin's (a revision
  triggered by (b) follows a MODEL_SELECT outcome and is ADAPTIVE). Revision: one SEX-blind HGB per task trained with weights
  w_s / pi_s(x) (cross-fitted pi clipped to [0.05, 0.95], normalized to mean 1), same settings and selection; release
  (1 - lam) twin + lam IPW model. It reuses the SPEND path (same twin) and refits its SBC path.
- **R2, decision-preserving STD.** Trigger (MODEL_FIT): lam* < 1 (or NO_FEASIBLE_STRENGTH) and an accuracy clause is the only failing
  clause at the next grid value. Revision: the twin's decision is kept with a margin-preserving variant (to avoid the exact-tie
  signature of challenger finding 10), the selection re-run on the same grid.
- Stop: two revisions used, or lam* None without a trigger (task-infeasible, never a protection result).

## 7. Registered predictions (registered grid, 8 cells; benefit = comparator AUC - DA_STD AUC)

- Designer A: TWIN pair / income / occupation 0.658 / 0.567 / 0.647; DA_STD 0.638 / 0.555 / 0.622 (benefit vs TWIN +0.020 [+0.008,
  +0.032]); both views fall; G1 passes (p 0.8); the 0.020 margin over NEURAL_RAW / H_LJ is not met (p 0.97); lam* = 1.0 in at least one
  version (p 0.6), w about 0.45.
- Coordinator (after the challenger): the permuted-SEX SBC falls short of CAND's spend (p 0.9) and PSEUDO2 is used; benefit vs the SBC
  bracket +0.008 [-0.004, +0.020]; vs the DQ SPEND bracket +0.010 [-0.002, +0.022]; D2 = IMPROVEMENT_BEYOND_MATCHED_CONTROLS p 0.15,
  NOT_SEPARATED or PARTIAL p 0.75, a missing bracket p 0.10; D3 p 0.04; D4 p 0.02.

Configurations: 8 x 2 = 16 (DA_TWIN, DA_STD, DA_SBC_LO, DA_SBC_HI, DA_SPD_LO, DA_SPD_HI, DA_COMP, DA_STD_OCC). CPU about 0.3-0.6 h for
the fits (up to four SBC attribute tries of 12 HGB fits each per version).

## 8. Novelty

NOT_CLAIMED. Published device: Pope and Sydnor (2011) standardization; Pearl's adjustment formula; Zliobaite and Custers (2016). Using
it as a defense against attribute inference from released probability vectors is an engineering combination.
