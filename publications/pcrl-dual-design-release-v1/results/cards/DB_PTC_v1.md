# Card DB_PTC_v1: propensity-tilt cancellation (family DB, Designer B)

Committed before the first fit it governs (coordinator, from `roles/designer_b/PROPOSAL.md`, the challenger's review
`roles/challenger/REVIEW_DESIGNS.md` and `PROTOCOL_AMENDMENT_1.md`). Code: `ddr/db/ptc.py` (entry `python3 -m ddr.db.ptc fit
--version V --card <this file>`), shared controls `ddr/ctlkit.py`, data-rich readers `ddr/richreaders.py`, deployment `ddr.db.predict`.

## 1. Operation

| Part | Definition | Fitted on |
|---|---|---|
| p_I, p_O | the cached task teachers (`TASK.pkl`, A1_PO_16 settings; pinned): the twin, unchanged | MODEL_FIT income, occupation (inherited) |
| q(x) = P(SEX = 1 \| x) | the cached SEX-side teacher `sex` (`SEXSIDE.pkl`; pinned), clipped to [0.01, 0.99] | MODEL_FIT SEX (inherited) |
| w_I, W_O[k], qbar, qbar_k | closed form on MODEL_FIT out-of-fold statistics: w_I = Cov(c(p_I), s) / Cov(q, s); W_O[k] = Cov(c(p_O), s \| k) / Cov(q, s \| k); qbar_k = mean(q \| k); c = centred log-probabilities; k = argmax p_O (the twin's decision). Degenerate (Cov <= 1e-4) globally: no release; a stratum with < 200 people or Cov(q, s \| k) <= 1e-4 uses the global occupation vector | MODEL_FIT SEX label |

**Release** (deterministic float64; decisions = argmax; clipped at 1e-6 and renormalized):
log r_I = log p_I - alpha (q - qbar) w_I + const;  log r_O = log p_O - alpha (q - qbar_k) W_O[k] + const.
At alpha = 1 the released centred log-probabilities have zero covariance with SEX on MODEL_FIT out-of-fold statistics (income overall;
occupation within each decision stratum), by Cov(f(x), s) = Cov(f(x), P(s | x)). alpha = 0 is the twin. The propensity is deployed at
runtime and written into the vector: the release is injective on its support, so any benefit is a reader-relative (first-moment)
effect with no data-processing guarantee (challenger finding 3, disclosed before any outcome).

**Strength and selection (MODEL_FIT only, per version):** grid alpha in {0.2, 0.35, 0.5, 0.65, 0.8, 1.0}; nested estimate: 5 outer
household folds (`DDR_DB_ncv_S<v>`), the weights refitted on the other folds' out-of-fold statistics and SEX labels and applied to the
held-out fold (nested for the weights, not for the inherited teachers). alpha* = the largest grid value such that it and every smaller
value keep estimate + 2 household-clustered SE within **S3** (occupation NLL 0.012, income NLL 0.006, occupation accuracy 0.008,
income accuracy 0.006, men's and women's occupation Brier 0.0045) **plus each task's summed Brier 0.0045** (the designer's added
components: a stricter, declared deviation). One CAND (the S3 selection; challenger finding 14). None qualifying: NO_FEASIBLE_STRENGTH.

## 2. Closest previous implementation and what differs

Closest in this repository: PRC's routing rule (`nrm/fa/prc.py` `route`): the person's runtime propensity, linear, moving them against
their SEX inside a task-cost objective. PTC's exponential tilt is argmin_r KL(r \|\| p) + alpha (pi - c) <r, W>: the cell-free,
continuous, closed-form version (challenger finding 2). It removes measured failure 4 (finite releases favour table readers) but uses
SEX information as PRC did, so the control that closed PRC (task-score-only HQ) is required here as ZTILT. Also close: RCCS (data-free,
never fitted). Not FB2 (randomized amplitude); FB2's SEX-blind channel is PTC's SHUF control.

## 3. Measured failure addressed

(1) SEX-blind controls at matched spend did as well: a shift whose amplitude is unrelated to the person's SEX leaves the release's
covariance with SEX at the twin's; PTC removes it. (2) Super-additive pair: both recipients are cancelled jointly. (4) Table readers:
continuous release. (5) Income accuracy: plug-in income accuracy loss 0.0033 at alpha 0.5.

## 4. Why usefulness may be preserved (label-free plug-in, Designer B, disclosed; not outcomes)

At alpha 0.5: KL 0.0046 (income) / 0.0027 (occupation), income accuracy loss 0.0033, men's occupation Brier +0.0023. Plug-in pair 0.642
(twin) -> 0.610 vs 0.634 for a spend-matched keyed-donor control; about 60% of the advantage over that control erodes as plug-in readers
get more data; with an estimated propensity the advantage halves.

## 5. Twin, controls and ablation (both versions; one fit unit per version)

| Configuration | Kind | Definition |
|---|---|---|
| DA/DA_TWIN | twin (shared; no new configuration) | the cached task teachers released unchanged (alpha = 0) |
| DB_PTC | candidate | alpha* |
| DB_SBC_LO, DB_SBC_HI | control (SBC path) | the same tilt law with SEX replaced by a SEX-independent target: a fixed-seed permutation of the MODEL_FIT SEX column first (propensity HGB refitted on it with the SEX-side settings and iteration count; weights by the same rule); if degenerate (Cov <= 1e-4) or unable to reach DB_PTC's nested spend + 2 SE, PSEUDO2 (`ddr.ctlkit`, gamma = 1; PROTOCOL_AMENDMENT_1). Grid: STRENGTHS x m, m = m0 x the smallest of (1, 2, 4, 8, 16) whose top point reaches CAND's spend + 2 SE (m0 equalizes the RMS log-shift with the candidate's). Two bracket points; profile (b) recorded |
| DB_SPD_LO, DB_SPD_HI | control (SPEND path) | DQ (`ddr.ctlkit`): subtractive-dithered quantization of the twin's standardized principal-component scores, steps (0.15, 0.25, 0.4, 0.6, 0.9, 1.4); two bracket points |
| DB_ZTILT | control (release-only; REQUIRED, reported beside D2) | the same tilt law with the amplitude q_z = P(SEX \| the twin's released log-probabilities) (cross-fitted HGB on the twin's log-probability features; weights by the same rule); scale = the first of alpha* x m_z0 x (0.5, 0.75, 1, 1.5, 2, 3) whose nested spend reaches DB_PTC's |
| DB_SHUF | control (reported beside D2) | DB_PTC's weights and strata with a keyed six-field-hash donor's propensity; scale = the first of alpha* x (0.5, 0.75, 1, 1.5, 2, 3) reaching DB_PTC's nested spend |
| DB_PTC_OCC | ablation | occupation tilt at alpha*, income = twin |

D2 reads DB_PTC against TWIN, the SBC path [TWIN, DB_SBC_LO, DB_SBC_HI] and the SPEND path [TWIN, DB_SPD_LO, DB_SPD_HI] (RULES.md 3).
**Declared readings beside D2** (descriptive): DB_PTC not resolved below DB_ZTILT (aggregate pair benefit - 2 se_agg <= 0) = PRC's
failure repeated (release-only information reproduces the gain); resolved below = the gain needs propensity information beyond the
release, which carries no removal guarantee. The data-rich reader check (`ddr.richreaders`): a D2 pass whose data-rich benefit over
DB_ZTILT or over a bracketing control is <= 0 in either version mean is labelled READER_BUDGET_DEPENDENT and is never called information
removal. MODEL_FIT cross-fitted readers on the nested releases are descriptive.

## 6. Revisions (at most two; each a new card committed before its fit)

- **R1, recipient-split strengths.** Trigger (MODEL_FIT selection, both versions): the binding S3 component belongs to one recipient
  while every component of the other has at least 40% slack. Revision: (alpha_I, alpha_O) on the product grid by the same rule
  (maximize alpha_I + alpha_O; ties to the smaller maximum). Reuses the SPEND path; refits the SBC path.
- **R2, joint-decision strata.** Trigger (after the grid, both versions): the pair benefit over the SBC bracket is below half the
  larger single-view benefit (a MODEL_SELECT outcome: ADAPTIVE). Revision: occupation weights stratified by (income decision x
  occupation decision), income weights by the occupation decision, fallback to the occupation stratum below 200 people.
- Stop: the benefit over the SBC bracket not positive in both versions, or the aggregate below 1 SE (designer's F1): the family closes
  without a revision; also two revisions used.

## 7. Registered predictions (registered grid; benefit = comparator AUC - DB_PTC AUC)

- Designer B: twin G1-feasible with pair 0.645-0.665 (p 0.8); alpha* in {0.35, 0.5}, binding income Brier / NLL then the men's guard
  (p 0.7); DB_PTC pair 0.012-0.030 below the twin (p 0.65); vs the SBC bracket +0.008-0.025 (positive in both versions p 0.75; resolved
  p 0.5); neither view worse than the twin's (p 0.9); the 0.020 margin over NEURAL_RAW / H_LJ not met, pair 0.620-0.645 (p 0.9); DB_PTC
  meets G1 (p 0.55).
- Challenger: benefit over a fair SBC +0.000-0.012, resolved p 0.25; the gain erodes with reader data.
- Coordinator: vs ZTILT +0.004 [-0.006, +0.014] (resolved p 0.2); D2 = IMPROVEMENT_BEYOND_MATCHED_CONTROLS p 0.12; READER_BUDGET_DEPENDENT
  if D2 passes p 0.6; D3 p 0.03; D4 p 0.01.

Configurations: 8 x 2 = 16 at most (DB_PTC, DB_SBC_LO, DB_SBC_HI, DB_SPD_LO, DB_SPD_HI, DB_ZTILT, DB_SHUF, DB_PTC_OCC). CPU about 0.2-0.4 h.

## 8. Novelty

NOT_CLAIMED. Closest published work: AttriGuard (Jia and Gong 2018; perturbing released vectors against attribute inference), LEACE
(Belrose et al. 2023; first-moment erasure), multiaccuracy-style use of a protected-attribute proxy (Kim, Ghorbani and Zou 2019),
exponential-tilt posterior adjustment (Saerens et al. 2002).
