# Investigator A hand-back for revision R2 (saved verbatim by the coordinator, about 06:08Z on 2026-10-24)

## FA_TFRB_R2: deploy the cross-fit bag the guard is measured on

I propose one second revision, **FA_TFRB_R2** (ADAPTIVE_R2). It deploys the average of the five fold models whose guard contrast is measured, instead of a separate full fit, and widens the guard's SE to cover run-to-run variance. The R2 selftest passes all 19 checks (48.2 CPU-s), and the preflight passes in both versions. No label file was opened, nothing new was computed from MODEL_SELECT, no git was run and no shared code was changed.

### Diagnosis (MS facts are the ones you gave plus the committed `claims/FA_R1/DECISIONS.json`, used as adaptive)
- **The failure is in the guard's men-minus-women contrast, not its level.**
  - Over SUCC on MS, the contrast is −0.0052 (v0) and +0.0145 (v1). The overall change is about −0.0011 / −0.0012.
  - The CALIBRATION level L predicted −0.0025 / −0.0039 (se 0.0017): within 0.8 / 1.6 SE.
  - The MODEL_FIT contrast C predicted −0.0017 / −0.0001 (se 0.0014): a 10-SE miss in v1.
- **C was measured on the five fold models, but a sixth run (the full fit) was deployed.**
  - The fold-averaged contrasts were near zero at every ρ: v1 +0.0016 / −0.0001 / +0.0003; v0 +0.0003 to −0.0017.
  - The deployed runs' MS contrasts have opposite signs, and FA_TFR_v1's fold contrasts also flipped sign across versions.
  - This fits a run-specific asymmetry: which SEX group pays the critic cost changes between training runs. An alternative is that SUCC's in-sample protection on MODEL_FIT changes how the residual behaves on new people; that alone does not explain opposite signs with near-zero fold contrasts, but it is not excluded.
- **Label-free checks** (no labels and no SEX are needed at λ = 0):
  - λ = 0 runs are nearly unique: the bag and the full fit differ by a mean KL of 4-7e-5, against a movement of 0.022-0.027.
  - R1's λ = 0 full fit reproduces bitwise, compared on trained weights, against FA_TFRO_R1_CONT and FA_TFRO_R1_TASK_LO in both versions.
- **In R1 v0 the mechanism did what FA asked**, so R2 keeps it and changes only what is deployed and how the guard uncertainty is formed.

### The mechanism change
- **Release:** occupation = softmax(centre(z_S + mean over k of r_k(φ))).
  - The r_k are R1's occupation-only residuals, fitted on MODEL_FIT minus fold k with R1's recipe and fold seeds.
  - Income is SUCC's bitwise.
  - There is no full fit at λ > 0.
- **Why the deployed object now matches the estimate:** each MODEL_FIT person is scored by the one model that did not see them, so the out-of-fold contrast is a first-order estimate of the deployed bag's contrast. Averaging also shrinks run-specific asymmetry.

### The rule
- **Kept from R1:**
  - budgets = SUCC's margins to the caps from the anchor table (men's guard 0.00058 / 0.00106);
  - part (a) on CALIBRATION;
  - the guard estimate Δ = L ± weight·C;
  - the ρ grid {0.3, …, 10}, scanned upward, stopping at the first failure.
- **Two changes:**
  - L and part (a) are measured on the deployed bag;
  - SE_C = max(household-clustered SE, SD of the five per-fold contrasts / √5), so the SE now includes run-to-run variance.
- **Remaining biases, stated in the card:**
  - any SEX-asymmetric in-sample term in C (in v0, C had the right sign but a third of the MS size);
  - the between-fold SD has only 4 degrees of freedom;
  - the estimate is first-order only;
  - CALIBRATION's SEX mix is assumed equal to MODEL_FIT's;
  - MS evaluation noise is not covered.

### Controls
- **CTRL:** R1's λ = 0 path (origin START, s = 0 is SUCC exactly), bracketed on CALIBRATION total NLL against R2's CAND.
  - Before any save, the fit checks that its recomputed λ = 0 fit matches R1's saved models bitwise. If not, it stops with REUSE_NOT_BITWISE and saves nothing.
  - Points that land on START (s = 0), FA_TFRO_R1_TASK_LO (s = 0.2 in v0, 0.3 in v1) or FA_TFRO_R1_CONT (s = 1) are reused at no new specification.
  - A full-fit λ = 0 path is a matched control for a bagged CAND because λ = 0 runs are nearly unique.
- **CONT:** FA_TFRO_R1_CONT, reused and gating.
- **Specifications:** at most 6 (CAND 2 + at most 4 new CTRL points); fewer when bracket points are reused.

### Predictions (registered in the card)
| Quantity | Prediction |
|---|---|
| ρ\* | 0.3: 0.40; 0.6: 0.20; 1: 0.10; ≥ 2: 0.05; none: 0.25 |
| MS men's guard ≤ 0.005 in both versions | 0.75 |
| All 28 conditions met at the point | 0.70 |
| MS contrast within ±0.006 | 0.65 |
| Pair AUC | 0.615 (80% interval 0.600-0.630) |
| Benefit over SUCC | −0.002 |
| Resolved benefit over both CTRL points | 0.08 |
| D3 / D4 | 0.02 / 0.01 |

I expect R2 to make the men's guard robust at the cost of the run-specific part of R1's protection. If G1 fails again, the asymmetry is systematic, and I would conclude that no MODEL_FIT/CALIBRATION estimate of the SEX contrast is reliable for a SUCC-anchored release.

### Built and tested
- `sar/fa/tfrb.py`: a new module, with tfr.py and tfro.py untouched. It writes `fa/FIT_R2_V<v>.json` with the per-fold contrasts, both SEs, the bracket, the reuse check and R1's reused points.
- `sar/fa/__init__.py`: routes `FA_TFRB_R2*` ids to tfrb.py; earlier ids keep their modules.
- **Selftest coverage:**
  - the bag algebra and deployment through the package router, both bitwise;
  - income bitwise SUCC, and s = 0 returns START exactly;
  - the between-fold SE takes over when invented fold asymmetries vary;
  - non-bitwise reuse is refused with nothing saved, and the scan stops at the first failure;
  - CONT is reused, not saved;
  - new CTRL points equal R1's λ = 0 fit and deploy bitwise.
- **CPU:** about 104 of the 300 design CPU-s used. The estimate for the real fit is about 3-5 CPU-min per version.

Files are in `<repository>/.claude/worktrees/pcrl-strong-anchor-release-v1/`:
- `sar/fa/tfrb.py`
- `sar/fa/__init__.py`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/CARD_DRAFT_R2.md`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/selftest_R2_output.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/preflight_R2_V0.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/preflight_R2_V1.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/r2_check_lambda0_labelfree.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/r2_check_reuse_weights.json`
- the scripts that produced the two check files, plus `r2_read_records.py` and `r2_read_guards.py`
