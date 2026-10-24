# Independent verifier hand-back, wave 2 (saved verbatim by the coordinator, about 06:51Z on 2026-10-24; this header first said 06:56Z, an estimate written before reading `date -u`, corrected at 06:51:33Z by `date -u`)

Wave 2 is done: all five items checked, **124 new checks, 124 PASS, 0 FAIL.** The grouped entries cover 17,404 elementary MODEL_SELECT comparisons, none failed. Across both waves it is **256 checks, 253 PASS, 3 FAIL**; the 3 are wave 1's and stay failed. Finished 06:49:52Z (`date -u`).

**CPU.** Queue units used 57.9 CPU-s across both waves:

| Unit | Interpreter | CPU-s |
|---|---|---|
| SAR_VERIFY_1 (wave 1) | python3 | 14.2 |
| SAR_VERIFY_2 (wave 1) | torch venv | 3.8 |
| SAR_VERIFY_3 | python3 | 20.3 |
| SAR_VERIFY_4 | torch venv | 2.3 |
| SAR_VERIFY_5 | python3 | 12.7 |
| SAR_VERIFY_6 | torch venv | 4.7 |

Work outside the queue (label-free scripts and inspection) is at most about 30 CPU-s for both waves. Total is under 90 of the 2,400 CPU-s cap.

**Code edits.** None of my units ran while I edited code; every verifier unit records an empty `code_changed_during_run`. Two of your units did pick up my new files as drift; neither imports my folder, but you may want to note them under deviation 11:
- SAR_GRID_SAR1_FA_R2_ROB_V1 recorded `v2_ms.py` and `v2_refit.py` (created about 06:36–06:37Z);
- SAR_CUSTODY_REHEARSAL_V1 recorded `v2_assemble.py` (06:47Z).

**1. FA_R2: all checks pass.**
- **Utility:** FA_TFRB_R2, both CTRL points and the reused CONT record match, including every paired cost; largest difference 2.8e-16. The v1 CTRL_HI is the frozen SUCC bitwise, and the candidate's income is SUCC's bitwise.
- **Selection:** I re-derived rho* = 0.6 / 1.0. The between-fold SE of the contrast is the binding one: 0.00339 vs a household SE of 0.00137 (v0), and 0.00250 vs 0.00134 (v1). The budgets equal SUCC's recomputed margins. The scan stops at rho 1.0 in v0 (men's bound 0.000636 > 0.000583) and at rho 2.0 in v1.
- **Guard contrast (yes, I did recompute it):** at rho*, from the saved bag in MODEL.pkl plus MODEL_FIT labels, without importing sar.fa. It is bitwise equal to the record: C = −0.000735 / +0.000289, with the five per-fold contrasts and the between-fold SE. The MODEL_FIT contrast inputs also equal FA_R1's at every common rho, so the bag reuses R1's fold fits.
- **CTRL bracket:** v0 LO s 0.4 and HI s 0.1 (both new points); v1 LO s 0.25 and HI s 0 (the origin). Roles and reuse match.
- **Grid:** every cell, member and view reproduces with difference 0.0.
- **Refits:** 6 of 6 bitwise, drawn with seed 20261025 (indices 44, 49, 54, 97, 193, 195 of 312). No seed equality on the minimum set.
- **Decisions:** every label reproduces.
  - D1 TASK_FEASIBLE.
  - UM MARGIN_GAIN_PROFILE_NOT_RETAINED: the men's-guard improvement is resolved (pooled −0.00285, se 0.00133), but the v0 occupation view is +0.0252.
  - D2 PARTIAL_CONT_ONLY: START is point-only and its occupation gate fails; CONT passes; at the CTRL bracket FA_R2_CTRL_HI the benefit is +0.00663 (se 0.0140), not resolved.
  - D3 NOT_SUPERIOR (passes W, TR100 and TT100 only).
  - D4 MATERIAL_0020_POINT_ONLY (+0.0241 vs NEURAL_RAW, +0.0208 vs H_LJ, not resolved).
  - Class FEASIBLE_NO_RELIABLE_GAIN.
- **Access and ordering:** plan_sha256 matches (054a8b4b…). The plan was written at 06:13:49Z and committed (a85bd035a) at 06:14:04Z, before the first reading read at 06:14:17Z. The card (d3b453d1d) was committed at 06:08:14Z, before the fits.

**2. FS_CPI: confirmed no candidate.**
- **v0:** lambda 0.025 fails income accuracy (bound 0.00519 > 0.0040), so NO_FEASIBLE_STRENGTH.
- **v1:** lambda* = 0.05.
- **Labels:** all reproduce, class TASK_INFEASIBLE.
- **No MODEL_SELECT work:** v0 saved nothing, and no MODEL_SELECT record, grid, role or label read touches any FS_CPI release.
- **Not recomputed:** the CALIBRATION spends themselves, which would need refitting the residual and START_CF. I only re-applied the rule to the recorded spends.

**3. FB_R1 D2R: ROBUSTNESS_NOT_CONFIRMED reproduces**, from my own recomputation of all 32 cells (24 robustness cells):
- NEURAL_RAW: robustness +0.01187 (se 0.00674), not resolved.
- FB_PFO_TASK: +0.01405 (se 0.00643), confirmed.
- ALT: +0.00115 registered → −0.00118 on the robustness cells (se 0.00701); the sign change is reproduced.
- The n_eff counts, seed qualifiers and seeds all match, and the robustness bank's registered cells equal the main bank's.
- The 2 robustness-cell refits are bitwise.

**4. FA_R2 D2R** (the lane had finished at 06:38:00Z, before I read it): ROBUSTNESS_NOT_CONFIRMED with no sign change reproduces.
- SUCC: +0.00341 (se 0.01393), not confirmed.
- CONT: +0.03741 (se 0.01556), confirmed.
- CTRL_HI: +0.00404 (se 0.01340), not confirmed.

**5. Housekeeping.**
- **Sidecars:** all five PLAN_SHA256.json files equal the SHA-256 of their plan and decision files.
- **FA_REG / FB_REG:** both are tracked in this worktree's git index with identical content. I read the index as a file and did not run git. Because I did not read HEAD's tree, "committed" is supported but not proven.
- **Access:** all 52 label reads logged so far pass the access rules.

**FAILs, all kept from wave 1**
1. **S6_FB_R1_plan_hash_recorded** and **S6_FA_R1_plan_hash_recorded:** the plan hash is missing from the decision records. The separate wave-2 entries W2_S6_FB_R1_plan_sidecar and W2_S6_FA_R1_plan_sidecar confirm that the sidecars hold the correct hashes.
2. **S1_FB_R1_income_bitwise_START:** a separate entry, W2_ADJ_S1_FB_R1_income_bitwise_START, confirms that deviation 10's bound of 7e-8 holds against my measured 6.84e-8.

**Still not checked**
- The household SE of FA_R2's contrast, and the contrasts at the stopping rho (those fold models are not saved).
- Any other fit-time CALIBRATION or MODEL_FIT spend.
- FS_CPI v1's saved releases.
- The full grid-validity pass, beyond the minimum-set seed checks.
- The deployment and custody units.

Files are in <repository>/.claude/worktrees/pcrl-strong-anchor-release-v1/results/pcrl_strong_anchor_release_v1/roles/verifier/:
- CHECKS.json — wave-2 entries appended with `"wave": 2`; wave-1 entries unchanged
- VERIFICATION_REPORT.md — "Wave 2" section appended
- work/ — ms2_fa_r2_fb_rob.json, ms2_fa_r2_rob.json, decide2.json, refit2.json, contrast2.json
- code/ — v2_ms.py, v2_refit.py, v2_contrast.py, v2_decide.py, v2_assemble.py
