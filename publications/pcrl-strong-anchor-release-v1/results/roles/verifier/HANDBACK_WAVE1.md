# Independent verifier hand-back, wave 1 (saved verbatim by the coordinator, about 06:21Z on 2026-10-24)

Wave 1 is done: **132 checks, 129 PASS, 3 FAIL.** None of the three failures changes a label in any reading. The grouped entries cover 8,808 elementary MODEL_SELECT comparisons, all PASS. Finished 06:19:51Z (`date -u`).

**Label reads and CPU.** Two queue units ran one at a time, both with my own code, both logged as operation 'verify':
- **SAR_VERIFY_1** (python3, 14.2 CPU-s): MODEL_SELECT income, occupation and SEX.
- **SAR_VERIFY_2** (torch venv, 3.8 CPU-s): MODEL_SELECT SEX, plus MODEL_FIT SEX for p_men.

Label-free scripts and inspection outside the queue add under 15 CPU-s, so about 33 CPU-s used of the 2,400 cap. I did not charge that remainder to the ledger.

**What passed**
- **Check 1, utility:** recomputed for FB_PFO, FB_PFO_TASK, FB_R1_CTRL_HI, FB_R1_ALT_HI, FA_TFRO_R1, FA_TFRO_R1_CONT, FA_R1_CTRL_LO and all 11 reference entries (NEURAL_RAW, H_LJ, SUCC, ...).
  - All 28 conditions with their SEs, all recorded paired costs, and the evaluated wire hashes match.
  - Largest difference is 2.3e-16 against a 1e-9 tolerance.
- **Check 2, selections:** all PASS.
  - FA_REG: rho 0.3 fails part (a) (income Brier; in v1 also income NLL and occupation accuracy) and part (b) (both guards) in both versions.
  - FB_REG: v0 k* = E1; v1 E1 fails income Brier (bound 0.00383 > 0.0025).
  - FB_R1: k* = S50 in both versions; S60 fails the women's guard.
  - FB_R1 brackets: CTRL HI = DQ 0.417725 / 0.37975; ALT HI = S160 / S130.
  - FA_R1: rho* = 1.0 / 0.6; the guard estimates re-derived exactly; CTRL LO s = 0.2 / 0.3, HI none / the origin.
  - The budgets equal SUCC's MODEL_SELECT margins as I recomputed them, and p_men = 0.5103 is confirmed on MODEL_FIT.
- **Check 3, grid:** I covered every cell, member and view of SAR1_FB_R1 and SAR1_FA_R1, not a sample.
  - Winner re-picks, SCORE AUCs and the 400 bootstrap draws, for registered and non-table winners, are all bitwise equal to the records.
  - Paired cells and their SEs show 0.0 difference.
  - All aggregates and labels for D2, D3, D4 and UM reproduce to 1e-12.
  - No seed-dependent recipe gives equal k0 and k1 scores on any minimum-set wire.
- **Check 4, refits:** 6 of 6 bitwise. The sample is seed 20261024, indices 12, 202, 293, 329, 451, 478, drawn from 600 registered-cell winners excluding U0 / Ucal.
- **Check 5, outcomes:** every label reproduces.
  - FA_REG / FB_REG: NO_FEASIBLE_STRENGTH, class TASK_INFEASIBLE.
  - FB_R1, D1 and UM: TASK_FEASIBLE and MARGIN_NOT_RESOLVED.
  - FB_R1, D2: PARTIAL_START+CONT_ONLY.
    - CTRL has no bracket: CAND − Q total-NLL is +0.003836 vs NEURAL_RAW and +0.001416 vs CTRL_HI.
    - ALT Q* = ALT_HI at −0.000754; benefit there is +0.00115 (se 0.00744), point only.
  - FB_R1, D3–class: D3 NOT_SUPERIOR (NEURAL_RAW only), D4 NOT_MET, class FEASIBLE_NO_RELIABLE_GAIN.
  - FA_R1: CAND_INFEASIBLE (v1 men's guard +0.009943, se 0.002016, against 0.005, a violation at 2 SE); v0 meets all 14 conditions; class TASK_INFEASIBLE.
- **Check 6, access:** all PASS apart from the plan-hash entries below.
  - First label read at 05:11:51Z, after the receipt (verified 05:11:39Z).
  - Every fitting read comes after its card's commit, and the cards are unchanged since.
  - No MODEL_SELECT read from a fitting module, no closed role, and no SEX read on CALIBRATION.
  - Every read is inside its unit.
  - The plans were committed at 05:38:20Z (FB_R1) and 05:45:00Z (FA_R1), before those readings' first reads at 05:38:26Z and 05:45:05Z.

**The three failures (kept as failures, not relaxed)**
1. **S6_FB_R1_plan_hash_recorded:** PROTOCOL 6 requires the plan's hash in the decision record. `claims/FB_R1/DECISIONS.json` has only `"plan": "FB_R1"`; the only SHA-256 in it is the rules hash. The missing hash is 349a6f6efbef305f015d0e9548e68daf7d0f918890a73c3aff11bc4fcaba5a45.
2. **S6_FA_R1_plan_hash_recorded:** same gap in `claims/FA_R1/DECISIONS.json`. The missing hash is 5a78be990f937bbeb20e7746120784fae28e1e97b4508c5e934017e2901151ab.
3. **S1_FB_R1_income_bitwise_START:** card FB_PFO_R1 item 1 says CAND's income equals START's bitwise, but on MODEL_SELECT it does not.
   - FB_PFO's income differs from the frozen NEURAL_RAW reference release by up to 6.47e-8 (v0) and 6.84e-8 (v1). Decisions are identical, but the wire hashes differ (53b756a3… vs f4de1c9f…).
   - The likely cause is that the family recomputes NEURAL_RAW's income network in its own float64 numpy forward pass.
   - No effect on any decision: income utility differences are below 1e-9, and every income-view grid cell is exactly 0, so the rules read it as ALIAS_ZERO even though the wires are not exact aliases.

**Not failures, but worth knowing**
- **Fineness flags:** D2's CONTROL_POINT_RULE_NOT_FOLLOWED flags for CTRL and ALT reproduce. With only HI saved, the lower point is the origin, while the full grid holds a larger qualifying LO, so the frozen rule reads SAVED_POINT_RULE_NOT_FOLLOWED. PROTOCOL 10 expected COARSE_BELOW for this case. The flag never gates.
- **Untracked REG records:** the git snapshot at session start showed `claims/FA_REG/` and `claims/FB_REG/` untracked. I did not check whether they have been committed since.
- **Ledger drift:** my folder is in the queue's code manifest, so my edit to `v_assemble.py` was recorded as `code_changed_during_run` in one of your recent units. Your code does not import mine, but you may want to note it.

**What I could not check**
- **No git run:** I took commit times from the worktree reflog file read as a file, and identified the plan commits by the hash prefixes in your commit log. I did not read commit trees.
- **Fit-time spends:** I did not recompute the MODEL_FIT and CALIBRATION spends, their SEs or the λ = 0 path spends. I re-applied each rule to the recorded values; only p_men was recomputed.
- **Taken from the record:** the TABLE_CV ("enhanced") winner AUCs, whose scores are not cached and which feed only a reported flag, and the reader-budget flags.
- **Grid validity:** I did not re-derive the full pass (2,880 seed checks and 16 explained equalities off the minimum set) beyond the minimum-set seed check.
- **D2R:** the FB_R1 record reads NOT_RUN (the robustness bank was still running), so it waits for the next wave.
- **Deployment and descriptive claims:** packaging equivalence and descriptive claims such as "lowest pair so far" are not checked.

**Files**
- results/pcrl_strong_anchor_release_v1/roles/verifier/CHECKS.json
- results/pcrl_strong_anchor_release_v1/roles/verifier/VERIFICATION_REPORT.md
- results/pcrl_strong_anchor_release_v1/roles/verifier/work/ (ms_recomputed.json, decide.json, refit.json)
- results/pcrl_strong_anchor_release_v1/roles/verifier/code/ (vlib.py, v_ms.py, v_refit.py, v_decide.py, v_assemble.py)
