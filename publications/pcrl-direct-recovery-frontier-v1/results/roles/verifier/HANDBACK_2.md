# Verifier hand-back 2 (verbatim; received by the coordinator 2026-10-25, saved 04:33Z)

D2R hand-back (04:32Z). My D2R for the R1 reading matches `claims/DRF_R1/D2R.json`: 434 of 434 items pass. I computed it before the record existed, then compared. The clean check-1 re-run on committed code also passes and reproduces the first run exactly.

**D2R (check 3e, label-free).** It uses the registered grid DRF1_R1 plus the robustness grid DRF1_R1_ROB, with 24 robustness cells per comparator (the same set the record uses).
- Result: ROBUSTNESS_NOT_CONFIRMED.
  - Against the CTRL bracket DR_CTRL_HI_R1 it is confirmed: registered +0.0340, robustness +0.0308, AGG_POSITIVE on the robustness cells and on all cells, both robustness recipient gates resolved.
  - Against DR_MI_R1 it is not confirmed: robustness +0.0054, point-only.
- Beside: TWB_C64 is confirmed (+0.0388). NEURAL_RAW (+0.0197) and H_LJ (+0.0123) are point-only. RETIRED_B3CTL is not positive and fails the robustness income gate.
- Headline caveats: SUCC changes sign (+0.0008 to −0.0002) and fails the robustness income gate; RETIRED_B3CTL also fails it.
- Refit: REFIT_GENUINELY_EFFECTIVE.
  - My R3/R4 assignment hashes are distinct and equal READER_SPLITS.json, and the SCORE overlaps are recorded and lie in [0, 1).
  - All 234 seed equalities in 6,144 seed checks are explained.
- The robustness grid's registered cells equal the main grid's (AUCs and SEs), and every per-comparator mean and SE agrees with the record within 1e-12.
- One cosmetic record issue: the record's `beside` list names SUCC twice. As a result `sign_changes`, `recipient_failures` and the headline caveat text repeat SUCC ("against SUCC, SUCC"). The sets equal mine.

**Check 1 clean re-run (DRF_VER_C1_V3).**
- Registered at ffb524dcc with every verifier file committed. `vlib.py` and `check1_finite_law.py` did not change during the run; the only drift was `check3_readers.py`, which check 1 doesn't import.
- v1, R1 and R2, both versions: 50,969 items pass.
- It reproduces DRF_VER_C1_V1 exactly (check 1r: same item names, values and results), so the earlier `vlib.py` drift had no effect.
- R2: for R1's four seeds, R2's per-seed best objectives equal R1's exactly, and R2's best objective is no worse than R1's, for both DR_AUC_R2 and DR_MI_R2.

**Check 2 with R2 (DRF_VER_C2_V2): 8,118 items pass.**
- DR_AUC_R2 and DR_MI_R2 pass all 28 conditions.
- DR_CTRL_LO_R2 fails the men's guard in both versions; DR_CTRL_HI_R2 fails the men's guard and occupation NLL.

**R2 labels, computed with the R2 decision record never opened (`--no-record`).** They are saved in `out/OUTOFQUEUE_20261025T0431Z_check3_DRF_R2_pre_record.json`.
- The reading is ADAPTIVE_R2 (MODEL_SELECT trigger, card committed after the first MODEL_SELECT read), so every label should carry the prefix `ADAPTIVE_R2__`.
- D1 TASK_FEASIBLE.
- D2 NOT_SEPARATED:
  - the CTRL frontier reads SEPARATED_ON_PRIMARY_COST_ONLY, with bracket DR_CTRL_HI_R2;
  - the OBJ spend state is OBJ_MI_SPENDS_AT_LEAST_AS_MUCH.
- FC FAMILY_CHANGE_NOT_RESOLVED, TWIN BELOW_TWIN_RESOLVED.
- D3 NOT_SUPERIOR: NEURAL_RAW and TWJ pass before the reader-budget credit rule is applied.
- D4 NOT_MET, STAGE_3, class FEASIBLE_NO_RELIABLE_GAIN.

**Disclosures.**
- The first start of check 3e stopped at `import drf` (the repository root wasn't on its path) before writing anything; I fixed it and re-ran.
- I changed `check3_readers.py`, `check4_refits.py` and `check2_task_metrics.py`, and added `check3e_d2r.py`, `check6_deploy.py` and `check1r_reproduce.py`.
- I charged VERIFIER_OUT_OF_QUEUE_3 (25 CPU-s).

**Next, when you resume me.**
- R2: compare my labels with DECISIONS.json (prefixed and unprefixed), then grid validity, confidence and reserve, and refits as a queue unit.
- Check the deployment packages once DRF_DEPLOY_CHECK_V1 is done.
- Run the final check 5 after closeout.
- Please commit the changed and new verifier files, `CHECKS.json` (135 records), `VERIFICATION_REPORT.md` (section 6 added) and `out/`.

Everything is in `results/pcrl_direct_recovery_frontier_v1/roles/verifier/`.
