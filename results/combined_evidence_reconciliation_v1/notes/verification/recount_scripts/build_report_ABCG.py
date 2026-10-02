#!/usr/bin/env python3
"""Assemble report items for checks A, B, C and G (PCRL half) from recount outputs + fixture outputs.
Writes ../report_parts/ABCG_pcrl.json. Standalone; reads only files written by the recount scripts/fixtures.
"""
import hashlib
import json
from pathlib import Path

V = Path(__file__).resolve().parent.parent
FX = V.parent.parent / "fixtures"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def J(name):
    return json.load(open(V / "recount_outputs" / name))


def ins(out_name, extra=()):
    d = J(out_name)
    return d["inputs"] + [{"location": str(V / "recount_outputs" / out_name), "sha256": sha(V / "recount_outputs" / out_name)}] + list(extra)


A, B, C = J("A_checkpoint_selection.json"), J("B_cross_purpose.json"), J("C_erase_vicreg.json")
CF = json.load(open(FX / "outputs" / "C_scale_invariance.json"))
GF = json.load(open(FX / "outputs" / "G_accuracy_guarantee.json"))
t = A["headline_grid_totals"]
pd = A["per_dataset"]
cmdA = "/opt/homebrew/bin/python3 notes/verification/recount_scripts/A_checkpoint_selection.py origin/main > notes/verification/recount_outputs/A_checkpoint_selection.json"
cmdB = "/opt/homebrew/bin/python3 notes/verification/recount_scripts/B_cross_purpose.py > notes/verification/recount_outputs/B_cross_purpose.json"
cmdC = "/opt/homebrew/bin/python3 notes/verification/recount_scripts/C_erase_vicreg.py > notes/verification/recount_outputs/C_erase_vicreg.json"
CKPT_NOTE = [{"location": "/Volumes/YOTUO/NathanSamson-Mac-relocated-2026-09-30/archives/fl-PCRL-main-checkpoints.tar::checkpoints/v2_adult_ROUND5_s0/final.pt",
              "sha256": "fb80c804eca8baa86b524ffce50ffe72aeb75252a85a239020ec119e7283a6e9"},
             {"location": "/Volumes/YOTUO/NathanSamson-Mac-relocated-2026-09-30/archives/fl-PCRL-main-checkpoints.tar::checkpoints/v2_adult_ROUND5_s0/best.pt",
              "sha256": "1d691fc69b62bf1c49fa59737c452cdb3b75c42dce48db299f92732445720c6f"}]
items = []
items.append({
    "id": "A1", "question": "Strict R2<=0.05 count on the NeurIPS headline grid (Adult R5 + HMDA R5 + Diabetes R7, 60 cells) under the FINAL-ITERATE rule (final.pt, epoch 199)",
    "inputs": ins("A_checkpoint_selection.json"), "command": cmdA,
    "result": {"final_rule_strict_le_0.05": f"{t['final_le']}/60", "per_dataset": {k: pd[k]["final_rule"]["strict_le"] for k in ["adult", "hmda", "diabetes"]},
               "same_under_lt": t["final_lt"] == t["final_le"],
               "final_source": "dominant_axis_audit.json r2_onehot (scripts/eval_round4_dominant_axis.py:137 loads final.pt); equals final_vs_best.json 'final' rows exactly (max |diff| 0.0) for Adult/HMDA"},
    "status": "independently recomputed", "scope": "in-sample OLS one-hot R2 fit and scored on TEST-split representations; 3 seeds; final.pt",
    "previous_assessment_value": "56/60 final.pt (23/24, 16/18, 17/18)",
    "corrected_value_or_confirmation": "Confirmed 56/60 = 23+16+17 under final-iterate rule. This is the rule the paper states (PDF(23) l.454-456 'We report the final iterate at Kw+Kc'; Table 4 'final iterate (epoch Kw+Kc)')."})
items.append({
    "id": "A2", "question": "Strict R2<=0.05 count on the same 60 cells under the BEST-VALIDATION / Cotter-selector rule (best.pt)",
    "inputs": ins("A_checkpoint_selection.json"), "command": cmdA,
    "result": {"best_rule_strict_le_0.05": f"{t['best_le']}/60", "per_dataset": {k: pd[k]["best_rule"]["strict_le"] for k in ["adult", "hmda", "diabetes"]},
               "flips_final_vs_best": {k: pd[k].get("flips_final_vs_best") for k in ["adult", "hmda", "diabetes"]},
               "cotter_kind_per_seed": {k: pd[k]["cotter_kind"] for k in ["adult", "hmda", "diabetes"]},
               "n_feasible_post_warmup": {k: pd[k]["n_feasible_post_warmup"] for k in ["adult", "hmda", "diabetes"]},
               "diabetes_R5_for_attribution": {"final": pd["diabetes_R5"]["final_rule"]["strict_le"], "best": pd["diabetes_R5"]["best_rule"]["strict_le"]},
               "best_source": "per_seed_results.json linear_r2 (run_v2_dataset.py@b158abc54:283-292 reloads best.pt before generate_report); equals final_vs_best.json 'best' rows within 7e-4"},
    "status": "independently recomputed", "scope": "same metric/split as A1; best.pt",
    "previous_assessment_value": "54/60 best.pt (21/24,16/18,17/18); 'Cotter fallback' implied for all seeds",
    "corrected_value_or_confirmation": "Confirmed 54/60 = 21+16+17. The two Adult flips are s0 employment_analysis/age_group (best 0.062 vs final 0.005) and /marital_status (0.081 vs 0.003). Minor correction: the Cotter selector was 'feasible' (not fallback) for Adult s1 (23 feasible epochs) and HMDA s1/s2 (147/58); Diabetes R7 is fallback on all seeds (0 feasible). Diabetes R5 (the run the lambda-floor/warmup fix actually applies to) is 16/18 final vs 18/18 best."})
mc = {k: pd[k]["mixed_rule_final_r2_x_best_health"]["clean_cells"] for k in ["adult", "hmda", "diabetes"]}
bc = {k: pd[k]["best_rule"]["clean_cells"] for k in ["adult", "hmda", "diabetes"]}
items.append({
    "id": "A3", "question": "'Cleanly compliant' (R2<=0.05 AND per_dim_std_mean>=0.5 AND eff_rank>=2, own purpose) per checkpoint rule; provenance of the paper's 7/60",
    "inputs": ins("A_checkpoint_selection.json", CKPT_NOTE), "command": cmdA + "  # code read: origin/main:scripts/identify_collapse_cells.py:57-131; origin/main:scripts/run_splince_benchmark.py:296-311,355-356",
    "result": {"paper_7_60_rule": "MIXED: final.pt R2 (dominant_axis_audit.json) x best.pt health (summary.json health_notes)",
               "mixed_rule": f"{t['mixed_clean']} clean / {t['mixed_collapse']} collapse / {t['mixed_failed']} failed",
               "best_rule_consistent": f"{t['best_clean']} clean / {t['best_collapse']} collapse / {t['best_failed']} failed",
               "final_rule_consistent": f"{t['final_clean']} clean / {t['final_collapse']} collapse / {t['final_failed']} failed (final.pt R2 x final.pt health; health = pre-projection 'pre_health' stored by scripts/run_splince_benchmark.py, which loads final.pt, eval mode, test split)",
               "final_rule_clean_cells": A["final_rule_clean_cells"],
               "mixed_clean_cells": mc, "best_clean_cells": bc,
               "cells_clean_only_under_mixing": [[0, "employment_analysis", "age_group", "best.pt R2 0.0619, aud delta 0.130"],
                                                 [0, "employment_analysis", "marital_status", "best.pt R2 0.0806, aud delta 0.264"]]},
    "status": "independently recomputed", "scope": "health on TEST-split reps of the cell's own purpose; per_dim_std is population std (ddof=0)",
    "previous_assessment_value": "7/49/4 reproduces only by mixing; best.pt-only 5/49/6; 'No final.pt health metrics are stored'",
    "corrected_value_or_confirmation": "CORRECTED: three rules, three counts. Final-iterate (final.pt R2 x final.pt health) = 6/60 (HMDA s2 underwriting race+ethnicity; Diabetes s1 and s2 billing_audit race+gender; Adult 0/24). Best-validation (best.pt x best.pt) = 5/60. The paper's 7/60 mixes final.pt R2 with best.pt health; its 3 Adult s0 employment cells are clean under neither single-checkpoint rule, and 2 of them FAIL R2 at the checkpoint whose health was used (0.062, 0.081). The previous assessment said no final.pt health was stored; it is stored as SPLINCE 'pre_health' (finding surfaced by the D fork, recomputed here)."})
items.append({
    "id": "A4", "question": "Which rule does the paper's 56/60 use, and which does the erase-pilot baseline 54/60 use?",
    "inputs": ins("A_checkpoint_selection.json") + [i for i in C["inputs"] if "ERASE_PILOT" in i["location"]],
    "command": cmdA + " ; " + cmdC + "  # code read: erase-layer-pilot-2026-05-17@65dd5c050:experiments/run_v2_dataset.py:329-361 (canonical>best>final)",
    "result": {"paper_56_60": "final-iterate (final.pt)", "erase_pilot_baseline_54_60": "best.pt (R5/R7 per_seed_results.json, i.e. A2)",
               "erase_pilot_itself": "best.pt (Cotter fallback on 9/9 seeds, 0 feasible epochs)",
               "consequence": "54/60 -> 60/60 is a like-for-like best.pt comparison; quoting it against the 56/60 headline mixes rules."},
    "status": "checked against code and recorded aggregates", "scope": "checkpoint provenance only",
    "previous_assessment_value": "56/60 final; pilot baseline is best.pt 54/60",
    "corrected_value_or_confirmation": "Confirmed. Present as two counts with rules: final-iterate 56/60; best-validation 54/60 (and erase pilot best.pt 60/60)."})
m0, m1 = B["M0_submission_original_PCRL_finalpt"], B["M1_rebuttal_eraselayer_unionLEACE_crosspurp_constraint_bestpt"]
items.append({
    "id": "B1", "question": "Exact metric definitions and the submitted Sec. 5.5 counts 26/33 (absolute) and 22/33 (incremental)",
    "inputs": ins("B_cross_purpose.json"), "command": cmdB,
    "result": {"ABS_definition": "mean over 3 PCRL seeds of (concat_acc - majority_test) > 1pp; acc = max over auditor seeds {11,22,33} of test accuracy (auditors fit on encoder train split)",
               "INCR_definition": "mean over 3 PCRL seeds of (concat_acc - max over the 3 single-purpose h_p of acc) > 1pp (single-purpose side includes purposes where the attribute is allowed; best of 3 purposes x 3 auditor seeds)",
               "ABS": f"{m0['ABS_gt_1pp']}/33", "INCR": f"{m0['INCR_gt_1pp']}/33", "per_dataset": m0["per_dataset"],
               "single_recipient_best_single_minus_majority_gt_1pp": f"{m0['single_recipient_best_single_minus_majority_gt_1pp']}/33",
               "model": "original PCRL R5 (Adult, HMDA) / R7 (Diabetes) final.pt (checkpoint path recorded in every raw file)",
               "max_abs_diff_vs_aggregate_json_pp": m0["max_abs_diff_vs_aggregate_json_pp"]},
    "status": "independently recomputed", "scope": "33 = 11 (dataset, attribute) x {LR, MLP, XGB}; test split = the tuning split",
    "previous_assessment_value": "26/33 ABS, 22/33 INCR, 25/33 best-single minus majority",
    "corrected_value_or_confirmation": "Confirmed 26/33 and 22/33 from per-seed raw files; they are two criteria, not a contradiction. Table 10 (PDF(23)) values are INCR gains (e.g. Adult age_group XGB +9.28) although its caption says 'gain over majority' (ABS = +20.0)."})
items.append({
    "id": "B2", "question": "Rebuttal 'unified protocol' 22/33 -> 8/33 and the previous assessment's '19/33 absolute'",
    "inputs": ins("B_cross_purpose.json"), "command": cmdB,
    "result": {"model": "erase-layer architecture with ONE frozen joint-LEACE eraser fit on the UNION of all purposes' disallowed attributes (128-d), LoRA on repr_proj only, plus a training-time h_concat linear-R2 constraint (tau_cross=0.10), 200 epochs; best.pt (local results.json 'checkpoint': 'best.pt' all datasets)",
               "INCR": f"{m1['INCR_gt_1pp']}/33 (recorded aggregate flags {m1['INCR_recorded_aggregate_flags']})",
               "ABS": f"{m1['ABS_gt_1pp']}/33 (local per-seed results.json give {m1['local_results_json_ABS_gt_1pp']}/33; max |diff| {m1['max_abs_diff_unified_vs_local_abs_pp']} pp)",
               "per_dataset": m1["per_dataset"], "paired_vs_submission": B["M0_vs_M1"]},
    "status": "independently recomputed", "scope": "same auditor definitions; different model and checkpoint rule than B1",
    "previous_assessment_value": "8/33 INCR; 19/33 ABS; ABS rose on 20/22 nonlinear cells (19.4 -> 24.2pp)",
    "corrected_value_or_confirmation": "Confirmed: INCR 22/33 (original PCRL final.pt) -> 8/33 (erase-layer union-LEACE + h_concat constraint, best.pt); ABS 26/33 -> 19/33 for the same pair of models; all 11 LR cells go to exactly 0pp because the union eraser removes every disallowed attribute linearly from every purpose; on the 22 nonlinear cells absolute leakage ROSE on 20 (mean +19.4 -> +24.2pp). 22->8 compares different models under one criterion; it is not a measurement repair."})
items.append({
    "id": "B3", "question": "Was the incremental criterion registered before results, or chosen after?",
    "inputs": [{"location": "Bostesa/PCRL@135e440e6:experiments/run_cross_purpose_attack_v2.py (lines 22-24, 494)", "sha256": None},
               {"location": "Bostesa/PCRL@a3875c618:results/cross_purpose_laftr/DUAL_CRITERIA.json", "sha256": None},
               {"location": "Bostesa/PCRL@a3875c618:results/cross_purpose_laftr/HEADLINE.txt", "sha256": None},
               {"location": "Bostesa/PCRL@17ef7d449 (merge msg: 'keeps ... the 22->26 cross-purpose criterion fix')", "sha256": None},
               {"location": "Bostesa/PCRL@4390835a1:results/rebuttal/cross_purpose/PLAN.md (2026-05-19, lines 8, 39)", "sha256": None},
               {"location": "Bostesa/PCRL@d39211214:results/rebuttal/cross_purpose/PAPER_PASTE.md (2026-06-05, lines 84-86)", "sha256": None}],
    "command": "git -C /Users/nathansamson/PCRL log --all --format='%h %ad %s' --date=iso -- <paths>; git show <ref>:<path>",
    "result": {"submission": "INCR is the attack script's built-in verdict (PASS iff concat - best_single <= 1pp), first committed 2026-05-05 02:13 in 135e440e6 together with the results (no earlier history; no registration file). ABS first appears in DUAL_CRITERIA.json committed 2026-05-07 06:49 (a3875c618), the hour of the final PDF(23) build; source draft (21)/PDF(22) abstract said '22 of 33' (INCR), PDF(23) says '26 of 33' (ABS); 17ef7d449 calls this the '22->26 criterion fix'.",
               "rebuttal": "PLAN.md (2026-05-19, before the Phase D runs) cites '22-26 of 33' and the Adult pilot flag count 14/15 -> 12/15, which is the ABS criterion; the 2026-06-05 PAPER_PASTE adopts INCR as headline after results and labels the ABS 19/33 'do NOT use as headline ... invites a reviewer to notice the protocol swap'.",
               "decision_note": "origin/main@a3875c618:results/cross_purpose_laftr/HEADLINE.txt ('TWO CRITERIA -- the verdict flips depending on which one you use'; 'PAPER INTEGRATION DECISION (resolve in the morning)': choose B if the paper quotes 22/33, A otherwise) - criterion selection conditioned on existing wording after both results were known (flagged by the D fork; read here)",
               "registration_files_found": "none (no prediction/registration file for either criterion on any ref searched)"},
    "status": "checked against code and recorded aggregates", "scope": "commit chronology; author-time ordering only",
    "previous_assessment_value": "headline criterion choice made after seeing results (PE-D08)",
    "corrected_value_or_confirmation": "Confirmed with chronology: neither criterion was registered. In the submission the headline moved INCR(22) -> ABS(26) hours before submission; in the rebuttal it moved back to INCR after the Phase D results. Both must be labelled post hoc."})
items.append({
    "id": "B4", "question": "LAFTR cross-purpose counts 29/33 (ABS) and 16/33 (INCR)",
    "inputs": [i for i in B["inputs"] if "laftr" in i["location"].lower()], "command": cmdB,
    "result": B["LAFTR"],
    "status": "checked against code and recorded aggregates",
    "scope": "ABS recomputed from per-(dataset,attr,arch) aggregate means only; INCR present only as a recorded count",
    "previous_assessment_value": "29/33 reproduces; 16/33 only from DUAL_CRITERIA.json (laftr aggregate lacks gain field)",
    "corrected_value_or_confirmation": "Confirmed 29/33 from aggregates (no per-seed LAFTR cross-purpose file located). Correction to this script's own note and to the previous assessment: the LAFTR aggregate rows DO carry a gain field; item D8 recounts 16/33 INCR from it. Both are aggregate-level (mean over seeds), not per-seed recomputations."})
pe = C["runs"]["erase_pilot_vicreg1"]
pv = C["runs"]["erase_vicreg5"]
pr = C["runs"]["erase_rank8_diabetes"]
bl = C["runs"]["baseline_R5R7_bestpt"]
items.append({
    "id": "C1", "question": "Erase-layer pilot per cell: strict compliance, clean compliance, per_dim_std, eff_rank, task utility (which split?)",
    "inputs": ins("C_erase_vicreg.json"), "command": cmdC,
    "result": {"strict": f"{pe['totals']['strict']}/60 (baseline best.pt {bl['totals']['strict']}/60)", "clean": f"{pe['totals']['clean']}/60 (baseline {bl['totals']['clean']}/60)",
               "per_dataset": {ds: {k: e[k] for k in ["strict", "clean", "r2_mean", "r2_max", "per_dim_std_mean_over_seed_purpose", "eff_rank_mean_over_seed_purpose", "task_acc_mean_TEST", "cotter_kind"]} for ds, e in pe["per_dataset"].items()},
               "baseline_task_acc_TEST": {ds: e["task_acc_mean_TEST"] for ds, e in bl["per_dataset"].items()},
               "utility_split": "TEST split via trainer.evaluate(test_loader) (erase-layer-pilot@65dd5c050:experiments/run_v2_dataset.py:385-386); this split also carries every R2/selection decision, so it is not an untouched held-out split",
               "null_floor_insample_R2_p64": C["null_floor_insample_R2_p64"]},
    "status": "independently recomputed", "scope": "per-seed files: Adult/HMDA/Diabetes local untracked results/rebuttal/erase_layer_pilot_aws/ (Diabetes from the recovery run); no copies on GitHub or the drive; best.pt",
    "previous_assessment_value": "60/60 strict, 0/60 clean; utility drops occupation .992->.753, education .9995->.834, loan_amount_band .498->.377, medication_change .999->.764",
    "corrected_value_or_confirmation": "Confirmed. Additional: pilot mean test R2 (0.0078/0.0057/0.0069) is within 1.2-1.8x of the in-sample OLS null floor p/(n-1) (0.0042/0.0047/0.0060), i.e. linear compliance is structural (frozen union-LEACE upstream of every trainable map), not an optimisation outcome."})
items.append({
    "id": "C2", "question": "Erase-layer pilot / VICReg x5 / rank-8: nonlinear auditor channel per cell",
    "inputs": ins("C_erase_vicreg.json"), "command": cmdC,
    "result": {"aud_delta_ge_0.02": {"baseline_bestpt": f"{bl['totals']['aud_delta_ge_0.02']}/60", "erase_pilot": f"{pe['totals']['aud_delta_ge_0.02']}/60", "vicreg5": f"{pv['totals']['aud_delta_ge_0.02']}/42", "rank8": f"{pr['totals']['aud_delta_ge_0.02']}/18"},
               "adj_pass(R2<0.05 and delta<0.02)": {"baseline_bestpt": f"{bl['totals']['adj_pass']}/60", "erase_pilot": f"{pe['totals']['adj_pass']}/60", "vicreg5": f"{pv['totals']['adj_pass']}/42", "rank8": f"{pr['totals']['adj_pass']}/18"},
               "paired_vs_baseline": C["paired_vs_baseline"],
               "auditor_protocol": "PostHocAuditorSuite fit on TRAIN reps (subsampled to 20k), best test accuracy minus test majority (certificates.py EmpiricalAudit)"},
    "status": "independently recomputed", "scope": "stored delta = empirical_best_acc - majority per cell; no auditor refit",
    "previous_assessment_value": "not recounted; project notes state 'Empirical-auditor channel unchanged' (erase pilot)",
    "corrected_value_or_confirmation": "OVERTURNED: the nonlinear auditor channel got worse, not unchanged. Delta rose on 48/60 paired cells (Adult 23/24, HMDA 17/18, Diabetes 8/18); mean delta Adult 0.068->0.160, HMDA 0.128->0.298, Diabetes 0.007->0.059; cells with delta>=0.02 went 27/60 -> 44/60; adjusted passes 32/60 -> 16/60 (VICReg x5: 0/42)."})
items.append({
    "id": "C3", "question": "VICReg x5 sweep and rank-8 ablation claims",
    "inputs": ins("C_erase_vicreg.json"), "command": cmdC,
    "result": {"vicreg5": {ds: {k: e[k] for k in ["strict", "clean", "per_dim_std_mean_over_seed_purpose", "eff_rank_mean_over_seed_purpose", "task_acc_mean_TEST"]} for ds, e in pv["per_dataset"].items()},
               "vicreg5_vs_pilot_std_delta": {ds: round(pv["per_dataset"][ds]["per_dim_std_mean_over_seed_purpose"] - pe["per_dataset"][ds]["per_dim_std_mean_over_seed_purpose"], 4) for ds in pv["per_dataset"]},
               "vicreg5_diabetes": "no per-seed file located (run cut by the 10h cap; not on GitHub, local, or drive inventories)",
               "rank8": {k: pr["per_dataset"]["diabetes"][k] for k in ["strict", "clean", "r2_mean", "r2_max", "per_dim_std_mean_over_seed_purpose", "eff_rank_mean_over_seed_purpose", "task_acc_mean_TEST"]},
               "rank24_matched_comparator_(erase pilot diabetes)": {k: pe["per_dataset"]["diabetes"][k] for k in ["per_dim_std_mean_over_seed_purpose", "eff_rank_mean_over_seed_purpose", "task_acc_mean_TEST"]},
               "rank24_R7_original_architecture": bl["per_dataset"]["diabetes"]["task_acc_mean_TEST"]},
    "status": "independently recomputed", "scope": "best.pt; TEST split",
    "previous_assessment_value": "rank-8 18/18 [0.0043,0.0087] reproduces; comparator mislabelled 'published Round 7'; medication_change 0.75 vs 0.999",
    "corrected_value_or_confirmation": "VICReg x5 confirmed: 42/42 strict, 0/42 clean, std +0.0018 (Adult) / -0.0142 (HMDA), eff_rank +1.1/+1.6. Rank-8 counts confirmed, but CORRECTION to its headline: the 'rank-24' column in rebuttal-evidence:results/rebuttal/erase_rank8_diabetes_cpu/HEADLINE.md mixes the erase-pilot (R2, std, eff_rank) with the original R7 task accuracies (medication_change 99.9%, primary_diagnosis 28.6%). Against the matched rank-24 erase pilot, medication_change is 76.35% -> 75.25% (-1.1pp), not -24.7pp; the 25pp loss belongs to the erase-layer architecture, not to LoRA rank."})
items.append({
    "id": "C4", "question": "Is the absolute per_dim_std>=0.5 bar invariant to harmless rescaling; is eff_rank scale-invariant? (synthetic fixture + stored scale factors)",
    "inputs": [{"location": str(FX / "C_scale_invariance.py"), "sha256": sha(FX / "C_scale_invariance.py")},
               {"location": str(FX / "outputs" / "C_scale_invariance.json"), "sha256": sha(FX / "outputs" / "C_scale_invariance.json")}],
    "command": "/opt/homebrew/bin/python3 fixtures/C_scale_invariance.py > fixtures/outputs/C_scale_invariance.json",
    "result": {"verdict": CF["verdict"], "max_spread_across_isotropic_scales": CF["max_spread_across_isotropic_scales"],
               "std_doubles_with_x2": CF["std_ratio_x2_over_x1"],
               "rotation_changes_std": [r for r in CF["rows"] if r["transform"] == "orthogonal rotation"][0]["per_dim_std_mean"],
               "uniform_scale_needed_for_every_purpose_std_ge_0.5": CF.get("stored_runs_uniform_scale_needed_for_every_purpose_std_ge_0.5")},
    "status": "independently recomputed", "scope": "synthetic fixture (deterministic, ~2 s) + stored per-purpose stds",
    "previous_assessment_value": "per_dim_std>=0.5 is scale-dependent (PE-F16, stated, not demonstrated)",
    "corrected_value_or_confirmation": "Demonstrated: R2, held-out linear AUC and kNN AUC are identical (spread 0) under isotropic rescaling while per_dim_std scales linearly and crosses 0.5; eff_rank is invariant to isotropic scale but changes under anisotropic scaling; per_dim_std_mean even changes under an orthogonal rotation. Multiplying the stored erase-pilot h_p by 1.15 (Adult) / 1.28 (HMDA) / 1.41 (Diabetes) would turn 0/60 into 60/60 'cleanly compliant' with leakage and (linear-head) utility unchanged."})
items.append({
    "id": "C5", "question": "What did the 'architectural 0.5 floor' diagnostic measure; is the floor a universal impossibility?",
    "inputs": [{"location": "Bostesa/PCRL@origin/rebuttal-evidence:scripts/diagnose_perdim_std_floor.py", "sha256": None},
               {"location": "Bostesa/PCRL@origin/rebuttal-evidence:results/rebuttal/erase_layer_vicreg_sweep_aws/diagnostic_perdim_std.json", "sha256": None},
               {"location": "Bostesa/PCRL@origin/erase-layer-pilot-2026-05-17:pcrl/training/independence/vicreg.py", "sha256": None}],
    "command": "git show origin/rebuttal-evidence:<paths>; code reading",
    "result": {"measured": "per_dim_std of a RECONSTRUCTED frozen, untrained (seeded Kaiming/Xavier) StandardEncoder [128,128]->64, BN at default stats, eval mode, at 9 (dataset, seed) points: (a) network 128-d 0.18-0.22, (b) post-LEACE 0.15-0.19, (c) repr_proj without LEACE 0.23-0.29, (d) repr_proj after LEACE 0.20-0.25. No per-cell quantity; '0/60 cells' is 9 backbone points broadcast over cells.",
               "not_measured": "any rescaling of h, any wider/normalised head, any LoRA scale, train-mode (dropout-on) std that VICReg actually sees (VICReg hinge target gamma=1.0 is applied during training with dropout 0.3 active; health is measured in eval mode)",
               "hmda_baseline_note": "baseline HMDA has a purpose with per_dim_std ~1.2e-5 (numerically constant representation) that still 'passes' R2"},
    "status": "checked against code and recorded aggregates",
    "scope": "code reading + stored JSON; the dropout/eval-mode variance gap is a hypothesis, not measured",
    "previous_assessment_value": "not assessed (claim recorded in project notes as 'architectural, not erasure-mechanism-side')",
    "corrected_value_or_confirmation": "Narrowed: the diagnostic shows that this particular frozen random feature map, at its initialisation scale, has per-dim std 0.18-0.29 before LoRA. Together with C4 it supports only 'under an absolute std threshold of 0.5 this fitted backbone/head scale falls short', not an architectural impossibility: a scalar on repr_proj or h_p (or a scale-free health metric) changes the verdict without changing leakage."})
items.append({
    "id": "C6", "question": "LoRA-rank claims: where are the LoRA adapters and the erase layer; is 'LEACE acts in 128-d backbone space, decoupled from LoRA rank' true of the trained architecture?",
    "inputs": [{"location": "Bostesa/PCRL@96c03d010 (erase-layer-pilot-2026-05-17):pcrl/models/encoder.py, pcrl/models/lora.py, pcrl/training/v2_trainer.py fit_erase_layer", "sha256": None},
               {"location": "Bostesa/PCRL@erase-layer-pilot-2026-05-17:results/rebuttal/erase_layer_pilot/NEXT_STEPS.md:33-35 (--use-erase-layer --lora-target repr_proj_only)", "sha256": None},
               {"location": "Bostesa/PCRL@b158abc54:pcrl/models/lora.py:183-206 (R5/R7: adapters on every backbone Linear)", "sha256": None}],
    "command": "git diff 17ef7d449 65dd5c050 -- pcrl/models/encoder.py pcrl/models/lora.py pcrl/training/v2_trainer.py experiments/run_v2_dataset.py",
    "result": {"erase_pilot": "frozen nn.Linear(128,128) erase layer (W = I - P_left P_right from concept_erasure.LeaceEraser fit on the union one-hot of all purposes' disallowed attributes, train split, eval mode) between backbone.network and repr_proj; marked _skip_lora; LoRA only on repr_proj (128->64), rank 8 (Adult/HMDA) / 24 (Diabetes) / 8 in the ablation",
               "R5_R7_original": "LEACE realised by rank-r SVD warm start of the repr_proj LoRA; LoRA on every Linear incl. hidden layers; rank floor sum(c_i-1) argued from Prop. 2",
               "implication": "In the pilot, every h_p is an affine map of LEACE-erased features, so zero train-set linear covariance holds for any LoRA rank by construction; the rank-8 18/18 is implied by the architecture and does not test the R7 rank-floor hypothesis."},
    "status": "checked against code and recorded aggregates", "scope": "code paths; launch flags taken from NEXT_STEPS.md and HEADLINE_PARTIAL.md (EC2 argv not committed)",
    "previous_assessment_value": "pilot = 'single-union LEACE + per-purpose heads, LoRA only on repr_proj'",
    "corrected_value_or_confirmation": "Confirmed: the claim is true for the erase-layer architecture (128-d frozen erase, LoRA downstream) and false/inapplicable for the R5/R7 headline architecture, whose rank-24 choice it was meant to address."})
G = GF
items.append({
    "id": "G1", "question": "Status of the invalid R2->accuracy guarantee on origin/main vs fix/retire-accuracy-guarantee (not merged)",
    "inputs": [{"location": f"Bostesa/PCRL@{G['refs']['origin/main']['commit']}:pcrl/purposes/verification.py", "sha256": G["refs"]["origin/main"]["file_sha256"]},
               {"location": f"Bostesa/PCRL@{G['refs']['origin/fix/retire-accuracy-guarantee']['commit']}:pcrl/purposes/verification.py", "sha256": G["refs"]["origin/fix/retire-accuracy-guarantee"]["file_sha256"]},
               {"location": str(FX / "G_accuracy_guarantee.py"), "sha256": sha(FX / "G_accuracy_guarantee.py")},
               {"location": str(FX / "outputs" / "G_accuracy_guarantee.json"), "sha256": sha(FX / "outputs" / "G_accuracy_guarantee.json")}],
    "command": "/opt/homebrew/bin/python3 fixtures/G_accuracy_guarantee.py > fixtures/outputs/G_accuracy_guarantee.json; for b in $(git branch -r); do git show $b:pcrl/purposes/verification.py | grep -c RETIRED; done",
    "result": {"origin_main": G["refs"]["origin/main"], "fix_branch": G["refs"]["origin/fix/retire-accuracy-guarantee"],
               "merged": G["fix_branch_merged_into_main"], "counterexample": G["counterexample"],
               "other_refs_with_retired_bound": "27 remote research/* and ablations branches carry a retired version; none is an ancestor of origin/main",
               "main_callers": "experiments/deployment_case_study.py, run_distribution_shift.py, run_extended_baselines.py, run_mine_audit.py, run_multi_seed.py, run_nonlinear_certificates.py still call certified_accuracy_bound on origin/main"},
    "status": "independently recomputed", "scope": "counterexample in exact rational arithmetic; code state by text inspection",
    "previous_assessment_value": "origin/main still ships the refuted bound; fix branch unmerged; validation after raise unreachable",
    "corrected_value_or_confirmation": "Confirmed: origin/main@55e4cb1 returns a numeric bound (0.5 at R2=0 where a threshold rule reaches 0.9); the fix branch raises NotImplementedError first (the old numeric return is dead code); not merged. Paper Prop. 3 (pi + k*sqrt(eps*pi*(1-pi))) and code (pi + sqrt(eps*k*pi*(1-pi))) differ in form but both give 0.5 here."})
items.append({
    "id": "G2", "question": "Classification of the NeurIPS PCRL guarantees and which 'failures' are inside vs outside each stated scope",
    "inputs": [{"location": "NeurIPS PDF(23) text scratchpad/recon/b.txt: Prop.1 l.259; Prop.2 l.396-405; Prop.3 l.478-491; Prop.4 l.692; Prop.5 l.798; Prop.6 l.1793", "sha256": None}],
    "command": "reading of statements against code (A3, B1-B2, C1-C6, G1)",
    "result": {
        "Prop2_joint_LEACE": {"class": "LEACE linear covariance guarantee on the fit law (train split)", "holds_for": "initialisation of R5/R7 (LoRA on all layers then moves the features); every epoch of the erase-layer pilot (frozen eraser upstream)", "outside_scope": "test-split R2 (sampling), nonlinear auditors"},
        "Prop3_linear_compliance_bound": {"class": "INVALID as stated (G1)", "note": "retired on branches, live on origin/main"},
        "Prop4_LoRA_erasure_floor": {"class": "population linear statement for a fixed f0 and last-layer rank edit", "outside_scope": "R5/R7 trained model (all-layer LoRA)"},
        "Prop5_convex_identity": {"class": "algebraic identity (exact under stated weighting)"},
        "Prop6_cross_purpose_linear_bound": {"class": "linear-scope bound on concatenation R2 given per-purpose R2 and lambda_min(R)", "inside_scope": "LR concatenation cells are linear (8/11 LR cells flag ABS on original PCRL, e.g. Adult marital_status +20.8pp), but Prop 6 bounds concatenation R2 with a 1/lambda_min(R) amplification, not accuracy; accuracy flags therefore do not test it directly and no stored artifact shows a violation of its R2 inequality", "outside_scope": "MLP/XGB concatenation recovery (B1/B2) does not refute it"},
        "Prop1_Zhao_Gordon": {"class": "restated information-theoretic population bound on task error"},
        "empirical_nonlinear_resistance": {"class": "empirical, test split, auditor-suite specific (C2 shows it worsens under the erase-layer pilot)"}},
    "status": "checked against code and recorded aggregates", "scope": "classification; no new proofs",
    "previous_assessment_value": "F4 (LEACE guarantee initialisation-only), F25/F26 (Prop.3 invalid)",
    "corrected_value_or_confirmation": "Confirmed and extended: in the erase-layer pilot the Prop.2 guarantee does hold throughout training (structural), which is exactly why strict R2 is 60/60 and why nonlinear recovery (outside every linear certificate's scope) rose."})
json.dump(items, open(V / "report_parts" / "ABCG_pcrl.json", "w"), indent=1)
print(len(items), "items")
