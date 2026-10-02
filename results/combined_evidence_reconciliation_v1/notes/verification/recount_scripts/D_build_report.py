"""Assemble report_parts/D.json from the D recount outputs (run the D_*.py recounts first).
Run: /opt/homebrew/bin/python3 D_build_report.py
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE.parent
RO = V / "recount_outputs"
FX = V.parent.parent / "fixtures"


def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def J(n):
    return json.loads((RO / n).read_text())


def ins(name, keep=lambda loc: True, limit=12):
    d = J(name)
    sel = [i for i in d["inputs"] if keep(i["location"])][:limit]
    return sel + [{"location": f"notes/verification/recount_outputs/{name}", "sha256": h(RO / name)}]


tw, sp, lh, lr, sr, lc = (J(x) for x in ["D_three_way.json", "D_splince.json", "D_laftr_hard.json",
                                       "D_leace_raw.json", "D_sep_rlace.json", "D_laftr_crosspurpose.json"])
PY = "/opt/homebrew/bin/python3"
items = [
 {"id": "D1", "question": "INLP/LAFTR/PCRL three-way (NeurIPS App. Q, Table 13): mean R2 0.014/0.038/0.270; strict pass 56/43/15 of 60 (93/72/25%).",
  "inputs": ins("D_three_way.json", lambda l: "inlp_results" in l or "dominant_axis" in l or l.endswith("purpose_0/seed_0/eval_reps.npz")),
  "command": f"{PY} recount_scripts/D_three_way.py",
  "result": {"mean_r2": [tw["pcrl_mean_r2"], tw["inlp_mean_r2"], tw["laftr_mean_r2"]],
             "pass_lt_0.05": [tw["pcrl_pass_lt_0.05"], tw["inlp_pass_lt_0.05"], tw["laftr_pass_lt_0.05"]],
             "by_dataset": tw["pass_by_dataset_lt"],
             "pcrl_column_equals_finalpt_DA_audit": tw["pcrl_column_vs_finalpt_DA_audit_mismatches"] == 0,
             "inlp_recomputed_from_drive_reps": tw["inlp_recomputed_from_drive_reps"],
             "laftr_adult_recomputed_from_git_history_reps": tw["laftr_adult"],
             "laftr_hmda_diabetes": tw["laftr_hmda_diabetes_provenance"]},
  "status": "independently recomputed",
  "scope": ("PCRL column = final.pt one-hot R2 (test split, in-sample OLS). INLP 60 rows recomputed from drive eval_reps (float64): 42/60 vs stored 43/60 "
            "(stored values match a float32 solve; 1 flip hmda/underwriting s2 ethnicity 0.039->0.069, R2 carried by directions with relative singular value ~1e-7). "
            "LAFTR Adult 24 rows recomputed exactly (0/24). LAFTR HMDA/Diabetes 36 rows: values only in the joined aggregate (from an S3 FINAL_BENCHMARK.csv in the 7-day-lifecycle bucket) -> "
            "reported but not independently reproduced for those 36 rows."),
  "previous_assessment_value": "0.0136/0.0382/0.2702; 56/43/15 reproduce from inlp_results.json; LAFTR HMDA/Diabetes per-seed not committed",
  "corrected_value_or_confirmation": ("Confirmed as stored (56/43/15). Under a float64 recomputation INLP is 42/60 (mean 0.0396). Only 24/60 LAFTR rows are reproducible. "
                                      "Neither INLP (stops on val-split LR accuracy within 1pp of majority) nor LAFTR (adversarial CE, best val task loss) was trained against R2<=0.05; "
                                      "both TRAIN their encoder while PCRL's backbone is a frozen seeded random MLP, so 'the same backbone anchors all three' holds for architecture only.")},
 {"id": "D2", "question": "Task accuracy: 'INLP averages 0.815 across cells and is comparable for PCRL' (App. Q).",
  "inputs": ins("D_three_way.json", lambda l: "inlp_results" in l or "per_seed_results" in l),
  "command": f"{PY} recount_scripts/D_three_way.py  (task_acc_pairs)",
  "result": {"inlp_mean_27": tw["inlp_task_acc_mean_27"], "pcrl_bestpt_mean_same_27": tw["pcrl_task_acc_mean_27_bestpt"],
             "cells_inlp_gt_pcrl_by_1pp": tw["cells_inlp_gt_pcrl_by_1pp"]},
  "status": "independently recomputed",
  "scope": "PCRL task accuracies are from per_seed_results.json (best.pt reload); no final.pt task accuracies stored. Test split.",
  "previous_assessment_value": "not checked",
  "corrected_value_or_confirmation": "INLP 0.815 confirmed; PCRL on the same 27 cells is 0.779 (best.pt), INLP higher by >1pp in 14/27 cells -> 'comparable' overstates PCRL's utility."},
 {"id": "D3", "question": "SPLINCE vs PCRL (App. R, Table 14): R2-only 60/60 vs 56/60; +Health 3/60 vs 7/60; +Delta_aud 2/60 vs 3/60.",
  "inputs": ins("D_splince.json", lambda l: "splince_benchmark" in l and ("adult_s0" in l or "hmda_s2" in l)),
  "command": f"{PY} recount_scripts/D_splince.py",
  "result": {"splince": {"r2_only": sp["splince_r2_only"], "r2_health": sp["splince_r2_and_health"], "r2_health_delta": sp["splince_r2_health_delta"],
                         "adj(r2+delta)": sp["splince_r2_and_delta(adj_pass)"], "fallback_to_LEACE": sp["splince_fallbacks_to_leace"],
                         "mean_task_drop_pp": sp["splince_mean_task_drop_pp"], "post_per_dim_std_range": sp["splince_post_per_dim_std_range"]},
             "warm_start": "final.pt of R5 (Adult/HMDA) and R7 (Diabetes)",
             "pcrl_clean_counts_by_rule": sp["pcrl_clean_counts_by_rule"]},
  "status": "independently recomputed",
  "scope": ("SPLINCE side recomputed from per-cell stored metrics (not from representations). SPLINCE is an in-house reimplementation of Holstege et al. Thm 1 "
            "(pcrl/baselines/splince.py), fit per (purpose, ONE attribute, seed) - each cell erases only the audited attribute, whereas PCRL's h_p must satisfy all "
            "of the purpose's attributes at once. Its input is NOT 'pre-LEACE features': R5/R7 code has no separate LEACE buffer (set_leace_projection added in "
            "940912cb9, after R7), the LEACE map is folded into the trained LoRA, so SPLINCE post-processes PCRL's own final.pt h_p (already constrained/collapsed; "
            "pre-projection per_dim_std 0.22-0.58). Health threshold is scale dependent: one SPLINCE cell has per_dim_std 968."),
  "previous_assessment_value": "not recounted (PCRL 7/60 noted as mixed-checkpoint; 5/60 on best.pt)",
  "corrected_value_or_confirmation": ("SPLINCE 60/3/2 and adj 38/60 confirmed. NEW: SPLINCE pre_health is PCRL final.pt health, giving a single-checkpoint final.pt "
                                      "clean count of 6/60 (hmda underwriting race+ethnicity s2; diabetes billing race+gender s1,s2) vs paper 7/60 (final R2 x best.pt health) "
                                      "vs 5/60 on best.pt. The head-to-head is SPLINCE-on-top-of-PCRL vs PCRL, per-attribute vs per-purpose.")},
 {"id": "D4", "question": "LAFTR-hard-R2 (branch laftr-hard-r2-2026-05-17): are full outputs present; Adult 0/24 COLLAPSED; HMDA/Diabetes?",
  "inputs": ins("D_laftr_hard.json"),
  "command": f"tar -xf <drive>/archives/wt-PCRL-superpowers-laftr-hard-r2-2026-05-17.tar laftr-hard-r2-2026-05-17/results/...; {PY} recount_scripts/D_laftr_hard.py",
  "result": {ds: {k: lh["per_dataset"][ds][k] for k in ("strict_r2_lt_0.05", "adj_pass(stored: R2<0.05 and delta<0.02)", "clean(R2<0.05, std>=0.5, eff_rank>=2)",
                                                        "mean_r2", "best_epochs", "cotter_selection", "pcrl_comparator_bestpt_strict")}
             for ds in ("adult", "hmda", "diabetes")} | {"totals": lh["totals"]},
  "status": "independently recomputed",
  "scope": ("Full 3-seed per_seed_results for all three datasets exist ONLY on the drive (not in any git ref; committed copy is the 1-seed 5-epoch smoke). "
            "Counts are on the Cotter-fallback best.pt (0 feasible epochs every seed); Adult selected epochs 11/15/17 of 200 - the same early-epoch fallback "
            "that produced PCRL Round-4 Adult 0/24 on best.pt vs 20/24 on final.pt. No final.pt evaluation stored, so the final-iterate count is unknown. "
            "'LAFTR-hard' = PCRL's frozen random backbone + LoRA + LEACE warm-start + proxy-Lagrangian R2 constraint + a fixed-weight adversary: an "
            "R2-trained comparator, but an ablation of PCRL rather than LAFTR. HEADLINE's 'LAFTR-Q' HMDA/Diabetes row was copied from the paper table (cb95e2502)."),
  "previous_assessment_value": "unverifiable (only smoke committed; full result reportedly S3 only)",
  "corrected_value_or_confirmation": ("Located and recounted: strict R2<0.05 Adult 0/24, HMDA 17/18, Diabetes 18/18 = 35/60 (vs PCRL 54/60 best.pt / 56/60 final.pt); "
                                      "adj pass 0+0+18 = 18/60; clean 0/60; STATUS COLLAPSED on all three. Adult 0/24 is selection-rule dependent (epoch 11-17).")},
 {"id": "D5", "question": "LEACE-on-raw threat baseline (App. P): '0/20 pair-seeds pass combined criterion' and 'reduces task accuracy to majority baseline on all three datasets'.",
  "inputs": ins("D_leace_raw.json"),
  "command": f"tar -xf <drive>/archives/fl-PCRL-main-results-ignored.tar results/<ds>_LEACE/erased_*.pt; {PY} recount_scripts/D_leace_raw.py",
  "result": {k: lr[k] for k in ("n_pairs", "max_abs_diff_vs_stored", "stored_r2_pass_lt_0.05", "r2_pass_after_1e-3_cutoff",
                                "heldout_LR_at_majority(<=0.5pp)", "delta_lt_2pp(nonlinear auditors)", "combined_pass_stored")}
            | {"task_acc_minus_majority": "8/9 tasks above majority (Adult income +7.6pp, HMDA amount_band +39.8pp, Diabetes primary_dx +7.5pp); only readmission -0.8pp"},
  "status": "independently recomputed",
  "scope": ("Official concept_erasure.LeaceFitter, but applied SEQUENTIALLY per attribute (not the joint fit PCRL uses), on raw inputs that contain the protected "
            "attributes; single deterministic run, 20 pairs (not pair x seed). Stored R2 reproduced exactly (max diff 0.0) from the stored erased test tensors. "
            "18/20 R2 'failures' live entirely in directions with relative singular value 1e-7..1e-4 (numerical residue of a train-fit eraser re-fit in-sample on "
            "test): dropping relative s.v.<1e-3 gives R2<0.03 on 20/20, and a train-fit LR is exactly at majority on 20/20. The robust part of the failure is "
            "nonlinear: Delta_aud>=2pp on 17/20."),
  "previous_assessment_value": "'LEACE-on-raw 0/20 exists only as markdown'",
  "corrected_value_or_confirmation": ("0/20 reproduces from stored per-pair JSON + drive tensors, but the linear leg is numerically fragile (20/20 pass after a 1e-3 spectral "
                                      "cutoff; combined would be 3/20). The task-accuracy sentence is contradicted by the stored per_task JSON (not majority on 8/9 tasks; "
                                      "above PCRL on income, amount_band, primary_dx).")},
 {"id": "D6", "question": "Per-purpose independent encoders (App. P): 5/20; Adult 0/8 R2 in [0.062,0.117]; HMDA 0/6 [0.134,0.170]; Diabetes passes via collapse.",
  "inputs": ins("D_sep_rlace.json", lambda l: "SEPARATE" in l),
  "command": f"{PY} recount_scripts/D_sep_rlace.py",
  "result": sr["separate"],
  "status": "independently recomputed",
  "scope": ("Stored per-pair results recounted (no representations). One seed (s0), 20 pairs. Trained with the pre-v2 PCRLTrainer (adversary + lambda_verify=50 soft R2 "
            "penalty, best val task loss), not the v2 hard-constraint/LoRA/LEACE stack; the comparator column '6/8, 5/6, 5.3/6' is pre-v2 PCRL, not the 56/60 model. "
            "'Shared frozen backbone is load-bearing' therefore compares two different training stacks."),
  "previous_assessment_value": "'App. P 5/20 is 20 pairs at one seed against a pre-v2 PCRL'",
  "corrected_value_or_confirmation": "Confirmed: 5/20 (Diabetes 5/6 with eff_rank 1.0-1.15), Adult 0/8 [0.0625,0.1165], HMDA 0/6 [0.1341,0.1697]."},
 {"id": "D7", "question": "R-LACE/LEACE diagnostic (results/rlace_diagnostic.json, 7/8 GREEN; cited for the RLACE/LEACE rank-(K-1) threshold).",
  "inputs": ins("D_sep_rlace.json", lambda l: "rlace" in l),
  "command": f"{PY} recount_scripts/D_sep_rlace.py",
  "result": sr["rlace"],
  "status": "checked against code and recorded aggregates",
  "scope": ("Adult only, one seed, stored per-pair values. 'R-LACE' is INLP-style iterative logistic null-space projection at rank 1/4/8, not the Ravfogel 2022 minimax "
            "(script docstring says so). Backbone is a TRAINED StandardEncoder, not PCRL's frozen random one. GREEN ignores nonlinear leakage: LEACE MLP Delta 0.09-0.46, 0/8 < 0.02."),
  "previous_assessment_value": "memory note: '7/8 GREEN; LEACE crushes R2<0.01 on every pair'",
  "corrected_value_or_confirmation": ("Confirmed 7/8 GREEN and LEACE R2 0.004-0.009 on 8/8 with <=3.3pp task drop - i.e. one-shot official LEACE on a trained backbone meets the "
                                      "linear criterion that PCRL trains for, a comparator absent from Table 13; under Delta_aud<0.02 it passes 0/8.")},
 {"id": "D8", "question": "LAFTR side of the §5.5 cross-purpose comparison: 29/33 absolute, 16/33 incremental.",
  "inputs": ins("D_laftr_crosspurpose.json"),
  "command": f"{PY} recount_scripts/D_laftr_crosspurpose.py",
  "result": {k: lc[k] for k in ("absolute_concat_minus_majority_gt_1pp", "incremental_concat_minus_best_single_gt_1pp", "by_dataset_abs_incr", "worst_abs")},
  "status": "checked against code and recorded aggregates",
  "scope": ("Recounted from stored per-(dataset,attribute,arch) aggregate rows (mean over 3 seeds); the incremental field exists in the LAFTR aggregate "
            "(concat_gain_over_best_single_pp_mean). LAFTR HMDA/Diabetes encoders/reps not located. results/cross_purpose_laftr/HEADLINE.txt records that the "
            "criterion (A vs B) was to be chosen to match the existing paper wording ('PAPER INTEGRATION DECISION') - post-hoc criterion selection."),
  "previous_assessment_value": "29/33 reproduces; LAFTR 16/33 only from DUAL_CRITERIA.json (laftr aggregate lacks gain field)",
  "corrected_value_or_confirmation": "Both 29/33 and 16/33 recount from the LAFTR aggregate itself (the gain field is present); per-seed LAFTR reps not located."},
 {"id": "D9", "question": "Implementation provenance: is PCRL's `LEACEEraser` (pcrl/models/baselines.py) LEACE?",
  "inputs": [{"location": "Bostesa/PCRL@origin/main:pcrl/models/baselines.py:120-190", "sha256": None},
             {"location": "fixtures/D_inhouse_leace_vs_leace.py", "sha256": h(FX / "D_inhouse_leace_vs_leace.py")},
             {"location": "fixtures/outputs/D_inhouse_leace_vs_leace.json", "sha256": h(FX / "outputs/D_inhouse_leace_vs_leace.json")}],
  "command": f"{PY} fixtures/D_inhouse_leace_vs_leace.py > fixtures/outputs/D_inhouse_leace_vs_leace.json",
  "result": "In-house class = orthogonal projection off span(OLS coefficients); on synthetic non-isotropic data it leaves one-hot R2 0.58-0.65 (raw 0.68-0.79) while the LEACE definition gives 0.0 (cross-cov ~1e-14).",
  "status": "independently recomputed",
  "scope": ("Synthetic fixture + code reading. The in-house class is used only by legacy pre-v2 scripts (run_extended_baselines, sweep_baselines, run_multi_seed, "
            "run_seeds_fixed, run_hmda_seeds, run_diabetes_all, run_leace_postprocess); every v2 path located (PCRL warm-start v2_trainer.leace_warm_start, "
            "run_leace_baseline, rlace_diagnostic) and durable-guarantees baseline_gauntlet use the official concept_erasure package. No manuscript number traced to the in-house class."),
  "previous_assessment_value": "not checked",
  "corrected_value_or_confirmation": "Misnamed in-house eraser (one INLP-like OLS null-space step), not LEACE; not used by located headline numbers."},
]
for it in items:
    for i in it["inputs"]:
        if i["sha256"] is None and "origin/main:pcrl/models/baselines.py" in i["location"]:
            import subprocess
            b = subprocess.run(["git", "-C", "/Users/nathansamson/PCRL", "show", "origin/main:pcrl/models/baselines.py"],
                               capture_output=True, check=True).stdout
            i["sha256"] = hashlib.sha256(b).hexdigest()
(V / "report_parts").mkdir(exist_ok=True)
(V / "report_parts" / "D.json").write_text(json.dumps(items, indent=1, default=float))
print("items", len(items))
