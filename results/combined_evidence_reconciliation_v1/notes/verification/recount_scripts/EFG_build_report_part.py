"""Assemble report_parts/EFG_aaai.json from the recount outputs (values are read, not typed).
Run after: E_utility_heldout.py F_audit59_recount.py F_table1_bars.py F_multiclass_worstpair.py
F_baseline_worstpair.py F_isolate_advantage_bars.py F_survivors_tier2_bars.py G_aaai_scope_classification.py
"""
import hashlib, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "recount_outputs")
PART = os.path.join(HERE, "..", "report_parts", "EFG_aaai.json")
DG = "/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/dg"
REL = "results/combined_evidence_reconciliation_v1/notes/verification/"


def L(name):
    return json.load(open(os.path.join(OUT, name)))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ins(name, script):
    d = L(name)
    base = list(d.get("inputs", []))
    base.append({"location": REL + "recount_outputs/" + name, "sha256": sha(os.path.join(OUT, name))})
    base.append({"location": REL + "recount_scripts/" + script, "sha256": sha(os.path.join(HERE, script))})
    return base


def code(path):
    return {"location": f"durable-guarantees@956f5c8:{path}", "sha256": sha(os.path.join(DG, path))}


PY = "/opt/homebrew/bin/python3 "
a59 = L("F_audit59_recount.json")
t1 = L("F_table1_bars.json")
mc = L("F_multiclass_worstpair.json")
bw = L("F_baseline_worstpair.json")
iso = L("F_isolate_advantage_bars.json")
sv = L("F_survivors_tier2_bars.json")
ut = L("E_utility_heldout.json")
g = L("G_aaai_scope_classification.json")
summ = mc["summary"]
bwr = {r["file"]: r for r in bw["rows"]}
items = []

items.append(dict(
    id="E1", question="AAAI Table 1 / headline utilities: in-sample vs held-out; do per-cell rankings change under held-out utility?",
    inputs=ins("E_utility_heldout.json", "E_utility_heldout.py") + [code("experiments/targeted_noise.py"), code("experiments/baseline_gauntlet.py")],
    command=PY + "E_utility_heldout.py",
    result={"conventions": "baselines: Table-1 utility = lift_best = max(own head in-sample on all rows, fresh LR head 75/25 on rows the "
                           "encoder saw) (baseline_gauntlet.py:375-380; targeted_noise.py:108-120); ours e2e: own head in-sample; "
                           "out-of-partition utility exists only for 9 'ours' points (fresh_partition_generalization.json)",
            "which_convention_table1_used": {f"{r['method']}|{r['cell']}|T{r['tier']}": r["table1_convention_uses"] for r in ut["table1_rows"]},
            "ranking_changes_all_7_methods": {k: v["changed_table1_vs_heldout"] for k, v in ut["rank_changes_all_methods"].items()},
            "tier1_passers_plus_ours": ut["tier1_passers_plus_ours_order"],
            "ours_in_vs_out_of_partition": ut["ours_fresh_partition"],
            "rows_lacking_heldout_values": ut["rows_without_any_heldout_value"]},
    status="checked against code and recorded aggregates",
    scope="Stored per-config utility means (no per-person predictions stored for utility). Held-out here means head-held-out "
          "(encoder/eraser still fit on all rows) for baselines; truly out-of-partition only for the 9 'ours' points.",
    previous_assessment_value="F-05/F-06: in-sample utility, six conventions, lift_best=max(own,LR); Adult clean lift 0.069 out vs 0.185 in",
    corrected_value_or_confirmation="CONFIRMED and extended: easy-cell ordering among passing rows + ours flips out of partition "
        "(in-partition subspace 102.2% > full-rank 94.6%; out-of-partition full-rank 70.8% > subspace 64.0%). All-method orderings change "
        "on easy and hard cells when every row is put on the LR-held-out convention (e.g. LAFTR-official easy 102.0% own vs 47.4% LR); "
        "middle unchanged. Table 1 LAFTR '3-17%' is not reproducible from the stored operating points (66-102% on easy)."))

items.append(dict(
    id="E2", question="Attacker access per tier from code; does Tier 2 receive the knowledge the paper describes?",
    inputs=[code("utils/battery.py"), code("experiments/diagnostic.py"), code("experiments/run_knows_q.py"),
            code("experiments/averaging_attack.py"), code("experiments/two_tier_certification.py"), code("experiments/mi_ceiling.py")],
    command="source inspection (sed -n on the listed files); no execution",
    result={"tier1": "XGB, MLP(256-256), rank-32 LoRA+ReLU probe (diagnostic.py:149-179), each trained on 75% of released rows WITH s labels "
                     "and scored on 25%; one fresh noise draw per probe seed; black box w.r.t. model parameters",
            "tier2": "Tier 1 + Gaussian class-conditional LRT fit on the CLEAN pre-noise representation of the attacker-train rows "
                     "(H_clean[tr], labels) with the exact noise covariance (sigma^2 I, or sigma^2 QQ^T i.e. knows the subspace basis Q), "
                     "scoring ONE noised release of each held-out row (battery.py:60-103). No access to the target's clean vector, to "
                     "the realised noise, to repeated releases, or to model weights/gradients. For deterministic baselines "
                     "Sigma_noise=0 so Tier 2 = Tier 1 + QDA on the release (no extra knowledge).",
            "knows_q": "separate experiment (run_knows_q.py:1-11): projects the release on span(Q)-complement, then XGB/MLP/LoRA; 0.98 easy",
            "averaging": "separate experiment (averaging_attack.py:1-25), easy subspace point only, N in {1,2,4,8,16}: "
                         "tier1 0.506/0.514/0.520/0.530/0.560; breach at N=16",
            "neural_or_whitebox_adaptive_attacker": "none located (no attacker uses parameters, gradients, or trains against the mechanism "
                                                   "beyond fitting on noised releases)",
            "paper_text": "paper.tex:251-254 'knows the clean representation and the noise we added'; battery.py:15-21 docstring adds "
                          "'an attacker who can average repeated queries'"},
    status="checked against code and recorded aggregates",
    scope="Code at dg@956f5c8; averaging numbers from results/averaging_attack.json.",
    previous_assessment_value="F-01: Tier 2 = white-box population access with one release, not insider/averaging",
    corrected_value_or_confirmation="CONFIRMED. Refinement: Tier 2 knows the noise DISTRIBUTION (and Q), not the noise realisation, "
        "and clean vectors of a labelled population sample, not of the target; the paper's 'knows ... the noise we added' over-states it, "
        "and battery.py's 'average repeated queries' is not implemented in Tier 2 (only in the separate N<=16 averaging run on one point)."))

sc = a59["stored_counts_fail_by_bar"]
items.append(dict(
    id="F1", question="Recount 59/67 (approved configurations failing the 0.55 bar) and its threshold sensitivity at 0.52/0.55/0.60",
    inputs=ins("F_audit59_recount.json", "F_audit59_recount.py"),
    command=PY + "F_audit59_recount.py",
    result={"stored_rule_max_xgb_mlp": {b: v["xgb_mlp (paper rule)"] for b, v in sc.items()},
            "with_lora_where_stored": {b: v["t1_with_lora_where_stored"] for b, v in sc.items()},
            "tier2_lrt_where_stored": {b: v["t2_with_lora_lrt_where_stored"] for b, v in sc.items()},
            "recomputed_from_probability_arrays": a59["independent_recount_from_scores"]["fail_counts_by_bar_max_xgb_mlp"],
            "n_rescored": a59["independent_recount_from_scores"]["n_configs_rescored"],
            "max_abs_auc_delta_vs_stored": a59["independent_recount_from_scores"]["max_abs_delta_vs_stored"],
            "near_bar": a59["near_bar_configs_max_xgb_mlp_in_[0.50,0.62]"]},
    status="independently recomputed",
    scope="Macro-OvR AUC re-scored from the stored held-out probabilities (analysis/tpr59_scores + tpr_ext_scores, drive) for all 67; "
          "the probabilities themselves come from the original attacker fits (re-run gate 0.01 in run_tpr_failing59.py). 67 rows = 64 distinct measurements.",
    previous_assessment_value="R03 67/59/8 match; R21 64/59/51 at 0.52/0.55/0.60",
    corrected_value_or_confirmation="CONFIRMED 64/59/51 from per-config probability arrays (max |dAUC| 1.7e-5). Three failures sit within "
        "0.004 of the bar (0.5502, 0.5511, 0.5535); adding LoRA changes nothing; adding the stored Tier-2 LRT gives 65/63/51."))

items.append(dict(
    id="F2", question="Decompose 59/67 by surface and attacker metric",
    inputs=ins("F_audit59_recount.json", "F_audit59_recount.py"),
    command=PY + "F_audit59_recount.py",
    result={"surface": a59["surface_decomposition"], "attacker_at_0.55": a59["attacker_decomposition_at_0.55_xgb_mlp"],
            "lora_flips": a59["configs_passing_xgb_mlp_but_failing_lora_at_0.55"],
            "mechanism": a59["by_mechanism_at_0.55"], "dataset_[n,fail]": a59["by_dataset_[n,fail]_at_0.55"],
            "metric": "66/67 binary attributes (macro OvR = AUC); 1 multiclass (HMDA/race noise sigma=8), see F3"},
    status="independently recomputed",
    scope="Representation surface only: no output-surface reading is stored for any of the 67 approved configurations.",
    previous_assessment_value="F-03: audit attacks representation only; 8 survivors never output-tested",
    corrected_value_or_confirmation="CONFIRMED: 67/67 scored on the representation, 0/67 on the output surface although the paper's audit "
        "question is 'either exposed surface' (paper.tex:306-308). Failure attribution: 52 both XGB and MLP, 3 XGB only, 4 MLP only; "
        "48/48 projections and 11/19 noise configs fail. Per-config output-surface battery would be needed (not located anywhere)."))

hm = a59["multiclass_row_hmda_race_noise_s8"]
items.append(dict(
    id="F3", question="Multiclass handling inside the audit: worst-class, worst-pair, unsupported classes",
    inputs=ins("F_audit59_recount.json", "F_audit59_recount.py"),
    command=PY + "F_audit59_recount.py (multiclass block)",
    result={"config": "E4S1 noise|hmda/race|noise sigma=8.0 (only multiclass audit row)",
            "worst_class_ovr_xgb_by_seed": {k: v["per_class_ovr_xgb"] for k, v in hm.items() if k.startswith("ps")},
            "heldout_class_counts": hm["ps0"]["test_fold_class_counts"],
            "unsupported_classes_(<3000 rows in cell)": {"3": 336, "4": 126},
            "means_over_probe_seeds": hm["means_over_probe_seeds"]},
    status="independently recomputed",
    scope="One row; classes 3-4 have 336/126 held-out rows per fold.",
    previous_assessment_value="F-13: worst-pair conventions, class 4 = 0.79% (126 rows)",
    corrected_value_or_confirmation="Under worst-class or all-pairs the row fails by a wide margin (0.64-0.65, driven by unsupported class 4); "
        "under the supported-pair criterion it PASSES (0.524-0.531), so 59/67 becomes 58/67 under the paper's own supported-class rule; "
        "under the macro rule it fails by 0.0011."))

c = t1["table1_counts"]
items.append(dict(
    id="F4", question="Table 1 pass counts ('7 of 42') at AUC bars 0.52/0.55/0.60",
    inputs=ins("F_table1_bars.json", "F_table1_bars.py"),
    command=PY + "F_table1_bars.py",
    result={k: {"pass": v["pass"], "undetermined": v["undetermined"], "fail": v["fail"], "passes": v["passes"],
                "undetermined_list": [x[:3] for x in v["undetermined_list"]]} for k, v in c.items()},
    status="checked against code and recorded aggregates",
    scope="Re-thresholds stored per-row AUC means using the stored selection rule (3-seed certification only for rows that cleared 0.55). "
          "UNDETERMINED = only a seed-0 sweep row clears the new bar; no 3-seed point exists at that bar.",
    previous_assessment_value="R08 7/42 match; R21 'gauntlet 0/1/4 of 36 pass at 0.52/0.55/0.60'",
    corrected_value_or_confirmation="7/42 CONFIRMED at 0.55 (identical with or without the output surface). 0.52: 0 certified passes "
        "(4 FARE combos undetermined). 0.60: 8 certified passes + 2 undetermined (VFAE middle/hard T1, seed-0 rows 0.585/0.587). "
        "Appendix's '4 of 36 pass at 0.60' (appendix.tex:399-401) counts those two single-seed rows as passes; certified count is 2."))

items.append(dict(
    id="F5", question="Worst-class / worst-pair for Table 1 passes and our operating points on the 5-class HMDA/race cells",
    inputs=ins("F_multiclass_worstpair.json", "F_multiclass_worstpair.py") + ins("F_baseline_worstpair.json", "F_baseline_worstpair.py"),
    command=PY + "F_multiclass_worstpair.py; " + PY + "F_baseline_worstpair.py",
    result={"VFAE_easy_T1_(Table-1 pass)": {k: round(v, 4) for k, v in bwr["bg_easy_VFAE_b1_sampled-z"].items() if isinstance(v, float)},
            "ours_paper_convention_(max over arch of mean)": summ,
            "heldout_class_counts_per_fold": mc["per_operating_point"]["subspace_easy"]["heldout_class_counts_per_fold"],
            "nulls_stored": {"all_pairs_max_over_27": mc["worstpair_null_stored"]["worstpair_max"],
                             "all_pairs_mean": mc["worstpair_null_stored"]["worstpair_mean"], "supported": 0.5217},
            "FARE": bw["fare"]},
    status="independently recomputed",
    scope="Re-scoring of stored held-out probability arrays (drive analysis/tpr_scores, tpr_ext_scores). Strict (max over draws) values "
          "reproduce worstpair_supported.json exactly (e.g. subspace easy rep 0.5793/0.5412). FARE certified points have no stored arrays.",
    previous_assessment_value="F-13 / R23: 0.01-0.11 above bar on nine affected passes; footnote 7 0.014/0.060/0.069",
    corrected_value_or_confirmation="CONFIRMED with additions: (i) the only non-FARE Table-1 pass (VFAE easy T1, macro 0.535) fails every "
        "worst-case criterion incl. supported pairs (0.566 > supported null 0.522); multiclass_dual_report.json lists VFAE as 'not_available' "
        "although its arrays exist on the drive. (ii) Our full-rank T1/T2 points (Table 1 'Ours, full-rank' checkmarks) fail the supported "
        "criterion on both 5-class cells at both tiers (rep supported pair 0.60/0.58 T1, 0.59/0.57 T2 with LRT). (iii) Footnote 7's "
        "0.014/0.060 use max-over-attackers-then-mean (maxseed_worstpair.json worst_mean); under the macro verdict's own aggregation "
        "(max over attackers of means) the subspace easy representation reads 0.547 (passes) and middle 0.587. (iv) 4 FARE multiclass "
        "passes unverifiable: would need per-class probability arrays for the FARE certified points (no such file located)."))

fr = iso["frontier_seed0_sweeps_by_bar"]
items.append(dict(
    id="F6", question="Isolate-then-noise advantage over full-rank noise at bars 0.52/0.55/0.60",
    inputs=ins("F_isolate_advantage_bars.json", "F_isolate_advantage_bars.py") + [code("experiments/run_isolate_vs_fullrank.py")],
    command=PY + "F_isolate_advantage_bars.py",
    result={"frontier_advantage_pp_T1_seed0_sweeps": {b: {cell: v["advantage_pp_t1"] for cell, v in per.items()} for b, per in fr.items()},
            "paper_matched_nearest_T1_gap_pp": {m["cell"]: m["gap_pp"] for m in iso["paper_matched_comparison_stored"]},
            "selected_points_in_vs_out_of_partition": [{k: o[k] for k in ("cell", "advantage_pp_in_partition", "advantage_pp_out_of_partition")}
                                                       for o in iso["selected_points_in_vs_out_of_partition"]],
            "tier2": "subspace channel never meets any bar <= 0.60 at Tier 2 (stored LRT 0.66-0.87)"},
    status="checked against code and recorded aggregates",
    scope="Seed-0 sweeps (single training seed) for the frontier; full-rank sweep grid starts at sigma=6/12/16 so lower-sigma full-rank "
          "points are not stored; in-sample utility except the out-of-partition row.",
    previous_assessment_value="F-08 / R19: matched rule picks over-protected sigma; easy ~100 -> ~46 pp; at the bar about 8/43/85 pp",
    corrected_value_or_confirmation="CONFIRMED direction; values: at 0.55 the frontier advantage is +4.4/+44.1/+87.9 pp (easy/middle/hard) "
        "vs the paper's matched 99.7/61.8/84.2; at 0.52 +48.3/+75.0/+82.3; at 0.60 +4.4/+44.1/+75.4. Out of partition at the paper's own "
        "selected Tier-1 points the easy-cell advantage REVERSES (-6.8 pp: full-rank 70.8% vs subspace 64.0%); middle +39.4, hard +99.8. "
        "The advantage is a hard/middle-cell, Tier-1-only result."))

items.append(dict(
    id="F7", question="'Eight survive; five of the eight still leak at Tier 2' at bars 0.52/0.55/0.60",
    inputs=ins("F_survivors_tier2_bars.json", "F_survivors_tier2_bars.py"),
    command=PY + "F_survivors_tier2_bars.py",
    result={"counts_by_bar": sv["counts_by_bar"], "rows": sv["survivor_rows_[config,tier1,tier2]"],
            "clean_accuracy_composition": sv["clean_accuracy_composition"]},
    status="checked against code and recorded aggregates",
    scope="Adult Tier-2 values come from mi_ceiling.json via number_reports.json, not from the audit suite.",
    previous_assessment_value="R28 match; R27 clean accuracies 'constructed'",
    corrected_value_or_confirmation="5/8 CONFIRMED at 0.55, but bar-fragile: 8/8 at 0.52, 0/8 at 0.60; two of the five exceed the bar by "
        "<= 0.0051 (0.5519, 0.5551). Clean accuracies 0.915/0.619/0.895 CONFIRMED as composed (fresh-partition eval-half majority + in-sample lift), not measured."))

gt = mc["gate_5seed_recount"]
items.append(dict(
    id="F8", question="Supported-class 5-seed gate (49.7% / 76.2% quoted) recomputed from per-seed shards",
    inputs=ins("F_multiclass_worstpair.json", "F_multiclass_worstpair.py"),
    command=PY + "F_multiclass_worstpair.py (gate block)",
    result={"max_abs_delta_vs_gate_5seed.json": {k: v["max_abs_delta_vs_gate_5seed"] for k, v in gt.items()},
            "util_5seed": {"easy": 41.0, "middle": 77.4}, "paper_quotes": {"easy": 49.7, "middle": 76.2}},
    status="independently recomputed",
    scope="Recomputes AUC aggregates from the 10 per-seed shards (drive analysis/gate_shards); utility read from gate_5seed.json.",
    previous_assessment_value="R24 / F-13: 3-seed 49.7% quoted while 5-seed gate reads 41.0%",
    corrected_value_or_confirmation="CONFIRMED exactly (delta 0.0): AUC aggregates reproduce; the 5-seed easy utility is 41.0% vs the 49.7% "
        "(3-seed) printed at paper.tex:748; the registered G4 'within 3 pp' prediction is FALSIFIED in the stored scoring."))

ours = {f"{o['cell']}_T{o['tier']}": o for o in t1["ours_e2e_fullrank_certified_points"]}
items.append(dict(
    id="F9", question="Aggregation order for the full-rank Tier-2 middle point (0.5496 pass vs 0.553)",
    inputs=ins("F_table1_bars.json", "F_table1_bars.py"),
    command=PY + "F_table1_bars.py (ours block)",
    result={"middle_T2": {k: ours["middle_T2"][k] for k in ("rep_max_of_means", "mean_of_per_seed_max", "worst_seed")},
            "all_points": {k: {kk: round(v[kk], 4) for kk in ("rep_max_of_means", "mean_of_per_seed_max", "worst_seed")} for k, v in ours.items()}},
    status="independently recomputed",
    scope="5 training seeds x per-seed attacker AUCs stored in two_tier_certification.json.",
    previous_assessment_value="F-07 / R16,R23: 'middle full-rank T2 0.5496 (mean-then-max) passes vs 0.553 (max-then-mean) fails'",
    corrected_value_or_confirmation="CORRECTED: max-then-mean also gives 0.5496 (the LRT is the max in every seed); 0.5533 is the WORST SINGLE "
        "SEED. The point passes under both aggregation orders and fails only under a worst-seed (max over seeds) rule."))

items.append(dict(
    id="F10", question="FARE 'leak race pairs at 0.603-0.610' (paper.tex:692-694)",
    inputs=ins("G_aaai_scope_classification.json", "G_aaai_scope_classification.py"),
    command=PY + "G_aaai_scope_classification.py",
    result=[x for x in g["items"] if x["certificate"].startswith("FARE")][0]["extra"],
    status="checked against code and recorded aggregates",
    scope="fare_gauntlet.json middle cell, seed-0 sweep rows.",
    previous_assessment_value="R30 / F-04: values are macro-OvR T1 AUCs of non-certified sweep rows",
    corrected_value_or_confirmation="CONFIRMED: 0.6028/0.6098/0.6058 are macro-OvR representation Tier-1 maxima of three seed-0 sweep rows with "
        "dp_ub=0.000 (14/14 available configs read 0.000); not pairwise AUCs, not certified points."))

for k, x in enumerate(g["items"], 1):
    items.append(dict(
        id=f"G-A{k}", question=f"Scope of certificate / claim: {x['certificate']}",
        inputs=ins("G_aaai_scope_classification.json", "G_aaai_scope_classification.py"),
        command=PY + "G_aaai_scope_classification.py",
        result={"stated_scope": x["stated_scope"], "failure_reported": x["failure_reported"], "extra": x.get("extra")},
        status="checked against code and recorded aggregates",
        scope="Classification by statement scope vs attack class; numbers pulled from stored JSON; no theorem re-proved.",
        previous_assessment_value={1: "F-15 certificate in-sample ridge R2", 2: "Table 1 LEACE row", 3: "Fair PCA (not assessed)",
                                   4: "F-04 FARE claim reworded", 5: "FNF (not assessed)", 6: "Obliviator row", 7: "F-04 LAFTR-official binary code on 5-class",
                                   8: "F-18 Prop 1 sound", 9: "F-16 Prop 2 over-generalized", 10: "F-17 Prop 3 scope; unclipped points not covered",
                                   11: "Stadler population claim"}.get(k, ""),
        corrected_value_or_confirmation=x["classification"]))

os.makedirs(os.path.dirname(PART), exist_ok=True)
json.dump(items, open(PART, "w"), indent=1, default=float)
print(PART, len(items))
