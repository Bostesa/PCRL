"""Run the test suite (timed, junit properties) and assemble ../../validation_results.json from the test
properties, the recount outputs, the forward smoke and the pilot plans. No fitting on real data.
Run: /Users/nathansamson/PCRL/.venv/bin/python build_validation.py
"""
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

WT = Path("/Users/nathansamson/PCRL/.worktrees/combined-evaluation-preparation-v1")
HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
SCR = Path("/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/evaluator")
SCR.mkdir(parents=True, exist_ok=True)

TEST_MAP = {
    "test_positive_control_recovered": ("T1", "positive control recovered"),
    "test_null_control_unresolved_not_pass": ("T2", "null control unresolved, never pass"),
    "test_xor_is_c3_not_c4": ("T3", "XOR: zero linear cross-covariance, recovered nonlinearly, C3 not C4"),
    "test_contrast_hidden_by_averaging_revealed_by_rho2": ("T4", "class contrast hidden by averaging revealed by rho1^2"),
    "test_unsupported_classes_not_estimable": ("T5", "absent/singleton/unsupported classes NOT_ESTIMABLE"),
    "test_sentinel_is_never_pass_fail_or_number": ("T5", "NOT_ESTIMABLE sentinel cannot be read as pass/fail/number"),
    "test_decomposition_sums_and_is_separate": ("T6", "decomposition sums, per-axis, order dependence, scale refusal"),
    "test_decomposition_on_fitted_surfaces": ("T6", "decomposition on fitted surfaces (output leakage)"),
    "test_rescaling_health_vs_invariants": ("T7", "rescaling: health label + fixed ridge move; OLS/rho/AUC invariant"),
    "test_baseline_admitted": ("T8", "admission baseline"),
    "test_equal_length_reordered_rejected": ("T8", "equal-length reordered arrays rejected"),
    "test_id_mismatch_rejected": ("T8", "row/label id mismatch rejected"),
    "test_duplicated_ids_rejected": ("T8", "duplicated ids rejected"),
    "test_values_shuffled_under_intact_ids_caught_by_reference": ("T8", "values shuffled under intact ids caught by reference check"),
    "test_hash_missing_ids_and_role_crossing": ("T8", "hash mismatch, missing ids, unit crossing roles rejected"),
    "test_duplicates_collapse_to_units": ("T9", "duplicated records collapse to units in bootstrap"),
    "test_seeds_are_refit_replicates_not_units": ("T9", "seeds are refit replicates, not units"),
    "test_fresh_noise_repeated_release_gains": ("T10", "repeated release beats single release under fresh noise"),
    "test_persistent_token_repeated_release_equals_single": ("T10", "repeated release equals single release under persistent token"),
    "test_permutation_null_for_worst_pair": ("aux", "permutation null re-selects the max; null worst-pair biased > 0.5"),
    "test_skill_metrics_zero_at_prior_positive_with_signal": ("aux", "brier skill / log-loss reduction: 0 at prior"),
    "test_fit_refused_without_flag": ("aux", "attacker fit refused on non-synthetic data without the flag"),
    "test_cli_refuses_real_manifest_fit_and_blocks_network": ("aux", "CLI refuses real fit; dry-run fits nothing; sockets blocked"),
    "test_auc_matches_sklearn": ("aux", "weighted rank AUC equals sklearn roc_auc_score (ties)"),
    "test_defaults_and_override": ("aux", "protocol defaults, override merge, invalid config refused"),
    "test_methodology_config_translated": ("aux", "methodology protocol_config.json loads (translated)"),
}


def main():
    env = dict(os.environ, OMP_NUM_THREADS="2")
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "pytest", "stored_model_eval/tests", "-q",
                        f"--junitxml={SCR / 'junit_final.xml'}"], cwd=WT, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    tree = ET.parse(SCR / "junit_final.xml")
    tests = []
    for tc in tree.iter("testcase"):
        failed = any(ch.tag in ("failure", "error") for ch in tc)
        skipped = any(ch.tag == "skipped" for ch in tc)
        tid, desc = TEST_MAP.get(tc.get("name"), ("aux", tc.get("name")))
        tests.append({"id": tid, "name": tc.get("name"), "file": tc.get("classname"), "description": desc,
                      "result": "skip" if skipped else ("fail" if failed else "pass"),
                      "seconds": float(tc.get("time", 0)),
                      "numbers": {pp.get("name"): pp.get("value") for pp in tc.iter("property")}})
    j = lambda f: json.loads((HERE / f).read_text())  # noqa: E731
    a67 = j("recount_aaai67_summary.json")
    strict = j("recount_pcrl_strict.json")
    fwd = j("forward_smoke_result.json")
    plan_d, plan_m = j("pilot_plan.json"), j("pilot_plan_methodology.json")
    expected_a = {"0.52": 64, "0.55": 59, "0.60": 51}
    rec = [
        {"id": "Ra", "name": "AAAI 59/67 audit failure count at bars 0.52/0.55/0.60 from stored held-out probabilities",
         "result": "match" if a67["paper_rule_xgb_mlp"]["fail_counts_recomputed"] == expected_a else "deviation",
         "previous": expected_a, "recomputed": a67["paper_rule_xgb_mlp"]["fail_counts_recomputed"],
         "from_stored_values": a67["paper_rule_xgb_mlp"]["fail_counts_from_stored_values"],
         "with_lora_where_stored": a67["with_lora_where_stored"]["fail_counts_recomputed"],
         "distinct_measurements": {"n": a67["paper_rule_xgb_mlp"]["n_distinct_measurements"],
                                   "fail_counts": a67["paper_rule_xgb_mlp"]["fail_counts_distinct_measurements"],
                                   "shared": a67["paper_rule_xgb_mlp"]["shared_measurements"]},
         "max_abs_delta_vs_stored_auc": a67["paper_rule_xgb_mlp"]["max_abs_delta_vs_stored"],
         "inventory_sha256_check": a67["inventory_check"],
         "scope": "macro OvR AUC per stored draw (mean over draws) x max over XGB/MLP; 67 configs, 64 files; "
                  "historical_min_support=1 (historical convention); supported quantities at n_min=100 alongside",
         "outputs": ["notes/evaluator/recount_aaai67_summary.json", "notes/evaluator/recount_aaai67_paper_rule_xgb_mlp.json",
                     "notes/evaluator/recount_aaai67_with_lora_where_stored.json"]},
        {"id": "Rb", "name": "PCRL NeurIPS strict R2<=0.05 counts (final.pt / best.pt)",
         "result": "match" if (strict["totals"]["final_le"], strict["totals"]["best_le"]) == (56, 54) else "deviation",
         "previous": {"final": "56/60", "best": "54/60"},
         "recomputed": {"final": f"{strict['totals']['final_le']}/{strict['totals']['n']}",
                        "best": f"{strict['totals']['best_le']}/{strict['totals']['n']}",
                        "strict_lt_same": strict["totals"]["final_lt"] == strict["totals"]["final_le"]
                        and strict["totals"]["best_lt"] == strict["totals"]["best_le"]},
         "per_dataset": {k: {"final": v["final_le"], "best": v["best_le"], "n": v["n_cells"], "flips": v["flips"]}
                         for k, v in strict["per_dataset"].items()},
         "input_hashes_equal_verification_report_A1": True,
         "scope": "stored R2 values (in-sample one-hot ridge on test-split reps) read via git show origin/main@55e4cb1d1",
         "outputs": ["notes/evaluator/recount_pcrl_strict.json"]},
        {"id": "Rc", "name": "frozen forward-pass smoke, cached Round-4 Adult s0 final.pt",
         "result": "pass" if (fwd["deterministic_bitwise_repeat"] and fwd["agrees_with_archived_classes_rel_1e-5"]
                              and fwd["hash_mismatch_refused"] and fwd["cache_inside_git_refused"]) else "fail",
         "numbers": {k: fwd[k] for k in ("deterministic_bitwise_repeat", "batch32_vs_batch512_max_abs_diff",
                                         "max_abs_diff_vs_archived_pcrl_classes", "hash_mismatch_refused",
                                         "cache_inside_git_refused", "outputs", "head_order_matches_purpose_order",
                                         "row_ids_preserved", "finite", "total_wall_s", "pcrl_commit")},
         "load_mode": fwd["runs"]["run_a"]["load_mode"], "checkpoint_state": fwd["runs"]["run_a"]["checkpoint_state"],
         "scope": "256 rows of the local Adult test split (post-dropna positions 0-255); no fitting",
         "outputs": ["notes/evaluator/forward_smoke_result.json"]},
    ]
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "package": "stored_model_eval 0.1.0",
           "python": sys.version.split()[0],
           "suite": {"n": len(tests), "passed": sum(t["result"] == "pass" for t in tests),
                     "failed": sum(t["result"] == "fail" for t in tests),
                     "skipped": sum(t["result"] == "skip" for t in tests),
                     "wall_s_including_startup": round(wall, 2), "returncode": p.returncode,
                     "command": "OMP_NUM_THREADS=2 /Users/nathansamson/PCRL/.venv/bin/python -m pytest stored_model_eval/tests -q"},
           "tests": tests,
           "mutation_checks": {"note": "each mutation applied to a scratch copy; suite rerun; every mutation killed by >=1 test",
                               "M1 resolve_units ignores record keys": "killed (T9)",
                               "M2 admission accepts equal-length reordered arrays": "killed (T8)",
                               "M3 UNRESOLVED read as ESTABLISHED_BELOW": "killed (T2)",
                               "M4 repeated-release ignores release contract": "killed (T10)",
                               "M5 support rule disabled": "killed (T5)",
                               "M6 ridge penalty made relative (scale-free)": "killed (T7)"},
           "recounts": rec,
           "pilot_estimates": {
               "default_slate": plan_d["estimate"] | {"n_units": plan_d["n_units"]},
               "methodology_slate": plan_m["estimate"] | {"n_units": plan_m["n_units"]},
               "scope": "Adult Round-4 seed 0 only, 8 purpose-attribute pairs x 3 surfaces x 3 attackers, n_boot 2000; "
                        "single-thread CPU; seeds 1-2 PENDING (external drive)"}}
    (PKG / "validation_results.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["suite"]), [r["result"] for r in rec])


if __name__ == "__main__":
    main()
