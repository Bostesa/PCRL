"""Assemble notes/verification/independent_checks.json from fixture outputs.

Reads outputs/*.json written by the fixtures; hashes every input (fixture
scripts, outputs, git blobs that were inspected, stored result files).
"""
import hashlib
import json
import subprocess
from pathlib import Path

FX = Path(__file__).resolve().parent
OUT = FX / "outputs"
NOTES = FX.parent / "notes" / "verification"
PCRL = "/Users/nathansamson/PCRL"
DG = "/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/dg"
SY = "/opt/homebrew/bin/python3"
VENV = "/Users/nathansamson/PCRL/.venv/bin/python"


def fsha(p):
    return {"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}


def gsha(ref, path, repo=PCRL):
    b = subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], capture_output=True, check=True).stdout
    rev = subprocess.run(["git", "-C", repo, "rev-parse", "--short=9", ref], capture_output=True, text=True).stdout.strip()
    return {"path": f"{repo}@{ref}({rev}):{path}", "sha256": hashlib.sha256(b).hexdigest()}


def j(name):
    return json.loads((OUT / name).read_text())


def fx(n, extra=()):
    return [fsha(FX / n), fsha(OUT / (n.split("_")[0] + ".json"))] + list(extra)


def main():
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10 = (j(f"F{i:02d}.json") for i in range(1, 11))
    OM, OS = j("orig_origin_main.json"), j("orig_research_pcrl-submission-finish-v1.json")
    om_ref, os_ref = "origin/main@55e4cb1d1", "research/pcrl-submission-finish-v1@55c0c5a35"
    checks = []
    I = F1["in_sample_float64"]
    checks.append({
        "id": "F1", "title": "One-hot aggregation identity ('Convex-Combination Identity', NeurIPS draft Proposition 4)",
        "question": "Is aggregate one-hot R2 exactly the variance-weighted average of per-class OvR R2, with w_k = pi_k(1-pi_k)/sum_j pi_j(1-pi_j); when does it fail; how large is the float32 discrepancy?",
        "input_files": fx("F01_onehot_identity.py", [
            gsha("origin/main", "pcrl/evaluation/certificates.py"), gsha("origin/main", "pcrl/purposes/verification.py"),
            gsha("research/pcrl-submission-finish-v1", "pcrl/evaluation/certificates.py"),
            gsha("research/pcrl-submission-finish-v1", "pcrl/purposes/verification.py"),
            fsha(OUT / "orig_origin_main.json"), fsha(OUT / "orig_research_pcrl-submission-finish-v1.json")]),
        "command": f"{SY} F01_onehot_identity.py > outputs/F01.json ; {VENV} orig_compare.py --root <export of ref> --label <ref>",
        "result": {
            "identity_in_sample_float64_residual": I["identity_residual"],
            "weights_written_out": {"pi": I["pi"], "w_k": I["weights_from_priors"], "per_class_r2": I["per_class_r2"],
                                     "aggregate": I["r2_aggregate_direct"], "sum_w_r2": I["r2_convex_combination"]},
            "max_abs_diff_weights_sstot_vs_priors": I["max_abs_weight_difference"],
            "held_out_residual_test_weights": F1["held_out"]["residual_test_weights_unclipped"],
            "held_out_residual_train_weights": F1["held_out"]["residual_train_weights"],
            "held_out_mixed_sign_residual_after_pcrl_style_clipping": F1["held_out_mixed_sign"]["residual_after_pcrl_clipping"],
            "float32_benign_abs_diff": F1["float32_benign"]["abs_diff"],
            "float32_stress": [{k: c[k] for k in ("tail_sd", "offset", "r2_float64", "r2_float32_gram_clipped_at_0")} for c in F1["float32_stress"]["cases"]],
            "original_origin_main": {"residual_float64_input": OM["F1"]["residual_float64"], "residual_float32_input": OM["F1"]["residual_float32"],
                                     "stress_cert_r2": [s["cert_r2_on_float32_H"] for s in OM["F1"]["stress"]],
                                     "stress_cert_certified_at_0.05": [s["cert_certified_at_0.05"] for s in OM["F1"]["stress"]],
                                     "stress_dominant_axis_float64": [s["dominant_axis_float64_per_class"][0] for s in OM["F1"]["stress"]]},
            "original_submission_finish": {"residual_float64_input": OS["F1"]["residual_float64"], "residual_float32_input": OS["F1"]["residual_float32"],
                                           "stress_cert_r2": [s["cert_r2_on_float32_H"] for s in OS["F1"]["stress"]]},
            "stored_audits_cert_vs_float64_identity": F8["pcrl_encoder"]["dominant_axis_audits_final_epoch"]["certificate_minus_float64_identity"],
        },
        "verdict": "confirmed",
        "scope": ("The identity is an exact algebraic identity (residual ~1e-15) for any column-wise least-squares fit (OLS or ridge), "
                  "in or out of sample, provided the weights use the variances of the SAME rows that are scored (train-set weights on test "
                  "R2 leave a residual 3.6e-4 here), every class has positive variance, and per-class R2 are not clipped (PCRL clips max(0,.) "
                  "per class; out of sample with mixed signs this breaks the identity, residual -0.041 in the fixture). It is a weighting "
                  "identity only: it says nothing about the max over linear contrasts (F2). The nonzero 'residuals' PCRL reports "
                  "(max 0.0021 multiclass, 0.0032 on a binary row) are not tolerance of an approximate identity but a numerical artifact: "
                  "origin/main LinearComplianceCertificate accumulates the Gram matrix in float32 when given float32 representations while "
                  "the dominant-axis code casts to float64; the certificate is never above the float64 value and understates it by up to "
                  "50% relative in stored audits (no pass/fail flips at tau=0.05). In a constructed ill-conditioned stress case the "
                  "origin/main certificate returns 0.0 (certified) where the float64 R2 is 0.578. submission-finish-v1 casts to float64 "
                  "and agrees with the independent value. The stress case is synthetic; it does not show that any stored PCRL verdict flips."),
        "original_implementation_compared": {"compared": True, "refs": [om_ref, os_ref],
                                             "functions": ["LinearComplianceCertificate.check", "compute_dominant_axis_r2"]},
    })
    c = F2["cases"]
    checks.append({
        "id": "F2", "title": "Dominant axis (max per-class R2) versus full categorical diagnostic (first canonical correlation)",
        "question": "Can every class-indicator R2 be small while a linear class contrast is highly recoverable; what does PCRL's dominant axis compute?",
        "input_files": fx("F02_multiclass_contrast.py", [gsha("origin/main", "pcrl/evaluation/certificates.py"),
                                                          gsha("research/pcrl-submission-finish-v1", "pcrl/evaluation/certificates.py"),
                                                          fsha(OUT / "orig_origin_main.json")]),
        "command": f"{SY} F02_multiclass_contrast.py > outputs/F02.json ; {VENV} orig_compare.py ...",
        "result": {"definition_dominant_axis": "max_k in-sample ridge R2 of 1{y=k} (origin/main pcrl/evaluation/certificates.py:47-117)",
                   "definition_canonical": "lambda_max(S_yy^-1/2 S_yz (S_zz+eps I)^-1 S_zy S_yy^-1/2), one reference class dropped, eps=1e-10 tr(S_zz)/d, eigen floor 1e-12; invariant to dropped class (max diff 3e-14)",
                   "proved_bound": "rho2 <= (K-1) * max_k R2_k (Cauchy-Schwarz); 300 random configurations: max ratio 1.000, never violated",
                   **{name: {k: v[k] for k in ("per_class_r2", "dominant_axis_max", "onehot_r2", "canonical_rho2",
                                                "gap_ratio_rho2_over_DA", "passes_tau_0.05", "optimal_contrast_normalised")} for name, v in c.items()},
                   "original_dominant_axis_origin_main": {k: v["r2_da"] for k, v in OM["F2"].items()},
                   "original_dominant_axis_submission_finish": {k: v["r2_da"] for k, v in OS["F2"].items()}},
        "verdict": "partially",
        "scope": ("Confirmed: the dominant axis is a max over the K indicator directions only and can miss contrast leakage: K=10 balanced, "
                  "every indicator R2 <= 0.047 and one-hot R2 = 0.041 (both pass tau=0.05) while the half-vs-half contrast has rho2 = 0.364. "
                  "Refuted as stated for K=3: 'every indicator small but a contrast highly recoverable' is impossible because rho2 <= (K-1) R2_DA "
                  "= 2 R2_DA; the e1-e2 contrast gives gap 1.33 (balanced) and 1.71 (imbalanced 0.8/0.1/0.1). The canonical diagnostic is still "
                  "only linear and in-sample; it bounds all linear contrasts, not nonlinear recoverability (F7 XOR). The original implementations "
                  "reproduce the independent per-class values exactly (both refs)."),
        "original_implementation_compared": {"compared": True, "refs": [om_ref, os_ref], "functions": ["compute_dominant_axis_r2", "LinearComplianceCertificate.check"]},
    })
    o3, s3 = OM["F3"], OS["F3"]
    checks.append({
        "id": "F3", "title": "Unsupported (absent / singleton / constant) classes",
        "question": "What do the original (origin/main) and fixed (submission-finish-v1) scorers return when a declared class is absent, constant, or a singleton?",
        "input_files": fx("F03_unsupported_classes.py", [gsha("origin/main", "pcrl/evaluation/certificates.py"), gsha("origin/main", "pcrl/purposes/verification.py"),
                                                          gsha("research/pcrl-submission-finish-v1", "pcrl/evaluation/certificates.py"),
                                                          gsha("research/pcrl-submission-finish-v1", "pcrl/purposes/verification.py"),
                                                          gsha("research/pcrl-submission-finish-v1", "tests/test_scoring_support.py"),
                                                          fsha(OUT / "orig_origin_main.json"), fsha(OUT / "orig_research_pcrl-submission-finish-v1.json")]),
        "command": f"{SY} F03_unsupported_classes.py > outputs/F03.json ; {VENV} orig_compare.py ...",
        "result": {
            "independent_report": {k: {"supports": [r["support"] for r in v["per_class"]], "status": [r["status"] for r in v["per_class"]],
                                       "dominant_axis": v["dominant_axis"]} for k, v in F3["cases"].items()},
            "origin_main": {k: {"cert_r2": v["cert_default_schema"].get("r_squared"), "cert_certified": v["cert_default_schema"].get("certified"),
                                "da_r2": v["da_default_schema"].get("r2_da"), "da_per_class": v["da_default_schema"].get("per_class_r2")} for k, v in o3.items()},
            "submission_finish_default_schema": {k: {"cert_r2": v["cert_default_schema"].get("r_squared"), "cert_certified": v["cert_default_schema"].get("certified"),
                                                     "coverage_complete": v["cert_default_schema"].get("coverage_complete"), "da_r2": v["da_default_schema"].get("r2_da")} for k, v in s3.items()},
            "submission_finish_declared_K4": {k: {"cert_r2": v["cert_declared_K4"].get("r_squared"), "cert_certified": v["cert_declared_K4"].get("certified"),
                                                  "da_r2": v["da_declared_K4"].get("r2_da"), "observed_r2_da": v["da_declared_K4"].get("observed_r2_da")} for k, v in s3.items()},
        },
        "verdict": "confirmed",
        "scope": ("origin/main: schema is inferred as max(label)+1, so an absent top class silently disappears and the one-hot certificate passes; "
                  "an absent middle class contributes a zero column (certificate passes, 0.035) while the dominant axis reports R2 = 1.0 for the "
                  "absent class (0/0 guarded by 1e-12 -> 1); all-constant labels give certificate R2 = 1.0 (fails, for the wrong reason). "
                  "submission-finish-v1: absent/constant classes give NaN and fail closed, BUT only if the caller passes num_classes for an absent "
                  "TOP class (default schema still certifies 0.019 with coverage_complete=True), and a singleton class is treated as valid: its "
                  "in-sample R2 is a single-row leverage statistic (0.020 for an ordinary row, 0.654 when that row is an outlier, pure-noise H). "
                  "A correct report gives counts and 'not estimable' (fixture rule: fewer than d+2 positives). Code not repaired."),
        "original_implementation_compared": {"compared": True, "refs": [om_ref, os_ref], "functions": ["LinearComplianceCertificate.check", "compute_dominant_axis_r2"]},
    })
    p1, p2 = F4["part1_two_recipients"], F4["part2_appended_channel"]
    pf = "research/pcrl-privacy-first-selector-v1"
    checks.append({
        "id": "F4", "title": "Nested singleton / ancestor predictors in coalition and appended-channel audits",
        "question": "Without nested predictors can a coalition report lower recovery than one view (spurious negative combination effect), and do the ACS coalition audits include nested predictors?",
        "input_files": fx("F04_nested_slate.py", [gsha(pf, "experiments/acs_coalition_audits.py"), gsha(pf, "experiments/acs_fixed_predictions_audits.py"),
                                                   gsha(pf, "experiments/pcrl_task_aligned_cuts_v1/audit.py"), gsha(pf, "experiments/pcrl_stochastic_channel_v1/slate.py"),
                                                   gsha(pf, "experiments/pcrl_stochastic_replacement_overnight_v1/replacement.py")]),
        "command": f"{SY} F04_nested_slate.py > outputs/F04.json ; git grep -n -i 'slate\\|ancestor\\|singleton\\|nested' <ref> -- experiments pcrl",
        "result": {"two_recipients": {s: {"A_alone_auc": v["A_alone"]["test_auc"], "AB_without_nested_auc": v["AB_without_nested"]["test_auc"],
                                          "AB_with_nested_auc": v["AB_with_nested"]["test_auc"], "effect_without": v["combination_effect_without_nested"],
                                          "effect_with": v["combination_effect_with_nested"], "paired_CI_without": v["paired_CI_AB_without_minus_A"]} for s, v in p1.items()},
                   "appended_channel": {s: {"J_auc": v["J_condition"]["test_auc"], "ext_H_only_auc": v["extension_H_only_ancestors"]["test_auc"],
                                            "ext_H_and_J_auc": v["extension_H_and_J_ancestors"]["test_auc"], "increment_H_only": v["increment_H_only"],
                                            "increment_H_and_J": v["increment_H_and_J"], "paired_CI_H_only": v["paired_CI_increment_H_only"]} for s, v in p2.items()},
                   "source_inspection": {
                       "coalition_singletons_included": f"{pf}:experiments/acs_coalition_audits.py:180-196 (inherit_singletons adds every A/B singleton candidate, column-projected, to AB); called at :310; parity asserted :318-327; metadata flag :337. Same file on shared-context-release-v1 (537e74a44) and task-directed-release-v1 (f4bdf4cd5).",
                       "fixed_predictions_H_anchor_ancestors": f"{pf}:experiments/acs_fixed_predictions_audits.py:295-308 (ancestor H candidates routed into every role) then inherit_singletons :313",
                       "route_bank": f"{pf}:experiments/pcrl_task_aligned_cuts_v1/audit.py:449-492 (own + same-role H-only ancestors; AB adds A same-release and B H-only); docstring :456 'J is a comparator, never an appended ancestor'",
                       "appended_channel_gap_documented_in_repo": f"{pf}:experiments/pcrl_stochastic_channel_v1/slate.py:1-21 states pcrl_utility_extension_v1 routed only H-only ancestors and audited ref_J as a separate condition, so 'nothing in the extension's own slate recovers a J-only predictor'; slate.py:84-110 (add_j_anchors) adds ignore-R J anchors",
                       "replacement_study": f"{pf}:experiments/pcrl_stochastic_replacement_overnight_v1/replacement.py:160-175 ancestor_inclusive best-of(view, H ancestor); final_prospective_v1 freeze.py:111-113 'J never an ancestor of a replacement release'"}},
        "verdict": "confirmed",
        "scope": ("Fixture: B is pure noise (150 dims, 300 fit rows), so the population combination effect is 0, yet the coalition slate without nested "
                  "predictors reports -0.16 to -0.20 AUC versus A alone (paired CIs exclude 0); adding column-projected singleton fits restores equality. "
                  "Appended channel: H-only ancestors leave an increment of -0.06 to -0.14 (spurious 'extension lowers recovery below J'); adding H+J "
                  "ancestors restores 0. The fixture magnitudes depend on the deliberately high noise dimension and are illustrative only. Source: the "
                  "ACS coalition audits on the three named refs DO include nested singletons and H ancestors; the appended-channel (J ancestor) gap "
                  "existed in pcrl_utility_extension_v1 and is documented and patched (for later studies) in pcrl_stochastic_channel_v1/slate.py. Whether "
                  "any stored utility-extension verdict changes was not rerun."),
        "original_implementation_compared": {"compared": False, "refs": [f"{pf}@8fdc61e39", "research/pcrl-shared-context-release-v1@537e74a44", "research/pcrl-task-directed-release-v1@f4bdf4cd5"],
                                             "note": "source inspection only; ACS audit code not executed (needs ACS data)"},
    })
    mc, an = F5["monte_carlo"], F5["analytic_auc"]
    checks.append({
        "id": "F5", "title": "Defense-aware attacker versus pre-noise (insider) access versus query averaging",
        "question": "Does knowing sigma give access to clean h; which of (a)-(d) does durable-guarantees Tier 2 implement?",
        "input_files": fx("F05_defense_aware_access.py", [fsha(Path(DG) / p) for p in ("utils/battery.py", "experiments/two_tier_certification.py",
                                                                                         "experiments/averaging_attack.py", "experiments/run_knows_q.py",
                                                                                         "experiments/mi_ceiling.py", "README.md", "results/averaging_attack.json", "results/knows_q.json")]),
        "command": f"{SY} F05_defense_aware_access.py > outputs/F05.json",
        "result": {"analytic_auc": an, "monte_carlo_auc": {k: v["auc"] for k, v in mc.items()},
                   "tier2_mapping": {
                       "implemented": "(a) defense-aware single-release attacker with clean TRAINING statistics: utils/battery.py:60-78 fits N(mu_k, Sigma_k + Sigma_noise) on clean train rows and scores the EXPOSED (single noised draw) held-out rows; gaussian_lrt :88-103; used by two_tier_certification.py:117-128 (full_point) with P from mi_ceiling.train_channel (one fresh-noise draw per row, mi_ceiling.py:251-262: P = h + sigma*randn, logits L = D(P)) and noise_cov = sigma^2 I (two_tier_certification.py:147)",
                       "not_implemented_by_tier2": "(b) clean test-time h; (c) multi-query averaging; (d) persistent noise",
                       "averaging": "(c) implemented separately in experiments/averaging_attack.py:10-16,110-124 (Tier-1 learners on the mean of N fresh draws); stored results/averaging_attack.json: T1 max 0.506 (N=1) -> 0.560 (N=16), breaching 0.55 at N=16",
                       "knows_q": "experiments/run_knows_q.py:1-7: for SUBSPACE channels knowing Q recovers h_perp exactly (stored max_recon_err 2.3e-5); knows-Q MLP 0.982 vs Tier-2 LRT 0.856 on the easy cell (results/knows_q.json) -> the Gaussian LRT is not an upper bound for defense-aware attackers on non-isotropic channels",
                       "claim_mismatch": "README.md:46-49 and utils/battery.py:15-21 describe Tier 2 as an attacker who 'has seen the representation before noise was added ... or anyone who can query the same row repeatedly and average'; the implemented LRT has neither test-time clean access nor averaging."}},
        "verdict": "confirmed",
        "scope": ("Gaussian toy (d=4, delta=1, tau=1, sigma=2): one-release defense-aware attackers all sit at the analytic 0.624 (logit on r 0.620, clean-fit LRT 0.620, "
                  "deconvolution LRT from released data only 0.620), insider clean h 0.756 (analytic 0.760), fresh averaging N=4/16 0.688/0.731 (0.691/0.736), "
                  "persistent noise N=16 0.621. Knowing sigma does not give access to h; clean training statistics only help estimation, not the test-time channel. "
                  "For isotropic noise the implemented Tier 2 is approximately Bayes-optimal among single-release attackers, so it is a valid (a); "
                  "it is not (b) or (c), and for subspace channels it is beaten by the knows-Q attacker."),
        "original_implementation_compared": {"compared": False, "refs": ["durable-guarantees@956f5c8"], "note": "source inspection + stored JSON; dg LRT reimplemented independently, not imported"},
    })
    af = F6["across_fits"]
    checks.append({
        "id": "F6", "title": "Accounting for an appended constant channel",
        "question": "Population increment is zero; can finite-sample fresh fits show nonzero/negative differences, and does paired uncertainty cover zero?",
        "input_files": fx("F06_constant_extension.py"),
        "command": f"{SY} F06_constant_extension.py > outputs/F06.json",
        "result": {"auc_diffs_ext_minus_base": F6["auc_diffs_ext_minus_base"], "n_negative": F6["n_negative_ext_minus_base"],
                   "per_seed_paired_people_ci_covers_zero_logloss": [v["ext_minus_base"]["covers_zero"] for v in F6["per_seed"].values()],
                   "per_seed_paired_people_auc_ci": [v["ext_minus_base"]["auc_diff_paired_boot_ci95"] for v in F6["per_seed"].values()],
                   "across_fits": af, "sd_ext_minus_base": F6["sd_ext_minus_base"], "sd_refit_control": F6["sd_refit_control"]},
        "verdict": "partially",
        "scope": ("Confirmed that a constant column (zero population increment) produces nonzero, mostly negative AUC differences (5/6 seeds; up to -0.0026) "
                  "and changes every person's loss. Partially: paired CIs over shared evaluation people alone cover zero in only 4/6 seeds (log loss) "
                  "and 5/6 (AUC) because they condition on the two fitted models; uncertainty over refits (6 seeds) covers zero for both AUC and log loss, "
                  "and the constant-extension spread (sd 0.0018) equals a pure refit control (sd 0.0020). Accounting must include fit randomness, "
                  "not only paired evaluation-sample uncertainty."),
        "original_implementation_compared": {"compared": False, "refs": []},
    })
    checks.append({
        "id": "F7", "title": "Positive and null controls for an adaptive attacker slate, with replay",
        "question": "Does a validation-selected slate recover a directly included attribute, an XOR, and a contrast missed by max-per-class; does it stay at chance on a null; does a second seed agree?",
        "input_files": fx("F07_slate_controls.py", [fsha(OUT / "F07_first_attempt_noise1.35.json")]),
        "command": f"{SY} F07_slate_controls.py > outputs/F07.json",
        "result": {s: {k: {"selected": v["selected"], "auc": v["selected_test_auc"], "ci95": v["ci95"],
                           **({"logit_auc": v["all"]["logit"]["test_auc"]} if k == "C2_xor" else {}),
                           **({"dominant_axis_r2": v["dominant_axis_r2_on_test_rows"], "half_membership_auc": v["half_membership_auc_from_selected_probs"]} if k.startswith("C3") else {})}
                       for k, v in F7[s].items()} for s in ("seed0", "seed1_replay")} | {"verdicts_seed0": F7["verdicts_seed0"], "replay_agrees": F7["replay_agrees"]},
        "verdict": "confirmed",
        "scope": ("All four controls behave as designed on two independent seeds. Disclosed: the first C3 design (noise 1.35) put the seed-1 dominant-axis "
                  "value at 0.0509, on the 0.05 threshold, so replay disagreed; the planted noise was raised to 1.5 (fixture design, not an empirical "
                  "claim) and the first output is kept. These controls validate a slate on synthetic data only; they do not certify the ACS or encoder slates."),
        "original_implementation_compared": {"compared": False, "refs": []},
    })
    H8, G8, C8, P8, E8 = F8["honest_reaudit"], F8["master_gauntlet"], F8["continuous_cost"], F8["pcrl_encoder"], F8["erase_layer_pilot"]
    checks.append({
        "id": "F8", "title": "Independent recount of headline counts from stored results",
        "question": "Do stored per-row results reproduce 18/21 -> 3 survivors, 1/36, r/rho, and PCRL 56/60 and pilot 60/60, including final-vs-best?",
        "input_files": [fsha(FX / "F08_recount_headlines.py"), fsha(OUT / "F08.json")] + F8["input_files"] + [fsha(Path(DG) / "README.md")],
        "command": f"{SY} F08_recount_headlines.py <dg clone> <export of origin/main results> /Users/nathansamson/PCRL/results/rebuttal/erase_layer_pilot_aws > outputs/F08.json",
        "result": {"honest_reaudit": {k: H8[k] for k in ("rows_total", "rows_old_verdict_breach", "rows_old_verdict_stopped", "survivors_stored_flag",
                                                         "collapse_among_stopped", "distinct_stopped_measurements", "distinct_survivor_measurements", "near_bar_rows")},
                   "master_gauntlet": {k: G8[k] for k in ("combinations", "certified", "certified_list")},
                   "continuous_cost": {k: C8[k] for k in C8 if k.startswith("cost_")},
                   "pcrl_encoder": P8 | {},
                   "erase_pilot": {k: E8[k] for k in ("total_strict", "comparison_json_pilot_strict", "comparison_json_base_strict", "checkpoint_selector")} |
                                  {ds: {k: E8[ds][k] for k in ("strict_pass", "cleanly_compliant", "status_field", "adj_pass_counts_per_seed", "final_and_best_both_stored")} for ds in ("adult", "hmda", "diabetes")}},
        "verdict": "partially",
        "scope": ("Reproduced exactly: 3 survivors; 1/36 (VFAE tier1 hmda/race/loan_decision); r/rho 0.7949/0.8482 (n=20) and 0.7991/0.8282 (n=27); "
                  "PCRL 56/60 one-hot and 55/60 dominant-axis strict passes, 33 multiclass pair-seeds, median amplification 1.43, max 12.65, 14 >1.5, 7 >2, "
                  "one hidden case; pilot 60/60. Qualifications: (1) the honest-reaudit denominator 21 includes 3 rows whose certificate verdict was 'breach' "
                  "(sigma 0/0.25/0.5) -- certificate-approved rows are 18, of which 15 collapse; 3 survivors are 2 distinct measurements (E2 sigma=8 and E4S1 "
                  "sigma=8 are the same numbers); (2) 56/60 is the FINAL-epoch (epoch 199) audit; the selected-checkpoint per_seed_results give 54/60 "
                  "(Adult 21, HMDA 16, Diabetes R7 17), which is the baseline the rebuttal pilot comparison uses; final_vs_best R5: Adult best 21 / final 23, "
                  "HMDA 16/16, Diabetes best 18 / final 16; (3) the pilot stores only one checkpoint (best_epoch < 199, selector 'fallback' with 0 feasible "
                  "epochs in all 9 seeds; summary STATUS 'COLLAPSED'), so final-vs-best is not verifiable for it; (4) only the adult continuous-cost shard is "
                  "tracked, hmda/diabetes shard files are absent (merged rows used)."),
        "original_implementation_compared": {"compared": False, "refs": ["durable-guarantees@956f5c8", om_ref, "local untracked erase_layer_pilot_aws"],
                                             "note": "stored-result recount only; no model rerun"},
    })
    checks.append({
        "id": "F9", "title": "Withdrawn R2-to-accuracy guarantee: 20-observation counterexample and current exposure",
        "question": "Does affine least-squares R2 = 0 coexist with 90% threshold accuracy; is the withdrawn guarantee still exposed on origin/main?",
        "input_files": fx("F09_accuracy_guarantee.py", [gsha("origin/main", "pcrl/purposes/verification.py"), gsha("origin/main", "pcrl/evaluation/certificates.py"),
                                                         gsha("fix/retire-accuracy-guarantee", "pcrl/purposes/verification.py"),
                                                         gsha("research/pcrl-guarantee-review-v1", "docs/ACCURACY_CERTIFICATE_RETIREMENT.md"),
                                                         fsha(OUT / "orig_origin_main.json"), fsha(OUT / "orig_research_pcrl-submission-finish-v1.json")]),
        "command": f"{SY} F09_accuracy_guarantee.py > outputs/F09.json ; {VENV} orig_compare.py ... ; git grep -n certified_accuracy_bound <ref> -- pcrl",
        "result": {"doc_example": {k: F9["doc_example"][k] for k in ("exact_cov_h_A", "ols_r2", "best_threshold_accuracy", "majority_rate")},
                   "own_example": {k: F9["own_example"][k] for k in ("exact_cov_h_A", "ols_r2", "best_threshold_accuracy")},
                   "origin_main_certified_accuracy_bound(0,0.5,2)": OM["F9"]["certified_accuracy_bound_r2_0_pi_0.5"],
                   "origin_main_certificate_on_example": OM["F9"]["cert_r2_on_counterexample"],
                   "submission_finish_certified_accuracy_bound": OS["F9"]["certified_accuracy_bound_r2_0_pi_0.5"],
                   "exposure": {"origin/main": "pcrl/purposes/verification.py:1-11 (module docstring 'formal theorem (Linear Compliance Guarantee)'), :204-358 live function; called by pcrl/evaluation/certificates.py:611 (generate_report prints the bound) and verification.py:529 (NonlinearComplianceCertificate)",
                                "fix/retire-accuracy-guarantee@5d4eda046": "raises NotImplementedError; NOT an ancestor of origin/main (unmerged)",
                                "research/pcrl-submission-finish-v1": "retired (verification.py:179-195 raises NotImplementedError)"}},
        "verdict": "confirmed",
        "scope": ("Exact rational arithmetic: Cov(h,A) = 0, OLS slope 0, R2 = 0, threshold accuracy 18/20 = 0.90 vs the withdrawn bound 0.50, for the repo's "
                  "example and an independent one. origin/main still exposes and calls the withdrawn guarantee (returns 0.5 and certifies R2 = 0 at the "
                  "example). Report only; nothing changed."),
        "original_implementation_compared": {"compared": True, "refs": [om_ref, os_ref], "functions": ["certified_accuracy_bound", "LinearComplianceCertificate.check"]},
    })
    checks.append({
        "id": "F10", "title": "Scope of leakage statements conditioned on a coarsened output",
        "question": "Can leakage measured through / conditional on a coarsened output differ from the same quantity with the full continuous output, in either direction?",
        "input_files": fx("F10_coarse_conditioning.py", [fsha(Path(DG) / "docs/imputation_baseline.md"), fsha(Path(DG) / "results/master_gauntlet_table.txt")]),
        "command": f"{SY} F10_coarse_conditioning.py > outputs/F10.json",
        "result": F10,
        "verdict": "confirmed",
        "scope": ("Exact discrete examples (bits): E1 I(S;C)=0 but I(S;Y)=0.278 (Bayes accuracy 0.50 vs 0.80); E2 I(S;Z|C)=0.278 but I(S;Z|Y)=0; "
                  "E3 I(S;Z|C)=0 but I(S;Z|Y)=1. So statements made from a coarse label (the durable-guarantees label-coupling predictor is fit on the "
                  "1-D discrete task label, docs/imputation_baseline.md:12-17) or conditional on it do not transfer to the continuous output "
                  "(the 'out' attack reads task logits, master_gauntlet_table.txt footer), in either direction for conditional statements; for the "
                  "unconditional output leak the coarse version can only be lower (C is a function of Y). Illustrative, not a claim about any cell."),
        "original_implementation_compared": {"compared": False, "refs": []},
    })
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / "independent_checks.json").write_text(json.dumps(checks, indent=1, default=str))
    print(f"wrote {len(checks)} checks")


if __name__ == "__main__":
    main()
