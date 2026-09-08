"""Reporting-only analysis for a fresh nonlinear-protection release pilot.

No models are fit and no configurations are selected here. Primary choices
must be read from saved validation selections; test results remain descriptive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


KINDS = ("linear", "mlp")
TASK_TARGETS = ("p1_U", "p2_V")
FORBIDDEN_TARGETS = ("p1_V", "p1_S", "p2_U", "p2_S", "combined_S")
UTILITY_FLOOR = 0.99
THRESHOLDS = {key: 0.1 if key == "combined_S" else 0.05 for key in FORBIDDEN_TARGETS}
METHODS = {
    "A_oracle": "Oracle control",
    "B_prediction_only": "Prediction-only control",
    "C_saved_task_only": "Saved prior task-only reference",
    "R_prior_0.1": "Saved prior protection 0.1 reference",
    "R_prior_1": "Saved prior protection 1 reference",
    "C_matched_task_only": "Matched new task-only control",
    "D_fixed_0.01": "Exploratory fixed weight 0.01",
    "D_fixed_0.1": "Exploratory fixed weight 0.1",
    "E_dual_0.01": "Exploratory adaptive dual, initialization 0.01",
    "E_dual_0.1": "Exploratory adaptive dual, initialization 0.1",
}
SELECTED = {"D_selected": "D", "E_selected": "E"}
SUMMARY_COLUMNS = (
    "mlp_task_p1_U", "mlp_task_p2_V", "worst_individual_linear",
    "worst_individual_mlp", "combined_S_linear", "combined_S_mlp",
)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def aggregate(values):
    values = list(values)
    observed = [float(value) for value in values if finite(value)]
    return {
        "n_seeds": len(values), "n_defined": len(observed),
        "complete": len(observed) == len(values),
        "mean": statistics.mean(observed) if observed else None,
        "sample_sd": statistics.stdev(observed) if len(observed) > 1 else None,
    }


def scored_value(score):
    if score is None:
        return None
    value = score.get("r2")
    return float(value) if score.get("defined", finite(value)) and finite(value) else None


def difference(left, right):
    return float(left - right) if finite(left) and finite(right) else None


def defined_max(values):
    values = list(values)
    return max(values) if values and all(finite(value) for value in values) else None


def defined_mean(values):
    values = list(values)
    return statistics.mean(values) if values and all(finite(value) for value in values) else None


def number(value, signed=False):
    if not finite(value):
        return "undefined"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def cell(summary, signed=False):
    text = f"{number(summary['mean'], signed)} ± {number(summary['sample_sd'])}"
    if not summary["complete"]:
        text += f" ({summary['n_defined']}/{summary['n_seeds']} defined)"
    return text


def threshold_precision_note(report):
    close_failures = []
    for seed, row in report["arms"]["E_selected"]["per_seed"].items():
        value = row["test_metrics"]["mlp_task_p1_U"]
        if finite(value) and 0 < UTILITY_FLOOR - value < 0.0001:
            close_failures.append(f"seed {seed} P1→U test R²={value!r}")
    text = "Threshold decisions use unrounded scores; all table values and SDs are displayed to six decimal places."
    if close_failures:
        text += " For E_selected, " + "; ".join(close_failures) + ". These values are below 0.99 and fail the fresh-test task floor; rounding must not turn them into passes."
    return text


def score_at(evaluation, role, kind, target):
    return scored_value(evaluation[role][kind][target])


def assessment(evaluation):
    """Recompute the prespecified criteria for reporting, without selection."""
    utility = {target: score_at(evaluation, "task", "mlp", target) for target in TASK_TARGETS}
    leakage = {target: {kind: score_at(evaluation, "attack", kind, target) for kind in KINDS} for target in FORBIDDEN_TARGETS}
    utility_defined = all(finite(value) for value in utility.values())
    leakage_defined = all(finite(value) for row in leakage.values() for value in row.values())
    utility_pass = utility_defined and all(value >= UTILITY_FLOOR for value in utility.values())
    protection_pass = utility_defined and leakage_defined and all(value <= THRESHOLDS[target] for target, row in leakage.items() for value in row.values())
    return {
        "utility": utility, "utility_defined": utility_defined,
        "utility_pass": utility_pass, "leakage": leakage,
        "protection_defined": leakage_defined, "protection_pass": protection_pass,
        "feasible": utility_pass and protection_pass,
        "defined": utility_defined and leakage_defined,
    }


def metrics(evaluation):
    result = {f"{kind}_task_{target}": score_at(evaluation, "task", kind, target) for kind in KINDS for target in TASK_TARGETS}
    for kind in KINDS:
        result[f"mean_task_{kind}"] = defined_mean(result[f"{kind}_task_{target}"] for target in TASK_TARGETS)
        result[f"worst_individual_{kind}"] = defined_max(score_at(evaluation, "attack", kind, target) for target in FORBIDDEN_TARGETS[:-1])
        result[f"combined_S_{kind}"] = score_at(evaluation, "attack", kind, "combined_S")
    return result


def key_for(record, arm):
    return record["selection"][SELECTED[arm]]["key"] if arm in SELECTED else arm


def direct_scores(method, split):
    values = method.get("direct_prediction", {}).get(split, {})
    return {target: scored_value(values.get(target, values.get(target.split("_")[-1]))) for target in TASK_TARGETS}


def training_diagnostic(method):
    training = method.get("training")
    if training is None:
        return None
    stages = training["diagnostics"]
    initial = next(row for row in stages if row["stage"] == "initial_before_adversary_warmup")
    warmed = next(row for row in stages if row["stage"] == "after_adversary_warmup")
    final = stages[-1]
    transfer = {key: scored_value(value) for key, value in method["training_adversary_validation_final_release"].items()}
    independent = {key: score_at(method["validation"], "attack", "mlp", key) for key in FORBIDDEN_TARGETS}
    weights = training["final_weights"]
    return {
        "initial_training_surrogate_by_target": initial["adversary_training_surrogate_r2"],
        "after_warmup_training_surrogate_by_target": warmed["adversary_training_surrogate_r2"],
        "final_training_surrogate_by_target": final["adversary_training_surrogate_r2"],
        "initial_training_surrogate_max": defined_max(initial["adversary_training_surrogate_r2"].values()),
        "after_warmup_training_surrogate_max": defined_max(warmed["adversary_training_surrogate_r2"].values()),
        "final_training_surrogate_max": defined_max(final["adversary_training_surrogate_r2"].values()),
        "final_training_task_r2_by_purpose": final["task_r2_by_purpose"],
        "transfer_validation_r2_by_target": transfer,
        "transfer_validation_r2_max": defined_max(transfer.values()),
        "independent_mlp_validation_r2_by_target": independent,
        "independent_mlp_validation_r2_max": defined_max(independent.values()),
        "final_weight_min": min(weights), "final_weight_max": max(weights),
        "encoder_optimizer_steps": training["optimizer_steps"],
        "adversary_optimizer_steps": training["adversary_optimizer_steps"],
        "dual_update_count": training["dual_update_count"],
        "final_diagnostic_stage": final["stage"],
        "diagnostic_split": final["diagnostic_split"],
    }


def build_report(records, inputs):
    seeds = [record["seed"] for record in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate completed seeds")
    arms = {}
    for arm in (*METHODS, *SELECTED):
        per_seed = {}
        for record in records:
            key = key_for(record, arm)
            method = record["methods"][key]
            validation = assessment(method["validation"])
            test = assessment(method["test"])
            # Detect an aggregation/schema error instead of silently relabeling
            # the runner's saved utility or protection decision.
            for split, recalculated in (("validation", validation), ("test", test)):
                saved = method[f"assessment_{split}"]
                for flag in ("utility_pass", "protection_pass", "feasible", "defined"):
                    if bool(saved[flag]) != recalculated[flag]:
                        raise ValueError(f"Saved {split} {flag} differs for seed {record['seed']} {key}")
            selection = record["selection"][SELECTED[arm]] if arm in SELECTED else None
            if selection is not None and bool(selection["utility_feasible"]) != validation["utility_pass"]:
                raise ValueError("Selected arm's utility eligibility differs from saved validation evidence")
            if selection is not None and (selection["selection_split"] != "validation"
                                          or bool(selection["diagnostic_only"]) == validation["utility_pass"]):
                raise ValueError("Saved selected arm must retain its validation-only diagnostic status")
            per_seed[str(record["seed"])] = {
                "method": key,
                "selection": selection,
                "primary_eligible": validation["utility_pass"] if selection is not None else None,
                "selection_status": ("primary_utility_eligible" if validation["utility_pass"] else "diagnostic_only_utility_floor_failed") if selection is not None else "predeclared_control_or_exploratory_configuration",
                "validation": validation, "test_assessment": test,
                "test_metrics": metrics(method["test"]),
                "forbidden_test": {kind: {target: score_at(method["test"], "attack", kind, target) for target in FORBIDDEN_TARGETS} for kind in KINDS},
                "direct_prediction": {split: direct_scores(method, split) for split in ("validation", "test")},
                "provenance": method.get("provenance"),
                "release_dimensions": method["release_dimensions"],
            }
        arms[arm] = {
            "description": METHODS.get(arm, "Previously validation-selected configuration; failed-floor fallback is diagnostic only"),
            "per_seed": per_seed,
            "summary": {key: aggregate(row["test_metrics"][key] for row in per_seed.values()) for key in next(iter(per_seed.values()))["test_metrics"]},
            "forbidden_summary": {kind: {target: aggregate(row["forbidden_test"][kind][target] for row in per_seed.values()) for target in FORBIDDEN_TARGETS} for kind in KINDS},
            "validation_counts": {flag: sum(row["validation"][flag] for row in per_seed.values()) for flag in ("utility_pass", "protection_pass", "feasible", "defined")},
            "direct_prediction_summary": {split: {target: aggregate(row["direct_prediction"][split][target] for row in per_seed.values()) for target in TASK_TARGETS} for split in ("validation", "test")},
        }
    comparisons = {}
    for left, right in (("E_dual_0.01", "D_fixed_0.01"), ("E_dual_0.1", "D_fixed_0.1"),
                        ("E_selected", "D_selected"), ("D_selected", "C_matched_task_only"),
                        ("E_selected", "C_matched_task_only")):
        per_seed = {}
        for seed in map(str, seeds):
            lhs, rhs = arms[left]["per_seed"][seed], arms[right]["per_seed"][seed]
            per_seed[seed] = {
                "left_method": lhs["method"], "right_method": rhs["method"],
                "both_validation_utility_eligible": lhs["validation"]["utility_pass"] and rhs["validation"]["utility_pass"],
                "both_validation_joint_feasible": lhs["validation"]["feasible"] and rhs["validation"]["feasible"],
                "changes": {key: difference(lhs["test_metrics"][key], rhs["test_metrics"][key]) for key in lhs["test_metrics"]},
            }
        comparisons[f"{left}_minus_{right}"] = {
            "left": left, "right": right, "per_seed": per_seed,
            "n_utility_eligible_pairs": sum(row["both_validation_utility_eligible"] for row in per_seed.values()),
            "n_joint_feasible_pairs": sum(row["both_validation_joint_feasible"] for row in per_seed.values()),
            "summary_all_declared_pairs": {key: aggregate(row["changes"][key] for row in per_seed.values()) for key in next(iter(per_seed.values()))["changes"]},
        }
    competence = {}
    for split in ("validation", "test"):
        per_seed = {str(record["seed"]): {kind: {target: score_at(record["diagnostics"]["exposed_target"][split], "attack", kind, target) for target in FORBIDDEN_TARGETS} for kind in KINDS} for record in records}
        competence[split] = {
            "per_seed": per_seed,
            "summary": {kind: {target: aggregate(row[kind][target] for row in per_seed.values()) for target in FORBIDDEN_TARGETS} for kind in KINDS},
        }
    training = {
        str(record["seed"]): {key: training_diagnostic(method) for key, method in record["methods"].items() if method.get("training") is not None}
        for record in records
    }
    return {
        "schema_version": 1, "seeds": seeds, "input_metrics": inputs,
        "utility_floor_each_purpose": UTILITY_FLOOR, "protection_thresholds": THRESHOLDS,
        "definitions": {
            "scores": "Out-of-sample predictive R2; negative scores retained. Undefined evidence fails criteria.",
            "utility": "P1->U and P2->V MLP task-probe R2 must each meet 0.99; an average cannot rescue a failed purpose.",
            "protection": "Every prohibited pair meets its threshold for both independently fitted linear and MLP attackers.",
            "selected": "Aliases read saved validation selections. A failed-utility-floor fallback is labeled diagnostic only.",
            "pairing": "Left minus right within each seed, then mean and sample SD of paired differences. All declared pairs are retained, even if ineligible.",
            "sample_sd": "ddof=1; undefined for fewer than two observed seeds. Complete/count fields expose undefined values.",
            "direct_prediction": "Native released predictions for oracle/prediction-only controls; independent task-probe utility is a separate metric.",
            "scope": "Finite-budget empirical evaluations, not a universal privacy guarantee. Oracle and prediction-only controls can differ in representation dimension and access to latent labels.",
        },
        "arms": arms, "paired_comparisons": comparisons,
        "attacker_competence_exposed_targets": competence,
        "training_adversary_diagnostics": training,
    }


def render_report(report):
    n = len(report["seeds"])
    headings = "P1→U MLP task | P2→V MLP task | Worst linear leakage | Worst MLP leakage | Combined S linear | Combined S MLP"
    lines = [
        "# Nonlinear protection release: held-out analysis", "",
        f"Completed seeds: {', '.join(map(str, report['seeds']))}. All utility/leakage values are predictive R² on the final test split. Means use paired seed observations; SD is sample SD, undefined for a single seed. Negative R² is retained.", "",
        threshold_precision_note(report), "",
        "The validation utility floor is **0.99 for each purpose separately**. Protection requires every P1→V,S and P2→U,S score ≤0.05 and combined→S ≤0.10 for **both linear and MLP probes**. The combined recipient may access U,V. These thresholds define the empirical pilot criteria; they are not universal privacy guarantees.", "",
        "## All predeclared controls, references and exploratory configurations", "",
        "Every fixed/dual configuration is shown. Test performance does not choose the primary configurations. Saved prior references retain their original training history; matched comparisons are identified separately. Oracle and prediction-only controls are reference releases with different information/dimension constraints, not equivalent learned representation methods.", "",
        f"| Method | Release dimensions P1/P2/combined | {headings} |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in METHODS:
        row = report["arms"][arm]
        dimensions = sorted({tuple(value["release_dimensions"]) for value in row["per_seed"].values()})
        dimension_text = "; ".join("/".join(map(str, value)) for value in dimensions)
        lines.append(f"| {arm} | {dimension_text} | " + " | ".join(cell(row["summary"][key]) for key in SUMMARY_COLUMNS) + " |")
    lines += ["", "## Previously validation-selected configurations", "",
              "A selection is eligible as a primary result only if both purpose utility floors passed on validation. If no candidate met that floor, the saved fallback is diagnostic only. All such selected observations remain visible below; their test outcomes are not used to replace the selection.", "",
              f"| Selected family | Validation utility eligible | Validation protection pass | Validation jointly feasible | {headings} |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in SELECTED:
        row = report["arms"][arm]
        counts = row["validation_counts"]
        lines.append(f"| {arm} | {counts['utility_pass']}/{n} | {counts['protection_pass']}/{n} | {counts['feasible']}/{n} | " + " | ".join(cell(row["summary"][key]) for key in SUMMARY_COLUMNS) + " |")
    lines += ["", "| Seed | Family | Saved configuration | Selection status |", "|---|---|---|---|"]
    for arm in SELECTED:
        for seed, row in report["arms"][arm]["per_seed"].items():
            lines.append(f"| {seed} | {arm} | {row['method']} | {row['selection_status']} |")
    lines += ["", "## Utility and protection criteria by arm", "",
              "Counts below are validation decisions. Passing leakage while failing either task floor does not satisfy the intended joint criterion.", "",
              "| Arm | Utility floor passed | Protection passed | Jointly feasible | Per-seed utility/protection/joint |",
              "|---|---:|---:|---:|---|"]
    for arm, row in report["arms"].items():
        counts = row["validation_counts"]
        detail = "; ".join(f"{seed}: {int(value['validation']['utility_pass'])}/{int(value['validation']['protection_pass'])}/{int(value['validation']['feasible'])}" for seed, value in row["per_seed"].items())
        lines.append(f"| {arm} | {counts['utility_pass']}/{n} | {counts['protection_pass']}/{n} | {counts['feasible']}/{n} | {detail} |")
    lines += ["", "## Paired comparisons", "",
              "Changes are left minus right within seed. Positive utility change favors the left arm; negative leakage change lowers that probe's predictive success. Sample SD is computed on paired changes. These are all declared paired observations, including any that failed the utility floor; eligible-pair counts prevent treating diagnostic fallbacks as successful primary results.", "",
              f"| Comparison | Utility-eligible pairs | {headings} |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["paired_comparisons"].items():
        lines.append(f"| {name} | {row['n_utility_eligible_pairs']}/{n} | " + " | ".join(cell(row["summary_all_declared_pairs"][key], signed=True) for key in SUMMARY_COLUMNS) + " |")
    lines += ["", "## Native direct prediction versus independently fitted task probes", "",
              "Native U/V predictive R² is reported separately from the MLP that is trained to recover U/V from the release. An oracle receives the true permitted signal; prediction-only uses the saved task prediction. Neither control is evidence for a richer representation's novelty.", "",
              "| Control | Direct P1→U test R² | Task-probe P1→U MLP R² | Direct P2→V test R² | Task-probe P2→V MLP R² |",
              "|---|---:|---:|---:|---:|"]
    for arm in ("A_oracle", "B_prediction_only"):
        row = report["arms"][arm]
        lines.append(f"| {arm} | {cell(row['direct_prediction_summary']['test']['p1_U'])} | {cell(row['summary']['mlp_task_p1_U'])} | {cell(row['direct_prediction_summary']['test']['p2_V'])} | {cell(row['summary']['mlp_task_p2_V'])} |")
    lines += ["", "## Attacker competence positive control", "",
              "The exposed-target diagnostic gives each view U,V,S directly and uses the same independent attacker budgets. Recovery here checks basic probe competence; good recovery does not prove those probes detect every nonlinear code. Oracle/prediction-only leakage remains reported in the main table.", "",
              "| Split | Probe | P1→V | P1→S | P2→U | P2→S | Combined→S |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for split, row in report["attacker_competence_exposed_targets"].items():
        for kind in KINDS:
            lines.append(f"| {split} | {kind} | " + " | ".join(cell(row["summary"][kind][key]) for key in FORBIDDEN_TARGETS) + " |")
    lines += ["", "## Training adversary learning and transfer", "",
              "Before warmup, after warmup, and final values are maxima of the normalized 1−MSE training surrogate over the five prohibited targets on a fixed training batch. They are not held-out predictive scores. Final is measured after the last training-eraser refresh. Transfer and independent-audit columns are predictive R² on the same validation rows and final calibrated release: the first reuses the frozen training adversaries and their training normalization, while the second uses newly fitted validation-selected MLP auditors. The final training task pair is also an in-sample diagnostic, separate from task-probe utility.", "",
              "A weak training adversary or poor transfer across the final eraser can hide leakage from the optimization objective. The independent audit remains the selection/evaluation evidence; neither a low training surrogate nor a low transfer score establishes protection.", "",
              "| Seed | Arm | Initial surrogate max | After warmup max | Final surrogate max | Transfer validation max | Independent MLP validation max | Final training task U/V | Final λ min/max |",
              "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for seed, methods in report["training_adversary_diagnostics"].items():
        for arm, row in methods.items():
            values = " | ".join(number(row[key]) for key in (
                "initial_training_surrogate_max", "after_warmup_training_surrogate_max",
                "final_training_surrogate_max", "transfer_validation_r2_max", "independent_mlp_validation_r2_max",
            ))
            tasks = "/".join(number(value) for value in row["final_training_task_r2_by_purpose"])
            weights = f"{number(row['final_weight_min'])}/{number(row['final_weight_max'])}"
            lines.append(f"| {seed} | {arm} | {values} | {tasks} | {weights} |")
    lines += ["", "## Each prohibited target", "",
              "| Arm | Probe | P1→V | P1→S | P2→U | P2→S | Combined→S |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for arm, row in report["arms"].items():
        for kind in KINDS:
            lines.append(f"| {arm} | {kind} | " + " | ".join(cell(row["forbidden_summary"][kind][key]) for key in FORBIDDEN_TARGETS) + " |")
    lines += ["", "## Per-seed, per-purpose results", "",
              f"| Seed | Arm | {headings} | Validation P1→U | Validation P2→V | Validation utility/protection/joint |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm, row in report["arms"].items():
        for seed, value in row["per_seed"].items():
            flags = "/".join(str(int(value["validation"][flag])) for flag in ("utility_pass", "protection_pass", "feasible"))
            validation_utility = " | ".join(number(value["validation"]["utility"][target]) for target in TASK_TARGETS)
            lines.append(f"| {seed} | {arm} | " + " | ".join(number(value["test_metrics"][key]) for key in SUMMARY_COLUMNS) + f" | {validation_utility} | {flags} |")
    lines += ["", "## Per-seed native direct predictions", "",
              "| Seed | Control | Validation P1→U | Validation P2→V | Test P1→U | Test P2→V |",
              "|---|---|---:|---:|---:|---:|"]
    for arm in ("A_oracle", "B_prediction_only"):
        for seed, row in report["arms"][arm]["per_seed"].items():
            lines.append(f"| {seed} | {arm} | " + " | ".join(number(row["direct_prediction"][split][target]) for split in ("validation", "test") for target in TASK_TARGETS) + " |")
    lines += ["", "All values, paired per-seed differences, saved selection records, source provenance and input metric hashes are available in release_summary.json. This report performs no fitting or selection and preserves the original result files.", ""]
    return "\n".join(lines)


def render_table(report):
    headings = "P1→U MLP task | P2→V MLP task | Worst individual linear | Worst individual MLP | Combined S linear | Combined S MLP"
    n = len(report["seeds"])
    lines = [
        "# Nonlinear release pilot: complete results", "",
        "Final-test predictive R². Utility is reported for each purpose separately. Worst individual leakage takes the maximum over P1→V,S and P2→U,S within each seed and probe family. Only S is prohibited for the combined recipient.", "",
        threshold_precision_note(report), "",
        "D_fixed_* and E_dual_* are the exploratory configurations. D_selected/E_selected reuse the saved validation choice within each family; failed-floor fallbacks are diagnostic only. Oracle and prediction-only releases are one-dimensional controls; full representations and prior protection references are separate methods.", "",
        "## Mean ± sample SD across completed seeds", "",
        "SD is undefined for one completed seed. Validation flags require utility R²≥0.99 for each purpose and protection R²≤0.05 for each individual forbidden pair, ≤0.10 for combined S, for both probe families.", "",
        f"| Arm | {headings} | Validation utility/protection/joint passed |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm, row in report["arms"].items():
        counts = " · ".join(f"{row['validation_counts'][flag]}/{n}" for flag in ("utility_pass", "protection_pass", "feasible"))
        lines.append(f"| {arm} | " + " | ".join(cell(row["summary"][key]) for key in SUMMARY_COLUMNS) + f" | {counts} |")
    lines += ["", "## Every completed seed and purpose", "",
              f"| Seed | Arm | {headings} | Validation P1→U | Validation P2→V | Validation utility/protection/joint |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for seed in map(str, report["seeds"]):
        for arm, summary in report["arms"].items():
            row = summary["per_seed"][seed]
            validation = row["validation"]
            flags = "/".join(str(int(validation[flag])) for flag in ("utility_pass", "protection_pass", "feasible"))
            values = " | ".join(number(row["test_metrics"][key]) for key in SUMMARY_COLUMNS)
            val_tasks = " | ".join(number(validation["utility"][target]) for target in TASK_TARGETS)
            lines.append(f"| {seed} | {arm} | {values} | {val_tasks} | {flags} |")
    lines += ["", "## Saved selections", "",
              "| Seed | Family | Saved configuration | Status |", "|---|---|---|---|"]
    for seed in map(str, report["seeds"]):
        for arm in SELECTED:
            row = report["arms"][arm]["per_seed"][seed]
            lines.append(f"| {seed} | {arm} | {row['method']} | {row['selection_status']} |")
    lines += ["", "ANALYSIS.md provides each forbidden target, native direct prediction metrics, exposed-target attacker competence, release dimensions and paired fixed-versus-dual changes. release_summary.json preserves all paired per-seed values and source metric hashes. No failed empirical attack is a universal privacy guarantee.", ""]
    return "\n".join(lines)


def summarize(out):
    out = Path(out)
    paths = sorted(out.glob("seed_*/metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No completed seed metrics in {out}; no outputs written")
    records, inputs = [], []
    for path in paths:
        contents = path.read_bytes()
        records.append(json.loads(contents))
        inputs.append({"path": str(path.relative_to(out)), "sha256": hashlib.sha256(contents).hexdigest()})
    records.sort(key=lambda record: record["seed"])
    report = build_report(records, inputs)
    (out / "release_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "ANALYSIS.md").write_text(render_report(report))
    (out / "TABLE.md").write_text(render_table(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.out)
    print(f"Wrote TABLE.md, ANALYSIS.md and release_summary.json for seeds {report['seeds']} in {args.out}")


if __name__ == "__main__":
    main()
