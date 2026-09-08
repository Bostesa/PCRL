"""Aggregate an existing fresh nonlinear-conflict pilot; never fit or select models.

Reads only seed_*/metrics.json in --out. Rewrites this script's ANALYSIS.md and
paired_comparisons.json as completed seeds arrive. The selected positive arm
is taken verbatim from the previously saved validation-only selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ARMS = {
    "A": "A_frozen_no_erasure",
    "B": "B_frozen_leace",
    "C": "C_task_only",
    "D0.1": "D_protection_0.1",
    "D1": "D_protection_1",
    "positive_validation_selected": None,
}
KINDS = ("linear", "mlp")
TASKS = ("p1_U", "p2_V")
FORBIDDEN = ("p1_V", "p1_S", "p2_U", "p2_S", "combined_S")
COMPARISONS = (
    ("B_minus_A", "B", "A", "Erasure utility change"),
    ("C_minus_B", "C", "B", "Task-only adaptation recovery"),
    ("D0.1_minus_C", "D0.1", "C", "Protection 0.1 versus task-only"),
    ("D1_minus_C", "D1", "C", "Protection 1 versus task-only"),
    ("positive_validation_selected_minus_C", "positive_validation_selected", "C", "Validation-selected positive protection versus task-only"),
)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def aggregate(values):
    values = list(values)
    observed = [float(value) for value in values if finite(value)]
    return {
        "n_seeds": len(values),
        "n_defined": len(observed),
        "complete": len(observed) == len(values),
        "mean": statistics.mean(observed) if observed else None,
        "sample_sd": statistics.stdev(observed) if len(observed) > 1 else None,
    }


def delta(left, right):
    return float(left - right) if finite(left) and finite(right) else None


def method_key(seed, arm):
    if arm == "positive_validation_selected":
        selection = seed["positive_selection"]
        if selection["selection_split"] != "validation":
            raise ValueError("Positive-arm selection must be explicitly validation-only")
        return selection["key"]
    return ARMS[arm]


def get_score(seed, arm, split, role, kind, target):
    score = seed["methods"][method_key(seed, arm)][split][role][kind][target]
    value = score["r2"]
    return float(value) if score["defined"] and finite(value) else None


def utility(seed, arm, kind):
    values = [get_score(seed, arm, "test", "task", kind, target) for target in TASKS]
    return statistics.mean(values) if all(finite(value) for value in values) else None


def worst_individual(seed, arm, kind):
    values = [get_score(seed, arm, "test", "attack", kind, target) for target in FORBIDDEN[:-1]]
    return max(values) if all(finite(value) for value in values) else None


def combined(seed, arm, kind):
    return get_score(seed, arm, "test", "attack", kind, "combined_S")


def validation_feasibility(seed, arm, thresholds):
    probes = {}
    for kind in KINDS:
        scores = {key: get_score(seed, arm, "validation", "attack", kind, key) for key in FORBIDDEN}
        defined = all(finite(value) for value in scores.values())
        probes[kind] = {
            "scores": scores,
            "defined": defined,
            "feasible": defined and all(scores[key] <= thresholds[key] for key in FORBIDDEN),
            "excess": {key: max(0.0, scores[key] - thresholds[key]) if finite(scores[key]) else None for key in FORBIDDEN},
        }
    defined = all(row["defined"] for row in probes.values())
    worst = {
        key: max(probes[kind]["scores"][key] for kind in KINDS) if defined else None
        for key in FORBIDDEN
    }
    return {
        "method": method_key(seed, arm),
        "defined": defined,
        "feasible": all(row["feasible"] for row in probes.values()),
        "probes": probes,
        "worst_probe_by_target": worst,
        "total_normalized_excess": sum(max(0.0, worst[key] - thresholds[key]) / thresholds[key] for key in FORBIDDEN) if defined else None,
    }


def build_analysis(records, input_files):
    seeds = [record["seed"] for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seed identifiers")
    thresholds = records[0]["configuration"]["thresholds"]
    if set(thresholds) != set(FORBIDDEN) or any(not finite(value) or value <= 0 for value in thresholds.values()):
        raise ValueError("Expected all five positive recipient/target thresholds")
    if any(record["configuration"]["thresholds"] != thresholds for record in records):
        raise ValueError("Seed thresholds differ")
    arms = {}
    for arm in ARMS:
        arms[arm] = {
            "method_by_seed": {str(record["seed"]): method_key(record, arm) for record in records},
            "test_utility": {kind: aggregate(utility(record, arm, kind) for record in records) for kind in KINDS},
            "test_forbidden_targets": {
                kind: {target: aggregate(get_score(record, arm, "test", "attack", kind, target) for record in records) for target in FORBIDDEN}
                for kind in KINDS
            },
            "test_worst_individual": {kind: aggregate(worst_individual(record, arm, kind) for record in records) for kind in KINDS},
            "validation": {str(record["seed"]): validation_feasibility(record, arm, thresholds) for record in records},
        }
    paired = {}
    for name, left, right, description in COMPARISONS:
        per_seed = {}
        for record in records:
            per_seed[str(record["seed"])] = {
                kind: {
                    "mean_task_r2_change": delta(utility(record, left, kind), utility(record, right, kind)),
                    "worst_individual_r2_change": delta(worst_individual(record, left, kind), worst_individual(record, right, kind)),
                    "combined_S_r2_change": delta(combined(record, left, kind), combined(record, right, kind)),
                } for kind in KINDS
            }
        paired[name] = {
            "left": left, "right": right, "description": description,
            "per_seed": per_seed,
            "summary": {
                kind: {metric: aggregate(row[kind][metric] for row in per_seed.values()) for metric in (
                    "mean_task_r2_change", "worst_individual_r2_change", "combined_S_r2_change",
                )} for kind in KINDS
            },
        }
    headroom = {}
    for kind in KINDS:
        per_seed = {str(record["seed"]): {
            "A_mean_task_r2": utility(record, "A", kind),
            "B_mean_task_r2": utility(record, "B", kind),
            "B_headroom_1_minus_B": delta(1.0, utility(record, "B", kind)),
            "LEACE_utility_loss_A_minus_B": delta(utility(record, "A", kind), utility(record, "B", kind)),
        } for record in records}
        headroom[kind] = {
            "per_seed": per_seed,
            "summary": {key: aggregate(row[key] for row in per_seed.values()) for key in next(iter(per_seed.values()))},
        }
    return {
        "schema_version": 1,
        "seeds": seeds,
        "input_metrics": input_files,
        "thresholds": thresholds,
        "definitions": {
            "utility": "Mean of independently fitted task-probe predictive R2 for P1->U and P2->V within each seed.",
            "paired_change": "Left arm minus right arm within each seed, then mean and sample SD across paired differences.",
            "worst_individual": "Maximum predictive R2 among P1->V,S and P2->U,S within each seed and probe family.",
            "combined": "Combined recipient may access U,V; only S is prohibited.",
            "sample_sd": "Sample standard deviation (ddof=1), undefined with fewer than two defined seeds.",
            "undefined": "Undefined inputs remain null and counts expose coverage; undefined validation evidence cannot pass.",
            "scope": "Empirical finite-budget attacks and utility only; no universal privacy guarantee.",
            "selection": "Positive selected arm is read from each seed's saved validation selection; this script performs no selection.",
        },
        "arms": arms,
        "paired_comparisons": paired,
        "headroom": headroom,
    }


def number(value, signed=False):
    return f"{value:+.5f}" if finite(value) and signed else f"{value:.5f}" if finite(value) else "undefined"


def summary_cell(row, signed=False):
    text = f"{number(row['mean'], signed)} ± {number(row['sample_sd'])}"
    return text if row["complete"] else f"{text} ({row['n_defined']}/{row['n_seeds']} defined)"


def render_markdown(report):
    text = [
        "# Nonlinear conflict pilot: paired analysis", "",
        f"Completed seeds: {', '.join(map(str, report['seeds']))}. Predictive R² is out of sample; negative scores are retained. Cells show mean ± sample SD across seeds. SD is undefined with only one seed.", "",
        "A = frozen pretrained encoder; B = A plus per-purpose LEACE; C = task-only upstream adaptation plus recalibrated LEACE; D0.1/D1 = matched adaptation with protection strength 0.1/1 plus recalibrated LEACE. The positive selected arm reuses the previously saved validation choice between D0.1 and D1.", "",
        "## Utility and remaining headroom", "",
        "Task utility averages P1→U and P2→V within each seed. Headroom is the distance from B to predictive R²=1; LEACE loss is the paired A−B difference.", "",
        "| Probe | A utility | B utility | B headroom (1−B) | LEACE utility loss (A−B) |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in KINDS:
        row = report["headroom"][kind]["summary"]
        text.append(f"| {kind} | " + " | ".join(summary_cell(row[key]) for key in (
            "A_mean_task_r2", "B_mean_task_r2", "B_headroom_1_minus_B", "LEACE_utility_loss_A_minus_B",
        )) + " |")
    text += ["", "## Every forbidden target", "",
             "The combined recipient intentionally receives both task signals; only combined→S is prohibited. These are results for the fitted probe families and budgets.", "",
             "| Arm | Probe | P1→V | P1→S | P2→U | P2→S | Combined→S |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for arm, row in report["arms"].items():
        for kind in KINDS:
            text.append(f"| {arm} | {kind} | " + " | ".join(summary_cell(row["test_forbidden_targets"][kind][key]) for key in FORBIDDEN) + " |")
    text += ["", "## Paired changes", "",
             "All differences are left minus right within the same seed. Positive utility change means improvement. Negative leakage change means lower predictive success for that probe. B−A records the signed erasure utility change; A−B above expresses its loss. SD below is computed from paired differences, not from separate arm SDs.", "",
             "| Comparison | Probe | Mean task R² change | Worst individual leakage change | Combined S leakage change |",
             "|---|---|---:|---:|---:|"]
    for name, row in report["paired_comparisons"].items():
        for kind in KINDS:
            text.append(f"| {name} | {kind} | " + " | ".join(summary_cell(row["summary"][kind][key], signed=True) for key in (
                "mean_task_r2_change", "worst_individual_r2_change", "combined_S_r2_change",
            )) + " |")
    text += ["", "## Validation feasibility", "",
             "An arm is feasible only when every prohibited target meets its saved threshold for both linear and MLP attackers. This is an empirical validation criterion. Undefined observations fail. Thresholds: " + ", ".join(f"{key}≤{value:g}" for key, value in report["thresholds"].items()) + ".", "",
             "| Arm | Linear feasible seeds | MLP feasible seeds | Both feasible seeds | Both, by seed |",
             "|---|---:|---:|---:|---|"]
    for arm, row in report["arms"].items():
        validation = row["validation"]
        denominator = len(validation)
        counts = [sum(seed["probes"][kind]["feasible"] for seed in validation.values()) for kind in KINDS]
        both = sum(seed["feasible"] for seed in validation.values())
        flags = ", ".join(f"{seed}: {'pass' if result['feasible'] else 'fail'}" for seed, result in validation.items())
        text.append(f"| {arm} | {counts[0]}/{denominator} | {counts[1]}/{denominator} | {both}/{denominator} | {flags} |")
    text += ["", "Saved positive-strength selections:", ""]
    for seed, method in report["arms"]["positive_validation_selected"]["method_by_seed"].items():
        text.append(f"- Seed {seed}: {method}.")
    text += ["", "## Per-seed paired changes", "",
             "| Seed | Comparison | Probe | Mean task R² change | Worst individual leakage change | Combined S leakage change |",
             "|---|---|---|---:|---:|---:|"]
    for name, row in report["paired_comparisons"].items():
        for seed, kinds in row["per_seed"].items():
            for kind, metrics in kinds.items():
                text.append(f"| {seed} | {name} | {kind} | " + " | ".join(number(metrics[key], signed=True) for key in (
                    "mean_task_r2_change", "worst_individual_r2_change", "combined_S_r2_change",
                )) + " |")
    text += ["", "This report performs no fitting or model selection. All selections come from saved validation records. Small or negative measured leakage is not a universal privacy guarantee. Per-target coverage, validation violations, input hashes, and all paired values are in paired_comparisons.json. TABLE.md contains the runner's raw per-seed arm results.", ""]
    return "\n".join(text)


def summarize(out):
    out = Path(out)
    paths = sorted(out.glob("seed_*/metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No completed seed metrics in {out}; no reports written")
    inputs, records = [], []
    for path in paths:
        contents = path.read_bytes()
        records.append(json.loads(contents))
        inputs.append({"path": str(path.relative_to(out)), "sha256": hashlib.sha256(contents).hexdigest()})
    records.sort(key=lambda record: record["seed"])
    report = build_analysis(records, inputs)
    (out / "paired_comparisons.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "ANALYSIS.md").write_text(render_markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.out)
    print(f"Wrote ANALYSIS.md and paired_comparisons.json for seeds {result['seeds']} in {args.out}")


if __name__ == "__main__":
    main()
