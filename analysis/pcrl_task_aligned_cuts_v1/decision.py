"""Recompute aggregate inner-pilot point screens from verified public tables.

This script does no model fitting, person-level loading, or outcome selection.
It evaluates the registered point rule on the already completed 2018
development pilot, and is deliberately separate from outer inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


TASK = "utility:A/same_residence"
PROTECTED = ("attack:A/SEX", "attack:A/RAC1P", "attack:AB/SEX", "attack:AB/RAC1P")
WEIGHTINGS = ("U", "PWGTP")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows: list[dict], name_key: str) -> dict:
    answer = {}
    for row in rows:
        key = (row[name_key], row["role"], row["weighting"])
        if key in answer:
            raise ValueError(f"duplicate endpoint: {key}")
        answer[key] = row
    return answer


def score(candidate: dict, comparator: dict) -> dict:
    task = {}
    protected = {}
    if set(candidate) != set(comparator):
        raise ValueError("candidate/comparator endpoint families differ")
    for role, weighting in candidate:
        left, right = candidate[(role, weighting)], comparator[(role, weighting)]
        if (left["n_people"], left["n_households"]) != (right["n_people"], right["n_households"]):
            raise ValueError(f"unpaired aggregate scope: {role}/{weighting}")
        if abs(float(left["H_loss"]) - float(right["H_loss"])) > 1e-10:
            raise ValueError(f"different H reference: {role}/{weighting}")
        if role == TASK:
            task[weighting] = float(left["loss"]) - float(right["loss"])
        elif role in PROTECTED:
            protected[f"{role}/{weighting}"] = (float(left["H_minus_release"])
                                                  - float(right["H_minus_release"]))
        else:
            raise ValueError(f"unexpected role {role}")
    if len(task) != 2 or len(protected) != 8:
        raise ValueError("incomplete registered endpoints")
    ab_race = [protected[f"attack:AB/RAC1P/{w}"] for w in WEIGHTINGS]
    other = [v for k, v in protected.items() if not k.startswith("attack:AB/RAC1P/")]
    h_gain = {w: float(candidate[(TASK, w)]["H_minus_release"]) for w in WEIGHTINGS}
    checks = {
        "task_both_le_plus_0.001": all(task[w] <= 0.001 for w in WEIGHTINGS),
        "all_other_protected_le_plus_0.001": all(v <= 0.001 for v in other),
        "AB_RAC1P_both_le_minus_0.002": all(v <= -0.002 for v in ab_race),
        "task_gain_over_H_both_ge_0.01": all(h_gain[w] >= 0.01 for w in WEIGHTINGS),
    }
    return {"task_loss_candidate_minus_comparator": task,
            "sensitive_recovery_candidate_minus_comparator": protected,
            "task_gain_over_H": h_gain,
            "point_checks": checks,
            "all_point_checks_pass": all(checks.values())}


def main(core_path: Path, controls_path: Path, gradient_path: Path, output: Path) -> dict:
    core = list(csv.DictReader(core_path.open(newline="")))
    controls = list(csv.DictReader(controls_path.open(newline="")))
    selected = json.loads(gradient_path.read_text())
    gradient_unit = selected["selected_unit_id"]
    if not gradient_unit or selected.get("outer_pool_opened") is not False:
        raise ValueError("no eligible, inner-only gradient selection")
    core = [r for r in core if r["anchor"] == "0" and r["audit_status"] == "complete"]
    controls = [r for r in controls if r["anchor"] == "0" and r["audit_status"] in ("complete", "alias_complete")]
    by_core = indexed(core, "configuration")
    by_control = indexed(controls, "control_label")
    core_names = sorted({r["configuration"] for r in core})
    if len(core_names) != 12:
        raise ValueError("anchor-0 crossed pilot incomplete")

    def core_group(name: str) -> dict:
        return {(role, w): {**by_core[(name, role, w)],
                             "loss": by_core[(name, role, w)]["release_loss"]}
                for role in (TASK, *PROTECTED) for w in WEIGHTINGS}

    def control_group(name: str) -> dict:
        return {(role, w): {**by_control[(name, role, w)],
                             "loss": by_control[(name, role, w)]["candidate_loss"],
                             "H_minus_release": by_control[(name, role, w)]["H_minus_candidate"]}
                for role in (TASK, *PROTECTED) for w in WEIGHTINGS}

    comparisons = {}
    for name in core_names:
        comparisons[name] = {base: score(core_group(name), control_group(base))
                             for base in ("D17", "D_U1")}
    center = "a0_u1p1_z000"
    chosen_rows = [r for r in selected["full_curve"] if r["unit_id"] == gradient_unit]
    if (len(chosen_rows) != 1 or not chosen_rows[0]["eligible"]
            or chosen_rows[0]["exact_Q_alias"]
            or chosen_rows[0]["source_array_sha256"] != selected["selected_source_array_sha256"]):
        raise ValueError("selected gradient is not a unique verified feasible map")
    gradient_label = chosen_rows[0]["label"]
    for base in ("MILP", gradient_label):
        for role in (TASK, *PROTECTED):
            for w in WEIGHTINGS:
                if by_control[(base, role, w)]["fixed_bank_feasible"] != "true":
                    raise ValueError(f"matched control is not fixed-bank feasible: {base}")
    center_controls = {base: score(core_group(center), control_group(base))
                       for base in ("MILP", gradient_label)}
    result = {
        "schema": "pcrl-inner-pilot-point-screens-v1",
        "scope": "2018 used-data inner development pilot; no final inference",
        "registered_rule": {"task_max_nats": 0.001,
                            "other_sensitive_recovery_max_nats": 0.001,
                            "AB_RAC1P_recovery_max_nats": -0.002,
                            "task_gain_over_H_min_nats": 0.01},
        "input_sha256": {"CORE_FACTORIAL.csv": digest(core_path),
                         "MATCHED_CONTROLS.csv": digest(controls_path),
                         "GRADIENT_SELECTION.json": digest(gradient_path)},
        "core_configurations": len(core_names),
        "core_vs_D17_and_D_U1": comparisons,
        "both_unprotected_comparator_screens_pass": [name for name, pair in comparisons.items()
                                                        if all(v["all_point_checks_pass"] for v in pair.values())],
        "center_vs_matched_controls": center_controls,
        "selected_gradient_label": gradient_label,
        "selected_gradient_unit": gradient_unit,
        "outer_assessment_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, output)
    return {"core_configurations": len(core_names),
            "both_unprotected_comparator_screens_pass": result["both_unprotected_comparator_screens_pass"],
            "selected_gradient_label": gradient_label, "output_sha256": digest(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    base = args.root / "results/pcrl_task_aligned_cuts_v1"
    print(json.dumps(main(base / "CORE_FACTORIAL.csv", base / "MATCHED_CONTROLS.csv",
                          base / "agents/controls_data/GRADIENT_SELECTION.json",
                          base / "POINT_SCREENS.json"), sort_keys=True))
