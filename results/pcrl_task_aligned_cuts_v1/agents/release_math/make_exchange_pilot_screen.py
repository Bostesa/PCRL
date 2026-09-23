"""Build an aggregate-only, hash-pinned 2018 exchange pilot comparison."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path("results/pcrl_task_aligned_cuts_v1")
PRIVATE = ROOT / "private"
WEIGHTS = ("U", "PWGTP")
TASK = "utility:A/same_residence"
PROTECTED = ("attack:A/SEX", "attack:A/RAC1P",
             "attack:AB/SEX", "attack:AB/RAC1P")
ROLES = (TASK, *PROTECTED)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_panel(relative, receipt_name):
    directory = PRIVATE / relative
    receipt_path = directory / receipt_name
    receipt = json.loads(receipt_path.read_text())
    panel_path = directory / "INNER_PANEL.json"
    if receipt.get("artifacts", {}).get("INNER_PANEL.json") != sha(panel_path):
        raise ValueError(f"panel differs from immutable receipt: {relative}")
    panel = json.loads(panel_path.read_text())
    if (panel.get("slate") != "standard" or panel.get("outer_pool_opened") is not False or
            panel.get("split_assignment_sha256") !=
            "9624c02a5dfc1797c602c0ab49b55f0853d2ca12a2a631ebbf2bf307253c27bf" or
            set(panel.get("roles", {})) != set(ROLES)):
        raise ValueError(f"pilot panel role/split/scope differs: {relative}")
    return {
        "panel_relative_path": str(panel_path.relative_to(ROOT)),
        "panel_sha256": sha(panel_path),
        "receipt_relative_path": str(receipt_path.relative_to(ROOT)),
        "receipt_sha256": sha(receipt_path),
        "channel_array_sha256": panel["channel_sha256"],
        "unit_id": receipt["unit_id"],
        "task_loss_nats": {w: panel["roles"][TASK]["candidate"][w] for w in WEIGHTS},
        "task_improvement_over_H_nats": {
            w: panel["roles"][TASK]["H_minus_candidate"][w] for w in WEIGHTS},
        "protected_recovery_over_H_nats": {
            role: {w: panel["roles"][role]["H_minus_candidate"][w] for w in WEIGHTS}
            for role in PROTECTED},
        "n_scored_households": {role: panel["roles"][role]["n_scored_households"]
                                 for role in ROLES},
        "_H_scores": {role: {w: panel["roles"][role]["H"][w] for w in WEIGHTS}
                      for role in ROLES},
    }


def main():
    selection_path = ROOT / "agents/controls_data/GRADIENT_SELECTION.json"
    gradient_selection = json.loads(selection_path.read_text())
    gradient_id = gradient_selection["selected_unit_id"]
    definitions = {
        "exchange_stochastic": ("exchange_r01_audit", "COMPLETE.json"),
        "exchange_deterministic_MILP": ("exchange_r01_deterministic_audit", "COMPLETE.json"),
        "round_zero_center": ("run/a0_u1p1_z000_audit", "COMPLETE.json"),
        "D17_unprotected": ("sidecar_audits/a0_simple_D17", "SIDECAR_COMPLETE.json"),
        "D_U1_unprotected": ("sidecar_audits/a0_simple_D_U1", "SIDECAR_COMPLETE.json"),
        "initial_bank_MILP": ("sidecar_audits/a0_matched_MILP", "SIDECAR_COMPLETE.json"),
        "validation_selected_gradient": (f"sidecar_audits/{gradient_id}", "SIDECAR_COMPLETE.json"),
    }
    panels = {name: read_panel(*where) for name, where in definitions.items()}
    reference = panels["exchange_stochastic"]
    h_scores = reference.pop("_H_scores")
    households = reference["n_scored_households"]
    for name, panel in panels.items():
        if name == "exchange_stochastic":
            continue
        if panel.pop("_H_scores") != h_scores or panel["n_scored_households"] != households:
            raise ValueError(f"H baseline or scoring households differ: {name}")
    contrasts = {}
    for name, panel in panels.items():
        if name == "exchange_stochastic":
            continue
        task = {w: reference["task_loss_nats"][w] - panel["task_loss_nats"][w]
                for w in WEIGHTS}
        protected = {
            role: {w: reference["protected_recovery_over_H_nats"][role][w]
                   - panel["protected_recovery_over_H_nats"][role][w]
                   for w in WEIGHTS}
            for role in PROTECTED}
        useful_beyond_h = all(
            reference["task_improvement_over_H_nats"][w] >= .01 for w in WEIGHTS)
        point_rule = useful_beyond_h and all(task[w] <= .001 for w in WEIGHTS) and all(
            protected["attack:AB/RAC1P"][w] <= -.002 for w in WEIGHTS) and all(
            protected[role][w] <= .001 for role in PROTECTED
            if role != "attack:AB/RAC1P" for w in WEIGHTS)
        contrasts[name] = {
            "task_loss_exchange_minus_comparator_nats": task,
            "protected_recovery_exchange_minus_comparator_nats": protected,
            "registered_primary_point_rule_pass": point_rule,
            "candidate_useful_beyond_H_point_rule_pass": useful_beyond_h,
            "status": "descriptive_point_comparison_only",
        }
    result = {
        "schema": "pcrl-exchange-inner-pilot-screen-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "2018 used-data inner_pilot, anchor 0, standard validation-selected slate; no outer outcomes",
        "not_confirmation": True,
        "outer_pool_opened": False,
        "primary_point_rule": {
            "task_loss_candidate_minus_comparator_max_nats": .001,
            "AB_RAC1P_recovery_candidate_minus_comparator_max_nats": -.002,
            "other_protected_recovery_candidate_minus_comparator_max_nats": .001,
            "task_improvement_over_H_min_nats": .01,
            "weightings": list(WEIGHTS),
            "not_an_inferential_test": True,
        },
        "panels": panels,
        "exchange_minus_comparator": contrasts,
        "gradient_selection": {
            "selected_unit_id": gradient_id,
            "selection_file_relative_path": str(selection_path.relative_to(ROOT)),
            "selection_file_sha256": sha(selection_path),
            "selection_stage_limitation": gradient_selection.get("limitation"),
        },
        "optimization_context": {
            "exchange_bank_sha256": "1b933956edcedf4d14dc7a73ec3fa6f4e7b7be7da1b24db98e97039d3ac4a18d",
            "exchange_stochastic_and_deterministic_same_324_cut_bank": True,
            "round_zero_and_initial_MILP_bank_cut_count": 312,
            "D17_and_D_U1_unprotected": True,
            "selected_gradient_bank_cut_count": 312,
        },
        "limits": "Point comparisons only; fitted standard slates and reused 2018 pools, no new outer assessment or household interval. H-selected sensitive routes may produce equal measured recovery despite different channels. A fixed-bank integrality gap does not imply a fresh-probe advantage.",
        "reporting_correction": "The first uncommitted aggregate screen omitted the registered task-improvement-over-H capability clause from its point-rule Boolean. This corrected screen includes it; the candidate passes that clause in both weightings, and no comparator verdict changes. Original bytes are retained in agents/release_math/EXCHANGE_PILOT_SCREEN_PRE_CAPABILITY_FIX.json.",
    }
    out = ROOT / "EXCHANGE_PILOT_SCREEN.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"file": str(out), "sha256": sha(out),
                      "comparators": len(contrasts),
                      "point_rule_passes": sum(v["registered_primary_point_rule_pass"]
                                               for v in contrasts.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
