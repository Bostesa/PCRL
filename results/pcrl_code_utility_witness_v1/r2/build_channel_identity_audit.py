"""Reconcile prior A identity evidence from public aggregate receipts only.

This program does not open role archives, person arrays, private checkpoints,
or the old assessment. It recomputes every reported identity cost and margin
from the saved aggregate task metrics, and preserves the original fit status.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "pcrl_latent_and_sampled_methods_v1"
PREFIT = OLD / "r04/A_PREFIT_INDEPENDENT_REPLAY.json"
FITS = OLD / "r03/A_PRODUCER_FIT_CLOSEOUT.json"
BANK = OLD / "r01/BANK_IDENTITY_AUDIT.json"
INNER_POINTS = OLD / "r02/A_FIXED_POINT_ROWS_R2.csv"
BANK_SOURCE = OLD / "sampled_channel_v1/impl/bank.py"
SOURCE = OLD / "sampled_channel_v1/impl/channel.py"
DET_SOURCE = OLD / "sampled_channel_v1/controls/deterministic_policy.py"
FACTOR_SOURCE = OLD / "sampled_channel_v1/controls/factor_controls.py"
SEQ_SOURCE = OLD / "sampled_channel_v1/controls/seq_train.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def status(cost: float, cap: float) -> str:
    return "PASS" if cost <= cap else "FAIL"


def build() -> tuple[list[dict], list[dict], dict]:
    old = load(PREFIT)
    fits = load(FITS)
    bank = load(BANK)
    assert old["status"] == "PREFIT_SOURCE_DECODER_IDENTITY_PASS"
    assert fits["fit_plan_count"] == 16 and len(fits["rows"]) == 16
    assert set(old["seeds"]) == {"0", "1"}
    assert "ordered = (identity, *neighbors[:2], medoid[c])" in BANK_SOURCE.read_text()
    assert "q = np.exp(logits - np.max(logits))" in SOURCE.read_text()
    assert "winner = int(np.argmax(self.gate.logits(features)[slots]))" in DET_SOURCE.read_text()
    assert "mass = np.exp(logits - np.max(logits))" in FACTOR_SOURCE.read_text()
    rows = []
    summary = {}
    for seed in ("0", "1"):
        one = old["seeds"][seed]
        summary[seed] = {}
        for recipient, task in (("r1", "income"), ("r2", "occupation")):
            bank_count = bank["seeds"][seed][recipient]["canonical_output_vector_decision_pairs"]
            assert bank_count == one["banks"][recipient]["unique_public_actions"]
            assert one["banks"][recipient]["identity_decision_class_preserved"]
            summary[seed][task] = {
                "recipient": recipient,
                "public_bank_states": bank_count,
                "action_identity_first": True,
                "exact_identity_diagnostic_exists": True,
            "original_stochastic_parameterization_has_explicit_exact_identity_checkpoint": False,
                "original_deterministic_parameterization_exact_identity": True,
            }
            for role in ("OSF_DEFENSE_FIT", "INNER_SELECTION"):
                item = one["roles"][role]
                assert item["identity_token_exact"] and item["source_cell_exact"]
                actual = item["identity_task"][task]
                for reference in ("U0", "Ucal"):
                    baseline = item["references"][task][reference]
                    reported = item["identity_point_gate"]["rows"][task][reference]
                    for metric, cap in (("accuracy", .010), ("nll", .010), ("brier", .005)):
                        if metric == "accuracy":
                            cost = baseline[metric] - actual[metric]
                            old_key = "accuracy_drop"
                        else:
                            cost = actual[metric] - baseline[metric]
                            old_key = metric + "_excess"
                        margin = cap - cost
                        assert abs(cost - reported["cost"][old_key]) < 1e-12
                        assert abs(margin - reported["margin"][old_key]) < 1e-12
                        rows.append({
                            "seed": int(seed), "recipient": recipient, "task": task,
                            "role": role, "reference": reference, "metric": metric,
                            "identity_value": actual[metric],
                            "reference_value": baseline[metric],
                            "cost_direction": "reference_minus_identity" if metric == "accuracy" else "identity_minus_reference",
                            "cost": cost, "cap": cap, "margin_cap_minus_cost": margin,
                            "old_fit_constraint_applies": role == "OSF_DEFENSE_FIT" and metric in ("nll", "brier"),
                            "point_status": status(cost, cap),
                        })
    assert len(rows) == 48
    for seed in ("0", "1"):
        for task in ("income", "occupation"):
            subset = [row for row in rows if row["seed"] == int(seed) and row["task"] == task]
            summary[seed][task]["I2_fitting_costs_pass"] = all(
                row["point_status"] == "PASS" for row in subset
                if row["old_fit_constraint_applies"])
            summary[seed][task]["I3_common_inner_point_pass"] = all(
                row["point_status"] == "PASS" for row in subset
                if row["role"] == "INNER_SELECTION")
    # The later fixed endpoint reconstruction is a separate aggregate source.
    # Its identity and reference points must equal the earlier source replay.
    checked_inner_points = 0
    with INNER_POINTS.open(newline="") as stream:
        for point in csv.DictReader(stream):
            if (point["arm"] not in {"A_IDENTITY", "U0", "Ucal"} or
                    point["draw"] != "0" or
                    point["scope"] not in {"income", "occupation"} or
                    point["metric"] not in {"accuracy", "nll", "brier"}):
                continue
            source = old["seeds"][point["seed"]]["roles"]["INNER_SELECTION"]
            metrics = (source["identity_task"][point["scope"]] if point["arm"] == "A_IDENTITY"
                       else source["references"][point["scope"]][point["arm"]])
            assert abs(float(point["point"]) - metrics[point["metric"]]) < 1e-12
            checked_inner_points += 1
    assert checked_inner_points == 36
    fit_rows = []
    for row in sorted(fits["rows"], key=lambda r: (r["arm_id"], r["seed"])):
        arm = row["arm_id"]
        fit_rows.append({
            "seed": row["seed"], "arm_id": arm,
            "prior_queue_status": row["queue_status"],
            "prior_scientific_status": "FAILED_AFTER_INFEASIBLE_STAGE1" if arm == "A_SEQ21" else row["scientific_status"],
            "prior_returncode": row["returncode"],
            "selected_policy_present": row["selected_policy_present"],
            "explicit_exact_identity_checkpoint_in_old_fit": arm in ("A_JOINT_DET", "A_TASK_DET"),
            "identity_action_support": "PRESENT_FIRST_GLOBAL_CANONICAL_CODE",
            "identity_fitting_utility": "FAIL_OCCUPATION_U0_NLL" + ("_AND_BRIER" if row["seed"] == 1 else ""),
            "identity_common_inner_utility": "PASS_POINT_BOTH_REFERENCES",
            "disposition": (
                "STAGE2_NOT_RUN_AFTER_NO_FEASIBLE_FIRST_STAGE" if arm == "A_SEQ21" else
                "COMPLETE_SEARCH_NO_SELECTED_FEASIBLE_CHECKPOINT"
            ),
            "interpretation": "No global infeasibility certificate; exact identity itself fails FIT caps.",
            "prior_fit_receipt_sha256": row["private_fit_receipt_sha256"],
            "prior_queue_receipt_sha256": row["queue_receipt_sha256"],
        })
    assert len(fit_rows) == 16
    evidence = {
        "schema": "pcrl-code-utility-r2-identity-source-audit-v1",
        "scope": "Public aggregate replay and static old implementation only; no role/person arrays or new fits",
        "sources_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                           (PREFIT, FITS, BANK, INNER_POINTS, BANK_SOURCE, SOURCE,
                            DET_SOURCE, FACTOR_SOURCE, SEQ_SOURCE)},
        "metric_conventions": {
            "accuracy": "numpy argmax first tie; unweighted mean; cost reference minus identity",
            "nll": "natural log; probability floor 1e-12; unweighted mean",
            "brier": "sum over all classes, then unweighted mean",
            "comparison": "same ordered people within each OSF_DEFENSE_FIT or INNER_SELECTION role; U0 and Ucal separately",
            "point_boundary": "inclusive cost <= cap",
            "caps": {"accuracy_drop": .010, "nll_excess": .010, "brier_excess": .005},
            "old_fit_constraints": "four NLL/Brier means on OSF_DEFENSE_FIT; accuracy is a common-INNER service criterion, not an old fitting constraint",
        },
        "I1": {
            "action_and_public_diagnostic": "PASS_BOTH_SEEDS_BOTH_TASKS",
            "original_stochastic_parametric_law": "FAIL_NO_EXPLICIT_EXACT_IDENTITY_CHECKPOINT; finite real softmax weights give positive mass to every available action",
            "original_deterministic_control": "PASS_EXACT_ARGMAX_ACTION_ZERO_AT_IDENTITY_BIASED_START",
            "original_product_local_sequential_stochastic": "FAIL_NO_EXPLICIT_EXACT_IDENTITY_CHECKPOINT; finite real factor softmax weights give positive mass to every available action",
            "numeric_note": "Floating-point underflow at extreme logits is not a declared exact identity checkpoint and was not the old identity-biased initialization.",
            "note": "An explicit eta=0 mixture could repair I1 for a compatible stochastic family, but does not repair I2.",
        },
        "I2": "FAIL_BOTH_SEEDS_OCCUPATION_U0_NLL; SEED1_ALSO_U0_BRIER",
        "I3": "PASS_POINT_BOTH_SEEDS_BOTH_TASKS_BOTH_REFERENCES; NO_INTERVAL_CLAIM",
        "identity_is_continuous_U0": False,
        "identity_privacy_constraint": "Old protected FIT local Bayes ceilings equal identity by construction; identity meets them at equality. Pair Bayes was an objective, not a hard pair-benefit cap. Utility I2 already fails.",
        "a_continuation_gate": "STOP_NO_ADMITTED_IDENTITY_FIT_WITNESS_UNDER_CURRENT_FORMULATION",
        "prior_searches": {"planned": 16, "complete_no_selected": 14,
                           "incomplete_after_infeasible_first_stage": 2,
                           "selected_protected_policy": 0},
        "independent_saved_aggregate_crosscheck": {
            "source_replay_identity_and_reference_point_metrics_vs_later_fixed_endpoint":
                checked_inner_points,
            "maximum_tolerance": 1e-12,
        },
        "per_seed_task": summary,
    }
    return rows, fit_rows, evidence


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    utility, fits, evidence = build()
    write_csv(HERE / "CHANNEL_IDENTITY_AUDIT.csv", utility)
    write_csv(HERE / "CHANNEL_SEARCH_DISPOSITION.csv", fits)
    (HERE / "CHANNEL_IDENTITY_RECEIPT.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n")
