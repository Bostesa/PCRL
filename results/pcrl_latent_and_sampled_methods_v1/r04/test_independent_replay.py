"""Independent metric and selection fixtures, no private Adult arrays."""

import numpy as np
import pytest
import json
from pathlib import Path

from independent_replay import (
    comparison_margins,
    d_finite_family_ids,
    diagnostic_fallback,
    nominate_within_family,
    reader_recovery,
    simultaneous_group_bounds,
    strongest_matched_control,
    task_metrics,
    task_point_gate,
    weighted_binary_auc,
)


def test_independent_full_d_family_matches_prefit_ordered_manifest():
    path = Path(__file__).resolve().parents[1] / "r02" / "D_FAMILY_ORDER.json"
    manifest = json.loads(path.read_text())
    assert tuple(manifest["ordered_ids"]) == d_finite_family_ids()
    assert manifest["counts"] == {"utility": 252, "arm_pair_auc": 189,
                                  "u_anchor_auc": 126, "total": 567}


def test_task_metric_math_and_wire_decision():
    p = np.array([[0.75, 0.25], [0.2, 0.8]])
    result = task_metrics(p, np.array([0, 1]), decision=np.array([0, 1]), majority_class=0)
    assert result["accuracy"] == 1
    assert result["brier"] == pytest.approx((2 * 0.25**2 + 2 * 0.2**2) / 2)
    assert result["class_recall"] == [1.0, 1.0]
    assert result["accuracy_gain_over_majority"] == 0.5
    with pytest.raises(ValueError):
        task_metrics(p, np.array([0, 1]), decision=np.array([1, 1]))


def test_recovery_ties_weighting_and_all_view_margins():
    score = np.array([0.2, 0.2, 0.9, 0.1])
    sex = np.array([0, 1, 1, 0])
    assert weighted_binary_auc(score, sex) == pytest.approx(0.875)
    assert weighted_binary_auc(score, sex, np.array([2, 1, 1, 1])) == pytest.approx(5 / 6)
    report = reader_recovery({"tied": score, "reversed": 1 - score}, sex)
    assert report["auc"] == pytest.approx(0.875)
    margin = comparison_margins({"r1": .7, "r2": .8, "pair": .75},
                                {"r1": .69, "r2": .8, "pair": .79})
    assert margin == pytest.approx({"r1_local": 0, "r2_local": .01, "pair_benefit": .02})


def test_task_gate_and_fallback_use_only_task_costs():
    candidate = {task: {"accuracy": .8, "nll": .5, "brier": .3}
                 for task in ("income", "occupation")}
    refs = {task: {ref: {"accuracy": .8, "nll": .5, "brier": .3}
                   for ref in ("U0", "Ucal")} for task in candidate}
    assert task_point_gate(candidate, refs)["point_pass"]
    costs = {"D-K": [{"accuracy_drop": .012, "nll_excess": 0, "brier_excess": 0}],
             "D-C": [{"accuracy_drop": .010, "nll_excess": .013, "brier_excess": 0}]}
    assert diagnostic_fallback(costs, ("D-C", "D-K"))["arm"] == "D-K"


def test_strongest_control_and_family_nomination():
    selected = strongest_matched_control({"T": .78, "L": .72, "S12": .75},
                                         ("T", "L", "S12"), ("T", "L", "S12"))
    assert selected["control"] == "L"
    base = {"admission_valid": True, "completed_controls": ["T", "L"],
            "point_feasible_all_seeds": True,
            "local_margins_by_control": {"T": {"r1": 0, "r2": .01},
                                         "L": {"r1": .005, "r2": .01}},
            "coalition_benefit_by_control": {"T": .04, "L": .025},
            "simplicity_rank": 1}
    weak = {**base, "coalition_benefit_by_control": {"T": .04, "L": .015},
            "simplicity_rank": 0}
    nominee = nominate_within_family({"D-K": base, "D-C": weak},
                                     required_control_ids=("T", "L"), tie_order=("D-K", "D-C"))
    assert nominee["nominee"] == "D-K"
    assert nominee["status"] == "POINT_LEAD"


def test_group_bootstrap_resamples_duplicates_together():
    group = np.array([1, 1, 2, 3])
    seen = []

    def evaluate(weights):
        seen.append(weights.copy())
        return [float(np.mean(weights[:2]) - weights[2])]

    bounds = simultaneous_group_bounds(group, ["fixed-contrast"], evaluate,
                                       seed=3, replicates=30)
    assert all(row[0] == row[1] for row in seen)
    assert bounds.names == ("fixed-contrast",)
