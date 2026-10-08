"""Pure utility and reporting checks for the focused feature study."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "results/pcrl_method_portfolio_v1/shared"))

from endpoints import public_per_class, score_task_bundle


def record(prob):
    n = len(prob)
    return {"row_id": np.arange(1, n + 1, dtype=np.int64),
            "unit": np.arange(101, 101 + n, dtype=np.int64),
            "prob": np.asarray(prob, dtype=np.float64)}


def labels(y):
    n = len(y)
    return {"row_id": np.arange(1, n + 1, dtype=np.int64),
            "unit": np.arange(101, 101 + n, dtype=np.int64),
            "income": np.asarray(y, dtype=np.int64)}


def test_three_references_are_active_and_F_can_fail_alone():
    y = labels([0, 0, 1, 1, 0, 1])
    candidate = record([[0.7, 0.3], [0.6, 0.4], [0.4, 0.6],
                        [0.3, 0.7], [0.8, 0.2], [0.2, 0.8]])
    perfect = record([[0.99, 0.01], [0.99, 0.01], [0.01, 0.99],
                      [0.01, 0.99], [0.99, 0.01], [0.01, 0.99]])
    output = score_task_bundle(
        candidate, y, {"U0": candidate, "Ucal": candidate, "F": perfect},
        task="income", fit_constant_prob=np.asarray([0.5, 0.5]),
    )
    assert output["references"]["U0"]["point_eligible"] is True
    assert output["references"]["Ucal"]["point_eligible"] is True
    assert output["references"]["F"]["point_eligible"] is False
    assert output["point_eligible_all_references"] is False


def test_utility_rejects_row_permutation_and_uses_fit_constant():
    y = labels([0, 0, 0, 0, 1, 1])
    candidate = record([[0.8, 0.2]] * 4 + [[0.2, 0.8]] * 2)
    shifted = record(candidate["prob"])
    shifted["row_id"] = shifted["row_id"][::-1]
    with pytest.raises(ValueError, match="identity"):
        score_task_bundle(candidate, y, {"U0": shifted, "Ucal": candidate,
                                         "F": candidate}, task="income",
                          fit_constant_prob=np.asarray([0.1, 0.9]))
    result = score_task_bundle(candidate, y, {"U0": candidate, "Ucal": candidate,
                                               "F": candidate}, task="income",
                               fit_constant_prob=np.asarray([0.1, 0.9]))
    assert result["candidate"]["constant_accuracy"] == 2 / 6
    assert result["candidate"]["accuracy_gain_over_constant"] == 1 - 2 / 6


@pytest.mark.parametrize("field", ["row_id", "unit"])
def test_utility_rejects_non_int64_identity_before_inherited_cast(field):
    y = labels([0, 1])
    candidate = record([[0.8, 0.2], [0.2, 0.8]])
    bad = record(candidate["prob"])
    bad[field] = bad[field].astype(np.float64) + 0.9
    with pytest.raises(ValueError, match="exact int64"):
        score_task_bundle(candidate, y, {"U0": bad, "Ucal": candidate,
                                         "F": candidate}, task="income",
                          fit_constant_prob=np.asarray([0.5, 0.5]))


def test_public_per_class_suppresses_small_support_recall():
    metrics = {"class_counts": [10, 4, 0], "per_class_recall": [0.8, 0.5, None]}
    result = public_per_class(metrics)
    assert result == [
        {"class_id": 0, "support": 10, "recall": 0.8, "recall_status": "REPORTED"},
        {"class_id": 1, "support": 4, "recall": None,
         "recall_status": "SUPPRESSED_SUPPORT_LT5"},
        {"class_id": 2, "support": 0, "recall": None,
         "recall_status": "UNDEFINED_NO_SUPPORT"},
    ]
