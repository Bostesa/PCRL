"""Pre-run supplementary L2 analytic witness check."""

import numpy as np

from finite_channel import evaluate
from test_finite_channel import _common_law, _identity


def test_l2_classwise_more_confident_witness_meets_original_caps():
    law = _common_law("L2")
    q1, q2 = _identity(law)
    baseline = evaluate(law, q1, q2)["utility"]
    q = np.zeros((4, 4))
    q[[0, 1, 2, 3], [1, 1, 3, 3]] = 1
    repaired = evaluate(law, q, q)
    assert repaired["accuracy"]["pair"] == .5
    for r in ("r1", "r2"):
        assert repaired["utility"][r]["nll"] <= baseline[r]["nll"] + .010
        assert repaired["utility"][r]["brier"] <= baseline[r]["brier"] + .005
