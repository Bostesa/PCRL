"""Synthetic tests for the historical D0 INNER utility gate."""

import unittest

import numpy as np

from .d0_utility import evaluate_inner_utility


class D0UtilityTest(unittest.TestCase):
    def test_two_task_two_reference_conjunction(self):
        labels = {"income": np.array([0, 1], dtype=np.int64),
                  "occupation": np.array([0, 1], dtype=np.int64)}
        q1 = np.array([[.9, .1], [.1, .9]], dtype=np.float64)
        q2 = np.array([[.9, .1, 0, 0, 0, 0],
                       [.1, .9, 0, 0, 0, 0]], dtype=np.float64)
        release = {"q1": q1, "q2": q2,
                   "d1": np.array([0, 1], dtype=np.int64),
                   "d2": np.array([0, 1], dtype=np.int64)}
        refs = {"U0": release, "Ucal": release}
        result = evaluate_inner_utility(release, refs, labels)
        self.assertTrue(result["point_eligible_all_tasks_refs"])
        self.assertEqual(result["tasks"]["income"]["candidate"]["accuracy"], 1.0)
        self.assertEqual(result["tasks"]["occupation"]["comparisons"]["Ucal"]["delta"]["nll"], 0.0)

    def test_one_failed_clause_fails_cell(self):
        labels = {"income": np.array([0, 1], dtype=np.int64),
                  "occupation": np.array([0, 1], dtype=np.int64)}
        q1 = np.array([[.9, .1], [.1, .9]], dtype=np.float64)
        q2 = np.array([[.9, .1, 0, 0, 0, 0],
                       [.1, .9, 0, 0, 0, 0]], dtype=np.float64)
        worse = np.array([[.8, .2], [.2, .8]], dtype=np.float64)
        release = {"q1": worse, "q2": q2,
                   "d1": np.array([0, 1], dtype=np.int64),
                   "d2": np.array([0, 1], dtype=np.int64)}
        refs = {"U0": {"q1": q1, "q2": q2},
                "Ucal": {"q1": q1, "q2": q2}}
        result = evaluate_inner_utility(release, refs, labels)
        self.assertFalse(result["point_eligible_all_tasks_refs"])
        self.assertTrue(result["tasks"]["occupation"]["comparisons"]["U0"]["point_eligible"])
        self.assertFalse(result["tasks"]["income"]["comparisons"]["U0"]["point_eligible"])

    def test_decision_disagreement_fails_closed(self):
        labels = {"income": np.array([0], dtype=np.int64),
                  "occupation": np.array([0], dtype=np.int64)}
        release = {"q1": np.array([[.9, .1]], dtype=np.float64),
                   "q2": np.array([[.9, .1, 0, 0, 0, 0]], dtype=np.float64),
                   "d1": np.array([1], dtype=np.int64),
                   "d2": np.array([0], dtype=np.int64)}
        with self.assertRaises(ValueError):
            evaluate_inner_utility(release, {"U0": release, "Ucal": release}, labels)


if __name__ == "__main__":
    unittest.main()
