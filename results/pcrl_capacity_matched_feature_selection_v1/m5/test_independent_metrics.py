"""Synthetic arithmetic checks; no candidate scorer or Adult artifact imported."""

import unittest

import numpy as np

from independent_metrics import binary_auc, task_metrics


class IndependentMetricsTests(unittest.TestCase):
    def test_auc_counts_exact_ties_and_reversed_reader(self):
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        self.assertEqual(binary_auc(labels, [0.1, 0.9, 0.1, 0.9]), 1.0)
        self.assertEqual(binary_auc(labels, [0.9, 0.1, 0.9, 0.1]), 0.0)
        self.assertEqual(binary_auc(labels, [0.5, 0.5, 0.5, 0.5]), 0.5)
        self.assertEqual(binary_auc(labels, [0.2, 0.8, 0.8, 0.2]), 0.5)

    def test_binary_task_metrics_use_summed_brier_and_natural_log(self):
        labels = np.array([0, 1], dtype=np.int64)
        p = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
        got = task_metrics(labels, p, width=2)
        self.assertEqual(got["accuracy"], 1.0)
        self.assertAlmostEqual(got["nll"], (-np.log(.8) - np.log(.7)) / 2)
        self.assertAlmostEqual(got["brier"], (.08 + .18) / 2)
        self.assertEqual(got["class_counts"], [1, 1])
        self.assertEqual(got["per_class_recall"], [1.0, 1.0])

    def test_missing_class_and_invalid_probabilities(self):
        got = task_metrics([0, 1], [[.9, .1, 0], [.1, .9, 0]], width=3)
        self.assertEqual(got["class_counts"], [1, 1, 0])
        self.assertEqual(got["per_class_recall"], [1.0, 1.0, None])
        with self.assertRaises(ValueError):
            task_metrics([0], [[.2, .2]], width=2)
        with self.assertRaises(ValueError):
            binary_auc([1, 1], [.2, .8])


if __name__ == "__main__":
    unittest.main()
