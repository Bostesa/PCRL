"""Synthetic arithmetic checks; no PCRL role or release files are read."""

import math
import unittest

import numpy as np

from .loss_metrics import code_states, distillation_kl, task_metrics


class LossMetricsTest(unittest.TestCase):
    def test_binary_task_arithmetic_and_class_recall(self):
        p = np.array([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]], dtype=np.float64)
        y = np.array([0, 1, 1], dtype=np.int64)
        result = task_metrics(p, y, "income")
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["nll"], -math.log(0.8 * 0.6 * 0.3) / 3)
        self.assertAlmostEqual(result["brier"], (0.08 + 0.32 + 0.98) / 3)
        self.assertEqual(result["class_counts"], [1, 2])
        self.assertEqual(result["per_class_recall"], [1.0, 0.5])
        self.assertAlmostEqual(result["accuracy_gain_over_constant"], 0.0)

    def test_teacher_kl_and_ten_bit_states(self):
        p = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)
        q = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
        expected = 0.8 * math.log(1.6) + 0.2 * math.log(0.4)
        self.assertAlmostEqual(distillation_kl(p, q, "income"), expected)
        bits = np.zeros((4, 10), dtype=np.uint8)
        bits[1, 0] = bits[2, 0] = 1
        result = code_states(bits)
        self.assertEqual(result["global_capacity_states"], 1024)
        self.assertEqual(result["occupied_states"], 2)
        self.assertAlmostEqual(result["observed_entropy_bits"], 1)

    def test_reject_invalid_probability(self):
        with self.assertRaises(ValueError):
            task_metrics(np.array([[0.4, 0.4]], dtype=np.float64),
                         np.array([1], dtype=np.int64), "income")


if __name__ == "__main__":
    unittest.main()
