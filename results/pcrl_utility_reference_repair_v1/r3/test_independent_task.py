import unittest

import numpy as np

from independent_task import CAPS, gate_against_references, task_metrics


class IndependentTaskTests(unittest.TestCase):
    def test_natural_log_and_summed_multiclass_brier(self):
        p = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.int64)
        got = task_metrics(p, y)
        self.assertEqual(got["accuracy"], 1.0)
        self.assertAlmostEqual(got["nll"], (-np.log(0.8) - np.log(0.9)) / 2)
        self.assertAlmostEqual(got["brier"], ((0.2**2 + 0.2**2) + (0.1**2 + 0.1**2)) / 2)
        self.assertEqual(got["recall"], [1.0, 1.0])

    def test_inclusive_caps_and_both_references(self):
        baseline = {"accuracy": 0.9, "nll": 0.4, "brier": 0.2}
        candidate = {"accuracy": 0.89, "nll": 0.41, "brier": 0.205}
        result = gate_against_references(candidate, {"U0": baseline, "Ucal": baseline})
        self.assertEqual(result["caps"], CAPS)
        self.assertTrue(result["point_eligible"])
        self.assertEqual(result["margins"]["U0"], {"accuracy": 0.0, "nll": 0.0, "brier": 0.0})

    def test_one_reference_failure_blocks_joint_gate(self):
        candidate = {"accuracy": 0.88, "nll": 0.4, "brier": 0.2}
        result = gate_against_references(candidate, {
            "U0": {"accuracy": 0.9, "nll": 0.4, "brier": 0.2},
            "Ucal": {"accuracy": 0.88, "nll": 0.39, "brier": 0.2},
        })
        self.assertFalse(result["point_eligible"])
        self.assertFalse(result["by_reference"]["U0"])
        self.assertTrue(result["by_reference"]["Ucal"])

    def test_rejects_misaligned_or_bad_probabilities(self):
        for p, y in [
            (np.array([[0.5, 0.5]]), np.array([0, 1])),
            (np.array([[0.6, 0.5]]), np.array([0])),
            (np.array([[1.0, 0.0]]), np.array([2])),
        ]:
            with self.assertRaises(ValueError):
                task_metrics(p, y)


if __name__ == "__main__":
    unittest.main()
