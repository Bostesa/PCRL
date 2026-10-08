"""Synthetic-only evaluator checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from common_eval import (apply_global_temperature, assert_same_people, fit_constant_probability,
                         fit_global_temperature, load_calibrator, save_calibrator, task_metrics,
                         utility_against_references)


class CommonEvalTest(unittest.TestCase):
    def test_calibration_and_package_binding(self):
        p = np.array([[0.95, 0.05], [0.95, 0.05], [0.05, 0.95], [0.05, 0.95]])
        y = np.array([0, 1, 1, 0], dtype=np.int64)
        rec = fit_global_temperature(p, y, task="income", arm_id="toy", model_seed=0,
                                     row_id=np.arange(4, dtype=np.int64), unit=np.arange(4, dtype=np.int64))
        self.assertLess(rec["alpha"], 1)
        q = apply_global_temperature(p, rec)
        np.testing.assert_array_equal(np.argmax(q, axis=1), np.argmax(p, axis=1))
        self.assertLess(task_metrics(q, y, task="income")["nll"], task_metrics(p, y, task="income")["nll"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibrator.json"
            digest = save_calibrator(rec, path)
            self.assertEqual(load_calibrator(path, digest), rec)
            with self.assertRaises(ValueError):
                load_calibrator(path, "0" * 64)
            with self.assertRaises(FileExistsError):
                save_calibrator(rec, path)

    def test_brier_full_multiclass_and_constant(self):
        p = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7], [0.2, 0.5, 0.3], [0.3, 0.4, 0.3]])
        # Use occupation width six, padding zero probabilities.
        p = np.pad(p, ((0, 0), (0, 3)))
        y = np.array([0, 2, 1, 5], dtype=np.int64)
        const = fit_constant_probability(y, task="occupation")
        m = task_metrics(p, y, task="occupation", constant_prob=const)
        expected = np.mean(np.sum((p - np.eye(6)[y]) ** 2, axis=1))
        self.assertAlmostEqual(m["brier"], expected)
        self.assertEqual(m["class_counts"], [1, 1, 1, 0, 0, 1])
        self.assertIsNone(m["per_class_recall"][3])
        self.assertIn("accuracy_gain_over_constant", m)

    def test_ordered_pair_identity_and_two_reference_gate(self):
        a = {"row_id": np.array([1, 2], dtype=np.int64), "unit": np.array([4, 5], dtype=np.int64)}
        b = {"row_id": np.array([1, 2], dtype=np.int64), "unit": np.array([5, 4], dtype=np.int64)}
        with self.assertRaisesRegex(ValueError, "mismatch"):
            assert_same_people(a, b)
        base = {"accuracy": 0.8, "nll": 0.5, "brier": 0.3}
        candidate = {"accuracy": 0.795, "nll": 0.507, "brier": 0.304}
        ucal = {"accuracy": 0.8, "nll": 0.498, "brier": 0.3}
        self.assertTrue(utility_against_references(candidate, base, ucal)["point_eligible_both"])
        ucal["nll"] = 0.496
        self.assertFalse(utility_against_references(candidate, base, ucal)["point_eligible_both"])


if __name__ == "__main__":
    unittest.main()
