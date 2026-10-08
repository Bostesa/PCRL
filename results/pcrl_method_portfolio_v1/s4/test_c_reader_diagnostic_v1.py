"""Data-free checks for the one-unit C LR_C1 prediction diagnostic."""

import unittest

import numpy as np

from run_c_reader_diagnostic_v1 import compare_predictions, matrix_layout


class PredictionDiagnosticTests(unittest.TestCase):
    def test_tiny_probability_drift_preserves_all_binary_auc_ranks(self) -> None:
        saved = np.array([0.1, 0.2, 0.2, 0.4], dtype=np.float64)
        restored = saved.copy()
        restored[0] = np.nextafter(saved[0], 1.0)
        report = compare_predictions(saved, restored)
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["mismatch_count"], 1)
        self.assertFalse(report["exact_equal"])
        self.assertTrue(report["all_binary_label_auc_invariant"])
        self.assertEqual(report["rank_changed_count"], 0)
        self.assertEqual(report["tie_groups_saved"], 1)
        self.assertEqual(report["tie_groups_restored"], 1)
        self.assertAlmostEqual(report["absolute_delta_max"],
                               float(restored[0] - saved[0]))

    def test_breaking_a_saved_tie_marks_auc_as_potentially_changed(self) -> None:
        saved = np.array([0.1, 0.2, 0.2, 0.4], dtype=np.float64)
        restored = saved.copy()
        restored[2] = np.nextafter(saved[2], 1.0)
        report = compare_predictions(saved, restored)
        self.assertEqual(report["mismatch_count"], 1)
        self.assertEqual(report["rank_changed_count"], 2)
        self.assertEqual(report["tie_groups_saved"], 1)
        self.assertEqual(report["tie_groups_restored"], 0)
        self.assertFalse(report["all_binary_label_auc_invariant"])

    def test_nonfinite_prediction_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            compare_predictions(np.array([0.1, 0.9]),
                                np.array([0.1, np.nan]))

    def test_layout_reports_strides_without_exposing_values(self) -> None:
        x = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
        report = matrix_layout(x)
        self.assertEqual(report["shape"], [3, 4])
        self.assertEqual(report["dtype"], "float64")
        self.assertEqual(report["strides_bytes"], [8, 24])
        self.assertTrue(report["f_contiguous"])
        self.assertFalse(report["c_contiguous"])


if __name__ == "__main__":
    unittest.main()
