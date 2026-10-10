"""Data-free tests for the bank-anchored expected-output utility contract."""

import unittest

import numpy as np

from bank_anchored.r3_utility import evaluate_law, point_clause


class BankExpectedUtilityTests(unittest.TestCase):
    def setUp(self):
        self.table0 = np.array([[.9, .1], [.1, .9], [.6, .4]], dtype=np.float64)
        self.table1 = np.array([
            [.8, .04, .04, .04, .04, .04],
            [.1, .5, .1, .1, .1, .1],
        ], dtype=np.float64)
        self.law = {
            "code_r1": np.array([[0, 2], [1, 1]], dtype=np.int64),
            "code_r2": np.array([[0, 0], [1, 1]], dtype=np.int64),
            "mass": np.array([[.5, .5], [1., 0.]], dtype=np.float64),
            "valid": np.array([[True, True], [True, False]]),
            "b0_code_r1": np.array([0, 1], dtype=np.int64),
            "b0_code_r2": np.array([0, 1], dtype=np.int64),
        }
        self.labels = (np.array([0, 1], dtype=np.int64),
                       np.array([0, 1], dtype=np.int64))

    def test_expected_loss_scores_each_draw_not_the_mean_probability(self):
        # Break caught: replacing E[loss(p_A)] with loss(E[p_A]).
        result = evaluate_law(self.law, (self.table0, self.table1), self.labels)
        first_nll = .5 * -np.log(.9) + .5 * -np.log(.6)
        self.assertAlmostEqual(result["income"]["expected"]["nll"][0], first_nll)
        self.assertAlmostEqual(result["income"]["expected"]["brier"][0], .17)
        self.assertAlmostEqual(result["income"]["expected"]["accuracy"][0], 1.)
        self.assertNotAlmostEqual(first_nll, -np.log(.75))

    def test_identity_has_exact_zero_bank_relative_cost(self):
        # Break caught: softmax approximation or wrong baseline action.
        identity = {**self.law, "mass": np.array([[1., 0.], [1., 0.]]),
                    "valid": np.array([[True, True], [True, False]])}
        result = evaluate_law(identity, (self.table0, self.table1), self.labels)
        for task in ("income", "occupation"):
            for metric in ("accuracy", "nll", "brier"):
                np.testing.assert_array_equal(result[task]["expected"][metric],
                                              result[task]["b0"][metric])

    def test_duplicate_public_action_mass_is_aggregated_by_expectation(self):
        # Break caught: counting a repeated action as a new public state.
        duplicate = {**self.law, "code_r1": np.array([[0, 0], [1, 1]]),
                     "mass": np.array([[.25, .75], [1., 0.]])}
        result = evaluate_law(duplicate, (self.table0, self.table1), self.labels)
        for task in ("income", "occupation"):
            for metric in ("accuracy", "nll", "brier"):
                np.testing.assert_array_equal(result[task]["expected"][metric],
                                              result[task]["b0"][metric])

    def test_invalid_mass_and_class_switch_are_rejected(self):
        # Break caught: a nonzero invalid slot or action outside B0 class.
        bad = {**self.law, "mass": np.array([[.5, .5], [.9, .1]])}
        with self.assertRaises(ValueError):
            evaluate_law(bad, (self.table0, self.table1), self.labels)
        bad = {**self.law, "code_r1": np.array([[0, 1], [1, 1]])}
        with self.assertRaises(ValueError):
            evaluate_law(bad, (self.table0, self.table1), self.labels)

    def test_inclusive_original_and_fit_boundaries_are_distinct(self):
        # Break caught: reusing FIT tolerances as original INNER caps.
        self.assertEqual(point_clause("accuracy", -.010, "inner")["status"], "PASS")
        self.assertEqual(point_clause("nll", .010, "inner")["status"], "PASS")
        self.assertEqual(point_clause("brier", .005, "inner")["status"], "PASS")
        self.assertEqual(point_clause("accuracy", -.002, "fit")["status"], "PASS")
        self.assertEqual(point_clause("nll", .001, "fit")["status"], "PASS")
        self.assertEqual(point_clause("brier", .0005, "fit")["status"], "PASS")
        self.assertEqual(point_clause("nll", .0011, "fit")["status"], "FAIL")
        self.assertEqual(point_clause("nll", .0011, "inner")["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
