"""Data-free checks for full-K bounds and AUC direction conventions."""

import unittest

from .inference_rule import (
    CRITICAL_95, interval_bounds, orient_auc_contrast, utility_status,
)


class InferenceRuleTest(unittest.TestCase):
    def test_full_family_student_critical_and_zero_se(self):
        self.assertAlmostEqual(CRITICAL_95, 4.262627512538624, places=14)
        self.assertEqual(interval_bounds(0.03, 0.0), (0.03, 0.03))
        lo, hi = interval_bounds(0.03, 0.01)
        self.assertAlmostEqual(lo, 0.03 - CRITICAL_95 * 0.01)
        self.assertAlmostEqual(hi, 0.03 + CRITICAL_95 * 0.01)

    def test_pair_benefit_reverses_raw_candidate_minus_control(self):
        result = orient_auc_contrast(
            "mean:arm_pair:J_B16:S12_B16:pair:auc",
            point=-0.03, lower=-0.04, upper=-0.02,
            candidate="J_B16", control="S12_B16")
        self.assertEqual(result["direction"], "control_minus_candidate")
        self.assertEqual((result["point"], result["lower"], result["upper"]),
                         (0.03, 0.02, 0.04))
        self.assertEqual(result["status"], "PASS")

    def test_local_worsening_keeps_raw_candidate_minus_control(self):
        result = orient_auc_contrast(
            "s0:arm_pair:L_B10:J_B10:r1:auc",
            point=-0.005, lower=-0.012, upper=0.002,
            candidate="J_B10", control="L_B10")
        self.assertEqual(result["direction"], "candidate_minus_control")
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertEqual((result["lower"], result["upper"]), (-0.002, 0.012))

    def test_reference_order_and_unmatched_clause_rejected(self):
        result = orient_auc_contrast(
            "s1:ref_pair:J_B10:F:pair:auc", point=-0.04,
            lower=-0.05, upper=-0.03, candidate="J_B10", control="F")
        self.assertEqual((result["point"], result["lower"], result["upper"]),
                         (0.04, 0.03, 0.05))
        with self.assertRaises(ValueError):
            orient_auc_contrast("s1:arm_pair:J_B10:C_B10:pair:auc",
                                point=0, lower=-1, upper=1,
                                candidate="J_B10", control="F")

    def test_utility_inclusive_boundaries_and_missing(self):
        self.assertEqual(utility_status("accuracy", -.010, -.010, -.009), "PASS")
        self.assertEqual(utility_status("nll", .010, .009, .010), "PASS")
        self.assertEqual(utility_status("brier", .005, .004, .006), "UNRESOLVED")
        self.assertEqual(utility_status("brier", .008, .006, .010), "VIOLATION")
        self.assertEqual(utility_status("nll", None, None, None),
                         "NOT_ESTIMABLE_DEPENDENCY")


if __name__ == "__main__":
    unittest.main()
