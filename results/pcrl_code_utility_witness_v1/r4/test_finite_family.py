"""Data-free tests for the full prospective two-capacity clause order."""

import unittest

from .finite_family import ARMS, REFERENCES, ordered_clause_ids


class FiniteFamilyTest(unittest.TestCase):
    def test_full_registry_contains_both_capacity_branches(self):
        self.assertEqual(len(ARMS), 16)
        self.assertEqual(ARMS[:7], ("T0_B10", "TPLUS_B10", "L_B10", "J_B10",
                                    "S12_B10", "S21_B10", "C_B10"))
        self.assertEqual(ARMS[7:14], ("T0_B16", "TPLUS_B16", "L_B16", "J_B16",
                                      "S12_B16", "S21_B16", "C_B16"))
        self.assertEqual(ARMS[-2:], ("W_Q", "W_TSTAR"))
        self.assertEqual(REFERENCES, ("F", "U0", "Ucal"))

    def test_exact_clause_count_and_order(self):
        names = ordered_clause_ids()
        self.assertEqual(len(names), 2376)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "s0:utility:T0_B10:income:F:accuracy")
        self.assertEqual(names[-1], "mean:ref_pair:W_TSTAR:Ucal:pair:auc")
        self.assertIn("s1:arm_pair:J_B16:S12_B16:pair:auc", names)
        self.assertIn("mean:ref_pair:J_B10:F:r1:auc", names)

    def test_absent_branch_still_has_registered_ids(self):
        names = ordered_clause_ids()
        self.assertIn("s0:utility:TPLUS_B10:occupation:Ucal:brier", names)
        self.assertIn("mean:utility:TPLUS_B16:occupation:Ucal:brier", names)
        self.assertIn("mean:arm_pair:J_B10:J_B16:pair:auc", names)


if __name__ == "__main__":
    unittest.main()
