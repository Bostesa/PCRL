"""Fixed comparison family must cover every possible selected arm pair."""

import unittest

from bank_anchored.r3_family import ordered_clause_ids


class FiniteFamilyTests(unittest.TestCase):
    def test_all_pairwise_and_utility_slots_are_fixed_before_fits(self):
        # Break caught: shrinking multiplicity to observed or promoted arms.
        ids = ordered_clause_ids()
        self.assertEqual(len(ids), 1575)
        self.assertEqual(len(set(ids)), 1575)
        self.assertIn("s0:utility:J:income:Ucal:nll", ids)
        self.assertIn("mean:utility:P_STAR:occupation:U0:brier", ids)
        self.assertIn("s1_d2:recovery:J:P_STAR:pair:auc", ids)
        self.assertIn("mean:recovery:B0:J:r1:auc", ids)
        self.assertIn("s0_mean:recovery:S12:S21:pair:auc", ids)


if __name__ == "__main__":
    unittest.main()
