import unittest

import numpy as np

from bank_anchored.r5_replay_intervals import independent_utility_intervals


class IndependentUtilityIntervalReplay(unittest.TestCase):
    def fixture(self, nll_excess=0.0):
        row = np.asarray([1, 2, 3, 4], dtype=np.int64)
        group = np.asarray([11, 11, 12, 13], dtype=np.int64)
        record = {"row_id": row, "unit": group}
        for source in ("candidate", "B0", "U0", "Ucal"):
            for task in ("income", "occupation"):
                for metric in ("accuracy", "nll", "brier"):
                    value = nll_excess if source == "candidate" and metric == "nll" else 0.0
                    record[f"{source}_{task}_{metric}"] = np.full(4, value, dtype=np.float64)
        return {("J", 0): record, ("J", 1): record}

    def test_exact_zero_margin_and_missing_family_slots(self):
        out = independent_utility_intervals(self.fixture(), expected_groups=3,
                                            bootstrap_draws=32)
        self.assertEqual(len(out), 360)
        self.assertEqual(out["mean:utility:J:income:U0:nll"]["status"], "PASS")
        self.assertEqual(out["mean:utility:J:income:U0:nll"]["se"], 0.0)
        self.assertEqual(out["mean:utility:T:income:U0:nll"]["status"],
                         "NOT_ESTIMABLE_DEPENDENCY")

    def test_constant_nll_above_cap_is_violation(self):
        out = independent_utility_intervals(self.fixture(.02), expected_groups=3,
                                            bootstrap_draws=32)
        self.assertEqual(out["mean:utility:J:occupation:Ucal:nll"]["status"],
                         "VIOLATION")


if __name__ == "__main__":
    unittest.main()
