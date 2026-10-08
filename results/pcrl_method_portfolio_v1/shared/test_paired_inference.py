"""Synthetic inference contract checks; shortened replicate count for unit test."""

import unittest
from unittest.mock import patch

import numpy as np

import paired_inference as pi


class PairedInferenceTest(unittest.TestCase):
    def test_draw0_only_and_exact_zero_se(self):
        unit = np.arange(2234, dtype=np.int64)
        calls = []

        def margin_at(idx, draw):
            calls.append(draw)
            return np.array([0.1, float(np.mean((idx % 2) * 0.1) - 0.2)])

        with patch.object(pi, "REPLICATES", 16):
            out = pi.paired_group_inference(unit, ("exact_positive", "variable_negative"),
                                            margin_at, b"p" * 32, replicates=16)
        self.assertEqual(set(calls), {0})
        self.assertEqual(out["clauses"][0]["status"], "PASS")
        self.assertEqual(out["clauses"][0]["se"], 0.0)
        self.assertEqual(out["clauses"][0]["numerical_se_status"], "NUMERICALLY_CONSTANT")
        self.assertEqual(out["mechanism_draws_in_primary_interval"], 1)
        self.assertEqual(out["inner_groups"], 2234)

    def test_eight_draw_spread_is_descriptive(self):
        out = pi.eight_draw_sensitivity(3, ("one",),
                                        lambda idx, draw: np.array([float(draw) / 10]))
        self.assertEqual(out["draws"], 8)
        self.assertEqual(out["minimum"], [0.0])
        self.assertEqual(out["maximum"], [0.7])
        self.assertIsNone(out["selected_draw"])


if __name__ == "__main__":
    unittest.main()
