"""Small, label-free synthetic checks of prospective utility inference."""

import unittest
import numpy as np

from .finite_family import ordered_clause_ids
from .utility_intervals import (BOOTSTRAP_SEED, _weights, cost_vectors,
                                evaluate_utility_family)


class UtilityIntervalTest(unittest.TestCase):
    def test_draws_match_locked_pcg64_integer_sequence(self):
        units = np.array([20, 20, 10, 30], dtype=np.int64)
        draw, inverse = _weights(units)
        sampled = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
            0, 3, size=(2048, 3), dtype=np.int64)
        expected = np.stack([np.bincount(row, minlength=3)
                             for row in sampled])
        self.assertTrue(np.array_equal(draw, expected))
        self.assertTrue(np.array_equal(inverse, [1, 1, 0, 2]))

    def test_cost_signs_and_metrics(self):
        y = np.array([0, 1], dtype=np.int64)
        candidate = np.array([[.8, .2], [.6, .4]], dtype=np.float64)
        reference = np.array([[.6, .4], [.2, .8]], dtype=np.float64)
        costs = cost_vectors(candidate, reference, y, "income")
        self.assertTrue(np.array_equal(costs["accuracy"], [0., -1.]))
        self.assertAlmostEqual(costs["nll"][0], np.log(.6 / .8))
        self.assertAlmostEqual(costs["brier"][1],
                               np.sum((candidate[1] - [0, 1]) ** 2)
                               - np.sum((reference[1] - [0, 1]) ** 2))

    def test_full_family_retains_missing_and_paired_mean(self):
        units = np.array([11, 11, 12, 13], dtype=np.int64)
        y = np.array([0, 1, 0, 1], dtype=np.int64)
        ref = np.array([[.8, .2], [.2, .8], [.7, .3], [.4, .6]], dtype=np.float64)
        same = ref.copy()
        worse = np.array([[.6, .4], [.4, .6], [.6, .4], [.6, .4]], dtype=np.float64)
        candidates = {(0, "T0_B10", "income"): same,
                      (1, "T0_B10", "income"): worse}
        references = {(seed, "income", "U0"): ref for seed in (0, 1)}
        result = evaluate_utility_family(candidates, references,
                                         {"income": y}, units)
        self.assertEqual(result["bootstrap_seed"], BOOTSTRAP_SEED)
        self.assertEqual(result["ordered_ids"], list(ordered_clause_ids()))
        first = result["clauses"]["s0:utility:T0_B10:income:U0:nll"]
        second = result["clauses"]["s1:utility:T0_B10:income:U0:nll"]
        mean = result["clauses"]["mean:utility:T0_B10:income:U0:nll"]
        self.assertAlmostEqual(first["point"], 0.)
        self.assertAlmostEqual(mean["point"], second["point"] / 2)
        self.assertIsNone(result["clauses"]["s0:utility:T0_B16:income:U0:nll"]["point"])
        self.assertEqual(result["clauses"]["s0:utility:T0_B16:income:U0:nll"]["status"],
                         "NOT_ESTIMABLE_DEPENDENCY")
        self.assertIsNone(result["clauses"]["s0:arm_pair:T0_B10:TPLUS_B10:r1:auc"]["point"])
        self.assertEqual(len(result["clauses"]), 2376)


if __name__ == "__main__":
    unittest.main()
