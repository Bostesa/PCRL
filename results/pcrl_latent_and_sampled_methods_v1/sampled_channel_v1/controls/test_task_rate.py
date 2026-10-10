"""Task-only score-channel rate is computed on emitted public pair IDs."""

import math
import unittest

from task_rate import empirical_public_pair_rate


class TaskRateTest(unittest.TestCase):
    def test_same_random_public_law_for_all_inputs_has_zero_rate(self):
        rows = [[(1, 4, 0.5), (2, 5, 0.5)],
                [(1, 4, 0.5), (2, 5, 0.5)]]
        result = empirical_public_pair_rate(rows)
        self.assertAlmostEqual(result.rate_nats, 0.0)
        self.assertAlmostEqual(result.marginal_entropy_nats, math.log(2))
        self.assertAlmostEqual(result.conditional_entropy_nats, math.log(2))

    def test_deterministic_distinct_outputs_have_rate_equal_to_entropy(self):
        rows = [[(1, 4, 1.0)], [(2, 5, 1.0)]]
        result = empirical_public_pair_rate(rows)
        self.assertAlmostEqual(result.rate_nats, math.log(2))
        self.assertEqual(result.conditional_entropy_nats, 0.0)
        self.assertEqual(result.occupied_pairs, 2)

    def test_duplicate_actions_are_aggregated_before_entropy(self):
        rows = [[(1, 4, 0.25), (1, 4, 0.25), (2, 5, 0.5)],
                [(1, 4, 0.5), (2, 5, 0.5)]]
        self.assertAlmostEqual(empirical_public_pair_rate(rows).rate_nats, 0.0)

    def test_weights_apply_to_people_not_to_public_actions(self):
        rows = [[(1, 4, 1.0)], [(2, 5, 1.0)]]
        result = empirical_public_pair_rate(rows, weights=[1.0, 3.0])
        expected = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
        self.assertAlmostEqual(result.rate_nats, expected)

    def test_invalid_public_law_refuses(self):
        for rows in ([], [[(1, 4, 0.7)]], [[(True, 4, 1.0)]],
                     [[(1, 4, float("nan"))]], [[(1, 4, -0.1), (2, 5, 1.1)]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                empirical_public_pair_rate(rows)


if __name__ == "__main__":
    unittest.main()
