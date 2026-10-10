import unittest

import numpy as np

from bank_anchored.r5_independent import (
    audit_table,
    aggregate_pair_law,
    expected_costs,
    public_id_from_bits,
    saved_law_to_pairs,
)


class IndependentScalarChecks(unittest.TestCase):
    def test_global_id_is_ten_little_endian_bits(self):
        bits = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
        self.assertEqual(public_id_from_bits(bits).tolist(), [5, 1023])
        with self.assertRaises(ValueError):
            public_id_from_bits(np.array([[2] * 10], dtype=np.uint8))

    def test_table_rejects_duplicate_probability_and_decision(self):
        table = np.array([[0.8, 0.2], [0.8, 0.2]])
        with self.assertRaises(ValueError):
            audit_table(table, np.array([0, 0]))

    def test_table_decision_and_simplex(self):
        self.assertEqual(audit_table(np.array([[0.8, 0.2], [0.3, 0.7]]),
                                     np.array([0, 1]))["states"], 2)
        with self.assertRaises(ValueError):
            audit_table(np.array([[0.8, 0.2]]), np.array([1]))

    def test_duplicate_actions_aggregate_into_one_public_pair(self):
        actions = np.array([[[0, 1], [0, 1], [1, 0]]], dtype=np.int64)
        mass = np.array([[0.2, 0.3, 0.5]])
        law = aggregate_pair_law(actions, mass, (2, 2))
        self.assertEqual(law, [{(0, 1): 0.5, (1, 0): 0.5}])
        with self.assertRaises(ValueError):
            aggregate_pair_law(actions, np.array([[0.2, 0.3, 0.4]]), (2, 2))

    def test_expected_loss_scores_emitted_vectors_not_average(self):
        tables = (np.array([[0.9, 0.1], [0.1, 0.9]]),
                  np.array([[0.8, 0.2], [0.2, 0.8]]))
        law = [{(0, 0): 0.5, (1, 1): 0.5}]
        out = expected_costs(law, tables, np.array([[0, 0]]))
        self.assertAlmostEqual(out["income"]["nll"],
                               -0.5 * (np.log(0.9) + np.log(0.1)))
        self.assertAlmostEqual(out["occupation"]["brier"],
                               0.5 * (2 * 0.2**2 + 2 * 0.8**2))
        self.assertGreater(out["income"]["nll"], -np.log(0.5))

    def test_identity_expected_cost_matches_bank_scalar(self):
        tables = (np.array([[0.9, 0.1], [0.1, 0.9]]),
                  np.array([[0.8, 0.2], [0.2, 0.8]]))
        law = [{(0, 1): 1.0}, {(1, 0): 1.0}]
        out = expected_costs(law, tables, np.array([[0, 1], [1, 0]]))
        self.assertAlmostEqual(out["income"]["nll"], -np.log(0.9))
        self.assertAlmostEqual(out["occupation"]["nll"], -np.log(0.8))
        self.assertAlmostEqual(out["income"]["accuracy"], 1.0)
        self.assertAlmostEqual(out["occupation"]["accuracy"], 1.0)

    def test_saved_law_requires_exact_identity_action_and_zero_invalid_mass(self):
        tables = (np.array([[.9, .1], [.7, .3]]),
                  np.array([[.8, .2], [.6, .4]]))
        law = {
            "row_id": np.array([2, 3], dtype=np.int64),
            "unit": np.array([2, 3], dtype=np.int64),
            "code_r1": np.array([[0, 1], [1, 1]], dtype=np.int64),
            "code_r2": np.array([[0, 1], [1, 1]], dtype=np.int64),
            "mass": np.array([[.5, .5], [1., 0.]], dtype=np.float64),
            "valid": np.array([[True, True], [True, False]]),
            "b0_code_r1": np.array([0, 1], dtype=np.int64),
            "b0_code_r2": np.array([0, 1], dtype=np.int64),
        }
        self.assertEqual(saved_law_to_pairs(law, tables, law["row_id"],
                                            law["unit"])[0],
                         {(0, 0): .5, (1, 1): .5})
        bad = {**law, "mass": np.array([[.5, .5], [.9, .1]])}
        with self.assertRaises(ValueError):
            saved_law_to_pairs(bad, tables, law["row_id"], law["unit"])
        bad = {**law, "code_r1": np.array([[1, 0], [1, 1]])}
        with self.assertRaises(ValueError):
            saved_law_to_pairs(bad, tables, law["row_id"], law["unit"])


if __name__ == "__main__":
    unittest.main()
