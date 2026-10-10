"""Data-free fixtures for the separate R5 arithmetic boundary."""

import unittest

import numpy as np

from independent_checks import (
    minimal_public_bits, oriented_auc, point_utility_pass,
    sampled_expected_losses, select_capacity, simultaneous_group_bounds, task_means,
    task_rows, utility_differences, verify_hard_code_wire,
)


class IndependentChecksTest(unittest.TestCase):
    def test_multiclass_loss_tie_and_clipping(self):
        p = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
        rows = task_rows(np.array([0, 1]), p, 3, decisions=np.array([0, 0]))
        self.assertEqual(rows["accuracy"].tolist(), [1.0, 0.0])
        self.assertAlmostEqual(rows["nll"][0], np.log(2))
        self.assertAlmostEqual(rows["nll"][1], -np.log(1e-12))
        self.assertAlmostEqual(rows["brier"][0], 0.5)
        self.assertAlmostEqual(rows["brier"][1], 2.0)

    def test_point_gate_inclusive(self):
        ref = {"accuracy": .8, "nll": .5, "brier": .2}
        cand = {"accuracy": .79, "nll": .51, "brier": .205}
        diff = utility_differences(cand, ref)
        self.assertAlmostEqual(diff["accuracy"], -.010)
        self.assertAlmostEqual(diff["nll"], .010)
        # Decimal input subtraction is not an exact binary representation of
        # the cap. Test the inclusive comparison on the exact cap operands.
        self.assertTrue(point_utility_pass({"accuracy": -.010, "nll": .010,
                                            "brier": .005}))
        self.assertFalse(point_utility_pass({"accuracy": -.010, "nll": .0101,
                                             "brier": .005}))

    def test_public_state_capacity_includes_extra_tag(self):
        p = np.array([[.7, .3], [.7, .3], [.3, .7]])
        d = np.array([0, 0, 1])
        self.assertEqual(minimal_public_bits(p, d)["distinct_public_states"], 2)
        self.assertEqual(minimal_public_bits(p, d, np.array([0, 1, 0]))["distinct_public_states"], 3)
        self.assertEqual(minimal_public_bits(p, d, public_code_ids=np.array([0, 1, 2]))
                         ["distinct_public_states"], 3)

    def test_public_code_reconstruction_and_bypass_rejection(self):
        table = np.repeat(np.array([[.6, .4]]), 1024, axis=0)
        table[1] = [.2, .8]
        bits = np.zeros((2, 10), dtype=np.int8)
        bits[1, 0] = 1
        p = np.array([[.6, .4], [.2, .8]])
        self.assertEqual(verify_hard_code_wire(bits, p, np.array([0, 1]), table)["occupied_states"], 2)
        with self.assertRaises(ValueError):
            verify_hard_code_wire(bits, np.array([[.6, .4], [.3, .7]]), np.array([0, 1]), table)

    def test_tied_auc_and_orientation(self):
        y = np.array([0, 1, 0, 1])
        self.assertEqual(oriented_auc(np.zeros(4), y), .5)
        self.assertEqual(oriented_auc(np.array([0., 1., 0., 1.]), y), 1.)
        self.assertEqual(oriented_auc(np.array([1., 0., 1., 0.]), y), 1.)

    def test_group_bootstrap_paired_constant(self):
        group = np.array([4, 4, 5, 6])
        def fixed(w):
            return np.array([task_means({"accuracy": np.ones(4), "nll": np.zeros(4),
                                         "brier": np.zeros(4)}, w)["accuracy"]])
        receipt = simultaneous_group_bounds(group, fixed, 1, 123, 32)
        self.assertEqual(receipt["point"][0], 1.)
        self.assertEqual(receipt["lower"][0], 1.)
        self.assertEqual(receipt["upper"][0], 1.)

    def test_full_family_bonferroni_critical_uses_missing_slots(self):
        from scipy.stats import t
        group = np.array([0, 1, 2, 3])
        def mean(w):
            return np.array([float(np.dot(w, np.array([0., 1., 0., 1.])) / w.sum())])
        out = simultaneous_group_bounds(group, mean, 1, 123, 128,
                                        full_family_count=2376)
        expected = float(t.ppf(1 - .05 / (2 * 2376), 3))
        self.assertAlmostEqual(out["critical_95"], expected)
        self.assertGreater(out["upper"][0], out["point"][0])

    def test_sampled_loss_is_linear_not_loss_of_mean_probability(self):
        bank = np.array([[.9, .1], [.1, .9]])
        q = np.array([[.5, .5]])
        out = sampled_expected_losses(q, bank, np.array([0]))
        self.assertAlmostEqual(out["nll"][0], (-np.log(.9) - np.log(.1))/2)
        self.assertGreater(out["nll"][0], -np.log(.5))
        one = sampled_expected_losses(np.array([[1., 0.]]), bank, np.array([0]))
        self.assertAlmostEqual(one["nll"][0], -np.log(.9))

    def test_capacity_ladder_requires_all_two_seed_task_reference_clauses(self):
        def rows(bits, nll=0.0):
            return [{"bits": bits, "seed": seed, "task": task, "reference": ref,
                     "accuracy": 0.0, "nll": nll, "brier": 0.0}
                    for seed in (0, 1) for task in ("income", "occupation")
                    for ref in ("U0", "Ucal")]
        self.assertEqual(select_capacity(rows(10))["selected_bits"], 10)
        first = rows(10)
        first[3]["nll"] = .02
        self.assertEqual(select_capacity(first)["status"], "SIXTEEN_BIT_STAGE_REQUIRED")
        self.assertEqual(select_capacity(first + rows(16))["selected_bits"], 16)
        self.assertEqual(select_capacity(first + rows(16, nll=.02))["status"],
                         "LEARNER_RECIPE_NOT_UTILITY_FEASIBLE")
        with self.assertRaises(ValueError):
            select_capacity(rows(10) + rows(10))


if __name__ == "__main__":
    unittest.main()
