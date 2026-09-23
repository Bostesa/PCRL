import math
import unittest

import numpy as np

from experiments.pcrl_full_view_protection_v1.finite import conditional_mi
from experiments.pcrl_full_view_protection_v1.synthetic import fixtures, task_cost, task_information
from experiments.pcrl_full_view_protection_v1.optimize import (
    optimize_channel, _conditional_tables, _cmi_gradient,
)


class SyntheticTests(unittest.TestCase):
    def test_registered_laws_normalized_and_xor(self):
        suite = fixtures()
        for name, case in suite.items():
            for law in case["laws"]:
                self.assertAlmostEqual(law.sum(), 1, places=12, msg=name)
                self.assertGreaterEqual(law.min(), 0, msg=name)
        xor = suite["xor"]["laws"][0]
        self.assertAlmostEqual(conditional_mi(xor, np.eye(2), (1,)), math.log(2), places=12)

    def test_safe_task_full_information_without_leakage(self):
        law = fixtures()["safe_independent"]["laws"][0]
        self.assertAlmostEqual(conditional_mi(law, np.eye(2), (1,)), 0, places=12)
        self.assertAlmostEqual(task_cost(law, np.eye(2)), 0, places=12)
        self.assertAlmostEqual(task_information(law, np.eye(2)), math.log(2), places=12)

    def test_xor_zero_budget_forces_no_useful_binary_channel(self):
        law = fixtures()["xor"]["laws"][0]
        result = optimize_channel(law, [law], budget=0.0, method="robust")
        self.assertLessEqual(result["max_full_view_cmi"], 1e-7)
        self.assertAlmostEqual(result["cost"], 0.5, delta=1e-6)
        self.assertLessEqual(result["objective_lower_bound"], result["cost"] + 1e-9)
        self.assertLess(result["cost"] - result["objective_lower_bound"], 1.1e-7)

    def test_positive_budget_has_affine_dual_bound(self):
        law = fixtures()["xor"]["laws"][0]
        result = optimize_channel(law, [law], budget=0.01, method="robust")
        self.assertLessEqual(result["objective_lower_bound"], result["cost"] + 1e-9)
        self.assertLess(result["cost"] - result["objective_lower_bound"], 1e-5)

    def test_cmi_gradient_matches_simplex_direction(self):
        law = fixtures()["partly_coupled"]["laws"][0]
        q = np.array([[0.7, 0.3], [0.2, 0.8]])
        direction = np.array([[1.0, -1.0], [-1.0, 1.0]])
        eps = 1e-6
        for role, axes in (("A", (1,)), ("AB", (1, 2))):
            grad = _cmi_gradient(_conditional_tables(law, role), q)
            finite_difference = (conditional_mi(law, q + eps * direction, axes) -
                                 conditional_mi(law, q - eps * direction, axes)) / (2 * eps)
            self.assertAlmostEqual(float(np.sum(grad * direction)), finite_difference, places=7)

    def test_prior_sensitivity_uses_coherent_nine_law_set(self):
        case = fixtures()["prior_sensitivity"]
        self.assertEqual(len(case["laws"]), 9)
        priors = []
        for law in case["laws"]:
            self.assertAlmostEqual(law.sum(), 1, places=12)
            self.assertAlmostEqual(law.sum(axis=(0, 2, 3, 4))[0], 0.5, places=12)
            priors.append(law[1, 0].sum() / law[:, 0].sum())
            for s in range(2):
                for ha in range(2):
                    self.assertAlmostEqual(law[s, ha, 1 - (s ^ ha)].sum(), 0, places=12)
        self.assertEqual(set(np.round(priors, 12)), {0.2, 0.25, 0.3})


if __name__ == "__main__":
    unittest.main()
