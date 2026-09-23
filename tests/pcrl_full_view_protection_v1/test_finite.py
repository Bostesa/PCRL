import math
import unittest

import numpy as np

from experiments.pcrl_full_view_protection_v1.finite import (
    conditional_mi,
    information_radius_bracket,
    chain_rule_terms,
)


class FiniteTheoryTests(unittest.TestCase):
    def test_xor_coarse_zero_full_log_two(self):
        # Rows are S,H,T,Y probability masses. Z=T deterministically.
        law = np.zeros((2, 2, 2, 2))
        for s in range(2):
            for h in range(2):
                t = s ^ h
                law[s, h, t, t] = 0.25
        q = np.eye(2)
        self.assertAlmostEqual(conditional_mi(law, q, (1,)), math.log(2), places=12)
        terms = chain_rule_terms(law, q, np.zeros(2, dtype=int))
        self.assertAlmostEqual(terms["coarse"], 0, places=12)
        self.assertAlmostEqual(terms["full"], math.log(2), places=12)
        self.assertAlmostEqual(terms["full"], terms["coarse"] + terms["positive"] - terms["negative"], places=12)

    def test_bsc_radius_has_upper_and_lower_orientation(self):
        q = np.array([[0.75, 0.25], [0.25, 0.75]])
        b = information_radius_bracket(q, tol=1e-12)
        exact = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
        self.assertLessEqual(b["lower"] - 1e-12, exact)
        self.assertGreaterEqual(b["upper"] + 1e-12, exact)
        self.assertLess(b["upper"] - b["lower"], 1e-9)

    def test_unseen_deployment_row_is_included(self):
        q = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        b = information_radius_bracket(q)
        self.assertAlmostEqual(b["upper"], math.log(2), places=9)

    def test_zero_column_and_constant_channel(self):
        q = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = information_radius_bracket(q)
        self.assertAlmostEqual(b["upper"], 0, places=12)

    def test_coalition_conditioning_uses_all_service_axes(self):
        law = np.zeros((2, 1, 2, 2, 2))
        for s in range(2):
            for hb in range(2):
                t = s ^ hb
                law[s, 0, hb, t, t] = 0.25
        q = np.eye(2)
        self.assertAlmostEqual(conditional_mi(law, q, (1,)), 0, places=12)
        self.assertAlmostEqual(conditional_mi(law, q, (1, 2)), math.log(2), places=12)


if __name__ == "__main__":
    unittest.main()
