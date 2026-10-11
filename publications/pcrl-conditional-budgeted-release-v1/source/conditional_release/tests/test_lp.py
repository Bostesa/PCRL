"""Locked tiny-law correctness tests; run only after engineering seal."""
import unittest
import numpy as np
from scipy import sparse
from conditional_release.lp import (AffineLaw, ConstraintRows, build_mixture_problem,
    solve_protection, solve_linear, direct_bayes, linear_program, solve_program,
    certify_infeasibility, omitted_column_report)


def fixture(mode="conditional", heterogeneous=True, cap=1.):
    identity = np.arange(4, dtype=np.int64)
    donor_code = np.array([4, 4, 5, 5], dtype=np.int64)
    emitted = np.c_[identity, donor_code]
    law = build_mixture_problem(emitted, emitted, identity, identity,
                                np.tile([0., 1.], (4, 1)), np.array([0, 0, 1, 1], np.int64), mode)
    cost = np.c_[np.zeros(4), [1., 1., 9., 9.] if heterogeneous else np.ones(4)]
    offset, coeff = law.expected_coefficients(cost)
    return law, ConstraintRows(["pooled/taskcost"], coeff[None], np.array([cap - offset]))


class LockedLPTests(unittest.TestCase):
    def test_heterogeneity(self):
        sex = np.array([0, 1, 0, 1], np.int64)
        g, gc = fixture("scalar")
        c, cc = fixture()
        gr = solve_protection(g, sex, gc)
        cr = solve_protection(c, sex, cc)
        self.assertTrue(gr["success"] and cr["success"])
        self.assertAlmostEqual(gr["theta"][0], .2)
        np.testing.assert_allclose(cr["theta"], [1., 1 / 9])
        self.assertAlmostEqual(direct_bayes(g, gr["theta"], sex)[0], 9 / 5)
        self.assertAlmostEqual(direct_bayes(c, cr["theta"], sex)[0], 13 / 9)
        intercept, coef = c.change_coefficients()
        change = solve_linear(c, cc, coef, intercept, maximize=True)
        self.assertTrue(change["success"])
        self.assertAlmostEqual(intercept + coef @ change["theta"], 5 / 9)

    def test_homogeneous(self):
        sex = np.array([0, 1, 0, 1], np.int64)
        for mode in ("scalar", "conditional"):
            law, rows = fixture(mode, heterogeneous=False, cap=.5)
            answer = solve_protection(law, sex, rows)
            self.assertTrue(answer["success"])
            self.assertAlmostEqual(direct_bayes(law, answer["theta"], sex)[0], 1.5)

    def test_no_headroom(self):
        law, rows = fixture(cap=0.)
        base, coeff = law.change_coefficients()
        answer = solve_linear(law, rows, coeff, base, maximize=True)
        self.assertTrue(answer["success"])
        np.testing.assert_array_equal(answer["theta"], [0., 0.])

    def test_actual_alias_not_parameter(self):
        codes = np.arange(4, dtype=np.int64)
        emitted = np.repeat(codes[:, None], 2, axis=1)
        law = build_mixture_problem(emitted, emitted, codes, codes,
                                    np.tile([0., 1.], (4, 1)), np.arange(4, dtype=np.int64))
        intercept, coeff = law.change_coefficients()
        self.assertEqual(intercept, 0.)
        np.testing.assert_array_equal(coeff, np.zeros(4))
        np.testing.assert_allclose(law.probabilities(np.ones(4)).sum(1), 1)

    def test_infeasibility_needs_certificate(self):
        law, _ = fixture()
        rows = ConstraintRows(["impossible"], np.zeros((1, 2)), np.array([-.1]))
        program = linear_program(law, rows, np.zeros(2))
        answer = solve_program(program)
        self.assertFalse(answer["success"])
        self.assertFalse(answer["certificate"]["certified"])
        proof = certify_infeasibility(program)
        self.assertTrue(proof["infeasibility_certified"])
        self.assertAlmostEqual(proof["certificate"]["dual_objective"], .1)

    def test_dual_sign_boundary(self):
        law, _ = fixture("scalar")
        answer = solve_linear(law, ConstraintRows([], sparse.csr_matrix((0, 1)), np.empty(0)), [-1.])
        self.assertTrue(answer["success"])
        self.assertEqual(answer["upper_dual"][0], -1.)
        self.assertEqual(answer["lower_dual"][0], 0.)

    def test_simplex_equality_and_duplicate_emitted_actions(self):
        emitted = np.array([[0, 1, 1]], np.int64)
        law = AffineLaw(emitted, emitted, np.array([0], np.int64), np.array([0], np.int64),
                        np.zeros((1, 3)), sparse.eye(3), np.tile([0., 1.], (3, 1)),
                        equality_matrix=np.ones((1, 3)), equality_rhs=np.array([1.]))
        rows = ConstraintRows([], sparse.csr_matrix((0, 3)), np.empty(0))
        base, coeff = law.change_coefficients()
        answer = solve_linear(law, rows, coeff, base, maximize=True)
        self.assertTrue(answer["success"])
        self.assertAlmostEqual(coeff @ answer["theta"], 1.)
        self.assertAlmostEqual(direct_bayes(law, answer["theta"], np.array([0], np.int64))[0], 2.)

    def test_omitted_column_trap(self):
        law, _ = fixture("scalar")
        # min x with x>=1 has a certified restricted optimum 1.
        law.bounds[:] = [0., np.inf]
        program = linear_program(law, ConstraintRows(["demand"], [[-1.]], np.array([-1.])), [1.])
        answer = solve_program(program)
        self.assertTrue(answer["success"])
        pricing = omitted_column_report(program, answer["inequality_dual"], answer["equality_dual"],
                                         [[-1.]], sparse.csr_matrix((0, 1)), [0.])
        self.assertFalse(pricing["all_supplied_columns_dual_feasible"])
        self.assertAlmostEqual(pricing["minimum_reduced_cost"], -1.)


if __name__ == "__main__":
    unittest.main()
