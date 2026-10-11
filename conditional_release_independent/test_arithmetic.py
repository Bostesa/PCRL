"""Fixed data-free fixtures. Must be hashed in engineering lock before run."""
import unittest
import numpy as np
from conditional_release_independent.arithmetic import (
    check_law, change_rate, bayes_objective, utility_rows, integrated_auc,
    exhaustive_auc, validate_group_pairing, lp_certificate, cap_status,
)


class ArithmeticTests(unittest.TestCase):
    def test_aliases_and_real_change(self):
        rows = check_law([[((1,2), .2), ((1,2), .3), ((3,2), .5)]])
        self.assertEqual(change_rate(rows, [(1,2)]), .5)

    def test_mass_failures(self):
        with self.assertRaises(ValueError):
            check_law([[((1,2), .9)]])
        with self.assertRaises(ValueError):
            check_law([[((1,2), 1.1), ((3,2), -.1)]])

    def test_actual_action_loss(self):
        rows = check_law([[((0,0), .5), ((1,0), .5)]])
        costs = utility_rows(rows, [0], {0:[.9,.1], 1:[.6,.4]}, 0)
        self.assertAlmostEqual(costs[0,1], -(.5*np.log(.9)+.5*np.log(.6)))
        self.assertGreater(costs[0,1], -np.log(.75))
        self.assertAlmostEqual(costs[0,2], .5*.02+.5*.32)
        self.assertEqual(costs[0,0], 1)

    def test_coalition_complement(self):
        rows = check_law([[((0,0),.5),((1,1),.5)], [((0,1),.5),((1,0),.5)]])
        self.assertEqual(bayes_objective(rows, [0,1]), 1.5)

    def test_exact_auc_ties_and_exhaustion(self):
        laws = [{0:.25,1:.75}, {0:.5,1:.5}, {0:.75,1:.25}, {.5:1}]
        sex = [1,1,0,0]
        for weights in ([1,1,1,1], [2,1,3,1]):
            self.assertAlmostEqual(integrated_auc(laws, sex, weights), exhaustive_auc(laws, sex, weights))

    def test_constant_auc(self):
        self.assertEqual(integrated_auc([{.1:1},{.1:1}], [0,1]), .5)

    def test_duplicate_group_sex(self):
        self.assertEqual(validate_group_pairing(['a','a','b'],[0,0,1]), {'a':0,'b':1})
        with self.assertRaises(ValueError):
            validate_group_pairing(['a','a'],[0,1])

    def test_dual_sign_boundary(self):
        # min -x, x<=.5, 0<=x<=1. Optimum .5 and inequality marginal -1.
        result = lp_certificate(np.array([-1.]), np.array([.5]),
            np.array([[1.]]), np.array([.5]), np.zeros((0,1)), np.array([]),
            [0.],[1.],[-1.],[],[0.],[0.])
        self.assertEqual(result['gap'],0)
        self.assertEqual(result['stationarity_residual'],0)
        self.assertEqual(result['dual_sign_violation'],0)
        wrong = lp_certificate(np.array([-1.]), np.array([.5]),
            np.array([[1.]]), np.array([.5]), np.zeros((0,1)), np.array([]),
            [0.],[1.],[1.],[],[0.],[0.])
        self.assertGreater(wrong['dual_sign_violation'],0)

    def test_exact_bound_conventions(self):
        self.assertEqual(cap_status(.01,.01,.01), 'PASS')
        self.assertEqual(cap_status(.009,.011,.01), 'UNRESOLVED')
        self.assertEqual(cap_status(.011,.012,.01), 'VIOLATION')
        self.assertEqual(cap_status(.02,.03,.02,'lower'), 'PASS')


if __name__ == '__main__':
    unittest.main()
