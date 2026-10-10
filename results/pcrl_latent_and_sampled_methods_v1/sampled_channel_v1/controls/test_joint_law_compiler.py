"""Exact small-law and numeric replay checks for sequential sampling aliases."""

from fractions import Fraction
import unittest

from joint_law_compiler import (compile_joint_actions, compile_both_orders,
                                verify_fitted_rows)


class JointLawCompilerTest(unittest.TestCase):
    def test_both_orders_replay_a_correlated_joint_law_exactly(self):
        actions = [
            (3, 11, Fraction(1, 4)),
            (3, 12, Fraction(1, 4)),
            (4, 12, Fraction(1, 2)),
        ]
        for compiled in compile_both_orders(actions, (3, 4), (11, 12)):
            self.assertEqual(compiled.replay(), {
                (3, 11): Fraction(1, 4),
                (3, 12): Fraction(1, 4),
                (4, 12): Fraction(1, 2),
            })

    def test_duplicate_public_pair_actions_are_aggregated(self):
        actions = [(3, 11, Fraction(1, 4)), (3, 11, Fraction(1, 4)),
                   (4, 12, Fraction(1, 2))]
        compiled = compile_joint_actions(actions, (3, 4), (11, 12), order="12")
        self.assertEqual(compiled.replay(), {
            (3, 11): Fraction(1, 2), (4, 12): Fraction(1, 2)
        })

    def test_zero_marginal_conditionals_choose_lowest_admissible_code(self):
        actions = [(3, 12, Fraction(1, 1))]
        forward, reverse = compile_both_orders(actions, (3, 4), (11, 12))
        self.assertEqual(forward.conditional_for(4), {11: Fraction(1), 12: Fraction(0)})
        self.assertEqual(reverse.conditional_for(11), {3: Fraction(1), 4: Fraction(0)})
        self.assertEqual(forward.replay(), {(3, 12): Fraction(1)})
        self.assertEqual(reverse.replay(), {(3, 12): Fraction(1)})

    def test_invalid_ids_and_nonprobabilities_refuse(self):
        for actions in (
            [(9, 11, 1.0)],
            [(3, 11, -0.1), (4, 12, 1.1)],
            [(3, 11, 0.7)],
            [(3, 11, float("nan"))],
        ):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                compile_joint_actions(actions, (3, 4), (11, 12), order="12")

    def test_boolean_action_id_cannot_alias_public_bank_id_one(self):
        with self.assertRaises(ValueError):
            compile_joint_actions([(True, 11, Fraction(1))],
                                  (1, 2), (11, 12), order="12")

    def test_float_replay_residual_is_small(self):
        actions = [(3, 11, 0.1), (3, 12, 0.2), (4, 11, 0.3), (4, 12, 0.4)]
        for compiled in compile_both_orders(actions, (3, 4), (11, 12)):
            replay = compiled.replay()
            self.assertLess(max(abs(replay[(a, b)] - p) for a, b, p in actions), 1e-14)

    def test_every_fitted_row_replays_and_records_numeric_residual(self):
        rows = [
            ([(3, 11, Fraction(1, 2)), (4, 12, Fraction(1, 2))], (3, 4), (11, 12)),
            ([(7, 13, 0.4), (7, 14, 0.6)], (7, 8), (13, 14)),
        ]
        receipt = verify_fitted_rows(rows)
        self.assertEqual(receipt["rows"], 2)
        self.assertEqual(receipt["exact_rational_rows"], 1)
        self.assertEqual(receipt["max_pair_replay_residual"], 0.0)
        self.assertEqual(receipt["max_row_sum_residual"], 0.0)
        self.assertEqual(receipt["orders"], ("12", "21"))

    def test_every_fitted_row_rejects_an_inadmissible_pair(self):
        rows = [
            ([(3, 11, 1.0)], (3,), (11,)),
            ([(7, 13, 1.0)], (7,), (14,)),
        ]
        with self.assertRaisesRegex(ValueError, "row 1"):
            verify_fitted_rows(rows)

    def test_float_mass_residual_is_disclosed(self):
        rows = [([(3, 11, 1.0 + 5e-13)], (3,), (11,))]
        receipt = verify_fitted_rows(rows)
        self.assertGreater(receipt["max_row_sum_residual"], 0.0)
        self.assertLess(receipt["max_row_sum_residual"], 1e-12)

    def test_zero_weight_action_does_not_change_exact_law(self):
        rows = [([(3, 11, Fraction(1)), (4, 12, Fraction(0))],
                 (3, 4), (11, 12))]
        self.assertEqual(verify_fitted_rows(rows)["max_pair_replay_residual"], 0.0)


if __name__ == "__main__":
    unittest.main()
