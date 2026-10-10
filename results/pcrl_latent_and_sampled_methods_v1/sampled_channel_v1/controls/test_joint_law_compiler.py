"""Exact small-law and numeric replay checks for sequential sampling aliases."""

from fractions import Fraction
import unittest

from joint_law_compiler import (compile_joint_actions, compile_both_orders,
                                verify_admitted_rows, verify_fitted_rows)


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

    def test_admitted_rows_bind_complete_order_and_decision_classes(self):
        rows = [
            ("person-a", 0, 1, [(0, 2, Fraction(1))], (0, 1), (2,)),
            ("person-b", 1, 0, [(2, 0, Fraction(1))], (2,), (0, 1)),
        ]
        receipt = verify_admitted_rows(
            rows, expected_row_ids=("person-a", "person-b"),
            bank_decision1=(0, 0, 1), bank_decision2=(0, 0, 1))
        self.assertEqual(receipt["rows"], 2)
        self.assertEqual(len(receipt["row_order_sha256"]), 64)
        self.assertEqual(receipt["max_pair_replay_residual"], 0.0)

    def test_admitted_rows_refuse_wrong_order_missing_row_and_wrong_class(self):
        good = ("person-a", 0, 1, [(0, 2, 1.0)], (0, 1), (2,))
        bad_class = ("person-b", 1, 0, [(2, 0, 1.0)], (2,), (0, 2))
        kwargs = dict(expected_row_ids=("person-a", "person-b"),
                      bank_decision1=(0, 0, 1), bank_decision2=(0, 0, 1))
        for rows in ((good,), (bad_class, good), (good, bad_class)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                verify_admitted_rows(rows, **kwargs)


if __name__ == "__main__":
    unittest.main()
