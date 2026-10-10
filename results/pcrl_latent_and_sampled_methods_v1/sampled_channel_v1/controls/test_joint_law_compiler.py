"""Exact small-law and numeric replay checks for sequential sampling aliases."""

from fractions import Fraction
import unittest

from joint_law_compiler import compile_joint_actions, compile_both_orders


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

    def test_float_replay_residual_is_small(self):
        actions = [(3, 11, 0.1), (3, 12, 0.2), (4, 11, 0.3), (4, 12, 0.4)]
        for compiled in compile_both_orders(actions, (3, 4), (11, 12)):
            replay = compiled.replay()
            self.assertLess(max(abs(replay[(a, b)] - p) for a, b, p in actions), 1e-14)


if __name__ == "__main__":
    unittest.main()
