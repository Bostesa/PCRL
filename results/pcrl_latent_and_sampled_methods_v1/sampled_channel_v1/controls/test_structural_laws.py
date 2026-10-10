"""Pure public-output laws for matched product and freeze-first controls."""

from fractions import Fraction
import unittest

from structural_laws import compose_frozen_first, product_public_law


class StructuralLawsTest(unittest.TestCase):
    def test_product_uses_independent_draws_from_two_factors(self):
        law = product_public_law(
            [(1, Fraction(3, 4)), (2, Fraction(1, 4))],
            [(7, Fraction(1, 3)), (8, Fraction(2, 3))])
        self.assertEqual(law, ((1, 7, Fraction(1, 4)),
                               (1, 8, Fraction(1, 2)),
                               (2, 7, Fraction(1, 12)),
                               (2, 8, Fraction(1, 6))))

    def test_product_aggregates_duplicate_public_ids(self):
        law = product_public_law(
            [(1, Fraction(1, 4)), (1, Fraction(3, 4))],
            [(7, Fraction(1))])
        self.assertEqual(law, ((1, 7, Fraction(1)),))

    def test_freeze_first_exact_expectation_is_correlated(self):
        first = [(1, Fraction(1, 2)), (2, Fraction(1, 2))]
        second = {1: [(7, Fraction(1))], 2: [(8, Fraction(1))]}
        self.assertEqual(compose_frozen_first(first, second, order="12"),
                         ((1, 7, Fraction(1, 2)), (2, 8, Fraction(1, 2))))
        self.assertEqual(compose_frozen_first(first, second, order="21"),
                         ((7, 1, Fraction(1, 2)), (8, 2, Fraction(1, 2))))

    def test_missing_first_code_conditional_refuses(self):
        with self.assertRaises(ValueError):
            compose_frozen_first([(1, 0.5), (2, 0.5)],
                                 {1: [(7, 1.0)]}, order="12")

    def test_invalid_factor_refuses(self):
        for factor in ([], [(True, 1.0)], [(1, 0.8)],
                       [(1, -0.1), (2, 1.1)]):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                product_public_law(factor, [(7, 1.0)])


if __name__ == "__main__":
    unittest.main()
