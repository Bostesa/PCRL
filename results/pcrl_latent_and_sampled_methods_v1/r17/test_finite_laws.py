"""Data-free, exact finite-law checks independent of candidate code."""

import math
import unittest
from fractions import Fraction as F

from finite_laws import (
    aggregate_public_pairs,
    binary_hamming_kernel,
    canonical_public_bank,
    deduplicate_before_normalizing,
    empirical_biased_mmd2,
    empirical_public_pair_laws,
    expected_sampled_cost,
    loss_of_averaged_vector,
    public_output_key,
)


class HardCodeMMDTests(unittest.TestCase):
    def test_xor_has_zero_local_mmd_and_positive_pair_mmd(self):
        # Every (a,b) occurs once; SEX = a XOR b. Recipient marginals match.
        rows = [(a ^ b, (a,), (b,)) for a in (0, 1) for b in (0, 1)]
        for recipient in (1, 2):
            group0 = [row[recipient] for row in rows if row[0] == 0]
            group1 = [row[recipient] for row in rows if row[0] == 1]
            self.assertEqual(
                empirical_biased_mmd2(group0, group1, lambda x, y: binary_hamming_kernel(x, y, F(1, 2))),
                0,
            )
        pair0 = [(r[1], r[2]) for r in rows if r[0] == 0]
        pair1 = [(r[1], r[2]) for r in rows if r[0] == 1]

        def product(x, y):
            return binary_hamming_kernel(x[0], y[0], F(1, 2)) * binary_hamming_kernel(x[1], y[1], F(1, 2))

        self.assertEqual(empirical_biased_mmd2(pair0, pair1, product), F(1, 4))

    def test_balanced_null_law_has_zero_in_every_view(self):
        rows = [(sex, (a,), (b,)) for sex in (0, 1) for a in (0, 1) for b in (0, 1)]
        kernel = lambda x, y: binary_hamming_kernel(x, y, F(1, 2))
        for index in (1, 2):
            self.assertEqual(empirical_biased_mmd2(
                [r[index] for r in rows if r[0] == 0],
                [r[index] for r in rows if r[0] == 1], kernel), 0)
        pair = lambda x, y: kernel(x[0], y[0]) * kernel(x[1], y[1])
        self.assertEqual(empirical_biased_mmd2(
            [(r[1], r[2]) for r in rows if r[0] == 0],
            [(r[1], r[2]) for r in rows if r[0] == 1], pair), 0)

    def test_xor_multiscale_product_kernel_detects_pair_only(self):
        rows = [(a ^ b, (a,), (b,)) for a in (0, 1) for b in (0, 1)]
        widths = (F(1, 2), F(1, 4))

        def multiscale(a, b):
            return sum((binary_hamming_kernel(a, b, rho) for rho in widths), F(0)) / len(widths)

        pair = lambda x, y: multiscale(x[0], y[0]) * multiscale(x[1], y[1])
        group0 = [(r[1], r[2]) for r in rows if r[0] == 0]
        group1 = [(r[1], r[2]) for r in rows if r[0] == 1]
        self.assertEqual(empirical_biased_mmd2(group0, group1, pair), F(25, 64))

    def test_xor_with_proposed_four_gamma_scales(self):
        rows = [(a ^ b, (a,), (b,)) for a in (0, 1) for b in (0, 1)]
        gammas = (0.25, 0.5, 1.0, 2.0)

        def multiscale(a, b):
            distance = sum(x != y for x, y in zip(a, b))
            return sum(math.exp(-gamma * distance) for gamma in gammas) / len(gammas)

        group0 = [(r[1], r[2]) for r in rows if r[0] == 0]
        group1 = [(r[1], r[2]) for r in rows if r[0] == 1]
        pair = lambda x, y: multiscale(x[0], y[0]) * multiscale(x[1], y[1])
        rho_bar = sum(math.exp(-gamma) for gamma in gammas) / len(gammas)
        self.assertAlmostEqual(float(empirical_biased_mmd2(group0, group1, pair)), (1 - rho_bar) ** 2, places=14)
        self.assertEqual(empirical_biased_mmd2([r[1] for r in rows if r[0] == 0],
                                                [r[1] for r in rows if r[0] == 1], multiscale), 0)
        self.assertEqual(empirical_biased_mmd2([r[2] for r in rows if r[0] == 0],
                                                [r[2] for r in rows if r[0] == 1], multiscale), 0)

    def test_task_preserving_collapse_removes_a_sex_bit(self):
        rows = [(task, sex, (task, sex), (task,)) for task in (0, 1) for sex in (0, 1)]
        kernel = lambda x, y: binary_hamming_kernel(x, y, F(1, 2))
        original = empirical_biased_mmd2(
            [r[2] for r in rows if r[1] == 0],
            [r[2] for r in rows if r[1] == 1], kernel)
        repaired = empirical_biased_mmd2(
            [r[3] for r in rows if r[1] == 0],
            [r[3] for r in rows if r[1] == 1], kernel)
        self.assertGreater(original, 0)
        self.assertEqual(repaired, 0)
        self.assertTrue(all(task == code[0] for task, _, _, code in rows))

    def test_no_complementarity_when_first_code_already_reveals_sex(self):
        rows = [(a, (a,), (b,)) for a in (0, 1) for b in (0, 1)]
        kernel = lambda x, y: binary_hamming_kernel(x, y, F(1, 2))
        pair = lambda x, y: kernel(x[0], y[0]) * kernel(x[1], y[1])
        self.assertEqual(empirical_biased_mmd2(
            [r[1] for r in rows if r[0] == 0], [r[1] for r in rows if r[0] == 1], kernel), 1)
        self.assertEqual(empirical_biased_mmd2(
            [r[2] for r in rows if r[0] == 0], [r[2] for r in rows if r[0] == 1], kernel), 0)
        self.assertEqual(empirical_biased_mmd2(
            [(r[1], r[2]) for r in rows if r[0] == 0],
            [(r[1], r[2]) for r in rows if r[0] == 1], pair), F(3, 4))
        self.assertTrue(all(sex == code1[0] for sex, code1, _ in rows))


class SampledChannelTests(unittest.TestCase):
    def test_expected_draw_loss_exceeds_loss_of_averaged_vector(self):
        choices = ((F(9, 10), F(1, 10)), (F(3, 5), F(2, 5)))
        weights = (F(1, 2), F(1, 2))
        draw_nll, draw_brier = expected_sampled_cost(weights, choices, label=0)
        mean_nll, mean_brier = loss_of_averaged_vector(weights, choices, label=0)
        self.assertAlmostEqual(draw_nll, -math.log(F(27, 50)) / 2)
        self.assertAlmostEqual(mean_nll, -math.log(F(3, 4)))
        self.assertGreater(draw_nll, mean_nll)
        self.assertEqual(draw_brier, F(17, 100))
        self.assertEqual(mean_brier, F(1, 8))
        self.assertEqual(draw_brier - mean_brier, F(9, 200))

    def test_duplicate_choices_are_removed_before_normalization(self):
        a = (F(9, 10), F(1, 10))
        b = (F(3, 5), F(2, 5))
        choices, weights = deduplicate_before_normalizing((a, a, b), (F(1), F(1), F(1)))
        self.assertEqual(choices, (a, b))
        self.assertEqual(weights, (F(1, 2), F(1, 2)))
        # Normalizing three slots first would have silently doubled A's mass.
        self.assertNotEqual(weights[0], F(2, 3))

    def test_global_ids_ignore_input_local_positions(self):
        a = (F(9, 10), F(1, 10))
        b = (F(3, 5), F(2, 5))
        c = (F(1, 5), F(4, 5))
        choices_sex0 = (a, b)
        choices_sex1 = (b, a)
        bank1 = canonical_public_bank(choices_sex0 + choices_sex1)
        self.assertEqual(bank1, canonical_public_bank(choices_sex1 + choices_sex0))
        self.assertEqual(len(bank1), 2)
        bank2 = canonical_public_bank((c, c))
        self.assertEqual(choices_sex0[0], choices_sex1[1])
        self.assertNotEqual(choices_sex0.index(a), choices_sex1.index(a))
        law_sex0 = aggregate_public_pairs(((F(1), choices_sex0[0], c),), bank1, bank2)
        law_sex1 = aggregate_public_pairs(((F(1), choices_sex1[1], c),), bank1, bank2)
        self.assertEqual(law_sex0, law_sex1)

    def test_calibrated_confidence_can_beat_decision_only_for_same_class(self):
        high = (F(9, 10), F(1, 10))
        low = (F(3, 5), F(2, 5))
        class_only = (F(3, 4), F(1, 4))
        # Ten rows at each context implement the stated Bernoulli laws exactly.
        rows = [(high, 0)] * 9 + [(high, 1)] + [(low, 0)] * 6 + [(low, 1)] * 4
        fine = [expected_sampled_cost((F(1),), (vector,), label) for vector, label in rows]
        coarse = [expected_sampled_cost((F(1),), (class_only,), label) for _, label in rows]
        self.assertTrue(all(public_output_key(vector)[1] == 0 for vector, _ in rows))
        self.assertLess(sum(x[0] for x in fine), sum(x[0] for x in coarse))
        self.assertLess(sum((x[1] for x in fine), F(0)), sum((x[1] for x in coarse), F(0)))

    def test_same_public_pair_aggregates_all_action_mass(self):
        a = (F(9, 10), F(1, 10))
        b = (F(3, 5), F(2, 5))
        c = (F(1, 5), F(4, 5))
        bank1 = canonical_public_bank((a, b))
        bank2 = canonical_public_bank((c,))
        law = aggregate_public_pairs(((F(1, 4), a, c), (F(1, 4), a, c), (F(1, 2), b, c)), bank1, bank2)
        self.assertEqual(law[(bank1[public_output_key(a)], bank2[public_output_key(c)])], F(1, 2))
        self.assertEqual(sum(law.values()), 1)

    def test_sex_group_laws_merge_input_local_action_positions_globally(self):
        a = (F(9, 10), F(1, 10))
        b = (F(3, 5), F(2, 5))
        c = (F(1, 5), F(4, 5))
        bank1 = canonical_public_bank((a, b))
        bank2 = canonical_public_bank((c,))
        rows = (
            (0, ((F(1), a, c), (F(0), b, c))),
            (1, ((F(0), b, c), (F(1), a, c))),
        )
        laws = empirical_public_pair_laws(rows, bank1, bank2)
        self.assertEqual(laws[0], laws[1])
        self.assertEqual(laws[0][(bank1[public_output_key(a)], bank2[public_output_key(c)])], 1)


if __name__ == "__main__":
    unittest.main()
