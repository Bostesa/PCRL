"""Data-free transport checks for the fixed sampled-channel output bank."""

import unittest

import numpy as np

from bank import FixedBank, build_actions, symmetric_kl


class FixedBankTest(unittest.TestCase):
    def test_duplicate_public_vectors_have_one_global_id(self):
        source = np.array([[0.8, 0.2], [0.8, 0.2], [0.7, 0.3],
                           [0.2, 0.8]], dtype=np.float64)
        bank = FixedBank.from_source(source, np.arange(len(source)),
                                     np.argmax(source, axis=1))
        self.assertEqual(len(bank.probability), 3)
        self.assertEqual(bank.old_to_global[0], bank.old_to_global[1])
        self.assertTrue(np.array_equal(
            bank.probability[bank.old_to_global[0]], source[0]))
        reordered_source = source[[3, 2, 1, 0]]
        reordered = FixedBank.from_source(reordered_source, np.arange(len(source)),
                                          np.argmax(reordered_source, axis=1))
        self.assertTrue(np.array_equal(bank.probability, reordered.probability))
        self.assertTrue(np.array_equal(bank.decision, reordered.decision))

    def test_actions_are_same_class_global_ids_with_fixed_order(self):
        source = np.array([[.9, .1], [.85, .15], [.8, .2], [.7, .3],
                           [.6, .4], [.2, .8]], dtype=np.float64)
        bank = FixedBank.from_source(source, np.arange(len(source)),
                                     np.argmax(source, axis=1))
        actions = build_actions(
            bank, input_class=np.array([0, 0, 1], dtype=np.int64),
            fine_to_old_token=np.array([0, 1, 5], dtype=np.int64))
        first = actions[0]
        self.assertEqual(first[0], bank.old_to_global[0])
        self.assertEqual(first[1:3], (bank.old_to_global[1], bank.old_to_global[2]))
        self.assertEqual(len(first), len(set(first)))
        self.assertLessEqual(len(first), 4)
        for predicted, choices in zip((0, 0, 1), actions):
            self.assertTrue(all(bank.decision[j] == predicted for j in choices))
        self.assertEqual(actions[2], (bank.old_to_global[5],))

    def test_distance_floor_does_not_change_transmitted_bank(self):
        source = np.array([[1.0, 0.0], [.8, .2], [0.0, 1.0]])
        before = source.copy()
        bank = FixedBank.from_source(source, np.arange(len(source)),
                                     np.argmax(source, axis=1))
        self.assertTrue(np.isfinite(symmetric_kl(source[0], source[1])))
        self.assertTrue(np.array_equal(source, before))
        self.assertTrue(np.array_equal(bank.probability[bank.old_to_global[0]], source[0]))

    def test_ambiguous_class_is_refused(self):
        with self.assertRaises(ValueError):
            source = np.array([[.5, .5], [.8, .2]])
            FixedBank.from_source(source, np.arange(len(source)),
                                  np.argmax(source, axis=1))

    def test_source_token_order_and_class_must_be_pinned(self):
        source = np.array([[.8, .2], [.2, .8]])
        with self.assertRaises(ValueError):
            FixedBank.from_source(source, np.array([1, 0]), np.array([0, 1]))
        with self.assertRaises(ValueError):
            FixedBank.from_source(source, np.array([0, 1]), np.array([1, 0]))

    def test_single_output_sixth_class_keeps_only_identity_action(self):
        source = np.eye(6, dtype=np.float64)
        bank = FixedBank.from_source(source, np.arange(6, dtype=np.int64),
                                     np.arange(6, dtype=np.int64))
        choices = build_actions(bank, input_class=np.array([5], np.int64),
                                fine_to_old_token=np.array([5], np.int64))
        self.assertEqual(choices, ((bank.old_to_global[5],),))
        self.assertEqual(int(bank.decision[choices[0][0]]), 5)


if __name__ == "__main__":
    unittest.main()
