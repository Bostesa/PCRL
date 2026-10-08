"""Synthetic-only group-shuffle custody check; no fitted control job."""

import unittest

import numpy as np

from attack_controls import groupwise_shuffled_labels


class AttackControlTest(unittest.TestCase):
    def test_group_vectors_move_whole_and_repeat(self):
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 5], dtype=np.int64)
        labels = np.array([0, 1, 1, 1, 0, 0, 0, 1], dtype=np.int64)
        a = groupwise_shuffled_labels(labels, groups)
        b = groupwise_shuffled_labels(labels, groups)
        np.testing.assert_array_equal(a, b)
        original_pairs = sorted(tuple(labels[groups == g]) for g in (1, 2, 3))
        shuffled_pairs = sorted(tuple(a[groups == g]) for g in (1, 2, 3))
        self.assertEqual(original_pairs, shuffled_pairs)
        self.assertEqual(int(np.sum(a)), int(np.sum(labels)))


if __name__ == "__main__":
    unittest.main()
