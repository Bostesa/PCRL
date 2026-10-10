"""Synthetic independent original-ID action-map checks."""

import unittest

import numpy as np

from bank_anchored.r5_actions import original_id_actions


class OriginalActionAudit(unittest.TestCase):
    def test_permutation_recovers_original_identity_ids(self):
        table = np.array([[.8, .2], [.7, .3], [.2, .8], [.3, .7]], dtype=np.float64)
        classes = np.array([0, 0, 1, 1], dtype=np.int64)
        old = np.arange(4, dtype=np.int64)
        actions, mapping = original_id_actions(table, classes, old, classes)
        self.assertEqual(tuple(row[0] for row in actions), (0, 1, 2, 3))
        self.assertEqual(set(mapping.tolist()), {0, 1, 2, 3})
        self.assertFalse(np.array_equal(mapping, old))

    def test_duplicate_decoded_state_is_rejected(self):
        table = np.array([[.8, .2], [.8, .2]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "duplicate bank state"):
            original_id_actions(table, np.array([0, 0], dtype=np.int64),
                                np.array([0], dtype=np.int64),
                                np.array([0], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
