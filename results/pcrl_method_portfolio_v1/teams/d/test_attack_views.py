"""Tiny complete-view and exact hard-code reader checks; use S3's queue."""

import unittest

import numpy as np

from attack_views import (AlignedDViews, CellReader, fixed_cell_bank,
                          validate_recipient_release, view_matrix)


def release(purpose, ids, *, calibration="H-GLOBAL-TEMP"):
    ids = np.asarray(ids, dtype=np.int64).reshape(-1, 1)
    p = np.column_stack((np.where(ids[:, 0] == 0, .8, .2),
                         np.where(ids[:, 0] == 0, .2, .8)))
    return {"code_ids": ids, "decoded": ids.astype(np.float32),
            "probabilities": p, "decision": p.argmax(1),
            "metadata": {"purpose": purpose, "code_length_bits": 10,
                         "block_count": 1, "fallback": False,
                         "stochastic": False, "kind": "LOCAL",
                         "calibration": calibration, "calibrator_sha256": "f" * 64}}


class AttackViewTests(unittest.TestCase):
    def test_complete_view_includes_code_and_public_head_fields(self):
        ids = np.arange(4, dtype=np.int64)
        r0, r1 = release(0, [0, 0, 1, 1]), release(1, [0, 1, 0, 1])
        views = AlignedDViews(ids, ids, ids, ids, r0, r1, "a" * 64)
        single = view_matrix(views, "recipient0")
        pair = view_matrix(views, "pair")
        self.assertEqual(len(pair), 4)
        self.assertEqual(pair.shape[1], 2 * single.shape[1])
        np.testing.assert_array_equal(views.codes("pair"), [[0, 0], [0, 1], [1, 0], [1, 1]])
        self.assertEqual(float(single[0, 0]), 1.0)
        self.assertEqual(float(single[0, 1]), 0.0)

    def test_wrong_order_invalid_id_and_unregistered_wire_fail_closed(self):
        ids = np.arange(4, dtype=np.int64)
        r0, r1 = release(0, [0, 0, 1, 1]), release(1, [0, 1, 0, 1])
        with self.assertRaises(ValueError):
            AlignedDViews(ids, ids, ids, ids[::-1], r0, r1, "a" * 64)
        bad = release(0, [0, 0, 1, 1024])
        with self.assertRaises(ValueError):
            validate_recipient_release(bad, 4, 0)
        bad = release(0, [0, 0, 1, 1])
        bad["metadata"]["fallback"] = True
        with self.assertRaises(ValueError):
            validate_recipient_release(bad, 4, 0)
        bad = release(0, [0, 0, 1, 1])
        bad["extra_score"] = np.zeros(4)
        with self.assertRaises(ValueError):
            validate_recipient_release(bad, 4, 0)
        bad = release(0, [0, 0, 1, 1])
        del bad["metadata"]["calibrator_sha256"]
        with self.assertRaises(ValueError):
            validate_recipient_release(bad, 4, 0)
        bad = release(0, [0, 0, 1, 1])
        bad["metadata"]["kind"] = np.asarray(["a", "b", "a", "b"])
        with self.assertRaises(ValueError):
            validate_recipient_release(bad, 4, 0)
        bad = release(0, [0, 0, 1, 1])
        bad["metadata"]["calibration"] = ["H-GLOBAL-TEMP"] * 4
        with self.assertRaises(ValueError):
            validate_recipient_release(bad, 4, 0)
        bad = release(0, [0, 0, 1, 1])
        bad["probabilities"][0, 0] += 1e-7
        with self.assertRaises(ValueError):
            validate_recipient_release(bad, 4, 0)
        with self.assertRaises(ValueError):
            AlignedDViews(ids, ids, ids, ids, release(0, [0, 0, 1, 1], calibration="UNFITTED"),
                          r1, "a" * 64)

    def test_pair_cell_reader_detects_complementary_xor_clue(self):
        code0 = np.tile([0, 0, 1, 1], 3).reshape(-1, 1)
        code1 = np.tile([0, 1, 0, 1], 3).reshape(-1, 1)
        sex = np.tile([0, 1, 1, 0], 3)
        pair = np.concatenate((code0, code1), axis=1)
        for alpha, reader in fixed_cell_bank(pair, sex).items():
            with self.subTest(alpha=alpha):
                prob = reader.predict_proba(pair)[:, 1]
                self.assertGreater(float(prob[1]), float(prob[0]))
                self.assertGreater(float(prob[2]), float(prob[3]))
        local = CellReader.fit(code0, sex, 1.0).predict_proba(code0)[:, 1]
        np.testing.assert_allclose(local, .5)

    def test_unseen_pair_fallback_preserves_marginal_clue(self):
        train = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int64)
        sex = np.array([0, 0, 1, 1])
        reader = CellReader.fit(train, sex, 1.0)
        unseen = reader.predict_proba(np.array([[0, 1]], dtype=np.int64))
        self.assertEqual(unseen.shape, (1, 2))
        self.assertTrue(np.isfinite(unseen).all())
        np.testing.assert_allclose(unseen.sum(1), 1.0)


if __name__ == "__main__":
    unittest.main()
