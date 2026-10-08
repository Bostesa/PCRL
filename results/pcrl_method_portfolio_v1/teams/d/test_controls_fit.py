"""Tiny synthetic control fit/deployment checks; use S3's numerical queue."""

import unittest

import numpy as np
import torch

from controls_fit import (ControlFitConfig, fit_continuous, fit_local_or_sequential,
                          fit_raw_joint_reference, prepare_control_cache)
from trainer import DTrainingData


ROW_SHA = "a" * 64
SOURCE_SHA = ("b" * 64, "c" * 64)


def fixture(sex=True):
    h0 = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.],
                   [2., 0.], [2., 1.], [3., 0.], [3., 1.]], dtype=np.float32)
    h1 = h0[:, ::-1].copy()
    y0 = (h0[:, 0] > 1).astype(np.int64)
    y1 = (h1[:, 0] > 0).astype(np.int64)
    p0 = np.column_stack((1 - (.2 + .6 * y0), .2 + .6 * y0)).astype(np.float32)
    p1 = np.column_stack((1 - (.2 + .6 * y1), .2 + .6 * y1)).astype(np.float32)
    ids = np.arange(len(h0), dtype=np.int64)
    s = np.array([0, 1, 0, 1, 1, 0, 1, 0]) if sex else None
    return DTrainingData(h0, h1, y0, y1, p0, p1, s, ids, ids, "SYNTHETIC")


CFG = ControlFitConfig(codes=4, hidden=4, epochs=1, batch_size=8, pair_features=16)


class ControlFitTests(unittest.TestCase):
    def test_cache_requires_task_only_source_and_binds_identity(self):
        task = fixture(sex=False)
        with self.assertRaises(ValueError):
            prepare_control_cache(fixture(sex=True), 7, CFG,
                                  source_model_sha256=SOURCE_SHA, row_order_sha256=ROW_SHA)
        cache = prepare_control_cache(task, 7, CFG,
                                      source_model_sha256=SOURCE_SHA, row_order_sha256=ROW_SHA)
        cache.validate(task, 7, CFG, SOURCE_SHA, ROW_SHA)
        cache.validate(fixture(sex=True), 7, CFG, SOURCE_SHA, ROW_SHA)
        with self.assertRaises(ValueError):
            cache.validate(task, 8, CFG, SOURCE_SHA, ROW_SHA)
        with self.assertRaises(ValueError):
            cache.validate(task, 7, CFG, SOURCE_SHA, "d" * 64)
        self.assertEqual(cache.pooled_centroids.shape, (4, 2))
        self.assertEqual(len(cache.purpose_centroids), 2)

    def test_local_and_both_sequential_orders_are_purpose_local_at_runtime(self):
        data = fixture(sex=True)
        task = fixture(sex=False)
        cache = prepare_control_cache(task, 7, CFG,
                                      source_model_sha256=SOURCE_SHA, row_order_sha256=ROW_SHA)
        for mode in ("LOCAL", "SEQ12", "SEQ21"):
            with self.subTest(mode=mode):
                result = fit_local_or_sequential(data, 7, mode, CFG,
                                                 normalizer=cache.normalizer,
                                                 initializers=cache.purpose_centroids)
                before = result.release.recipient(0, data.h0)
                result.release.recipient(1, data.h1[::-1].copy())
                after = result.release.recipient(0, data.h0)
                np.testing.assert_array_equal(before["code_ids"], after["code_ids"])
                np.testing.assert_allclose(before["probabilities"], after["probabilities"])
                np.testing.assert_array_equal(before["decision"],
                                              before["probabilities"].argmax(1))
                self.assertEqual(before["metadata"]["code_length_bits"], 2)
                self.assertEqual(len(result.history), 2)
                self.assertFalse(result.map_metadata["inference_requires_other_recipient"])

    def test_misaligned_pairing_refused_before_sequential_fit(self):
        d = fixture(sex=True)
        with self.assertRaises(ValueError):
            DTrainingData(d.h0, d.h1, d.y0, d.y1, d.teacher0, d.teacher1,
                          d.sex, d.row_id0, d.row_id1[::-1], "SYNTHETIC")

    def test_unchanged_and_official_leace_have_label_free_runtime(self):
        task = fixture(sex=False)
        private = fixture(sex=True)
        unchanged = fit_continuous(task, 7, "UNCHANGED", CFG)
        with self.assertRaises(ValueError):
            fit_continuous(private, 7, "UNCHANGED", CFG)
        leace = fit_continuous(private, 7, "LEACE", CFG)
        for result in (unchanged, leace):
            out = result.release.recipient(0, task.h0)
            self.assertEqual(out["code_ids"].shape, (len(task.h0), 0))
            self.assertEqual(out["metadata"]["code_length_bits"], 0)
            self.assertEqual(out["metadata"]["calibration"], "UNFITTED")
            np.testing.assert_allclose(out["probabilities"].sum(1), 1)
        self.assertEqual(leace.map_metadata["map_fits"], 2)
        self.assertEqual(len(leace.map_metadata["leace_fit_row_id_sha256"]), 2)

    def test_raw_joint_adapter_requires_exact_source_pin(self):
        task = fixture(sex=False)
        encode = lambda purpose, x: x if purpose == 0 else x[:, ::-1].copy()
        release, fitted = fit_raw_joint_reference(task, encode, "b" * 64, "b" * 64,
                                                  7, CFG)
        out = release.recipient(1, task.h0)
        self.assertEqual(out["metadata"]["kind"], "RAW_J_ADMITTED")
        self.assertEqual(fitted.mode, "UNCHANGED")
        with self.assertRaises(ValueError):
            fit_raw_joint_reference(task, encode, "b" * 64, "c" * 64, 7, CFG)

    def test_nonfinite_control_weights_rejected(self):
        with self.assertRaises(ValueError):
            ControlFitConfig(learning_rate=float("nan"))
        with self.assertRaises(ValueError):
            ControlFitConfig(reconstruction_weight=float("inf"))


if __name__ == "__main__":
    unittest.main()
