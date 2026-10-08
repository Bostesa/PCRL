"""Data-free regression for exact C replay across equal-value matrix layouts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from replay_c_readers_v1 import replay_model_predictions as replay_v1
from replay_c_readers_v2 import replay_model_predictions as replay_v2


class LayoutSensitiveModel:
    """Synthetic stand-in for a model whose reduction follows input strides."""

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        probability = 0.2 + 0.1 * matrix[:, 0]
        if matrix.flags.f_contiguous and not matrix.flags.c_contiguous:
            probability = probability.copy()
            probability[0] = np.nextafter(probability[0], 1.0)
        return np.column_stack((1.0 - probability, probability))


class ExactLayoutReplayTests(unittest.TestCase):
    def test_raw_lr_uses_c_contiguous_values_and_keeps_exact_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "synthetic.joblib"
            joblib.dump(LayoutSensitiveModel(), model_path)
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            att = np.ascontiguousarray(np.array([[1.0, 2.0], [3.0, 4.0],
                                                [5.0, 6.0]]))
            inner_c = np.ascontiguousarray(np.array([[1.0, 2.0], [3.0, 4.0],
                                                    [5.0, 6.0]]))
            inner_f = np.asfortranarray(inner_c)
            self.assertTrue(np.array_equal(inner_c, inner_f))
            self.assertTrue(inner_f.flags.f_contiguous)
            self.assertFalse(inner_f.flags.c_contiguous)
            tune = np.array([0, 2], dtype=np.int64)
            saved_tune = LayoutSensitiveModel().predict_proba(att[tune])[:, 1]
            saved_inner = LayoutSensitiveModel().predict_proba(inner_c)[:, 1]
            with self.assertRaisesRegex(ValueError, "inner prediction differs"):
                replay_v1(model_path, model_sha, "RAW_LR_C1",
                          {"x_raw_only": att}, {"x_raw_only": inner_f},
                          tune, saved_tune, saved_inner)
            replay_v2(model_path, model_sha, "RAW_LR_C1",
                      {"x_raw_only": att}, {"x_raw_only": inner_f},
                      tune, saved_tune, saved_inner)
            bad = saved_inner.copy()
            bad[1] = np.nextafter(bad[1], 1.0)
            with self.assertRaisesRegex(ValueError, "inner prediction differs"):
                replay_v2(model_path, model_sha, "RAW_LR_C1",
                          {"x_raw_only": att}, {"x_raw_only": inner_f},
                          tune, saved_tune, bad)


if __name__ == "__main__":
    unittest.main()
