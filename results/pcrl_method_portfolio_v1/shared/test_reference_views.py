"""Synthetic complete-view checks; no Adult role files are read."""

import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from reference_views import feature_ref_raw83_u


class FeatureReferenceViewTests(unittest.TestCase):
    def test_all_wire_components_and_public_recomputation(self):
        names = tuple(f"synthetic_{j}" for j in range(83))
        digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
        row = np.arange(5, dtype=np.int64)
        unit = row + 100
        features = {"row_id": row, "unit": unit,
                    "X": np.arange(5 * 83, dtype=np.float32).reshape(5, 83),
                    "feature_names": names}
        q1 = np.tile([0.2, 0.8], (5, 1))
        q2 = np.tile([0.1, 0.2, 0.3, 0.15, 0.15, 0.1], (5, 1))
        refs = {"row_id": row.copy(), "unit": unit.copy(),
                "Ucal_1": q1, "Ucal_2": q2,
                "U0_1": q1.copy(), "U0_2": q2.copy(),
                "d_1": np.ones(5, dtype=np.int64),
                "d_2": np.full(5, 2, dtype=np.int64)}
        with patch("reference_views.FEATURE_NAMES_SHA256", digest):
            views = feature_ref_raw83_u(features, refs, source_model_sha256="a" * 64)
        self.assertEqual(views["r1"].x.shape, (5, 87))
        self.assertEqual(views["r2"].x.shape, (5, 95))
        self.assertEqual(views["pair"].x.shape, (5, 182))
        self.assertEqual(views["pair"].x_defense_aware.shape, (5, 190))
        self.assertEqual(views["pair"].x_raw_only.shape, (5, 83))
        self.assertEqual(len(views["pair"].wire_constants), 8)
        np.testing.assert_array_equal(views["pair"].x[:, :83], features["X"])
        np.testing.assert_array_equal(views["pair"].x[:, 87:170], features["X"])

    def test_rejects_reference_row_permutation(self):
        names = tuple(f"synthetic_{j}" for j in range(83))
        digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
        features = {"row_id": np.arange(2, dtype=np.int64),
                    "unit": np.arange(2, dtype=np.int64),
                    "X": np.zeros((2, 83), dtype=np.float32), "feature_names": names}
        refs = {"row_id": np.array([1, 0]), "unit": np.array([1, 0])}
        with patch("reference_views.FEATURE_NAMES_SHA256", digest), self.assertRaises(ValueError):
            feature_ref_raw83_u(features, refs, source_model_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
