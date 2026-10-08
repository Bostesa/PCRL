"""Synthetic-only private D initializer-cache custody checks (S3 queue required)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cache_store import load_private_cache, save_private_cache
from controls_fit import ControlFitConfig, prepare_control_cache
from trainer import DTrainingData


def fixture() -> DTrainingData:
    rng = np.random.default_rng(41)
    h0 = rng.normal(size=(20, 2)).astype(np.float32)
    h1 = rng.normal(size=(20, 2)).astype(np.float32)
    y0 = np.arange(20, dtype=np.int64) % 2
    y1 = np.arange(20, dtype=np.int64) % 3
    p0 = np.full((20, 2), .5, dtype=np.float32)
    p1 = np.full((20, 3), 1 / 3, dtype=np.float32)
    rid = np.arange(20, dtype=np.int64)
    return DTrainingData(h0, h1, y0, y1, p0, p1, None, rid, rid, "SYNTHETIC")


class CacheStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "owner_only"
        self.root.mkdir(mode=0o700)
        self.data = fixture()
        self.config = ControlFitConfig(codes=4, hidden=8, epochs=1,
                                       batch_size=10, pair_features=8)
        self.source = ("a" * 64, "a" * 64)
        self.row_sha = "b" * 64
        self.binding = {"campaign_lock_sha256": "c" * 64,
                        "data_manifest_sha256": "d" * 64,
                        "controls_fit_sha256": "e" * 64,
                        "encoder_adapter_sha256": "f" * 64}

    def test_round_trip_and_exact_alias_validation(self) -> None:
        cache = prepare_control_cache(self.data, 7, self.config,
                                      source_model_sha256=self.source,
                                      row_order_sha256=self.row_sha)
        target = self.root / "cache_7"
        save_private_cache(cache, target, self.binding, private_root=self.root)
        restored = load_private_cache(target, self.binding, private_root=self.root)
        restored.validate(self.data, 7, self.config, self.source, self.row_sha)
        np.testing.assert_array_equal(restored.pooled_centroids, cache.pooled_centroids)
        np.testing.assert_array_equal(restored.normalizer.mean, cache.normalizer.mean)
        self.assertEqual(restored.receipt["cache_id"], cache.receipt["cache_id"])
        with self.assertRaises(FileExistsError):
            save_private_cache(cache, target, self.binding, private_root=self.root)

    def test_corruption_and_public_root_refusal(self) -> None:
        cache = prepare_control_cache(self.data, 7, self.config,
                                      source_model_sha256=self.source,
                                      row_order_sha256=self.row_sha)
        target = self.root / "cache_7"
        save_private_cache(cache, target, self.binding, private_root=self.root)
        with (target / "arrays.npz").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_private_cache(target, self.binding, private_root=self.root)
        public_root = Path(self.temp.name) / "public"
        public_root.mkdir(mode=0o755)
        os.chmod(public_root, 0o755)
        with self.assertRaisesRegex(ValueError, "owner-only"):
            save_private_cache(cache, public_root / "new", self.binding,
                               private_root=public_root)


if __name__ == "__main__":
    unittest.main()
