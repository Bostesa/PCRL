"""Synthetic-only private D package and pre-access seal checks (S3 queue)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from deployment import DCalibratedDeployment, calibrator_record_hash
from encoder_adapter import FrozenUEncoder, hash_file, schema_hash
from jcv.train import Model
from private_runner import _check_seal, _save_role_package
from prototype import PurposeNormalizer
from trainer import TorchDRelease, _Head


def synthetic_temperature(prob: np.ndarray, record: dict) -> np.ndarray:
    shifted = record["alpha"] * np.log(np.maximum(prob, 1e-12))
    exp = np.exp(shifted - shifted.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class FakeRoleStore:
    def __init__(self, feature: dict):
        self.feature = feature
        self.requests = []

    def read(self, role: str, field: str, procedure=None) -> dict:
        self.requests.append((role, field, procedure))
        if role != "ATTACK_FIT_NEW" or field != "features" or procedure is not None:
            raise AssertionError("unexpected label or role read")
        return self.feature


class PrivateRunnerTests(unittest.TestCase):
    def test_draft_seal_refuses_before_role_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            lock_path = folder / "CAMPAIGN_LOCK.json"
            lock_path.write_text(json.dumps({"schema": "pcrl-method-portfolio-campaign-lock-v1",
                                             "status": "BLOCKING_DRAFT_CONTINUATION_NOT_SEALED",
                                             "adult_fit_permission": False}) + "\n")
            with self.assertRaisesRegex(PermissionError, "not effective"):
                _check_seal(lock_path, hash_file(lock_path),
                            folder / "LOCAL_SEAL_RECEIPT.json",
                            folder / "DATA_ACCESS_MANIFEST.json", folder / "shared")

    def test_private_complete_wire_has_no_labels_or_clean_latents(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            private_root = folder / "private"
            private_root.mkdir(mode=0o700)
            run_dir = private_root / "d_run"
            run_dir.mkdir(mode=0o700)
            source = folder / "model.pt"
            torch.save(Model(83, [2, 6], 4, masks=None).state_dict(), source)
            columns = tuple(f"permitted_{i}" for i in range(83))
            encoder = FrozenUEncoder.load(source, hash_file(source), columns,
                                          schema_hash(columns), seed=4)
            normalizer = PurposeNormalizer(np.zeros((2, 16)), np.ones((2, 16)))
            release = TorchDRelease(normalizer, torch.zeros((1024, 16)),
                                    (_Head(16, 64, 2), _Head(16, 64, 6)))
            records = tuple({"schema": "pcrl-portfolio-global-temp-v1",
                             "task": task, "arm_id": "D_VQ_JOINT", "model_seed": 4,
                             "alpha": 1.0} for task in ("income", "occupation"))
            hashes = tuple(calibrator_record_hash(record) for record in records)
            deployed = DCalibratedDeployment(encoder, release, records, hashes,
                                             "D_VQ_JOINT", 4, synthetic_temperature)
            feature = {"row_id": np.arange(3, dtype=np.int64),
                       "unit": np.arange(3, dtype=np.int64),
                       "X": np.zeros((3, 83), dtype=np.float32),
                       "feature_names": np.asarray(columns)}
            store = FakeRoleStore(feature)
            manifest = {"real_file_specs": {"ATTACK_FIT_NEW:features": {
                "row_unit_pair_sha256": "e" * 64, "sha256": "f" * 64}}}
            fit_record = {"checkpoint_weights_sha256": "a" * 64,
                          "codebook_sha256": "b" * 64,
                          "head_probability_conversion":
                          "torch float32 softmax -> float64 cast -> row normalization"}
            summary = _save_role_package(run_dir, private_root, store, manifest,
                                         encoder, deployed, "ATTACK_FIT_NEW", fit_record)
            self.assertEqual(store.requests, [("ATTACK_FIT_NEW", "features", None)])
            self.assertEqual(summary["rows"], 3)
            self.assertTrue(summary["source_parity"])
            package = run_dir / "ATTACK_FIT_NEW"
            with np.load(package / "wire.npz", allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"row_id", "unit",
                    "r0_code_ids", "r0_decoded", "r0_probabilities", "r0_decision",
                    "r1_code_ids", "r1_decoded", "r1_probabilities", "r1_decision"})
                np.testing.assert_array_equal(archive["unit"], feature["unit"])
            self.assertEqual(json.loads((package / "manifest.json").read_text())
                             ["calibrator_sha256"], list(hashes))
            original_complete = (package / "COMPLETE.json").read_bytes()
            (package / "COMPLETE.json").unlink()
            resumed = _save_role_package(run_dir, private_root, store, manifest,
                                         encoder, deployed, "ATTACK_FIT_NEW", fit_record)
            self.assertEqual(resumed["wire_npz_sha256"], summary["wire_npz_sha256"])
            self.assertEqual((package / "COMPLETE.json").read_bytes(), original_complete)


if __name__ == "__main__":
    unittest.main()
