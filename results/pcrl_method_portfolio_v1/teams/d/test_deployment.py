"""Synthetic head-on-code deployment checks; numerical run only via S3."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from deployment import DCalibratedDeployment, calibrator_record_hash
from encoder_adapter import FrozenUEncoder, hash_file, schema_hash
from jcv.train import Model
from prototype import PurposeNormalizer
from trainer import TorchDRelease, _Head


def synthetic_temperature(prob: np.ndarray, record: dict) -> np.ndarray:
    shifted = record["alpha"] * np.log(np.maximum(prob, 1e-12))
    exp = np.exp(shifted - shifted.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class DDeploymentTests(unittest.TestCase):
    def test_calibrated_wire_depends_only_on_own_hard_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            torch.save(Model(83, [2, 6], 4, masks=None).state_dict(), path)
            columns = tuple(f"permitted_{i}" for i in range(83))
            encoder = FrozenUEncoder.load(path, hash_file(path), columns,
                                          schema_hash(columns), seed=4)
            normalizer = PurposeNormalizer(np.zeros((2, 16)), np.ones((2, 16)))
            # Every input gets hard ID 0. Any probability variation would be a
            # clean-input/head bypass rather than a head-on-code release.
            codebook = torch.zeros((1024, 16), dtype=torch.float32)
            release = TorchDRelease(normalizer, codebook, (_Head(16, 64, 2),
                                                            _Head(16, 64, 6)))
            records = tuple({"schema": "pcrl-portfolio-global-temp-v1",
                             "task": task, "arm_id": "D_VQ_JOINT", "model_seed": 4,
                             "alpha": 1.3} for task in ("income", "occupation"))
            hashes = tuple(calibrator_record_hash(record) for record in records)
            with self.assertRaisesRegex(ValueError, "calibrator"):
                DCalibratedDeployment(encoder, release, records,
                                      ("a" * 64, hashes[1]), "D_VQ_JOINT", 4,
                                      synthetic_temperature)
            deployed = DCalibratedDeployment(encoder, release, records,
                                             hashes, "D_VQ_JOINT", 4,
                                             synthetic_temperature)
            x = np.zeros((3, 83), dtype=np.float32)
            x[1:] = 2.0
            out0 = deployed.recipient(0, x, columns)
            self.assertEqual(out0["code_ids"].shape, (3, 1))
            self.assertTrue(np.all(out0["code_ids"] == 0))
            np.testing.assert_array_equal(out0["probabilities"],
                                          np.repeat(out0["probabilities"][:1], 3, axis=0))
            self.assertEqual(out0["metadata"]["calibration"], "H-GLOBAL-TEMP")
            self.assertEqual(out0["metadata"]["calibrator_sha256"], hashes[0])
            records[0]["alpha"] = 2.0
            np.testing.assert_array_equal(deployed.recipient(0, x, columns)["probabilities"],
                                          out0["probabilities"])
            # Recipient 0 never evaluates recipient 1's frozen encoder.
            encoder._state["enc.1.0.weight"].fill_(float("nan"))
            again = deployed.recipient(0, x, columns)
            np.testing.assert_array_equal(again["probabilities"], out0["probabilities"])
            deployed.calibrators[0]["alpha"] = 3.0
            with self.assertRaisesRegex(ValueError, "mutated"):
                deployed.recipient(0, x, columns)
            with self.assertRaises(ValueError):
                deployed.recipient(1, x, columns, role="OSF_DEVELOPMENT_ASSESSMENT")


if __name__ == "__main__":
    unittest.main()
