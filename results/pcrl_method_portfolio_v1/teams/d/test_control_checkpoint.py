"""Small synthetic custody and replay fixtures; execute only in S3's queue."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from control_checkpoint import (load_private_control_checkpoint,
                                save_private_control_checkpoint)
from controls_fit import (ContinuousRelease, ControlFitConfig, ControlFitResult,
                          TwoCodeRelease)
from prototype import PurposeNormalizer
from trainer import _Head


BINDING = {"campaign_lock_sha256": "a" * 64,
           "role_export_sha256": "b" * 64,
           "source_model_sha256": "c" * 64,
           "trainer_sha256": "d" * 64}
CONFIG = ControlFitConfig(codes=4, hidden=5, epochs=1, batch_size=4)


def fixture(mode: str) -> ControlFitResult:
    normalizer = PurposeNormalizer(np.zeros((2, 2)), np.ones((2, 2)))
    torch.manual_seed(7)
    heads = (_Head(2, 5, 2), _Head(2, 5, 3))
    if mode in ("LOCAL", "SEQ12", "SEQ21"):
        books = (torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]]),
                 torch.tensor([[0., -2.], [0., -1.], [0., 1.], [0., 2.]]))
        release = TwoCodeRelease(normalizer, books, heads, mode)
    else:
        maps = (None, None)
        if mode == "LEACE":
            from stored_model_eval.defenses import LeaceMap
            audits = {"mean_z": np.zeros(2), "sigma_xx_used": np.eye(2),
                      "sigma_xz": np.zeros((2, 2)),
                      "singular_values_whitened_xz": np.zeros(2)}
            maps = tuple(LeaceMap(np.array([[1.], [0.]]),
                                   np.array([[1., 0.]]), np.zeros(2),
                                   {"provenance": "synthetic fixture"}, audits)
                         for _ in (0, 1))
        release = ContinuousRelease(normalizer, heads, maps, mode)
    return ControlFitResult(release, mode, 0, 8, [{"epoch": 1}],
                            {"synthetic_fixture": True})


class PrivateControlCheckpointTest(unittest.TestCase):
    def test_local_sequence_and_continuous_replay(self):
        probe = np.array([[-1.25, .3], [1.25, -.3]], dtype=np.float32)
        for mode in ("LOCAL", "SEQ12", "SEQ21", "UNCHANGED", "LEACE"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                fit = fixture(mode)
                target = root / "fit"
                manifest = save_private_control_checkpoint(
                    fit, target, BINDING, private_root=root, config=CONFIG)
                self.assertEqual(manifest["status"], "UNCALIBRATED_CONTROL_FIT")
                restored = load_private_control_checkpoint(
                    target, BINDING, private_root=root, expected_mode=mode)
                for purpose in (0, 1):
                    original = fit.release.recipient(purpose, probe)
                    replayed = restored.release.recipient(purpose, probe)
                    for field in ("code_ids", "decoded", "probabilities", "decision"):
                        np.testing.assert_array_equal(original[field], replayed[field])
                with self.assertRaises(FileExistsError):
                    save_private_control_checkpoint(fit, target, BINDING,
                                                    private_root=root, config=CONFIG)
                with self.assertRaises(ValueError):
                    load_private_control_checkpoint(target, BINDING, private_root=root,
                                                    expected_mode="LEACE" if mode != "LEACE" else "LOCAL")

    def test_corruption_and_binding_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "fit"
            save_private_control_checkpoint(fixture("LOCAL"), target, BINDING,
                                            private_root=root, config=CONFIG)
            wrong = dict(BINDING, source_model_sha256="e" * 64)
            with self.assertRaises(ValueError):
                load_private_control_checkpoint(target, wrong, private_root=root,
                                                expected_mode="LOCAL")
            with (target / "weights.npz").open("ab") as stream:
                stream.write(b"altered")
            with self.assertRaises(ValueError):
                load_private_control_checkpoint(target, BINDING, private_root=root,
                                                expected_mode="LOCAL")

    def test_public_root_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            root.chmod(0o755)
            with self.assertRaises(ValueError):
                save_private_control_checkpoint(fixture("UNCHANGED"), root / "fit", BINDING,
                                                private_root=root, config=CONFIG)


if __name__ == "__main__":
    unittest.main()
