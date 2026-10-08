"""Private checkpoint and exact code/head deployment replay tests.

Run only through S3's numerical queue. These fixtures contain no Adult data.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import load_private_checkpoint, save_private_checkpoint
from prototype import PurposeNormalizer
from trainer import DFitResult, DTrainConfig, TorchDRelease, _Head


class CheckpointTests(unittest.TestCase):
    def _fit_fixture(self):
        torch.manual_seed(17)
        normalizer = PurposeNormalizer(np.zeros((2, 2)), np.ones((2, 2)))
        codebook = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
        heads = (_Head(2, 4, 2), _Head(2, 4, 3))
        release = TorchDRelease(normalizer, codebook, heads)
        config = DTrainConfig(codes=2, hidden=4, epochs=1, batch_size=4, pair_features=8)
        return DFitResult(release, history=[{"epoch": 1, "objective": 1.0}],
                          role="SYNTHETIC", seed=17, protected=True, fit_rows=4,
                          fit_code_change={"rows": 4}, config=config)

    def _binding(self):
        return {"campaign_lock_sha256": "a" * 64,
                "role_export_sha256": "b" * 64,
                "source_model_sha256": "c" * 64,
                "trainer_sha256": "d" * 64}

    def test_private_save_restore_replays_wire_outputs_bitwise(self):
        fit = self._fit_fixture()
        h = np.array([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]], dtype=np.float32)
        before = fit.release.recipient(0, h)
        self.assertEqual(before["metadata"]["calibration"], "UNFITTED")
        with tempfile.TemporaryDirectory() as tmp:
            private_root = Path(tmp)
            target = private_root / "checkpoint_v1"
            save_private_checkpoint(fit, target, self._binding(), private_root=private_root)
            self.assertEqual((target.stat().st_mode & 0o777), 0o700)
            self.assertEqual(((target / "weights.npz").stat().st_mode & 0o777), 0o600)
            restored = load_private_checkpoint(target, self._binding(), private_root=private_root)
            after = restored.release.recipient(0, h)
            for key in ("code_ids", "decoded", "probabilities", "decision"):
                np.testing.assert_array_equal(after[key], before[key])
            self.assertEqual(after["metadata"], before["metadata"])
            self.assertEqual(restored.config, fit.config)
            self.assertEqual(restored.fit_code_change, fit.fit_code_change)

    def test_checkpoint_refuses_corrupt_weights_and_wrong_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            private_root = Path(tmp)
            target = private_root / "checkpoint_v1"
            save_private_checkpoint(self._fit_fixture(), target, self._binding(), private_root=private_root)
            wrong = dict(self._binding(), source_model_sha256="e" * 64)
            with self.assertRaises(ValueError):
                load_private_checkpoint(target, wrong, private_root=private_root)
            with (target / "weights.npz").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaises(ValueError):
                load_private_checkpoint(target, self._binding(), private_root=private_root)

    def test_checkpoint_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            private_root = Path(tmp)
            target = private_root / "checkpoint_v1"
            fit = self._fit_fixture()
            save_private_checkpoint(fit, target, self._binding(), private_root=private_root)
            with self.assertRaises(FileExistsError):
                save_private_checkpoint(fit, target, self._binding(), private_root=private_root)

    def test_public_permission_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            public_root = Path(tmp) / "public"
            public_root.mkdir(mode=0o755)
            with self.assertRaises(ValueError):
                save_private_checkpoint(self._fit_fixture(), public_root / "checkpoint_v1",
                                        self._binding(), private_root=public_root)
            self.assertFalse((public_root / "checkpoint_v1").exists())


if __name__ == "__main__":
    unittest.main()
