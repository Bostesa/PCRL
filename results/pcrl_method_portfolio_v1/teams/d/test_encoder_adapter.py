"""Engineering fixtures for the frozen-U preprivacy latent adapter.

These use only invented 83-column inputs. Numerical execution needs S3 admission.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from encoder_adapter import FrozenUEncoder, hash_file, schema_hash


class FrozenUEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        from jcv.train import Model

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model_path = Path(self.temp.name) / "model.pt"
        model = Model(83, [2, 6], seed=7, masks=None).eval()
        torch.save(model.state_dict(), self.model_path)
        self.sha = hash_file(self.model_path)
        self.columns = tuple(f"permitted_{i:02d}" for i in range(83))
        self.schema_sha = schema_hash(self.columns)
        self.x = np.arange(4 * 83, dtype=np.float32).reshape(4, 83) / 1000

    def test_exact_preprivacy_source_parity_and_locality(self) -> None:
        adapter = FrozenUEncoder.load(self.model_path, self.sha, self.columns,
                                      self.schema_sha, seed=7)
        h0, h1, receipt = adapter.encode_with_receipt(
            self.x, self.columns, role="OSF_DEFENSE_FIT")
        self.assertEqual(h0.shape, (4, 16))
        self.assertEqual(h1.shape, (4, 16))
        self.assertEqual(h0.dtype, np.float32)
        self.assertTrue(receipt["bitwise_source_parity"])
        self.assertEqual(receipt["tensor"], "Model.encode(i, X) before maps/head")
        self.assertEqual(receipt["source_model_sha256"], self.sha)
        self.assertNotIn("path", receipt)
        np.testing.assert_array_equal(adapter.encode_one(0, self.x, self.columns,
                                                         role="DEPLOYMENT"), h0)
        np.testing.assert_array_equal(adapter.encode_one(1, self.x, self.columns,
                                                         role="DEPLOYMENT"), h1)
        changed = self.x.copy()
        changed[0, 0] += 1
        other0, other1 = adapter.encode_pair(changed, self.columns,
                                            role="OSF_DEFENSE_FIT")
        np.testing.assert_array_equal(h0[1:], other0[1:])
        np.testing.assert_array_equal(h1[1:], other1[1:])
        # The recipient-0 path never reads the other purpose's encoder weights.
        adapter._state["enc.1.0.weight"].fill_(float("nan"))
        np.testing.assert_array_equal(adapter.encode_one(0, self.x, self.columns,
                                                         role="DEPLOYMENT"), h0)

    def test_rejects_wrong_model_schema_role_and_nonfinite_x(self) -> None:
        with self.assertRaisesRegex(ValueError, "source checkpoint SHA"):
            FrozenUEncoder.load(self.model_path, "0" * 64, self.columns,
                                self.schema_sha, seed=7)
        adapter = FrozenUEncoder.load(self.model_path, self.sha, self.columns,
                                      self.schema_sha, seed=7)
        with self.assertRaisesRegex(ValueError, "83 permitted columns"):
            adapter.encode_pair(self.x[:, :-1], self.columns,
                                role="OSF_DEFENSE_FIT")
        with self.assertRaisesRegex(ValueError, "column order"):
            adapter.encode_pair(self.x, self.columns[::-1],
                                role="OSF_DEFENSE_FIT")
        with self.assertRaisesRegex(ValueError, "forbidden role"):
            adapter.encode_with_receipt(self.x, self.columns,
                                        role="OSF_DEVELOPMENT_ASSESSMENT")
        with self.assertRaisesRegex(ValueError, "forbidden role"):
            adapter.encode_pair(self.x, self.columns, role="DEPLOYMENT")
        with self.assertRaisesRegex(ValueError, "forbidden role"):
            adapter.encode_one(0, self.x, self.columns,
                               role="OSF_DEVELOPMENT_ASSESSMENT")
        bad = self.x.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            adapter.encode_pair(bad, self.columns, role="OSF_DEFENSE_FIT")

    def test_model_state_must_be_exact_u_architecture(self) -> None:
        state = torch.load(self.model_path, weights_only=True, map_location="cpu")
        state["enc.0.0.weight"] = state["enc.0.0.weight"][:, :-1]
        torch.save(state, self.model_path)
        with self.assertRaisesRegex(ValueError, "U architecture"):
            FrozenUEncoder.load(self.model_path, hash_file(self.model_path),
                                self.columns, self.schema_sha, seed=7)


if __name__ == "__main__":
    unittest.main()
