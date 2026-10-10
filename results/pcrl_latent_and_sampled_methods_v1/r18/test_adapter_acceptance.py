"""Small invented-input acceptance fixtures; run only through R03's sealed queue."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import (
    FixedScaler, HardCodeModel, HardCodeRelease,
)
from results.pcrl_latent_and_sampled_methods_v1.r18.acceptance_contract import (
    check_a_public_release, check_cached_pair, check_d_public_release,
    check_raw_schema, check_source_checkpoint,
)


class _SyntheticFrozenEncoder:
    def __init__(self, source_sha256: str, columns: tuple[str, ...]):
        self.model_sha256 = source_sha256
        self.columns = columns

    def encode_one(self, purpose: int, raw: np.ndarray,
                   columns: tuple[str, ...], role: str) -> np.ndarray:
        if tuple(columns) != self.columns or role != "DEPLOYMENT":
            raise ValueError("source encoder schema or role differs")
        if raw.shape != (1, 83) or raw.dtype != np.float32:
            raise ValueError("source encoder raw shape differs")
        return np.asarray(raw[:, :16] * (purpose + 1), dtype=np.float32)


class DeploymentAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = tuple(f"allowed_{i:02d}" for i in range(83))
        self.schema_sha = hashlib.sha256("\n".join(self.columns).encode()).hexdigest()
        self.raw = np.arange(83, dtype=np.float32).reshape(1, 83) / 83
        self.source_sha = "a" * 64

    def test_raw_schema_order_exclusions_and_checkpoint(self) -> None:
        check_raw_schema(self.raw, self.columns, self.columns, self.schema_sha)
        with self.assertRaisesRegex(ValueError, "order"):
            check_raw_schema(self.raw, self.columns[::-1], self.columns, self.schema_sha)
        excluded = list(self.columns)
        excluded[1] = "SEX"
        excluded = tuple(excluded)
        with self.assertRaisesRegex(ValueError, "excluded"):
            check_raw_schema(self.raw, excluded, excluded,
                             hashlib.sha256("\n".join(excluded).encode()).hexdigest())
        with self.assertRaisesRegex(ValueError, "float32"):
            check_raw_schema(self.raw.astype(np.float64), self.columns,
                             self.columns, self.schema_sha)
        with tempfile.TemporaryDirectory() as scratch:
            checkpoint = Path(scratch) / "source.pt"
            checkpoint.write_bytes(b"invented checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            check_source_checkpoint(checkpoint, digest)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                check_source_checkpoint(checkpoint, "0" * 64)

    def test_d_adapter_code_only_service_and_source_pin(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3)
            model = HardCodeModel(16, (2, 6))
        scaler = FixedScaler(np.zeros((2, 16), np.float32),
                             np.ones((2, 16), np.float32))
        adapter = HardCodeRelease(model, scaler, "synthetic-v1", self.source_sha)
        encoder = _SyntheticFrozenEncoder(self.source_sha, self.columns)
        for purpose in (0, 1):
            first = adapter.recipient_from_raw(purpose, self.raw, self.columns, encoder)
            second = adapter.recipient_from_raw(purpose, self.raw, self.columns, encoder)
            check_d_public_release(first, purpose, self.source_sha,
                                   adapter.probabilities_from_code)
            for field in ("code", "probabilities", "decision"):
                np.testing.assert_array_equal(first[field], second[field])
            self.assertEqual(first["metadata"], second["metadata"])
            changed = dict(first, latent=self.raw[:, :16])
            with self.assertRaisesRegex(ValueError, "hidden"):
                check_d_public_release(changed, purpose, self.source_sha,
                                       adapter.probabilities_from_code)
            changed = dict(first, probabilities=np.roll(first["probabilities"], 1, axis=1))
            with self.assertRaisesRegex(ValueError, "code"):
                check_d_public_release(changed, purpose, self.source_sha,
                                       adapter.probabilities_from_code)
        encoder.model_sha256 = "b" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            adapter.recipient_from_raw(0, self.raw, self.columns, encoder)

    def test_a_global_bank_wire_and_cached_joint_draw(self) -> None:
        bank = {"income:0": ([0.8, 0.2], 0), "income:1": ([0.3, 0.7], 1)}
        metadata = {"schema": "synthetic-global-bank-v1", "version": "synthetic-v1"}
        first = {"code_id": "income:0", "probability": [0.8, 0.2],
                 "decision": 0, "metadata": metadata}
        second = {"code_id": "income:1", "probability": [0.3, 0.7],
                  "decision": 1, "metadata": metadata}
        check_a_public_release(first, bank, metadata)
        check_a_public_release(second, bank, metadata)
        check_cached_pair((first, second), (dict(first), dict(second)))
        for hidden in ("fine_token", "action_index", "gate_probabilities",
                       "random_seed", "teacher_score"):
            with self.assertRaisesRegex(ValueError, "hidden"):
                check_a_public_release(dict(first, **{hidden: 1}), bank, metadata)
        with self.assertRaisesRegex(ValueError, "public bank"):
            check_a_public_release(dict(first, code_id="input-local:0"), bank, metadata)
        with self.assertRaisesRegex(ValueError, "public bank"):
            check_a_public_release(dict(first, probability=[0.79, 0.21]), bank, metadata)
        with self.assertRaisesRegex(ValueError, "cached"):
            check_cached_pair((first, second), (first, first))


if __name__ == "__main__":
    unittest.main()
