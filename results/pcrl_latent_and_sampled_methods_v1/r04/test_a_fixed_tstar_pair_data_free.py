"""Data-free arithmetic and wire serialization checks for the R04 T* verifier."""

from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from .replay_a_fixed_tstar_pair import same_bytes, wire_digest


class FixedTStarVerifierFixture(unittest.TestCase):
    def test_finite_wire_exact_json_line(self) -> None:
        q = np.array([[0.75, 0.25]], dtype=np.float64)
        d = np.array([0], dtype=np.int64)
        code = np.array([2], dtype=np.int64)
        binding = "a" * 64
        bank_hash = "b" * 64
        literal = {"schema": "SCV1-reference-score-release-v1", "recipient": 1,
                   "probability": [0.75, 0.25], "decision": 0,
                   "release_id": "fixture", "code_id": 2,
                   "public_metadata": {"model_id": "fixture",
                                       "source_binding_sha256": binding,
                                       "bank_sha256": bank_hash,
                                       "code_length_bits": 2}}
        expected = hashlib.sha256((json.dumps(literal, sort_keys=True,
                                               separators=(",", ":")) + "\n").encode()).hexdigest()
        self.assertEqual(wire_digest(q, d, code, recipient=1,
                                     release_id="fixture", source_binding=binding,
                                     bank_hash=bank_hash, bank_size=4), expected)

    def test_output_parity_is_byte_exact(self) -> None:
        values = np.array([[0.1, 0.9]], dtype=np.float64)
        self.assertTrue(same_bytes(values, values.copy()))
        self.assertFalse(same_bytes(values, values.astype(np.float32)))

    def test_complete_pair_pack_keeps_high_code_bits(self) -> None:
        left = np.array([0, 64], dtype=np.int64)
        right = np.array([1, 1], dtype=np.int64)
        pair = left * 321 + right
        self.assertEqual(len(np.unique(pair)), 2)


if __name__ == "__main__":
    unittest.main()
