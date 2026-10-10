import tempfile
import unittest
from pathlib import Path

import numpy as np

from results.pcrl_code_utility_witness_v1.r5.snapshot_restore_worker import (
    _bank_decode, _wire,
)


class SnapshotRestoreWorkerTest(unittest.TestCase):
    def test_public_bank_id_decodes_without_bank_implementation(self):
        code = np.zeros((2, 10), dtype=np.uint8)
        code[1, 1] = 1
        table = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])
        prob, decision = _bank_decode(code, table)
        np.testing.assert_array_equal(prob, table[[0, 2]])
        np.testing.assert_array_equal(decision, [0, 1])

    def test_wire_refuses_extra_latent_export(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "bad.npz"
            np.savez(path, code=np.zeros((32, 10), dtype=np.uint8),
                     prob=np.tile([0.8, 0.2], (32, 1)),
                     decision=np.zeros(32, dtype=np.int64),
                     latent=np.zeros((32, 16), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "unexpected member"):
                _wire(path, 10)


if __name__ == "__main__":
    unittest.main()
