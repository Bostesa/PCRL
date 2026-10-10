"""Data-free write-once one-realization custody and public-wire replay."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

from release import CachedJointRelease
from source_bridge import sha_array
from test_release import toy_policy
from materialize import materialize_role, reconstruct_wire_from_codes


class MaterializeTest(unittest.TestCase):
    def test_one_realization_saves_only_code_ids_and_replayable_wire_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = CachedJointRelease(
                toy_policy(), cache_path=root / "cache.sqlite3",
                release_id="arm-s0", source_seed=0, role="INNER_SELECTION",
                realization=2, source_binding_sha256="a" * 64,
                private_root=b"x" * 32, offline=True)
            row = np.array([2, 4, 6], np.int64)
            unit = np.array([1, 3, 3], np.int64)
            score1 = np.array([[.9, .1], [.8, .2], [.8, .2]], np.float64)
            score2 = np.array([[.8, .2], [.7, .3], [.7, .3]], np.float64)
            cell = np.zeros(3, np.int64)
            expected = sha_array(np.column_stack((row, unit)))
            receipt = materialize_role(
                service, row_id=row, unit=unit, score1=score1, score2=score2,
                cell1=cell, cell2=cell, output_dir=root,
                expected_row_unit_sha256=expected,
                source_identity={"tstar_bank_admission_receipt_sha256": "b" * 64,
                                 "tstar_role_tokens_file_sha256": "d" * 64,
                                 "reference_file_sha256": "c" * 64})
            self.assertEqual(receipt["schema"], "SCV1-private-realization-v1")
            self.assertEqual(receipt["row_unit_pair_sha256"], expected)
            self.assertEqual(receipt["role"], "INNER_SELECTION")
            self.assertEqual(receipt["private_stream_root_sha256"],
                             hashlib.sha256(b"x" * 32).hexdigest())
            with ZipFile(root / "release_codes.npz") as archive:
                self.assertEqual(set(archive.namelist()),
                                 {"row_id.npy", "unit.npy", "code_r1.npy",
                                  "code_r2.npy"})
            with np.load(root / "release_codes.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["code_r1"][1], saved["code_r1"][2])
                self.assertEqual(saved["code_r2"][1], saved["code_r2"][2])
                for recipient in (1, 2):
                    wires = reconstruct_wire_from_codes(service,
                        saved[f"code_r{recipient}"], recipient=recipient)
                    content = b"".join(json.dumps(wire, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode() + b"\n"
                        for wire in wires)
                    self.assertEqual(hashlib.sha256(content).hexdigest(),
                                     receipt["wire_sha256_by_recipient"][f"r{recipient}"])
            with self.assertRaises(FileExistsError):
                materialize_role(service, row_id=row, unit=unit, score1=score1,
                                 score2=score2, cell1=cell, cell2=cell,
                                 output_dir=root, expected_row_unit_sha256=expected,
                                 source_identity={"tstar_bank_admission_receipt_sha256": "b" * 64,
                                                  "tstar_role_tokens_file_sha256": "d" * 64,
                                                  "reference_file_sha256": "c" * 64})


if __name__ == "__main__":
    unittest.main()
