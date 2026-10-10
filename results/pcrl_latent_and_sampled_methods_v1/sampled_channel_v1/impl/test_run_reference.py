"""Data-free pre-role gate and T* token-to-wire tests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import run_reference as target


class ReferenceWorkerTest(unittest.TestCase):
    def test_no_role_reader_when_queue_gate_refuses(self):
        with (patch.object(target, "assert_active_from_env",
                           side_effect=PermissionError("R03 denied")),
              patch.object(target, "RoleOnlyStore",
                           side_effect=AssertionError("role opened"))):
            with self.assertRaisesRegex(PermissionError, "R03 denied"):
                target.run(unit_id="synthetic", lock_path="missing",
                           amendment_path="missing", amendment_sha256="a" * 64,
                           reference_id="U0", seed=0, role="ATTACK_FIT_NEW",
                           repair_manifest_path="missing", source_audit_path="missing",
                           historical_audit_path="missing", portfolio_package="missing",
                           historical_cache_root="missing", output_dir="missing")

    def test_private_output_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with self.assertRaises(PermissionError):
                target._private_output(root / "public")
            private = root / ".private"
            private.mkdir(mode=0o700)
            self.assertEqual(target._private_output(private / "fresh"),
                             (private / "fresh").resolve())

    def test_tstar_token_maps_to_actual_bank_row(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "SOURCE_AUDIT.json").write_text(json.dumps({"schema": "synthetic"}))
            np.savez(root / "references.npz", ignored=np.array([1]))
            row = np.array([11, 12], np.int64)
            unit = np.array([21, 22], np.int64)
            refs = {"row_id": row, "unit": unit}
            cells = {"row_id": row, "unit": unit,
                     "token_1": np.array([1, 0], np.int64),
                     "token_2": np.array([0, 1], np.int64)}
            p1 = np.array([[.8, .2], [.2, .8]], np.float64)
            p2 = np.array([[.8, .04, .04, .04, .04, .04],
                           [.04, .8, .04, .04, .04, .04]], np.float64)

            class Store:
                manifest = {"role_file_sha256": {"ATTACK_FIT_NEW:references_s0": "a" * 64},
                            "roles": {"ATTACK_FIT_NEW": {"row_unit_pair_sha256": "b" * 64}}}

                def read(self, role, field):
                    self._ = (role, field)
                    return refs

            class Source:
                source_binding_sha256 = "c" * 64
                receipt = {"release_id": "U|FINE-TASK|i8o64|H-GLOBAL-TEMP"}

                def __init__(self, source_root):
                    self.root = source_root

                def read_role_cells(self, role):
                    return cells

                def read_bank(self, recipient):
                    p = p1 if recipient == 1 else p2
                    return {"probability": p,
                            "token_class": np.argmax(p, axis=1).astype(np.int64)}

            (root / "BANK_ADMISSION_RECEIPT.json").write_bytes(b"bank")
            (root / "tokens_ATTACK_FIT_NEW.npz").write_bytes(b"tokens")
            with (patch.object(target, "RoleOnlyStore", return_value=Store()),
                  patch.object(target, "TStarSourceStore", return_value=Source(root))):
                result = target.collect_reference_role(
                    reference_id="T_STAR", seed=0, role="ATTACK_FIT_NEW",
                    repair_manifest_path=root / "repair.json",
                    source_audit_path=root / "SOURCE_AUDIT.json",
                    historical_audit_path=root / "historical.json",
                    portfolio_package=root, historical_cache_root=root)
            self.assertEqual(result["interface_kind"], "finite")
            self.assertEqual(result["source_release_id"],
                             "U|FINE-TASK|i8o64|H-GLOBAL-TEMP")
            self.assertTrue(np.array_equal(result["code1"], cells["token_1"]))
            self.assertTrue(np.array_equal(result["prob1"], p1[cells["token_1"]]))
            self.assertTrue(np.array_equal(result["decision2"],
                                           np.argmax(p2, axis=1)[cells["token_2"]]))


if __name__ == "__main__":
    unittest.main()
