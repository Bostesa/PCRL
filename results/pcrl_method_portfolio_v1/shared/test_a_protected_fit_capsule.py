"""Synthetic member and identity tests for the pure protected FIT capsule."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from a_protected_fit_capsule import (
    MEMBERS, SCHEMA, _sha_arr, _sha_file, load_protected_capsule,
)


def _fixture(root: Path) -> tuple[Path, Path, str]:
    row_id = np.array([10, 11, 12], dtype=np.int64)
    unit = np.array([7, 8, 8], dtype=np.int64)
    bank_1 = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    bank_2 = np.eye(6, dtype=np.float64)
    u0_1 = bank_1[[0, 1, 0]]
    u0_2 = bank_2[[0, 1, 2]]
    arrays = {
        "row_id": row_id, "unit": unit,
        "cell_1": np.array([0, 1, 0], dtype=np.int64),
        "cell_2": np.array([0, 1, 2], dtype=np.int64),
        "income": np.array([0, 1, 0], dtype=np.int64),
        "occupation": np.array([0, 1, 2], dtype=np.int64),
        "sex": np.array([1, 0, 1], dtype=np.int64),
        "U0_1": u0_1, "U0_2": u0_2,
        "Ucal_1": u0_1.copy(), "Ucal_2": u0_2.copy(),
        "bank_1": bank_1, "bank_2": bank_2,
        "input_class_1": np.array([0, 1], dtype=np.int64),
        "input_class_2": np.arange(6, dtype=np.int64),
        "fine_to_hcal_token_1": np.array([0, 1], dtype=np.int64),
        "fine_to_hcal_token_2": np.arange(6, dtype=np.int64),
        "token_class_1": np.array([0, 1], dtype=np.int64),
        "token_class_2": np.arange(6, dtype=np.int64),
        "source_binding_sha256": np.array("a" * 64),
    }
    assert set(arrays) == MEMBERS
    npz = root / "protected_fit.npz"
    np.savez_compressed(npz, **arrays)
    pair = np.column_stack((row_id, unit)).astype(np.int64)
    receipt = {
        "schema": SCHEMA, "status": "FROZEN", "model_seed": 0,
        "role": "OSF_DEFENSE_FIT", "rows": len(row_id), "groups": 2,
        "members": sorted(MEMBERS), "npz_sha256": _sha_file(npz),
        "array_sha256": {name: _sha_arr(value) for name, value in arrays.items()},
        "row_unit_pair_sha256": _sha_arr(pair),
        "source_binding_sha256": "a" * 64,
        "bank_admission_receipt_sha256": "b" * 64,
        "s4_bank_report_sha256": "c" * 64,
        "fit_sex_role_file_sha256": "d" * 64,
        "task_capsule_receipt_sha256_by_recipient": {"r1": "e" * 64,
                                                       "r2": "f" * 64},
        "s4_task_capsule_report_sha256_by_recipient": {"r1": "4" * 64,
                                                         "r2": "5" * 64},
        "data_access_manifest_sha256": "1" * 64,
        "campaign_lock_sha256": "2" * 64,
        "producer_sha256": "3" * 64,
    }
    rp = root / "RECEIPT.json"
    rp.write_text(json.dumps(receipt, sort_keys=True))
    return npz, rp, _sha_file(rp)


class ProtectedCapsuleTests(unittest.TestCase):
    def test_exact_twenty_member_capsule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            npz, receipt, sha = _fixture(Path(directory))
            result = load_protected_capsule(npz, receipt,
                                            expected_receipt_sha256=sha,
                                            model_seed=0)
            self.assertEqual(set(result), MEMBERS | {"receipt"})
            self.assertEqual(result["sex"].tolist(), [1, 0, 1])

    def test_mismatched_receipt_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            npz, receipt, _ = _fixture(Path(directory))
            with self.assertRaises(PermissionError):
                load_protected_capsule(npz, receipt,
                                       expected_receipt_sha256="0" * 64,
                                       model_seed=0)

    def test_extra_member_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            npz, receipt, _ = _fixture(Path(directory))
            with np.load(npz, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            arrays["forbidden_partner"] = np.array([1], dtype=np.int64)
            np.savez_compressed(npz, **arrays)
            record = json.loads(receipt.read_text())
            record["npz_sha256"] = _sha_file(npz)
            receipt.write_text(json.dumps(record, sort_keys=True))
            with self.assertRaises(PermissionError):
                load_protected_capsule(npz, receipt,
                                       expected_receipt_sha256=_sha_file(receipt),
                                       model_seed=0)


if __name__ == "__main__":
    unittest.main()
