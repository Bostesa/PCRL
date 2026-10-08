"""Synthetic-only recipient capsule and no-broad-loader boundary checks."""

from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import a_task_only_capsule as capsule


def _fixture(root: Path, *, extra: bool = False) -> tuple[Path, Path, str]:
    bank = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    row = np.array([10, 11, 12], dtype=np.int64)
    unit = np.array([30, 31, 31], dtype=np.int64)
    arrays = {"row_id": row, "unit": unit,
              "cell": np.array([0, 1, 0], dtype=np.int64),
              "task_label": np.array([0, 1, 1], dtype=np.int64),
              "U0": np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]], dtype=np.float64),
              "Ucal": np.array([[0.65, 0.35], [0.25, 0.75], [0.55, 0.45]], dtype=np.float64),
              "bank": bank, "input_class": np.array([0, 1], dtype=np.int64),
              "fine_to_hcal_token": np.array([0, 1], dtype=np.int64),
              "token_class": np.array([0, 1], dtype=np.int64),
              "source_binding_sha256": np.array("a" * 64)}
    if extra:
        arrays["sex"] = np.array([0, 1, 0], dtype=np.int64)
    path = root / "task_only.npz"
    np.savez_compressed(path, **arrays)
    receipt = {"schema": capsule.SCHEMA, "status": "FROZEN", "recipient": 1,
               "model_seed": 0, "task": "income", "role": "OSF_DEFENSE_FIT",
               "npz_sha256": capsule._sha_file(path),
               "members": sorted(capsule.MEMBERS), "rows": len(row), "groups": 2,
               "row_unit_pair_sha256": capsule._sha_arr(np.column_stack((row, unit))),
               "source_binding_sha256": "a" * 64,
               "array_sha256": {key: capsule._sha_arr(value) for key, value in arrays.items()
                                if key in capsule.MEMBERS}}
    receipt_path = root / "RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return path, receipt_path, capsule._sha_file(receipt_path)


class TaskCapsuleTests(unittest.TestCase):
    def test_exact_recipient_only_schema_and_reference_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, receipt, receipt_sha = _fixture(Path(directory))
            result = capsule.load_task_capsule(path, receipt,
                                               expected_receipt_sha256=receipt_sha,
                                               recipient=1, model_seed=0)
            self.assertEqual(result["cell"].tolist(), [0, 1, 0])
            self.assertFalse(any(key in result for key in ("sex", "occupation", "U0_2")))

    def test_forbidden_sensitive_member_is_rejected_before_member_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, receipt, receipt_sha = _fixture(Path(directory), extra=True)
            with self.assertRaisesRegex(PermissionError, "missing or forbidden members"):
                capsule.load_task_capsule(path, receipt,
                                          expected_receipt_sha256=receipt_sha,
                                          recipient=1, model_seed=0)

    def test_probability_sum_tolerance_has_no_relative_slack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, receipt_path, _ = _fixture(Path(directory))
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
            arrays["U0"][0, 0] += 2e-6
            np.savez_compressed(path, **arrays)
            receipt = json.loads(receipt_path.read_text())
            receipt["npz_sha256"] = capsule._sha_file(path)
            receipt["array_sha256"]["U0"] = capsule._sha_arr(arrays["U0"])
            receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "probability simplex"):
                capsule.load_task_capsule(path, receipt_path,
                                          expected_receipt_sha256=capsule._sha_file(receipt_path),
                                          recipient=1, model_seed=0)

    def test_receipt_task_identity_cannot_be_swapped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, receipt_path, _ = _fixture(Path(directory))
            receipt = json.loads(receipt_path.read_text())
            receipt["task"] = "occupation"
            receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
            with self.assertRaisesRegex(PermissionError, "identity or member contract"):
                capsule.load_task_capsule(path, receipt_path,
                                          expected_receipt_sha256=capsule._sha_file(receipt_path),
                                          recipient=1, model_seed=0)

    def test_pure_loader_import_refuses_broad_data_module(self) -> None:
        source = Path(capsule.__file__)
        spec = importlib.util.spec_from_file_location("isolated_a_capsule", source)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name in ("safe_data_guard", "hcal", "qpc"):
                raise AssertionError("task-only worker imported a broad source loader")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            spec.loader.exec_module(module)
        self.assertEqual(module.MEMBERS, capsule.MEMBERS)


if __name__ == "__main__":
    unittest.main()
