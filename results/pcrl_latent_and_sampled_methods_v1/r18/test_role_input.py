"""Synthetic custody tests; no real role array is opened by this suite."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r18.role_input import (
    _sha_array, load_one_admitted_feature_row,
)


class RoleInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "package"
        self.root.mkdir()
        self.columns = tuple(f"allowed_{i:02d}" for i in range(83))
        self.columns_sha = hashlib.sha256("\n".join(self.columns).encode()).hexdigest()
        self.x = np.arange(3 * 83, dtype=np.float32).reshape(3, 83) / 100
        self.ids = np.asarray([11, 12, 13], dtype=np.int64)
        self.units = np.asarray([4, 4, 5], dtype=np.int64)
        self.export = self.root / ".private/role_export_v1/INNER_SELECTION/features.npz"
        self.export.parent.mkdir(parents=True)
        np.savez(self.export, X=self.x, feature_names=np.asarray(self.columns),
                 row_id=self.ids, unit=self.units)
        self.manifest = self.root / "m1/DATA_ACCESS_MANIFEST.json"
        self.manifest.parent.mkdir()
        self.record = {
            "schema": "pcrl-utility-repair-role-admission-v1",
            "source_schema_sha256": self.columns_sha,
            "role_file_relative_path": {
                "INNER_SELECTION:features": ".private/role_export_v1/INNER_SELECTION/features.npz"},
            "role_file_sha256": {
                "INNER_SELECTION:features": hashlib.sha256(self.export.read_bytes()).hexdigest()},
            "roles": {"INNER_SELECTION": {
                "rows": 3, "groups": 2,
                "row_id_sha256": _sha_array(np.sort(self.ids)),
                "group_id_sha256": _sha_array(np.unique(self.units)),
                "row_unit_pair_sha256": _sha_array(np.column_stack((self.ids, self.units)))}}}
        self._write_manifest()

    def _write_manifest(self) -> str:
        self.manifest.write_text(json.dumps(self.record, sort_keys=True))
        return hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def test_one_admitted_role_row_and_no_label_member(self) -> None:
        sha = self._write_manifest()
        row, columns, row_id, unit = load_one_admitted_feature_row(
            self.root, self.manifest, sha, "INNER_SELECTION", 1, self.columns_sha)
        np.testing.assert_array_equal(row, self.x[1:2])
        self.assertEqual(columns, self.columns)
        self.assertEqual((row_id, unit), (12, 4))
        self.assertEqual(row.shape, (1, 83))

    def test_refuses_changed_export_or_unexpected_member(self) -> None:
        sha = self._write_manifest()
        np.savez(self.export, X=self.x, feature_names=np.asarray(self.columns),
                 row_id=self.ids, unit=self.units, sex=np.zeros(3, np.int8))
        with self.assertRaisesRegex(ValueError, "hash"):
            load_one_admitted_feature_row(self.root, self.manifest, sha,
                                          "INNER_SELECTION", 0, self.columns_sha)
        self.record["role_file_sha256"]["INNER_SELECTION:features"] = hashlib.sha256(
            self.export.read_bytes()).hexdigest()
        sha = self._write_manifest()
        with self.assertRaisesRegex(ValueError, "members"):
            load_one_admitted_feature_row(self.root, self.manifest, sha,
                                          "INNER_SELECTION", 0, self.columns_sha)

    def test_refuses_barred_role_wrong_manifest_and_changed_identity(self) -> None:
        sha = self._write_manifest()
        with self.assertRaisesRegex(ValueError, "not admitted"):
            load_one_admitted_feature_row(self.root, self.manifest, sha,
                                          "OSF_DEVELOPMENT_ASSESSMENT", 0, self.columns_sha)
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            load_one_admitted_feature_row(self.root, self.manifest, "0" * 64,
                                          "INNER_SELECTION", 0, self.columns_sha)
        self.record["roles"]["INNER_SELECTION"]["row_unit_pair_sha256"] = "0" * 64
        sha = self._write_manifest()
        with self.assertRaisesRegex(ValueError, "identity"):
            load_one_admitted_feature_row(self.root, self.manifest, sha,
                                          "INNER_SELECTION", 0, self.columns_sha)


if __name__ == "__main__":
    unittest.main()
