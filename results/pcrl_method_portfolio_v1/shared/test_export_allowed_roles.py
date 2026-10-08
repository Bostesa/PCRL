"""Sentinel-only exporter tests: no real source member values are opened."""

from __future__ import annotations

import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import export_allowed_roles as E
from safe_data_guard import sha_file


class ExporterSentinelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_forbidden_values_never_enter_role_outputs_or_logs(self):
        names = np.array([f"safe_feature_{i}" for i in range(83)])
        values = np.zeros((5, 83), dtype=np.float32)
        values[4, :] = 777777.0  # synthetic forbidden assessment row
        source = {
            "feature_names": names, "X": values,
            "y_income": np.array([0, 1, 0, 1, 888888], dtype=np.int64),
            "y_occupation_group": np.array([0, 1, 2, 3, 888888], dtype=np.int64),
            "sex": np.array([0, 1, 0, 1, 888888], dtype=np.int64),
        }
        positions = {role: np.array([i], dtype=np.int64) for i, role in enumerate(E.EXPORT_ROLES)}
        row_id = np.arange(10, 60, 10, dtype=np.int64)
        unit = np.arange(1, 6, dtype=np.int64)
        assigned = np.array([*E.EXPORT_ROLES, "OSF_DEVELOPMENT_ASSESSMENT"])
        output = io.StringIO()
        error = io.StringIO()
        h = hashlib.sha256("\n".join(names.tolist()).encode()).hexdigest()
        with patch.object(E, "FEATURE_NAMES_SHA", h), \
             patch("smf.data.refit_numeric", side_effect=lambda x, n, m: (x.astype(np.float32), {})), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            receipts = E._export_features_labels(source, self.root, positions, row_id, unit, assigned)
        self.assertEqual(len(receipts), 13)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(error.getvalue(), "")
        for path in self.root.rglob("*.npz"):
            with np.load(path, allow_pickle=False) as archive:
                self.assertFalse(any(np.any(archive[name] == 777777) for name in archive.files
                                     if np.issubdtype(archive[name].dtype, np.number)))
                self.assertFalse(any(np.any(archive[name] == 888888) for name in archive.files
                                     if np.issubdtype(archive[name].dtype, np.number)))

    def test_archive_bytes_deterministic(self):
        arrays = {"row_id": np.array([10, 20], dtype=np.int64), "unit": np.array([1, 2], dtype=np.int64),
                  "income": np.array([0, 1], dtype=np.int64)}
        a, b = self.root / "a.npz", self.root / "b.npz"
        E._write_archive(a, arrays)
        E._write_archive(b, arrays)
        self.assertEqual(sha_file(a), sha_file(b))

    def test_public_output_path_refused_before_source_access(self):
        with patch.object(E, "sha_file", side_effect=AssertionError("source accessed")):
            with self.assertRaises(E.ExportRefused):
                E.export(self.root / "public_output")

    def test_failure_log_scrubs_exception_value(self):
        output, error = io.StringIO(), io.StringIO()
        with patch.object(E, "export", side_effect=ValueError("888888_SECRET_SENTINEL")), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = E.main(["--output", str(self.root / "unused")])
        self.assertEqual(code, 1)
        self.assertNotIn("888888_SECRET_SENTINEL", output.getvalue() + error.getvalue())


if __name__ == "__main__":
    unittest.main()
