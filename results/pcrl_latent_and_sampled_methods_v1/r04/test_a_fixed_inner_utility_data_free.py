"""Data-free checks for canonical diagnostics and actual-draw task arithmetic."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from .audit_diagnostic_q_parity import _canonical, _distance
from .independent_replay import task_metrics, task_point_gate
from .independent_a_source import sha_file
from .replay_a_fixed_inner_utility import load_role


class FixedInnerUtilityFixture(unittest.TestCase):
    def test_canonical_old_id_permutation_and_class_medoid(self) -> None:
        old = np.array([[0.8, 0.2], [0.9, 0.1], [0.1, 0.9]], np.float64)
        bank, decision, mapping = _canonical(
            old, np.arange(3, dtype=np.int64), np.array([0, 0, 1], np.int64))
        self.assertEqual(len(np.unique(mapping)), 3)
        self.assertTrue(np.array_equal(bank[mapping], old))
        ids = np.flatnonzero(decision == 0)
        medoid = min((sum(_distance(bank[i], bank[j]) for j in ids), int(i))
                     for i in ids)[1]
        self.assertIn(medoid, ids)
        self.assertEqual(int(decision[medoid]), 0)

    def test_actual_probability_vector_task_gate(self) -> None:
        y = np.array([0, 1], np.int64)
        candidate = task_metrics(np.array([[0.9, 0.1], [0.6, 0.4]]), y)
        reference = task_metrics(np.array([[0.9, 0.1], [0.6, 0.4]]), y)
        self.assertAlmostEqual(candidate["brier"], 0.37)
        gate = task_point_gate({"income": candidate, "occupation": candidate},
                               {"income": {"U0": reference, "Ucal": reference},
                                "occupation": {"U0": reference, "Ucal": reference}})
        self.assertTrue(gate["point_pass"])

    def test_label_archive_rejects_extra_member_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "m1/DATA_ACCESS_MANIFEST.json"
            manifest_path.parent.mkdir()
            relative = ".private/role_export_v1/INNER_SELECTION/income.npz"
            source = root / relative
            source.parent.mkdir(parents=True)
            np.savez(source, row_id=np.array([1], np.int64),
                     unit=np.array([2], np.int64), income=np.array([0], np.int64),
                     forbidden_sex=np.array([1], np.int64))
            manifest = {"role_file_relative_path": {"INNER_SELECTION:income": relative},
                        "role_file_sha256": {"INNER_SELECTION:income": sha_file(source)}}
            with self.assertRaisesRegex(ValueError, "role schema"):
                load_role(manifest_path, manifest, "income")


if __name__ == "__main__":
    unittest.main()
