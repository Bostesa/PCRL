"""Synthetic-only checks. No PCRL private data file is opened."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from safe_data_guard import AccessRefused, SafeRoleStore, sha_arr, sha_file


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = np.array([10, 20, 30], dtype=np.int64)
        self.groups = np.array([1, 2, 3], dtype=np.int64)
        self.names = np.array(["age", "workclass=A", "education=B"])
        self.x = np.zeros((3, 3), dtype=np.float64)
        self.manifest = {
            "schema": "pcrl-method-portfolio-data-access-v1",
            "source": {"columns": 3,
                       "feature_names_sha256_newline_join": hashlib.sha256("\n".join(self.names).encode()).hexdigest()},
            "roles": {"OSF_DEFENSE_FIT": {"rows": 3, "groups": 3,
                       "row_id_sha256": sha_arr(self.ids), "group_id_sha256": sha_arr(self.groups)}},
            "label_allowlist": {"defense_fit": {"OSF_DEFENSE_FIT": ["income", "sex"]}},
            "real_file_specs": {}
        }

    def store(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        return SafeRoleStore(path)

    def admit(self, field, **arrays):
        path = self.root / f"role_only_{field}.npz"
        np.savez(path, row_id=self.ids, unit=self.groups, **arrays)
        self.manifest["real_file_specs"][f"OSF_DEFENSE_FIT:{field}"] = {
            "role": "OSF_DEFENSE_FIT", "field": field, "path": str(path),
            "sha256": sha_file(path), "members": ["row_id", "unit", *arrays],
            "row_unit_pair_sha256": sha_arr(np.column_stack((self.ids, self.groups)).astype(np.int64))
        }
        return path

    def test_no_registered_file_refuses_before_file_read(self):
        with patch("safe_data_guard.sha_file", side_effect=AssertionError("file touched")):
            with self.assertRaises(AccessRefused):
                self.store().read("OSF_DEFENSE_FIT", "features")

    def test_forbidden_role_or_label_refuses_before_file_read(self):
        self.admit("income", income=np.array([0, 1, 0], dtype=np.int64))
        with patch("safe_data_guard.sha_file", side_effect=AssertionError("file touched")):
            with self.assertRaises(AccessRefused):
                self.store().read("OSF_DEVELOPMENT_ASSESSMENT", "income", "defense_fit")
            with self.assertRaises(AccessRefused):
                self.store().read("OSF_DEFENSE_FIT", "income", "attack")

    def test_extra_sentinel_member_rejected_before_numpy_member_read(self):
        self.admit("features", X=self.x, feature_names=self.names,
                   assessment_sex=np.array([9, 9, 9], dtype=np.int64))
        # Leave the file hash correct, but propose only the nominal members.
        self.manifest["real_file_specs"]["OSF_DEFENSE_FIT:features"]["members"] = [
            "row_id", "unit", "X", "feature_names"]
        with patch("safe_data_guard.np.load", side_effect=AssertionError("member read")):
            with self.assertRaises(AccessRefused):
                self.store().read("OSF_DEFENSE_FIT", "features")

    def test_role_only_allowed_arrays_and_label(self):
        self.admit("features", X=self.x, feature_names=self.names)
        self.admit("income", income=np.array([0, 1, 0], dtype=np.int64))
        store = self.store()
        features = store.read("OSF_DEFENSE_FIT", "features")
        label = store.read("OSF_DEFENSE_FIT", "income", "defense_fit")
        self.assertEqual(set(features), {"row_id", "unit", "X", "feature_names"})
        self.assertEqual(set(label), {"row_id", "unit", "income"})
        np.testing.assert_array_equal(features["row_id"], label["row_id"])

    def test_row_to_group_swap_rejected_despite_separate_hashes(self):
        path = self.admit("income", income=np.array([0, 1, 0], dtype=np.int64))
        np.savez(path, row_id=self.ids, unit=np.array([2, 1, 3], dtype=np.int64),
                 income=np.array([0, 1, 0], dtype=np.int64))
        self.manifest["real_file_specs"]["OSF_DEFENSE_FIT:income"]["sha256"] = sha_file(path)
        with self.assertRaisesRegex(AccessRefused, "pairing"):
            self.store().read("OSF_DEFENSE_FIT", "income", "defense_fit")

    def test_known_all_role_filename_refused_before_hash(self):
        path = self.admit("income", income=np.array([0, 1, 0], dtype=np.int64))
        all_role_name = self.root / "adult_jcv.npz"
        path.rename(all_role_name)
        self.manifest["real_file_specs"]["OSF_DEFENSE_FIT:income"]["path"] = str(all_role_name)
        with patch("safe_data_guard.sha_file", side_effect=AssertionError("all-role file touched")):
            with self.assertRaises(AccessRefused):
                self.store().read("OSF_DEFENSE_FIT", "income", "defense_fit")

    def test_inherited_loader_transiently_reads_forbidden_sentinel(self):
        # Independently demonstrates why masking on return is insufficient.
        from osf import data as old
        reads = []

        class Label:
            def __init__(self, name): self.name = name
            def __getitem__(self, positions):
                reads.append((self.name, tuple(np.asarray(positions).tolist())))
                return np.array([0, 7], dtype=np.int64)[positions]

        class Archive:
            def __getitem__(self, key):
                if key in old.LABEL_KEYS: return Label(key)
                if key == "role": return np.array(["defense_train", "assessment"])
                if key == "unit": return np.array([10, 11], dtype=np.int64)
                if key == "row_id": return np.array([0, 1], dtype=np.int64)
                if key == "X": return np.zeros((2, 83), dtype=np.float64)
                if key == "feature_names": return np.array([f"x{i}" for i in range(83)])
                raise KeyError(key)

        role = np.array(["OSF_DEFENSE_FIT", "OSF_DEVELOPMENT_ASSESSMENT"])
        sub = np.array(["", ""])
        pool = np.array(["", "ORIG_ASSESSMENT"])
        with patch.object(old.np, "load", return_value=Archive()), \
             patch.object(old, "assign_roles", return_value=(role, sub, pool, {})), \
             patch.object(old, "check_partition"), \
             patch.object(old.SD, "refit_numeric", side_effect=lambda X, names, fit: (X, {})):
            returned = old.load(verify=False, unseal=False)
        self.assertEqual({name for name, _ in reads}, set(old.LABEL_KEYS))
        self.assertTrue(all(1 in positions for _, positions in reads))
        self.assertTrue(all(returned[name][1] == -1 for name in old.LABEL_KEYS))


if __name__ == "__main__":
    unittest.main()
