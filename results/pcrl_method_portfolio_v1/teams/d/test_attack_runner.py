"""Tiny synthetic custody checks for Team D's private attack runner."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import attack_runner


class DAttackRunnerCustodyTests(unittest.TestCase):
    def test_unregistered_attack_unit_refused_before_role_store_load(self):
        args = types.SimpleNamespace(
            campaign_lock="lock", lock_sha256="a" * 64,
            local_seal_receipt="seal", data_manifest="manifest",
            shared_root="shared", arm_id="D_VQ_JOINT", seed=0,
            attacker_seed=0, source_run_id="source_s0",
            attack_run_id="unregistered_s0")
        lock = {"arms": {"D_VQ_JOINT": {"family": "D", "track": "feature"}}}
        with patch.object(attack_runner.base, "_check_seal",
                          return_value=(lock, {}, "b" * 64)), \
             patch.object(attack_runner.base, "_load_shared",
                          side_effect=AssertionError("role store must stay unopened")):
            with self.assertRaises(PermissionError):
                attack_runner._admit(args)

    def _package(self, root: Path):
        role = "ATTACK_FIT_NEW"
        directory = root / role
        directory.mkdir(mode=0o700)
        rows = np.array([1, 2], dtype=np.int64)
        units = rows.copy()
        arrays = {"row_id": rows, "unit": units}
        for purpose in (0, 1):
            arrays[f"r{purpose}_code_ids"] = np.full((2, 1), purpose, dtype=np.int64)
            arrays[f"r{purpose}_decoded"] = np.full((2, 16), purpose, dtype=np.float32)
            arrays[f"r{purpose}_probabilities"] = (np.tile([0.7, 0.3], (2, 1)) if purpose == 0
                                                     else np.tile([0.5, 0.2, 0.1, 0.1, 0.06, 0.04], (2, 1)))
            arrays[f"r{purpose}_decision"] = np.zeros(2, dtype=np.int64)
        wire = directory / "wire.npz"
        wire_sha = attack_runner.base._write_npz_once(wire, arrays)
        cal = ["a" * 64, "b" * 64]
        metadata = [{"purpose": purpose, "code_length_bits": 10,
                     "block_count": 1, "fallback": False, "stochastic": False,
                     "kind": "D_VQ_JOINT", "calibration": "H-GLOBAL-TEMP",
                     "calibrator_sha256": cal[purpose]} for purpose in (0, 1)]
        package = {"schema": "team-d-private-calibrated-wire-v1", "role": role,
                   "arm_id": "D_VQ_JOINT", "seed": 0, "rows": 2,
                   "source_model_sha256": attack_runner.base.U_MODEL_SHA[0],
                   "row_unit_pair_sha256": "c" * 64,
                   "role_features_sha256": "d" * 64,
                   "calibrator_sha256": cal, "wire_metadata": metadata,
                   "wire_members": sorted(arrays), "wire_npz_sha256": wire_sha}
        manifest_path = directory / "manifest.json"
        attack_runner.base._write_json(manifest_path, package)
        complete = {"schema": package["schema"],
                    "manifest_sha256": attack_runner.base._sha256(manifest_path),
                    "wire_npz_sha256": wire_sha}
        complete_path = directory / "COMPLETE.json"
        attack_runner.base._write_json(complete_path, complete)
        release = {"calibrator_sha256": cal,
                   "fit_source_model_sha256": attack_runner.base.U_MODEL_SHA[0],
                   "packages": [{"role": role,
                                 "package_complete_sha256": attack_runner.base._sha256(complete_path),
                                 "wire_npz_sha256": wire_sha}]}
        admitted = {"roles": {role: {"rows": 2}},
                    "real_file_specs": {f"{role}:features": {
                        "row_unit_pair_sha256": "c" * 64, "sha256": "d" * 64}}}
        args = types.SimpleNamespace(arm_id="D_VQ_JOINT", seed=0)
        return release, admitted, args, package, manifest_path

    def test_exact_synthetic_wire_load_and_sex_alignment(self):
        with tempfile.TemporaryDirectory(prefix="d-attack-wire-synthetic-") as temp:
            root = Path(temp)
            release, manifest, args, _, _ = self._package(root)
            aligned = attack_runner._load_wire(root, "ATTACK_FIT_NEW",
                                                release, manifest, args)
            self.assertEqual(aligned.codes("pair").tolist(), [[0, 1], [0, 1]])
            attack_runner._check_sex_alignment(
                aligned, {"row_id": np.array([1, 2]), "unit": np.array([1, 2]),
                          "sex": np.array([0, 1])}, "ATTACK_FIT_NEW")
            with self.assertRaises(ValueError):
                attack_runner._check_sex_alignment(
                    aligned, {"row_id": np.array([2, 1]), "unit": np.array([2, 1]),
                              "sex": np.array([0, 1])}, "ATTACK_FIT_NEW")

    def test_package_manifest_mutation_refused(self):
        with tempfile.TemporaryDirectory(prefix="d-attack-wire-synthetic-") as temp:
            root = Path(temp)
            release, manifest, args, package, path = self._package(root)
            package["role_features_sha256"] = "e" * 64
            path.write_text(json.dumps(package))
            with self.assertRaises(ValueError):
                attack_runner._load_wire(root, "ATTACK_FIT_NEW", release, manifest, args)

    def test_three_view_fit_freezes_before_inner_label_read(self):
        events = []

        class FakeStore:
            def read(self, role, field, procedure):
                events.append(("read", role, field, procedure))
                return {"row_id": np.array([1, 2], dtype=np.int64),
                        "unit": np.array([1, 2], dtype=np.int64),
                        "sex": np.array([0, 1], dtype=np.int64)}

        class FakeAligned:
            row_id = np.array([1, 2], dtype=np.int64)
            unit_id = np.array([1, 2], dtype=np.int64)
            recipient0 = {"code_ids": np.empty((2, 0), dtype=np.int64)}
            recipient1 = {"code_ids": np.empty((2, 0), dtype=np.int64)}

        class FakeS2View:
            def content_hashes(self):
                return {"synthetic_only": "a" * 64}

        class FakeAttack:
            AttackView = object

            @staticmethod
            def fit_slate(att, sex_att, inner, **kwargs):
                events.append(("fit", kwargs["attacker_seed"], tuple(kwargs["required_readers"])))
                kwargs["progress_dir"].mkdir(mode=0o700)
                return {"reader_order": list(kwargs["required_readers"]),
                        "reader_progress": [], "external_readers": []}

            @staticmethod
            def add_ignore_recipient_aliases(pair, local0, local1):
                events.append(("alias",))
                return pair

            @staticmethod
            def save_fitted_slate(fitted, directory, **kwargs):
                events.append(("freeze", directory.parent.name))
                directory.mkdir(mode=0o700)
                return {"private_receipt_sha256": "a" * 64}

            @staticmethod
            def score_slate(fitted, sex_inner):
                events.append(("score",))
                return {"primary_envelope_auc": 0.5}

        with tempfile.TemporaryDirectory(prefix="d-attack-flow-synthetic-") as temp:
            root = Path(temp)
            source, target = root / "source", root / "attack"
            source.mkdir(mode=0o700)
            attack_runner.base._write_json(source / "RELEASE_COMPLETE.json", {})
            args = types.SimpleNamespace(arm_id="D_U_FROZEN", seed=0,
                                         attacker_seed=0, lock_sha256="b" * 64)
            admitted = ({}, {}, "c" * 64, FakeAttack, FakeStore(), source, target, root)
            with patch.object(attack_runner, "_admit", return_value=admitted), \
                 patch.object(attack_runner, "_source_release_record", return_value={}), \
                 patch.object(attack_runner, "_load_wire", return_value=FakeAligned()), \
                 patch.object(attack_runner, "as_s2_attack_view", return_value=FakeS2View()):
                receipt = attack_runner.run_attack(args)
            self.assertEqual(receipt["reader_fit_units"], 12)
            self.assertEqual(sum(event[0] == "fit" for event in events), 3)
            self.assertEqual(sum(event[0] == "freeze" for event in events), 3)
            self.assertEqual(sum(event[0] == "score" for event in events), 3)
            inner_open = next(i for i, event in enumerate(events)
                              if event[:2] == ("read", "INNER_SELECTION"))
            self.assertLess(events.index(("alias",)), inner_open)
            self.assertTrue(all(i < inner_open for i, event in enumerate(events)
                                if event[0] == "freeze"))
            self.assertTrue(all(event[0] == "fit" for event in events[1:4]))
            self.assertEqual(events[0], ("read", "ATTACK_FIT_NEW", "sex", "attack"))


if __name__ == "__main__":
    unittest.main()
