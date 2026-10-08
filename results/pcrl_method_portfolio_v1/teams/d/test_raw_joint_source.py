"""Synthetic source-custody refusals; no Adult checkpoint or role input is read."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from encoder_adapter import schema_hash
from raw_joint_source import FrozenRawJointEncoder, RAW_MODEL_SHA


class RawJointSourceIdentityTest(unittest.TestCase):
    def test_package_path_and_public_admission_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            columns = tuple(f"x{i}" for i in range(83))
            feature_sha = schema_hash(columns)
            admission = root / "SOURCE_ADMISSION.json"
            admission.write_text(json.dumps({
                "schema": "hcal-source-admission-v1", "verdict": "ADMITTED",
                "source_tip": "976202546a100f2b89a863cb3785c3465cf76c6a",
                "admitted_teacher_dirs": {"rel__s0__RAW-J_b0.3": {
                    "model.pt": RAW_MODEL_SHA[0]}},
            }))
            wrong_package = root / "rel__s0__U" / "model.pt"
            with self.assertRaisesRegex(PermissionError, "package path"):
                FrozenRawJointEncoder.load(wrong_package, admission,
                                           columns, feature_sha, seed=0)
            correct_package = root / "rel__s0__RAW-J_b0.3" / "model.pt"
            admission.write_text(json.dumps({"schema": "hcal-source-admission-v1",
                                             "verdict": "PENDING", "units": {}}))
            with self.assertRaisesRegex(PermissionError, "hcal admission"):
                FrozenRawJointEncoder.load(correct_package, admission,
                                           columns, feature_sha, seed=0)
            # A valid public manifest reaches the independent checkpoint-byte
            # gate; this dummy model is never treated as an admitted source.
            admission.write_text(json.dumps({
                "schema": "hcal-source-admission-v1", "verdict": "ADMITTED",
                "source_tip": "976202546a100f2b89a863cb3785c3465cf76c6a",
                "admitted_teacher_dirs": {"rel__s0__RAW-J_b0.3": {
                    "model.pt": RAW_MODEL_SHA[0]}},
            }))
            correct_package.parent.mkdir()
            correct_package.write_bytes(b"synthetic-unadmitted-model")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
                FrozenRawJointEncoder.load(correct_package, admission,
                                           columns, feature_sha, seed=0)

    def test_unregistered_seed_refused(self):
        with self.assertRaisesRegex(PermissionError, "unregistered"):
            FrozenRawJointEncoder.load(Path("model.pt"), Path("admission.json"),
                                       tuple(f"x{i}" for i in range(83)),
                                       "a" * 64, seed=3)


if __name__ == "__main__":
    unittest.main()
