"""Data-free package/checkpoint fixtures for independent C release replay."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from replay_c_release_v1 import load_packages, predict_release


def _names() -> tuple[str, ...]:
    return ("age", "education-num", "capital-gain", "capital-loss",
            "hours-per-week") + tuple(
                f"{name}={index}" for name, count in (
                    ("workclass", 8), ("education", 16),
                    ("marital-status", 7), ("relationship", 6),
                    ("native-country", 41)) for index in range(count))


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class CReleaseReplayTest(unittest.TestCase):
    def test_two_public_packages_replay_and_hash_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ".private"
            root.mkdir()
            names = _names()
            schema_sha = hashlib.sha256("\n".join(names).encode()).hexdigest()
            source = {
                "OSF_DEFENSE_FIT:features": "a" * 64,
                "OSF_DEFENSE_FIT:income": "b" * 64,
                "OSF_DEFENSE_FIT:occupation": "c" * 64,
                "CALIBRATION_HELDOUT:features": "d" * 64,
                "CALIBRATION_HELDOUT:income": "e" * 64,
                "CALIBRATION_HELDOUT:occupation": "f" * 64,
            }
            units, heads, calibrators = [], [], []
            for purpose, task, width in (("income", "income", 2),
                                         ("occupation_group", "occupation", 6)):
                model_id = f"C-TEST:s0:{purpose}"
                meta = {
                    "schema": "team-c-head-checkpoint-v1", "purpose": purpose,
                    "model_id": model_id, "selected_atoms": ["age"],
                    "selected_columns": 1, "spec": {"hidden_width": 2},
                    "receipt": {"seed": 0, "feature_schema_sha256": schema_sha},
                    "state_keys": ["0.weight", "0.bias", "2.weight", "2.bias",
                                   "4.weight", "4.bias"],
                }
                head_path = root / f"{purpose}.npz"
                np.savez_compressed(
                    head_path, meta=np.array(json.dumps(meta)),
                    state_0=np.zeros((2, 1), dtype=np.float32),
                    state_1=np.zeros(2, dtype=np.float32),
                    state_2=np.zeros((2, 2), dtype=np.float32),
                    state_3=np.zeros(2, dtype=np.float32),
                    state_4=np.zeros((width, 2), dtype=np.float32),
                    state_5=np.zeros(width, dtype=np.float32),
                )
                head_sha = hashlib.sha256(head_path.read_bytes()).hexdigest()
                cal = {
                    "schema": "pcrl-portfolio-global-temp-v1", "task": task,
                    "arm_id": model_id, "model_seed": 0,
                    "fit_role": "CALIBRATION_HELDOUT", "draw_id": "deterministic",
                    "fit_row_unit_pair_sha256": "1" * 64,
                    "alpha": 1.0, "argmax_preserved": True,
                }
                cal_path = root / f"{purpose}.json"
                cal_sha = _write(cal_path, json.dumps(cal).encode())
                hashes = {"fit_features": source["OSF_DEFENSE_FIT:features"],
                          "fit_label": source[f"OSF_DEFENSE_FIT:{task}"],
                          "calibration_features": source["CALIBRATION_HELDOUT:features"],
                          "calibration_label": source[f"CALIBRATION_HELDOUT:{task}"]}
                package = {
                    "schema": "team-c-private-release-package-v1",
                    "purpose": purpose, "arm_id": model_id, "seed": 0,
                    "selected_atoms": ["age"], "feature_schema_sha256": schema_sha,
                    "head_path": head_path.name, "head_sha256": head_sha,
                    "calibrator_path": cal_path.name, "calibrator_sha256": cal_sha,
                    "calibrator_fit_row_unit_pair_sha256": "1" * 64,
                    "campaign_lock_sha256": "2" * 64,
                    "data_manifest_sha256": "3" * 64,
                    "source_role_hashes": hashes,
                }
                package_path = root / f"package_{purpose}.json"
                package_sha = _write(package_path, json.dumps(package).encode())
                units.append({"purpose": purpose,
                              "package_relative_path": package_path.name,
                              "package_sha256": package_sha,
                              "source_role_hashes": hashes})
                heads.append(f"{model_id}:{head_sha[:16]}")
                calibrators.append(f"H-GLOBAL-TEMP:{cal_sha[:16]}")
            fit = {"schema": "team-c-fit-handoff-v1", "arm_id": "C-TEST",
                   "seed": 0, "campaign_lock_sha256": "2" * 64,
                   "data_manifest_sha256": "3" * 64,
                   "fit_cal_role_file_hashes": source,
                   "selected_atoms": {purpose: ["age"] for purpose in
                                      ("income", "occupation_group")},
                   "head_ids": heads, "calibrator_ids": calibrators,
                   "arm_receipt": {"package_units": units}}
            restored = load_packages(fit, root, names, "2" * 64, "3" * 64, source)
            raw, calibrated = predict_release(
                restored["income"], np.zeros((3, 83)), names)
            np.testing.assert_array_equal(raw, np.full((3, 2), .5))
            np.testing.assert_array_equal(calibrated, raw)
            wrong = json.loads(json.dumps(fit))
            wrong["arm_receipt"]["package_units"][0]["package_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "artifact path or SHA"):
                load_packages(wrong, root, names, "2" * 64, "3" * 64, source)


if __name__ == "__main__":
    unittest.main()
