"""Synthetic guard tests for future protected FIT capsule preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import prepare_a_protected_fit_capsule as producer
from a_protected_fit_capsule import MEMBERS, load_protected_capsule, _sha_file
from test_a_protected_fit_capsule import _fixture


def _task_parts(values: dict) -> tuple[dict, dict, dict]:
    shared = {"row_id": values["row_id"], "unit": values["unit"],
              "source_binding_sha256": values["source_binding_sha256"]}
    tasks = []
    for recipient, label in ((1, "income"), (2, "occupation")):
        task = dict(shared)
        task.update({"cell": values[f"cell_{recipient}"],
                     "task_label": values[label],
                     "U0": values[f"U0_{recipient}"],
                     "Ucal": values[f"Ucal_{recipient}"],
                     "bank": values[f"bank_{recipient}"],
                     "input_class": values[f"input_class_{recipient}"],
                     "fine_to_hcal_token": values[f"fine_to_hcal_token_{recipient}"],
                     "token_class": values[f"token_class_{recipient}"]})
        tasks.append(task)
    sex = {"row_id": values["row_id"], "unit": values["unit"],
           "sex": values["sex"]}
    return tasks[0], tasks[1], sex


class ProtectedProducerTests(unittest.TestCase):
    def test_assembled_task_marginals_are_bitwise_equal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            npz, receipt, sha = _fixture(Path(directory))
            values = load_protected_capsule(npz, receipt,
                                            expected_receipt_sha256=sha,
                                            model_seed=0)
            task1, task2, sex = _task_parts(values)
            assembled = producer._assemble_arrays(task1, task2, sex)
            self.assertEqual(set(assembled), MEMBERS)
            for name in MEMBERS:
                self.assertTrue(np.array_equal(assembled[name], values[name]))

    def test_misalignment_and_extra_sex_field_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            npz, receipt, sha = _fixture(Path(directory))
            values = load_protected_capsule(npz, receipt,
                                            expected_receipt_sha256=sha,
                                            model_seed=0)
            task1, task2, sex = _task_parts(values)
            task2 = dict(task2)
            task2["unit"] = np.array([7, 8, 9], dtype=np.int64)
            with self.assertRaises(PermissionError):
                producer._assemble_arrays(task1, task2, sex)
            _, task2, sex = _task_parts(values)
            sex["occupation"] = np.array([0, 1, 2], dtype=np.int64)
            with self.assertRaises(PermissionError):
                producer._assemble_arrays(task1, task2, sex)

    def test_missing_protected_stage_refuses_before_role_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            lock_path = campaign / "CAMPAIGN_LOCK.json"
            lock_path.write_text(json.dumps({
                "status": "LOCAL_GIT_SEALED_EFFECTIVE",
                "adult_fit_permission": True,
                "families": {"A": {"fit_admission": "DENIED",
                                   "protected_capsule_preprocessing_admission": "DENIED"}},
            }))
            with mock.patch.object(producer, "CAMPAIGN", campaign), \
                    mock.patch.object(producer, "SafeRoleStore",
                                      side_effect=AssertionError("role read")):
                with self.assertRaises(PermissionError):
                    producer._admit(0, _sha_file(lock_path), {"gate": {}})

    def test_decoy_script_is_refused_before_role_read(self) -> None:
        seed, lock_sha = 0, "a" * 64
        uid = "S2-A-PROTECTED-FIT-CAPSULE-S0"
        output = producer.PRIVATE / "a_protected_fit_capsule_s0_v1" / "RECEIPT.json"
        spec = {"unit_id": uid, "mode": "real_data", "family": "SHARED",
                "team": "SHARED", "arm": "A_PROTECTED_FIT_CAPSULE",
                "arm_id": "A_PROTECTED_FIT_CAPSULE",
                "stage": "a_protected_fit_capsule_preprocess",
                "data_access": producer.DATA_ACCESS, "seed": seed,
                "runner_sha256": _sha_file(Path(producer.__file__)),
                "outputs": [str(output)],
                "argv": ["python", "/tmp/decoy.py", "--seed", "0",
                         "--campaign-lock-sha256", lock_sha,
                         "--queue-root", "/tmp/queue"]}
        state = {"units": {uid: {"status": "running", "spec": spec,
                                 "child_pgid": producer.os.getpgrp()}}}
        with mock.patch.object(producer, "Queue") as queue:
            queue.return_value.status.return_value = state
            with self.assertRaises(PermissionError):
                producer._running_unit(seed, lock_sha, Path("/tmp/queue"))

    def test_missing_independent_task_report_refuses_before_task_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            (campaign / "S4_A_SCORE_BANK_VERIFICATION_s0.json").write_text("{}")
            family = {"task_capsule_receipt_sha256_by_seed_recipient": {
                          "s0:r1": "a" * 64, "s0:r2": "b" * 64},
                      "task_capsule_s4_sha256_by_seed_recipient": {
                          "s0:r1": "c" * 64, "s0:r2": "d" * 64}}
            with mock.patch.object(producer, "CAMPAIGN", campaign), \
                    mock.patch.object(producer, "PRIVATE", campaign / "private"), \
                    mock.patch.object(producer, "load_task_capsule",
                                      side_effect=AssertionError("task read")):
                with self.assertRaises(PermissionError):
                    producer._task_capsules(0, {"families": {"A": family}}, {})


if __name__ == "__main__":
    unittest.main()
