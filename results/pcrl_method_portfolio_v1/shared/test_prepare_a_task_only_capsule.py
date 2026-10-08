"""Synthetic-only S2 task-capsule production and source-access sentinel."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import a_task_only_capsule as pure
import prepare_a_task_only_capsule as producer
from safe_data_guard import sha_file


class SyntheticStore:
    def __init__(self, manifest_path: Path, row: np.ndarray, unit: np.ndarray):
        self.manifest_path = manifest_path
        self.row = row
        self.unit = unit
        self.requests = []
        self.manifest = {"real_file_specs": {
            "OSF_DEFENSE_FIT:income": {"sha256": "c" * 64},
            "OSF_DEFENSE_FIT:references_s0": {"sha256": "b" * 64}}}

    def read(self, role, field, procedure=None):
        self.requests.append((role, field, procedure))
        if role != "OSF_DEFENSE_FIT" or field == "sex" or field == "occupation":
            raise AssertionError("task-only producer requested forbidden label")
        if field == "income" and procedure == "defense_fit":
            return {"row_id": self.row, "unit": self.unit,
                    "income": np.array([0, 1, 0], dtype=np.int64)}
        if field == "references_s0" and procedure is None:
            return {"row_id": self.row, "unit": self.unit,
                    "U0_1": np.array([[.7, .3], [.2, .8], [.6, .4]], dtype=np.float64),
                    "Ucal_1": np.array([[.65, .35], [.25, .75], [.55, .45]], dtype=np.float64),
                    "U0_2": np.full((3, 6), 1 / 6, dtype=np.float64),
                    "Ucal_2": np.full((3, 6), 1 / 6, dtype=np.float64)}
        raise AssertionError("task-only producer requested unplanned field")


class ProducerTests(unittest.TestCase):
    def test_queued_decoy_script_is_refused_before_role_read(self):
        unit_id = "S2-A-TASK-CAPSULE-S0-R1"
        output = producer.PRIVATE / "a_task_only_capsule_s0_r1_v1" / "RECEIPT.json"
        queue_root = Path("/synthetic/queue")
        state = {"units": {unit_id: {"status": "running", "child_pgid": os.getpgrp(),
                                     "spec": {"unit_id": unit_id, "mode": "real_data",
                                              "family": "SHARED", "team": "SHARED",
                                              "stage": "a_task_capsule_preprocess",
                                              "seed": 0, "recipient": 1,
                                              "runner_sha256": sha_file(Path(producer.__file__)),
                                              "outputs": [str(output)],
                                              "argv": ["python", "/synthetic/decoy.py",
                                                       "--seed", "0", "--recipient", "1",
                                                       "--campaign-lock-sha256", "d" * 64,
                                                       "--queue-root", str(queue_root)]}}}}

        class SyntheticQueue:
            def __init__(self, root):
                self.root = root

            def status(self):
                return state

        with patch.object(producer, "Queue", SyntheticQueue):
            with self.assertRaisesRegex(PermissionError, "identity differs"):
                producer._running_unit(0, 1, "d" * 64, queue_root)

    def test_one_recipient_artifact_never_contains_sex_or_partner_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / ".private"
            bank_root = private / "tstar_score_bank_s0_v1"
            bank_root.mkdir(parents=True)
            row = np.array([10, 11, 12], dtype=np.int64)
            unit = np.array([30, 31, 31], dtype=np.int64)
            np.savez_compressed(bank_root / "tokens_OSF_DEFENSE_FIT.npz",
                                row_id=row, unit=unit,
                                cell_1=np.array([0, 1, 0], dtype=np.int64),
                                cell_2=np.array([0, 0, 0], dtype=np.int64),
                                token_1=np.array([0, 1, 0], dtype=np.int64),
                                token_2=np.array([0, 0, 0], dtype=np.int64))
            np.savez_compressed(bank_root / "bank_r1.npz",
                                probability=np.array([[.8, .2], [.1, .9]], dtype=np.float64),
                                token_class=np.array([0, 1], dtype=np.int64),
                                token_id=np.array([0, 1], dtype=np.int64),
                                input_class=np.array([0, 1], dtype=np.int64),
                                fine_to_hcal_token=np.array([0, 1], dtype=np.int64))
            bank_receipt = {
                "source_binding_sha256": "a" * 64,
                "private_output_sha256": {
                    name: sha_file(bank_root / name)
                    for name in ("tokens_OSF_DEFENSE_FIT.npz", "bank_r1.npz")},
                "role_receipts": {"OSF_DEFENSE_FIT": {
                    "row_unit_pair_sha256": pure._sha_arr(np.column_stack((row, unit))),
                    "source_reference_sha256": "b" * 64}}}
            (bank_root / "BANK_ADMISSION_RECEIPT.json").write_text(json.dumps(bank_receipt))
            (root / "S4_A_SCORE_BANK_VERIFICATION_s0.json").write_text('{}\n')
            manifest_path = root / "DATA_ACCESS_MANIFEST.json"
            manifest_path.write_text('{}\n')
            store = SyntheticStore(manifest_path, row, unit)
            with patch.object(producer, "PRIVATE", private), \
                    patch.object(producer, "CAMPAIGN", root), \
                    patch.object(producer, "_running_unit", return_value={}), \
                    patch.object(producer, "_admit", return_value=(store, bank_root, bank_receipt)):
                result = producer.run(0, 1, "d" * 64, root / "queue")
            self.assertEqual(store.requests,
                             [("OSF_DEFENSE_FIT", "income", "defense_fit"),
                              ("OSF_DEFENSE_FIT", "references_s0", None)])
            output = private / "a_task_only_capsule_s0_r1_v1"
            with np.load(output / "task_only.npz", allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), pure.MEMBERS)
                self.assertNotIn("sex", archive.files)
                self.assertNotIn("U0_2", archive.files)
            admitted = pure.load_task_capsule(output / "task_only.npz", output / "RECEIPT.json",
                                              expected_receipt_sha256=result["private_receipt_sha256"],
                                              recipient=1, model_seed=0)
            self.assertEqual(admitted["task_label"].tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
