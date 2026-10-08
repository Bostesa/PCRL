"""Synthetic admission checks for immutable V2 anchor retry units."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
from compute_queue import Queue


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AnchorR2AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.campaign = self.repo / "results/pcrl_method_portfolio_v1"
        self.shared = self.campaign / "shared"
        self.shared.mkdir(parents=True)
        self.uid = "S2-FEATURE-REF-RAW83-U-S0-V2-PROBE-R2"
        self.parent_id = "S2-FEATURE-REF-RAW83-U-S0"
        self.script = self.shared / "run_feature_ref_raw83_u_v2_probe_r2.py"
        self.script.write_text("# synthetic reviewed runner\n")
        self.reader_shas = {"r1": "1" * 64, "r2": "2" * 64, "pair": "3" * 64}
        self.lock = self.campaign / "CONTROL_V2_LOCK.json"
        self.lock.write_text(json.dumps({
            "probe_unit_ids": {"FEATURE_REF_RAW83_U:s0": self.uid},
            "unit_specs": {self.uid: {
                "family": "SHARED", "arm": "FEATURE_REF_RAW83_U", "seed": 0,
                "stage": "postrelease_control_v2", "parent_unit_id": self.parent_id}},
            "anchor_probe_runner_sha256": sha(self.script),
            "data_access_manifest_sha256": "d" * 64,
            "accepted_fit_lock_sha256": ["a" * 64],
        }))
        self.seal = self.campaign / "CONTROL_V2_LOCAL_SEAL_RECEIPT.json"
        self.seal.write_text("{}\n")
        self.gate = {"safe_inputs": True, "v2_local_seal_path": str(self.seal),
                     "v2_local_seal_sha256": sha(self.seal)}

    def test_exact_r2_anchor_identity_and_versioned_output_are_admitted(self):
        output = (self.shared / ".private" / "feature_ref_raw83_u_s0_v2_probe_r2" /
                  "V2_ANCHOR_PROBE_RECEIPT.json")
        spec = {
            "unit_id": self.uid, "mode": "real_data", "family": "SHARED",
            "team": "SHARED", "arm": "FEATURE_REF_RAW83_U",
            "arm_id": "FEATURE_REF_RAW83_U", "seed": 0,
            "stage": "postrelease_control_v2", "cwd": str(self.repo),
            "data_access": "ATTACK_INNER_FEATURES_AND_SEX_V2_CONTROL",
            "parent_unit_id": self.parent_id, "deps": [self.parent_id],
            "parent_anchor_receipt_sha256": "b" * 64,
            "reader_receipt_sha256": self.reader_shas,
            "runner_sha256": sha(self.script), "outputs": [str(output)],
            "argv": [sys.executable, str(self.script), "--seed", "0",
                     "--unit-id", self.uid,
                     "--control-lock-sha256", sha(self.lock),
                     "--control-seal-sha256", sha(self.seal),
                     "--data-manifest-sha256", "d" * 64,
                     "--parent-anchor-receipt-sha256", "b" * 64],
        }
        with mock.patch.object(Queue, "_local_v2_seal_ok", return_value=True):
            self.assertIsNone(Queue._v2_admission_reason(self.gate, spec))
            changed = dict(spec, argv=[sys.executable, str(self.script),
                                       "--seed", "0", "--unit-id", "other",
                                       *spec["argv"][6:]])
            self.assertIsNotNone(Queue._v2_admission_reason(self.gate, changed))
            old_output = str(output).replace("_v2_probe_r2/", "_v2_probe/")
            self.assertIsNotNone(Queue._v2_admission_reason(
                self.gate, dict(spec, outputs=[old_output])))

    def test_reader_receipt_map_must_equal_hash_pinned_parent_before_launch(self):
        queue = Queue(Path(self.temp.name) / "queue")
        parent_output = Path(self.temp.name) / "ANCHOR_RECEIPT.json"
        parent_output.write_text(json.dumps({"reader_receipt_sha256": self.reader_shas}) + "\n")
        parent_receipt = queue.root / "units" / self.parent_id / "RECEIPT.json"
        parent_receipt.parent.mkdir(parents=True)
        parent_receipt.write_text('{"status":"done","returncode":0}\n')
        lock = json.loads(self.lock.read_text())
        lock.update({
            "parent_queue_receipt_sha256_by_unit": {self.parent_id: sha(parent_receipt)},
            "parent_anchor_receipt_sha256_by_unit": {self.parent_id: sha(parent_output)},
        })
        self.lock.write_text(json.dumps(lock))
        state = queue.status()
        state["gate"] = self.gate
        state["units"][self.parent_id] = {
            "status": "done", "returncode": 0,
            "spec": {"unit_id": self.parent_id, "team": "SHARED", "seed": 0,
                     "stage": "feature_anchor", "outputs": [str(parent_output)],
                     "argv": [sys.executable, "parent.py", "--campaign-lock-sha256", "a" * 64]},
            "output_sha256": {str(parent_output): sha(parent_output)},
        }
        child = {
            "unit_id": self.uid, "family": "SHARED", "arm_id": "FEATURE_REF_RAW83_U",
            "seed": 0, "stage": "postrelease_control_v2", "parent_unit_id": self.parent_id,
            "deps": [self.parent_id], "parent_receipt_sha256": sha(parent_receipt),
            "parent_anchor_receipt_sha256": sha(parent_output),
            "parent_fit_lock_sha256": "a" * 64,
            "reader_receipt_sha256": self.reader_shas,
            "outputs": [str(Path(self.temp.name) / ".private" / "new" /
                            "V2_ANCHOR_PROBE_RECEIPT.json")],
            "argv": [sys.executable, "worker.py", "--queue-root", str(queue.root)],
        }
        self.assertIsNone(queue._v2_parent_reason(state, child))
        changed = dict(child)
        changed.pop("reader_receipt_sha256")
        self.assertEqual(queue._v2_parent_reason(state, changed),
                         "v2_parent_reader_receipts_mismatch")
        swapped = dict(child, reader_receipt_sha256={
            "r1": self.reader_shas["r2"], "r2": self.reader_shas["r1"],
            "pair": self.reader_shas["pair"]})
        self.assertEqual(queue._v2_parent_reason(state, swapped),
                         "v2_parent_reader_receipts_mismatch")


if __name__ == "__main__":
    unittest.main()
