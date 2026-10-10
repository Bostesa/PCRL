"""Boundary tests: public queue rows cannot disclose private command or output paths."""

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("queue_unit_status.py")


class QueueUnitStatusTests(unittest.TestCase):
    def test_finished_and_pending_rows_are_whitelisted_and_receipt_hashed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            done_id = "D-FIRST-01"
            receipt_dir = root / "units" / done_id
            receipt_dir.mkdir(parents=True)
            receipt = {
                "unit_id": done_id,
                "stage": "d_first",
                "status": "done",
                "direct_child_cpu_s": 4.0,
                "charged_cpu_s": 5.0,
                "wall_s": 5.0,
                "sampled_tree_rss_peak_bytes": 123456,
                "private_note": "/Users/person/DO_NOT_DISCLOSE",
            }
            receipt_path = receipt_dir / "RECEIPT.json"
            receipt_path.write_text(json.dumps(receipt))
            receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            state = {
                "schema": "pcrl-latent-sampled-queue-v1",
                "units": {
                    done_id: {
                        **receipt,
                        "spec": {
                            "unit_id": done_id,
                            "stage": "d_first",
                            "cwd": "/Users/person/DO_NOT_DISCLOSE",
                            "argv": ["/private/DO_NOT_DISCLOSE"],
                            "outputs": ["/private/DO_NOT_DISCLOSE"],
                        },
                        "output_sha256": {"/private/DO_NOT_DISCLOSE": "a" * 64},
                    },
                    "D-NEXT-02": {
                        "status": "queued",
                        "spec": {"unit_id": "D-NEXT-02", "stage": "d_followup", "cwd": "/private/DO_NOT_DISCLOSE"},
                    },
                    "A-HELD-03": {
                        "status": "held",
                        "hold_reason": "/private/DO_NOT_DISCLOSE",
                        "spec": {"unit_id": "A-HELD-03", "stage": "a", "cwd": "/private/DO_NOT_DISCLOSE"},
                    },
                },
            }
            state_path = root / "state.json"
            state_path.write_text(json.dumps(state))
            result = subprocess.run([sys.executable, str(SCRIPT), "--state", str(state_path)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("DO_NOT_DISCLOSE", result.stdout)
            self.assertNotIn(str(root), result.stdout)
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            self.assertEqual(list(rows[0]), ["record_kind", "record_id", "unit_id", "charge_id",
                                              "stage", "status", "direct_child_cpu_s",
                                              "charged_cpu_s", "wall_s", "sampled_tree_rss_peak_bytes",
                                              "receipt_sha256"])
            self.assertEqual(rows, [
                {"record_kind": "unit", "record_id": "A-HELD-03", "unit_id": "A-HELD-03", "charge_id": "",
                 "stage": "a", "status": "held",
                 "direct_child_cpu_s": "", "charged_cpu_s": "", "wall_s": "",
                 "sampled_tree_rss_peak_bytes": "", "receipt_sha256": ""},
                {"record_kind": "unit", "record_id": done_id, "unit_id": done_id, "charge_id": "",
                 "stage": "d_first", "status": "done", "direct_child_cpu_s": "4.0",
                 "charged_cpu_s": "5.0", "wall_s": "5.0", "sampled_tree_rss_peak_bytes": "123456",
                 "receipt_sha256": receipt_hash},
                {"record_kind": "unit", "record_id": "D-NEXT-02", "unit_id": "D-NEXT-02", "charge_id": "",
                 "stage": "d_followup", "status": "queued",
                 "direct_child_cpu_s": "", "charged_cpu_s": "", "wall_s": "",
                 "sampled_tree_rss_peak_bytes": "", "receipt_sha256": ""},
            ])

    def test_supersession_and_charge_receipts_are_verified_and_included(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_id = "A-OLD"
            supersession = {"schema": "pcrl-queue-unstarted-supersession-v1", "old_unit_id": old_id,
                            "new_unit_id": "A-NEW", "old_child_started": False, "old_cpu_charge_s": 0,
                            "reason": "/Users/person/DO_NOT_DISCLOSE"}
            sup_path = root / "units" / old_id / "SUPERSESSION.json"
            sup_path.parent.mkdir(parents=True)
            sup_path.write_text(json.dumps(supersession))
            sup_hash = hashlib.sha256(sup_path.read_bytes()).hexdigest()
            charge = {"schema": "pcrl-r03-setup-cost-v1", "charge_id": "R03_SETUP",
                      "stage": "admission", "charged_cpu_s": 60.0,
                      "basis": "/private/DO_NOT_DISCLOSE"}
            charge_path = root / "SETUP_COST.json"
            charge_path.write_text(json.dumps(charge))
            charge_hash = hashlib.sha256(charge_path.read_bytes()).hexdigest()
            state = {"schema": "pcrl-latent-sampled-queue-v1",
                     "units": {old_id: {"status": "superseded", "spec": {"unit_id": old_id, "stage": "a"},
                                        "superseded_by": "A-NEW", "supersession_receipt_path": str(sup_path),
                                        "supersession_receipt_sha256": sup_hash}},
                     "charges": {"R03_SETUP": {"charge_id": "R03_SETUP", "stage": "admission", "cpu_s": 60.0,
                                               "receipt_path": str(charge_path), "receipt_sha256": charge_hash}}}
            state_path = root / "state.json"
            state_path.write_text(json.dumps(state))
            result = subprocess.run([sys.executable, str(SCRIPT), "--state", str(state_path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("DO_NOT_DISCLOSE", result.stdout)
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["record_kind"], "unit")
            self.assertEqual(rows[0]["status"], "superseded")
            self.assertEqual(rows[0]["receipt_sha256"], sup_hash)
            self.assertEqual(rows[1]["record_kind"], "charge")
            self.assertEqual(rows[1]["charge_id"], "R03_SETUP")
            self.assertEqual(rows[1]["charged_cpu_s"], "60.0")
            self.assertEqual(rows[1]["receipt_sha256"], charge_hash)

            charge_path.write_text(json.dumps({**charge, "charged_cpu_s": 61.0}))
            rejected = subprocess.run([sys.executable, str(SCRIPT), "--state", str(state_path)],
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(rejected.stdout, "")

            charge_path.write_text(json.dumps(charge))
            orphan = root / "units" / "ORPHAN" / "RECEIPT.json"
            orphan.parent.mkdir(parents=True)
            orphan.write_text("{}")
            rejected_orphan = subprocess.run([sys.executable, str(SCRIPT), "--state", str(state_path)],
                                             capture_output=True, text=True)
            self.assertNotEqual(rejected_orphan.returncode, 0)
            self.assertEqual(rejected_orphan.stdout, "")

    def test_path_like_unit_id_fails_closed_without_echoing_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = "/Users/person/DO_NOT_DISCLOSE"
            (root / "state.json").write_text(json.dumps({
                "schema": "pcrl-latent-sampled-queue-v1",
                "units": {marker: {"status": "queued", "spec": {"stage": "d_first"}}},
            }))
            result = subprocess.run([sys.executable, str(SCRIPT), "--state", str(root / "state.json")],
                                    capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("queue export rejected", result.stderr)
            self.assertNotIn("DO_NOT_DISCLOSE", result.stderr)


if __name__ == "__main__":
    unittest.main()
