"""Synthetic preaccess checks for the separate S4 Adult replay worker.

No Adult role path is opened by this test. The missing module is the intended
test-first RED until the independent worker is implemented and S3 admits it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from run_adult_replay_v3 import parent_output_root, running_unit_gate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AdultReplayPreaccessTest(unittest.TestCase):
    def test_parent_receipt_binds_single_immutable_private_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / ".private" / "attempt" / "ATTEMPT_INDEX.json"
            output.parent.mkdir(parents=True)
            output.write_text('{"status":"FROZEN_FOR_INDEPENDENT_REPLAY"}\n')
            receipt = root / "RECEIPT.json"
            receipt.write_text(json.dumps({
                "unit_id": "B-SCREEN-B-TASK-S0-R2", "status": "done",
                "returncode": 0, "output_sha256": {str(output): digest(output)}
            }, sort_keys=True) + "\n")
            resolved = parent_output_root(
                receipt, digest(receipt), digest(output),
                "ATTEMPT_INDEX.json", "B-SCREEN-B-TASK-S0-R2")
            self.assertEqual(resolved, output.parent.resolve())
            with self.assertRaises((ValueError, PermissionError)):
                parent_output_root(receipt, digest(receipt), "0" * 64,
                                   "ATTEMPT_INDEX.json", "B-SCREEN-B-TASK-S0-R2")
            extra = root / ".private" / "other" / "ATTEMPT_INDEX.json"
            extra.parent.mkdir(parents=True)
            extra.write_bytes(output.read_bytes())
            payload = json.loads(receipt.read_text())
            payload["output_sha256"][str(extra)] = digest(extra)
            receipt.write_text(json.dumps(payload, sort_keys=True) + "\n")
            with self.assertRaises((ValueError, PermissionError)):
                parent_output_root(receipt, digest(receipt), digest(output),
                                   "ATTEMPT_INDEX.json", "B-SCREEN-B-TASK-S0-R2")

    def test_running_s3_unit_requires_exact_script_argv_scope_and_pgid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            script = root / "run_adult_replay_v3.py"
            script.write_text("# synthetic script path\n")
            report = root / ".private" / "s4_report.json"
            unit_id = "S4-B-TASK-S0"
            argv = ["/synthetic/python", str(script), "--unit-id", unit_id,
                    "--private-report", str(report)]
            unit_spec = {
                "unit_id": unit_id, "kind": "B_SCREEN", "arm": "B-TASK",
                "seed": 0,
                "data_access": "S4_B_FIT_TASK_ATT_INNER_ROLE_ONLY_REPLAY",
            }
            state = {"gate": {"safe_inputs": True,
                              "occupancy_confirmed": True,
                              "manually_closed": False},
                     "units": {unit_id: {
                         "status": "running", "child_pgid": os.getpgrp(),
                         "spec": {"unit_id": unit_id, "mode": "s4_adult_replay",
                                  "family": "S4", "team": "S4",
                                  "stage": "independent_b_v2_replay",
                                  "arm": "B-TASK", "seed": 0,
                                  "data_access": unit_spec["data_access"],
                                  "argv": argv, "outputs": [str(report)]}}}}
            running_unit_gate(state, unit_spec, script, argv, report)
            wrong = json.loads(json.dumps(state))
            wrong["units"][unit_id]["spec"]["seed"] = 1
            with self.assertRaises((ValueError, PermissionError)):
                running_unit_gate(wrong, unit_spec, script, argv, report)
            wrong = json.loads(json.dumps(state))
            wrong["units"][unit_id]["spec"]["argv"] = [
                "/synthetic/python", "/synthetic/decoy.py", str(script), *argv[2:]]
            with self.assertRaises((ValueError, PermissionError)):
                running_unit_gate(wrong, unit_spec, script, argv, report)
            wrong = json.loads(json.dumps(state))
            wrong["units"][unit_id]["child_pgid"] = os.getpgrp() + 1
            with self.assertRaises((ValueError, PermissionError)):
                running_unit_gate(wrong, unit_spec, script, argv, report)
            wrong = json.loads(json.dumps(state))
            wrong["units"][unit_id]["spec"]["outputs"] = [str(root / "other.json")]
            with self.assertRaises((ValueError, PermissionError)):
                running_unit_gate(wrong, unit_spec, script, argv, report)


if __name__ == "__main__":
    unittest.main()
