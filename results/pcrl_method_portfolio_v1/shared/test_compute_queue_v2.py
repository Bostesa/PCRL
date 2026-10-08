"""Synthetic prefit checks for the separate V2 control admission boundary."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
from compute_queue import Queue


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V2QueueAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.campaign = self.repo / "results/pcrl_method_portfolio_v1"
        self.campaign.mkdir(parents=True)
        self.lock = self.campaign / "CONTROL_V2_LOCK.json"
        self.lock.write_text(json.dumps({
            "schema": "pcrl-control-v2-lock-v1",
            "status": "LOCAL_GIT_SEALED_EFFECTIVE",
            "scope": "B_CONTROL_V2",
            "control_version": 2,
            "adult_probe_permission": True,
            "accepted_fit_lock_sha256": ["a" * 64],
            "probe_unit_ids": {"B-TASK:s0": "B-V2-REAL-PROBES-B-TASK-S0"},
        }))
        subprocess.run(["git", "-C", str(self.repo), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "results"], check=True)
        env = dict(os.environ, GIT_AUTHOR_NAME="Queue Test", GIT_AUTHOR_EMAIL="queue@test.invalid",
                   GIT_COMMITTER_NAME="Queue Test", GIT_COMMITTER_EMAIL="queue@test.invalid")
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "V2 fixture"],
                       check=True, env=env)
        self.commit = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                                              text=True).strip()
        self.review = self.campaign / "S4_CONTROL_V2_PREFIT_REVIEW.json"
        self.review.write_text(json.dumps({"status": "PASS", "scope": "B_CONTROL_V2",
                                           "lock_commit_sha": self.commit,
                                           "lock_sha256": _sha(self.lock),
                                           "control_lock_sha256": _sha(self.lock)}))
        self.receipt = self.campaign / "CONTROL_V2_LOCAL_SEAL_RECEIPT.json"
        self.receipt.write_text(json.dumps({
            "schema": "pcrl-control-v2-local-seal-receipt-v1",
            "status": "PASS", "scope": "B_CONTROL_V2",
            "commit_sha": self.commit, "lock_commit_sha": self.commit,
            "control_lock_sha256": _sha(self.lock), "lock_sha256": _sha(self.lock),
            "review_path": self.review.name, "review_sha256": _sha(self.review),
        }))
        self.gate = {"safe_inputs": True, "v2_local_seal_path": str(self.receipt),
                     "v2_local_seal_sha256": _sha(self.receipt)}

    def test_v2_requires_its_own_committed_reviewed_seal(self):
        self.assertTrue(Queue._local_v2_seal_ok(self.gate))
        changed = dict(self.gate, v2_local_seal_sha256="0" * 64)
        self.assertFalse(Queue._local_v2_seal_ok(changed))
        receipt = json.loads(self.receipt.read_text())
        receipt["scope"] = "ALL_FAMILIES"
        self.receipt.write_text(json.dumps(receipt))
        changed = dict(self.gate, v2_local_seal_sha256=_sha(self.receipt))
        self.assertFalse(Queue._local_v2_seal_ok(changed))

    def test_v2_probe_stage_does_not_inherit_b_screen_seal(self):
        queue = Queue(Path(self.temp.name) / "queue")
        queue.update_gate(safe_inputs=True, lock_pushed=False, occupancy_confirmed=True,
                          external_heavy=0, local_seal_path=str(self.receipt),
                          local_seal_sha256=_sha(self.receipt))
        spec = {"unit_id": "B-V2-REAL-PROBES-B-TASK-S0", "family": "B", "team": "B",
                "arm": "B-TASK", "arm_id": "B-TASK", "seed": 0,
                "stage": "postrelease_control_v2", "mode": "real_data",
                "cwd": str(self.repo), "argv": [sys.executable, "missing.py"],
                "deps": [], "max_wall_s": 10, "cpu_reserve_s": 10, "max_rss_gib": 1.0}
        with self.assertRaisesRegex(ValueError, "local_seal_invalid"):
            queue.enqueue(spec)

    def test_b_v2_exact_unit_script_parent_and_output_mapping(self):
        script = self.campaign / "teams/b/real_control_probe_v2.py"
        script.parent.mkdir(parents=True)
        script.write_text("# synthetic B V2 worker\n")
        uid = "B-V2-REAL-PROBES-B-TASK-S0"
        parent_id = "B-SCREEN-B-TASK-S0-R2"
        lock = json.loads(self.lock.read_text())
        lock.update({
            "unit_specs": {uid: {"arm": "B-TASK", "seed": 0, "stage": "postrelease_control_v2",
                                  "screen_unit_id": parent_id, "v1_probe_index_sha256": None}},
            "probe_runner_sha256": _sha(script),
            "parent_attempt_index_sha256": {"B-TASK:s0": "b" * 64},
            "screen_config_sha256": "c" * 64,
            "data_access_manifest_sha256": "d" * 64,
        })
        self.lock.write_text(json.dumps(lock))
        output = self.repo / "results/pcrl_method_portfolio_v1/teams/b/.private" / (
            f"real_probe_v2__B-TASK__s0__{parent_id}") / "REAL_CONTROL_V2_INDEX.json"
        spec = {
            "unit_id": uid, "mode": "real_data", "family": "B", "team": "B",
            "arm": "B-TASK", "arm_id": "B-TASK", "seed": 0,
            "stage": "postrelease_control_v2", "cwd": str(self.repo),
            "data_access": "ATTACK_INNER_FEATURES_AND_SEX_V2_CONTROL",
            "parent_unit_id": parent_id, "deps": [parent_id],
            "parent_attempt_index_sha256": "b" * 64,
            "parent_fit_lock_sha256": "a" * 64,
            "v1_probe_index_sha256": None,
            "outputs": [str(output)],
            "argv": [sys.executable, str(script), "--arm", "B-TASK", "--seed", "0",
                     "--unit-id", uid, "--screen-unit-id", parent_id,
                     "--attempt-index-sha256", "b" * 64,
                     "--parent-fit-lock-sha256", "a" * 64,
                     "--control-v2-lock-sha256", _sha(self.lock),
                     "--control-v2-lock", str(self.lock),
                     "--v2-local-seal-sha256", self.gate["v2_local_seal_sha256"],
                     "--v2-local-seal-receipt", str(self.receipt),
                     "--config-sha256", "c" * 64, "--manifest-sha256", "d" * 64,
                     "--private-output", str(output.parent)],
        }
        with mock.patch.object(Queue, "_local_v2_seal_ok", return_value=True):
            self.assertIsNone(Queue._v2_admission_reason(self.gate, spec))
            self.assertIsNotNone(Queue._v2_admission_reason(self.gate, dict(spec, unit_id="extra-fit")))
            self.assertIsNotNone(Queue._v2_admission_reason(self.gate, dict(spec, seed=1)))
            self.assertIsNotNone(Queue._v2_admission_reason(self.gate, dict(
                spec, argv=[sys.executable, "-c", "pass", str(script), *spec["argv"][2:]])))
            self.assertIsNotNone(Queue._v2_admission_reason(self.gate, dict(spec, outputs=[])))
            self.assertIsNotNone(Queue._v2_admission_reason(self.gate, dict(spec, parent_unit_id="other")))

    def test_v2_parent_hash_is_rechecked_after_enqueue(self):
        uid = "B-V2-REAL-PROBES-B-TASK-S0"
        parent_id = "B-SCREEN-B-TASK-S0-R2"
        queue = Queue(Path(self.temp.name) / "parent_queue")
        parent_output = Path(self.temp.name) / "ATTEMPT_INDEX.json"
        parent_output.write_text('{"frozen":true}\n')
        parent_receipt = queue.root / "units" / parent_id / "RECEIPT.json"
        parent_receipt.parent.mkdir(parents=True)
        parent_receipt.write_text('{"status":"done","returncode":0}\n')
        parent_spec = {"unit_id": parent_id, "mode": "real_data", "team": "B", "arm": "B-TASK",
                       "seed": 0, "stage": "screen", "outputs": [str(parent_output)],
                       "argv": [sys.executable, "screen.py", "--lock-sha256", "a" * 64]}
        state = queue.status()
        state["gate"] = self.gate
        state["units"][parent_id] = {"spec": parent_spec, "status": "done", "returncode": 0,
                                      "output_sha256": {str(parent_output): _sha(parent_output)}}
        queue._save(state)
        lock = json.loads(self.lock.read_text())
        lock["parent_attempt_index_sha256"] = {"B-TASK:s0": _sha(parent_output)}
        lock["parent_queue_receipt_sha256_by_unit"] = {parent_id: _sha(parent_receipt)}
        self.lock.write_text(json.dumps(lock))
        child = {"unit_id": uid, "family": "B", "arm_id": "B-TASK", "seed": 0,
                 "stage": "postrelease_control_v2", "parent_unit_id": parent_id,
                 "deps": [parent_id], "parent_receipt_sha256": _sha(parent_receipt),
                 "parent_attempt_index_sha256": _sha(parent_output),
                 "parent_fit_lock_sha256": "a" * 64,
                 "outputs": [str(Path(self.temp.name) / ".private" / "new" / "REAL_CONTROL_V2_INDEX.json")],
                 "argv": [sys.executable, "worker.py", "--attempt", str(parent_output.parent),
                          "--queue-root", str(queue.root)]}
        self.assertIsNone(queue._v2_parent_reason(queue.status(), child))
        changed_lock = json.loads(self.lock.read_text())
        changed_lock["parent_queue_receipt_sha256_by_unit"][parent_id] = "0" * 64
        self.lock.write_text(json.dumps(changed_lock))
        self.assertEqual(queue._v2_parent_reason(queue.status(), child), "v2_parent_receipt_not_locked")
        lock["parent_queue_receipt_sha256_by_unit"][parent_id] = _sha(parent_receipt)
        self.lock.write_text(json.dumps(lock))
        parent_output.write_text('{"changed":true}\n')
        self.assertEqual(queue._v2_parent_reason(queue.status(), child), "v2_parent_output_changed")


if __name__ == "__main__":
    unittest.main()
