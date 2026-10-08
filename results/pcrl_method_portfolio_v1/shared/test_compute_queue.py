"""Lightweight behavior checks for the portfolio's local compute queue."""

import fcntl
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from compute_queue import Queue


class ComputeQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.queue = Queue(self.root)
        self.marker = self.root / "ran.txt"
        self.spec = {
            "unit_id": "synthetic_smoke_s0",
            "argv": [sys.executable, "-c", f"from pathlib import Path; Path({str(self.marker)!r}).write_text('ok')"],
            "cwd": str(self.root),
            "deps": [],
            "max_wall_s": 4,
            "cpu_reserve_s": 4,
            "max_rss_gib": 0.25,
            "stage": "setup",
        }
        self.queue.enqueue(self.spec)

    def test_gate_is_closed_until_all_admissions_are_recorded(self):
        self.assertEqual(self.queue.run_once()["status"], "gate_closed")
        self.assertFalse(self.marker.exists())
        self.queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True, external_heavy=2)
        self.assertEqual(self.queue.run_once()["status"], "external_capacity_full")
        self.assertFalse(self.marker.exists())

    def test_unit_runs_once_with_receipt_and_complete_marker(self):
        self.queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True, external_heavy=0)
        result = self.queue.run_once()
        self.assertEqual(result["status"], "done")
        self.assertEqual(self.marker.read_text(), "ok")
        self.assertEqual(self.queue.run_once()["status"], "empty")
        state = self.queue.status()
        self.assertEqual(state["units"]["synthetic_smoke_s0"]["status"], "done")
        self.assertGreater(state["used_cpu_s"], 0)
        receipts = [json.loads(line) for line in (self.root / "RECEIPTS.jsonl").read_text().splitlines()]
        self.assertEqual([r["event"] for r in receipts if r.get("unit_id") == "synthetic_smoke_s0"],
                         ["enqueue", "start", "finish"])

    def test_both_flock_slots_prevent_an_unregistered_third_job(self):
        self.queue.update_gate(safe_inputs=True, lock_pushed=True, occupancy_confirmed=True, external_heavy=0)
        handles = [(self.root / f"slot{i}.lock").open("a+") for i in (0, 1)]
        self.addCleanup(lambda: [h.close() for h in handles])
        for handle in handles:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertEqual(self.queue.run_once()["status"], "portfolio_capacity_full")
        self.assertFalse(self.marker.exists())

    def test_prelock_synthetic_unit_requires_registered_law_and_test_hash(self):
        law = self.root / "law.json"
        law.write_text('{"fixed": true}')
        test_file = self.root / "test_tiny.py"
        test_file.write_text("# frozen synthetic-only test\n")
        law_hash = hashlib.sha256(law.read_bytes()).hexdigest()
        test_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()
        lock = self.root / "PHASE1_LOCK.json"
        lock.write_text(json.dumps({
            "shared_laws": {"path": "law.json", "sha256": law_hash},
            "pre_algorithm_tests": {"team_c": {"path": "test_tiny.py", "sha256": test_hash}},
        }))
        spec = dict(self.spec, unit_id="synthetic_c", mode="prelock_synthetic",
                    team="team_c", data_access="NONE", test_file=str(test_file),
                    test_sha256=test_hash)
        self.queue.enqueue(spec)
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, phase1_lock_path=str(lock),
                               phase1_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
        result = self.queue.run_once()
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["unit_id"], "synthetic_c")
        self.assertEqual(self.queue.status()["units"]["synthetic_smoke_s0"]["status"], "planned")

    def test_prelock_synthetic_unit_rejects_unregistered_test(self):
        law = self.root / "law.json"
        law.write_text("{}")
        lock = self.root / "PHASE1_LOCK.json"
        lock.write_text(json.dumps({"shared_laws": {"path": "law.json", "sha256": hashlib.sha256(law.read_bytes()).hexdigest()},
                                    "pre_algorithm_tests": {}}))
        test_file = self.root / "unregistered.py"
        test_file.write_text("# not locked\n")
        self.queue.enqueue(dict(self.spec, unit_id="bad_synthetic", mode="prelock_synthetic",
                                team="team_c", data_access="NONE", test_file=str(test_file),
                                test_sha256=hashlib.sha256(test_file.read_bytes()).hexdigest()))
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, phase1_lock_path=str(lock),
                               phase1_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
        self.assertEqual(self.queue.run_once()["status"], "phase1_test_unregistered")
        self.assertFalse(self.marker.exists())

    def test_legacy_unqueued_checks_are_charged_once(self):
        self.queue.record_deviation("prequeue-light-tests", 180, "synthetic engineering checks")
        self.queue.record_deviation("prequeue-light-tests", 180, "synthetic engineering checks")
        state = self.queue.status()
        self.assertEqual(state["used_cpu_s"], 180)
        with self.assertRaises(ValueError):
            self.queue.record_deviation("prequeue-light-tests", 181, "changed charge")

    def test_prelock_accepts_one_of_multiple_registered_team_tests(self):
        law = self.root / "law.json"
        law.write_text("{}")
        first = self.root / "test_first.py"
        first.write_text("# first\n")
        second = self.root / "other_worktree" / "results" / "pcrl_method_portfolio_v1" / "teams" / "b" / "test_second.py"
        second.parent.mkdir(parents=True)
        second.write_text("# second\n")
        lock = self.root / "PHASE1_LOCK_AMENDMENT.json"
        lock.write_text(json.dumps({
            "shared_laws": {"path": "law.json", "sha256": hashlib.sha256(law.read_bytes()).hexdigest()},
            "pre_algorithm_tests": {"team_b": [
                {"path": "test_first.py", "sha256": hashlib.sha256(first.read_bytes()).hexdigest()},
                {"path": "results/pcrl_method_portfolio_v1/teams/b/test_second.py", "sha256": hashlib.sha256(second.read_bytes()).hexdigest()},
            ]},
        }))
        self.queue.enqueue(dict(self.spec, unit_id="b_second", mode="prelock_synthetic",
                                team="team_b", data_access="NONE", test_file=str(second),
                                test_sha256=hashlib.sha256(second.read_bytes()).hexdigest()))
        self.queue.update_gate(safe_inputs=False, lock_pushed=False, occupancy_confirmed=True,
                               external_heavy=0, phase1_lock_path=str(lock),
                               phase1_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
        self.assertEqual(self.queue.run_once()["status"], "done")


if __name__ == "__main__":
    unittest.main()
