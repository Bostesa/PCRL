"""Isolated control tests; no campaign job or real data is run here."""
from __future__ import annotations

import hashlib
import datetime as dt
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from compute_queue import Queue, atomic, digest


class QueueControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = Queue(self.root / "queue")
        self.seal = self.root / "seal.json"
        self.seal.write_text('{"effective": true}\n')

    def spec(self, uid: str, sleep: float = 0) -> Path:
        spec = {
            "unit_id": uid,
            "stage": "d_first",
            "argv": [sys.executable, "-c", f"import os,time; print(os.environ['OMP_NUM_THREADS']); time.sleep({sleep})"],
            "cwd": str(self.root),
            "max_wall_s": 5,
            "cpu_reserve_s": 5,
            "max_rss_gib": 0.25,
            "projected_output_bytes": 100,
            "sources": [{"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))}],
        }
        target = self.root / f"{uid}.json"
        target.write_text(json.dumps(spec))
        return target

    def test_seal_admission_and_resource_receipt(self):
        self.queue.enqueue(self.spec("D_TEST"))
        self.assertEqual(self.queue.run_once()["reason"], "protocol_seal_missing")
        self.queue.open(self.seal, digest(self.seal))
        result = self.queue.run_once()
        self.assertEqual(result["status"], "done")
        receipt = json.loads(Path(result["receipt"]).read_text())
        self.assertGreater(receipt["charged_cpu_s"], 0)
        self.assertIsInstance(receipt["child_pid"], int)
        self.assertEqual(receipt["returncode"], 0)
        self.assertIn("1", (self.root / "queue" / "units" / "D_TEST" / "stdout.log").read_text())
        self.assertEqual(self.queue.status()["capacity_now"], 1)

    def test_one_slot_and_milestone_gate(self):
        self.queue.open(self.seal, digest(self.seal))
        self.queue.enqueue(self.spec("D_SLEEP", 1.0))
        worker = subprocess.Popen([sys.executable, str(Path(__file__).with_name("compute_queue.py")),
                                   "--root", str(self.root / "queue"), "run-once"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for _ in range(80):
                if self.queue.status()["units"]["D_SLEEP"]["status"] == "running":
                    break
                time.sleep(0.02)
            self.assertEqual(self.queue.run_once()["status"], "capacity_full")
        finally:
            stdout, stderr = worker.communicate(timeout=10)
        self.assertEqual(worker.returncode, 0, stderr.decode())
        self.assertEqual(json.loads(stdout)["status"], "done")
        a_spec = json.loads(self.spec("A_TEST").read_text())
        a_spec["stage"] = "a"
        path = self.root / "A_TEST.json"
        path.write_text(json.dumps(a_spec))
        self.queue.enqueue(path)
        self.assertEqual(self.queue.run_once()["reason"], "d_and_verification_first")

    def test_immutable_cache(self):
        source = self.root / "source.bin"
        source.write_bytes(b"private fixture")
        first = self.queue.cache_put(source)
        source.write_bytes(b"changed fixture")
        self.assertEqual(Path(first["path"]).read_bytes(), b"private fixture")
        self.assertEqual(self.queue.cache_put(source)["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_owned_descendant_is_in_receipt(self):
        self.queue.open(self.seal, digest(self.seal))
        path = self.spec("D_CHILD")
        spec = json.loads(path.read_text())
        spec["argv"] = [sys.executable, "-c",
                        "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(1.3)']); p.wait()"]
        path.write_text(json.dumps(spec))
        self.queue.enqueue(path)
        result = self.queue.run_once()
        self.assertEqual(result["status"], "done")
        receipt = json.loads(Path(result["receipt"]).read_text())
        self.assertTrue(receipt["descendants_visible"])
        self.assertGreaterEqual(len(receipt["observed_tree_pids"]), 2)

    def test_append_only_amendment_requires_new_unit_identity(self):
        self.queue.open(self.seal, digest(self.seal))
        amendment = self.root / "2026-10-10-amendment.json"
        amendment.write_text('{"scope":"d_first"}\n')
        value = self.queue.amend(amendment, digest(amendment), dt.datetime.now(dt.timezone.utc).isoformat(), "d_first")
        self.queue.enqueue(self.spec("D_OLD"))
        self.assertEqual(self.queue.run_once()["reason"], "affected_stage_requires_latest_amendment_sha256")
        path = self.spec("D_NEW")
        spec = json.loads(path.read_text())
        spec["amendment_sha256"] = value["sha256"]
        path.write_text(json.dumps(spec))
        self.queue.enqueue(path)
        self.assertEqual(self.queue.run_once()["status"], "done")
        self.assertEqual(self.queue.status()["seal"]["sha256"], digest(self.seal))

    def test_external_cpu_charge_is_idempotent(self):
        receipt = self.root / "light.json"
        receipt.write_text('{"basis":"conservative"}\n')
        self.queue.charge("LIGHT", "admission", 60.0, receipt)
        self.queue.charge("LIGHT", "admission", 60.0, receipt)
        self.assertEqual(self.queue.status()["used_cpu_s"], 60.0)

    def test_orphan_blocks_until_process_checked_and_charged(self):
        self.queue.open(self.seal, digest(self.seal))
        self.queue.enqueue(self.spec("D_ORPHAN"))
        self.queue.enqueue(self.spec("D_NEXT"))
        with self.queue.mutex():
            state = self.queue.read()
            state["units"]["D_ORPHAN"].update(status="running", slot=0,
                                               wrapper_pid=99999999, child_pid=99999998)
            atomic(self.queue.state, state)
        self.assertEqual(self.queue.run_once()["status"], "orphan_unresolved")
        recovered = self.queue.reconcile("D_ORPHAN")
        self.assertEqual(recovered["status"], "incomplete")
        self.assertEqual(self.queue.status()["used_cpu_s"], 5.0)
        self.assertEqual(self.queue.run_once()["status"], "done")


if __name__ == "__main__":
    unittest.main()
