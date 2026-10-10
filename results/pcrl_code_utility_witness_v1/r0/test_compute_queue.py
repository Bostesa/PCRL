"""Data-free admission checks for the focused campaign queue."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from results.pcrl_code_utility_witness_v1.r0.compute_queue import (
    CPU_LIMIT,
    Queue,
    WITNESS_RESERVE,
)


class QueueAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.queue = Queue(self.root / "queue")
        self.seal = self.root / "seal.json"
        self.seal.write_text("{}\n")
        self.queue.open(self.seal, hashlib.sha256(self.seal.read_bytes()).hexdigest())

    def tearDown(self):
        self.temp.cleanup()

    def spec(self, stage):
        return {
            "stage": stage,
            "max_wall_s": 30,
            "cpu_reserve_s": 30,
            "projected_output_bytes": 0,
            "sources": [{"path": str(self.seal),
                         "sha256": hashlib.sha256(self.seal.read_bytes()).hexdigest()}],
            "deps": [],
            "max_rss_gib": 1,
        }

    def test_protection_requires_witness_pass(self):
        self.assertEqual(self.queue._admit_reason(self.queue.read(), self.spec("protection")),
                         "learned_code_witness_gate")

    def test_channel_keeps_witness_and_other_reserves(self):
        state = self.queue.read()
        state["milestones"]["channel_identity_pass"] = {
            "path": str(self.seal),
            "sha256": hashlib.sha256(self.seal.read_bytes()).hexdigest(),
        }
        state["used_cpu_s"] = CPU_LIMIT - WITNESS_RESERVE
        self.assertEqual(self.queue._admit_reason(state, self.spec("channel")),
                         "cpu_budget_or_protected_reserve")

    def test_witness_stage_is_admitted_under_floors(self):
        self.assertIsNone(self.queue._admit_reason(self.queue.read(), self.spec("witness")))

    def test_no_fresh_occupancy_claim_means_no_heavy_slot(self):
        self.assertEqual(self.queue._capacity(self.queue.read()), 0)

    def test_channel_fit_requires_identity_gate(self):
        self.assertEqual(self.queue._admit_reason(self.queue.read(), self.spec("channel")),
                         "channel_identity_gate")

    def test_invalid_witness_receipt_cannot_open_protection_gate(self):
        receipt = self.root / "wrong-witness.json"
        receipt.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "does not establish gate"):
            self.queue.mark("witness_pass", receipt)


if __name__ == "__main__":
    unittest.main()
