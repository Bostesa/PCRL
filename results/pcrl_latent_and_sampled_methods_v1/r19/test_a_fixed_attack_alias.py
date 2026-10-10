"""A corrected physical attack may fill one old scientific cell, not erase attempts."""

import unittest

from a_fixed_attack_alias import attach_aliases


def queue_row(uid, status, digit):
    return {"unit_id": uid, "stage": "a", "status": status,
            "charged_cpu_s": "1.0", "wall_s": "1.0",
            "sampled_tree_rss_peak_bytes": "100",
            "receipt_sha256": digit * 64}


class FixedAttackAliasTests(unittest.TestCase):
    def setUp(self):
        self.old_failed = "A-REFERENCE-ATTACK-P_STAR-S0"
        self.old_unqueued = "A-REFERENCE-ATTACK-Ucal-S1"
        self.old_done = "A-DIAG-ATTACK-A_CLASS_ONLY-S0"
        self.retry_failed = self.old_failed + "-R1"
        self.retry_unqueued = self.old_unqueued + "-R1"
        self.rows = [{"unit_id": self.old_failed, "kind": "reference_attack", "status": "failed"},
                     {"unit_id": self.old_unqueued, "kind": "reference_attack", "status": "PLANNED_NOT_ENQUEUED"},
                     {"unit_id": self.old_done, "kind": "diagnostic_attack", "status": "done"}]
        self.queue = {self.old_failed: queue_row(self.old_failed, "failed", "a"),
                      self.retry_failed: queue_row(self.retry_failed, "done", "b"),
                      self.retry_unqueued: queue_row(self.retry_unqueued, "done", "c"),
                      self.old_done: queue_row(self.old_done, "done", "d")}
        self.correction = {"physical_attempts": {
            self.retry_failed: {"original_attack_unit_id": self.old_failed,
                                "original_attempt_status": "FAILED_TECHNICAL_PRESERVED"},
            self.retry_unqueued: {"original_attack_unit_id": self.old_unqueued,
                                  "original_attempt_status": "NOT_ENQUEUED_SOURCE_CORRECTION"}},
            "old_attempt_failed_queue_receipt_sha256": {self.old_failed: "a" * 64}}
        self.completion = {"status": "ALL_18_SCIENTIFIC_CELLS_HAVE_COMPLETE_PHYSICAL_ATTACK",
                           "scientific_cell_count": 3, "corrected_physical_done_count": 2,
                           "original_physical_done_count": 1,
                           "original_failed_preserved_count": 1,
                           "original_unqueued_superseded_count": 1,
                           "cells": [
                               self.cell(self.old_failed, self.retry_failed, "FAILED_TECHNICAL_PRESERVED", "b"),
                               self.cell(self.old_unqueued, self.retry_unqueued, "NOT_ENQUEUED_SOURCE_CORRECTION", "c"),
                               self.cell(self.old_done, self.old_done, "SAME_PHYSICAL_ID", "d"),
                           ]}
        self.outputs = {self.retry_failed: {"e" * 64},
                        self.retry_unqueued: {"e" * 64}, self.old_done: {"e" * 64}}

    @staticmethod
    def cell(scientific, physical, original_status, queue_digit):
        return {"scientific_cell_id": scientific, "physical_unit_id": physical,
                "original_physical_status": original_status,
                "queue_receipt_sha256": queue_digit * 64,
                "attack_index_sha256": "e" * 64,
                "control_status": {"r1": "PASS", "r2": "PASS", "pair": "PASS"}}

    def test_retries_fill_logical_cells_and_preserve_old_failed_attempt_charge(self):
        logical, attempts = attach_aliases(self.rows, self.queue, self.correction,
                                           self.completion, self.outputs,
                                           expected_aliases=2, expected_cells=3,
                                           expected_old_failed=1, expected_old_unqueued=1)
        by_id = {row["unit_id"]: row for row in logical}
        self.assertEqual(by_id[self.old_failed]["status"], "COMPLETE_VIA_R1_ALIAS")
        self.assertEqual(by_id[self.old_failed]["original_attempt_status"], "FAILED_TECHNICAL_PRESERVED")
        self.assertEqual(by_id[self.old_failed]["original_queue_receipt_sha256"], "a" * 64)
        self.assertEqual(by_id[self.old_failed]["physical_unit_id"], self.retry_failed)
        self.assertEqual(by_id[self.old_unqueued]["original_attempt_status"], "NOT_ENQUEUED_SOURCE_CORRECTION")
        self.assertEqual(by_id[self.old_done]["status"], "done")
        self.assertEqual(len(attempts), 4)
        self.assertEqual({row["attempt_status"] for row in attempts}, {"done", "failed"})
        self.assertEqual(sum(row["attempt_status"] == "failed" for row in attempts), 1)

    def test_control_failure_cannot_be_reported_complete(self):
        self.completion["cells"][0]["control_status"]["pair"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "fixed attack controls"):
            attach_aliases(self.rows, self.queue, self.correction, self.completion,
                           self.outputs, expected_aliases=2, expected_cells=3,
                           expected_old_failed=1, expected_old_unqueued=1)

    def test_missing_retry_receipt_rejects_alias(self):
        self.queue.pop(self.retry_unqueued)
        with self.assertRaisesRegex(ValueError, "physical queue receipt"):
            attach_aliases(self.rows, self.queue, self.correction, self.completion,
                           self.outputs, expected_aliases=2, expected_cells=3,
                           expected_old_failed=1, expected_old_unqueued=1)


if __name__ == "__main__":
    unittest.main()
