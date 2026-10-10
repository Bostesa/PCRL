"""The A planned-unit exporter preserves a null parent's unrun descendants."""

import unittest

from a_220_status import build_rows


PARENT = "A-FIT-A_JOINT_STOCH-S0-R1"
QUEUE_SHA = "a" * 64
FIT_SHA = "b" * 64


class A220StatusTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            PARENT: {"kind": "fit", "arm_id": "A_JOINT_STOCH", "seed": 0},
            "A-RELEASE-CHILD": {"kind": "release", "arm_id": "A_JOINT_STOCH", "seed": 0,
                                "fit_unit_id": PARENT, "deps": [PARENT]},
            "A-Q-CHILD": {"kind": "q_preprocess", "arm_id": "A_JOINT_STOCH", "seed": 0,
                          "fit_unit_id": PARENT, "deps": [PARENT]},
            "A-ATTACK-CHILD": {"kind": "attack", "arm_id": "A_JOINT_STOCH", "seed": 0,
                               "fit_unit_id": PARENT, "deps": [PARENT, "A-RELEASE-CHILD", "A-Q-CHILD"]},
            "A-FIXED-CONTROL": {"kind": "diagnostic_release", "arm_id": "A_TASK_DET", "seed": 0},
            "A-FIT-SEQ21": {"kind": "fit", "arm_id": "A_SEQ21", "seed": 0},
            "A-SEQ21-RELEASE": {"kind": "release", "arm_id": "A_SEQ21", "seed": 0,
                                "fit_unit_id": "A-FIT-SEQ21", "deps": ["A-FIT-SEQ21"]},
        }
        self.queue = {PARENT: {"unit_id": PARENT, "stage": "a", "status": "done",
                               "charged_cpu_s": "14.728", "wall_s": "14.728",
                               "sampled_tree_rss_peak_bytes": "484016128",
                               "receipt_sha256": QUEUE_SHA},
                      "A-FIT-SEQ21": {"unit_id": "A-FIT-SEQ21", "stage": "a", "status": "failed",
                                      "charged_cpu_s": "8.73", "wall_s": "8.73",
                                      "sampled_tree_rss_peak_bytes": "380698624",
                                      "receipt_sha256": "d" * 64}}
        self.checkpoint = {"unit_id": PARENT, "scientific_status": "COMPLETE_NO_FEASIBLE_POLICY",
                           "selected_policy_present": False, "feasible_checkpoint_count": 0,
                           "checkpoint_count": 26, "start_count": 2,
                           "private_queue_receipt_sha256": QUEUE_SHA,
                           "private_fit_receipt_sha256": FIT_SHA}
        self.closeout = {"fit_plan_count": 2, "fit_done_rc0_count": 1,
                         "fit_failed_rc1_count": 1, "selected_policy_count": 0,
                         "rows": [
                             {"unit_id": PARENT, "queue_status": "done", "returncode": 0,
                              "scientific_status": "COMPLETE_NO_FEASIBLE_POLICY",
                              "selected_policy_present": False,
                              "queue_receipt_sha256": QUEUE_SHA,
                              "private_fit_receipt_sha256": FIT_SHA},
                             {"unit_id": "A-FIT-SEQ21", "queue_status": "failed", "returncode": 1,
                              "scientific_status": "INCOMPLETE_RESOURCE",
                              "selected_policy_present": False,
                              "queue_receipt_sha256": "d" * 64,
                              "private_fit_receipt_sha256": "e" * 64},
                         ]}

    def test_null_parent_marks_only_its_absent_release_q_attack_children_not_triggered(self):
        rows = build_rows(self.plan, self.queue, self.checkpoint, self.closeout,
                          expected_fit_count=2, expected_null_children=3, expected_blocked_children=1)
        by_id = {row["unit_id"]: row for row in rows}
        self.assertEqual(len(rows), 7)
        self.assertEqual(by_id[PARENT]["status"], "done")
        self.assertEqual(by_id[PARENT]["scientific_status"], "COMPLETE_NO_FEASIBLE_POLICY")
        self.assertEqual(by_id[PARENT]["fit_receipt_sha256"], FIT_SHA)
        for uid in ("A-RELEASE-CHILD", "A-Q-CHILD", "A-ATTACK-CHILD"):
            self.assertEqual(by_id[uid]["status"], "NOT_TRIGGERED_PARENT_NO_FEASIBLE_POLICY")
            self.assertEqual(by_id[uid]["fit_unit_id"], PARENT)
            self.assertEqual(by_id[uid]["charged_cpu_s"], "")
        self.assertEqual(by_id["A-FIXED-CONTROL"]["status"], "PLANNED_NOT_ENQUEUED")
        self.assertEqual(by_id["A-FIT-SEQ21"]["scientific_status"], "INCOMPLETE_RESOURCE")
        self.assertEqual(by_id["A-SEQ21-RELEASE"]["status"], "BLOCKED_PARENT_TECHNICAL_INCOMPLETE")

    def test_queued_child_conflicts_with_no_selected_policy(self):
        self.queue["A-RELEASE-CHILD"] = {"unit_id": "A-RELEASE-CHILD", "stage": "a",
                                          "status": "done", "receipt_sha256": "c" * 64}
        with self.assertRaisesRegex(ValueError, "null-parent descendant was queued"):
            build_rows(self.plan, self.queue, self.checkpoint, self.closeout,
                       expected_fit_count=2, expected_null_children=3, expected_blocked_children=1)

    def test_parent_receipt_mismatch_rejects_export(self):
        self.checkpoint["private_queue_receipt_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "parent queue receipt mismatch"):
            build_rows(self.plan, self.queue, self.checkpoint, self.closeout,
                       expected_fit_count=2, expected_null_children=3, expected_blocked_children=1)

    def test_checkpoint_count_change_rejects_exact_first_fit_claim(self):
        self.checkpoint["checkpoint_count"] = 25
        with self.assertRaisesRegex(ValueError, "unverified null-policy checkpoint"):
            build_rows(self.plan, self.queue, self.checkpoint, self.closeout,
                       expected_fit_count=2, expected_null_children=3, expected_blocked_children=1)

    def test_failed_seq21_cannot_be_reported_as_completed_scientific_null(self):
        self.closeout["rows"][1]["scientific_status"] = "COMPLETE_NO_FEASIBLE_POLICY"
        with self.assertRaisesRegex(ValueError, "fit closeout status mismatch"):
            build_rows(self.plan, self.queue, self.checkpoint, self.closeout,
                       expected_fit_count=2, expected_null_children=3, expected_blocked_children=1)


if __name__ == "__main__":
    unittest.main()
