"""Synthetic S4 replay identities across two fit and two V2 control locks.

No Adult role file, candidate implementation, or shared evaluator is imported.
The test intentionally precedes verifier_identity_v3.py under the S3 queue.
"""

import unittest

from verifier_identity_v3 import admit_anchor_replay_identity, admit_b_replay_identity


FIT_OLD = "a" * 64
FIT_NEW = "b" * 64
CONTROL_OLD = "c" * 64
CONTROL_R2 = "d" * 64
ATTEMPT = "e" * 64
ANCHOR = "f" * 64
PROBE = "1" * 64


def verifier():
    return {
        "schema": "pcrl-s4-verifier-admission-v3",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "accepted_fit_lock_sha256": [FIT_OLD, FIT_NEW],
        "accepted_control_v2_lock_sha256": [CONTROL_OLD, CONTROL_R2],
        "unit_specs": {},
    }


def b_case(*, fit_sha=FIT_OLD, control_sha=CONTROL_OLD):
    v = verifier()
    unit = "S4-B-TASK-S0"
    screen = "B-SCREEN-B-TASK-S0-R2"
    probe_unit = "B-V2-REAL-PROBES-B-TASK-S0"
    v["unit_specs"][unit] = {
        "kind": "B_SCREEN", "arm": "B-TASK", "seed": 0,
        "screen_unit_id": screen, "fit_lock_sha256": fit_sha,
        "attempt_index_sha256": ATTEMPT,
        "control_v2_unit_id": probe_unit,
        "control_v2_lock_sha256": control_sha,
        "control_v2_index_sha256": PROBE,
    }
    fit_lock = {
        "schema": "pcrl-method-portfolio-campaign-lock-v1",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE", "fit_admission": "B_ONLY",
        "admitted_arms": {"B": ["B-TASK"]},
        "families": {"B": {"screen_unit_ids": {"B-TASK:s0": screen}}},
    }
    attempt = {
        "schema": "pcrl-team-b-private-attempt-index-v1",
        "status": "FROZEN_FOR_INDEPENDENT_REPLAY", "arm": "B-TASK",
        "seed": 0, "screen_unit_id": screen,
        "campaign_lock_sha256": fit_sha,
    }
    control = {
        "schema": "pcrl-team-b-real-control-index-v2",
        "status": "PASS", "arm": "B-TASK", "seed": 0,
        "queue_unit_id": probe_unit, "screen_unit_id": screen,
        "fit_lock_sha256": fit_sha,
        "control_v2_lock_sha256": control_sha,
        "attempt_index_sha256": ATTEMPT,
        "view_report_sha256": {"income": "2" * 64,
                               "occupation": "3" * 64, "pair": "4" * 64},
        "view_custody_receipt_sha256": {"income": "5" * 64,
                                       "occupation": "6" * 64,
                                       "pair": "7" * 64},
    }
    return v, unit, fit_lock, attempt, control


def check_b(case, *, fit_sha=FIT_OLD, control_sha=CONTROL_OLD,
            attempt_sha=ATTEMPT, probe_sha=PROBE):
    v, unit, fit_lock, attempt, control = case
    return admit_b_replay_identity(
        verifier=v, unit_id=unit, arm="B-TASK", seed=0,
        fit_lock=fit_lock, fit_lock_sha256=fit_sha,
        attempt_index=attempt, attempt_index_sha256=attempt_sha,
        control_v2_index=control, control_v2_index_sha256=probe_sha,
        control_v2_lock_sha256=control_sha)


def anchor_case(*, fit_sha=FIT_NEW, control_sha=CONTROL_R2):
    v = verifier()
    unit = "S4-FEATURE-ANCHOR-S1"
    shared = "S2-FEATURE-REF-RAW83-U-S1"
    probe_unit = shared + "-V2-PROBE-R2"
    v["unit_specs"][unit] = {
        "kind": "FEATURE_ANCHOR", "arm": "FEATURE_REF_RAW83_U",
        "seed": 1, "shared_unit_id": shared,
        "fit_lock_sha256": fit_sha, "anchor_receipt_sha256": ANCHOR,
        "control_v2_unit_id": probe_unit,
        "control_v2_lock_sha256": control_sha,
        "control_v2_receipt_sha256": PROBE,
    }
    fit_lock = {
        "schema": "pcrl-method-portfolio-campaign-lock-v1",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE", "fit_admission": "B_ONLY",
        "admitted_shared_unit_ids": [shared],
    }
    anchor = {
        "schema": "pcrl-feature-ref-raw83-u-anchor-receipt-v1",
        "status": "EXPLORATORY_DEVELOPMENT", "arm": "FEATURE_REF_RAW83_U",
        "seed": 1, "campaign_lock_sha256": fit_sha,
    }
    control = {
        "schema": "pcrl-feature-ref-raw83-u-v2-probe-receipt-v1",
        "status": "PASS", "arm": "FEATURE_REF_RAW83_U", "seed": 1,
        "s3_unit_id": probe_unit, "parent_anchor_receipt_sha256": ANCHOR,
        "parent_campaign_lock_sha256": fit_sha,
        "control_lock_sha256": control_sha,
        "probe_report_sha256": {"r1": "2" * 64,
                                "r2": "3" * 64, "pair": "4" * 64},
        "probe_custody_receipt_sha256": {"r1": "5" * 64,
                                        "r2": "6" * 64,
                                        "pair": "7" * 64},
    }
    return v, unit, fit_lock, anchor, control


def check_anchor(case, *, fit_sha=FIT_NEW, control_sha=CONTROL_R2,
                 anchor_sha=ANCHOR, probe_sha=PROBE):
    v, unit, fit_lock, anchor, control = case
    return admit_anchor_replay_identity(
        verifier=v, unit_id=unit, seed=1,
        fit_lock=fit_lock, fit_lock_sha256=fit_sha,
        anchor_receipt=anchor, anchor_receipt_sha256=anchor_sha,
        control_v2_receipt=control,
        control_v2_receipt_sha256=probe_sha,
        control_v2_lock_sha256=control_sha)


class BIdentityTest(unittest.TestCase):
    def test_two_fit_locks_and_per_unit_control_lock(self):
        self.assertEqual(check_b(b_case())["fit_lock_sha256"], FIT_OLD)
        self.assertEqual(check_b(b_case(fit_sha=FIT_NEW),
                                 fit_sha=FIT_NEW)["fit_lock_sha256"], FIT_NEW)
        self.assertEqual(check_b(b_case(control_sha=CONTROL_R2),
                                 control_sha=CONTROL_R2)["control_v2_lock_sha256"],
                         CONTROL_R2)

    def test_swapped_fit_attempt_control_or_unit_is_refused(self):
        for mutate in (
            lambda x: x[3].update(campaign_lock_sha256=FIT_NEW),
            lambda x: x[3].update(screen_unit_id="B-SCREEN-OTHER"),
            lambda x: x[4].update(attempt_index_sha256=FIT_NEW),
            lambda x: x[4].update(queue_unit_id="B-V2-OTHER"),
            lambda x: x[4].update(control_v2_lock_sha256=CONTROL_R2),
            lambda x: x[0]["unit_specs"][x[1]].update(
                control_v2_index_sha256=FIT_NEW),
        ):
            with self.subTest(mutate=mutate):
                case = b_case()
                mutate(case)
                with self.assertRaises((ValueError, PermissionError)):
                    check_b(case)
        with self.assertRaises((ValueError, PermissionError)):
            check_b(b_case(), fit_sha=FIT_NEW)
        with self.assertRaises((ValueError, PermissionError)):
            check_b(b_case(), control_sha=CONTROL_R2)

    def test_failed_v2_control_keeps_replay_but_blocks_comparison(self):
        case = b_case()
        case[4]["status"] = "FAIL"
        result = check_b(case)
        self.assertEqual(result["v2_probe_status"], "FAIL")
        self.assertFalse(result["comparison_eligible"])


class AnchorIdentityTest(unittest.TestCase):
    def test_r2_anchor_binds_its_own_fit_and_control_lock(self):
        result = check_anchor(anchor_case())
        self.assertEqual(result["fit_lock_sha256"], FIT_NEW)
        self.assertEqual(result["control_v2_lock_sha256"], CONTROL_R2)

    def test_swapped_parent_or_v2_receipt_is_refused(self):
        for mutate in (
            lambda x: x[3].update(campaign_lock_sha256=FIT_OLD),
            lambda x: x[4].update(parent_anchor_receipt_sha256=FIT_OLD),
            lambda x: x[4].update(s3_unit_id="OLD-ANCHOR-V2-UNIT"),
            lambda x: x[4].update(control_lock_sha256=CONTROL_OLD),
            lambda x: x[0]["unit_specs"][x[1]].update(
                control_v2_receipt_sha256=FIT_OLD),
        ):
            with self.subTest(mutate=mutate):
                case = anchor_case()
                mutate(case)
                with self.assertRaises((ValueError, PermissionError)):
                    check_anchor(case)

    def test_failed_anchor_v2_control_blocks_local_cap_comparison(self):
        case = anchor_case()
        case[4]["status"] = "FAIL"
        result = check_anchor(case)
        self.assertEqual(result["v2_probe_status"], "FAIL")
        self.assertFalse(result["comparison_eligible"])


if __name__ == "__main__":
    unittest.main()
