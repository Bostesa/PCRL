"""Synthetic identity checks for independent B replay across two fit locks.

No Adult role file, candidate implementation, or shared evaluator is imported.
The test is intentionally registered before the identity implementation.
"""

import unittest

from verifier_identity_v2 import admit_b_replay_identity


OLD = "a" * 64
NEW = "b" * 64
CONTROL = "c" * 64
ATTEMPT = "d" * 64
PROBE = "e" * 64


def fixture(fit_sha=OLD):
    arm, seed, unit = "B-TASK", 0, "S4-B-TASK-S0"
    screen = "B-SCREEN-B-TASK-S0-R2"
    probe_unit = "B-V2-REAL-PROBES-B-TASK-S0"
    verifier = {
        "schema": "pcrl-s4-verifier-admission-v2",
        "status": "LOCAL_GIT_SEALED_EFFECTIVE",
        "accepted_fit_lock_sha256": [OLD, NEW],
        "control_v2_lock_sha256": CONTROL,
        "unit_specs": {unit: {
            "kind": "B_SCREEN", "arm": arm, "seed": seed,
            "screen_unit_id": screen, "fit_lock_sha256": fit_sha,
            "attempt_index_sha256": ATTEMPT,
            "control_v2_unit_id": probe_unit,
            "control_v2_index_sha256": PROBE,
        }},
    }
    fit_lock = {"schema": "pcrl-method-portfolio-campaign-lock-v1",
                "status": "LOCAL_GIT_SEALED_EFFECTIVE",
                "fit_admission": "B_ONLY",
                "admitted_arms": {"B": [arm]},
                "families": {"B": {"screen_unit_ids": {f"{arm}:s{seed}": screen}}}}
    attempt = {"status": "FROZEN_FOR_INDEPENDENT_REPLAY",
               "arm": arm, "seed": seed, "screen_unit_id": screen,
               "campaign_lock_sha256": fit_sha}
    v2 = {"schema": "pcrl-team-b-real-control-index-v2",
          "status": "PASS", "arm": arm, "seed": seed,
          "queue_unit_id": probe_unit,
          "screen_unit_id": screen, "fit_lock_sha256": fit_sha,
          "control_v2_lock_sha256": CONTROL,
          "attempt_index_sha256": ATTEMPT,
          "view_report_sha256": {view: digit * 64 for view, digit in
                                  (("income", "1"), ("occupation", "2"),
                                   ("pair", "3"))},
          "view_custody_receipt_sha256": {view: digit * 64 for view, digit in
                                          (("income", "4"), ("occupation", "5"),
                                           ("pair", "6"))}}
    return verifier, unit, fit_lock, attempt, v2


def check(values, *, fit_sha=OLD, attempt_sha=ATTEMPT, v2_sha=PROBE):
    verifier, unit, fit_lock, attempt, v2 = values
    return admit_b_replay_identity(
        verifier=verifier, unit_id=unit, arm="B-TASK", seed=0,
        fit_lock=fit_lock, fit_lock_sha256=fit_sha,
        attempt_index=attempt, attempt_index_sha256=attempt_sha,
        control_v2_index=v2, control_v2_index_sha256=v2_sha)


class HistoricalFitAndV2IdentityTest(unittest.TestCase):
    def test_old_and_amended_fit_locks_are_separately_admitted(self):
        old = fixture(OLD)
        new = fixture(NEW)
        self.assertEqual(check(old)["fit_lock_sha256"], OLD)
        self.assertEqual(check(new, fit_sha=NEW)["fit_lock_sha256"], NEW)

    def test_swapped_parent_or_probe_is_refused(self):
        base = fixture()
        for mutate in (
            lambda x: x[3].update(campaign_lock_sha256=NEW),
            lambda x: x[3].update(screen_unit_id="B-SCREEN-OTHER"),
            lambda x: x[4].update(attempt_index_sha256=NEW),
            lambda x: x[4].update(queue_unit_id="B-V2-OTHER"),
            lambda x: x[4].update(control_v2_lock_sha256=NEW),
            lambda x: x[0]["unit_specs"][x[1]].update(control_v2_index_sha256=NEW),
        ):
            with self.subTest(mutate=mutate):
                values = fixture()
                mutate(values)
                with self.assertRaises((ValueError, PermissionError)):
                    check(values)
        with self.assertRaises((ValueError, PermissionError)):
            check(base, fit_sha=NEW)

    def test_v2_is_mandatory_and_v1_does_not_grant_admission(self):
        values = fixture()
        values[4].clear()
        with self.assertRaises((ValueError, PermissionError)):
            check(values)
        values = fixture()
        values[4]["status"] = "FAIL"
        result = check(values)
        self.assertEqual(result["v2_probe_status"], "FAIL")
        self.assertFalse(result["comparison_eligible"])


if __name__ == "__main__":
    unittest.main()
