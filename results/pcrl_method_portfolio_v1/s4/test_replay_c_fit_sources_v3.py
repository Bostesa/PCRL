"""Data-free RED for C R3 fit-stage-specific source provenance."""

from __future__ import annotations

import unittest

from replay_c_attempt_v3 import admit_fit_role_hashes


BASE = (
    "OSF_DEFENSE_FIT:features",
    "OSF_DEFENSE_FIT:income",
    "OSF_DEFENSE_FIT:occupation",
    "CALIBRATION_HELDOUT:features",
    "CALIBRATION_HELDOUT:income",
    "CALIBRATION_HELDOUT:occupation",
)
SEX = "OSF_DEFENSE_FIT:sex"


class StageBoundFitSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest_hashes = {
            key: f"{i + 1:064x}" for i, key in enumerate((*BASE, SEX))
        }

    def test_privacy_fit_requires_manifest_bound_sex(self) -> None:
        expected = dict(self.manifest_hashes)
        self.assertEqual(admit_fit_role_hashes(
            self.manifest_hashes, "privacy_fit_cal_batch", expected), expected)
        changed = dict(expected)
        changed.pop(SEX)
        with self.assertRaisesRegex(ValueError, "fit source"):
            admit_fit_role_hashes(self.manifest_hashes,
                                  "privacy_fit_cal_batch", changed)
        changed = dict(expected)
        changed[SEX] = "f" * 64
        with self.assertRaisesRegex(ValueError, "fit source"):
            admit_fit_role_hashes(self.manifest_hashes,
                                  "privacy_fit_cal_batch", changed)

    def test_both_task_only_stages_require_exact_six_manifest_sources(self) -> None:
        expected = {key: self.manifest_hashes[key] for key in BASE}
        for stage in ("task_only_fit_cal", "task_pair_fit_cal"):
            with self.subTest(stage=stage):
                self.assertEqual(admit_fit_role_hashes(
                    self.manifest_hashes, stage, expected), expected)
                changed = dict(expected)
                changed[SEX] = self.manifest_hashes[SEX]
                with self.assertRaisesRegex(ValueError, "fit source"):
                    admit_fit_role_hashes(self.manifest_hashes, stage, changed)
                changed = dict(expected)
                changed["OSF_DEFENSE_FIT:income"] = "f" * 64
                with self.assertRaisesRegex(ValueError, "fit source"):
                    admit_fit_role_hashes(self.manifest_hashes, stage, changed)

    def test_unknown_stage_or_missing_manifest_hash_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "fit stage"):
            admit_fit_role_hashes(self.manifest_hashes, "unapproved_fit", {})
        missing = dict(self.manifest_hashes)
        missing.pop("CALIBRATION_HELDOUT:occupation")
        with self.assertRaisesRegex(ValueError, "fit source"):
            admit_fit_role_hashes(missing, "task_pair_fit_cal",
                                  {key: self.manifest_hashes[key] for key in BASE})


if __name__ == "__main__":
    unittest.main()
