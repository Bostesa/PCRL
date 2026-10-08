"""Small synthetic checks of the prefit D task/distortion matching rule."""

import unittest

import numpy as np

from control_match import (BETAS, METRIC_CONVENTION_ID, MatchObservation, gaps, recheck_inner,
                           select_match, utility_observation)


CAL_SHA = "a" * 64
INNER_SHA = "b" * 64
CONVENTION = METRIC_CONVENTION_ID
FIT_IDS = {b: f"fit-beta-{b}" for b in BETAS}
CAL_IDS = {b: f"cal-beta-{b}" for b in BETAS}


def obs(accuracy=0.8, nll=0.3, brier=0.2, distortion=1.0):
    return MatchObservation(np.tile([accuracy, nll, brier], (2, 1)),
                            np.repeat(distortion, 2))


def select(candidate, grid):
    return select_match(candidate, grid, role="CALIBRATION_HELDOUT",
                        candidate_fit_id="joint-fit", candidate_calibration_id="joint-cal",
                        control_fit_ids=FIT_IDS, control_calibration_ids=CAL_IDS,
                        row_order_sha256=CAL_SHA, metric_convention_id=CONVENTION)


class ControlMatchTests(unittest.TestCase):
    def test_all_utility_boundaries_are_inclusive(self):
        candidate = obs(accuracy=0.0, nll=0.0, brier=0.0)
        boundary = obs(accuracy=0.002, nll=0.002, brier=0.001)
        self.assertTrue(gaps(candidate, boundary)["eligible"])

    def test_no_eligible_arm_does_not_promote_closest_miss(self):
        candidate = obs()
        grid = {b: obs(accuracy=0.803) for b in BETAS}
        result = select(candidate, grid)
        self.assertEqual(result["status"], "NO_MATCH")
        self.assertIsNone(result["chosen_beta"])

    def test_tie_chooses_smaller_beta_with_frozen_ids(self):
        candidate = obs()
        result = select(candidate, {b: obs() for b in BETAS})
        self.assertEqual(result["chosen_beta"], 0.0)
        self.assertEqual(result["selected_control_fit_id"], FIT_IDS[0.0])
        self.assertEqual(result["calibration_row_order_sha256"], CAL_SHA)

    def test_distortion_is_per_purpose_with_fixed_denominator_floor(self):
        candidate = MatchObservation(np.tile([0.8, 0.3, 0.2], (2, 1)), [0.0, 1.0])
        within = MatchObservation(candidate.utility, [0.0, 1.05])
        outside = MatchObservation(candidate.utility, [0.0, 1.11])
        self.assertTrue(gaps(candidate, within)["eligible"])
        self.assertFalse(gaps(candidate, outside)["eligible"])

    def test_cal_only_and_inner_frozen_identity(self):
        candidate = obs()
        result = select(candidate, {b: obs() for b in BETAS})
        with self.assertRaises(ValueError):
            select_match(candidate, {b: obs() for b in BETAS}, role="INNER_SELECTION",
                         candidate_fit_id="joint-fit", candidate_calibration_id="joint-cal",
                         control_fit_ids=FIT_IDS, control_calibration_ids=CAL_IDS,
                         row_order_sha256=INNER_SHA, metric_convention_id=CONVENTION)
        kwargs = dict(role="INNER_SELECTION", frozen_receipt=result,
                      observed_control_fit_id=FIT_IDS[0.0],
                      observed_control_calibration_id=CAL_IDS[0.0],
                      candidate_fit_id="joint-fit", candidate_calibration_id="joint-cal",
                      row_order_sha256=INNER_SHA, metric_convention_id=CONVENTION)
        self.assertTrue(recheck_inner(candidate, obs(), **kwargs)["match_persists"])
        with self.assertRaises(ValueError):
            recheck_inner(candidate, obs(), **(kwargs | {"observed_control_fit_id": "wrong"}))
        with self.assertRaises(ValueError):
            recheck_inner(candidate, obs(), **(kwargs | {"role": "CALIBRATION_HELDOUT"}))
        with self.assertRaises(ValueError):
            recheck_inner(candidate, obs(), **(kwargs | {"metric_convention_id": "other"}))

    def test_multiclass_brier_is_class_sum(self):
        y = (np.array([0], dtype=np.int64), np.array([1], dtype=np.int64))
        p = (np.array([[0.8, 0.2]]), np.array([[0.2, 0.8]]))
        h = (np.array([[1.0]]), np.array([[1.0]]))
        z = (np.array([[0.0]]), np.array([[0.0]]))
        observed = utility_observation(y, p, h, z)
        np.testing.assert_allclose(observed.utility[:, 2], [0.08, 0.08])
        np.testing.assert_allclose(observed.utility[:, 1], [-np.log(0.8)] * 2)
        np.testing.assert_allclose(observed.distortion, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
