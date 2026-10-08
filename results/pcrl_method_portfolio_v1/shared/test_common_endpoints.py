"""Synthetic-only registered margin order and paired-person refusal."""

import unittest

import numpy as np

from common_endpoints import calculate_margins, margin_names


class CommonEndpointTest(unittest.TestCase):
    def test_all_utility_and_every_control_margin(self):
        n = 8
        rid = np.arange(n, dtype=np.int64)
        unit = rid.copy()
        y1 = np.array([0, 1] * 4, dtype=np.int64)
        y2 = np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int64)
        sex = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        p1 = np.column_stack((np.where(y1 == 0, .8, .2), np.where(y1 == 1, .8, .2)))
        p2 = .2 * np.ones((n, 6))
        p2[np.arange(n), y2] = .0
        p2 /= p2.sum(axis=1, keepdims=True)
        scores = {view: {"LR_C1": np.linspace(.1, .9, n)} for view in ("r1", "r2", "pair")}
        identity = {"row_id": rid, "unit": unit}
        release = {**identity, "prob_1": p1, "prob_2": p2, "reader_predictions": scores}
        references = {**identity, "U0_1": p1, "U0_2": p2, "Ucal_1": p1, "Ucal_2": p2}
        truth = {**identity, "income": y1, "occupation": y2, "sex": sex}
        item = {"truth": truth, "candidate": release, "controls": {"CTL": release},
                "references": references}
        result = calculate_margins({0: item, 1: item}, ("CTL",), np.arange(n))
        self.assertEqual(len(margin_names(("CTL",))), 15)
        np.testing.assert_allclose(result["values"][:12], [.010, .010, .005] * 4)
        np.testing.assert_allclose(result["values"][-3:], [.010, .010, -.020])
        changed = dict(release)
        changed["unit"] = unit[::-1]
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            calculate_margins({0: {**item, "candidate": changed}}, ("CTL",), np.arange(n))


if __name__ == "__main__":
    unittest.main()
