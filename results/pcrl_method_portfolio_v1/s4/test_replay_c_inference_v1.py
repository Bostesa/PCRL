"""Data-free contract checks for S4 C paired inference; run only via S3."""

from __future__ import annotations

import unittest

import numpy as np

from replay_c_inference_v1 import evaluate, margin_names, prepare
from run_c_aggregate_replay_v1 import ARMS


class IndependentCInferenceSynthetic(unittest.TestCase):
    def _records(self):
        n = 2235
        row = np.arange(n, dtype=np.int64)
        unit = row.copy()
        unit[-1] = unit[-2]
        income = row % 2
        occupation = row % 6
        sex = row % 2
        p1 = np.column_stack((np.full(n, .6), np.full(n, .4)))
        p2 = np.full((n, 6), 1 / 6, dtype=np.float64)
        reader = np.linspace(.1, .9, n)
        item = {"row_id": row, "unit": unit, "income": income,
                "occupation": occupation, "sex": sex,
                "prob_1": p1, "prob_2": p2}
        for view in ("r1", "r2", "pair"):
            for name in ("LR_C1", "MLP_64x32", "HGB_15x200",
                         "DA_MLP_64x32", "RAW_LR_C1", "RAW_HGB_15x200"):
                item[f"reader_{view}__{name}"] = reader
        reference = {"U0_1": p1, "Ucal_1": p1,
                     "U0_2": p2, "Ucal_2": p2}
        return {seed: {"truth": {key: item[key] for key in
                                ("row_id", "unit", "income", "occupation", "sex")},
                       "references": reference,
                       "arms": {arm: item for arm in ARMS}}
                for seed in (0, 1)}

    def test_full_family_and_frozen_margin_orientation(self):
        prepared = prepare(self._records(), ARMS)
        names = margin_names(ARMS)
        values = evaluate(prepared, np.arange(2235, dtype=np.int64))
        self.assertEqual(len(names), 264)
        self.assertEqual(values.shape, (264,))
        self.assertAlmostEqual(values[0], .010)
        self.assertAlmostEqual(values[12], .010)
        self.assertAlmostEqual(values[13], .010)
        self.assertAlmostEqual(values[14], -.020)
        self.assertTrue(np.isfinite(values).all())

    def test_order_and_truth_mismatch_refuse(self):
        records = self._records()
        changed = dict(records[1]["arms"][ARMS[0]])
        changed["row_id"] = changed["row_id"][::-1]
        records[1]["arms"] = dict(records[1]["arms"])
        records[1]["arms"][ARMS[0]] = changed
        with self.assertRaisesRegex(ValueError, "person order"):
            prepare(records, ARMS)


if __name__ == "__main__":
    unittest.main()
