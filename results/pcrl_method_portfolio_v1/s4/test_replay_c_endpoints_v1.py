"""Data-free fixtures for independent C point endpoint reconstruction."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from replay_c_endpoints_v1 import (pair_aliases, replay_score_receipt,
                                   score_view, task_metrics)
from replay_c_readers_v1 import READERS, oriented_auc


class CEndpointReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sex = np.array([0, 1, 0, 1], dtype=np.int64)
        self.labels = {"income": np.array([0, 1, 1, 0]),
                       "occupation": np.array([0, 1, 2, 5])}
        self.probability = {
            "income": np.array([[.8, .2], [.3, .7], [.1, .9], [.6, .4]]),
            "occupation": np.eye(6, dtype=np.float64)[self.labels["occupation"]],
        }
        self.predictions = {
            view: {reader: np.array([.1, .9, .2, .8], dtype=np.float64)
                   for reader in READERS}
            for view in ("r1", "r2", "pair")
        }
        self.validation = {
            view: {
                reader: {**oriented_auc(self.sex, prediction),
                         "selected_orientation": "positive"}
                for reader, prediction in bank.items()
            }
            for view, bank in self.predictions.items()
        }

    def _receipt(self) -> dict:
        pair, aliases = pair_aliases(self.predictions["r1"],
                                     self.predictions["r2"],
                                     self.predictions["pair"])
        return {
            "schema": "team-c-arm-score-stage-v1", "status": "SCORED",
            "task_metrics": {
                "income": task_metrics(self.probability["income"],
                                       self.labels["income"], 2),
                "occupation": task_metrics(self.probability["occupation"],
                                           self.labels["occupation"], 6),
            },
            "view_scores": {
                view: score_view(pair if view == "pair" else self.predictions[view],
                                 self.sex, READERS[0], "positive")
                for view in self.predictions
            },
            "ignore_aliases": aliases,
        }

    def test_independent_accuracy_brier_and_auc(self) -> None:
        income = task_metrics(self.probability["income"],
                              self.labels["income"], 2)
        self.assertEqual(income["rows"], 4)
        self.assertEqual(income["accuracy"], 1.0)
        self.assertEqual(income["class_counts"], [2, 2])
        self.assertAlmostEqual(income["brier"], .15)
        score = score_view(self.predictions["r1"], self.sex,
                           READERS[0], "positive")
        self.assertEqual(score["primary_envelope_auc"], 1.0)
        self.assertEqual(score["validation_winner_inner_auc"], 1.0)

    def test_all_three_views_aliases_and_saved_score_parity(self) -> None:
        saved = self._receipt()
        replayed = replay_score_receipt(saved, self.probability, self.labels,
                                        self.sex, self.predictions, self.validation)
        self.assertEqual(len(replayed["ignore_aliases"]), 12)
        self.assertEqual(len(replayed["view_scores"]["pair"]["readers"]), 18)
        self.assertEqual(replayed["ignore_aliases"][0]["alias"],
                         "IGNORE_R2__LR_C1")
        for field, value in (("task_metrics", .2), ("view_scores", .3)):
            altered = copy.deepcopy(saved)
            if field == "task_metrics":
                altered[field]["income"]["accuracy"] = value
            else:
                altered[field]["pair"]["primary_envelope_auc"] = value
            with self.assertRaisesRegex(ValueError, "numerical endpoint differs"):
                replay_score_receipt(altered, self.probability, self.labels,
                                     self.sex, self.predictions, self.validation)

    def test_partial_reader_slate_refused(self) -> None:
        partial = {view: dict(bank) for view, bank in self.predictions.items()}
        partial["r1"].pop(READERS[-1])
        with self.assertRaisesRegex(ValueError, "six-reader order"):
            replay_score_receipt(self._receipt(), self.probability, self.labels,
                                 self.sex, partial, self.validation)


if __name__ == "__main__":
    unittest.main()
