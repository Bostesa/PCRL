import unittest

import numpy as np

from bank_anchored.r5_replay_attack import replay_reader_scores


class IndependentAttackScoreReplay(unittest.TestCase):
    def test_validation_orientation_and_inner_envelope(self):
        y_tune = np.asarray([0, 0, 1, 1], dtype=np.int64)
        y_inner = np.asarray([0, 1, 0, 1], dtype=np.int64)
        tune = {"reader_a": np.asarray([.9, .8, .2, .1]),
                "reader_b": np.asarray([.1, .4, .6, .9])}
        inner = {"reader_a": np.asarray([.8, .2, .6, .4]),
                 "reader_b": np.asarray([.2, .7, .3, .8])}
        result = replay_reader_scores(y_tune, y_inner, tune, inner,
                                      ("reader_a", "reader_b"))
        self.assertEqual(result["validation_winner"], "reader_a")
        self.assertEqual(result["validation_winner_orientation"], "reverse")
        self.assertEqual(result["primary_envelope_auc"], 1.0)
        self.assertEqual(result["readers"]["reader_a"]["positive_alias_auc"], 0.0)
        self.assertEqual(result["validation_winner_inner_auc"], 1.0)

    def test_rejects_missing_prediction_member(self):
        y = np.asarray([0, 1], dtype=np.int64)
        with self.assertRaises(ValueError):
            replay_reader_scores(y, y, {"a": np.asarray([.1, .9])}, {}, ("a",))


if __name__ == "__main__":
    unittest.main()
