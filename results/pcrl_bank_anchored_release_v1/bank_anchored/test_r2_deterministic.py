"""Tiny fixed-law parity with the inherited deterministic optimizer."""

import unittest
from dataclasses import asdict

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls import deterministic_trainer as inherited
from results.pcrl_bank_anchored_release_v1.bank_anchored import r2_source_deterministic as retaining


class RetainedDeterministicParity(unittest.TestCase):
    def test_inherited_winner_and_private_replay_checkpoints_agree(self):
        n = 8
        features = np.zeros((n, 12), dtype=np.float64)
        features[:, 0] = np.linspace(-1, 1, n)
        pair = np.full((n, 16, 2), -1, dtype=np.int64)
        pair[:, 0, :] = (0, 0)
        pair[:, 1, :] = (1, 0)
        valid = np.zeros((n, 16), dtype=np.bool_)
        valid[:, :2] = True
        costs = np.zeros((n, 16, 4), dtype=np.float64)
        costs[:, 1, :] = .01
        args = (features, pair, valid, costs, np.zeros(n, dtype=np.int64),
                (2, 1), np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64))
        params = dict(mode="private", steps=2, learning_rate=.01,
                      temperature_start=1., temperature_end=.5, rho=100.,
                      dual_every=1, replay_every=1, grad_clip=5.,
                      cpu_limit_seconds=20., seed=42,
                      bias_deltas=(-.5, -.1, .1, .5))
        caps = (1., 1., 1., 1.)
        old = inherited.fit_deterministic(
            inherited.DeterministicBatch(*args),
            inherited.DeterministicConfig(**params), caps=caps)
        new = retaining.fit_deterministic(
            retaining.DeterministicBatch(*args),
            retaining.DeterministicConfig(**params), caps=caps)
        self.assertTrue(old.complete and new.complete)
        self.assertEqual(old.success, new.success)
        self.assertEqual(asdict(old.metrics), asdict(new.metrics))
        self.assertEqual([asdict(v) for v in old.starts],
                         [asdict(v) for v in new.starts])
        for name in ("w1", "b1", "w2", "b2"):
            np.testing.assert_array_equal(getattr(old.weights, name),
                                          getattr(new.weights, name))
        self.assertEqual([phase for _, _, phase, _, _ in new.checkpoints].count("initial"), 2)
        self.assertGreaterEqual(
            [phase for _, _, phase, _, _ in new.checkpoints].count("replay"), 4)
        self.assertEqual([phase for _, _, phase, _, _ in new.checkpoints].count("terminal"), 2)


if __name__ == "__main__":
    unittest.main()
