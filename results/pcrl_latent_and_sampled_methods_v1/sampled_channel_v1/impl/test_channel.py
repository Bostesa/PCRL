"""Data-free finite-channel arithmetic and public-identity checks."""

import math
import unittest

import numpy as np

from bank import FixedBank
from channel import (GateWeights, JointPolicy, aggregate_public_pairs,
                     bayes_recovery, expected_cost)


class JointChannelTest(unittest.TestCase):
    def policy(self):
        source1 = np.array([[.9, .1], [.6, .4]])
        source2 = np.array([[.8, .2], [.7, .3]])
        bank1 = FixedBank.from_source(source1, np.arange(2), np.array([0, 0]))
        bank2 = FixedBank.from_source(source2, np.arange(2), np.array([0, 0]))
        actions1 = (tuple(bank1.old_to_global),)
        actions2 = (tuple(bank2.old_to_global),)
        gate = GateWeights.zeros(input_width=8)
        return JointPolicy(bank1, bank2, actions1, actions2, gate)

    def test_gate_normalizes_after_unique_public_actions(self):
        policy = self.policy()
        triples = policy.sparse_law(np.array([.9, .1]), np.array([.8, .2]), 0, 0)
        self.assertEqual(len(triples), 4)
        self.assertAlmostEqual(sum(p for _, _, p in triples), 1)
        self.assertEqual(len({(i, j) for i, j, _ in triples}), 4)
        self.assertTrue(all(abs(p - .25) < 1e-12 for _, _, p in triples))

    def test_actual_public_pairs_aggregate_duplicate_action_positions(self):
        triples = aggregate_public_pairs(
            [(2, 3), (2, 3), (2, 4)], np.array([.2, .3, .5]))
        self.assertEqual(triples, ((2, 3, .5), (2, 4, .5)))

    def test_expected_loss_is_loss_of_sample_not_loss_of_mean(self):
        policy = self.policy()
        triples = policy.sparse_law(np.array([.9, .1]), np.array([.8, .2]), 0, 0)
        cost = expected_cost(policy.bank1, policy.bank2, triples, y1=0, y2=0)
        self.assertAlmostEqual(cost["r1"]["nll"],
                               (-math.log(.9) - math.log(.6)) / 2)
        self.assertAlmostEqual(cost["r1"]["brier"],
                               ((.9 - 1)**2 + .1**2 + (.6 - 1)**2 + .4**2) / 2)
        self.assertGreater(cost["r1"]["nll"], -math.log(.75))
        self.assertGreater(cost["r1"]["brier"], 2 * (.75 - 1)**2)

    def test_empirical_bayes_recovery_uses_actual_global_ids(self):
        policy = self.policy()
        row = policy.sparse_law(np.array([.9, .1]), np.array([.8, .2]), 0, 0)
        recovery = bayes_recovery([row, row], np.array([0, 1], dtype=np.int64))
        self.assertEqual(recovery, {"r1": .5, "r2": .5, "pair": .5})

    def test_gate_uses_frozen_tanh_hidden_activation(self):
        w1 = np.zeros((8, 16))
        w1[0, 0] = 2.0
        w2 = np.zeros((16, 16))
        w2[0, 0] = 3.0
        gate = GateWeights(w1, np.zeros(16), w2, np.zeros(16))
        features = np.zeros(8)
        features[0] = .9
        self.assertAlmostEqual(gate.logits(features)[0], 3 * np.tanh(1.8))

    def test_action_ids_cannot_be_truncated_from_floats_or_bools(self):
        policy = self.policy()
        for bad in (True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                JointPolicy(policy.bank1, policy.bank2, ((0, bad),),
                            policy.actions2, policy.gate)

    def test_masked_gate_uses_fixed_four_by_four_action_slots(self):
        source1 = np.array([[.9, .1], [.8, .2]])
        source2 = np.array([[.9, .1], [.8, .2], [.7, .3]])
        bank1 = FixedBank.from_source(source1, np.arange(2), np.zeros(2, dtype=np.int64))
        bank2 = FixedBank.from_source(source2, np.arange(3), np.zeros(3, dtype=np.int64))
        bias = np.arange(16, dtype=np.float64) / 10
        gate = GateWeights(np.zeros((8, 16)), np.zeros(16),
                           np.zeros((16, 16)), bias)
        policy = JointPolicy(bank1, bank2, (tuple(bank1.old_to_global),),
                             (tuple(bank2.old_to_global),), gate)
        law = policy.sparse_law(source1[0], source2[0], 0, 0)
        slots = np.array([0, 1, 2, 4, 5, 6])
        expected = np.exp(bias[slots] - np.max(bias[slots]))
        expected /= expected.sum()
        by_pair = {(i, j): mass for i, j, mass in law}
        for k, pair in enumerate((i, j) for i in bank1.old_to_global
                                 for j in bank2.old_to_global):
            self.assertAlmostEqual(by_pair[pair], expected[k])

    def test_bayes_recovery_refuses_negative_or_nonfinite_mass(self):
        for row in (((0, 0, 1.1), (0, 1, -.1)),
                    ((0, 0, float("nan")),),
                    ((-1, 0, 1.0),)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                bayes_recovery([row], np.array([0], dtype=np.int64))

    def test_private_teacher_score_requires_exact_float64_bytes(self):
        policy = self.policy()
        for bad in (np.array([.9, .1], dtype=np.float32),
                    np.array([".9", ".1"])):
            with self.subTest(dtype=bad.dtype), self.assertRaises(ValueError):
                policy.sparse_law(bad, np.array([.8, .2], dtype=np.float64), 0, 0)


if __name__ == "__main__":
    unittest.main()
