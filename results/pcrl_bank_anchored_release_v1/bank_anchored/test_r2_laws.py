"""Data-free identities and factorization checks for the bank channel."""

import unittest

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import FixedBank
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import (
    GateWeights, JointPolicy, expected_cost,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.factor_controls import (
    FactorGate, ProductPolicy, LocalPolicy,
)
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import (
    IdentityPolicy, BoundaryJointPolicy, BoundaryFactorPolicy,
    identity_pair, mix_joint, mix_product, offline_factor_uniforms,
    wire_fixed_bank_and_actions,
)


def _fixture():
    bank1 = FixedBank(np.array([[.8, .2], [.7, .3]], dtype=np.float64),
                      np.array([0, 0], dtype=np.int64), (0, 1))
    bank2 = FixedBank(np.array([[.7, .3], [.6, .4]], dtype=np.float64),
                      np.array([0, 0], dtype=np.int64), (0, 1))
    gate = GateWeights.zeros(input_width=8)
    base = JointPolicy(bank1, bank2, ((0, 1),), ((0, 1),), gate)
    score1 = np.array([.85, .15], dtype=np.float64)
    score2 = np.array([.75, .25], dtype=np.float64)
    return base, score1, score2


def _factor(input_width):
    return FactorGate(np.zeros((input_width, 16)), np.zeros(16),
                      np.zeros((16, 4)), np.zeros(4))


class BankBoundaryTests(unittest.TestCase):
    def test_source_action_choices_remap_to_original_public_token_ids(self):
        # The source canonical sorter changes these original row IDs.
        table = np.array([[.8, .2], [.7, .3], [.2, .8], [.3, .7]], dtype=np.float64)
        token_class = np.array([0, 0, 1, 1], dtype=np.int64)
        fine_to_old = np.array([0, 1, 2, 3], dtype=np.int64)
        wire_bank, actions = wire_fixed_bank_and_actions(
            table, token_class, fine_to_old, token_class)
        np.testing.assert_array_equal(wire_bank.probability, table)
        self.assertEqual(wire_bank.old_to_global, (0, 1, 2, 3))
        self.assertEqual(tuple(row[0] for row in actions), (0, 1, 2, 3))
        base = JointPolicy(wire_bank, wire_bank, actions, actions,
                           GateWeights.zeros(input_width=8))
        for token in range(4):
            score = table[token].copy()
            self.assertEqual(IdentityPolicy(base).sample_public_pair(
                score, score, token, token), (token, token))

    def test_exact_identity_is_source_bank_action_and_zero_excess(self):
        base, s1, s2 = _fixture()
        identity = IdentityPolicy(base)
        self.assertEqual(identity_pair(base, s1, s2, 0, 0), (0, 0))
        self.assertEqual(identity.sparse_law(s1, s2, 0, 0), ((0, 0, 1.0),))
        self.assertEqual(identity.sample_public_pair(s1, s2, 0, 0), (0, 0))
        self.assertEqual(BoundaryJointPolicy(base, 0).sparse_law(s1, s2, 0, 0),
                         identity.sparse_law(s1, s2, 0, 0))
        cost = expected_cost(base.bank1, base.bank2,
                             identity.sparse_law(s1, s2, 0, 0), y1=0, y2=0)
        bank_cost = expected_cost(base.bank1, base.bank2, ((0, 0, 1.0),),
                                  y1=0, y2=0)
        self.assertEqual(cost, bank_cost)
        self.assertEqual(base.bank1.decision[0], np.argmax(s1))
        self.assertEqual(base.bank2.decision[0], np.argmax(s2))

    def test_joint_mixture_aggregates_identity_and_preserves_bank(self):
        base, s1, s2 = _fixture()
        law = BoundaryJointPolicy(base, .4).sparse_law(s1, s2, 0, 0)
        self.assertAlmostEqual(sum(x[2] for x in law), 1)
        self.assertEqual([x[:2] for x in law], [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertAlmostEqual(law[0][2], .7)
        self.assertTrue(all(base.bank1.decision[i] == 0 and
                            base.bank2.decision[j] == 0 for i, j, _ in law))
        direct = mix_joint((0, 0), ((0, 0, .25), (0, 1, .25),
                                    (1, 0, .25), (1, 1, .25)), .4)
        self.assertEqual(law, direct)

    def test_product_boundaries_use_independent_recipient_coins(self):
        base, s1, s2 = _fixture()
        product = ProductPolicy(base.bank1, base.bank2, base.actions1,
                                base.actions2, base.gate, _factor(8), _factor(8))
        law = BoundaryFactorPolicy(product, .2, .6).sparse_law(s1, s2, 0, 0)
        mass = {(i, j): p for i, j, p in law}
        self.assertAlmostEqual(sum(mass.values()), 1)
        self.assertAlmostEqual(mass[0, 0], .9 * .7)
        self.assertAlmostEqual(mass[0, 1], .9 * .3)
        self.assertAlmostEqual(mass[1, 0], .1 * .7)
        self.assertAlmostEqual(mass[1, 1], .1 * .3)
        self.assertEqual(BoundaryFactorPolicy(product, 0, 0).sparse_law(
            s1, s2, 0, 0), ((0, 0, 1.0),))
        policy = BoundaryFactorPolicy(product, .2, .6)
        self.assertEqual(policy.sample_public_pair(s1, s2, 0, 0, (.05, .95)), (0, 1))
        self.assertEqual(policy.sample_public_pair(s1, s2, 0, 0, (.95, .05)), (1, 0))
        with self.assertRaisesRegex(ValueError, "two independent uniforms"):
            policy.sample_public_pair(s1, s2, 0, 0, .5)

    def test_factor_hmac_domains_are_distinct_and_repeatable(self):
        root = bytes(range(32))
        a = offline_factor_uniforms(root, source_seed=0, role="INNER_SELECTION",
                                    realization=1, unit=17)
        b = offline_factor_uniforms(root, source_seed=0, role="INNER_SELECTION",
                                    realization=1, unit=17)
        c = offline_factor_uniforms(root, source_seed=0, role="INNER_SELECTION",
                                    realization=2, unit=17)
        self.assertEqual(a, b)
        self.assertNotEqual(a[0], a[1])
        self.assertNotEqual(a, c)
        self.assertTrue(all(0 <= x < 1 for x in a))

    def test_local_boundary_preserves_own_score_contract(self):
        base, s1, s2 = _fixture()
        local = LocalPolicy(base.bank1, base.bank2, base.actions1,
                            base.actions2, base.gate, _factor(4), _factor(4))
        law = BoundaryFactorPolicy(local, .3, .5).sparse_law(s1, s2, 0, 0)
        self.assertAlmostEqual(sum(p for _, _, p in law), 1)
        self.assertEqual(len(law), 4)

    def test_factor_boundary_refuses_correlated_joint_source(self):
        with self.assertRaisesRegex(ValueError, "not a recipient product"):
            mix_product((0, 0), ((0, 0, .5), (1, 1, .5)), .2, .2)

    def test_malformed_boundaries_and_class_mismatch_fail_closed(self):
        base, s1, s2 = _fixture()
        with self.assertRaises(ValueError):
            BoundaryJointPolicy(base, -1e-4)
        with self.assertRaises(ValueError):
            BoundaryJointPolicy(base, np.nan)
        with self.assertRaises(ValueError):
            IdentityPolicy(base).sparse_law(np.array([.1, .9]), s2, 0, 0)
        with self.assertRaises(TypeError):
            BoundaryFactorPolicy(base, 0, 0)


if __name__ == "__main__":
    unittest.main()
