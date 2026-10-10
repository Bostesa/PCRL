"""Data-free laws for central product, local, and freeze-first score controls."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "impl"))
from bank import FixedBank
from channel import GateWeights
from release import _mechanism_hash, CachedJointRelease

from factor_controls import (FactorGate, ProductPolicy, LocalPolicy,
                             SequentialPolicy, public_probability_embedding,
                             initial_factor_gate, _factor_law)


def fixture():
    s1 = np.array([.9, .1], dtype=np.float64)
    s2 = np.array([.8, .2], dtype=np.float64)
    b1 = FixedBank.from_source(np.array([[.9, .1], [.6, .4]]),
                               np.arange(2), np.zeros(2, dtype=np.int64))
    b2 = FixedBank.from_source(np.array([[.8, .2], [.7, .3]]),
                               np.arange(2), np.zeros(2, dtype=np.int64))
    a1 = (tuple(map(int, b1.old_to_global)),)
    a2 = (tuple(map(int, b2.old_to_global)),)
    base = GateWeights.zeros(input_width=8)
    return s1, s2, b1, b2, a1, a2, base


def factor(width, bias=(0.0, 0.0, 0.0, 0.0), embedding=None):
    return FactorGate(np.zeros((width, 16)), np.zeros(16),
                      np.zeros((16, 4)), np.array(bias), embedding)


class FactorControlTest(unittest.TestCase):
    def test_start_zero_and_uniform_factor_laws(self):
        first = initial_factor_gate(8, np.random.default_rng(7),
                                    identity_biased=True)
        uniform = initial_factor_gate(8, np.random.default_rng(8),
                                      identity_biased=False)
        self.assertTrue(np.array_equal(first.w2, np.zeros((16, 4))))
        self.assertTrue(np.array_equal(uniform.w2, np.zeros((16, 4))))
        q0 = _factor_law(first, np.ones(8), (3, 1))
        q1 = _factor_law(uniform, np.ones(8), (3, 1))
        self.assertGreater(q0[0][1], .9)
        self.assertEqual(q1, ((3, .5), (1, .5)))

    def test_central_product_has_independent_factors_and_distinct_hash(self):
        s1, s2, b1, b2, a1, a2, base = fixture()
        q = ProductPolicy(b1, b2, a1, a2, base,
                          factor(8, (0.0, 1.0, 0.0, 0.0)), factor(8))
        law = q.sparse_law(s1, s2, 0, 0)
        self.assertEqual(len(law), 4)
        first = {i: sum(p for ii, _, p in law if ii == i) for i in a1[0]}
        self.assertGreater(first[a1[0][1]], first[a1[0][0]])
        self.assertAlmostEqual(sum(p for _, _, p in law), 1.0)
        altered = ProductPolicy(b1, b2, a1, a2, base,
                                factor(8, (0.0, 2.0, 0.0, 0.0)), factor(8))
        self.assertNotEqual(_mechanism_hash(q), _mechanism_hash(altered))

    def test_local_policy_requires_own_input_width(self):
        s1, s2, b1, b2, a1, a2, base = fixture()
        local = LocalPolicy(b1, b2, a1, a2, base, factor(4), factor(4))
        self.assertAlmostEqual(sum(p for _, _, p in local.sparse_law(s1, s2, 0, 0)), 1)
        with self.assertRaises(ValueError):
            LocalPolicy(b1, b2, a1, a2, base, factor(8), factor(4))

    def test_product_and_local_release_nonmonotone_action_ids(self):
        s1, s2, b1, b2, a1, a2, base = fixture()
        reverse1 = (tuple(sorted(a1[0], reverse=True)),)
        reverse2 = (tuple(sorted(a2[0], reverse=True)),)
        policies = (
            ProductPolicy(b1, b2, reverse1, reverse2, base, factor(8), factor(8)),
            LocalPolicy(b1, b2, reverse1, reverse2, base, factor(4), factor(4)),
        )
        for index, policy in enumerate(policies):
            with self.subTest(kind=policy.law_kind):
                law = policy.sparse_law(s1, s2, 0, 0)
                self.assertEqual(tuple((i, j) for i, j, _ in law),
                                 tuple(sorted((i, j) for i, j, _ in law)))
                with tempfile.TemporaryDirectory() as directory:
                    service = CachedJointRelease(
                        policy, cache_path=Path(directory) / "release.sqlite",
                        release_id=f"factor-{index}", source_seed=0,
                        role="SYNTHETIC", realization=0,
                        source_binding_sha256="0" * 64,
                        private_root=b"r" * 32, offline=True)
                    left = service.recipient(recipient=1, unit="p", score1=s1,
                                             score2=s2, cell1=0, cell2=0)
                    right = service.recipient(recipient=2, unit="p", score1=s1,
                                              score2=s2, cell1=0, cell2=0)
                    self.assertIn((left["code_id"], right["code_id"]),
                                  {(i, j) for i, j, mass in law if mass > 0})

    def test_sequential_conditions_on_actual_first_public_code(self):
        s1, s2, b1, b2, a1, a2, base = fixture()
        embedding = np.zeros((len(b1.probability), 16))
        embedding[a1[0][1], 0] = 2.0
        second = factor(8, embedding=embedding)
        # A nonzero second head reads the first-code embedding.
        second_w2 = second.w2.copy()
        second_w2[0, 1] = 3.0
        second = FactorGate(second.w1, second.b1, second_w2,
                            second.b2, second.embedding)
        seq = SequentialPolicy(b1, b2, a1, a2, base, factor(8), second,
                               order="12", first_checkpoint_sha256="0" * 64)
        law = seq.sparse_law(s1, s2, 0, 0)
        by_first = {i: {j: p for ii, j, p in law if ii == i} for i in a1[0]}
        self.assertGreater(by_first[a1[0][1]][a2[0][1]],
                           by_first[a1[0][0]][a2[0][1]])
        self.assertAlmostEqual(sum(p for _, _, p in law), 1)
        for uniform in (0.0, .15, .5, .75, .999999):
            sampled = seq.sample_public_pair(s1, s2, 0, 0, uniform)
            self.assertIn(sampled, {(i, j) for i, j, mass in law if mass > 0})

    def test_unseen_first_code_has_public_bank_initialization(self):
        s1, s2, b1, b2, a1, a2, base = fixture()
        initial = public_probability_embedding(b2.probability)
        np.testing.assert_array_equal(initial[1],
                                      b2.probability[1, np.arange(16) % 2])
        # Suppose only code 0 appears in the frozen FIT draw. Code 1 remains
        # available at deployment and the full-law replay with its fixed E0.
        second = factor(8, embedding=initial)
        seq = SequentialPolicy(b1, b2, a1, a2, base, factor(8), second,
                               order="21", first_checkpoint_sha256="a" * 64)
        law = seq.sparse_law(s1, s2, 0, 0)
        self.assertTrue(any(j == 1 and mass > 0 for _, j, mass in law))
        self.assertAlmostEqual(sum(mass for _, _, mass in law), 1)


if __name__ == "__main__":
    unittest.main()
