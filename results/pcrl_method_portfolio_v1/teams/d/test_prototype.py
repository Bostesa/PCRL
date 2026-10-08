"""Synthetic checks for Team D's frozen-latent release adapter.

Run with: python3 -m unittest discover -s results/pcrl_method_portfolio_v1/teams/d -p 'test_prototype.py'
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prototype import (
    FrozenLatentRelease,
    MLPHead,
    ProductVQ,
    PurposeNormalizer,
    SoftmaxHead,
    occupancy_report,
    tuple_lookup_scores,
)


class ProductVQTests(unittest.TestCase):
    def _toy_model(self):
        # Two one-coordinate blocks, with the exact midpoint tied to code 0.
        codebooks = np.array([[[0.0], [2.0]], [[0.0], [2.0]]])
        normalizer = PurposeNormalizer(
            mean=np.zeros((2, 2)), scale=np.ones((2, 2))
        )
        return ProductVQ(codebooks, normalizer)

    def test_hard_map_is_many_to_one_and_ties_choose_lowest_code(self):
        vq = self._toy_model()
        latent = np.array([[0.0, 0.0], [0.2, -0.2], [1.0, 1.0], [2.0, 2.0]])
        ids, decoded = vq.encode(0, latent)
        np.testing.assert_array_equal(ids, [[0, 0], [0, 0], [0, 0], [1, 1]])
        np.testing.assert_array_equal(decoded[:3], np.zeros((3, 2)))
        np.testing.assert_array_equal(decoded[3], [2.0, 2.0])
        self.assertEqual(vq.code_length_bits, 2)

    def test_locked_full_vector_capacity_is_ten_bits(self):
        # The revised candidate is one full 16-D block with 1024 entries.
        codebooks = np.zeros((1, 1024, 16))
        codebooks[0, :, 0] = np.arange(1024)
        normalizer = PurposeNormalizer(np.zeros((2, 16)), np.ones((2, 16)))
        vq = ProductVQ(codebooks, normalizer)
        latent = np.zeros((2, 16))
        latent[:, 0] = [500.0, 500.1]
        ids, decoded = vq.encode(0, latent)
        np.testing.assert_array_equal(ids, [[500], [500]])
        np.testing.assert_array_equal(decoded[0], decoded[1])
        self.assertEqual(vq.code_length_bits, 10)

    def test_release_head_depends_only_on_protected_code(self):
        vq = self._toy_model()
        head = SoftmaxHead(np.array([[1.0, 0.0], [-1.0, 0.0]]), np.zeros(2))
        release = FrozenLatentRelease(vq, (head, head))
        a = release.recipient(0, np.array([[0.0, 0.0]]))
        b = release.recipient(0, np.array([[0.2, -0.2]]))
        self.assertEqual(set(a), {"code_ids", "decoded", "probabilities", "decision", "metadata"})
        np.testing.assert_array_equal(a["code_ids"], b["code_ids"])
        np.testing.assert_array_equal(a["decoded"], b["decoded"])
        np.testing.assert_array_equal(a["probabilities"], b["probabilities"])
        self.assertEqual(a["metadata"]["purpose"], 0)

    def test_recipient_requires_only_its_own_latent(self):
        vq = self._toy_model()
        head = SoftmaxHead(np.zeros((2, 2)), np.array([0.2, -0.2]))
        release = FrozenLatentRelease(vq, (head, head))
        first = release.recipient(0, np.array([[0.0, 0.0]]))
        second = release.recipient(1, np.array([[2.0, 2.0]]))
        np.testing.assert_array_equal(first["code_ids"], [[0, 0]])
        np.testing.assert_array_equal(second["code_ids"], [[1, 1]])

    def test_xor_fixture_has_only_coalition_clue(self):
        vq = self._toy_model()
        head = SoftmaxHead(np.zeros((2, 2)), np.zeros(2))
        release = FrozenLatentRelease(vq, (head, head))
        bit_a = np.array([0, 0, 1, 1])
        bit_b = np.array([0, 1, 0, 1])
        sex = bit_a ^ bit_b
        h0 = np.column_stack((bit_a * 2.0, np.zeros(4)))
        h1 = np.column_stack((bit_b * 2.0, np.zeros(4)))
        c0 = release.recipient(0, h0)["code_ids"][:, 0]
        c1 = release.recipient(1, h1)["code_ids"][:, 0]
        for local in (c0, c1):
            for code in (0, 1):
                self.assertEqual(sex[local == code].mean(), 0.5)
        np.testing.assert_array_equal(c0 ^ c1, sex)

    def test_shape_and_nonfinite_latent_refused(self):
        vq = self._toy_model()
        with self.assertRaises(ValueError):
            vq.encode(0, np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            vq.encode(0, np.array([[np.nan, 0.0]]))

    def test_fit_only_purpose_standardization_handles_constant_coordinate(self):
        h0 = np.array([[0.0, 4.0], [2.0, 4.0]])
        h1 = np.array([[10.0, -1.0], [12.0, -1.0]])
        normalizer = PurposeNormalizer.fit((h0, h1))
        np.testing.assert_allclose(normalizer.transform(0, h0)[:, 0], [-1.0, 1.0])
        np.testing.assert_allclose(normalizer.transform(1, h1)[:, 0], [-1.0, 1.0])
        self.assertTrue(np.isfinite(normalizer.transform(0, h0)).all())

    def test_near_constant_scale_has_fixed_floor(self):
        h0 = np.array([[0.0, 1.0], [2.0, 1.0 + 1e-10]])
        h1 = np.array([[10.0, -1.0], [12.0, -1.0 + 1e-10]])
        normalizer = PurposeNormalizer.fit((h0, h1))
        np.testing.assert_allclose(normalizer.scale[:, 1], [1e-6, 1e-6])

    def test_mlp_head_receives_only_decoded_code_vector(self):
        vq = self._toy_model()
        head = MLPHead(
            weight1=np.array([[1.0, 0.0], [-1.0, 0.0]]),
            bias1=np.zeros(2),
            weight2=np.array([[1.0, 0.0], [0.0, 1.0]]),
            bias2=np.zeros(2),
        )
        release = FrozenLatentRelease(vq, (head, head))
        a = release.recipient(0, np.array([[0.0, 0.0]]))
        b = release.recipient(0, np.array([[0.2, -0.2]]))
        np.testing.assert_array_equal(a["probabilities"], b["probabilities"])
        self.assertEqual(a["decision"].item(), 0)

    def test_released_centroid_is_shared_normalized_code_vector(self):
        normalizer = PurposeNormalizer(
            mean=np.array([[0.0, 0.0], [10.0, 10.0]]),
            scale=np.ones((2, 2)),
        )
        vq = ProductVQ(np.array([[[0.0, 0.0], [2.0, 2.0]]]), normalizer)
        ids0, decoded0 = vq.encode(0, np.array([[0.0, 0.0]]))
        ids1, decoded1 = vq.encode(1, np.array([[10.0, 10.0]]))
        np.testing.assert_array_equal(ids0, ids1)
        np.testing.assert_array_equal(decoded0, decoded1)
        np.testing.assert_array_equal(decoded1, [[0.0, 0.0]])

    def test_pair_occupancy_exposes_near_unique_full_code(self):
        # Each local block has repeated IDs, but the paired tuple is unique.
        ids0 = np.array([[0], [0], [1], [1]])
        ids1 = np.array([[0], [1], [0], [1]])
        report = occupancy_report(ids0, ids1)
        self.assertEqual(report["recipient_0"]["singleton_rows"], 0)
        self.assertEqual(report["recipient_1"]["singleton_rows"], 0)
        self.assertEqual(report["coalition"]["singleton_rows"], 4)
        self.assertEqual(report["coalition"]["singleton_fraction"], 1.0)

    def test_tuple_lookup_recovers_complementary_xor_but_not_local(self):
        # Exhaustive L3 law marginalized over task bits B1/B2: U and U XOR S.
        u = np.array([0, 0, 1, 1])
        v = np.array([0, 1, 0, 1])
        sex = u ^ v
        local = tuple_lookup_scores(u[:, None], sex, u[:, None])
        pair = tuple_lookup_scores(np.column_stack((u, v)), sex, np.column_stack((u, v)))
        np.testing.assert_allclose(local, 0.5)
        self.assertTrue(np.all(pair[sex == 1] > pair[sex == 0]))

    def test_unseen_tuple_uses_fitting_prior(self):
        train = np.array([[0], [0], [1], [1]])
        sex = np.array([0, 0, 0, 1])
        scores = tuple_lookup_scores(train, sex, np.array([[2]]))
        np.testing.assert_allclose(scores, [0.25])


if __name__ == "__main__":
    unittest.main()
