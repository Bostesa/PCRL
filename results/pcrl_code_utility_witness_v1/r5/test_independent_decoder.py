"""Data-free serialized-tensor forward fixtures for the separate R5 replay."""

import unittest

import numpy as np
import torch

from independent_decoder import (
    continuous_head_probabilities, hard_bits_from_latent,
    probabilities_from_hard_bits, public_head_table,
)


def flat_state(bits: int) -> dict[str, torch.Tensor]:
    state = {}
    for purpose, classes in ((0, 2), (1, 6)):
        m = f"mappers.{purpose}."
        h = f"heads.{purpose}."
        state[m + "layer1.weight"] = torch.zeros((64, 16), dtype=torch.float32)
        state[m + "layer1.bias"] = torch.zeros(64, dtype=torch.float32)
        state[m + "layer2.weight"] = torch.zeros((bits, 64), dtype=torch.float32)
        state[m + "layer2.bias"] = torch.zeros(bits, dtype=torch.float32)
        state[h + "layer1.weight"] = torch.zeros((64, bits), dtype=torch.float32)
        state[h + "layer1.bias"] = torch.zeros(64, dtype=torch.float32)
        state[h + "layer2.weight"] = torch.zeros((classes, 64), dtype=torch.float32)
        state[h + "layer2.bias"] = torch.zeros(classes, dtype=torch.float32)
    return state


class IndependentDecoderTest(unittest.TestCase):
    def test_hard_mapper_and_public_head(self):
        state = flat_state(10)
        latent = np.zeros((2, 16), dtype=np.float32)
        center = np.zeros((2, 16), dtype=np.float32)
        divisor = np.ones((2, 16), dtype=np.float32)
        code = hard_bits_from_latent(state, latent, center, divisor, 0, 10)
        self.assertEqual(code.shape, (2, 10))
        self.assertTrue(np.all(code == 1))  # sigmoid(0) equals the hard tie threshold
        probability = probabilities_from_hard_bits(state, code, 0, 10)
        self.assertTrue(np.array_equal(probability, np.full((2, 2), .5)))
        self.assertTrue(np.array_equal(public_head_table(state, 0, 10)[-1], probability[0]))

    def test_sealed_capacity_and_no_soft_code(self):
        state = flat_state(16)
        code = np.zeros((1, 16), dtype=np.uint8)
        self.assertEqual(probabilities_from_hard_bits(state, code, 1, 16).shape, (1, 6))
        with self.assertRaises(ValueError):
            probabilities_from_hard_bits(state, np.full((1, 16), .5), 1, 16)
        with self.assertRaises(ValueError):
            probabilities_from_hard_bits(state, code, 1, 10)

    def test_global_alpha_is_public_code_only(self):
        state = flat_state(10)
        state["heads.0.layer2.bias"][0] = 1.0
        code = np.zeros((1, 10), dtype=np.uint8)
        p1 = probabilities_from_hard_bits(state, code, 0, 10)
        p2 = probabilities_from_hard_bits(state, code, 0, 10, alpha=2.0)
        self.assertGreater(p2[0, 0], p1[0, 0])

    def test_old_continuous_head_is_feature_derived(self):
        state = {}
        for purpose, classes in ((0, 2), (1, 6)):
            prefix = f"head{purpose}."
            state[prefix + "layer1.weight"] = torch.zeros((64, 16), dtype=torch.float32)
            state[prefix + "layer1.bias"] = torch.zeros(64, dtype=torch.float32)
            state[prefix + "layer2.weight"] = torch.zeros((classes, 64), dtype=torch.float32)
            state[prefix + "layer2.bias"] = torch.zeros(classes, dtype=torch.float32)
        state["head0.layer2.bias"][0] = 1.0
        features = np.zeros((2, 16), dtype=np.float32)
        raw = continuous_head_probabilities(state, features, 0)
        calibrated = continuous_head_probabilities(state, features, 0, 2.0)
        self.assertGreater(calibrated[0, 0], raw[0, 0])
        with self.assertRaises(ValueError):
            continuous_head_probabilities(state, features.astype(np.float64), 0)


if __name__ == "__main__":
    unittest.main()
