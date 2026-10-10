"""Data-free complete-wire fixtures for the independent replay boundary."""

import unittest

import numpy as np
import torch

from independent_release import replay_bank, replay_feature, replay_hard, tensor_digest
from test_independent_decoder import flat_state


class IndependentReleaseTest(unittest.TestCase):
    def setUp(self):
        self.ids = np.arange(3, dtype=np.int64)
        self.units = np.arange(3, dtype=np.int64)
        self.labels = (np.array([0, 1, 1]), np.array([0, 1, 2]))

    def test_hard_wire_reconstructs_without_private_teacher(self):
        state = {key: value.numpy() for key, value in flat_state(10).items()}
        wire = {"row_id": self.ids, "unit": self.units,
                "code0": np.zeros((3, 10), dtype=np.uint8),
                "code1": np.ones((3, 10), dtype=np.uint8),
                "prob0": np.full((3, 2), .5),
                "prob1": np.full((3, 6), 1/6),
                "decision0": np.zeros(3, dtype=np.int64),
                "decision1": np.zeros(3, dtype=np.int64)}
        result = replay_hard(wire, self.ids, self.units, self.labels,
                             state, 10, (1.0, 1.0))
        self.assertEqual(result["states"]["income"]["occupied_states"], 1)
        wire["prob0"] = np.full((3, 2), [.6, .4])
        with self.assertRaises(ValueError):
            replay_hard(wire, self.ids, self.units, self.labels, state, 10, (1.0, 1.0))

    def test_bank_code_only_and_global_capacity(self):
        table0 = np.array([[.8, .2], [.1, .9]], dtype=np.float64)
        table1 = np.full((1, 6), 1/6, dtype=np.float64)
        wire = {"row_id": self.ids, "unit": self.units,
                "code0": np.zeros((3, 10), dtype=np.uint8),
                "code1": np.zeros((3, 10), dtype=np.uint8),
                "prob0": np.tile(table0[0], (3, 1)),
                "prob1": np.tile(table1[0], (3, 1)),
                "decision0": np.zeros(3, dtype=np.int64),
                "decision1": np.zeros(3, dtype=np.int64)}
        result = replay_bank(wire, self.ids, self.units, self.labels, (table0, table1))
        self.assertEqual(result["states"]["income"]["global_table_states"], 2)
        wire["code0"][0, 1] = 1
        with self.assertRaises(ValueError):
            replay_bank(wire, self.ids, self.units, self.labels, (table0, table1))

    def test_feature_service_requires_transmitted_features(self):
        state = {}
        for purpose, classes in ((0, 2), (1, 6)):
            prefix = f"head{purpose}."
            state[prefix + "layer1.weight"] = np.zeros((64, 16), dtype=np.float32)
            state[prefix + "layer1.bias"] = np.zeros(64, dtype=np.float32)
            state[prefix + "layer2.weight"] = np.zeros((classes, 64), dtype=np.float32)
            state[prefix + "layer2.bias"] = np.zeros(classes, dtype=np.float32)
        wire = {"row_id": self.ids, "unit": self.units,
                "features0": np.zeros((3, 16), dtype=np.float32),
                "features1": np.zeros((3, 16), dtype=np.float32),
                "prob0": np.full((3, 2), .5),
                "prob1": np.full((3, 6), 1/6),
                "decision0": np.zeros(3, dtype=np.int64),
                "decision1": np.zeros(3, dtype=np.int64)}
        result = replay_feature(wire, self.ids, self.units, self.labels,
                                state, (1.0, 1.0))
        self.assertAlmostEqual(result["metrics"]["income"]["accuracy"], 1/3)
        self.assertEqual(len(tensor_digest(state)), 64)


if __name__ == "__main__":
    unittest.main()
