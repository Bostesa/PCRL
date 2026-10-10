"""Tiny data-free checks of separately trained deterministic controls."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from deterministic_trainer import (
    DeterministicBatch, DeterministicConfig, evaluate_selection,
    fit_deterministic, batch_from_prepared, as_public_gate, GateWeights,
)


def _batch(*, sex=None):
    n = 4
    pair_ids = np.full((n, 16, 2), -1, dtype=np.int64)
    pair_ids[:, 0] = (0, 0)
    pair_ids[:, 1] = (1, 1)
    valid = np.zeros((n, 16), dtype=bool)
    valid[:, :2] = True
    costs = np.zeros((n, 16, 4), dtype=np.float64)
    costs[:, 0, :] = 0.1
    costs[:, 1, :] = 0.9
    return DeterministicBatch(
        features=np.zeros((n, 12), dtype=np.float64),
        pair_ids=pair_ids, valid=valid, costs=costs,
        identity_slot=np.zeros(n, dtype=np.int64),
        bank_sizes=(2, 2),
        sex=None if sex is None else np.asarray(sex, dtype=np.int64),
    )


def _config(*, mode):
    return DeterministicConfig(
        mode=mode, steps=2, learning_rate=0.01,
        temperature_start=1.0, temperature_end=0.5,
        rho=10.0, dual_every=1, replay_every=1, grad_clip=5.0,
        cpu_limit_seconds=30.0, seed=7,
        bias_deltas=(-0.5, -0.1, 0.1, 0.5),
    )


class DeterministicTrainerTest(unittest.TestCase):
    def test_hard_task_rate_equals_entropy_of_emitted_public_pairs(self):
        batch = _batch()
        metrics = evaluate_selection(batch, np.array([0, 1, 0, 1]),
                                     caps=(1.0, 1.0, 1.0, 1.0), mode="task")
        self.assertAlmostEqual(metrics.primary, math.log(2))
        self.assertEqual(metrics.occupied_pairs, 2)
        self.assertEqual(metrics.occupied_codes, (2, 2))

    def test_literal_cap_boundary_rejects_positive_excess(self):
        batch = _batch()
        baseline = evaluate_selection(batch, np.zeros(4, dtype=np.int64),
                                      caps=(1.0, 1.0, 1.0, 1.0), mode="task")
        tighter = math.nextafter(baseline.costs[0], -math.inf)
        checked = evaluate_selection(batch, np.zeros(4, dtype=np.int64),
                                     caps=(tighter, 1.0, 1.0, 1.0), mode="task")
        self.assertFalse(checked.feasible)

    def test_variable_padded_slots_and_actual_public_gate_adapter(self):
        valid = np.zeros((2, 16), dtype=bool)
        valid[0, (0, 4)] = True   # 2 × 1 actions
        valid[1, (0, 1, 2)] = True  # 1 × 3 actions
        c1 = np.zeros((2, 16), dtype=np.int64)
        c2 = np.zeros((2, 16), dtype=np.int64)
        c1[0, 4] = 1
        c2[1, 1:3] = (1, 2)
        common = np.zeros((2, 16), dtype=np.float64)
        prepared = SimpleNamespace(
            features=np.zeros((2, 12), dtype=np.float64), valid=valid,
            code1=c1, code2=c2, z1=2, z2=3,
            r1_nll=common, r1_brier=common,
            r2_nll=common, r2_brier=common,
        )
        batch = batch_from_prepared(prepared,
                                    identity_public_pair=np.zeros((2, 2), dtype=np.int64))
        metrics = evaluate_selection(batch, np.array([4, 2], dtype=np.int64),
                                     caps=(0.0, 0.0, 0.0, 0.0), mode="task")
        self.assertEqual(metrics.occupied_pairs, 2)
        self.assertEqual(metrics.occupied_codes, (2, 2))
        with self.assertRaises(ValueError):
            batch_from_prepared(prepared, identity_public_pair=np.ones((2, 2), dtype=np.int64))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "impl"))
        from channel import GateWeights as PublicGateWeights
        weights = GateWeights(np.zeros((12, 16)), np.zeros(16),
                              np.zeros((16, 16)), np.arange(16, dtype=float))
        deployed = as_public_gate(weights, PublicGateWeights)
        self.assertIsInstance(deployed, PublicGateWeights)
        np.testing.assert_array_equal(deployed.logits(np.zeros(12)), weights.b2)

    def test_private_primary_is_unnormalized_joint_plus_half_locals(self):
        batch = _batch(sex=[0, 1, 0, 1])
        metrics = evaluate_selection(batch, np.array([0, 1, 0, 1]),
                                     caps=(1.0, 1.0, 1.0, 1.0), mode="private",
                                     local_caps=(1.0, 1.0))
        self.assertEqual(metrics.recovery, (1.0, 1.0, 1.0))
        self.assertAlmostEqual(metrics.primary, 2.0)
        hidden = evaluate_selection(batch, np.zeros(4, dtype=np.int64),
                                    caps=(1.0, 1.0, 1.0, 1.0), mode="private",
                                    local_caps=(1.0, 1.0))
        self.assertAlmostEqual(hidden.primary, 1.0)

    def test_task_worker_refuses_sex_and_private_worker_requires_it(self):
        with self.assertRaises(ValueError):
            fit_deterministic(_batch(sex=[0, 1, 0, 1]), _config(mode="task"),
                              caps=(0.2, 0.2, 0.2, 0.2))
        with self.assertRaises(ValueError):
            fit_deterministic(_batch(), _config(mode="private"),
                              caps=(0.2, 0.2, 0.2, 0.2), local_caps=(1.0, 1.0))

    def test_two_independent_starts_complete_with_deployable_gate(self):
        result = fit_deterministic(_batch(), _config(mode="task"),
                                   caps=(0.2, 0.2, 0.2, 0.2))
        self.assertTrue(result.complete)
        self.assertTrue(result.metrics.feasible)
        self.assertEqual(result.metrics.occupied_pairs, 1)
        self.assertEqual(tuple(x.start for x in result.starts),
                         ("identity_biased", "uniform"))
        self.assertTrue(all(x.optimizer_steps == 2 for x in result.starts))
        self.assertTrue(all(x.polish_trials == 64 for x in result.starts))
        self.assertEqual(result.weights.w1.shape, (12, 16))
        self.assertEqual(result.weights.w2.shape, (16, 16))


if __name__ == "__main__":
    unittest.main()
