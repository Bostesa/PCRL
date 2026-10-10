"""Data-free exact first-stage boundary and conditional deployment checks."""

import unittest

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import FixedBank
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import GateWeights
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.fit import (
    TaskOnlyData, ProtectedData, prepare_law,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.factor_controls import (
    FactorGate, SequentialPolicy, public_probability_embedding,
)
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_source_seq_train import (
    _first_arrays, _first_exact, _project_first,
)
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_sequential import (
    BoundarySequentialPolicy,
)


class FirstStageBoundaryTests(unittest.TestCase):
    def test_raw_first_fails_but_interior_projection_is_feasible(self):
        bank1 = FixedBank(np.array([[.95, .05], [.55, .45]], dtype=np.float64),
                          np.array([0, 0], dtype=np.int64), (0, 1))
        bank2 = FixedBank(np.array([[.95, .01, .01, .01, .01, .01]],
                                   dtype=np.float64),
                          np.array([0], dtype=np.int64), (0,))
        first = FactorGate(np.zeros((12, 16)), np.zeros(16),
                           np.zeros((16, 4)), np.zeros(4))
        second = FactorGate(np.zeros((12, 16)), np.zeros(16),
                            np.zeros((16, 4)), np.zeros(4),
                            public_probability_embedding(bank1.probability))
        template = SequentialPolicy(
            bank1, bank2, ((0, 1), (1, 0)), ((0,),),
            GateWeights.zeros(input_width=12), first, second,
            order="12", first_checkpoint_sha256="0"*64)
        n = 12
        cells = np.array([0]*9 + [1]*3, dtype=np.int64)
        task = TaskOnlyData(
            np.tile(np.array([.95, .05]), (n, 1)),
            np.tile(np.array([.95, .01, .01, .01, .01, .01]), (n, 1)),
            cells, np.zeros(n, dtype=np.int64),
            np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64))
        data = ProtectedData(task, np.array([0]*9+[1]*3, dtype=np.int64))
        prepared = prepare_law(task, template)
        arrays = _first_arrays(data, template)
        raw = _first_exact(data, prepared, first, arrays)
        base_nll = (9*(-np.log(.95)) + 3*(-np.log(.55)))/12
        base_brier = (9*(2*.05**2) + 3*(2*.45**2))/12
        limits = {"r1_nll": base_nll+.001,
                  "r1_brier": base_brier+.0005,
                  "r1_bayes": 1.0}
        self.assertGreater(raw["r1_nll"], limits["r1_nll"])
        projected = _project_first(data, prepared, first, arrays, limits)
        self.assertGreater(projected["eta"], 0.0)
        self.assertLess(projected["eta"], 1.0)
        self.assertLessEqual(projected["costs"]["r1_nll"], limits["r1_nll"])
        self.assertLessEqual(projected["costs"]["r1_brier"], limits["r1_brier"])
        self.assertLess(projected["bayes"], 1.0)
        policy = BoundarySequentialPolicy(template, projected["eta"], 0.0)
        law = policy.sparse_law(task.score1[0], task.score2[0], 0, 0)
        self.assertAlmostEqual(sum(p for _, _, p in law), 1.0)
        self.assertEqual({j for _, j, p in law if p > 0}, {0})
        self.assertGreater(sum(p for i, _, p in law if i == 1), 0.0)


if __name__ == "__main__":
    unittest.main()
