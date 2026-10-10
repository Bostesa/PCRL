"""Tiny fixed-law checks for exact B0-relative checkpoint projection."""

import unittest

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import FixedBank
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import GateWeights, JointPolicy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.fit import (
    TaskOnlyData, prepare_law, _probabilities_numpy,
)
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_fit_core import (
    PROXY_TOLERANCE, project_joint_segment,
)


class SegmentProjectionTests(unittest.TestCase):
    def test_identity_is_exactly_feasible_and_costs_are_expected(self):
        bank1 = FixedBank(np.array([[.8, .2], [.6, .4]], dtype=np.float64),
                          np.array([0, 0], dtype=np.int64), (0, 1))
        bank2 = FixedBank(np.array([[.7, .3], [.55, .45]], dtype=np.float64),
                          np.array([0, 0], dtype=np.int64), (0, 1))
        policy = JointPolicy(bank1, bank2, ((0, 1),), ((0, 1),),
                             GateWeights.zeros(input_width=8))
        n = 8
        data = TaskOnlyData(np.tile(np.array([.8, .2]), (n, 1)),
                            np.tile(np.array([.7, .3]), (n, 1)),
                            np.zeros(n, dtype=np.int64),
                            np.zeros(n, dtype=np.int64),
                            np.zeros(n, dtype=np.int64),
                            np.zeros(n, dtype=np.int64))
        prepared = prepare_law(data, policy)
        q1 = _probabilities_numpy(prepared, policy.gate)
        projected = project_joint_segment(prepared, q1, objective_kind="task_rate")
        self.assertTrue(0 <= projected["eta"] <= 1)
        self.assertAlmostEqual(projected["b0_fit_costs"]["r1_nll"], -np.log(.8))
        for key, cap in PROXY_TOLERANCE.items():
            self.assertLessEqual(projected["fit_excess"][key], cap + 1e-12)
        # Verify an actual expected loss, not loss of the average probability.
        q = np.zeros_like(q1)
        q[:, 0] = 1 - projected["eta"]
        q += projected["eta"] * q1
        self.assertAlmostEqual(projected["fit_costs"]["r1_nll"],
                               float(np.mean(np.sum(q * prepared.r1_nll, axis=1))))

    def test_source_law_must_be_normalized(self):
        bank = FixedBank(np.array([[.8, .2]], dtype=np.float64),
                         np.array([0], dtype=np.int64), (0,))
        policy = JointPolicy(bank, bank, ((0,),), ((0,),), GateWeights.zeros(input_width=8))
        data = TaskOnlyData(np.array([[.8, .2]]), np.array([[.8, .2]]),
                            np.array([0], dtype=np.int64), np.array([0], dtype=np.int64),
                            np.array([0], dtype=np.int64), np.array([0], dtype=np.int64))
        prepared = prepare_law(data, policy)
        with self.assertRaises(ValueError):
            project_joint_segment(prepared, np.zeros((1, 16)), objective_kind="task_rate")


if __name__ == "__main__":
    unittest.main()
