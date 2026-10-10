"""Guard against central-arm completion falling into another arm branch."""

import unittest
from unittest.mock import patch

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import FixedBank
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import GateWeights, JointPolicy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.fit import TaskOnlyData, ProtectedData
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import IdentityPolicy
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_dispatch import fit_arm


class CentralCompletion(unittest.TestCase):
    def test_completed_t_and_j_return_selected_policy(self):
        b1 = FixedBank(np.array([[.8, .2]], dtype=np.float64),
                       np.array([0], dtype=np.int64), (0,))
        b2 = FixedBank(np.array([[.6, .1, .1, .1, .05, .05]], dtype=np.float64),
                       np.array([0], dtype=np.int64), (0,))
        base = JointPolicy(b1, b2, ((0,),), ((0,),), GateWeights.zeros(input_width=12))
        task = TaskOnlyData(np.tile(b1.probability[0], (2, 1)),
                            np.tile(b2.probability[0], (2, 1)),
                            np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64),
                            np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64))
        selected = IdentityPolicy(base)
        with patch("results.pcrl_bank_anchored_release_v1.bank_anchored.r2_dispatch.fit_central_boundary",
                   return_value={"complete": True, "policy": selected,
                                 "selection": {"identity_alias": True,
                                               "start": "exact_identity", "update": 0},
                                 "source": {"complete": True}}):
            for arm, capability in (("T", task),
                                    ("J", ProtectedData(task, np.array([0, 1], dtype=np.int64)))):
                policy, summary = fit_arm(
                    arm, capability, base, source_seed=0, recipe_seed=1,
                    cpu_seconds=1., unit_ids=[0, 1], source_binding_sha256="a"*64)
                self.assertIsInstance(policy, IdentityPolicy)
                self.assertEqual(summary["status"], "COMPLETE_IDENTITY_ALIAS")


if __name__ == "__main__":
    unittest.main()
