"""Data-free package round trips and source-byte refusal."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import FixedBank
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import GateWeights, JointPolicy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.factor_controls import FactorGate, ProductPolicy
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import (
    IdentityPolicy, BoundaryJointPolicy, BoundaryFactorPolicy,
)
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_package import (
    save_policy, load_policy,
)


def fixture():
    bank = FixedBank(np.array([[.8, .2], [.7, .3]], dtype=np.float64),
                     np.array([0, 0], dtype=np.int64), (0, 1))
    base = JointPolicy(bank, bank, ((0, 1),), ((0, 1),), GateWeights.zeros(input_width=8))
    factor = FactorGate(np.zeros((8, 16)), np.zeros(16),
                        np.zeros((16, 4)), np.zeros(4))
    product = ProductPolicy(bank, bank, ((0, 1),), ((0, 1),),
                            base.gate, factor, factor)
    return base, product


class PackageTests(unittest.TestCase):
    def _roundtrip(self, arm, policy):
        with tempfile.TemporaryDirectory() as path:
            os.chmod(path, 0o700)
            receipt = save_policy(path, policy, arm_id=arm, source_seed=0,
                                  source_binding_sha256="a" * 64,
                                  fit_receipt_sha256="b" * 64,
                                  selected_start="identity_biased",
                                  selected_update=25)
            loaded, observed = load_policy(path, receipt_sha256=receipt["receipt_sha256"])
            s = np.array([.85, .15], dtype=np.float64)
            self.assertEqual(policy.sparse_law(s, s, 0, 0),
                             loaded.sparse_law(s, s, 0, 0))
            self.assertEqual(receipt["policy_file_sha256"], observed["policy_file_sha256"])
            with self.assertRaises(ValueError):
                load_policy(path, receipt_sha256="0" * 64)

    def test_identity_joint_and_product_round_trip(self):
        base, product = fixture()
        self._roundtrip("B0", IdentityPolicy(base))
        self._roundtrip("J", BoundaryJointPolicy(base, .23))
        self._roundtrip("P", BoundaryFactorPolicy(product, .11, .37))


if __name__ == "__main__":
    unittest.main()
