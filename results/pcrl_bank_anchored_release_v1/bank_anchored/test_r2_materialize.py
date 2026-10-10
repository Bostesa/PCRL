"""Synthetic exact role law and cached release serialization checks."""

import os
import tempfile
import unittest

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import FixedBank
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import GateWeights, JointPolicy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.factor_controls import FactorGate, ProductPolicy
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import IdentityPolicy, BoundaryFactorPolicy, BoundaryJointPolicy
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_materialize import (
    LAW_KEYS, materialize_arrays, write_materialization,
)


class MaterializationTests(unittest.TestCase):
    def test_eta_one_full_sixteen_action_support_keeps_identity_slot(self):
        p1 = np.array([[v, 1-v] for v in (.8, .75, .7, .65)], dtype=np.float64)
        p2 = np.array([[v, (1-v)/5, (1-v)/5, (1-v)/5, (1-v)/5,
                        1-v-4*((1-v)/5)] for v in (.6, .58, .56, .54)],
                      dtype=np.float64)
        b1 = FixedBank(p1, np.zeros(4, dtype=np.int64), (0,1,2,3))
        b2 = FixedBank(p2, np.zeros(4, dtype=np.int64), (0,1,2,3))
        base = JointPolicy(b1, b2, ((0,1,2,3),), ((0,1,2,3),),
                           GateWeights.zeros(input_width=12))
        arrays = {"row_id": np.array([1], dtype=np.int64),
                  "unit": np.array([1], dtype=np.int64),
                  "score1": p1[:1], "score2": p2[:1],
                  "cell1": np.array([0], dtype=np.int64),
                  "cell2": np.array([0], dtype=np.int64)}
        law, _ = materialize_arrays(BoundaryJointPolicy(base, 1.0), **arrays,
            source_seed=0, role="INNER_SELECTION", realization=0,
            private_root=bytes(range(32)))
        self.assertEqual(int(law["valid"].sum()), 16)
        self.assertEqual((int(law["code_r1"][0,0]),
                          int(law["code_r2"][0,0])), (0,0))
        self.assertAlmostEqual(float(law["mass"].sum()), 1.0)

    def _fixture(self):
        bank1 = FixedBank(np.array([[.8, .2], [.7, .3]]),
                          np.array([0, 0], dtype=np.int64), (0, 1))
        bank2 = FixedBank(np.array([[.6, .1, .1, .1, .05, .05],
                                    [.5, .1, .1, .1, .1, .1]]),
                          np.array([0, 0], dtype=np.int64), (0, 1))
        base = JointPolicy(bank1, bank2, ((0, 1),), ((0, 1),),
                           GateWeights.zeros(input_width=12))
        s1 = np.array([[.8, .2], [.8, .2], [.8, .2]], dtype=np.float64)
        s2 = np.array([[.6, .1, .1, .1, .05, .05]] * 3, dtype=np.float64)
        arrays = {"row_id": np.array([10, 11, 12], dtype=np.int64),
                  "unit": np.array([5, 5, 6], dtype=np.int64),
                  "score1": s1, "score2": s2,
                  "cell1": np.zeros(3, dtype=np.int64),
                  "cell2": np.zeros(3, dtype=np.int64)}
        return base, arrays

    def test_identity_and_product_release_have_exact_schema(self):
        base, arrays = self._fixture()
        root = bytes(range(32))
        law, release = materialize_arrays(IdentityPolicy(base), **arrays,
            source_seed=0, role="INNER_SELECTION", realization=0, private_root=root)
        self.assertEqual(set(law), LAW_KEYS)
        self.assertEqual(law["mass"].shape, (3, 16))
        np.testing.assert_array_equal(law["mass"][:, 0], np.ones(3))
        self.assertTrue(np.all(law["mass"][:, 1:] == 0))
        self.assertTrue(np.all(release["code_r1"] == 0))
        factor = FactorGate(np.zeros((12, 16)), np.zeros(16),
                            np.zeros((16, 4)), np.zeros(4))
        product = ProductPolicy(base.bank1, base.bank2, base.actions1,
                                base.actions2, base.gate, factor, factor)
        protected = BoundaryFactorPolicy(product, .3, .5)
        law2, rel2 = materialize_arrays(protected, **arrays,
            source_seed=0, role="INNER_SELECTION", realization=1, private_root=root)
        self.assertTrue(np.allclose(law2["mass"].sum(axis=1), 1))
        self.assertEqual(rel2["code_r1"][0], rel2["code_r1"][1])
        self.assertEqual(rel2["code_r2"][0], rel2["code_r2"][1])
        with tempfile.TemporaryDirectory() as path:
            os.chmod(path, 0o700)
            receipt = write_materialization(path, law2, rel2, arm_id="P",
                source_seed=0, role="INNER_SELECTION", realization=1,
                source_binding_sha256="a" * 64, policy_receipt_sha256="b" * 64)
            self.assertEqual(receipt["rows"], 3)
            with np.load(f"{path}/law.npz", allow_pickle=False) as stored:
                self.assertEqual(set(stored.files), LAW_KEYS)
            with self.assertRaises(FileExistsError):
                write_materialization(path, law2, rel2, arm_id="P", source_seed=0,
                    role="INNER_SELECTION", realization=1,
                    source_binding_sha256="a" * 64, policy_receipt_sha256="b" * 64)


if __name__ == "__main__":
    unittest.main()
