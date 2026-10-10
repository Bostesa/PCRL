"""Selected control package restores actual public law and mechanism hash."""

import importlib
import tempfile
import unittest

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import FixedBank
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import GateWeights, JointPolicy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.policy_package import save_policy_package
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.release import _bank_hash, _mechanism_hash
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.source_bridge import action_map_sha256

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.deterministic_policy import DeterministicJointPolicy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.factor_controls import (
    FactorGate, ProductPolicy, LocalPolicy, SequentialPolicy,
    public_probability_embedding)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.control_package import (
    save_control_policy_package, load_selected_policy_package)


def fixture():
    p1 = np.array([[.9, .1], [.6, .4]], dtype=np.float64)
    p2 = np.array([[.8, .2], [.7, .3]], dtype=np.float64)
    b1 = FixedBank.from_source(p1, np.arange(2), np.zeros(2, dtype=np.int64))
    b2 = FixedBank.from_source(p2, np.arange(2), np.zeros(2, dtype=np.int64))
    a1 = (tuple(map(int, b1.old_to_global)),)
    a2 = (tuple(map(int, b2.old_to_global)),)
    gate = GateWeights.zeros(input_width=8)
    def factor(width, embedding=None):
        return FactorGate(np.zeros((width, 16)), np.zeros(16),
                          np.zeros((16, 4)), np.zeros(4), embedding)
    return p1[0], p2[0], b1, b2, a1, a2, gate, factor


class ControlPackageTest(unittest.TestCase):
    def test_normal_package_fit_imports(self):
        prefix = "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls."
        for module in ("fit_product", "fit_local", "seq_train"):
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.import_module(prefix + module))

    def test_common_loader_dispatches_existing_joint_package(self):
        s1, s2, b1, b2, a1, a2, gate, _ = fixture()
        policy = JointPolicy(b1, b2, a1, a2, gate)
        with tempfile.TemporaryDirectory() as directory:
            receipt = save_policy_package(
                directory, policy, arm_id="A_JOINT_STOCH", release_id="joint-release",
                source_seed=0, source_binding_sha256="0" * 64,
                row_unit_pair_sha256="1" * 64,
                bank_sha256=(_bank_hash(b1), _bank_hash(b2)),
                action_sha256=(action_map_sha256(a1), action_map_sha256(a2)),
                fit_unit_id="tiny-fit", start="identity_biased", update=0,
                fit_receipt_sha256="2" * 64)
            restored, _ = load_selected_policy_package(
                directory, receipt_sha256=receipt["receipt_sha256"])
            self.assertEqual(policy.sparse_law(s1, s2, 0, 0),
                             restored.sparse_law(s1, s2, 0, 0))

    def test_each_control_law_restores_from_exact_private_package(self):
        s1, s2, b1, b2, a1, a2, gate, factor = fixture()
        policies = (
            DeterministicJointPolicy(b1, b2, a1, a2, gate),
            DeterministicJointPolicy(b1, b2, a1, a2, gate),
            ProductPolicy(b1, b2, a1, a2, gate, factor(8), factor(8)),
            LocalPolicy(b1, b2, a1, a2, gate, factor(4), factor(4)),
            SequentialPolicy(b1, b2, a1, a2, gate, factor(8),
                             factor(8, public_probability_embedding(b1.probability)),
                             order="12", first_checkpoint_sha256="a" * 64),
            SequentialPolicy(b1, b2, a1, a2, gate, factor(8),
                             factor(8, public_probability_embedding(b2.probability)),
                             order="21", first_checkpoint_sha256="b" * 64),
        )
        arms = ("A_TASK_DET", "A_JOINT_DET", "A_CENTRAL_PRODUCT", "A_LOCAL",
                "A_SEQ12", "A_SEQ21")
        for arm_id, policy in zip(arms, policies, strict=True):
            with self.subTest(arm=arm_id), tempfile.TemporaryDirectory() as directory:
                receipt = save_control_policy_package(
                    directory, policy, arm_id=arm_id,
                    release_id="tiny-release", source_seed=0,
                    source_binding_sha256="0" * 64,
                    row_unit_pair_sha256="1" * 64,
                    bank_sha256=(_bank_hash(b1), _bank_hash(b2)),
                    action_sha256=(action_map_sha256(a1), action_map_sha256(a2)),
                    fit_unit_id="tiny-fit",
                    start=("composite" if arm_id in ("A_LOCAL", "A_SEQ12", "A_SEQ21")
                           else "identity_biased"), update=0,
                    fit_receipt_sha256="2" * 64)
                restored, checked = load_selected_policy_package(
                    directory, receipt_sha256=receipt["receipt_sha256"])
                self.assertEqual(_mechanism_hash(policy), _mechanism_hash(restored))
                self.assertEqual(policy.sparse_law(s1, s2, 0, 0),
                                 restored.sparse_law(s1, s2, 0, 0))
                self.assertEqual(checked["law_kind"], policy.law_kind)
                with self.assertRaises(ValueError):
                    load_selected_policy_package(directory,
                                                 receipt_sha256="f" * 64)

    def test_arm_law_kind_mismatch_is_rejected(self):
        _, _, b1, b2, a1, a2, gate, _ = fixture()
        policy = DeterministicJointPolicy(b1, b2, a1, a2, gate)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "arm.*law"):
                save_control_policy_package(
                    directory, policy, arm_id="A_SEQ21", release_id="tiny-release",
                    source_seed=0, source_binding_sha256="0" * 64,
                    row_unit_pair_sha256="1" * 64,
                    bank_sha256=(_bank_hash(b1), _bank_hash(b2)),
                    action_sha256=(action_map_sha256(a1), action_map_sha256(a2)),
                    fit_unit_id="tiny-fit", start="identity_biased", update=0,
                    fit_receipt_sha256="2" * 64)

    def test_sequential_order_mismatch_is_rejected(self):
        _, _, b1, b2, a1, a2, gate, factor = fixture()
        policy = SequentialPolicy(
            b1, b2, a1, a2, gate, factor(8),
            factor(8, public_probability_embedding(b1.probability)),
            order="12", first_checkpoint_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "sequential order"):
                save_control_policy_package(
                    directory, policy, arm_id="A_SEQ21", release_id="tiny-release",
                    source_seed=0, source_binding_sha256="0" * 64,
                    row_unit_pair_sha256="1" * 64,
                    bank_sha256=(_bank_hash(b1), _bank_hash(b2)),
                    action_sha256=(action_map_sha256(a1), action_map_sha256(a2)),
                    fit_unit_id="tiny-fit", start="identity_biased", update=0,
                    fit_receipt_sha256="2" * 64)


if __name__ == "__main__":
    unittest.main()
