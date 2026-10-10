"""Selected control package restores actual public law and mechanism hash."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "impl"))
from bank import FixedBank
from channel import GateWeights, JointPolicy
from policy_package import save_policy_package
from release import _bank_hash, _mechanism_hash
from source_bridge import action_map_sha256

from deterministic_policy import DeterministicJointPolicy
from factor_controls import (FactorGate, ProductPolicy, LocalPolicy,
                             SequentialPolicy, public_probability_embedding)
from control_package import save_control_policy_package, load_selected_policy_package


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
    def test_common_loader_dispatches_existing_joint_package(self):
        s1, s2, b1, b2, a1, a2, gate, _ = fixture()
        policy = JointPolicy(b1, b2, a1, a2, gate)
        with tempfile.TemporaryDirectory() as directory:
            receipt = save_policy_package(
                directory, policy, arm_id="joint-tiny", release_id="joint-release",
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
            ProductPolicy(b1, b2, a1, a2, gate, factor(8), factor(8)),
            LocalPolicy(b1, b2, a1, a2, gate, factor(4), factor(4)),
            SequentialPolicy(b1, b2, a1, a2, gate, factor(8),
                             factor(8, public_probability_embedding(b1.probability)),
                             order="12", first_checkpoint_sha256="a" * 64),
        )
        for policy in policies:
            with self.subTest(law=policy.law_kind), tempfile.TemporaryDirectory() as directory:
                receipt = save_control_policy_package(
                    directory, policy, arm_id=policy.law_kind,
                    release_id="tiny-release", source_seed=0,
                    source_binding_sha256="0" * 64,
                    row_unit_pair_sha256="1" * 64,
                    bank_sha256=(_bank_hash(b1), _bank_hash(b2)),
                    action_sha256=(action_map_sha256(a1), action_map_sha256(a2)),
                    fit_unit_id="tiny-fit", start="identity_biased", update=0,
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


if __name__ == "__main__":
    unittest.main()
