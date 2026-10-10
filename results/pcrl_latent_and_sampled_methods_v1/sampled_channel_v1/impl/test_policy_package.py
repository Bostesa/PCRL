"""Data-free custody tests for one fitted stochastic score policy."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bank import FixedBank
from channel import GateWeights, JointPolicy
from policy_package import load_policy_package, save_policy_package
from release import _bank_hash, _mechanism_hash
from source_bridge import action_map_sha256


class PolicyPackageTest(unittest.TestCase):
    def _policy(self):
        p = np.array([[.9, .1], [.7, .3]], dtype=np.float64)
        bank = FixedBank.from_source(p, np.arange(2, dtype=np.int64),
                                     np.zeros(2, dtype=np.int64))
        gate = GateWeights.zeros(input_width=8)
        return JointPolicy(bank, bank, ((0, 1),), ((0, 1),), gate)

    def test_write_once_restore_exact_law_and_tamper(self):
        policy = self._policy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private"
            root.mkdir(mode=0o700)
            receipt = save_policy_package(
                root, policy, arm_id="JOINT-STOCH", release_id="JOINT-STOCH-s0",
                source_seed=0, source_binding_sha256="a" * 64,
                row_unit_pair_sha256="b" * 64,
                bank_sha256=(_bank_hash(policy.bank1), _bank_hash(policy.bank2)),
                action_sha256=(action_map_sha256(policy.actions1),
                               action_map_sha256(policy.actions2)),
                fit_unit_id="A-JOINT-STOCH-S0", start="identity_biased",
                update=300, fit_receipt_sha256="1" * 64)
            restored, metadata = load_policy_package(root, receipt_sha256=receipt["receipt_sha256"])
            self.assertEqual(_mechanism_hash(restored), _mechanism_hash(policy))
            self.assertEqual(metadata["source_seed"], 0)
            self.assertEqual(metadata["arm_id"], "JOINT-STOCH")
            self.assertEqual(restored.sparse_law(np.array([.8, .2]), np.array([.8, .2]), 0, 0),
                             policy.sparse_law(np.array([.8, .2]), np.array([.8, .2]), 0, 0))
            with self.assertRaises(FileExistsError):
                save_policy_package(root, policy, arm_id="JOINT-STOCH",
                                    release_id="JOINT-STOCH-s0",
                                    source_seed=0, source_binding_sha256="a" * 64,
                                    row_unit_pair_sha256="b" * 64,
                                    bank_sha256=(_bank_hash(policy.bank1), _bank_hash(policy.bank2)),
                                    action_sha256=(action_map_sha256(policy.actions1),
                                                   action_map_sha256(policy.actions2)),
                                    fit_unit_id="A-JOINT-STOCH-S0", start="identity_biased",
                                    update=300, fit_receipt_sha256="1" * 64)
            os.chmod(root / "policy.npz", 0o600)
            with (root / "policy.npz").open("ab") as output:
                output.write(b"tamper")
            with self.assertRaises(ValueError):
                load_policy_package(root, receipt_sha256=receipt["receipt_sha256"])


if __name__ == "__main__":
    unittest.main()
