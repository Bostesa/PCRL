"""Data-free bank attack lock and direct-run refusal tests."""

import unittest

from .attack_worker import READER_PLAN
from .bank_attack_worker import validate_bank_attack_lock


class BankAttackWorkerTest(unittest.TestCase):
    def test_public_bank_lock_binds_table_policy_and_two_releases(self):
        unit = {"arm_id": "W_Q", "seed": 0, "bits": 10,
                "source_model_sha256": "s" * 64,
                "public_table_sha256": "t" * 64,
                "policy_sha256": "p" * 64,
                "decoder_sha256": None,
                "release_sha256": {"ATTACK_FIT_NEW": "a" * 64,
                                   "INNER_SELECTION": "i" * 64}}
        lock = {"schema": "pcrl-code-witness-bank-attack-lock-v1",
                "status": "LOCALLY_SEALED_ATTACKS",
                "campaign_seal_sha256": "c" * 64,
                "role_manifest_sha256": "r" * 64,
                "worker_sha256": "w" * 64,
                "reader_plan": READER_PLAN,
                "units": {"ATTACK_W_Q_S0": unit}}
        args = {"campaign_seal_sha256": "c" * 64,
                "role_manifest_sha256": "r" * 64,
                "worker_sha256": "w" * 64,
                "unit_id": "ATTACK_W_Q_S0", **unit}
        validate_bank_attack_lock(lock, **args)
        args["public_table_sha256"] = "z" * 64
        with self.assertRaisesRegex(PermissionError, "bank attack lock"):
            validate_bank_attack_lock(lock, **args)


if __name__ == "__main__":
    unittest.main()
