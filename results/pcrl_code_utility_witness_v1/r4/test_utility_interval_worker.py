"""Synthetic preflight checks for the queue-only utility interval reader."""

import unittest

from .utility_interval_worker import validate_interval_config


class IntervalWorkerTest(unittest.TestCase):
    def test_config_binds_randomness_source_and_candidate_inventory(self):
        config = {
            "schema": "pcrl-r4-utility-interval-config-v1",
            "unit_id": "R4_UTILITY_INTERVALS", "worker_sha256": "w" * 64,
            "utility_source_sha256": "u" * 64,
            "inference_rule_source_sha256": "n" * 64,
            "finite_family_source_sha256": "f" * 64,
            "inference_proposal": {"path": "/proposal/INFERENCE_RULE_PROPOSAL.json",
                                   "sha256": "p" * 64},
            "finite_family_proposal": {"path": "/proposal/D_FINITE_FAMILY_PROPOSAL.json",
                                       "sha256": "q" * 64},
            "role_manifest": {"path": "/repair/m1/DATA_ACCESS_MANIFEST.json",
                              "sha256": "r" * 64},
            "seal": {"path": "/study/CAMPAIGN_LOCK.json", "sha256": "s" * 64},
            "inference_lock": {"path": "/study/INFERENCE_RANDOMNESS_LOCK.json",
                               "sha256": "i" * 64},
            "candidates": [{"arm_id": "T0_B10", "kind": "learned", "seed": 0,
                            "source_model_sha256": "m" * 64,
                            "receipt": {"path": "/receipt", "sha256": "x" * 64},
                            "release": {"path": "/private/release", "sha256": "y" * 64},
                            "service_manifest": {"path": "/private/manifest", "sha256": "z" * 64}}],
            "output": "/study/r4/UTILITY_INTERVALS.json",
        }
        validate_interval_config(config, unit_id="R4_UTILITY_INTERVALS",
                                 worker_sha256="w" * 64,
                                 utility_source_sha256="u" * 64,
                                 inference_rule_source_sha256="n" * 64,
                                 finite_family_source_sha256="f" * 64,
                                 seal_sha256="s" * 64,
                                 inference_lock_sha256="i" * 64)
        config["utility_source_sha256"] = "q" * 64
        with self.assertRaisesRegex(PermissionError, "interval config"):
            validate_interval_config(config, unit_id="R4_UTILITY_INTERVALS",
                                     worker_sha256="w" * 64,
                                     utility_source_sha256="u" * 64,
                                     inference_rule_source_sha256="n" * 64,
                                     finite_family_source_sha256="f" * 64,
                                     seal_sha256="s" * 64,
                                     inference_lock_sha256="i" * 64)


if __name__ == "__main__":
    unittest.main()
