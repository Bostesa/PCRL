import unittest

from bank_anchored.r5_prepare_utility import verify_provenance


class IndependentReplayAdmission(unittest.TestCase):
    def fixture(self):
        sha = lambda letter: letter * 64
        config = {"schema": "pcrl-bank-r3-utility-job-v1", "stage": "protection",
                  "unit_id": "R3_J_S0_INNER_G1", "arm": "J", "seed": 0,
                  "role": "INNER_SELECTION", "law": {"sha256": sha("a")},
                  "seal": {"sha256": sha("i")}}
        index = {"schema": "pcrl-bank-r3-protection-unit-index-v1",
                 "unit_id": config["unit_id"], "arm": "J", "seed": 0,
                 "role": "INNER_SELECTION", "config_sha256": sha("b"),
                 "queue_spec_sha256": sha("c"),
                 "r2_law_sha256": sha("a"),
                 "r2_release_receipt_sha256": sha("d"),
                 "r2_policy_receipt_sha256": sha("e"),
                 "r2_selected_unit_receipt_sha256": sha("f"),
                 "r2_selected_unit_id": "R2_J_S0_FIT_G1",
                 "source_binding_sha256": sha("h")}
        spec = {"unit_id": config["unit_id"], "stage": "protection",
                "deps": ["R2_J_S0_FIT_G1"],
                "sources": [{"path": "/tmp/config", "sha256": sha("b")},
                            {"path": "/tmp/law", "sha256": sha("a")},
                            {"path": "/tmp/release", "sha256": sha("d")},
                            {"path": "/tmp/policy", "sha256": sha("e")},
                            {"path": "/tmp/unit", "sha256": sha("f")}]}
        r3 = {"schema": "pcrl-bank-r3-utility-receipt-v1",
              "unit_id": config["unit_id"], "arm": "J", "seed": 0,
              "role": "INNER_SELECTION", "law_sha256": sha("a"),
              "output_csv_sha256": sha("g")}
        release = {"schema": "BAR1-role-release-v1", "arm_id": "J",
                   "source_seed": 0, "role": "INNER_SELECTION", "realization": 0,
                   "law_sha256": sha("a"), "policy_receipt_sha256": sha("e"),
                   "source_binding_sha256": sha("h")}
        policy = {"schema": "BAR1-selected-policy-v1", "arm_id": "J",
                  "source_seed": 0, "source_binding_sha256": sha("h")}
        unit = {"schema": "BAR1-r2-unit-receipt-v1", "unit_id": "R2_J_S0_FIT_G1",
                "arm_id": "J", "source_seed": 0,
                "status": "COMPLETE_SELECTED_PROTECTED",
                "source_lock_sha256": sha("i"),
                "source_binding_sha256": sha("h"),
                "policy_receipt_sha256": sha("e"),
                "role_releases": [{"role": "INNER_SELECTION", "realization": 0,
                                   "law_sha256": sha("a"),
                                   "receipt_sha256": sha("d")} ]}
        return index, config, spec, r3, release, policy, unit

    def test_exact_selected_release_is_admitted(self):
        result = verify_provenance(*self.fixture(), config_sha="b" * 64,
                                   spec_sha="c" * 64, csv_sha="g" * 64)
        self.assertEqual(result["r2_unit_id"], "R2_J_S0_FIT_G1")

    def test_substituted_selected_unit_receipt_is_rejected(self):
        docs = list(self.fixture())
        docs[0]["r2_selected_unit_receipt_sha256"] = "z" * 64
        with self.assertRaises(PermissionError):
            verify_provenance(*docs, config_sha="b" * 64,
                              spec_sha="c" * 64, csv_sha="g" * 64)


if __name__ == "__main__":
    unittest.main()
