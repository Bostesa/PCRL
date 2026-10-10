import unittest

from bank_anchored.r5_utility_links import validate_links


class ProtectedUtilityLinks(unittest.TestCase):
    def fixture(self):
        config = {"schema": "pcrl-bank-r3-utility-job-v1", "stage": "protection",
                  "unit_id": "R3_J_S0_INNER", "arm": "J", "seed": 0,
                  "role": "INNER_SELECTION", "law": {"sha256": "a" * 64}}
        r3 = {"schema": "pcrl-bank-r3-utility-receipt-v1", "unit_id": config["unit_id"],
              "arm": "J", "seed": 0, "role": "INNER_SELECTION",
              "law_sha256": "a" * 64, "output_csv_sha256": "b" * 64}
        release = {"schema": "BAR1-role-release-v1", "arm_id": "J",
                   "source_seed": 0, "role": "INNER_SELECTION", "realization": 0,
                   "law_sha256": "a" * 64, "policy_receipt_sha256": "c" * 64,
                   "source_binding_sha256": "d" * 64}
        policy = {"schema": "BAR1-selected-policy-v1", "arm_id": "J",
                  "source_seed": 0, "source_binding_sha256": "d" * 64}
        unit = {"schema": "BAR1-r2-unit-receipt-v1", "unit_id": "R2_J_S0_FIT_F1",
                "arm_id": "J", "source_seed": 0, "status": "COMPLETE_SELECTED_PROTECTED",
                "policy_receipt_sha256": "c" * 64,
                "role_releases": [{"role": "INNER_SELECTION", "realization": 0,
                                   "law_sha256": "a" * 64,
                                   "receipt_sha256": "e" * 64}]}
        return config, r3, release, policy, unit

    def test_exact_selected_law_passes(self):
        docs = self.fixture()
        result = validate_links(*docs, csv_sha="b" * 64,
                                release_sha="e" * 64, policy_sha="c" * 64)
        self.assertEqual(result["status"], "PASS")

    def test_different_evaluated_law_is_rejected(self):
        docs = list(self.fixture())
        docs[1]["law_sha256"] = "f" * 64
        with self.assertRaises(PermissionError):
            validate_links(*docs, csv_sha="b" * 64,
                           release_sha="e" * 64, policy_sha="c" * 64)


if __name__ == "__main__":
    unittest.main()
