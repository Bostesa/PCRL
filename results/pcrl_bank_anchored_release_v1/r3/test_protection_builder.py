"""Data-free protection unit assembly checks."""

import unittest

from bank_anchored.r3_make_protection_config import derive_documents


class ProtectionBuilderTests(unittest.TestCase):
    def test_law_receipt_identity_and_unique_outputs_are_required(self):
        base = {"schema": "pcrl-bank-r3-utility-job-v1", "stage": "witness",
                "arm": "B0", "seed": 0, "role": "INNER_SELECTION",
                "unit_id": "R3_B0_S0_INNER-R1", "law": None,
                "outputs": {"csv": "/x/old.csv", "private_vectors": "/x/.private/old.npz",
                            "receipt": "/x/old.json"}}
        source = {"unit_id": base["unit_id"], "stage": "witness",
                  "argv": ["/opt/homebrew/opt/python@3.14/bin/python3.14",
                           "-m", "bank_anchored.r3_worker", "--config", "/x/base"],
                  "sources": [{"path": "/x/base", "sha256": "a" * 64}],
                  "deps": [], "outputs": list(base["outputs"].values())}
        release = {"schema": "BAR1-role-release-v1", "arm_id": "J",
                   "source_seed": 0, "role": "INNER_SELECTION", "realization": 0,
                   "law_sha256": "b" * 64, "policy_receipt_sha256": "e" * 64,
                   "source_binding_sha256": "f" * 64}
        policy = {"schema": "BAR1-selected-policy-v1", "arm_id": "J",
                  "source_seed": 0, "source_binding_sha256": "f" * 64,
                  "policy_file_sha256": "g" * 64}
        identity = {"schema": "BAR1-r2-full-fit-identity-v1", "status": "PASS",
                    "source_seed": 0, "role": "OSF_DEFENSE_FIT",
                    "source_lock_sha256": "a" * 64,
                    "source_binding_sha256": "f" * 64}
        selected_unit = {"schema": "BAR1-r2-unit-receipt-v1",
                         "status": "COMPLETE_PROTECTED",
                         "unit_id": "R2_J_S0_FIT_F1", "arm_id": "J", "source_seed": 0,
                         "source_lock_sha256": "a" * 64,
                         "source_binding_sha256": "f" * 64,
                         "policy_receipt_sha256": "e" * 64,
                         "role_releases": [{"role": "INNER_SELECTION", "realization": 0,
                                            "law_sha256": "b" * 64,
                                            "receipt_sha256": "c" * 64}]}
        base["seal"] = {"path": "/x/seal", "sha256": "a" * 64}
        args = dict(base=base, source_spec=source, release=release, arm="J",
                    unit_id="R3_J_S0_INNER", law_record={"path": "/x/law", "sha256": "b" * 64},
                    release_record={"path": "/x/release", "sha256": "c" * 64},
                    policy=policy, policy_record={"path": "/x/policy_receipt", "sha256": "e" * 64},
                    policy_package_record={"path": "/x/policy.npz", "sha256": "g" * 64},
                    identity=identity, identity_record={"path": "/x/identity", "sha256": "h" * 64},
                    selected_unit=selected_unit,
                    selected_unit_record={"path": "/x/unit_receipt", "sha256": "i" * 64},
                    amendment_record={"path": "/x/amend", "sha256": "d" * 64},
                    config_path="/x/.private/new_config.json",
                    outputs={"csv": "/x/new.csv", "private_vectors": "/x/.private/new.npz",
                             "receipt": "/x/new.json"}, deps=["R2_J_S0_FIT_F1"])
        cfg, spec = derive_documents(**args)
        self.assertEqual(cfg["stage"], "protection")
        self.assertEqual(cfg["law"]["sha256"], "b" * 64)
        self.assertEqual(spec["deps"], ["R2_J_S0_FIT_F1"])
        self.assertIn(args["selected_unit_record"], spec["sources"])
        self.assertEqual(spec["argv"][0], source["argv"][0])
        self.assertEqual(spec["amendment_sha256"], "d" * 64)
        self.assertIn(args["policy_package_record"], spec["sources"])
        self.assertIn(args["identity_record"], spec["sources"])
        with self.assertRaises(PermissionError):
            derive_documents(**(args | {"release": release | {"law_sha256": "e" * 64}}))
        with self.assertRaises(PermissionError):
            derive_documents(**(args | {"outputs": base["outputs"]}))
        with self.assertRaises(PermissionError):
            derive_documents(**(args | {"release": release | {"policy_receipt_sha256": "z" * 64}}))
        with self.assertRaises(PermissionError):
            derive_documents(**(args | {"identity": identity | {"source_binding_sha256": "z" * 64}}))
        with self.assertRaises(PermissionError):
            derive_documents(**(args | {"source_spec": source | {"argv": ["python3"]}}))
        with self.assertRaises(PermissionError):
            derive_documents(**(args | {"selected_unit": selected_unit | {
                "policy_receipt_sha256": "z" * 64}}))


if __name__ == "__main__":
    unittest.main()
