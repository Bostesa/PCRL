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
                  "sources": [{"path": "/x/base", "sha256": "a" * 64}],
                  "deps": [], "outputs": list(base["outputs"].values())}
        release = {"schema": "BAR1-role-release-v1", "arm_id": "J",
                   "source_seed": 0, "role": "INNER_SELECTION", "realization": 0,
                   "law_sha256": "b" * 64}
        args = dict(base=base, source_spec=source, release=release, arm="J",
                    unit_id="R3_J_S0_INNER", law_record={"path": "/x/law", "sha256": "b" * 64},
                    release_record={"path": "/x/release", "sha256": "c" * 64},
                    amendment_record={"path": "/x/amend", "sha256": "d" * 64},
                    config_path="/x/.private/new_config.json",
                    outputs={"csv": "/x/new.csv", "private_vectors": "/x/.private/new.npz",
                             "receipt": "/x/new.json"}, deps=["R2_J_S0_INNER"])
        cfg, spec = derive_documents(**args)
        self.assertEqual(cfg["stage"], "protection")
        self.assertEqual(cfg["law"]["sha256"], "b" * 64)
        self.assertEqual(spec["deps"], ["R2_J_S0_INNER"])
        self.assertEqual(spec["amendment_sha256"], "d" * 64)
        with self.assertRaises(PermissionError):
            derive_documents(**(args | {"release": release | {"law_sha256": "e" * 64}}))
        with self.assertRaises(PermissionError):
            derive_documents(**(args | {"outputs": base["outputs"]}))


if __name__ == "__main__":
    unittest.main()
