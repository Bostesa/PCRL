import unittest

from .witness_worker import SCHEMA, SOURCES, SPEC_KEYS, validate_spec


class WitnessWorkerContractTests(unittest.TestCase):
    def setUp(self):
        sha = "c" * 64
        self.spec = {key: "/private/file" for key in SPEC_KEYS}
        self.spec.update({
            "schema": SCHEMA, "unit_id": "R4_B0_S0", "arm_id": "B0",
            "source_seed": 0, "realization": 0, "attacker_seed": 17,
            "release_path_by_role": {
                "ATTACK_FIT_NEW": "/private/att.npz",
                "INNER_SELECTION": "/private/inner.npz"},
            "release_sha256_by_role": {
                "ATTACK_FIT_NEW": sha, "INNER_SELECTION": sha},
            "source_pins": [{"path": f"/sources/{name}", "sha256": sha}
                            for name in sorted(SOURCES)],
        })
        for key in SPEC_KEYS:
            if key.endswith("sha256"):
                self.spec[key] = sha

    def test_b0_and_wq_have_distinct_decoder_contracts(self):
        validate_spec(self.spec)
        validate_spec(self.spec | {"arm_id": "W_Q", "decoder_sha256": None})
        with self.assertRaisesRegex(ValueError, "does not have"):
            validate_spec(self.spec | {"arm_id": "W_Q"})

    def test_rejects_missing_role_or_source(self):
        with self.assertRaisesRegex(ValueError, "both witness"):
            validate_spec(self.spec | {"release_path_by_role":
                                       {"ATTACK_FIT_NEW": "/private/att.npz"}})
        with self.assertRaisesRegex(ValueError, "source inventory"):
            validate_spec(self.spec | {"source_pins": self.spec["source_pins"][:-1]})


if __name__ == "__main__":
    unittest.main()
