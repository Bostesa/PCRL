import unittest

from . import witness_worker
from .witness_worker import SCHEMA, SOURCES, SPEC_KEYS, validate_spec


class WitnessWorkerContractTests(unittest.TestCase):
    def test_b0_inner_gate_v2_and_wq_historical_gate(self):
        b0 = {"schema": "pcrl-bank-r3-inner-attack-eligibility-v2",
              "arm_id": "B0", "inner_clauses": {"passed": 24, "total": 24},
              "inner_original_caps_pass": True,
              "all_two_seed_task_reference_point_caps_pass": True}
        witness_worker.validate_attack_gate(b0, "B0")
        with self.assertRaises(PermissionError):
            witness_worker.validate_attack_gate(
                b0 | {"schema": "pcrl-bank-r3-arm-point-eligibility-v1"}, "B0")
        wq = {"schema": "pcrl-bank-r3-historical-original-inner-eligibility-v1",
              "arm_id": "W_Q", "original_inner_point_clauses_passed": 24,
              "original_inner_point_clauses_total": 24,
              "all_two_seed_task_reference_point_caps_pass": True}
        witness_worker.validate_attack_gate(wq, "W_Q")

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
