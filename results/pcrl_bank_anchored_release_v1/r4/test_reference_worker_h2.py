"""Data-free structural checks for the P* H2 successor worker."""

import unittest

from .reference_worker_h2 import REQUIRED_SOURCES, SPEC_KEYS, SPEC_SCHEMA, validate_spec


class PStarH2WorkerTests(unittest.TestCase):
    def test_exact_h2_source_inventory_and_id(self):
        sha = "a" * 64
        spec = {key: "/private/path" for key in SPEC_KEYS}
        spec.update({"schema": SPEC_SCHEMA,
                     "unit_id": "R4_ATTACK_P_STAR_S0_R0_H2",
                     "arm_id": "P_STAR", "source_seed": 0,
                     "realization": 0, "attacker_seed": 100000,
                     "source_pins": [{"path": f"/sources/{name}", "sha256": sha}
                                     for name in sorted(REQUIRED_SOURCES)]})
        for key in SPEC_KEYS:
            if key.endswith("sha256"):
                spec[key] = sha
        validate_spec(spec)
        self.assertIn("pstar_views_h2.py", REQUIRED_SOURCES)
        self.assertNotIn("reference_worker.py", REQUIRED_SOURCES)
        with self.assertRaises(ValueError):
            validate_spec(spec | {"source_pins": spec["source_pins"][:-1]})


if __name__ == "__main__":
    unittest.main()
