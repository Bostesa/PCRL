import unittest

from .reference_worker import (LOCK_SCHEMA, REQUIRED_SOURCES, SPEC_KEYS,
                               SPEC_SCHEMA, validate_live_sources, validate_spec,
                               validate_stage_lock)


class FreshPStarWorkerTests(unittest.TestCase):
    def setUp(self):
        self.sha = "a" * 64
        self.spec = {key: "/private/path" for key in SPEC_KEYS}
        self.spec.update({"schema": SPEC_SCHEMA, "unit_id": "R4_PSTAR_S0",
                          "arm_id": "P_STAR", "source_seed": 0,
                          "realization": 0, "attacker_seed": 17,
                          "source_pins": [
                              {"path": f"/sources/{name}", "sha256": self.sha}
                              for name in sorted(REQUIRED_SOURCES)]})
        for key in SPEC_KEYS:
            if key.endswith("sha256"):
                self.spec[key] = self.sha

    def test_exact_spec(self):
        validate_spec(self.spec)

    def test_wrong_arm_or_realization_refused(self):
        for change in ({"arm_id": "W_TSTAR"}, {"realization": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_spec(self.spec | change)

    def test_missing_transitive_source_refused(self):
        with self.assertRaisesRegex(ValueError, "source inventory"):
            validate_spec(self.spec | {"source_pins": self.spec["source_pins"][:-1]})

    def test_unregistered_stage_lock_refused_before_state_read(self):
        lock = {"schema": LOCK_SCHEMA, "status": "LOCALLY_SEALED_ATTACKS",
                "campaign_seal_sha256": self.sha, "units": {}}
        with self.assertRaisesRegex(PermissionError, "did not admit"):
            validate_stage_lock(lock, self.spec, self.sha,
                                stage_lock_sha256=self.sha)

    def test_live_source_inventory_rejects_unrelated_same_named_file(self):
        with self.assertRaises((PermissionError, FileNotFoundError)):
            validate_live_sources(self.spec["source_pins"])


if __name__ == "__main__":
    unittest.main()
