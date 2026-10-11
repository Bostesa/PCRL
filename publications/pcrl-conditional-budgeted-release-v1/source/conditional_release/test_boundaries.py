"""Explicit equality tests for the frozen strict/non-strict endpoint rules."""
import json
from pathlib import Path
import unittest

from .inference import lower_strict_status, lower_target_status, upper_cap_status


class Boundaries(unittest.TestCase):
    def test_strict_positive_effect(self):
        p = Path(__file__).resolve().parents[1] / "results/pcrl_conditional_budgeted_release_v1/r5/BOUNDARY_FIXTURES.json"
        fixture = json.loads(p.read_text())
        for case in fixture["strict_cases"]:
            self.assertEqual(lower_strict_status(case, fixture["target"]), case["status"])

    def test_unchanged_nonstrict_service_rules(self):
        self.assertEqual(lower_target_status({"lower":0.02,"upper":0.03}, 0.02), "PASS")
        self.assertEqual(upper_cap_status({"lower":0.0,"upper":0.01}, 0.01), "PASS")
        self.assertEqual(upper_cap_status({"lower":0.01,"upper":0.02}, 0.01), "UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
