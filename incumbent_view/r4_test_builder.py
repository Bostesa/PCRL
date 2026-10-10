"""Data-free attack-spec and public-index admission checks."""
from pathlib import Path
import json
import tempfile
import unittest
from incumbent_view.r4_build_specs import _policy_pinset, _public_relative, _private_from_public, _ensure_custody_root
from incumbent_view.r4_aggregate import fixed_slots


class BuilderChecks(unittest.TestCase):
    def test_owner_attack_custody_root_exists_before_unit_specs(self):
        with tempfile.TemporaryDirectory() as name:
            path=_ensure_custody_root(Path(name)/".private")
            self.assertTrue(path.is_dir())
            self.assertEqual(path.stat().st_mode&0o077,0)

    def test_generated_unit_ids_equal_prefit_registry(self):
        path=Path(__file__).parents[1]/"results/pcrl_incumbent_view_risk_v1/r4/PLANNED_UNITS.json"
        planned=json.loads(path.read_text())
        self.assertEqual([unit["unit_id"] for unit in planned["units"]],
                         [unit["unit_id"] for unit in fixed_slots()])

    def test_p0_identity_has_no_selected_package(self):
        self.assertIsNone(_policy_pinset("P0",{"policy_package":None}))
        with self.assertRaises(PermissionError):
            _policy_pinset("P0",{"policy_package":{"receipt":{},"archive":{}}})

    def test_selected_package_requires_both_pins(self):
        for value in (None,{}, {"receipt":{}}, {"archive":{}}):
            with self.subTest(value=value),self.assertRaises(ValueError):
                _policy_pinset("V_J",{"policy_package":value})

    def test_public_index_has_relative_private_locator_only(self):
        with tempfile.TemporaryDirectory() as name:
            repo=Path(name)
            path=repo/"results/pcrl_incumbent_view_risk_v1/r4/.private/worker_specs/U.json"
            locator=_public_relative(path,repo)
            self.assertFalse(locator.startswith("/"))
            self.assertNotIn(name,locator)
            self.assertEqual(_private_from_public(locator,repo),path)
            for bad in ("/tmp/elsewhere", "../private/spec.json", "results/not-private.json"):
                with self.subTest(bad=bad),self.assertRaises(PermissionError):
                    _private_from_public(bad,repo)


if __name__=="__main__":unittest.main()
