"""Data-free checks for the finite prelock core unit superset."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from results.pcrl_capacity_matched_feature_selection_v1.m0.unit_plan import (
    ROLE_FIELDS, build_core_unit_specs, build_endpoint_inference_specs,
    build_replay_specs,
)


class UnitPlanTests(unittest.TestCase):
    def test_independent_replay_potential_roster(self):
        with tempfile.TemporaryDirectory() as location:
            base = Path(location)
            repo = base / "repo"
            study = repo / "results" / "pcrl_capacity_matched_feature_selection_v1"
            (study / "m5").mkdir(parents=True)
            (study / "m5/replay_worker.py").write_text("pass\n")
            (study / "m5/replay_aggregate.py").write_text("pass\n")
            prior = base / "prior"
            prior.mkdir()
            role = prior / "role.npz"
            role.write_bytes(b"synthetic admitted fixture")
            digest = hashlib.sha256(role.read_bytes()).hexdigest()
            keys = set().union(*(set(ROLE_FIELDS["independent_replay"](s))
                                 for s in (0, 1, 2)))
            manifest = {"role_file_relative_path": {key: "role.npz" for key in keys},
                        "role_file_sha256": {key: digest for key in keys}}
            geometry = json.loads((Path(__file__).resolve().parents[1] /
                                   "m1/CAPACITY_SPEC.json").read_text())
            caps = {stage: {"cpu_reserve_s": 60, "wall_limit_s": 60,
                            "max_rss_gib": 1}
                    for stage in ("independent_replay", "independent_aggregate")}
            specs = build_replay_specs(
                worktree=repo, study=study, prior=prior, manifest=manifest,
                source_map_sha="manifest-hash", geometry=geometry, caps=caps,
                python=Path(sys.executable))
            self.assertEqual(len(specs), 77)
            pairs = [s for s in specs.values() if s["stage"] == "independent_replay"]
            self.assertEqual(len(pairs), 75)
            self.assertTrue(all(len(s["outputs"]) == 2 and
                                "OSF_DEFENSE_FIT:sex" not in s["role_inputs"]
                                for s in pairs))
            for phase in ("SCREEN", "FINALIST"):
                aggregate = specs["M5-REPLAY-AGGREGATE-" + phase]
                self.assertEqual(aggregate["data_access"], "NONE_PRIVATE_DERIVED_ONLY")
                self.assertEqual(aggregate["role_inputs"], {})
                self.assertEqual(aggregate["deps"],
                                 ["M4-ENDPOINTS-" + phase, "M3-INFERENCE-" + phase])

    def test_endpoint_and_derived_only_inference_rosters(self):
        with tempfile.TemporaryDirectory() as location:
            base = Path(location)
            repo = base / "repo"
            study = repo / "results" / "pcrl_capacity_matched_feature_selection_v1"
            (study / "m3").mkdir(parents=True)
            (study / "m4").mkdir()
            (study / "m3/inference_cli.py").write_text("pass\n")
            (study / "m4/endpoint_worker.py").write_text("pass\n")
            prior = base / "prior"
            prior.mkdir()
            role = prior / "role.npz"
            role.write_bytes(b"data-free role fixture")
            digest = hashlib.sha256(role.read_bytes()).hexdigest()
            keys = {"INNER_SELECTION:features", "INNER_SELECTION:income",
                    "INNER_SELECTION:occupation", "INNER_SELECTION:sex",
                    "OSF_DEFENSE_FIT:income", "OSF_DEFENSE_FIT:occupation"}
            keys |= {f"INNER_SELECTION:references_s{s}" for s in (0, 1, 2)}
            manifest = {"role_file_relative_path": {key: "role.npz" for key in keys},
                        "role_file_sha256": {key: digest for key in keys}}
            caps = {stage: {"cpu_reserve_s": 60, "wall_limit_s": 60,
                            "max_rss_gib": 1}
                    for stage in ("endpoint_index", "simultaneous_inference")}
            specs = build_endpoint_inference_specs(
                worktree=repo, study=study, prior=prior, manifest=manifest,
                source_map_sha="manifest-hash", caps=caps,
                python=Path(sys.executable))
            self.assertEqual(set(specs), {
                "M4-ENDPOINTS-FIRST", "M4-ENDPOINTS-SCREEN",
                "M4-ENDPOINTS-FINALIST", "M3-INFERENCE-SCREEN",
                "M3-INFERENCE-FINALIST"})
            self.assertEqual({key for key in specs["M4-ENDPOINTS-FIRST"]["role_inputs"]
                              if key.startswith("INNER_SELECTION:references_")},
                             {"INNER_SELECTION:references_s0"})
            self.assertEqual({key for key in specs["M4-ENDPOINTS-FINALIST"]["role_inputs"]
                              if key.startswith("INNER_SELECTION:references_")},
                             {f"INNER_SELECTION:references_s{s}" for s in (0, 1, 2)})
            for phase in ("SCREEN", "FINALIST"):
                unit = specs["M3-INFERENCE-" + phase]
                self.assertEqual(unit["role_inputs"], {})
                self.assertEqual(unit["role_access"], [])
                self.assertEqual(unit["inputs"], {})
                self.assertEqual(unit["data_access"], "NONE_PRIVATE_DERIVED_ONLY")
                self.assertEqual(unit["deps"], ["M4-ENDPOINTS-" + phase])

    def test_546_fixed_units_and_coalition_parent_order(self):
        with tempfile.TemporaryDirectory() as location:
            base = Path(location)
            repo = base / "repo"
            study = repo / "results" / "pcrl_capacity_matched_feature_selection_v1"
            (study / "m2").mkdir(parents=True)
            (study / "m4").mkdir()
            for script in (study / "m2/fit_worker.py", study / "m4/worker.py"):
                script.write_text("pass\n")
            prior = base / "prior"
            prior.mkdir()
            role = prior / "role.npz"
            role.write_bytes(b"synthetic role fixture, no Adult values")
            digest = hashlib.sha256(role.read_bytes()).hexdigest()
            core_stages = ("fit_bank", "raw_union", "release", "attack_view",
                           "v2_probe")
            keys = set().union(*(set(ROLE_FIELDS[stage](0)) for stage in core_stages))
            keys |= set(ROLE_FIELDS["fit_bank"](1)) | set(ROLE_FIELDS["fit_bank"](2))
            manifest = {"role_file_relative_path": {key: "role.npz" for key in keys},
                        "role_file_sha256": {key: digest for key in keys}}
            actual = Path(__file__).resolve().parents[1] / "m1/CAPACITY_SPEC.json"
            geometry = json.loads(actual.read_text())
            caps = {stage: {"cpu_reserve_s": 60, "wall_limit_s": 60,
                            "max_rss_gib": 1} for stage in core_stages}
            specs = build_core_unit_specs(
                worktree=repo, study=study, prior=prior, manifest=manifest,
                source_map_sha="m1-manifest-hash", geometry=geometry, caps=caps,
                python=Path(sys.executable),
            )
            self.assertEqual(len(specs), 546)
            self.assertEqual({stage: sum(s["stage"] == stage for s in specs.values())
                              for stage in caps},
                             {"fit_bank": 3, "raw_union": 18, "release": 75,
                              "attack_view": 225, "v2_probe": 225})
            self.assertTrue(all(s["seal_sha256"] == "$LOCAL_SEAL_SHA256"
                                and s["allow_child_processes"] is False
                                for s in specs.values()))
            self.assertFalse(any("HEAD_VALIDATION" in str(s["role_inputs"])
                                 or "DEVELOPMENT_ASSESSMENT" in str(s["role_inputs"])
                                 for s in specs.values()))
            self.assertTrue(all(f"M2-FIT-S{seed}" in spec["deps"]
                                for seed in (0, 1, 2)
                                for spec in specs.values()
                                if spec["stage"] == "raw_union" and
                                spec["unit_id"].startswith(f"M4-RAW-S{seed}-")))
            for spec in specs.values():
                if spec["stage"] == "attack_view" and spec["unit_id"].endswith("COALITION"):
                    self.assertEqual(len(spec["deps"]), 5)
                    self.assertIn(spec["unit_id"].removesuffix("COALITION") + "INCOME", spec["deps"])
                    self.assertIn(spec["unit_id"].removesuffix("COALITION") + "OCCUPATION", spec["deps"])


if __name__ == "__main__":
    unittest.main()
