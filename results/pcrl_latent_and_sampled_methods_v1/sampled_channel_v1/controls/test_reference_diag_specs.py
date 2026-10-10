"""Data-free spec-rendering checks for A source releases."""

import json
import stat
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import reference_diag_specs as target


CAMPAIGN = Path(__file__).resolve().parents[2]


class SourceReleaseSpecTest(unittest.TestCase):
    def _synthetic_context(self, root):
        campaign = root / "worktree/results/campaign"
        campaign.mkdir(parents=True)
        parent = campaign / ".private"
        parent.mkdir(mode=0o700)
        source = campaign / "r01"
        source.mkdir()
        (source / "SOURCE_AUDIT.json").write_text("{}")
        (source / "A_HISTORICAL_SCORE_SOURCE_AUDIT.json").write_text("{}")
        for extension in ("json", "npz"):
            (source / f"A_FIXED_MAP_T_STAR_S0.{extension}").write_bytes(b"public map")
        lock = campaign / "CAMPAIGN_LOCK.json"
        lock.write_text("{}")
        portfolio = root / "portfolio"
        portfolio.mkdir()
        portfolio_manifest = portfolio / "DATA_ACCESS_MANIFEST.json"
        portfolio_manifest.write_text("{}")
        role_package = root / "repair"
        repair = role_package / "m1/DATA_ACCESS_MANIFEST.json"
        repair.parent.mkdir(parents=True)
        role_relative = ".private/role_export_v1/ATTACK_FIT_NEW/references_s0.npz"
        reference = role_package / role_relative
        reference.parent.mkdir(parents=True)
        reference.write_bytes(b"private role archive not opened")
        bank = portfolio / "shared/.private/tstar_score_bank_s0_v1"
        bank.mkdir(parents=True)
        (bank / "BANK_ADMISSION_RECEIPT.json").write_text("{}")
        for filename in ("tokens_ATTACK_FIT_NEW.npz", "bank_r1.npz", "bank_r2.npz"):
            (bank / filename).write_bytes(filename.encode())
        role_manifest = {
            "portfolio_manifest_sha256": target._sha(portfolio_manifest),
            "role_file_relative_path": {"ATTACK_FIT_NEW:references_s0": role_relative},
            "role_file_sha256": {"ATTACK_FIT_NEW:references_s0": target._sha(reference)},
        }
        repair.write_text(json.dumps(role_manifest))
        source_audit = {"tstar_banks": {"0": {
            "receipt_sha256": target._sha(bank / "BANK_ADMISSION_RECEIPT.json"),
            "file_sha256": {name: target._sha(bank / name) for name in
                            ("tokens_ATTACK_FIT_NEW.npz", "bank_r1.npz", "bank_r2.npz")},
        }}}
        (source / "SOURCE_AUDIT.json").write_text(json.dumps(source_audit))
        amendment = json.loads((CAMPAIGN /
            "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_RETRY_001.json").read_text())
        amendment["implementation_source_sha256"] = {}
        amendment["source_sha256"] = {
            f"r01/A_FIXED_MAP_T_STAR_S0.{extension}":
                target._sha(source / f"A_FIXED_MAP_T_STAR_S0.{extension}")
            for extension in ("json", "npz")}
        amendment["role_bindings"]["repair_manifest_sha256"] = target._sha(repair)
        amended = campaign / "AMENDMENT.json"
        amended.write_text(json.dumps(amendment))
        queue = root / "state.json"
        queue.write_text(json.dumps({
            "seal": {"sha256": target._sha(lock)},
            "amendments": [{"stages": ["a"], "path": str(amended),
                            "sha256": target._sha(amended)}],
        }))
        cache = root / "cache"
        cache.mkdir()
        kwargs = dict(amendment_path=amended, amendment_sha256=target._sha(amended),
                      seal_path=lock, seal_sha256=target._sha(lock),
                      repair_manifest=repair, portfolio_package=portfolio,
                      historical_cache_root=cache, queue_state=queue,
                      private_parent=parent)
        return kwargs, reference, bank

    def test_sealed_source_release_inventory_is_exact(self):
        amendment = json.loads((CAMPAIGN /
            "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_RETRY_001.json").read_text())
        references, diagnostics = target.planned_source_release_ids(amendment)
        self.assertEqual(len(references), 28)
        self.assertEqual(len(diagnostics), 8)
        self.assertEqual(references[0], "A-REFERENCE-RELEASE-D0_LOCAL-S0-ATT")
        self.assertEqual(diagnostics[-1], "A-DIAG-RELEASE-A_IDENTITY-S1-INNER")

    def test_source_release_inventory_refuses_missing_role(self):
        amendment = json.loads((CAMPAIGN /
            "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_RETRY_001.json").read_text())
        del amendment["unit_plan"]["A-DIAG-RELEASE-A_IDENTITY-S1-INNER"]
        with self.assertRaises(ValueError):
            target.planned_source_release_ids(amendment)

    def test_role_only_reference_and_canonical_diagnostic_specs(self):
        with tempfile.TemporaryDirectory() as temp:
            kwargs, reference, bank = self._synthetic_context(Path(temp).resolve())
            with patch.object(target, "compile_graph", return_value={"unit_count": 220}):
                u0 = target.source_release_spec(
                    unit_id="A-REFERENCE-RELEASE-U0-S0-ATT", **kwargs)
                diagnostic = target.source_release_spec(
                    unit_id="A-DIAG-RELEASE-A_IDENTITY-S0-ATT", **kwargs)
            self.assertEqual(u0["reference_id"], "U0")
            self.assertEqual(u0["cpu_reserve_s"], 30)
            self.assertEqual(u0["deps"], [])
            self.assertEqual(diagnostic["arm_id"], "A_IDENTITY")
            self.assertEqual(diagnostic["cpu_reserve_s"], 10)
            self.assertEqual(diagnostic["outputs"], [str(kwargs["private_parent"] /
                "A-DIAG-RELEASE-A_IDENTITY-S0-ATT/REFERENCE_RELEASE_RECEIPT.json")])
            self.assertIn("--private-output", diagnostic["argv"])
            self.assertIn("--output-dir", u0["argv"])
            pinned = {row["path"] for row in diagnostic["sources"]}
            self.assertIn(str(reference), pinned)
            self.assertIn(str(bank / "tokens_ATTACK_FIT_NEW.npz"), pinned)
            self.assertIn(str(bank / "bank_r1.npz"), pinned)

    def test_reference_cap_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            kwargs, _, _ = self._synthetic_context(Path(temp).resolve())
            amendment = json.loads(kwargs["amendment_path"].read_text())
            amendment["unit_plan"]["A-REFERENCE-RELEASE-U0-S0-ATT"]["cpu_seconds"] = 31
            kwargs["amendment_path"].write_text(json.dumps(amendment))
            kwargs["amendment_sha256"] = target._sha(kwargs["amendment_path"])
            state = json.loads(kwargs["queue_state"].read_text())
            state["amendments"][-1]["sha256"] = kwargs["amendment_sha256"]
            kwargs["queue_state"].write_text(json.dumps(state))
            with patch.object(target, "compile_graph", return_value={"unit_count": 220}):
                with self.assertRaises(ValueError):
                    target.source_release_spec(
                        unit_id="A-REFERENCE-RELEASE-U0-S0-ATT", **kwargs)

    def test_cli_writes_owner_only_spec_once(self):
        with tempfile.TemporaryDirectory() as temp:
            kwargs, _, _ = self._synthetic_context(Path(temp).resolve())
            manifest = kwargs["amendment_path"].parent / "DOWNSTREAM.json"
            registration = kwargs["amendment_path"].parent / "REGISTRATION.json"
            manifest.write_text(json.dumps({"new_source_sha256": {},
                                            "attack_plan": {"inherited_shared_source_sha256": {}}}))
            registration.write_text("{}")
            output = kwargs["private_parent"] / "rendered.json"
            argv = ["--unit-id", "A-REFERENCE-RELEASE-U0-S0-ATT",
                    "--amendment", str(kwargs["amendment_path"]),
                    "--amendment-sha256", kwargs["amendment_sha256"],
                    "--seal", str(kwargs["seal_path"]),
                    "--seal-sha256", kwargs["seal_sha256"],
                    "--repair-manifest", str(kwargs["repair_manifest"]),
                    "--portfolio-package", str(kwargs["portfolio_package"]),
                    "--historical-cache-root", str(kwargs["historical_cache_root"]),
                    "--queue-state", str(kwargs["queue_state"]),
                    "--private-parent", str(kwargs["private_parent"]),
                    "--downstream-manifest", str(manifest),
                    "--downstream-manifest-sha256", target._sha(manifest),
                    "--downstream-registration", str(registration),
                    "--downstream-registration-sha256", target._sha(registration),
                    "--spec-output", str(output)]
            with (patch.object(target, "compile_graph", return_value={"unit_count": 220}),
                  patch.object(target, "verify_downstream_sidecar"),
                  patch.object(target, "verify_downstream_registration"),
                  patch.object(target, "validate_committed_registration")):
                target.main(argv)
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    target.main(argv)

    def test_guarded_renderer_pins_sidecar_and_registration_without_worker_change(self):
        with tempfile.TemporaryDirectory() as temp:
            kwargs, _, _ = self._synthetic_context(Path(temp).resolve())
            campaign = kwargs["amendment_path"].parent
            manifest = campaign / "DOWNSTREAM.json"
            registration = campaign / "REGISTRATION.json"
            manifest.write_text(json.dumps({"new_source_sha256": {},
                                            "attack_plan": {"inherited_shared_source_sha256": {}}}))
            registration.write_text("{}")
            with (patch.object(target, "compile_graph", return_value={"unit_count": 220}),
                  patch.object(target, "verify_downstream_sidecar"),
                  patch.object(target, "verify_downstream_registration"),
                  patch.object(target, "validate_committed_registration")):
                spec = target.guarded_source_release_spec(
                    unit_id="A-DIAG-RELEASE-A_IDENTITY-S0-ATT",
                    downstream_manifest=manifest,
                    downstream_manifest_sha256=target._sha(manifest),
                    downstream_registration=registration,
                    downstream_registration_sha256=target._sha(registration),
                    **kwargs)
            pinned = {row["path"]: row["sha256"] for row in spec["sources"]}
            self.assertEqual(pinned[str(manifest)], target._sha(manifest))
            self.assertEqual(pinned[str(registration)], target._sha(registration))
            self.assertEqual(spec["downstream_manifest_sha256"], target._sha(manifest))
            self.assertIn("controls.run_diagnostic_release", spec["argv"][2])

    def test_uncommitted_downstream_registration_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            kwargs, _, _ = self._synthetic_context(Path(temp).resolve())
            campaign = kwargs["amendment_path"].parent
            manifest = campaign / "DOWNSTREAM.json"
            registration = campaign / "REGISTRATION.json"
            manifest.write_text(json.dumps({"new_source_sha256": {},
                                            "attack_plan": {"inherited_shared_source_sha256": {}}}))
            registration.write_text("{}")
            with (patch.object(target, "compile_graph", return_value={"unit_count": 220}),
                  patch.object(target, "verify_downstream_sidecar"),
                  patch.object(target, "verify_downstream_registration"),
                  patch.object(target, "validate_committed_registration",
                               side_effect=PermissionError("registration lacks Git custody"))):
                with self.assertRaisesRegex(PermissionError, "Git custody"):
                    target.guarded_source_release_spec(
                        unit_id="A-REFERENCE-RELEASE-U0-S0-ATT",
                        downstream_manifest=manifest,
                        downstream_manifest_sha256=target._sha(manifest),
                        downstream_registration=registration,
                        downstream_registration_sha256=target._sha(registration),
                        **kwargs)


if __name__ == "__main__":
    unittest.main()
