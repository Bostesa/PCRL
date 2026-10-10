"""Admission must fail before any role or SEX access without an R03 slot."""

from argparse import Namespace
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls import run_admitted_controls as worker


class AdmissionTest(TestCase):
    def test_actual_corrected_a_envelope_and_unknown_key(self):
        campaign = Path(worker.__file__).resolve().parents[2]
        amendment = json.loads((campaign /
            "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_PATH_CORRECTION.json").read_text())
        worker.validate_a_amendment_envelope(amendment)
        with self.assertRaisesRegex(PermissionError, "top-level keys"):
            worker.validate_a_amendment_envelope({**amendment, "unreviewed_key": True})

    def test_selected_output_requires_resolved_private_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            with self.assertRaisesRegex(PermissionError, "private"):
                worker._checked_private_output(parent / "selected")
            self.assertFalse((parent / "selected").exists())

    def test_private_output_parent_guard_runs_without_creating_or_reading_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / ".private"
            private.mkdir(mode=0o700)
            target = private / "selected"
            self.assertEqual(worker._checked_private_output(target),
                             target.resolve(strict=False))
            self.assertFalse(target.exists())
            private.chmod(0o755)
            with self.assertRaisesRegex(PermissionError, "owner-only"):
                worker._checked_private_output(target)
            private.chmod(0o700)
            link = root / "linked"
            link.symlink_to(private, target_is_directory=True)
            with self.assertRaisesRegex(PermissionError, "owner-only"):
                worker._checked_private_output(link / "selected")

    def test_selected_provenance_matches_package_contract(self):
        self.assertEqual(worker._selected_provenance(
            "A_CENTRAL_PRODUCT", {"diagnostic_best": [0.5, 1, 125]}),
            ("uniform_logit", 125))
        self.assertEqual(worker._selected_provenance("A_LOCAL", {}),
                         ("composite", 0))
        self.assertEqual(worker._selected_provenance("A_SEQ21", {}),
                         ("composite", 0))
        metrics = {"primary": .5}
        self.assertEqual(worker._selected_provenance(
            "A_TASK_DET", {"selection": metrics, "starts": [
                {"start": "identity_biased", "terminal_metrics": metrics,
                 "chosen_checkpoint_update": 275}]}),
            ("identity_biased", 275))

    def test_direct_run_cannot_read_role_data(self):
        args = Namespace(unit_id="A_TASK_DET_S0", arm_id="A_TASK_DET", seed=0,
                         original_lock="absent", a_amendment="absent",
                         a_amendment_sha256="0" * 64, private_output="absent",
                         repair_manifest="absent", source_audit="absent",
                         portfolio_package="absent", campaign_root="absent")
        with patch.object(worker, "assert_active_from_env",
                          side_effect=RuntimeError("no queue slot")), \
             patch.object(worker, "RoleOnlyStore") as roles, \
             patch.object(worker, "TStarSourceStore") as source:
            with self.assertRaisesRegex(RuntimeError, "no queue slot"):
                worker.run(args)
        roles.assert_not_called()
        source.assert_not_called()

    def test_direct_admission_stops_at_queue_gate(self):
        with patch.object(worker, "assert_active_from_env",
                          side_effect=RuntimeError("no queue slot")):
            with self.assertRaisesRegex(RuntimeError, "no queue slot"):
                worker.admit_control_unit(unit_id="A_TASK_DET_S0",
                                          lock_path="absent", amendment_path="absent",
                                          amendment_sha256="0" * 64,
                                          output_dir="absent")
