"""Synthetic-only A score-bank preprocessing permission and queue checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import admit_a_score_bank as adapter
from safe_data_guard import sha_file


class AScoreBankStageTests(unittest.TestCase):
    def test_unpinned_source_manifest_is_refused_before_role_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "pcrl_method_portfolio_v1"
            campaign.mkdir()
            hcal = root / "pcrl_heldout_calibration_v1"
            lra = root / "pcrl_adult_learned_decoder_release_v1"
            hcal.mkdir()
            lra.mkdir()
            model = hcal / "MODEL_MANIFEST.json"
            source = hcal / "SOURCE_ADMISSION.json"
            lra_model = lra / "MODEL_MANIFEST.json"
            model.write_text('{}\n')
            source.write_text('{}\n')
            lra_model.write_text('{}\n')
            manifest = campaign / "DATA_ACCESS_MANIFEST.json"
            manifest.write_text('{}\n')
            lock = {"status": "LOCAL_GIT_SEALED_EFFECTIVE", "adult_fit_permission": True,
                    "families": {"A": {"fit_admission": "DENIED",
                                       "preprocessing_admission": "ALLOWED"}},
                    "admitted_a_score_bank_unit_ids": ["S2-A-SCORE-BANK-TSTAR-S0"],
                    "a_score_bank_runner_sha256": sha_file(Path(adapter.__file__)),
                    "data_admission": {"manifest_sha256": sha_file(manifest)},
                    "source_model_admission": {
                        "hcal_model_manifest_sha256": sha_file(model),
                        "hcal_source_admission_sha256": sha_file(source),
                        "lra_model_manifest_sha256": "0" * 64}}
            lock_path = campaign / "CAMPAIGN_LOCK.json"
            lock_path.write_text(json.dumps(lock))
            store = SimpleNamespace(manifest={"status": "ROLE_FILES_ADMITTED_S4_VERIFIED"},
                                    manifest_path=manifest,
                                    read=lambda *_: (_ for _ in ()).throw(
                                        AssertionError("role member values must not be read")))
            with patch.object(adapter, "CAMPAIGN", campaign), \
                    patch.object(adapter, "SHARED", root / "shared"), \
                    patch.object(adapter, "_running_preprocess_unit", return_value={}), \
                    patch.object(adapter.Queue, "_local_seal_ok", return_value=True), \
                    patch.object(adapter, "SafeRoleStore", return_value=store):
                with self.assertRaisesRegex(RuntimeError, "fine/policy source manifests"):
                    adapter.admit_one_seed(root, 0, "tstar", sha_file(lock_path), root)

    def test_decoy_script_under_correct_unit_is_refused(self) -> None:
        unit_id = "S2-A-SCORE-BANK-TSTAR-S0"
        queue_root = Path("/synthetic/queue")
        cache_root = Path("/synthetic/cache")
        output = adapter.SHARED / ".private" / "tstar_score_bank_s0_v1" / "BANK_ADMISSION_RECEIPT.json"
        state = {"units": {unit_id: {
            "status": "running", "child_pgid": os.getpgrp(),
            "spec": {"unit_id": unit_id, "mode": "real_data", "family": "SHARED",
                     "team": "SHARED", "stage": "a_score_bank_preprocess",
                     "data_access": adapter.DATA_ACCESS, "seed": 0,
                     "release_key": "tstar", "runner_sha256": sha_file(Path(adapter.__file__)),
                     "outputs": [str(output)],
                     "argv": ["python", "/synthetic/decoy.py", "--seed", "0",
                              "--release", "tstar", "--cache-root", str(cache_root),
                              "--campaign-lock-sha256", "a" * 64,
                              "--queue-root", str(queue_root)]}}}}

        class SyntheticQueue:
            def __init__(self, root):
                self.root = root

            def status(self):
                return state

        with patch.object(adapter, "Queue", SyntheticQueue):
            with self.assertRaisesRegex(PermissionError, "identity differs"):
                adapter._running_preprocess_unit(cache_root, 0, "tstar", "a" * 64, queue_root)

    def test_b_only_lock_denies_preprocessing_before_role_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            lock_path = campaign / "CAMPAIGN_LOCK.json"
            lock_path.write_text(json.dumps({"status": "LOCAL_GIT_SEALED_EFFECTIVE",
                                             "adult_fit_permission": True,
                                             "families": {"A": {"fit_admission": "DENIED",
                                                                "preprocessing_admission": "DENIED"}}}))
            with patch.object(adapter, "CAMPAIGN", campaign), \
                    patch.object(adapter, "_running_preprocess_unit", return_value={}), \
                    patch.object(adapter, "SafeRoleStore",
                                 side_effect=AssertionError("role loader must not be reached")):
                with self.assertRaisesRegex(RuntimeError, "preprocessing-only source unit"):
                    adapter.admit_one_seed(campaign, 0, "tstar", sha_file(lock_path), campaign)

    def test_preprocessing_permission_alone_does_not_bypass_local_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            lock_path = campaign / "CAMPAIGN_LOCK.json"
            lock_path.write_text(json.dumps({
                "status": "LOCAL_GIT_SEALED_EFFECTIVE", "adult_fit_permission": True,
                "families": {"A": {"fit_admission": "DENIED",
                                   "preprocessing_admission": "ALLOWED"}},
                "admitted_a_score_bank_unit_ids": ["S2-A-SCORE-BANK-TSTAR-S0"],
                "a_score_bank_runner_sha256": sha_file(Path(adapter.__file__))}))
            with patch.object(adapter, "CAMPAIGN", campaign), \
                    patch.object(adapter, "_running_preprocess_unit", return_value={}), \
                    patch.object(adapter.Queue, "_local_seal_ok", return_value=False), \
                    patch.object(adapter, "SafeRoleStore",
                                 side_effect=AssertionError("role loader must not be reached")):
                with self.assertRaisesRegex(RuntimeError, "committed local science seal absent"):
                    adapter.admit_one_seed(campaign, 0, "tstar", sha_file(lock_path), campaign)


if __name__ == "__main__":
    unittest.main()
