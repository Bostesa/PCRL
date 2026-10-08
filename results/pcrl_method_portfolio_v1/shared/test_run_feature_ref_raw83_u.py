"""Synthetic import and local-seal gate checks for the feature anchor."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_feature_ref_raw83_u as anchor
from compute_queue import Queue
from safe_data_guard import sha_file


class FeatureAnchorGateTests(unittest.TestCase):
    def test_runner_uses_real_queue_class(self) -> None:
        self.assertIs(anchor.Queue, Queue)

    def test_local_seal_denial_occurs_before_source_or_role_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            manifest_path = campaign / "DATA_ACCESS_MANIFEST.json"
            manifest_path.write_text('{"status":"ROLE_FILES_ADMITTED_S4_VERIFIED"}\n')
            lock = {
                "status": "LOCAL_GIT_SEALED_EFFECTIVE",
                "adult_fit_permission": True,
                "admitted_shared_unit_ids": ["S2-FEATURE-REF-RAW83-U-S0"],
                "data_admission": {"manifest_sha256": sha_file(manifest_path)},
                "attack_protocol": {
                    "source_module_sha256": sha_file(Path(anchor.__file__).with_name("attack_eval.py"))
                },
                "shared_attack_controls_sha256": sha_file(
                    Path(anchor.__file__).with_name("attack_controls.py")),
            }
            lock_path = campaign / "CAMPAIGN_LOCK.json"
            lock_path.write_text(json.dumps(lock, sort_keys=True) + "\n")
            (campaign / "LOCAL_SEAL_RECEIPT.json").write_text("{}\n")

            store = SimpleNamespace(
                manifest={"status": "ROLE_FILES_ADMITTED_S4_VERIFIED"},
                manifest_path=manifest_path)

            with patch.object(anchor, "CAMPAIGN", campaign), \
                    patch.object(Queue, "_local_seal_ok", return_value=False) as gate:
                with self.assertRaisesRegex(PermissionError, "committed local science seal absent"):
                    anchor._seal_and_source(0, store, sha_file(lock_path), campaign)
                gate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
