"""Synthetic identity checks for the versioned anchor V2 retry runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_feature_ref_raw83_u_v2_probe_r2 as runner
from safe_data_guard import sha_file


class AnchorV2R2Tests(unittest.TestCase):
    def test_parent_reader_map_and_versioned_output(self) -> None:
        parent_map = {"r1": "1" * 64, "r2": "2" * 64, "pair": "3" * 64}
        parent = {"reader_receipt_sha256": parent_map}
        runner._require_reader_receipt_binding(
            {"reader_receipt_sha256": dict(parent_map)}, parent)
        with self.assertRaisesRegex(PermissionError, "reader"):
            runner._require_reader_receipt_binding({}, parent)
        with self.assertRaisesRegex(PermissionError, "reader"):
            runner._require_reader_receipt_binding(
                {"reader_receipt_sha256": {**parent_map, "pair": "4" * 64}}, parent)
        self.assertEqual(
            runner._r2_output(0).name, "V2_ANCHOR_PROBE_RECEIPT.json")
        self.assertEqual(
            runner._r2_output(0).parent.name,
            "feature_ref_raw83_u_s0_v2_probe_r2")

    def _state(self, root: Path, *, seed: int, cli_unit_id: str | None = None):
        unit_id = f"S2-FEATURE-REF-RAW83-U-S{seed}-V2-PROBE-R2"
        parent_id = f"S2-FEATURE-REF-RAW83-U-S{seed}"
        parent_receipt = root / "units" / parent_id / "RECEIPT.json"
        parent_receipt.parent.mkdir(parents=True)
        parent_receipt.write_text(json.dumps({"unit_id": parent_id}) + "\n")
        parent_anchor_sha = "a" * 64
        actual_cli_unit = cli_unit_id or unit_id
        argv = ["python", str(Path(runner.__file__).resolve()),
                "--seed", str(seed), "--unit-id", actual_cli_unit,
                "--parent-anchor-receipt-sha256", parent_anchor_sha]
        state = {"units": {
            unit_id: {"status": "running", "child_pgid": os.getpgrp(), "spec": {
                "unit_id": unit_id, "mode": "real_data", "family": "SHARED",
                "team": "SHARED", "stage": "postrelease_control_v2",
                "data_access": "ATTACK_INNER_FEATURES_AND_SEX_V2_CONTROL",
                "arm": runner.ARM, "arm_id": runner.ARM, "seed": seed,
                "parent_unit_id": parent_id, "deps": [parent_id],
                "parent_anchor_receipt_sha256": parent_anchor_sha,
                "parent_receipt_sha256": sha_file(parent_receipt),
                "reader_receipt_sha256": {"r1": "1" * 64,
                                          "r2": "2" * 64,
                                          "pair": "3" * 64},
                "runner_sha256": sha_file(Path(runner.__file__)),
                "argv": argv}},
            parent_id: {"status": "done"}}}
        return state, unit_id, parent_anchor_sha

    def test_new_r2_unit_and_explicit_cli_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, unit_id, parent_anchor_sha = self._state(root, seed=0)

            class SyntheticQueue:
                def __init__(self, queue_root):
                    self.root = queue_root

                def status(self):
                    return state

            with patch.object(runner, "Queue", SyntheticQueue):
                result, spec = runner._running_unit(
                    0, unit_id,
                    {"--seed": "0", "--unit-id": unit_id,
                     "--parent-anchor-receipt-sha256": parent_anchor_sha},
                    root, parent_anchor_sha)
            self.assertIs(result, state)
            self.assertEqual(spec["unit_id"], unit_id)

    def test_wrong_cli_unit_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, unit_id, parent_anchor_sha = self._state(
                root, seed=1, cli_unit_id="S2-FEATURE-REF-RAW83-U-S1-V2-PROBE")

            class SyntheticQueue:
                def __init__(self, queue_root):
                    self.root = queue_root

                def status(self):
                    return state

            with patch.object(runner, "Queue", SyntheticQueue):
                with self.assertRaisesRegex(PermissionError, "identity"):
                    runner._running_unit(
                        1, unit_id,
                        {"--seed": "1", "--unit-id": unit_id,
                         "--parent-anchor-receipt-sha256": parent_anchor_sha},
                        root, parent_anchor_sha)


if __name__ == "__main__":
    unittest.main()
