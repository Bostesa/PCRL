"""Synthetic-only gate and saved-view tests for the separate anchor V2 probe."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from attack_eval import AttackView
import run_feature_ref_raw83_u_v2_probe as runner
from safe_data_guard import sha_file


class AnchorV2ProbeTests(unittest.TestCase):
    def test_role_allowlist_and_exact_v2_entrypoint(self) -> None:
        self.assertEqual(runner.ROLE_FIELDS,
                         {"ATTACK_FIT_NEW": ["features", "sex"],
                          "INNER_SELECTION": ["features", "sex"]})
        self.assertEqual(runner.VIEWS, ("r1", "r2", "pair"))
        self.assertTrue(hasattr(runner, "run_control_probes_v2"))

    def test_s0_queue_unit_rejects_seed_one_cli_before_role_read(self) -> None:
        unit_id = "S2-FEATURE-REF-RAW83-U-S0-V2-PROBE"
        parent_id = "S2-FEATURE-REF-RAW83-U-S0"
        state = {"units": {unit_id: {"status": "running", "child_pgid": os.getpgrp(),
                                      "spec": {"unit_id": unit_id, "mode": "real_data",
                                               "family": "SHARED", "team": "SHARED",
                                               "stage": "postrelease_control_v2",
                                               "arm": runner.ARM, "arm_id": runner.ARM,
                                               "seed": 0, "parent_unit_id": parent_id,
                                               "deps": [parent_id],
                                               "parent_anchor_receipt_sha256": "a" * 64,
                                               "runner_sha256": sha_file(Path(runner.__file__)),
                                               "argv": ["python", "runner.py", "--seed", "1"]}},
                           parent_id: {"status": "done"}}}

        class SyntheticQueue:
            def __init__(self, root):
                self.root = root

            def status(self):
                return state

        with patch.object(runner, "Queue", SyntheticQueue):
            with self.assertRaisesRegex(PermissionError, "differs from sealed V2 anchor identity"):
                runner._running_unit(0, {"--seed": "0"}, Path("/synthetic"), "a" * 64)

    def test_saved_view_hash_and_column_contract_are_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            view = AttackView(
                row_id=np.array([10, 11], dtype=np.int64),
                unit=np.array([20, 21], dtype=np.int64),
                x=np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float64),
                components=("probability",), wire_components=("probability", "constant"),
                component_columns=(("probability", 0, 2),),
                wire_constants=(("constant", "public"),)).validate()
            path = root / "view_att.npz"
            np.savez_compressed(path, row_id=view.row_id, unit=view.unit, x=view.x)
            receipt = {"files": {path.name: sha_file(path)},
                       "att_view_content_sha256": view.content_hashes(),
                       "view_schema": {"components": list(view.components),
                                       "wire_components": list(view.wire_components),
                                       "component_columns": [list(item) for item in view.component_columns],
                                       "wire_constants": [list(item) for item in view.wire_constants]}}
            restored = runner._restore_view(root, receipt, "att")
            np.testing.assert_array_equal(restored.x, view.x)
            np.savez_compressed(path, row_id=view.row_id, unit=view.unit, x=view.x + 1)
            receipt["files"][path.name] = sha_file(path)
            with self.assertRaisesRegex(PermissionError, "content differs"):
                runner._restore_view(root, receipt, "att")


if __name__ == "__main__":
    unittest.main()
