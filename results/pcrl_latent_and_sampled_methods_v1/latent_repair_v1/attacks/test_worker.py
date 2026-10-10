"""Synthetic admission boundary checks; no role file or attacker fit."""

import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ...r01.role_store import sha_array, sha_file
from ..impl.model import FixedScaler, HardCodeModel, HardCodeRelease
from .worker import _load_wire, _source_receipt, run


class WorkerAdmissionTest(unittest.TestCase):
    def test_direct_launch_refused_before_role_or_source_access(self):
        args = argparse.Namespace(stage="d_first", seed=0, arm_id="T", unit_id="D-ATT-TEST")
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "queue slot"):
            run(args)

    def test_first_stage_cannot_run_other_seed_or_arm(self):
        for seed, arm in ((1, "T"), (0, "L")):
            args = argparse.Namespace(stage="d_first", seed=seed, arm_id=arm)
            with self.assertRaises(PermissionError):
                run(args)

    def test_private_wire_replays_exact_code_only_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".private"
            root.mkdir(mode=0o700)
            service = HardCodeRelease(
                HardCodeModel(16, (2, 6)),
                FixedScaler(np.zeros((2, 16), dtype=np.float32),
                            np.ones((2, 16), dtype=np.float32)),
                "synthetic-fixture", "a" * 64)
            rows = np.array([11, 12], dtype=np.int64)
            units = rows.copy()
            latent = np.zeros((2, 16), dtype=np.float32)
            r0, r1 = service.recipient(0, latent), service.recipient(1, latent)
            arrays = {"row_id": rows, "unit": units,
                      "code0": r0["code"], "code1": r1["code"],
                      "prob0": r0["probabilities"], "prob1": r1["probabilities"],
                      "decision0": r0["decision"], "decision1": r1["decision"]}
            path = root / "release__T__ATTACK_FIT_NEW.npz"
            np.savez_compressed(path, **arrays)
            receipt = {"roles": {"ATTACK_FIT_NEW": {"T": {
                "private_release_sha256": sha_file(path)}}}}
            store = SimpleNamespace(manifest={"roles": {"ATTACK_FIT_NEW": {
                "rows": 2, "groups": 2,
                "row_unit_pair_sha256": sha_array(np.column_stack((rows, units)))}}})
            wire = _load_wire(root, "ATTACK_FIT_NEW", "T", receipt, store, service)
            self.assertEqual(wire["first"]["metadata"]["bit_order"], "c0_least_significant")
            arrays["prob0"] = arrays["prob0"].copy()
            arrays["prob0"][0] = arrays["prob0"][0, ::-1]
            np.savez_compressed(path, **arrays)
            receipt["roles"]["ATTACK_FIT_NEW"]["T"]["private_release_sha256"] = sha_file(path)
            with self.assertRaisesRegex(ValueError, "code-only"):
                _load_wire(root, "ATTACK_FIT_NEW", "T", receipt, store, service)

    def test_first_runner_receipt_sidecar_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ".private"
            root.mkdir(mode=0o700)
            endpoint = root / "endpoint__INNER_SELECTION.npz"
            endpoint.write_bytes(b"synthetic sidecar fixture")
            role_arms = {arm: {"rows": 2, "private_release_sha256": "b" * 64}
                         for arm in ("T", "D-K")}
            receipt = {
                "schema": "latent-repair-first-paired-seed-receipt-v1",
                "status": "T_DK_RELEASES_READY", "seed": 0,
                "unit_id": "D_FIRST_T_DK_S0", "seal_sha256": "a" * 64,
                "role_manifest_sha256": "c" * 64,
                "packages": {"T": {}, "D-K": {}},
                "roles": {"ATTACK_FIT_NEW": role_arms,
                          "INNER_SELECTION": {**role_arms,
                              "endpoint_labels_references_sha256": sha_file(endpoint)}},
            }
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(receipt))
            args = argparse.Namespace(source_receipt=str(path),
                                      source_receipt_sha256=sha_file(path),
                                      seed=0, source_unit_id="D_FIRST_T_DK_S0",
                                      seal_sha256="a" * 64,
                                      role_manifest_sha256="c" * 64,
                                      arm_id="T")
            store = SimpleNamespace(manifest={"roles": {
                "ATTACK_FIT_NEW": {"rows": 2}, "INNER_SELECTION": {"rows": 2}}})
            self.assertEqual(_source_receipt(args, store, root), receipt)
            endpoint.write_bytes(b"tampered")
            with self.assertRaisesRegex(PermissionError, "sidecar hash"):
                _source_receipt(args, store, root)


if __name__ == "__main__":
    unittest.main()
