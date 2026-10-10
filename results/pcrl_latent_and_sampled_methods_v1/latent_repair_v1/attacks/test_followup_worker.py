"""Synthetic wire and admission regressions for control-arm attacker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ...r01.role_store import sha_array, sha_file
from ..controls.unchanged import ContinuousHead, UnchangedLatentRelease
from ..impl.model import FixedScaler
from .followup_worker import _admit, _f_probabilities, _load_f_wire


def _service() -> UnchangedLatentRelease:
    heads = (ContinuousHead(16, 2), ContinuousHead(16, 6))
    scaler = FixedScaler(np.zeros((2, 16), dtype=np.float32),
                         np.ones((2, 16), dtype=np.float32))
    records = tuple({"schema": "pcrl-portfolio-global-temp-v1",
                     "task": task, "alpha": 0.8 + purpose * 0.2}
                    for purpose, task in enumerate(("income", "occupation")))
    return UnchangedLatentRelease(heads, scaler, "synthetic", "a" * 64, records)


class FollowupWorkerTests(unittest.TestCase):
    def test_f_transmitted_features_replay_exact_probabilities(self) -> None:
        service = _service()
        latent = np.arange(64, dtype=np.float32).reshape(4, 16) / 100
        for purpose in (0, 1):
            release = service.recipient(purpose, latent)
            self.assertTrue(np.array_equal(
                _f_probabilities(service, purpose, release["features"]),
                release["probabilities"]))

    def test_f_release_member_and_probability_handoff(self) -> None:
        service = _service()
        rows = np.array([1, 2, 3, 4], dtype=np.int64)
        units = np.array([1, 2, 3, 4], dtype=np.int64)
        latent = np.arange(64, dtype=np.float32).reshape(4, 16) / 100
        releases = [service.recipient(p, latent) for p in (0, 1)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "release__F__ATTACK_FIT_NEW.npz"
            np.savez_compressed(path, row_id=rows, unit=units,
                                features0=releases[0]["features"],
                                features1=releases[1]["features"],
                                prob0=releases[0]["probabilities"],
                                prob1=releases[1]["probabilities"],
                                decision0=releases[0]["decision"],
                                decision1=releases[1]["decision"])
            receipt = {"roles": {"ATTACK_FIT_NEW": {"F": {
                "private_release_sha256": sha_file(path)}}}}
            store = SimpleNamespace(manifest={"roles": {"ATTACK_FIT_NEW": {
                "rows": 4, "groups": 4,
                "row_unit_pair_sha256": sha_array(np.column_stack((rows, units)))}}})
            wire = _load_f_wire(root, "ATTACK_FIT_NEW", receipt, store, service)
            self.assertTrue(np.array_equal(wire["first"]["features"],
                                           releases[0]["features"]))
            self.assertEqual(wire["second"]["metadata"]["purpose"], 1)
            altered = releases[0]["probabilities"].copy()
            altered[0] = altered[0, ::-1]
            np.savez_compressed(path, row_id=rows, unit=units,
                                features0=releases[0]["features"],
                                features1=releases[1]["features"],
                                prob0=altered, prob1=releases[1]["probabilities"],
                                decision0=releases[0]["decision"],
                                decision1=releases[1]["decision"])
            receipt["roles"]["ATTACK_FIT_NEW"]["F"]["private_release_sha256"] = sha_file(path)
            with self.assertRaisesRegex(ValueError, "do not replay"):
                _load_f_wire(root, "ATTACK_FIT_NEW", receipt, store, service)

    def test_a3_attestation_precedes_role_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "lock.json"
            lock.write_text(json.dumps({"schema": "pcrl-latent-and-sampled-campaign-lock-v1",
                                        "status": "LOCALLY_SEALED_D_STAGE"}))
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            amendment = root / "a3.json"
            amendment.write_text(json.dumps({
                "schema": "pcrl-d-followup-amendment-v1",
                "status": "LOCALLY_SEALED_D_FOLLOWUP",
                "effective_seal_sha256": sha_file(lock),
                "control_attack_worker_sha256": "0" * 64,
                "role_manifest_sha256": sha_file(manifest),
                "authorized_unit_ids": ["D_ATTACK_F_S0"]}))
            state = {"units": {"D_FOLLOWUP_CONTROLS_S0": {"status": "done"}},
                     "amendments": [{"stages": ["d_followup"],
                                     "sha256": sha_file(amendment)}]}
            (root / "state.json").write_text(json.dumps(state))
            args = SimpleNamespace(stage="d_followup", seed=0, arm_id="F",
                                   source_unit_id="D_FOLLOWUP_CONTROLS_S0",
                                   unit_id="D_ATTACK_F_S0", seal=str(lock),
                                   seal_sha256=sha_file(lock),
                                   attack_amendment=str(amendment),
                                   attack_amendment_sha256=sha_file(amendment),
                                   role_manifest=str(manifest),
                                   role_manifest_sha256=sha_file(manifest))
            with patch.dict(os.environ, {"PCRL_QUEUE_ROOT": str(root)}), \
                 patch("results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.attacks.followup_worker.assert_active_from_env",
                       return_value={"seal_sha256": sha_file(lock)}):
                with self.assertRaisesRegex(PermissionError, "A3 does not authorize"):
                    _admit(args)


if __name__ == "__main__":
    unittest.main()
