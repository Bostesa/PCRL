"""Synthetic checks of guarded source binding and FIT receipt accounting."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import FixedBank
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import GateWeights, JointPolicy
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import IdentityPolicy
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_materialize import materialize_arrays
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_worker import (
    _admitted_source, _checkpoint_archive, _sha, _utility,
)
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_source_deterministic import (
    GateWeights as RetainedDeterministicWeights,
)


class WorkerDataFreeTests(unittest.TestCase):
    def test_identity_law_has_zero_six_fit_excesses(self):
        b1 = FixedBank(np.array([[.8, .2], [.7, .3]], dtype=np.float64),
                       np.array([0, 0], dtype=np.int64), (0, 1))
        b2 = FixedBank(np.array([[.6, .1, .1, .1, .05, .05]], dtype=np.float64),
                       np.array([0], dtype=np.int64), (0,))
        policy = IdentityPolicy(JointPolicy(
            b1, b2, ((0, 1),), ((0,),), GateWeights.zeros(input_width=12)))
        law, _ = materialize_arrays(
            policy, row_id=np.array([0, 1], dtype=np.int64),
            unit=np.array([0, 1], dtype=np.int64),
            score1=np.array([[.8, .2], [.7, .3]], dtype=np.float64),
            score2=np.tile(b2.probability[0], (2, 1)),
            cell1=np.zeros(2, dtype=np.int64),
            cell2=np.zeros(2, dtype=np.int64), source_seed=0,
            role="OSF_DEFENSE_FIT", realization=0, private_root=bytes(range(32)))
        result = _utility(law, policy, (np.array([0, 1]), np.array([0, 2])))
        for key, value in result.items():
            if not key.endswith("_cap"):
                self.assertAlmostEqual(value, 0.0, places=14)

    def test_incomplete_checkpoint_arrays_survive_private_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            weights = GateWeights.zeros(input_width=12)
            public, receipt = _checkpoint_archive(root, {
                "status": "INCOMPLETE_SEARCH",
                "_checkpoint_models": {"stage1": [(0, 25, weights)]},
            })
            self.assertEqual(public["status"], "INCOMPLETE_SEARCH")
            self.assertEqual(receipt["checkpoint_array_count"], 4)
            self.assertEqual(_sha(root / "CHECKPOINTS.npz"),
                             receipt["checkpoint_archive_sha256"])
            self.assertEqual((root / "CHECKPOINTS.npz").stat().st_mode & 0o777, 0o600)

    def test_deterministic_retained_gate_arrays_are_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            gate = RetainedDeterministicWeights(
                np.zeros((12, 16)), np.zeros(16),
                np.zeros((16, 16)), np.zeros(16))
            public, receipt = _checkpoint_archive(root, {
                "_checkpoint_models": [(0, 25, "replay", gate)]})
            self.assertEqual(receipt["checkpoint_array_count"], 4)
            self.assertEqual(public["_checkpoint_models"][0][2], "replay")

    def test_sequential_private_root_must_match_public_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_roots = {str(i): bytes([i + 1]) * 32 for i in range(3)}
            sequential_root = bytes([4]) * 32
            roots = {"schema": "scv1-private-stream-roots-v1",
                     "release_realization_roots_hex":
                     {k: v.hex() for k, v in release_roots.items()},
                     "seq_fit_root_hex": sequential_root.hex()}
            roots_path = root / "roots.json"
            roots_path.write_text(json.dumps(roots))
            h = lambda value: hashlib.sha256(value).hexdigest()
            public = {"schema": "scv1-public-stream-root-hashes-v1",
                      "private_file_sha256": _sha(roots_path),
                      "release_realization_root_sha256":
                      {k: h(v) for k, v in release_roots.items()},
                      "seq_fit_root_sha256": h(sequential_root)}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(public))
            lock = {"schema": "pcrl-bank-anchored-campaign-lock-v1",
                    "selected_bank": "W_TSTAR",
                    "sampling_root_hashes": {
                        "private_roots_source_sha256": _sha(roots_path),
                        "joint_SCv1": [h(release_roots[str(i)]) for i in range(3)]}}
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(lock))
            amendment_path = root / "amendment.json"
            amendment_path.write_text("{}")
            identity_path = root / "identity.json"
            identity_path.write_text(json.dumps({
                "schema": "BAR1-r2-full-fit-identity-v1", "status": "PASS",
                "source_seed": 0, "source_lock_sha256": _sha(lock_path)}))
            paths = {"campaign_lock": lock_path,
                     "protection_amendment": amendment_path,
                     "stream_roots": roots_path, "stream_manifest": manifest_path,
                     "identity_receipt": identity_path}
            for key in ("role_manifest", "portfolio_manifest", "source_audit",
                        "worker_source"):
                path = root / f"{key}.json"
                path.write_text("{}")
                paths[key] = path
            registered = {path.resolve(): _sha(path) for path in paths.values()}
            cfg = {"source_lock_sha256": _sha(lock_path),
                   "protection_amendment_sha256": _sha(amendment_path),
                   "source_seed": 0,
                   "pins": {key: {"path": str(path), "sha256": _sha(path)}
                            for key, path in paths.items()}}
            self.assertEqual(_admitted_source(cfg, registered)[-1], sequential_root)
            public["seq_fit_root_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(public))
            cfg["pins"]["stream_manifest"]["sha256"] = _sha(manifest_path)
            registered[manifest_path.resolve()] = _sha(manifest_path)
            with self.assertRaisesRegex(PermissionError, "sequential FIT root"):
                _admitted_source(cfg, registered)


if __name__ == "__main__":
    unittest.main()
