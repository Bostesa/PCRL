"""Synthetic control-runner stage and admission checks; no Adult file is opened."""

import hashlib
import json
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

import control_runner


class ControlRunnerGateTests(unittest.TestCase):
    def test_unadmitted_raw_joint_refused_before_role_store_context(self):
        args = types.SimpleNamespace(campaign_lock="lock", lock_sha256="hash",
                                     local_seal_receipt="seal", data_manifest="manifest",
                                     shared_root="shared", arm_id="D_RAW_J")
        with patch.object(control_runner.base, "_check_seal", return_value=({}, {}, "")):
            with patch.object(control_runner.base, "_context",
                              side_effect=AssertionError("role read should not occur")):
                with self.assertRaises(PermissionError):
                    control_runner._context(args)

    def test_unknown_control_refused_before_role_store_context(self):
        args = types.SimpleNamespace(campaign_lock="lock", lock_sha256="hash",
                                     local_seal_receipt="seal", data_manifest="manifest",
                                     shared_root="shared", arm_id="D_UNKNOWN")
        with patch.object(control_runner.base, "_check_seal", return_value=({}, {}, "")):
            with patch.object(control_runner.base, "_context",
                              side_effect=AssertionError("role read should not occur")):
                with self.assertRaises(PermissionError):
                    control_runner._context(args)


class ControlRunnerStageTests(unittest.TestCase):
    """Exercise the real stage wiring with tiny fake roles and fake fit kernels."""

    def test_fit_calibrate_release_control_modes_and_resume_refusal(self):
        class FakeStore:
            def __init__(self):
                self.calls = []

            def read(self, role, field, procedure=None):
                self.calls.append((role, field, procedure))
                n = 3 if role == "OSF_DEFENSE_FIT" else 2
                rows = np.arange(n, dtype=np.int64) + {
                    "OSF_DEFENSE_FIT": 0, "CALIBRATION_HELDOUT": 10,
                    "ATTACK_FIT_NEW": 20, "INNER_SELECTION": 30,
                }[role]
                out = {"row_id": rows, "unit": rows.copy()}
                if field == "features":
                    out.update(X=np.zeros((n, 2), dtype=np.float32),
                               feature_names=np.array(["a", "b"]))
                elif field == "income":
                    out["income"] = np.array([0, 1, 0], dtype=np.int64)[:n]
                elif field == "occupation":
                    out["occupation"] = np.array([0, 1, 2], dtype=np.int64)[:n]
                elif field == "sex":
                    out["sex"] = np.array([0, 1, 0], dtype=np.int64)[:n]
                else:
                    raise AssertionError(f"unregistered role field: {field}")
                return out

        class FakeEncoder:
            seed = 0
            model_sha256 = "b" * 64

            def encode_one(self, purpose, x, columns, role):
                return np.full((len(x), 16), float(purpose), dtype=np.float32)

            def encode_with_receipt(self, x, columns, role):
                return (self.encode_one(0, x, columns, role),
                        self.encode_one(1, x, columns, role),
                        {"role": role, "source_kind": "synthetic-only"})

        class FakeRelease:
            def __init__(self, mode):
                self.mode = mode

            def recipient(self, purpose, latent):
                n = len(latent)
                code = self.mode in ("LOCAL", "SEQ12", "SEQ21")
                probabilities = (np.tile([0.75, 0.25], (n, 1)) if purpose == 0
                                 else np.tile([0.45, 0.2, 0.15, 0.1, 0.06, 0.04], (n, 1)))
                return {
                    "code_ids": (np.full((n, 1), purpose, dtype=np.int64) if code
                                 else np.empty((n, 0), dtype=np.int64)),
                    "decoded": np.asarray(latent, dtype=np.float32),
                    "probabilities": probabilities,
                    "decision": np.zeros(n, dtype=np.int64),
                    "metadata": {"purpose": purpose, "code_length_bits": 10 if code else 0,
                                 "block_count": 1 if code else 0, "fallback": False,
                                 "stochastic": False, "kind": self.mode,
                                 "calibration": "UNFITTED"},
                }

        class FakeCache:
            normalizer = object()
            purpose_centroids = (object(), object())
            receipt = {"cache_id": "cache_s0"}

            def validate(self, data, seed, config, model_shas, pair_sha):
                self_test.assertEqual(data.sex is None, data.mode == "UNCHANGED")
                self_test.assertEqual(model_shas, ("b" * 64, "b" * 64))
                self_test.assertEqual(pair_sha, "c" * 64)

        self_test = self
        roles = ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT",
                 "ATTACK_FIT_NEW", "INNER_SELECTION")
        manifest = {
            "roles": {"OSF_DEFENSE_FIT": {"rows": 3}},
            "real_file_specs": {
                f"{role}:{field}": {"sha256": "d" * 64,
                                   "row_unit_pair_sha256": "c" * 64}
                for role in roles
                for field in ("features", "income", "occupation", "sex", "references_s0")
            },
        }
        store = FakeStore()
        encoder = FakeEncoder()
        lock = {"schema": "synthetic-only"}

        def fake_fit_data(store_arg, encoder_arg, seed, protected):
            self.assertIs(store_arg, store)
            self.assertIs(encoder_arg, encoder)
            sex = store_arg.read("OSF_DEFENSE_FIT", "sex", "defense_fit")["sex"] if protected else None
            data = types.SimpleNamespace(sex=sex,
                                         mode="LEACE" if protected else "UNCHANGED")
            return data, {"role": "OSF_DEFENSE_FIT"}

        def fake_fit(data, seed, mode, config, **kwargs):
            if data is not None:
                self.assertEqual(data.sex is None, mode == "UNCHANGED")
            self.assertIs(kwargs["normalizer"], FakeCache.normalizer)
            if mode in ("LOCAL", "SEQ12", "SEQ21"):
                self.assertEqual(kwargs["initializers"], FakeCache.purpose_centroids)
            return types.SimpleNamespace(release=FakeRelease(mode), fit_rows=3,
                                         history=[{"epoch": 0}], map_metadata={})

        def fake_checkpoint_api():
            def save(fit, directory, binding, *, private_root):
                directory.mkdir(mode=0o700)
                control_runner.base._write_json(directory / "COMPLETE.json",
                                                {"binding": binding})
                return {"synthetic_only": True}

            def load(directory, binding, *, private_root, expected_mode):
                self.assertEqual(json.loads((directory / "COMPLETE.json").read_text()),
                                 {"binding": binding})
                return fake_fit(None, 0, expected_mode, None, normalizer=FakeCache.normalizer,
                                initializers=FakeCache.purpose_centroids)

            return load, save

        def fake_fit_temp(prob, labels, *, task, arm_id, model_seed, row_id, unit):
            self.assertEqual(len(prob), len(row_id))
            return {"task": task, "arm_id": arm_id,
                    "model_seed": model_seed, "temperature": 1.0}

        def fake_save_cal(record, path):
            payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            path.write_bytes(payload)
            return hashlib.sha256(payload).hexdigest()

        def fake_load_cal(path, sha):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sha)
            return json.loads(path.read_text())

        with tempfile.TemporaryDirectory(prefix="d-control-synthetic-") as temp:
            private_root = Path(temp)
            private_root.chmod(0o700)
            cache_dir = private_root / "cache_s0"
            cache_dir.mkdir(mode=0o700)
            np.savez(cache_dir / "arrays.npz", dummy=np.array([0]))
            with ExitStack() as stack:
                stack.enter_context(patch.object(control_runner.base, "_check_seal",
                                                 return_value=(lock, manifest, "e" * 64)))
                stack.enter_context(patch.object(control_runner, "_check_control_lock",
                                                 side_effect=lambda _lock, arm: control_runner.CONTROL_ARMS[arm]))
                stack.enter_context(patch.object(control_runner.base, "_context",
                                                 return_value=(lock, manifest, "e" * 64,
                                                               encoder.model_sha256, store,
                                                               encoder, private_root)))
                stack.enter_context(patch.object(control_runner.base, "_fit_data",
                                                 side_effect=fake_fit_data))
                stack.enter_context(patch.object(control_runner, "load_private_cache",
                                                 return_value=FakeCache()))
                stack.enter_context(patch.object(control_runner, "fit_continuous",
                                                 side_effect=fake_fit))
                stack.enter_context(patch.object(control_runner, "fit_local_or_sequential",
                                                 side_effect=fake_fit))
                stack.enter_context(patch.object(control_runner, "_checkpoint_api",
                                                 side_effect=fake_checkpoint_api))
                stack.enter_context(patch.object(control_runner.base, "fit_global_temperature",
                                                 side_effect=fake_fit_temp, create=True))
                stack.enter_context(patch.object(control_runner.base, "save_calibrator",
                                                 side_effect=fake_save_cal, create=True))
                stack.enter_context(patch.object(control_runner.base, "load_calibrator",
                                                 side_effect=fake_load_cal, create=True))
                stack.enter_context(patch.object(control_runner.base, "apply_global_temperature",
                                                 side_effect=lambda p, record: p, create=True))
                stack.enter_context(patch.object(control_runner.base, "assert_same_people",
                                                 side_effect=lambda *records: None, create=True))
                for arm in ("D_U_FROZEN", "D_VQ_LOCAL", "D_VQ_SEQ12",
                            "D_VQ_SEQ21", "D_LEACE"):
                    args = types.SimpleNamespace(campaign_lock="lock", lock_sha256="a" * 64,
                                                 local_seal_receipt="seal", data_manifest="manifest",
                                                 shared_root="shared", arm_id=arm, seed=0,
                                                 cache_id="cache_s0", run_id=arm)
                    self.assertEqual(control_runner.run_fit(args)["stage"], "FIT")
                    self.assertEqual(control_runner.run_calibrate(args)["stage"], "CALIBRATE")
                    self.assertEqual(control_runner.run_release(args)["stage"], "RELEASE")
                    run_dir = private_root / arm
                    for role in ("CALIBRATION_HELDOUT", "ATTACK_FIT_NEW", "INNER_SELECTION"):
                        package = run_dir / role
                        self.assertTrue((package / "COMPLETE.json").is_file())
                        with np.load(package / "wire.npz", allow_pickle=False) as wire:
                            self.assertEqual(set(wire.files), {"row_id", "unit"} |
                                             {f"r{i}_{field}" for i in (0, 1) for field in
                                              ("code_ids", "decoded", "probabilities", "decision")})
                            self.assertEqual(wire["r0_probabilities"].dtype, np.float64)
                    with self.assertRaises(FileExistsError):
                        control_runner.run_release(args)
        self.assertFalse(any(role in ("HEAD_VALIDATION", "OSF_DEVELOPMENT_ASSESSMENT")
                             for role, _, _ in store.calls))
        self.assertEqual(sum(role == "OSF_DEFENSE_FIT" and field == "sex"
                             for role, field, _ in store.calls), 4)

if __name__ == "__main__":
    unittest.main()
