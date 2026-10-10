"""No-role-data tests for queued attack preflight and release replay."""

import unittest
from types import SimpleNamespace

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array

from .attack_worker import validate_archive_arrays, validate_attack_lock


def fixture(bits=16):
    rows = np.arange(4, dtype=np.int64)
    units = np.array([0, 0, 2, 3], dtype=np.int64)
    code0 = np.zeros((4, bits), dtype=np.uint8)
    code1 = np.zeros((4, bits), dtype=np.uint8)
    code0[1, 0] = 1
    code1[2, 1] = 1

    def probabilities(purpose, code):
        width = (2, 6)[purpose]
        p = np.full((len(code), width), 0.1 / (width - 1), dtype=np.float64)
        p[:, 0] = 0.9
        p[code.sum(axis=1) > 0] = 1.0 / width
        return p

    class Service:
        model = SimpleNamespace(latent_dim=16)

        def probabilities_from_code(self, purpose, code):
            return probabilities(purpose, code)

        def recipient(self, purpose, latent):
            code = np.zeros((len(latent), bits), dtype=np.uint8)
            probability = probabilities(purpose, code)
            return {"code": code, "probabilities": probability,
                    "decision": probability.argmax(axis=1).astype(np.int64),
                    "metadata": {"purpose": purpose, "bits": bits,
                                 "version": "synthetic", "schema": "hard-code"}}

    arrays = {"row_id": rows, "unit": units, "code0": code0, "code1": code1,
              "prob0": probabilities(0, code0),
              "prob1": probabilities(1, code1),
              "decision0": np.zeros(4, dtype=np.int64),
              "decision1": np.zeros(4, dtype=np.int64)}
    meta = {"rows": 4, "groups": 3,
            "row_unit_pair_sha256": sha_array(np.column_stack((rows, units)))}
    return arrays, meta, Service()


class WorkerPrefitTest(unittest.TestCase):
    def test_archive_replays_code_only_outputs(self):
        arrays, meta, service = fixture()
        actual = validate_archive_arrays(arrays, meta, service, bits=16)
        self.assertEqual(set(actual), {"row_id", "unit", "first", "second"})
        self.assertEqual(actual["first"]["code"].shape, (4, 16))

    def test_archive_rejects_extra_member_and_soft_bit(self):
        arrays, meta, service = fixture()
        arrays["teacher_score"] = arrays["prob0"]
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_archive_arrays(arrays, meta, service, bits=16)
        arrays, meta, service = fixture()
        arrays["code0"] = arrays["code0"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "hard code"):
            validate_archive_arrays(arrays, meta, service, bits=16)

    def test_archive_rejects_head_bypass_and_role_mismatch(self):
        arrays, meta, service = fixture()
        arrays["prob0"] = arrays["prob0"].copy()
        arrays["prob0"][0] = [0.8, 0.2]
        with self.assertRaisesRegex(ValueError, "code-only"):
            validate_archive_arrays(arrays, meta, service, bits=16)
        arrays, meta, service = fixture()
        arrays["unit"][3] = 7
        with self.assertRaisesRegex(PermissionError, "row/group"):
            validate_archive_arrays(arrays, meta, service, bits=16)

    def test_lock_binds_every_source_identity(self):
        digest = "a" * 64
        units = {"ATTACK_FIT_NEW": "b" * 64, "INNER_SELECTION": "c" * 64}
        registered = {"arm_id": "T0", "seed": 0, "bits": 16,
                      "source_model_sha256": "d" * 64,
                      "service_manifest_sha256": "e" * 64,
                      "release_sha256": units}
        lock = {"schema": "pcrl-code-witness-attack-lock-v1",
                "status": "LOCALLY_SEALED_ATTACKS",
                "campaign_seal_sha256": digest,
                "role_manifest_sha256": "f" * 64,
                "worker_sha256": "1" * 64,
                "reader_plan": {"core": ["LR_C1", "MLP_64x32", "HGB_15x200",
                                         "DA_MLP_64x32", "CELL_A1", "CODE_BIT_LR_C1"],
                                "probability": ["PROB_LR_C1", "PROB_HGB_15x200"],
                                "controls": ["SHUFFLED_LR", "PLANTED_LR"],
                                "views": ["r1", "r2", "pair"]},
                "units": {"ATTACK_T0_S0": registered}}
        inputs = dict(campaign_seal_sha256=digest, role_manifest_sha256="f" * 64,
                      worker_sha256="1" * 64, unit_id="ATTACK_T0_S0", arm_id="T0",
                      seed=0, bits=16, source_model_sha256="d" * 64,
                      service_manifest_sha256="e" * 64,
                      release_sha256=units)
        validate_attack_lock(lock, **inputs)
        inputs["bits"] = 10
        with self.assertRaisesRegex(PermissionError, "unit"):
            validate_attack_lock(lock, **inputs)


if __name__ == "__main__":
    unittest.main()
