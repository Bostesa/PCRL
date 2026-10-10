import unittest
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array

from .policy_worker import (ARMS, LAW_MEMBERS, RELEASE_MEMBERS, SCHEMA,
                            SOURCES, SPEC_KEYS, load_role_release, validate_spec)


class PolicyWorkerContractTests(unittest.TestCase):
    def setUp(self):
        sha = "b" * 64
        self.spec = {key: "/private/file" for key in SPEC_KEYS}
        self.spec.update({
            "schema": SCHEMA, "unit_id": "R4_J_S0_R1", "arm_id": "J",
            "source_seed": 0, "realization": 1, "attacker_seed": 28,
            "release_directory_by_role": {
                "ATTACK_FIT_NEW": "/private/att",
                "INNER_SELECTION": "/private/inner"},
            "release_receipt_sha256_by_role": {
                "ATTACK_FIT_NEW": sha, "INNER_SELECTION": sha},
            "source_pins": [{"path": f"/sources/{name}", "sha256": sha}
                            for name in sorted(SOURCES)],
        })
        for key in SPEC_KEYS:
            if key.endswith("sha256"):
                self.spec[key] = sha

    def test_all_declared_arms_and_draw_restriction(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                validate_spec(self.spec | {"arm_id": arm,
                                           "realization": 0 if arm == "D" else 1})
        with self.assertRaisesRegex(ValueError, "identity differs"):
            validate_spec(self.spec | {"arm_id": "D", "realization": 1})

    def test_missing_role_or_source_refused(self):
        with self.assertRaisesRegex(ValueError, "role-only"):
            validate_spec(self.spec | {"release_directory_by_role":
                                       {"ATTACK_FIT_NEW": "/private/att"}})
        with self.assertRaisesRegex(ValueError, "source inventory"):
            validate_spec(self.spec | {"source_pins": self.spec["source_pins"][:-1]})

    def test_private_member_sets_are_exact(self):
        self.assertEqual(RELEASE_MEMBERS,
                         {"row_id", "unit", "code_r1", "code_r2"})
        self.assertEqual(LAW_MEMBERS - RELEASE_MEMBERS,
                         {"mass", "valid", "b0_code_r1", "b0_code_r2"})

    def test_release_support_and_cached_group_contract(self):
        tables = (np.array([[.8, .2], [.3, .7]], dtype=np.float64),
                  np.array([[.7, .1, .05, .05, .05, .05],
                            [.1, .7, .05, .05, .05, .05]], dtype=np.float64))
        row = np.array([1, 2, 3], dtype=np.int64)
        unit = np.array([8, 9, 9], dtype=np.int64)
        c1 = np.array([0, 1, 1], dtype=np.int64)
        c2 = np.array([0, 1, 1], dtype=np.int64)
        law = {"row_id": row, "unit": unit,
               "code_r1": np.repeat(c1[:, None], 16, axis=1),
               "code_r2": np.repeat(c2[:, None], 16, axis=1),
               "mass": np.zeros((3, 16), dtype=np.float64),
               "valid": np.zeros((3, 16), dtype=np.bool_),
               "b0_code_r1": c1, "b0_code_r2": c2}
        law["mass"][:, 0] = 1
        law["valid"][:, 0] = True
        release = {"row_id": row, "unit": unit,
                   "code_r1": c1, "code_r2": c2}
        role_meta = {"rows": 3, "groups": 2,
                     "row_unit_pair_sha256": sha_array(np.column_stack((row, unit)))}
        with tempfile.TemporaryDirectory() as temp:
            private = Path(temp) / ".private"
            private.mkdir(mode=0o700)
            root = private / "unit"
            root.mkdir(mode=0o700)
            def write(arrays, filename):
                path = root / filename
                with path.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                os.chmod(path, 0o600)
                return hashlib.sha256(path.read_bytes()).hexdigest()
            law_sha = write(law, "law.npz")
            code_sha = write(release, "release_codes.npz")
            sha = "b" * 64
            receipt = {"schema": "BAR1-role-release-v1", "arm_id": "J",
                       "source_seed": 0, "role": "ATTACK_FIT_NEW",
                       "realization": 1, "source_binding_sha256": sha,
                       "policy_receipt_sha256": sha, "rows": 3,
                       "law_sha256": law_sha,
                       "release_codes_sha256": code_sha}
            receipt_path = root / "RELEASE_RECEIPT.json"
            receipt_path.write_text(json.dumps(receipt))
            os.chmod(receipt_path, 0o600)
            receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            out, _ = load_role_release(str(root), receipt_sha, arm_id="J",
                seed=0, realization=1, role="ATTACK_FIT_NEW",
                policy_receipt_sha256=sha, source_binding_sha256=sha,
                role_meta=role_meta, tables=tables)
            self.assertTrue(np.array_equal(out["prob0"], tables[0][c1]))
            release["code_r1"] = np.array([1, 1, 1], dtype=np.int64)
            code_sha = write(release, "release_codes.npz")
            receipt["release_codes_sha256"] = code_sha
            receipt_path.write_text(json.dumps(receipt))
            receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "positive law support"):
                load_role_release(str(root), receipt_sha, arm_id="J",
                    seed=0, realization=1, role="ATTACK_FIT_NEW",
                    policy_receipt_sha256=sha, source_binding_sha256=sha,
                    role_meta=role_meta, tables=tables)


if __name__ == "__main__":
    unittest.main()
