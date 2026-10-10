import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from bank_anchored.r5_pstar_view import array_sha, encode, role_fields, verify


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SeparateReferenceViewTests(unittest.TestCase):
    def test_full_finite_source_and_missing_code_bit(self):
        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            bank1 = np.array([[.7, .3], [.2, .8]], dtype=np.float64)
            bank2 = np.array([[.5, .1, .1, .1, .1, .1],
                              [.1, .1, .1, .1, .1, .5]], dtype=np.float64)
            fixed = root / "fixed.npz"
            arrays = {"bank_r1": bank1, "bank_r2": bank2,
                      "fine_to_code_r1": np.array([0, 1], np.int64),
                      "fine_to_code_r2": np.array([1, 0], np.int64)}
            np.savez_compressed(fixed, **arrays)
            binding = "a" * 64
            fixed_receipt = root / "fixed.json"
            fixed_receipt.write_text(json.dumps({
                "source_binding_sha256": binding, "npz_sha256": digest(fixed),
                "array_sha256": {name: array_sha(value)
                                 for name, value in arrays.items()}}))
            spec = {"arm_id": "P_STAR", "source_seed": 0,
                    "fixed_map_receipt_path": str(fixed_receipt),
                    "fixed_map_receipt_sha256": digest(fixed_receipt),
                    "fixed_map_archive_path": str(fixed),
                    "fixed_map_archive_sha256": digest(fixed)}
            role_fields_by_name = {}
            for role, ids in (("att", (np.array([0, 1], np.int64),
                                       np.array([1, 0], np.int64))),
                              ("inner", (np.array([1, 0], np.int64),
                                         np.array([0, 1], np.int64)))):
                directory = root / role
                directory.mkdir()
                values = {"row_id": np.array([1, 2], np.int64),
                          "unit": np.array([5, 6], np.int64),
                          "code_r1": ids[0], "code_r2": ids[1],
                          "prob_r1": bank1[ids[0]], "prob_r2": bank2[ids[1]],
                          "decision_r1": bank1[ids[0]].argmax(axis=1).astype(np.int64),
                          "decision_r2": bank2[ids[1]].argmax(axis=1).astype(np.int64)}
                data = directory / "reference_release.npz"
                np.savez_compressed(data, **values)
                receipt = {"codes_file_sha256": digest(data),
                           "source_binding_sha256": binding,
                           "interface_kind": "finite", "arm_id": "P_STAR",
                           "source_seed": 0, "role": "ATTACK_FIT_NEW" if role == "att" else "INNER_SELECTION",
                           "rows": 2, "groups": 2,
                           "row_unit_pair_sha256": array_sha(np.column_stack((values["row_id"], values["unit"]))),
                           "release_id": "pstar-fixture",
                           "bank_sha256_by_recipient": {"r1": array_sha(bank1),
                                                        "r2": array_sha(bank2)},
                           "wire_sha256_by_recipient": {}}
                for recipient in (1, 2):
                    bank = (bank1, bank2)[recipient - 1]
                    name = f"r{recipient}"
                    metadata = {"model_id": receipt["release_id"],
                                "source_binding_sha256": binding,
                                "bank_sha256": receipt["bank_sha256_by_recipient"][name],
                                "code_length_bits": (len(bank) - 1).bit_length()}
                    h = hashlib.sha256()
                    for i in range(2):
                        wire = {"schema": "SCV1-reference-score-release-v1",
                                "recipient": recipient,
                                "probability": values[f"prob_{name}"][i].tolist(),
                                "decision": int(values[f"decision_{name}"][i]),
                                "release_id": receipt["release_id"],
                                "public_metadata": metadata,
                                "code_id": int(values[f"code_{name}"][i])}
                        h.update(json.dumps(wire, sort_keys=True,
                                            separators=(",", ":")).encode() + b"\n")
                    receipt["wire_sha256_by_recipient"][name] = h.hexdigest()
                receipt_path = directory / "REFERENCE_RELEASE_RECEIPT.json"
                receipt_path.write_text(json.dumps(receipt))
                spec[f"{role}_package_dir"] = str(directory)
                spec[f"{role}_receipt_sha256"] = digest(receipt_path)
                role_fields_by_name[role] = (role_fields(values, receipt, 1, bank1),
                                             role_fields(values, receipt, 2, bank2))
            for view in ("r1", "r2", "pair"):
                def named(role):
                    fields = role_fields_by_name[role]
                    chosen = (0,) if view == "r1" else (1,) if view == "r2" else (0, 1)
                    return {((f"r{i+1}." if view == "pair" else "") + key): val
                            for i in chosen for key, val in fields[i].items()}
                codes = ("r1.code_id", "r2.code_id") if view == "pair" else ("code_id",)
                expected = encode(named("att"), named("inner"), code_fields=codes)
                out = root / view / "readers"
                out.mkdir(parents=True)
                for role in ("att", "inner"):
                    np.savez_compressed(out / f"view_{role}.npz",
                                        row_id=np.array([1, 2], np.int64),
                                        unit=np.array([5, 6], np.int64),
                                        **expected[role])
            self.assertEqual(verify(spec, root)["status"], "PASS")
            with np.load(root / "pair/readers/view_inner.npz", allow_pickle=False) as source:
                wrong = {key: source[key].copy() for key in source.files}
            wrong["x_code_bits"][0, 0] = 1 - wrong["x_code_bits"][0, 0]
            np.savez_compressed(root / "pair/readers/view_inner.npz", **wrong)
            with self.assertRaises(ValueError):
                verify(spec, root)


if __name__ == "__main__":
    unittest.main()
