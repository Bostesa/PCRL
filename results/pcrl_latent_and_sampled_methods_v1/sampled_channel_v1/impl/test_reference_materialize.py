"""Data-free exact public reference-wire and private-custody checks."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array

from .reference_materialize import materialize_reference_role, reconstruct_reference_wires


class ReferenceMaterializeTest(unittest.TestCase):
    def _inputs(self, root: Path, finite: bool) -> dict:
        row = np.array([11, 12], np.int64)
        unit = np.array([21, 22], np.int64)
        bank1 = np.array([[.8, .2], [.1, .9]], np.float64)
        bank2 = np.array([[.7, .1, .1, .05, .03, .02],
                          [.01, .75, .1, .05, .05, .04]], np.float64)
        code1 = np.array([0, 1], np.int64) if finite else np.full(2, -1, np.int64)
        code2 = np.array([0, 1], np.int64) if finite else np.full(2, -1, np.int64)
        return dict(
            row_id=row, unit=unit, prob1=bank1.copy(), prob2=bank2.copy(),
            decision1=np.array([0, 1], np.int64),
            decision2=np.array([0, 1], np.int64),
            code1=code1, code2=code2,
            arm_id="TSTAR" if finite else "U0", seed=0,
            role="ATTACK_FIT_NEW", release_id="TSTAR-s0-v1" if finite else "U0-s0-v1",
            source_binding_sha256="a" * 64,
            source_identity={"role_source_file_sha256": "b" * 64,
                             "reference_file_sha256": "c" * 64,
                             "bank_receipt_sha256": "d" * 64 if finite else None,
                             "source_audit_sha256": "e" * 64},
            interface_kind="finite" if finite else "continuous",
            bank1=bank1 if finite else None,
            bank2=bank2 if finite else None,
            bank_sha256_by_recipient={
                "r1": sha_array(bank1) if finite else None,
                "r2": sha_array(bank2) if finite else None},
            expected_row_unit_sha256=sha_array(np.column_stack((row, unit))),
            output_dir=root / ".private" / ("finite" if finite else "continuous"))

    def test_finite_codes_are_actual_bank_vectors_and_wire(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / ".private").mkdir(mode=0o700)
            inputs = self._inputs(root, True)
            receipt = materialize_reference_role(**inputs)
            self.assertEqual(receipt["status"], "FROZEN_DETERMINISTIC")
            wires = reconstruct_reference_wires(
                inputs["prob1"], inputs["decision1"], inputs["code1"],
                recipient=1, interface_kind="finite",
                release_id=inputs["release_id"],
                source_binding_sha256=inputs["source_binding_sha256"],
                bank_sha256=inputs["bank_sha256_by_recipient"]["r1"],
                bank_size=len(inputs["bank1"]))
            self.assertEqual(wires[0]["code_id"], 0)
            self.assertEqual(wires[0]["public_metadata"]["code_length_bits"], 1)
            digest = hashlib.sha256()
            for wire in wires:
                digest.update(json.dumps(wire, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode() + b"\n")
            self.assertEqual(receipt["wire_sha256_by_recipient"]["r1"],
                             digest.hexdigest())
            with self.assertRaises(PermissionError):
                materialize_reference_role(**inputs)
            bad = self._inputs(root, True)
            bad["output_dir"] = root / ".private" / "bad"
            bad["prob1"][0] = bad["prob1"][1]
            with self.assertRaises(ValueError):
                materialize_reference_role(**bad)

    def test_continuous_wire_has_no_fabricated_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / ".private").mkdir(mode=0o700)
            inputs = self._inputs(root, False)
            receipt = materialize_reference_role(**inputs)
            wires = reconstruct_reference_wires(
                inputs["prob1"], inputs["decision1"], inputs["code1"],
                recipient=1, interface_kind="continuous",
                release_id=inputs["release_id"],
                source_binding_sha256=inputs["source_binding_sha256"],
                bank_sha256=None, bank_size=None)
            self.assertNotIn("code_id", wires[0])
            self.assertIsNone(wires[0]["public_metadata"]["code_length_bits"])
            self.assertEqual(receipt["bank_sha256_by_recipient"],
                             {"r1": None, "r2": None})

    def test_duplicate_group_cannot_receive_two_deterministic_wires(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / ".private").mkdir(mode=0o700)
            inputs = self._inputs(root, False)
            inputs["unit"] = np.array([21, 21], np.int64)
            inputs["expected_row_unit_sha256"] = sha_array(
                np.column_stack((inputs["row_id"], inputs["unit"])))
            with self.assertRaisesRegex(ValueError, "duplicate unit"):
                materialize_reference_role(**inputs)


if __name__ == "__main__":
    unittest.main()
