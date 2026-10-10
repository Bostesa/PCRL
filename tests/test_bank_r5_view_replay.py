import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bank_anchored.r5_view_replay import matrices, public_wire, verify, wire_sha


class IndependentPublicViewTests(unittest.TestCase):
    def test_bits_bank_and_unknown_category(self):
        bank0 = np.tile(np.array([[0.7, 0.3]], dtype=np.float64), (16, 1))
        bank1 = np.tile(np.array([[0.4, 0.2, 0.1, 0.1, 0.1, 0.1]],
                                 dtype=np.float64), (20, 1))
        att = (np.array([2, 10], dtype=np.int64),
               np.array([3, 13], dtype=np.int64))
        inner = (np.array([10, 4], dtype=np.int64),
                 np.array([13, 5], dtype=np.int64))
        one = public_wire(att[0], bank0)
        self.assertEqual(one["code"].dtype, np.uint8)
        self.assertEqual(one["code"][0, :4].tolist(), [0, 1, 0, 0])
        self.assertEqual(one["code"][1, :4].tolist(), [0, 1, 0, 1])
        self.assertEqual(len(wire_sha(one)), 64)
        view = matrices(att, inner, (bank0, bank1), "pair")
        self.assertEqual(view["att"]["x"].shape, (2, 36))
        self.assertEqual(view["att"]["x_defense_aware"].shape, (2, 144))
        self.assertEqual(view["att"]["x_code_bits"].shape, (2, 20))
        self.assertEqual(view["inner"]["cell_key"].tolist(),
                         ['["10","13"]', '["4","5"]'])
        self.assertEqual(view["inner"]["x"][1, 12], 1.0)
        self.assertEqual(view["inner"]["x"][1, 28], 1.0)
        self.assertEqual(view["att"]["x_defense_aware"][0, -100:].sum(), 2.0)

    def test_reject_out_of_bank_code(self):
        bank = np.array([[0.6, 0.4]], dtype=np.float64)
        with self.assertRaises(ValueError):
            public_wire(np.array([1], dtype=np.int64), bank)

    def test_witness_original_wire_complete_views(self):
        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            banks = (np.array([[.7, .3], [.2, .8]], dtype=np.float64),
                     np.array([[.5, .1, .1, .1, .1, .1],
                               [.1, .1, .1, .1, .1, .5]], dtype=np.float64))
            table = root / "table.json"
            table.write_text(json.dumps({"families": {"W_Q": {
                "table0": banks[0].tolist(), "table1": banks[1].tolist()}}}))
            codes = {"att": (np.array([0, 1], dtype=np.int64),
                              np.array([1, 0], dtype=np.int64)),
                     "inner": (np.array([1, 0], dtype=np.int64),
                               np.array([0, 1], dtype=np.int64))}
            spec = {"arm_id": "W_Q", "public_table_path": str(table),
                    "public_table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                    "release_path_by_role": {}, "release_sha256_by_role": {}}
            public = {"public_side_information_by_role": {}}
            for role, source_role in (("att", "ATTACK_FIT_NEW"),
                                      ("inner", "INNER_SELECTION")):
                wires = [public_wire(codes[role][i], banks[i]) for i in (0, 1)]
                path = root / f"{role}.npz"
                np.savez_compressed(path, row_id=np.array([1, 2], np.int64),
                                    unit=np.array([5, 6], np.int64),
                                    code0=wires[0]["code"], code1=wires[1]["code"],
                                    prob0=wires[0]["prob"], prob1=wires[1]["prob"],
                                    decision0=wires[0]["decision"],
                                    decision1=wires[1]["decision"])
                spec["release_path_by_role"][source_role] = str(path)
                spec["release_sha256_by_role"][source_role] = hashlib.sha256(path.read_bytes()).hexdigest()
                public["public_side_information_by_role"][source_role] = {
                    str(i): {"wire_npz_sha256": wire_sha(wires[i])} for i in (0, 1)}
            for view in ("r1", "r2", "pair"):
                expected = matrices(codes["att"], codes["inner"], banks, view)
                target = root / view / "readers"
                target.mkdir(parents=True)
                for role in ("att", "inner"):
                    np.savez_compressed(target / f"view_{role}.npz",
                                        row_id=np.array([1, 2], np.int64),
                                        unit=np.array([5, 6], np.int64),
                                        **{k: expected[role][k] for k in
                                           ("x", "x_defense_aware", "x_code_bits", "cell_key")})
            self.assertEqual(verify(spec, public, root)["status"], "PASS")
            public["public_side_information_by_role"]["INNER_SELECTION"]["1"]["wire_npz_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                verify(spec, public, root)


if __name__ == "__main__":
    unittest.main()
