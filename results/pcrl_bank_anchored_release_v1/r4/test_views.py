import unittest
import hashlib
import io

import numpy as np

from .views import (bits_from_ids, ids_from_bits, materialize_role_from_ids,
                    prepare_complete_bank_views, public_wire_npz_sha256)


class BankCompleteViewTests(unittest.TestCase):
    def setUp(self):
        self.tables = (
            np.array([[.8, .2], [.3, .7]], dtype=np.float64),
            np.array([[.6, .1, .1, .1, .05, .05],
                      [.1, .6, .1, .1, .05, .05]], dtype=np.float64),
        )
        self.att = materialize_role_from_ids(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([11, 12, 12], dtype=np.int64),
            np.array([0, 1, 1], dtype=np.int64),
            np.array([1, 0, 0], dtype=np.int64), self.tables)
        self.inner = materialize_role_from_ids(
            np.array([4, 5], dtype=np.int64),
            np.array([14, 15], dtype=np.int64),
            np.array([1, 0], dtype=np.int64),
            np.array([0, 1], dtype=np.int64), self.tables)

    def test_exact_three_views_and_public_fields(self):
        metadata = ({"arm": "J", "seed": 0}, {"arm": "J", "seed": 0})
        out = prepare_complete_bank_views(
            self.att, self.inner, self.tables,
            metadata_att=metadata, metadata_inner=metadata)
        self.assertEqual(set(out), {"r1", "r2", "pair"})
        self.assertEqual(out["r1"].att.wire_components,
                         ("code", "prob", "decision", "metadata.arm", "metadata.seed"))
        self.assertEqual(out["pair"].att.x_code_bits.shape, (3, 20))
        self.assertEqual(out["pair"].att.cell_key.shape, (3,))
        self.assertEqual(out["pair"].att.x_defense_aware.shape[0], 3)
        self.assertEqual(out["pair"].probability_inner.shape, (2, 8))

    def test_ids_roundtrip_little_endian(self):
        ids = np.array([0, 1, 2, 513], dtype=np.int64)
        self.assertTrue(np.array_equal(ids_from_bits(bits_from_ids(ids, 514), 514), ids))

    def test_rejects_hidden_output_change(self):
        role = dict(self.inner)
        role["prob0"] = role["prob0"].copy()
        role["prob0"][0] = [0.7, 0.3]
        with self.assertRaisesRegex(ValueError, "frozen code table"):
            prepare_complete_bank_views(self.att, role, self.tables,
                                        metadata_att=({}, {}), metadata_inner=({}, {}))

    def test_rejects_duplicate_group_uncached_output(self):
        role = dict(self.att)
        role["code0"] = role["code0"].copy()
        role["prob0"] = role["prob0"].copy()
        role["decision0"] = role["decision0"].copy()
        role["code0"][2] = bits_from_ids(np.array([0], dtype=np.int64), 2)[0]
        role["prob0"][2] = self.tables[0][0]
        role["decision0"][2] = 0
        with self.assertRaisesRegex(ValueError, "different cached"):
            prepare_complete_bank_views(role, self.inner, self.tables,
                                        metadata_att=({}, {}), metadata_inner=({}, {}))

    def test_rejects_metadata_schema_change(self):
        with self.assertRaisesRegex(ValueError, "metadata field schema"):
            prepare_complete_bank_views(self.att, self.inner, self.tables,
                                        metadata_att=({"arm": "J"}, {}),
                                        metadata_inner=({}, {}))

    def test_public_wire_hash_matches_deployment_npz_bytes(self):
        stream = io.BytesIO()
        np.savez_compressed(stream, code=self.att["code0"],
                            prob=self.att["prob0"], decision=self.att["decision0"])
        self.assertEqual(public_wire_npz_sha256(self.att, 0, self.tables),
                         hashlib.sha256(stream.getvalue()).hexdigest())


if __name__ == "__main__":
    unittest.main()
