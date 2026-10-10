"""Small attack invariants; no Adult data or fit is opened."""

import unittest

import numpy as np

from .adapter import prepared_d_views
from .cell_reader import CellReader
from .control_wires import append_planted_public_bit, noisy_public_bit, shuffled_sex, xor_public_bits
from .hard_code import AlignedPair, PublicCode, code_ids, coverage


class HardCodeTest(unittest.TestCase):
    def test_code_bit_order_and_complete_wire(self):
        bits = np.zeros((2, 10), dtype=np.uint8)
        bits[0, 0] = 1
        bits[1, 6] = 1
        np.testing.assert_array_equal(code_ids(bits), [1, 64])
        q = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        release = {"code": bits, "probabilities": q,
                   "decision": np.array([0, 1], dtype=np.int64),
                   "metadata": {"purpose": 0, "bits": 10, "version": "fixture"}}
        a = PublicCode.from_release(np.array([101, 102]), release)
        b = PublicCode.from_release(np.array([101, 102]),
                                    {**release, "metadata": {**release["metadata"], "purpose": 1}})
        self.assertEqual(a.code_id.tolist(), [1, 64])
        self.assertEqual(AlignedPair(a, b).pair_ids.tolist(), [[1, 1], [64, 64]])
        with self.assertRaises(ValueError):
            PublicCode.from_release(np.array([101, 102]), {**release, "soft_bits": bits})
        with self.assertRaises(ValueError):
            PublicCode.from_release(np.array([101, 102]),
                                    {**release, "metadata": {"flag": np.array([0, 1])}})

    def test_pair_packing_does_not_wrap_uint16(self):
        fit = np.array([[0, 1], [64, 1]], dtype=np.uint16)
        screen = np.array([[64, 1], [65, 1]], dtype=np.uint16)
        receipt = coverage(fit, screen)
        self.assertEqual(receipt["fit_occupied"], 2)
        self.assertEqual(receipt["screen_unseen_people"], 1)

    def test_xor_pair_reader_exposes_complementary_clue(self):
        rows = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 20, dtype=np.uint16)
        sex = np.bitwise_xor(rows[:, 0], rows[:, 1]).astype(np.uint8)
        local0 = CellReader.fit(rows[:, 0], sex, 0.5)
        local1 = CellReader.fit(rows[:, 1], sex, 0.5)
        pair = CellReader.fit(rows, sex, 0.5)
        np.testing.assert_allclose(local0.predict_proba(rows[:, 0])[:, 1], 0.5)
        np.testing.assert_allclose(local1.predict_proba(rows[:, 1])[:, 1], 0.5)
        p = pair.predict_proba(rows)[:, 1]
        self.assertGreater(float(p[sex == 1].min()), 0.9)
        self.assertLess(float(p[sex == 0].max()), 0.1)

    def test_r02_complete_view_and_pair_public_cross_bits(self):
        def release(bit0, bit1, purpose):
            bits = np.zeros((2, 10), dtype=np.uint8)
            bits[:, 0] = bit0
            bits[:, 1] = bit1
            q = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=np.float32)
            return {"code": bits, "probabilities": q,
                    "decision": np.array([0, 1], dtype=np.int64),
                    "metadata": {"purpose": purpose, "bits": 10,
                                 "schema": "latent-repair-hardbits-v1", "version": "fixture",
                                 "fallback": False}}
        rows = np.array([101, 102], dtype=np.int64)
        groups = rows.copy()
        first = release([0, 1], [0, 0], 0)
        second = release([0, 1], [1, 0], 1)
        result = prepared_d_views(rows, groups, first, second,
                                  rows + 10, groups + 10, first, second)
        self.assertEqual(tuple(result), ("r1", "r2", "pair"))
        self.assertEqual(len(result["pair"].att.cell_key), 2)
        self.assertEqual(result["pair"].att.x_code_bits.shape, (2, 20))
        self.assertEqual(result["pair"].att.x_defense_aware.shape[1] -
                         result["pair"].att.x.shape[1], 104)
        self.assertTrue(any(f["name"] == "r1.code_bits" and f["att_categories"] is not None
                            for f in result["pair"].schema["fields"]))

    def test_grouped_controls_keep_duplicates_and_plant_public_category(self):
        sex = np.array([0, 0, 1, 1, 0, 1], dtype=np.uint8)
        unit = np.array([10, 10, 20, 20, 30, 40], dtype=np.int64)
        shuffled = shuffled_sex(sex, unit, 17)
        self.assertEqual(shuffled[0], shuffled[1])
        self.assertEqual(shuffled[2], shuffled[3])
        self.assertEqual(int(shuffled.sum()), int(sex.sum()))
        plant = noisy_public_bit(sex, unit, 17)
        self.assertEqual(plant[0], plant[1])
        b1, b2 = xor_public_bits(sex, unit, 17)
        self.assertEqual(b1[0], b1[1])
        self.assertEqual(b2[2], b2[3])
        from ...r02.complete_view import Field, WireBatch
        code = ((np.arange(6)[:, None] >> np.arange(10)[None, :]) & 1).astype(np.uint8)
        batch = WireBatch(np.arange(6, dtype=np.int64), unit,
                          {"code_bits": Field("bits", code)},
                          code_fields=("code_bits",))
        augmented = append_planted_public_bit(batch, plant)
        self.assertEqual(augmented.code_fields, ("control_code_id",))
        self.assertEqual(augmented.fields["control_code_id"].values.tolist(),
                         (np.arange(6) * 2 + plant).tolist())

    def test_actual_r07_synthetic_service_wire_reaches_r02(self):
        from ..impl.model import FixedScaler, HardCodeModel, HardCodeRelease
        model = HardCodeModel(16, (2, 6))
        scaler = FixedScaler(np.zeros((2, 16), dtype=np.float32),
                             np.ones((2, 16), dtype=np.float32))
        service = HardCodeRelease(model, scaler, "synthetic-fixture", "a" * 64)
        h = np.zeros((2, 16), dtype=np.float32)
        r1, r2 = service.recipient(0, h), service.recipient(1, h)
        rows = np.array([11, 12], dtype=np.int64)
        result = prepared_d_views(rows, rows, r1, r2, rows + 10, rows + 10, r1, r2)
        self.assertEqual(result["r1"].att.x_code_bits.shape, (2, 10))
        self.assertEqual(result["r2"].att.x_code_bits.shape, (2, 10))
        self.assertEqual(result["pair"].att.x_code_bits.shape, (2, 20))


if __name__ == "__main__":
    unittest.main()
