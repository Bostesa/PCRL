import unittest

import numpy as np

from bank_anchored.r5_view_replay import matrices, public_wire, wire_sha


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


if __name__ == "__main__":
    unittest.main()
