import unittest

from bank_anchored.r5_prepare_attack import _source_index_paths


class EffectiveAttackSourceTests(unittest.TestCase):
    def test_h1_non_pstar_uses_original_38_unit_index(self):
        worker, queue, _, _, count = _source_index_paths("R4_ATTACK_J_S1_R2_H1")
        self.assertEqual(worker.name, "ATTACK_UNIT_INDEX_H1.json")
        self.assertEqual(queue.name, "ATTACK_QUEUE_INDEX_H1.json")
        self.assertEqual(count, 38)

    def test_pstar_uses_only_h2_successor_index(self):
        worker, queue, _, _, count = _source_index_paths(
            "R4_ATTACK_P_STAR_S0_R0_H2")
        self.assertEqual(worker.name, "PSTAR_H2_UNIT_INDEX.json")
        self.assertEqual(queue.name, "PSTAR_H2_QUEUE_INDEX.json")
        self.assertEqual(count, 2)
        with self.assertRaises(PermissionError):
            _source_index_paths("R4_ATTACK_P_STAR_S0_R0_H1")

    def test_unregistered_successor_is_rejected(self):
        with self.assertRaises(PermissionError):
            _source_index_paths("R4_ATTACK_J_S0_R0_H2")


if __name__ == "__main__":
    unittest.main()
