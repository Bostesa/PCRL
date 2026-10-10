"""Data-free regression for the frozen INNER-only attack admission rule."""

import unittest

from bank_anchored.r3_attack_gate_v2 import correct_attack_gate, attack_nomination_rows


def old_combined(arm, fit_passed, inner_passed):
    return {
        "schema": "pcrl-bank-r3-arm-point-eligibility-v1",
        "arm_id": arm,
        "source_seal_sha256": "a" * 64,
        "fit_clauses": {"passed": fit_passed, "total": 12},
        "inner_clauses": {"passed": inner_passed, "total": 24},
        "fit_bank_proxy_pass": fit_passed == 12,
        "inner_original_caps_pass": inner_passed == 24,
        "all_two_seed_task_reference_point_caps_pass": fit_passed == 12 and inner_passed == 24,
        "source_receipts": {
            key: {"receipt_sha256": "b" * 64, "csv_sha256": "c" * 64,
                  "unit_id": "R3_" + arm + "_" + key}
            for key in ("s0_fit", "s0_inner", "s1_fit", "s1_inner")},
    }


class AttackGateV2Tests(unittest.TestCase):
    def test_j_inner_pass_admits_attacks_despite_fit_proxy_failure(self):
        gate = correct_attack_gate(old_combined("J", 10, 24), "d" * 64)
        self.assertEqual(gate["schema"], "pcrl-bank-r3-inner-attack-eligibility-v2")
        self.assertTrue(gate["all_two_seed_task_reference_point_caps_pass"])
        self.assertTrue(gate["inner_original_caps_pass"])
        self.assertFalse(gate["fit_bank_proxy_pass"])
        self.assertEqual(gate["fit_clauses"], {"passed": 10, "total": 12})
        self.assertEqual(gate["supersedes_combined_gate_sha256"], "d" * 64)
        rows = attack_nomination_rows({"J": gate, "S21": correct_attack_gate(
            old_combined("S21", 11, 22), "e" * 64)})
        self.assertEqual(next(row for row in rows if row["arm_id"] == "J")
                         ["nomination"], "J_NOMINEE")
        self.assertFalse(next(row for row in rows if row["arm_id"] == "S21")
                         ["fresh_attack_eligible"])

    def test_inner_failure_blocks_attack_even_if_fit_passes(self):
        gate = correct_attack_gate(old_combined("S21", 12, 22), "d" * 64)
        self.assertFalse(gate["all_two_seed_task_reference_point_caps_pass"])
        self.assertTrue(gate["fit_bank_proxy_pass"])
        self.assertEqual(gate["inner_clauses"], {"passed": 22, "total": 24})

    def test_inconsistent_old_gate_is_rejected(self):
        wrong = old_combined("J", 10, 24)
        wrong["all_two_seed_task_reference_point_caps_pass"] = True
        with self.assertRaises(PermissionError):
            correct_attack_gate(wrong, "d" * 64)


if __name__ == "__main__":
    unittest.main()
