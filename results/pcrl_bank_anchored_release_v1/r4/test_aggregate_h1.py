"""Data-free checks of public H1 attack receipt aggregation."""

import unittest

from .aggregate_h1 import rows_for_receipt


class H1AggregateTests(unittest.TestCase):
    def fixture(self):
        entry = {"unit_id": "R4_ATTACK_B0_S0_R0_H1", "arm_id": "B0",
                 "source_seed": 0, "realization": 0, "spec_sha256": "a" * 64}
        score = {"primary_envelope_auc": 0.7,
                 "validation_winner": "CELL_A1",
                 "validation_winner_inner_auc": 0.69,
                 "readers": {"CELL_A1": {"fixed_alias_max_auc": 0.7}}}
        receipt = {"schema": "pcrl-bank-r4-witness-fresh-attack-v1",
                   "status": "COMPLETE_PRIVATE_DEVELOPMENT_ONLY",
                   "unit_id": entry["unit_id"], "arm_id": "B0",
                   "source_seed": 0, "realization": 0,
                   "attacker_seed": 100000,
                   "unit_spec_sha256": "a" * 64,
                   "stage_lock_sha256": "b" * 64,
                   "attack": {"schema": "pcrl-next-frozen-attack-unit-v1",
                              "freeze_receipt_sha256": "c" * 64,
                              "scores": {v: dict(score) for v in ("r1", "r2", "pair")},
                              "controls": {v: {"scores": {"gates": {
                                  "null_exact": True, "planted": True}}}
                                  for v in ("r1", "r2", "pair")}}}
        return receipt, entry

    def test_records_all_views_and_control_status(self):
        receipt, entry = self.fixture()
        rows, unit = rows_for_receipt(receipt, entry, "b" * 64, "d" * 64)
        self.assertEqual([row["view"] for row in rows], ["r1", "r2", "pair"])
        self.assertEqual(unit["auc_by_view"], {v: 0.7 for v in ("r1", "r2", "pair")})
        self.assertTrue(all(row["all_control_gates_pass"] for row in rows))
        receipt["attack"]["controls"]["pair"]["scores"]["gates"]["planted"] = False
        rows, _ = rows_for_receipt(receipt, entry, "b" * 64, "d" * 64)
        self.assertFalse(rows[2]["all_control_gates_pass"])

    def test_rejects_identity_or_view_change(self):
        receipt, entry = self.fixture()
        with self.assertRaises(PermissionError):
            rows_for_receipt(receipt, entry, "0" * 64, "d" * 64)
        del receipt["attack"]["scores"]["pair"]
        with self.assertRaises(PermissionError):
            rows_for_receipt(receipt, entry, "b" * 64, "d" * 64)


if __name__ == "__main__":
    unittest.main()
