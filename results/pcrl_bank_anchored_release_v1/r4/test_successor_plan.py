"""Data-free regression for immutable H1 attack-unit succession."""

import unittest

from . import successor_plan


class SuccessorPlanTests(unittest.TestCase):
    def test_exact_successor_ids_preserve_scientific_units(self):
        units = [{"unit_id": "R4_ATTACK_J_S0_R0", "arm_id": "J",
                  "source_seed": 0, "realization": 0,
                  "spec_sha256": "a" * 64, "planned_fits": 30},
                 {"unit_id": "R4_ATTACK_J_S0_R1", "arm_id": "J",
                  "source_seed": 0, "realization": 1,
                  "spec_sha256": "b" * 64, "planned_fits": 30}]
        attack = {"schema": "pcrl-bank-r4-attack-unit-index-v2",
                  "campaign_seal_sha256": "c" * 64,
                  "unit_count": 2, "planned_fits": 60, "units": units,
                  "stage_lock_units": {
                      item["unit_id"]: {"spec_sha256": item["spec_sha256"],
                                        "arm_id": "J", "source_seed": 0,
                                        "realization": item["realization"],
                                        "point_eligible": True}
                      for item in units}}
        queue = {"schema": "pcrl-bank-r4-queue-unit-index-v2",
                 "attack_unit_index_sha256": "d" * 64,
                 "stage_lock_sha256": "e" * 64, "unit_count": 2,
                 "units": [{"unit_id": item["unit_id"],
                            "attack_spec_sha256": item["spec_sha256"],
                            "queue_spec_sha256": "f" * 64}
                           for item in units]}
        lock = {"schema": "pcrl-bank-attack-stage-lock-v1",
                "campaign_seal_sha256": "c" * 64,
                "units": attack["stage_lock_units"]}
        plan = successor_plan.build_plan(
            attack, queue, lock, attack_sha="d" * 64,
            queue_sha="g" * 64, lock_sha="e" * 64)
        self.assertEqual(plan["unit_count"], 2)
        self.assertEqual([u["successor_unit_id"] for u in plan["units"]],
                         ["R4_ATTACK_J_S0_R0_H1", "R4_ATTACK_J_S0_R1_H1"])
        self.assertEqual([u["attacker_seed"] for u in plan["units"]],
                         [100000, 100001])
        queue["units"][0]["attack_spec_sha256"] = "0" * 64
        with self.assertRaises(PermissionError):
            successor_plan.build_plan(
                attack, queue, lock, attack_sha="d" * 64,
                queue_sha="g" * 64, lock_sha="e" * 64)


if __name__ == "__main__":
    unittest.main()
