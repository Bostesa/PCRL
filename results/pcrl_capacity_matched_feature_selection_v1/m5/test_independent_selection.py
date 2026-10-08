"""Data-free selection replay cases with no M2 or M3 imports."""

from __future__ import annotations

import copy
import unittest

from independent_selection import replay_fit_selection


IDS = ("A", "B", "C")


def fixture() -> tuple[dict, dict]:
    references = {
        purpose: {name: {"accuracy": 0.90, "nll": 0.20, "brier": 0.30}
                  for name in ("U0", "Ucal", "F")}
        for purpose in ("income", "occupation_group")
    }
    nll = {"income": (0.201, 0.203, 0.205),
           "occupation_group": (0.204, 0.201, 0.206)}
    auc = {"income": (0.70, 0.65, 0.68),
           "occupation_group": (0.69, 0.71, 0.64)}
    masks = {}
    for purpose in ("income", "occupation_group"):
        masks[purpose] = []
        for i, mid in enumerate(IDS):
            masks[purpose].append({
                "mask_id": mid, "mask_sha256": mid.lower() * 64,
                "head_id": f"HEAD-{purpose}-{mid}",
                "task_validation_accuracy": 0.9,
                "task_validation_nll": nll[purpose][i],
                "task_validation_brier": 0.3,
                "local_search_auc": auc[purpose][i],
                "utility_feasible": True,
                "utility_margins_by_ref_metric": {
                    ref: {"accuracy": 0.01, "nll": 0.21 - nll[purpose][i],
                          "brier": 0.005}
                    for ref in ("U0", "Ucal", "F")
                },
            })
    diagonal_auc = {"A": 0.65, "B": 0.62, "C": 0.58}
    pairs = []
    for a in IDS:
        for b in IDS:
            ai, bi = IDS.index(a), IDS.index(b)
            same = a == b
            pairs.append({
                "pair_id": f"{a}__{b}", "income_mask_id": a,
                "occupation_mask_id": b,
                "income_mask_sha256": a.lower() * 64,
                "occupation_mask_sha256": b.lower() * 64,
                "union_group_count": 8 if same else 9,
                "union_encoded_columns": {"A": 42, "B": 66, "C": 75}[a]
                if same else 83,
                "union_sha256": (a.lower() if same else "f") * 64,
                "coalition_search_auc": diagonal_auc[a] if same else 0.70,
                "income_local_search_auc": auc["income"][ai],
                "occupation_local_search_auc": auc["occupation_group"][bi],
                "task_loss": nll["income"][ai] + nll["occupation_group"][bi],
                "pair_feasible": True,
            })
    return {"critic_input_kind": "RAW_UNION_ONLY", "income_masks": masks["income"],
            "occupation_masks": masks["occupation_group"], "pairs": pairs}, references


class IndependentSelectionTests(unittest.TestCase):
    def test_replays_joint_and_all_matched_rules_without_excluding_alias(self) -> None:
        table, refs = fixture()
        out = replay_fit_selection(table, refs)
        self.assertEqual(out["selected_pair_ids"], {
            "J": "C__C", "T": "A__B", "I": "B__C", "S12": "B__B",
            "S21": "C__C", "SHARED": "C__C", "T_U": "C__C",
            "I_U": "C__C", "S12_U": "C__C", "S21_U": "C__C",
        })
        self.assertEqual(out["union_quota"], (8, 75))
        self.assertIn("S21", out["same_pair_as_joint"])
        self.assertIn("T_U", out["same_pair_as_joint"])

    def test_refuses_false_utility_flag_and_union_cache_fork(self) -> None:
        table, refs = fixture()
        changed = copy.deepcopy(table)
        changed["income_masks"][0]["utility_feasible"] = False
        with self.assertRaisesRegex(ValueError, "utility"):
            replay_fit_selection(changed, refs)
        changed = copy.deepcopy(table)
        changed["pairs"][1]["coalition_search_auc"] = 0.51
        with self.assertRaisesRegex(ValueError, "union"):
            replay_fit_selection(changed, refs)
        changed = copy.deepcopy(table)
        changed["critic_input_kind"] = "COMPLETE_PAIR_VIEW"
        with self.assertRaisesRegex(ValueError, "RAW_UNION_ONLY"):
            replay_fit_selection(changed, refs)


if __name__ == "__main__":
    unittest.main()
