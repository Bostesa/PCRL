"""Data-free selected-control accounting with one ineligible arm."""

from __future__ import annotations

import unittest

from bank_anchored import r5_replay_selection_v2 as selection


def synthetic_family() -> list[dict]:
    rows = []
    for arm in selection.ESTIMABLE_ARMS:
        for seed in (0, 1):
            for draw in ((0, 1, 2) if arm in
                         ("T", "P", "L", "J", "S12") else (0,)):
                rows.append({"arm_id": arm, "source_seed": seed,
                             "realization": draw,
                             "auc_by_view": {view: .8 for view in selection.VIEWS}})
    return rows


class SelectionConditionalNullTest(unittest.TestCase):
    def test_all_38_releases_with_s21_explicitly_null(self) -> None:
        rows = synthetic_family()
        self.assertEqual(len(rows), 38)
        means = selection.equal_seed_mean(rows)
        self.assertEqual(set(means), set(selection.ARM_ORDER))
        self.assertEqual(means["S21"]["status"], "CONDITIONAL_NULL")
        self.assertIsNone(means["S21"]["mean"])
        for arm in selection.ESTIMABLE_ARMS:
            self.assertAlmostEqual(means[arm]["mean"]["pair"], .8)

    def test_another_missing_arm_is_not_accepted(self) -> None:
        rows = [row for row in synthetic_family() if row["arm_id"] != "P"]
        with self.assertRaisesRegex(ValueError, "missing arm"):
            selection.equal_seed_mean(rows)


if __name__ == "__main__":
    unittest.main()
