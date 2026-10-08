"""Synthetic-only checks of the frozen attack contract; no Adult arrays or fits."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from attack_eval import (AttackView, add_ignore_recipient_aliases, attack_fit_tune_split,
                         bootstrap_envelope, fit_slate, save_fitted_slate,
                         oriented_auc, score_slate)
from safe_data_guard import sha_arr


def toy_view(n: int = 8) -> AttackView:
    return AttackView(row_id=np.arange(n, dtype=np.int64), unit=np.arange(n, dtype=np.int64),
                      x=np.column_stack((np.arange(n), np.ones(n))),
                      components=("probability", "code_id"),
                      wire_components=("probability", "code_id"),
                      component_columns=(("probability", 0, 1), ("code_id", 1, 2)))


class AttackEvalTest(unittest.TestCase):
    def test_fixed_group_split_and_view_schema(self):
        units = np.arange(4061, dtype=np.int64)
        a = attack_fit_tune_split(units)
        b = attack_fit_tune_split(units)
        self.assertEqual(a["tune_groups"], 1015)
        self.assertEqual(a["fit_groups"], 3046)
        np.testing.assert_array_equal(a["tune"], b["tune"])
        self.assertEqual(len(np.intersect1d(a["fit"], a["tune"])), 0)
        toy_view().validate()
        bad = toy_view()
        object.__setattr__(bad, "wire_components", ("probability",))
        with self.assertRaisesRegex(ValueError, "wire schema"):
            bad.validate()
        bad = toy_view()
        object.__setattr__(bad, "component_columns", (("probability", 0, 1),))
        with self.assertRaisesRegex(ValueError, "omitted"):
            bad.validate()

    def test_orientation_aliases_and_group_bootstrap_envelope(self):
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        inverted = np.array([0.9, 0.8, 0.2, 0.1])
        right = 1 - inverted
        score = oriented_auc(y, inverted)
        self.assertEqual(score["positive_alias_auc"], 0.0)
        self.assertEqual(score["reverse_alias_auc"], 1.0)
        self.assertEqual(score["fixed_alias_max_auc"], 1.0)
        self.assertEqual(bootstrap_envelope({"reader_a": inverted, "reader_b": right}, y,
                                            np.arange(4, dtype=np.int64)), 1.0)

    def test_ordered_inner_sex_record_required(self):
        view = toy_view()
        p = np.linspace(0.1, 0.9, 8)
        fitted = {"predictions_inner": {"LR_C1": p},
                  "inner_row_id_sha256": sha_arr(view.row_id),
                  "inner_row_unit_pair_sha256": sha_arr(np.column_stack((view.row_id, view.unit))),
                  "validation_winner": "LR_C1", "validation_winner_orientation": "positive"}
        label = {"row_id": view.row_id.copy(), "unit": view.unit.copy(),
                 "sex": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)}
        self.assertEqual(score_slate(fitted, label)["primary_envelope_auc"], 1.0)
        label["unit"] = label["unit"][::-1]
        with self.assertRaisesRegex(ValueError, "pairing mismatch"):
            score_slate(fitted, label)

    def test_fit_refuses_label_order_or_component_permutation_before_training(self):
        n = 4061
        att = toy_view(n)
        inner = toy_view(n)
        label = {"row_id": att.row_id[::-1].copy(), "unit": att.unit[::-1].copy(),
                 "sex": (np.arange(n) % 2).astype(np.int64)}
        with self.assertRaisesRegex(ValueError, "ordered row/group identity mismatch"):
            fit_slate(att, label, inner, required_readers=("LR_C1", "MLP_64x32", "HGB_15x200"))
        label["row_id"] = att.row_id.copy()
        label["unit"] = att.unit.copy()
        object.__setattr__(inner, "component_columns", (("code_id", 0, 1), ("probability", 1, 2)))
        with self.assertRaisesRegex(ValueError, "component schemas differ"):
            fit_slate(att, label, inner, required_readers=("LR_C1", "MLP_64x32", "HGB_15x200"))

    def test_required_special_readers_cannot_disappear(self):
        n = 4061
        att, inner = toy_view(n), toy_view(n)
        label = {"row_id": att.row_id.copy(), "unit": att.unit.copy(),
                 "sex": (np.arange(n) % 2).astype(np.int64)}
        core = ("LR_C1", "MLP_64x32", "HGB_15x200")
        with self.assertRaisesRegex(ValueError, "required defense-aware"):
            fit_slate(att, label, inner, required_readers=core + ("DA_MLP_64x32",))
        with self.assertRaisesRegex(ValueError, "required categorical"):
            fit_slate(att, label, inner, required_readers=core + ("CELL_A1",))
        with self.assertRaisesRegex(ValueError, "required metadata"):
            fit_slate(att, label, inner, required_readers=core + ("META_LR_C1",))
        with self.assertRaisesRegex(ValueError, "required code-bit"):
            fit_slate(att, label, inner, required_readers=core + ("CODE_BIT_LR_C1",))

    def test_ignore_recipient_aliases_are_zero_fit_and_identity_bound(self):
        base = {"att_row_id_sha256": "a", "att_row_unit_pair_sha256": "b",
                "inner_row_id_sha256": "c", "inner_row_unit_pair_sha256": "d",
                "split": {"tune_group_id_sha256": "e"},
                "reader_order": ["LR_C1"], "predictions_inner": {"LR_C1": np.array([0.2, 0.8])},
                "predictions_tune": {"LR_C1": np.array([0.3, 0.7])},
                "validation": {"LR_C1": {"fixed_alias_max_auc": 0.7, "selected_orientation": "positive"}},
                "validation_winner": "LR_C1", "validation_winner_orientation": "positive"}
        merged = add_ignore_recipient_aliases(base, base, base)
        self.assertEqual(len(merged["ignore_recipient_aliases"]), 2)
        self.assertEqual(len(merged["predictions_inner"]), 3)
        self.assertFalse(any(a["new_fit"] for a in merged["ignore_recipient_aliases"]))
        bad = {**base, "inner_row_unit_pair_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "different ordered people"):
            add_ignore_recipient_aliases(base, bad, base)

    def test_private_reader_custody_is_write_once_and_replayable(self):
        fitted = {"models": {"LR_C1": {"weights": [1.0]}},
                  "predictions_tune": {"LR_C1": np.array([0.2, 0.8])},
                  "predictions_inner": {"LR_C1": np.array([0.3, 0.7])},
                  "att_row_id_sha256": "a", "att_row_unit_pair_sha256": "b",
                  "inner_row_id_sha256": "c", "inner_row_unit_pair_sha256": "d",
                  "split": {"tune_group_id_sha256": "e"}, "reader_order": ["LR_C1"],
                  "validation_winner": "LR_C1", "validation_winner_orientation": "positive",
                  "validation": {"LR_C1": {"fixed_alias_max_auc": 0.7}}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".private" / "readers"
            view = toy_view(2)
            fitted["att_row_id_sha256"] = fitted["inner_row_id_sha256"] = sha_arr(view.row_id)
            fitted["att_row_unit_pair_sha256"] = fitted["inner_row_unit_pair_sha256"] = \
                sha_arr(np.column_stack((view.row_id, view.unit)))
            fitted["att_view_content_sha256"] = fitted["inner_view_content_sha256"] = view.content_hashes()
            out = save_fitted_slate(fitted, path, att_view=view, inner_view=view)
            self.assertEqual(out["model_files"], 1)
            self.assertTrue((path / "model__LR_C1.joblib").is_file())
            with self.assertRaisesRegex(ValueError, "new private"):
                save_fitted_slate(fitted, path, att_view=view, inner_view=view)
            changed = toy_view(2)
            object.__setattr__(changed, "x", changed.x + 1)
            with self.assertRaisesRegex(ValueError, "values differ"):
                save_fitted_slate(fitted, Path(temp) / ".private" / "different",
                                  att_view=view, inner_view=changed)


if __name__ == "__main__":
    unittest.main()
