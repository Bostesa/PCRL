"""Synthetic exact-ID specialist attachment through S2's actual attack API."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import attack_eval
from attack_views import AlignedDViews
from d_cell_slate import fit_d_cell_readers
from s2_attack_adapter import as_s2_attack_view
from test_s2_attack_adapter import code_release


class DCellS2IntegrationTests(unittest.TestCase):
    def test_private_artifacts_attach_to_one_frozen_s2_envelope(self):
        n = 4061  # Actual S2 split requires this many distinct synthetic groups.
        row = np.arange(n, dtype=np.int64)
        code0 = np.tile([0, 0, 1, 1], (n + 3) // 4)[:n]
        code1 = np.tile([0, 1, 0, 1], (n + 3) // 4)[:n]
        sex = (code0 ^ code1).astype(np.int64)
        aligned = AlignedDViews(row, row, row, row,
                                code_release(0, code0), code_release(1, code1), "a" * 64)
        att = as_s2_attack_view(aligned, "pair", attack_eval.AttackView)
        inner = as_s2_attack_view(aligned, "pair", attack_eval.AttackView)
        sex_record = {"row_id": row, "unit": row, "sex": sex}
        split = attack_eval.attack_fit_tune_split(row)
        baseline_tune = np.full(len(split["tune"]), 0.5)
        baseline_inner = np.full(n, 0.5)
        validation = attack_eval.oriented_auc(sex[split["tune"]], baseline_tune)
        validation["selected_orientation"] = "positive"
        fitted = {
            "att_row_id_sha256": attack_eval.sha_arr(row),
            "att_row_unit_pair_sha256": attack_eval.sha_arr(np.column_stack((row, row))),
            "inner_row_id_sha256": attack_eval.sha_arr(row),
            "inner_row_unit_pair_sha256": attack_eval.sha_arr(np.column_stack((row, row))),
            "att_view_content_sha256": att.content_hashes(),
            "inner_view_content_sha256": inner.content_hashes(),
            "split": {"tune_group_id_sha256": split["tune_group_id_sha256"]},
            "reader_order": ["BASE"], "predictions_tune": {"BASE": baseline_tune},
            "predictions_inner": {"BASE": baseline_inner},
            "validation": {"BASE": validation},
            "validation_winner": "BASE", "validation_winner_orientation": "positive",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".private"
            root.mkdir(mode=0o700)
            readers = fit_d_cell_readers(aligned, aligned, "pair", att, inner,
                                         sex_record, attack_eval,
                                         model_dir=root / "models", private_root=root)
            attached = attack_eval.add_external_fixed_readers(
                fitted, att, sex_record, inner, readers)
            self.assertEqual(len(attached["external_readers"]), 3)
            self.assertEqual(len(attached["reader_order"]), 4)
            self.assertEqual(attached["validation_winner"], "D_CELL_A0.5")


if __name__ == "__main__":
    unittest.main()
