"""Synthetic-only tests for the separate shuffled-label control protocol V2."""

import unittest
import tempfile
from pathlib import Path

import joblib
import numpy as np

from attack_eval import AttackView
from attack_controls_v2 import (SHUFFLE_ATT_SALT, SHUFFLE_INNER_SALT,
                                groupwise_shuffled_labels, run_control_probes_v2,
                                save_control_probe_v2_custody)
from safe_data_guard import sha_file


def _view(row_id, unit, sex):
    return AttackView(
        row_id=np.asarray(row_id, dtype=np.int64),
        unit=np.asarray(unit, dtype=np.int64),
        x=np.asarray(sex, dtype=np.float64)[:, None],
        components=("public_sex_proxy",),
        wire_components=("public_sex_proxy",),
        component_columns=(("public_sex_proxy", 0, 1),),
    ).validate()


class ControlV2Tests(unittest.TestCase):
    def test_group_vectors_are_preserved_under_distinct_fixed_salts(self):
        group = np.array([1, 1, 2, 2, 3, 3, 4, 5], dtype=np.int64)
        sex = np.array([0, 1, 1, 1, 0, 0, 0, 1], dtype=np.int64)
        self.assertNotEqual(SHUFFLE_ATT_SALT, SHUFFLE_INNER_SALT)
        self.assertEqual(SHUFFLE_INNER_SALT,
                         "pcrl-method-portfolio-v1|independent-inner-sex-null|20261008|v2")
        for salt in (SHUFFLE_ATT_SALT, SHUFFLE_INNER_SALT):
            shuffled = groupwise_shuffled_labels(sex, group, salt=salt)
            np.testing.assert_array_equal(shuffled,
                                          groupwise_shuffled_labels(sex, group, salt=salt))
            self.assertEqual(sorted(tuple(sex[group == g]) for g in (1, 2, 3)),
                             sorted(tuple(shuffled[group == g]) for g in (1, 2, 3)))
            self.assertEqual(int(sex.sum()), int(shuffled.sum()))

    def test_exact_sex_proxy_exposes_old_false_failure_and_new_null_passes(self):
        # 4,061 ATT groups and 2,235 INNER people mirror inherited role sizes.
        att_unit = np.concatenate((np.arange(4061), np.array([0, 1, 2])))
        att_sex = np.arange(att_unit.size, dtype=np.int64) % 2
        inner_unit = np.concatenate((np.arange(10000, 12234), np.array([10000])))
        inner_sex = np.arange(inner_unit.size, dtype=np.int64) % 2
        att = _view(np.arange(att_unit.size), att_unit, att_sex)
        inner = _view(np.arange(10000, 12235), inner_unit, inner_sex)
        result, artifacts = run_control_probes_v2(
            att, {"row_id": att.row_id, "unit": att.unit, "sex": att_sex},
            inner, {"row_id": inner.row_id, "unit": inner.unit, "sex": inner_sex},
            return_artifacts=True)
        self.assertGreaterEqual(result["shuffled_att_true_inner_auc"], 0.99)
        self.assertLessEqual(result["shuffled_att_independent_inner_auc"], 0.60)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["additional_reader_fits"], 2)
        self.assertEqual(inner_unit.size, 2235)
        self.assertEqual(np.unique(inner_unit).size, 2234)
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory) / ".private" / "probe"
            custody = save_control_probe_v2_custody(artifacts, private)
            self.assertEqual(custody["private_receipt_sha256"],
                             sha_file(private / "CUSTODY_RECEIPT.json"))
            with np.load(private / "probe_replay.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["inner_shuffled_sex"],
                                              artifacts["inner_shuffled_sex"])
                np.testing.assert_allclose(saved["shuffled_prediction_inner"],
                                           artifacts["shuffled_prediction_inner"])
            restored = joblib.load(private / "shuffled_model.joblib")
            with np.load(private / "probe_replay.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(restored.predict_proba(saved["inner_x"])[:, 1],
                                           saved["shuffled_prediction_inner"])
                planted = joblib.load(private / "planted_model.joblib")
                np.testing.assert_allclose(
                    planted.predict_proba(np.column_stack((saved["inner_x"],
                                                            saved["inner_true_sex"])))[:, 1],
                    saved["planted_prediction_inner"])
            with self.assertRaisesRegex(ValueError, "new private directory"):
                save_control_probe_v2_custody(artifacts, private)

    def test_inner_label_order_mismatch_is_refused_before_fit(self):
        att_unit = np.concatenate((np.arange(4061), np.array([0, 1, 2])))
        att_sex = np.arange(att_unit.size, dtype=np.int64) % 2
        inner_unit = np.concatenate((np.arange(10000, 12234), np.array([10000])))
        inner_sex = np.arange(inner_unit.size, dtype=np.int64) % 2
        att = _view(np.arange(att_unit.size), att_unit, att_sex)
        inner = _view(np.arange(10000, 12235), inner_unit, inner_sex)
        with self.assertRaisesRegex(ValueError, "ordered row/group identity mismatch"):
            run_control_probes_v2(
                att, {"row_id": att.row_id, "unit": att.unit, "sex": att_sex},
                inner, {"row_id": inner.row_id[::-1], "unit": inner.unit[::-1],
                        "sex": inner_sex[::-1]})

    def test_wrong_inner_group_count_is_refused_before_fit(self):
        att_unit = np.concatenate((np.arange(4061), np.array([0, 1, 2])))
        att_sex = np.arange(att_unit.size, dtype=np.int64) % 2
        inner_unit = np.arange(10000, 12235, dtype=np.int64)
        inner_sex = np.arange(inner_unit.size, dtype=np.int64) % 2
        att = _view(np.arange(att_unit.size), att_unit, att_sex)
        inner = _view(inner_unit, inner_unit, inner_sex)
        with self.assertRaisesRegex(ValueError, "INNER_SELECTION row/group count mismatch"):
            run_control_probes_v2(
                att, {"row_id": att.row_id, "unit": att.unit, "sex": att_sex},
                inner, {"row_id": inner.row_id, "unit": inner.unit,
                        "sex": inner_sex})


if __name__ == "__main__":
    unittest.main()
