"""Data-free R01 role-array binding and task-only label boundary."""

import unittest

import numpy as np

from bank import FixedBank, build_actions
from release import _bank_hash
from source_bridge import (ExpectedIdentity, action_map_sha256, bind_task_inputs,
                           attach_protected, sha_array)


def fixture():
    p1 = np.array([[.9, .1], [.6, .4], [.1, .9]], np.float64)
    p2 = np.array([[.8, .2], [.7, .3], [.1, .9]], np.float64)
    records = []
    bank_hashes, action_hashes = [], []
    for p in (p1, p2):
        classes = np.argmax(p, axis=1).astype(np.int64)
        record = {"probability": p, "token_class": classes,
                  "token_id": np.arange(3, dtype=np.int64),
                  "input_class": classes.copy(),
                  "fine_to_hcal_token": np.arange(3, dtype=np.int64),
                  "canonical_action_id": np.arange(3, dtype=np.int64)}
        bank = FixedBank.from_source(p, record["token_id"], classes)
        actions = build_actions(bank, input_class=record["input_class"],
                                fine_to_old_token=record["fine_to_hcal_token"])
        records.append(record)
        bank_hashes.append(_bank_hash(bank))
        action_hashes.append(action_map_sha256(actions))
    row = np.array([2, 4], np.int64)
    unit = np.array([1, 3], np.int64)
    cells = {"row_id": row, "unit": unit,
             "cell_1": np.array([0, 2], np.int64),
             "cell_2": np.array([0, 2], np.int64),
             "token_1": np.array([0, 2], np.int64),
             "token_2": np.array([0, 2], np.int64)}
    references = {"row_id": row, "unit": unit,
                  "U0_1": p1[[0, 2]], "U0_2": p2[[0, 2]],
                  "Ucal_1": np.array([[.95, .05], [.05, .95]], np.float64),
                  "Ucal_2": np.array([[.9, .1], [.05, .95]], np.float64),
                  "d_1": np.array([0, 1], np.int64),
                  "d_2": np.array([0, 1], np.int64)}
    income = {"row_id": row, "unit": unit,
              "income": np.array([0, 1], np.int64)}
    occupation = {"row_id": row, "unit": unit,
                  "occupation": np.array([0, 1], np.int64)}
    expected = ExpectedIdentity(
        source_binding_sha256="a" * 64,
        row_unit_pair_sha256=sha_array(np.column_stack((row, unit))),
        bank_sha256=tuple(bank_hashes), action_sha256=tuple(action_hashes))
    return records, cells, references, income, occupation, expected


class SourceBridgeTest(unittest.TestCase):
    def test_task_only_binding_and_both_reference_caps(self):
        records, cells, references, income, occupation, expected = fixture()
        task, policy, caps, receipt = bind_task_inputs(
            records[0], records[1], cells, references, income, occupation,
            source_binding_sha256="a" * 64, expected=expected)
        self.assertFalse(hasattr(task, "sex"))
        self.assertEqual(task.score1.shape, (2, 2))
        self.assertLess(receipt["references"]["r1_nll"]["Ucal"],
                        receipt["references"]["r1_nll"]["U0"])
        self.assertEqual(caps["r1_nll"], receipt["references"]["r1_nll"]["Ucal"] + .010)
        self.assertEqual(caps["r2_brier"], receipt["references"]["r2_brier"]["Ucal"] + .005)
        self.assertEqual(receipt["row_unit_pair_sha256"], expected.row_unit_pair_sha256)
        self.assertEqual(policy.actions1[2][0], policy.bank1.old_to_global[2])

    def test_task_path_refuses_sensitive_member_and_wrong_source_hash(self):
        records, cells, references, income, occupation, expected = fixture()
        income = {**income, "sex": np.array([0, 1], np.int64)}
        with self.assertRaises(ValueError):
            bind_task_inputs(records[0], records[1], cells, references,
                             income, occupation, source_binding_sha256="a" * 64,
                             expected=expected)
        income.pop("sex")
        with self.assertRaises(ValueError):
            bind_task_inputs(records[0], records[1], cells, references,
                             income, occupation, source_binding_sha256="b" * 64,
                             expected=expected)

    def test_row_order_and_bank_identity_must_match_pins(self):
        records, cells, references, income, occupation, expected = fixture()
        swapped = {**cells, "row_id": cells["row_id"][::-1]}
        with self.assertRaises(ValueError):
            bind_task_inputs(records[0], records[1], swapped, references,
                             income, occupation, source_binding_sha256="a" * 64,
                             expected=expected)
        bad_bank = {**records[0], "probability": records[0]["probability"].copy()}
        bad_bank["probability"][0] = [.85, .15]
        with self.assertRaises(ValueError):
            bind_task_inputs(bad_bank, records[1], cells, references,
                             income, occupation, source_binding_sha256="a" * 64,
                             expected=expected)

    def test_protected_sex_is_attached_only_after_task_binding(self):
        records, cells, references, income, occupation, expected = fixture()
        task, _, _, _ = bind_task_inputs(
            records[0], records[1], cells, references, income, occupation,
            source_binding_sha256="a" * 64, expected=expected)
        sex = {"row_id": cells["row_id"], "unit": cells["unit"],
               "sex": np.array([0, 1], np.int64)}
        protected = attach_protected(task, sex, expected_row_unit_sha256=
                                     expected.row_unit_pair_sha256)
        self.assertEqual(protected.sex.tolist(), [0, 1])
        with self.assertRaises(ValueError):
            attach_protected(task, {**sex, "unit": sex["unit"][::-1]},
                             expected_row_unit_sha256=expected.row_unit_pair_sha256)


if __name__ == "__main__":
    unittest.main()
