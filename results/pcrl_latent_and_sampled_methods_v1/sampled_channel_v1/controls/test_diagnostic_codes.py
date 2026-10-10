"""Identity and class-only use frozen canonical code IDs without labels."""

import unittest

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import (
    FixedBank, build_actions, class_medoid,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.diagnostic_codes import (
    assert_group_coherent, select_codes,
)


class DiagnosticCodesTest(unittest.TestCase):
    def setUp(self):
        raw = np.asarray([[.8, .2], [.7, .3], [.2, .8], [.3, .7]],
                         dtype=np.float64)
        self.bank = FixedBank.from_source(raw, np.arange(4, dtype=np.int64),
                                          np.asarray([0, 0, 1, 1], np.int64))
        self.classes = np.asarray([0, 0, 1, 1], np.int64)
        self.actions = build_actions(self.bank, input_class=self.classes,
                                     fine_to_old_token=np.arange(4, dtype=np.int64))
        self.cells = np.asarray([0, 1, 2, 3, 0], dtype=np.int64)

    def test_identity_is_frozen_first_action(self):
        codes = select_codes(self.bank, self.actions, self.cells,
                             arm_id="A_IDENTITY")
        np.testing.assert_array_equal(codes,
                                      [self.actions[int(c)][0] for c in self.cells])

    def test_class_only_is_one_existing_bank_vector_per_decision(self):
        codes = select_codes(self.bank, self.actions, self.cells,
                             arm_id="A_CLASS_ONLY")
        identity = select_codes(self.bank, self.actions, self.cells,
                                arm_id="A_IDENTITY")
        for label in (0, 1):
            within = self.bank.decision[identity] == label
            self.assertEqual(set(codes[within]), {class_medoid(self.bank, label)})
        np.testing.assert_array_equal(self.bank.decision[codes],
                                      self.bank.decision[identity])

    def test_class_only_rejects_unsealed_action_row(self):
        actions = list(self.actions)
        actions[0] = (actions[0][0],)
        with self.assertRaisesRegex(ValueError, "absent"):
            select_codes(self.bank, tuple(actions), self.cells,
                         arm_id="A_CLASS_ONLY")

    def test_invalid_cells_refused(self):
        with self.assertRaisesRegex(ValueError, "fine cells"):
            select_codes(self.bank, self.actions, np.asarray([999], np.int64),
                         arm_id="A_IDENTITY")

    def test_duplicate_unit_must_receive_identical_pair(self):
        assert_group_coherent(np.asarray([1, 1, 2], np.int64),
                              np.asarray([3, 3, 4], np.int64),
                              np.asarray([5, 5, 6], np.int64))
        with self.assertRaisesRegex(ValueError, "duplicate unit"):
            assert_group_coherent(np.asarray([1, 1], np.int64),
                                  np.asarray([3, 4], np.int64),
                                  np.asarray([5, 5], np.int64))


if __name__ == "__main__":
    unittest.main()
