"""Data-free checks for the guarded fixed diagnostic release."""

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
import unittest
from unittest.mock import patch

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls import (
    run_diagnostic_release as worker,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import (
    FixedBank, build_actions,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.reference_materialize import (
    materialize_reference_role,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.release import _bank_hash
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.source_bridge import (
    action_map_sha256, sha_array,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.diagnostic_codes import (
    select_codes,
)


class DiagnosticWorkerTest(unittest.TestCase):
    def test_no_r03_slot_blocks_before_role_access(self):
        args = Namespace(unit_id="A-DIAG-RELEASE-A_IDENTITY-S0-ATT",
                         original_lock="absent", a_amendment="absent",
                         a_amendment_sha256="0" * 64, private_output="absent")
        with patch.object(worker, "assert_active_from_env",
                          side_effect=RuntimeError("no active slot")), \
             patch.object(worker, "TStarSourceStore") as source, \
             patch.object(worker, "RoleOnlyStore") as roles:
            with self.assertRaisesRegex(RuntimeError, "no active slot"):
                worker.run(args)
        source.assert_not_called()
        roles.assert_not_called()

    def test_output_custody_is_checked_without_creating_directory(self):
        with TemporaryDirectory(dir=Path(gettempdir()).resolve()) as directory:
            root = Path(directory)
            private = root / ".private"
            private.mkdir(mode=0o700)
            target = private / "diag"
            self.assertEqual(worker._private_output(target),
                             target.resolve(strict=False))
            self.assertFalse(target.exists())
            private.chmod(0o755)
            with self.assertRaisesRegex(PermissionError, "owner-only"):
                worker._private_output(target)
            private.chmod(0o700)
            link = root / "linked"
            link.symlink_to(private, target_is_directory=True)
            with self.assertRaisesRegex(PermissionError, "owner-only"):
                worker._private_output(link / "diag")

    def test_fixed_bank_requires_exact_full_bank_and_action_hashes(self):
        p = np.asarray([[.8, .2], [.7, .3], [.2, .8], [.3, .7]],
                       dtype=np.float64)
        classes = np.asarray([0, 0, 1, 1], np.int64)
        ids = np.arange(4, dtype=np.int64)
        record = {"probability": p, "token_id": ids,
                  "token_class": classes, "input_class": classes,
                  "fine_to_hcal_token": ids}
        bank = FixedBank.from_source(p, ids, classes)
        actions = build_actions(bank, input_class=classes,
                                fine_to_old_token=ids)
        actual_bank, actual_actions = worker._fixed_bank(
            record, expected_bank=_bank_hash(bank),
            expected_action=action_map_sha256(actions))
        np.testing.assert_array_equal(actual_bank.probability, bank.probability)
        self.assertEqual(actual_actions, actions)
        with self.assertRaisesRegex(PermissionError, "bank/action"):
            worker._fixed_bank(record, expected_bank="0" * 64,
                               expected_action=action_map_sha256(actions))

    def test_actual_reference_package_uses_full_canonical_codes(self):
        income = np.asarray([[.8, .2], [.7, .3], [.2, .8], [.3, .7]],
                            dtype=np.float64)
        occupation = np.full((6, 6), .06, dtype=np.float64)
        np.fill_diagonal(occupation, .70)
        banks = [FixedBank.from_source(income, np.arange(4, dtype=np.int64),
                                       np.asarray([0, 0, 1, 1], np.int64)),
                 FixedBank.from_source(occupation, np.arange(6, dtype=np.int64),
                                       np.arange(6, dtype=np.int64))]
        actions = [build_actions(banks[0],
                                 input_class=np.asarray([0, 0, 1, 1], np.int64),
                                 fine_to_old_token=np.arange(4, dtype=np.int64)),
                   build_actions(banks[1], input_class=np.arange(6, dtype=np.int64),
                                 fine_to_old_token=np.arange(6, dtype=np.int64))]
        cells1 = np.asarray([0, 1, 2], np.int64)
        cells2 = np.asarray([0, 1, 2], np.int64)
        row_id = np.asarray([10, 11, 12], np.int64)
        unit = np.asarray([20, 21, 22], np.int64)
        with TemporaryDirectory(dir=Path(gettempdir()).resolve()) as directory:
            private = Path(directory) / ".private"
            private.mkdir(mode=0o700)
            for arm in ("A_IDENTITY", "A_CLASS_ONLY"):
                codes = [select_codes(banks[0], actions[0], cells1, arm_id=arm),
                         select_codes(banks[1], actions[1], cells2, arm_id=arm)]
                receipt = materialize_reference_role(
                    row_id=row_id, unit=unit,
                    prob1=banks[0].probability[codes[0]],
                    prob2=banks[1].probability[codes[1]],
                    decision1=banks[0].decision[codes[0]],
                    decision2=banks[1].decision[codes[1]],
                    code1=codes[0], code2=codes[1],
                    arm_id=arm, seed=0, role="INNER_SELECTION",
                    release_id=f"{arm}-s0-v1", source_binding_sha256="a" * 64,
                    source_identity={"role_source_file_sha256": "b" * 64,
                                     "reference_file_sha256": "c" * 64,
                                     "bank_receipt_sha256": "d" * 64,
                                     "source_audit_sha256": "e" * 64},
                    interface_kind="finite",
                    bank_sha256_by_recipient={"r1": sha_array(banks[0].probability),
                                              "r2": sha_array(banks[1].probability)},
                    expected_row_unit_sha256=sha_array(np.column_stack((row_id, unit))),
                    output_dir=private / arm,
                    bank1=banks[0].probability, bank2=banks[1].probability)
                self.assertEqual(receipt["status"], "FROZEN_DETERMINISTIC")
                with np.load(private / arm / "reference_release.npz") as package:
                    np.testing.assert_array_equal(package["code_r1"], codes[0])
                    np.testing.assert_array_equal(package["code_r2"], codes[1])
                    self.assertEqual(package["prob_r1"].shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
