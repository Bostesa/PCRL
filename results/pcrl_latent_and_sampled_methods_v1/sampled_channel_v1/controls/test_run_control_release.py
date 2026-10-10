"""Control release wrapper must fail before common materialization without a slot."""

from argparse import Namespace
from unittest import TestCase
from unittest.mock import patch

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls import run_control_release as worker


class ControlReleaseTest(TestCase):
    def test_no_queue_slot_never_enters_common_release(self):
        args = Namespace(unit_id="A_LOCAL_S0_ATT_D0", arm_id="A_LOCAL")
        with patch.object(worker, "assert_active_from_env",
                          side_effect=RuntimeError("no slot")), \
             patch.object(worker, "run_common_release") as common:
            with self.assertRaisesRegex(RuntimeError, "no slot"):
                worker.run(args)
        common.assert_not_called()

    def test_noncontrol_arm_refused_before_source_or_role_reads(self):
        args = Namespace(unit_id="A_JOINT_STOCH_S0_ATT_D0",
                         arm_id="A_JOINT_STOCH")
        with patch.object(worker, "assert_active_from_env", return_value={}), \
             patch.object(worker, "run_common_release") as common:
            with self.assertRaisesRegex(PermissionError, "only R13"):
                worker.run(args)
        common.assert_not_called()
