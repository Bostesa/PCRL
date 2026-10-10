"""Control release wrapper must fail before common materialization without a slot."""

from argparse import Namespace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
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

    def test_executed_wrapper_must_be_campaign_pinned_source(self):
        with TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            campaign.mkdir()
            lock = campaign / "CAMPAIGN_LOCK.json"
            lock.write_text("{}")
            amendment = campaign / "A_AMENDMENT.json"
            amendment.write_text(json.dumps({"implementation_source_sha256": {
                worker.RELATIVE_SOURCE: "a" * 64}}))
            pinned_source = campaign / worker.RELATIVE_SOURCE
            pinned_source.parent.mkdir(parents=True)
            pinned_source.write_text("different source")
            queue = Path(temporary) / "queue"
            queue.mkdir()
            (queue / "state.json").write_text(json.dumps({"units": {
                "unit": {"spec": {"arm_id": "A_LOCAL", "sources": [{
                    "path": str(pinned_source), "sha256": "a" * 64}],
                    "outputs": [str(campaign / ".private" / "out" /
                                    "RELEASE_RECEIPT.json")]}}}}))
            args = Namespace(unit_id="unit", arm_id="A_LOCAL",
                             campaign_root=str(campaign), original_lock=str(lock),
                             a_amendment=str(amendment),
                             a_amendment_sha256="a" * 64,
                             private_output=str(campaign / ".private" / "out"))
            with patch.dict(os.environ, {"PCRL_QUEUE_ROOT": str(queue)}), \
                 patch.object(worker, "assert_active_from_env",
                              return_value={"seal_sha256": "a" * 64}), \
                 patch.object(worker, "_sha", return_value="a" * 64), \
                 patch.object(worker, "run_common_release") as common:
                with self.assertRaisesRegex(PermissionError, "source/output differs"):
                    worker.run(args)
            common.assert_not_called()
