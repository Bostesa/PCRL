"""Data-free checks for control dispatch permissions and private receipt fields."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.bank import FixedBank
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import GateWeights, JointPolicy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.fit import TaskOnlyData, ProtectedData
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls import control_dispatch as dispatch
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.deterministic_trainer import FitResult


OPTIMIZER = {"updates": 300, "checkpoint_every": 25, "learning_rate": .01,
             "rho": 100, "dual_every": 25, "gradient_norm": 5,
             "init_scale": .01, "identity_logit_bias": 4}
CAPS = {"r1_nll": 1., "r1_brier": 1., "r2_nll": 2., "r2_brier": 2.}


def fixture():
    b1 = FixedBank.from_source(np.array([[.9, .1]], dtype=np.float64),
                               np.array([0], dtype=np.int64),
                               np.array([0], dtype=np.int64))
    p2 = np.array([[.8, .04, .04, .04, .04, .04]], dtype=np.float64)
    b2 = FixedBank.from_source(p2, np.array([0], dtype=np.int64),
                               np.array([0], dtype=np.int64))
    policy = JointPolicy(b1, b2, ((0,),), ((0,),),
                         GateWeights.zeros(input_width=12))
    task = TaskOnlyData(np.repeat(b1.probability, 3, axis=0),
                        np.repeat(b2.probability, 3, axis=0),
                        np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64),
                        np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64))
    protected = ProtectedData(task, np.array([0, 1, 0], dtype=np.int64))
    return policy, task, protected


class DispatchTest(TestCase):
    def test_task_only_control_refuses_sex_capability(self):
        base, _, protected = fixture()
        with self.assertRaisesRegex(TypeError, "must not receive SEX"):
            dispatch.fit_control_arm("A_TASK_DET", protected, base, CAPS,
                                     optimizer=OPTIMIZER, recipe_seed=1,
                                     cpu_seconds=30, unit_ids=(0, 1, 2))

    def test_task_only_control_passes_no_sex_to_hard_fitter(self):
        base, task, _ = fixture()
        with patch.object(dispatch, "fit_deterministic",
                          return_value=FitResult(True, False, None, None, (), .1)) as fit:
            selected, receipt = dispatch.fit_control_arm(
                "A_TASK_DET", task, base, CAPS, optimizer=OPTIMIZER,
                recipe_seed=1, cpu_seconds=30, unit_ids=(0, 1, 2))
        self.assertIsNone(selected)
        self.assertIsNone(fit.call_args.args[0].sex)
        self.assertEqual(fit.call_args.kwargs["config"].mode, "task") if "config" in fit.call_args.kwargs else self.assertEqual(fit.call_args.args[1].mode, "task")
        self.assertEqual(receipt["arm_id"], "A_TASK_DET")

    def test_seq_private_person_codes_never_enter_public_receipt(self):
        base, _, protected = fixture()
        fake = {"complete": False, "success": False, "stage1_complete": True,
                "stage1_starts": [], "stage2_starts": [], "elapsed_cpu_seconds": .1,
                "private_first_codes": np.array([0, 0, 0], dtype=np.int64),
                "draw_cache_sha256": "a" * 64, "coverage": {"unique_fit_units": 3}}
        with patch.object(dispatch, "fit_sequential", return_value=fake):
            selected, receipt = dispatch.fit_control_arm(
                "A_SEQ12", protected, base, CAPS, optimizer=OPTIMIZER,
                recipe_seed=1, cpu_seconds=30, unit_ids=(0, 1, 2),
                source_seed=0, source_binding_sha256="b" * 64,
                seq_private_root=b"r" * 32, seq_private_cache_path="private.json")
        self.assertIsNone(selected)
        self.assertNotIn("private_first_codes", receipt)
        self.assertNotIn("private_first_codes", str(receipt))

    def test_local_template_receives_only_own_score_widths(self):
        base, _, _ = fixture()
        local = dispatch._template(base, "A_LOCAL")
        self.assertEqual(local.factor1.w1.shape[0], 4)
        self.assertEqual(local.factor2.w1.shape[0], 8)
