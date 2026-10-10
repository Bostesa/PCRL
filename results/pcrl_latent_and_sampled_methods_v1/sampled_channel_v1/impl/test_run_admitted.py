"""Pure synthetic fit-artifact integration; all queue and Adult readers mocked."""

from pathlib import Path
import json
import os
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .bank import FixedBank
from .channel import GateWeights, JointPolicy
from .release import _bank_hash
from .source_bridge import action_map_sha256
from . import run_admitted as worker


MODULE = worker.__name__


class RunnerCustodyTest(unittest.TestCase):
    def _policy(self):
        p = np.array([[.9, .1], [.7, .3]], np.float64)
        bank = FixedBank.from_source(p, np.arange(2, dtype=np.int64),
                                     np.zeros(2, dtype=np.int64))
        return JointPolicy(bank, bank, ((0, 1),), ((0, 1),),
                           GateWeights.zeros(input_width=8))

    def _run(self, *, feasible):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "private"
            private.mkdir(mode=0o700)
            output = private / "joint_s0"
            policy = self._policy()
            bank_sha = _bank_hash(policy.bank1)
            action_sha = action_map_sha256(policy.actions1)
            amendment = {
                "unit_plan": {"A-JOINT-STOCH-S0": {
                    "arm_id": "A_JOINT_STOCH", "seed": 0,
                    "objective_kind": "protected_bayes", "cpu_seconds": 30.0,
                    "wall_seconds": 30.0, "recipe_seed": 101}},
                "role_bindings": {"source_binding_sha256_by_seed": {"s0": "a" * 64},
                                  "fit_row_unit_pair_sha256": "b" * 64},
                "bank_action_sha256_by_seed": {"s0": {
                    "bank_r1": bank_sha, "bank_r2": bank_sha,
                    "action_r1": action_sha, "action_r2": action_sha}},
                "utility_caps": {"nll_excess": .010, "brier_excess": .005,
                                 "reference_rule": "min_U0_Ucal_on_FIT"},
                "optimizer_recipe": {"updates": 300, "checkpoint_every": 25,
                                     "learning_rate": .01, "rho": 100,
                                     "dual_every": 25, "gradient_norm": 5,
                                     "init_scale": .01,
                                     "identity_logit_bias": 4},
            }
            spec = {"mode": "real_data", "cpu_reserve_s": 30,
                    "max_wall_s": 30}
            args = SimpleNamespace(unit_id="A-JOINT-STOCH-S0",
                                   arm_id="A_JOINT_STOCH", seed=0,
                                   original_lock=root / "lock.json",
                                   a_amendment=root / "a.json",
                                   a_amendment_sha256="c" * 64,
                                   private_output=str(output),
                                   campaign_root=str(root),
                                   repair_manifest=str(root / "repair.json"),
                                   source_audit=str(root / "audit.json"),
                                   portfolio_package=str(root / "portfolio"))
            bindings = {"repair_manifest_sha256": "d" * 64,
                        "source_audit_sha256": "e" * 64,
                        "source_binding_sha256_by_seed": {"s0": "a" * 64},
                        "fit_row_unit_pair_sha256": "b" * 64}
            for name in ("repair.json", "audit.json"):
                (root / name).write_text("{}")
            (root / "portfolio").mkdir()

            class Source:
                source_binding_sha256 = "a" * 64

            checkpoint = {"update": 0, "weights": policy.gate,
                          "metrics": {"r1_nll": .1, "r1_brier": .1,
                                      "r2_nll": .1, "r2_brier": .1},
                          "residual": {"r1_nll": -.1},
                          "feasible": feasible, "objective": .5}

            def fake_fit(*_args, checkpoint_sink=None, **_kwargs):
                assert checkpoint_sink is not None
                checkpoint_sink("identity_biased", checkpoint)
                checkpoint_sink("uniform_logit", checkpoint)
                return {"starts": [
                    {"name": "identity_biased", "completed": True,
                     "elapsed_cpu_seconds": .1, "checkpoints": [checkpoint]},
                    {"name": "uniform_logit", "completed": True,
                     "elapsed_cpu_seconds": .2, "checkpoints": [checkpoint]}],
                    "complete": True, "success": feasible,
                    "policy": policy if feasible else None,
                    "diagnostic_best": {"objective": .5, "start": 0, "update": 0}
                    if feasible else None}

            with patch(f"{MODULE}.admit_fit_unit", return_value=(amendment, spec)), \
                 patch(f"{MODULE}._module_origin_guard"), \
                 patch(f"{MODULE}._binding", return_value=(
                     root / "repair.json", root / "audit.json", root / "portfolio", bindings)), \
                 patch(f"{MODULE}._independent_source_gate"), \
                 patch(f"{MODULE}.TStarSourceStore", return_value=Source()), \
                 patch(f"{MODULE}.RoleOnlyStore"), \
                 patch(f"{MODULE}.load_fit_capability", return_value=(
                     object(), policy, {"r1_nll": .2, "r1_brier": .2,
                                        "r2_nll": .2, "r2_brier": .2},
                     {"role_rows": 3})), \
                 patch(f"{MODULE}.fit_gate", side_effect=fake_fit):
                receipt = worker.run(args)
            self.assertEqual(receipt["status"], "SELECTED_FIT_POLICY" if feasible
                             else "COMPLETE_NO_FEASIBLE_POLICY")
            self.assertTrue((output / "identity_biased_u000.npz").is_file())
            self.assertTrue((output / "uniform_logit_u000.npz").is_file())
            self.assertEqual((output / "POLICY_RECEIPT.json").exists(), feasible)
            self.assertEqual(json.loads((output / "FIT_RECEIPT.json").read_text())
                             ["status"], receipt["status"])

    def test_selected_package_and_no_feasible_path(self):
        self._run(feasible=True)
        self._run(feasible=False)


if __name__ == "__main__":
    unittest.main()
