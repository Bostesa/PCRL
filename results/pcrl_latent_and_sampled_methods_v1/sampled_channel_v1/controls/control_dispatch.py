"""Pure dispatch from an admitted FIT capability to one matched A control.

This module opens no role files. The queue/admission worker must supply the
source-checked R12 capability, bank/action law, fixed recipe and private unit
identities. Returned public summaries never contain person-level draws.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math

import numpy as np

from ..impl.channel import JointPolicy
from ..impl.fit import FitRecipe, ProtectedData, TaskOnlyData, TASK_CAP_KEYS, exact_metrics, prepare_law
from .deterministic_policy import DeterministicJointPolicy
from .deterministic_trainer import (DeterministicConfig, as_public_gate,
                                    batch_from_prepared, fit_deterministic)
from .factor_controls import (FactorGate, LocalPolicy, ProductPolicy,
                              SequentialPolicy, public_probability_embedding)
from .fit_local import LocalFactorData, fit_local_pair
from .fit_product import fit_product
from .seq_train import fit_sequential


ARMS = frozenset(("A_TASK_DET", "A_JOINT_DET", "A_CENTRAL_PRODUCT",
                  "A_LOCAL", "A_SEQ12", "A_SEQ21"))
PROTECTED_ARMS = ARMS - {"A_TASK_DET"}
OPTIMIZER_KEYS = frozenset(("updates", "checkpoint_every", "learning_rate", "rho",
                            "dual_every", "gradient_norm", "init_scale",
                            "identity_logit_bias"))


def _stage_seed(base: int, arm: str, stage: str) -> int:
    """Independent fixed PCG64 seed domains, unaffected by fit outcomes."""
    if type(base) is not int or base < 0 or arm not in ARMS or not stage:
        raise ValueError("unsealed control RNG domain")
    payload = json.dumps(["SCV1-control-stage-seed-v1", base, arm, stage],
                         separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _recipe(optimizer: dict, *, updates: int, cpu: float, seed: int) -> FitRecipe:
    if (type(optimizer) is not dict or optimizer.keys() != OPTIMIZER_KEYS or
            optimizer["updates"] != 300 or optimizer["checkpoint_every"] != 25 or
            optimizer["dual_every"] != 25 or optimizer["learning_rate"] != .01 or
            optimizer["rho"] != 100 or optimizer["gradient_norm"] != 5 or
            optimizer["init_scale"] != .01 or
            optimizer["identity_logit_bias"] != 4 or
            updates not in (150, 300) or not math.isfinite(cpu) or cpu <= 0):
        raise PermissionError("control optimizer differs from frozen schedule")
    return FitRecipe(**{**optimizer, "updates": updates,
                        "cpu_seconds": cpu, "seed": seed})


def _factor(width: int, embedding=None) -> FactorGate:
    return FactorGate(np.zeros((width, 16)), np.zeros(16),
                      np.zeros((16, 4)), np.zeros(4), embedding)


def _template(base: JointPolicy, arm: str):
    width = base.bank1.probability.shape[1] + base.bank2.probability.shape[1] + 4
    if arm == "A_CENTRAL_PRODUCT":
        return ProductPolicy(base.bank1, base.bank2, base.actions1, base.actions2,
                             base.gate, _factor(width), _factor(width))
    if arm == "A_LOCAL":
        return LocalPolicy(base.bank1, base.bank2, base.actions1, base.actions2,
                           base.gate, _factor(base.bank1.probability.shape[1] + 2),
                           _factor(base.bank2.probability.shape[1] + 2))
    if arm in ("A_SEQ12", "A_SEQ21"):
        order = arm[-2:]
        first_bank = base.bank1 if order == "12" else base.bank2
        return SequentialPolicy(
            base.bank1, base.bank2, base.actions1, base.actions2, base.gate,
            _factor(width), _factor(width,
                                    public_probability_embedding(first_bank.probability)),
            order=order, first_checkpoint_sha256="0" * 64)
    raise ValueError("arm has no factor template")


def _safe_public(value):
    """Accept only JSON scalar/checkpoint structure; never serialize arrays."""
    if isinstance(value, np.generic):
        return _safe_public(value.item())
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [_safe_public(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _safe_public(item) for key, item in value.items()}
    raise TypeError("public control receipt contains an array or unsupported value")


def fit_control_arm(arm_id: str, capability, base: JointPolicy, caps: dict,
                    *, optimizer: dict, recipe_seed: int, cpu_seconds: float,
                    unit_ids=None, source_seed=None, source_binding_sha256=None,
                    seq_private_root=None, seq_private_cache_path=None):
    """Run one fully named control; return (selected policy or None, public receipt).

    A failed or incomplete arm has no policy package. The caller keeps private
    SEQ FIT codes/cache, all role labels and all checkpoint arrays private.
    """
    if (arm_id not in ARMS or type(base) is not JointPolicy or
            type(caps) is not dict or caps.keys() != TASK_CAP_KEYS or
            any(not math.isfinite(float(value)) for value in caps.values()) or
            not math.isfinite(cpu_seconds) or cpu_seconds <= 0):
        raise ValueError("control identity, source law or utility caps differ")
    if arm_id == "A_TASK_DET":
        if type(capability) is not TaskOnlyData:
            raise TypeError("task-only deterministic arm must not receive SEX")
        task = capability
    else:
        if type(capability) is not ProtectedData:
            raise TypeError("protected control requires admitted FIT SEX")
        task = capability.task
    if (type(unit_ids) not in (tuple, list) or len(unit_ids) != len(task.y1) or
            any(type(unit) not in (str, int, np.int64) for unit in unit_ids)):
        raise ValueError("aligned FIT unit identities are required")
    _recipe(optimizer, updates=300, cpu=cpu_seconds,
            seed=_stage_seed(recipe_seed, arm_id, "contract"))
    selected = None
    if arm_id in ("A_TASK_DET", "A_JOINT_DET"):
        prepared = prepare_law(task, base)
        identity = np.column_stack((
            [base.actions1[int(cell)][0] for cell in task.cell1],
            [base.actions2[int(cell)][0] for cell in task.cell2])).astype(np.int64)
        batch = batch_from_prepared(
            prepared, identity_public_pair=identity,
            sex=None if arm_id == "A_TASK_DET" else capability.sex)
        config = DeterministicConfig(
            mode="task" if arm_id == "A_TASK_DET" else "private",
            steps=300, learning_rate=.01, temperature_start=1.0,
            temperature_end=.5, rho=100., dual_every=25,
            replay_every=25, grad_clip=5., cpu_limit_seconds=cpu_seconds,
            seed=_stage_seed(recipe_seed, arm_id, "deterministic"),
            bias_deltas=(-.5, -.1, .1, .5))
        result = fit_deterministic(batch, config,
                                   caps=tuple(caps[key] for key in
                                              ("r1_nll", "r1_brier", "r2_nll", "r2_brier")))
        if result.success:
            selected = DeterministicJointPolicy(
                base.bank1, base.bank2, base.actions1, base.actions2,
                as_public_gate(result.weights, type(base.gate)))
            metrics = exact_metrics(capability, selected)
        else:
            metrics = None
        public = {"complete": result.complete, "success": result.success,
                  "starts": [asdict(start) for start in result.starts],
                  "metrics": metrics, "elapsed_cpu_seconds": result.cpu_seconds,
                  "selection": None if result.metrics is None else asdict(result.metrics)}
    elif arm_id == "A_CENTRAL_PRODUCT":
        result = fit_product(capability, _template(base, arm_id), caps=caps,
                             recipe=_recipe(optimizer, updates=300, cpu=cpu_seconds,
                                            seed=_stage_seed(recipe_seed, arm_id, "pair")))
        selected = result["policy"]
        public = {key: result[key] for key in
                  ("complete", "success", "starts", "metrics", "local_identity_guards",
                   "diagnostic_best", "elapsed_cpu_seconds")}
    elif arm_id == "A_LOCAL":
        units = tuple(str(value) for value in unit_ids)
        d1 = LocalFactorData(task.score1, task.cell1, task.y1, capability.sex,
                             base.bank1, base.actions1, units)
        d2 = LocalFactorData(task.score2, task.cell2, task.y2, capability.sex,
                             base.bank2, base.actions2, units)
        result = fit_local_pair(
            d1, d2, gate=base.gate,
            caps1=(caps["r1_nll"], caps["r1_brier"]),
            caps2=(caps["r2_nll"], caps["r2_brier"]),
            recipe1=_recipe(optimizer, updates=150, cpu=cpu_seconds,
                            seed=_stage_seed(recipe_seed, arm_id, "r1")),
            recipe2=_recipe(optimizer, updates=150, cpu=cpu_seconds,
                            seed=_stage_seed(recipe_seed, arm_id, "r2")))
        selected = result["policy"]
        public = {"complete": result["complete"], "success": result["success"],
                  "recipient1": {key: result["recipient1"][key] for key in
                                 ("complete", "success", "starts", "metrics", "identity_guard")},
                  "recipient2": None if result["recipient2"] is None else
                  {key: result["recipient2"][key] for key in
                   ("complete", "success", "starts", "metrics", "identity_guard")},
                  "metrics": result["metrics"],
                  "elapsed_cpu_seconds": result["elapsed_cpu_seconds"]}
    else:
        if (type(seq_private_root) is not bytes or len(seq_private_root) < 32 or
                seq_private_cache_path is None or source_seed not in (0, 1) or
                type(source_binding_sha256) is not str):
            raise ValueError("SEQ requires frozen private FIT stream and cache")
        result = fit_sequential(
            capability, _template(base, arm_id), caps=caps,
            recipe_stage1=_recipe(optimizer, updates=150, cpu=cpu_seconds / 2,
                                  seed=_stage_seed(recipe_seed, arm_id, "first")),
            recipe_stage2=_recipe(optimizer, updates=150, cpu=cpu_seconds / 2,
                                  seed=_stage_seed(recipe_seed, arm_id, "second")),
            total_cpu_seconds=cpu_seconds, unit_ids=unit_ids,
            private_root=seq_private_root, source_seed=source_seed,
            source_binding_sha256=source_binding_sha256,
            private_draw_cache_path=seq_private_cache_path)
        selected = result.get("policy")
        public = {key: result.get(key) for key in
                  ("complete", "success", "stage1_complete", "stage1_starts",
                   "stage2_starts", "elapsed_cpu_seconds", "metrics",
                   "first_checkpoint_sha256", "draw_cache_sha256", "coverage",
                   "stage1_metrics")}
    if selected is not None and any(
            exact_metrics(capability, selected)[key] > caps[key] for key in TASK_CAP_KEYS):
        raise AssertionError("selected control violates exact FIT utility cap")
    return selected, _safe_public({"arm_id": arm_id, **public})
