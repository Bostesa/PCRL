"""One predeclared bank-anchored fit per arm and source seed, loader-free.

The queue/admission worker supplies role-only FIT arrays and owns the resource
limit. This dispatcher reuses source A trainers with B0-relative absolute caps
and always records exact identity as an available feasible checkpoint. A
stopped source trainer never becomes a completed arm via the B0 fallback.
"""

from __future__ import annotations

from dataclasses import asdict
import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.fit import (
    FitRecipe, ProtectedData, TaskOnlyData, prepare_law, TASK_CAP_ORDER,
    exact_metrics,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import JointPolicy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.control_dispatch import (
    _template, _recipe, _stage_seed,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.deterministic_policy import DeterministicJointPolicy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.factor_controls import (
    ProductPolicy, LocalPolicy, SequentialPolicy,
)
from .r2_source_fit_product import fit_product as fit_product_retaining
from .r2_source_fit_local import LocalFactorData, fit_local_pair as fit_local_retaining
from .r2_factor_projection import project_product_checkpoint, project_local_checkpoint
from .r2_source_seq_train import fit_sequential as fit_sequential_retaining
from .r2_sequential import project_second_checkpoint
from .r2_fit_core import fit_central_boundary, PROXY_TOLERANCE, _identity_q, _task_means
from .r2_laws import IdentityPolicy, BoundaryFactorPolicy
from .r2_source_deterministic import (
    DeterministicConfig, as_public_gate, batch_from_prepared,
    fit_deterministic,
)


ARMS = ("T", "D", "P", "L", "J", "S12", "S21")
SOURCE_CONTROL = {"D": "A_JOINT_DET", "P": "A_CENTRAL_PRODUCT",
                  "L": "A_LOCAL", "S12": "A_SEQ12", "S21": "A_SEQ21"}
OPTIMIZER = {"updates": 300, "checkpoint_every": 25,
             "learning_rate": .01, "rho": 100, "dual_every": 25,
             "gradient_norm": 5, "init_scale": .01,
             "identity_logit_bias": 4}


def fit_arm(arm_id: str, capability, base: JointPolicy, *,
            source_seed: int, recipe_seed: int, cpu_seconds: float,
            unit_ids, source_binding_sha256: str,
            seq_private_root: bytes | None = None,
            seq_private_cache_path=None):
    if (arm_id not in ARMS or type(base) is not JointPolicy or
            source_seed not in (0, 1) or type(recipe_seed) is not int or
            recipe_seed < 0 or not np.isfinite(cpu_seconds) or cpu_seconds <= 0):
        raise ValueError("unregistered FIT arm, seed or source bank")
    if arm_id == "T":
        if type(capability) is not TaskOnlyData:
            raise TypeError("T fit must not receive SEX capability")
        task = capability
    else:
        if type(capability) is not ProtectedData:
            raise TypeError("protected fit requires admitted SEX capability")
        task = capability.task
    if len(unit_ids) != len(task.y1):
        raise ValueError("FIT unit IDs are not row-aligned")
    prepared = prepare_law(task, base)
    b0 = _task_means(prepared, _identity_q(prepared))
    caps = {key: b0[key] + PROXY_TOLERANCE[key] for key in TASK_CAP_ORDER}
    identity = IdentityPolicy(base)
    if arm_id in ("J", "T"):
        result = fit_central_boundary(
            capability, base,
            objective_kind="task_rate" if arm_id == "T" else "protected_bayes",
            recipe=FitRecipe(updates=300, checkpoint_every=25,
                             cpu_seconds=cpu_seconds, seed=recipe_seed))
        if not result["complete"]:
            return None, {"arm_id": arm_id, "status": "INCOMPLETE_SEARCH",
                          "b0_fit_costs": b0, "fit_caps": caps,
                          "source": result["source"]}
        selected = result["policy"]
        selection = result["selection"]
        if selection["identity_alias"]:
            selected = identity
        return selected, {"arm_id": arm_id,
                          "status": ("COMPLETE_IDENTITY_ALIAS" if
                                     selection["identity_alias"] else
                                     "COMPLETE_SELECTED_PROTECTED"),
                          "b0_fit_costs": b0, "fit_caps": caps,
                          "selection": selection,
                          "source": result["source"]}

    if arm_id == "D":
        identity_pairs = np.column_stack((
            [base.actions1[int(cell)][0] for cell in task.cell1],
            [base.actions2[int(cell)][0] for cell in task.cell2])).astype(np.int64)
        batch = batch_from_prepared(prepared, identity_public_pair=identity_pairs,
                                    sex=capability.sex)
        config = DeterministicConfig(
            mode="private", steps=300, learning_rate=.01,
            temperature_start=1.0, temperature_end=.5, rho=100.,
            dual_every=25, replay_every=25, grad_clip=5.,
            cpu_limit_seconds=cpu_seconds,
            seed=_stage_seed(recipe_seed, SOURCE_CONTROL[arm_id], "deterministic"),
            bias_deltas=(-.5, -.1, .1, .5))
        result = fit_deterministic(batch, config,
            caps=tuple(caps[key] for key in
                       ("r1_nll", "r1_brier", "r2_nll", "r2_brier")))
        checkpoints = [(start, update, phase, weights, asdict(metrics))
                       for start, update, phase, weights, metrics in result.checkpoints]
        source = {"complete": result.complete, "success": result.success,
                  "starts": [asdict(start) for start in result.starts],
                  "metrics": None, "elapsed_cpu_seconds": result.cpu_seconds,
                  "selection": None if result.metrics is None else asdict(result.metrics)}
        if not result.complete:
            return None, {"arm_id": arm_id, "status": "INCOMPLETE_SEARCH",
                          "b0_fit_costs": b0, "fit_caps": caps,
                          "source": source, "_checkpoint_models": checkpoints}
        if not result.success:
            return identity, {"arm_id": arm_id,
                              "status": "COMPLETE_IDENTITY_ALIAS",
                              "b0_fit_costs": b0, "fit_caps": caps,
                              "selection": {"start": "exact_identity", "update": 0,
                                            "identity_alias": True},
                              "source": source, "_checkpoint_models": checkpoints}
        selected = DeterministicJointPolicy(
            base.bank1, base.bank2, base.actions1, base.actions2,
            as_public_gate(result.weights, type(base.gate)))
        source["metrics"] = exact_metrics(capability, selected)
        matches = [start for start in source["starts"]
                   if start["terminal_metrics"] == source["selection"]]
        if not matches:
            raise RuntimeError("deterministic source winner lacks a start receipt")
        chosen = matches[0]
        return selected, {"arm_id": arm_id, "status": "COMPLETE_SELECTED_PROTECTED",
                          "b0_fit_costs": b0, "fit_caps": caps,
                          "selection": {"start": chosen["start"],
                                        "update": chosen["chosen_checkpoint_update"] or 0,
                                        "polish_trials": chosen["polish_trials"],
                                        "terminal_metrics": source["selection"]},
                          "source": source, "_checkpoint_models": checkpoints}
    if arm_id == "P":
        source = fit_product_retaining(
            capability, _template(base, SOURCE_CONTROL[arm_id]), caps=caps,
            recipe=_recipe(OPTIMIZER, updates=300, cpu=cpu_seconds,
                           seed=_stage_seed(recipe_seed, SOURCE_CONTROL[arm_id], "pair")))
        if not source["complete"]:
            return None, {"arm_id": arm_id, "status": "INCOMPLETE_SEARCH",
                          "b0_fit_costs": b0, "fit_caps": caps,
                          "source": {k: v for k, v in source.items()
                                     if k not in ("policy", "checkpoint_policies")},
                          "_checkpoint_models": source["checkpoint_policies"]}
        projected = []
        for start, update, checkpoint in source["checkpoint_policies"]:
            policy, metrics = project_product_checkpoint(capability, checkpoint)
            projected.append((metrics["objective"], start, update,
                              metrics["eta1"], metrics["eta2"], policy, metrics))
        _, start, update, _, _, selected, selection = min(
            projected, key=lambda row: row[:5])
        if selection["identity_alias"]:
            selected = identity
        summary = {"arm_id": arm_id,
                   "status": ("COMPLETE_IDENTITY_ALIAS" if selection["identity_alias"]
                              else "COMPLETE_SELECTED_PROTECTED"),
                   "b0_fit_costs": b0, "fit_caps": caps,
                   "selection": {"start": ("identity_biased", "uniform_logit")[start],
                                 "update": update, **selection},
                   "source": {k: v for k, v in source.items()
                              if k not in ("policy", "checkpoint_policies")},
                   "projected_checkpoints": [
                       {"start": ("identity_biased", "uniform_logit")[row[1]],
                        "update": row[2], **row[6]} for row in projected]}
        summary["_checkpoint_models"] = source["checkpoint_policies"]
        return selected, summary
    if arm_id == "L":
        units = tuple(str(v) for v in unit_ids)
        local1 = LocalFactorData(task.score1, task.cell1, task.y1, capability.sex,
                                 base.bank1, base.actions1, units)
        local2 = LocalFactorData(task.score2, task.cell2, task.y2, capability.sex,
                                 base.bank2, base.actions2, units)
        source = fit_local_retaining(
            local1, local2, gate=base.gate,
            caps1=(caps["r1_nll"], caps["r1_brier"]),
            caps2=(caps["r2_nll"], caps["r2_brier"]),
            recipe1=_recipe(OPTIMIZER, updates=150, cpu=cpu_seconds,
                            seed=_stage_seed(recipe_seed, SOURCE_CONTROL[arm_id], "r1")),
            recipe2=_recipe(OPTIMIZER, updates=150, cpu=cpu_seconds,
                            seed=_stage_seed(recipe_seed, SOURCE_CONTROL[arm_id], "r2")))
        if not source["complete"]:
            return None, {"arm_id": arm_id, "status": "INCOMPLETE_SEARCH",
                          "b0_fit_costs": b0, "fit_caps": caps,
                          "source": {"complete": source["complete"],
                                     "elapsed_cpu_seconds": source["elapsed_cpu_seconds"],
                                     "recipient1": {k: v for k, v in source["recipient1"].items()
                                                    if k not in ("factor", "checkpoint_factors")},
                                     "recipient2": None if source["recipient2"] is None else
                                     {k: v for k, v in source["recipient2"].items()
                                      if k not in ("factor", "checkpoint_factors")}},
                          "_checkpoint_models": {
                              "recipient1": source["recipient1"]["checkpoint_factors"],
                              "recipient2": ([] if source["recipient2"] is None else
                                             source["recipient2"]["checkpoint_factors"])}}
        chosen = []
        inventories = []
        for recipient, local in ((1, local1), (2, local2)):
            projection = []
            for start, update, factor in source[f"recipient{recipient}"]["checkpoint_factors"]:
                metrics = project_local_checkpoint(local, factor, recipient=recipient)
                projection.append((metrics["objective"], start, update,
                                   metrics["eta"], factor, metrics))
            _, start, update, _, factor, metrics = min(projection, key=lambda row: row[:4])
            chosen.append((factor, metrics["eta"], start, update, metrics))
            inventories.append([{"start": ("identity_biased", "uniform_logit")[row[1]],
                                 "update": row[2], **row[5]} for row in projection])
        selected = BoundaryFactorPolicy(
            LocalPolicy(base.bank1, base.bank2, base.actions1, base.actions2,
                        base.gate, chosen[0][0], chosen[1][0]),
            chosen[0][1], chosen[1][1])
        alias = chosen[0][1] == chosen[1][1] == 0.0
        if alias:
            selected = identity
        summary = {"arm_id": arm_id,
                   "status": "COMPLETE_IDENTITY_ALIAS" if alias else "COMPLETE_SELECTED_PROTECTED",
                   "b0_fit_costs": b0, "fit_caps": caps,
                   "selection": {"start": "+".join(
                       ("identity_biased", "uniform_logit")[v[2]] for v in chosen),
                       "update": max(v[3] for v in chosen),
                       "recipient1": chosen[0][4], "recipient2": chosen[1][4],
                       "eta1": chosen[0][1], "eta2": chosen[1][1],
                       "identity_alias": alias},
                   "source": {"complete": source["complete"],
                              "elapsed_cpu_seconds": source["elapsed_cpu_seconds"],
                              "recipient1": {k: v for k, v in source["recipient1"].items()
                                             if k not in ("factor", "checkpoint_factors")},
                              "recipient2": {k: v for k, v in source["recipient2"].items()
                                             if k not in ("factor", "checkpoint_factors")}},
                   "projected_checkpoints": {"r1": inventories[0], "r2": inventories[1]}}
        summary["_checkpoint_models"] = {
            "recipient1": source["recipient1"]["checkpoint_factors"],
            "recipient2": source["recipient2"]["checkpoint_factors"]}
        return selected, summary

    if arm_id in ("S12", "S21"):
        if (type(seq_private_root) is not bytes or len(seq_private_root) < 32 or
                seq_private_cache_path is None):
            raise ValueError("sequential arm needs frozen private FIT draw custody")
        template = _template(base, SOURCE_CONTROL[arm_id])
        source = fit_sequential_retaining(
            capability, template, caps=caps,
            recipe_stage1=_recipe(OPTIMIZER, updates=150, cpu=cpu_seconds/2,
                                  seed=_stage_seed(recipe_seed, SOURCE_CONTROL[arm_id], "first")),
            recipe_stage2=_recipe(OPTIMIZER, updates=150, cpu=cpu_seconds/2,
                                  seed=_stage_seed(recipe_seed, SOURCE_CONTROL[arm_id], "second")),
            total_cpu_seconds=cpu_seconds, unit_ids=unit_ids,
            private_root=seq_private_root, source_seed=source_seed,
            source_binding_sha256=source_binding_sha256,
            private_draw_cache_path=seq_private_cache_path)
        public_source = {k: v for k, v in source.items() if k not in
                         ("policy", "private_first_codes", "checkpoint_policies",
                          "stage1_checkpoint_gates")}
        if not source["complete"]:
            return None, {"arm_id": arm_id, "status": "INCOMPLETE_SEARCH",
                          "b0_fit_costs": b0, "fit_caps": caps,
                          "source": public_source,
                          "_checkpoint_models": {
                              "stage1": source["stage1_checkpoint_gates"],
                              "stage2": source.get("checkpoint_policies", [])}}
        candidates = []
        for start, update, checkpoint in source["checkpoint_policies"]:
            policy, metrics = project_second_checkpoint(
                capability, checkpoint, eta_first=source["first_eta"])
            candidates.append((metrics["objective"], start, update,
                               metrics["eta_second"], policy, metrics))
        _, start, update, _, selected, selection = min(
            candidates, key=lambda row: row[:4])
        if selection["identity_alias"]:
            selected = identity
        summary = {"arm_id": arm_id,
                          "status": ("COMPLETE_IDENTITY_ALIAS" if selection["identity_alias"]
                                     else "COMPLETE_SELECTED_PROTECTED"),
                          "b0_fit_costs": b0, "fit_caps": caps,
                          "selection": {"start": ("identity_biased", "uniform_logit")[start],
                                        "update": update, **selection},
                          "source": public_source,
                          "projected_checkpoints": [
                              {"start": ("identity_biased", "uniform_logit")[row[1]],
                               "update": row[2], **row[5]} for row in candidates]}
        summary["_checkpoint_models"] = {
            "stage1": source["stage1_checkpoint_gates"],
            "stage2": source["checkpoint_policies"]}
        return selected, summary

    raise AssertionError("registered arm has no dispatcher")
