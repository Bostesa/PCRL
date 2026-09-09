"""Explicit added-protection strengths from immutable historical forks.

This continuation-only study never changes historical configuration globals,
recreates source/observer warmup, or receives reserved utility labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import acs_coalition_training as historical
from experiments import acs_bottleneck_training as old
from experiments.acs_transfer_models import array_digest, state_digest

BETAS = (0., .025, .05, .1, .2)
NEW_BETAS = (.025, .05, .2)
REGIMES = ("Iplus", "J")


def coefficients(regime, beta):
    if regime not in REGIMES or beta not in BETAS:
        raise ValueError("Use the frozen Iplus/J regimes and beta grid")
    return {"individual": -.1, "extra_local": -float(beta) if regime == "Iplus" else 0.,
            "coalition": -float(beta) if regime == "J" else 0.}


def restore_fork(checkpoint, pre, interface, seed):
    """Reconstruct objects, then strictly restore every historical state."""
    state = checkpoint["model_state"]
    constructor = {key: state[key] for key in ("input_mean", "input_scale")}
    for key, value in state.items():
        if key.startswith("branches.A.mapper."):
            constructor[key[len("branches.A."):]] = value
        for purpose, tasks in historical.PURPOSE_TASKS.items():
            if any(key.startswith(f"branches.{purpose}.heads.{task}.") for task in tasks):
                constructor[key[len(f"branches.{purpose}."):]] = value
    model = historical.CoalitionModel(pre, constructor)
    model.load_state_dict(copy.deepcopy(state), strict=True)
    observers, _ = historical.make_observers(interface, seed)
    observers.load_state_dict(copy.deepcopy(checkpoint["adversary_state"]), strict=True)
    optimizer, observer_optimizer = old._adam(model.parameters()), old._adam(observers.parameters())
    optimizer.load_state_dict(copy.deepcopy(checkpoint["mapper_optimizer"]))
    observer_optimizer.load_state_dict(copy.deepcopy(checkpoint["adversary_optimizer"]))
    expected = {"model": state_digest(state), "adversaries": state_digest(checkpoint["adversary_state"]),
                "mapper_optimizer": old.tree_digest(checkpoint["mapper_optimizer"]),
                "adversary_optimizer": old.tree_digest(checkpoint["adversary_optimizer"])}
    if historical._hashes(model, observers, optimizer, observer_optimizer) != expected:
        raise AssertionError("Historical model/observer/Adam fork was not restored exactly")
    return model, observers, optimizer, observer_optimizer


def mapper_update(model, observers, optimizer, x, source, attributes, priors, interface, regime, beta):
    """Literal historical objective/update order with explicit extra strength."""
    coeff = coefficients(regime, beta)
    optimizer.zero_grad(set_to_none=True); observers.zero_grad(set_to_none=True)
    flags = [p.requires_grad for p in observers.parameters()]
    observers.requires_grad_(False)
    try:
        losses, detail = historical.components(model, observers, x, source, attributes, priors, interface)
        objective = losses["source"]
        for name, weight in coeff.items():
            if weight:
                objective = objective + weight * losses[name]
        if not torch.isfinite(objective): raise FloatingPointError("Nonfinite strength continuation objective")
        objective.backward(); optimizer.step()
    finally:
        for parameter, flag in zip(observers.parameters(), flags): parameter.requires_grad_(flag)
    return detail["source_support"]


def gradient_diagnostics(model, observers, x, source, attributes, priors, interface, regime, beta):
    """Actual raw/applied branch and head gradients without RNG/state changes."""
    before = old.tree_digest((model.state_dict(), observers.state_dict(), [p.grad for p in model.parameters()], [p.grad for p in observers.parameters()]))
    rng = torch.get_rng_state().clone()
    losses, detail = historical.components(model, observers, x, source, attributes, priors, interface)
    coeff = {"source": 1., **coefficients(regime, beta)}
    groups = {purpose + "_mapper": list(branch.mapper.parameters()) for purpose, branch in model.branches.items()}
    groups.update({purpose + "_heads": list(branch.heads.parameters()) for purpose, branch in model.branches.items()})
    groups["all_mappers"] = groups["A_mapper"] + groups["B_mapper"]
    result = {"coefficients": coeff, "condition": regime, "beta": float(beta), "interface": interface,
              "losses": {name: float(value.detach()) for name, value in losses.items()},
              "source_task_losses": {name: float(value.detach()) for name, value in detail["tasks"].items()},
              "observer_ce": {name: float(value.detach()) for name, value in detail["observer_ce"].items()}, "groups": {}}
    for group, parameters in groups.items():
        vectors = {}
        for name, value in losses.items():
            grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            vectors[name] = torch.cat([(torch.zeros_like(p) if grad is None else grad).reshape(-1) for p, grad in zip(parameters, grads)])
        applied = {name: coeff[name] * vector for name, vector in vectors.items()}; record = {}
        for convention, values in (("raw", vectors), ("applied", applied)):
            for name, vector in values.items(): record[f"{name}_{convention}_l2"] = float(torch.linalg.vector_norm(vector))
            for left, right in (("source", "individual"), ("source", "coalition"), ("source", "extra_local"), ("individual", "extra_local"), ("extra_local", "coalition"), ("individual", "coalition")):
                dot = float(torch.dot(values[left], values[right]))
                denominator = record[f"{left}_{convention}_l2"] * record[f"{right}_{convention}_l2"]
                record[f"{left}_{right}_{convention}_dot"] = dot
                record[f"{left}_{right}_{convention}_cosine"] = dot / denominator if denominator else None
        record["combined_protection_l2"] = float(torch.linalg.vector_norm(applied["individual"] + applied["extra_local"] + applied["coalition"]))
        record["objective_l2"] = float(torch.linalg.vector_norm(sum(applied.values())))
        if not all(value is None or np.isfinite(value) for value in record.values()): raise FloatingPointError("Nonfinite diagnostic")
        result["groups"][group] = record
    routing = {}
    for owner in ("A", "B"):
        local = torch.stack([detail["observer_ce"][owner + "__" + target] / priors[target]["entropy"] for target in historical.ROLE_TARGETS[owner]]).mean()
        extra = torch.stack([detail["observer_ce"][owner + "__" + target] / priors[target]["entropy"] for target in historical.ATTRIBUTE_SCHEMA]).mean()
        for destination in ("A", "B"):
            for name, value in (("local", local), ("extra", extra)):
                grads = torch.autograd.grad(value, groups[destination+"_mapper"], retain_graph=True, allow_unused=True)
                norm = float(torch.sqrt(sum((g*g).sum() for g in grads if g is not None))) if any(g is not None for g in grads) else 0.
                routing[f"{name}_{owner}_to_{destination}_mapper_l2"] = norm
                if owner != destination and norm: raise AssertionError("Local gradient crossed purpose boundary")
    result["routing"] = routing
    after = old.tree_digest((model.state_dict(), observers.state_dict(), [p.grad for p in model.parameters()], [p.grad for p in observers.parameters()]))
    if before != after or not torch.equal(rng, torch.get_rng_state()): raise AssertionError("Diagnostic changed state or RNG")
    result.update(state_unchanged=True, rng_unchanged=True)
    return result


@torch.no_grad()
def curve(model, observers, x, source, attributes, priors, interface, regime, beta, epoch, counts, xv=None, validation=None):
    result = historical._curve(model, observers, x, source, attributes, priors, interface, regime, epoch, counts, xv, validation)
    result["coefficients"] = {"source": 1., **coefficients(regime, beta)}
    losses, _ = historical.components(model, observers, x, source, attributes, priors, interface)
    result["objective"] = float(sum(result["coefficients"][name] * value for name, value in losses.items()))
    result["beta"] = float(beta)
    return result


def train_continuation(pca_fit, source_y, attr_y, pre, fork_checkpoint, historical_metadata, seed, out, *,
                       interface, regime, beta, pca_val=None, source_val=None, miniature=False):
    """Continue one exact historical fork; science fits only the three new betas."""
    started = time.perf_counter(); coeff = coefficients(regime, beta)
    if interface not in ("F", "P") or seed not in (0, 1, 2): raise ValueError("Frozen interface/seed required")
    if not miniature and beta not in NEW_BETAS: raise ValueError("Scientific beta=0/.1 endpoints must reuse historical fits")
    if historical_metadata["seed"] != seed or historical_metadata["miniature"] != miniature:
        raise ValueError("Historical metadata seed/test mode mismatch")
    xraw = old._features(pca_fit)
    source = old._labels(source_y, len(xraw), historical.SOURCE_SCHEMA, "source fitting")
    attributes = old._labels(attr_y, len(xraw), historical.ATTRIBUTE_SCHEMA, "attribute fitting")
    if (pca_val is None) != (source_val is None): raise ValueError("Both source validation inputs and labels required")
    xvraw = old._features(pca_val) if pca_val is not None else None
    validation = old._labels(source_val, len(xvraw), historical.SOURCE_SCHEMA, "source validation") if xvraw is not None else None
    for name, actual in (("fit_input_sha256", array_digest(xraw)),
                        ("source_label_hashes", {k: array_digest(y.numpy()) for k, y in source.items()}),
                        ("attribute_label_hashes", {k: array_digest(y.numpy()) for k, y in attributes.items()})):
        if historical_metadata[name] != actual: raise ValueError("Historical fitting identity mismatch: " + name)
    if xvraw is not None:
        if historical_metadata["source_validation_input_sha256"] != array_digest(xvraw) or historical_metadata["source_validation_label_hashes"] != {k: array_digest(y.numpy()) for k, y in validation.items()}:
            raise ValueError("Historical source-validation identity mismatch")
    path = Path(fork_checkpoint); fork_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    original_tree = old.tree_digest(checkpoint)
    config = copy.deepcopy(historical.CONFIG)
    config.update(study="acs_coalition_strength_v1", historical_training_recipe="acs_coalition_v1")
    if miniature: config.update(warm_base_epochs=1, warm_adversary_epochs=1, continuation_epochs=2, batch_size=16, curve_interval=1)
    for key in ("continuation_epochs", "batch_size", "lr", "betas", "eps", "weight_decay", "adversary_updates_per_mapper_step"):
        if historical_metadata["config"][key] != config[key]: raise ValueError("Historical recipe mismatch: " + key)
    if checkpoint["schedule_state"]["phase"] != "continuation" or checkpoint["schedule_state"]["completed_epochs"] != 0 or checkpoint["schedule_state"]["next_minibatch_index"] != 0:
        raise ValueError("Only the exact pre-continuation historical fork is eligible")
    if checkpoint["schedule_state"]["schedules"] != historical_metadata["schedules"]:
        raise ValueError("Historical schedule descriptor mismatch")
    if checkpoint["schedule_state"]["rows"] != len(xraw) or checkpoint["schedule_state"]["batch_size"] != config["batch_size"]:
        raise ValueError("Historical schedule rows/batches mismatch")
    schedule = historical_metadata["schedules"]["continuation"]
    if schedule["seed"] != 1280000 + 100 * seed + 200: raise ValueError("Unexpected historical continuation schedule seed")
    orders, schedule_digest = old._orders(len(xraw), config["continuation_epochs"], schedule["seed"])
    if schedule_digest != schedule["sha256"]: raise ValueError("Historical batch schedule changed")
    diagnostic_indices = np.random.default_rng(1280000+100*seed).permutation(len(xraw))[:config["batch_size"]]
    if array_digest(diagnostic_indices) != historical_metadata["diagnostic_indices_sha256"]:
        raise ValueError("Historical diagnostic minibatch changed")
    model, observers, optimizer, observer_optimizer = restore_fork(checkpoint, pre, interface, seed)
    fork_hashes = historical._hashes(model, observers, optimizer, observer_optimizer)
    if fork_hashes != historical_metadata["arms"][interface+"_"+regime]["fork_hashes"]:
        raise ValueError("Loaded fork is not the exact reviewed interface fork")
    x, xv = model.standardize(xraw), model.standardize(xvraw) if xvraw is not None else None
    if array_digest(x.numpy()) != historical_metadata["standardized_fit_sha256"]: raise ValueError("Fitting preprocessing changed")
    priors = historical.label_priors(source, attributes)
    if priors != historical_metadata["prior_entropies"]: raise ValueError("Fitting label-prior normalization changed")
    counts = copy.deepcopy(checkpoint["counts"])
    n_batch = (len(x) + config["batch_size"] - 1) // config["batch_size"]
    if counts != {"mapper_optimizer_steps": config["warm_base_epochs"] * n_batch,
                  "adversary_optimizer_steps": config["warm_adversary_epochs"] * n_batch}:
        raise ValueError("Historical optimizer counts are not the common fork")
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    rng_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(checkpoint["torch_rng_state"])
        historical._checkpoint(out/"fork.pt", model, observers, optimizer, observer_optimizer, counts,
                               "exact interface fork", schedule_state=checkpoint["schedule_state"])
        if old.tree_digest(torch.load(out/"fork.pt", map_location="cpu", weights_only=True)) != original_tree:
            raise AssertionError("Saved strength fork does not exactly reproduce complete historical state")
        idx = diagnostic_indices
        diagnostic_args = (model, observers, x[idx], old._batch(source, idx), old._batch(attributes, idx), priors, interface, regime, beta)
        fork_diagnostic = gradient_diagnostics(*diagnostic_args)
        trajectory = [curve(model, observers, x, source, attributes, priors, interface, regime, beta, 0, counts, xv, validation)]
        for epoch, order in enumerate(orders, 1):
            for start in range(0, len(order), config["batch_size"]):
                ix = order[start:start+config["batch_size"]]
                ys, ya = old._batch(source, ix), old._batch(attributes, ix)
                for _ in range(config["adversary_updates_per_mapper_step"]):
                    historical.observer_update(model, observers, observer_optimizer, x[ix], {**ys, **ya}, priors, interface)
                    counts["adversary_optimizer_steps"] += 1
                mapper_update(model, observers, optimizer, x[ix], ys, ya, priors, interface, regime, beta)
                counts["mapper_optimizer_steps"] += 1
            if epoch % config["curve_interval"] == 0 or epoch == config["continuation_epochs"]:
                trajectory.append(curve(model, observers, x, source, attributes, priors, interface, regime, beta, epoch, counts, xv, validation))
        final_schedule = copy.deepcopy(checkpoint["schedule_state"]); final_schedule["completed_epochs"] = config["continuation_epochs"]
        historical._checkpoint(out/"final.pt", model, observers, optimizer, observer_optimizer, counts,
                               "fixed final iterate", schedule_state=final_schedule)
        final_diagnostic = gradient_diagnostics(*diagnostic_args)
        if not torch.equal(checkpoint["torch_rng_state"], torch.get_rng_state()): raise AssertionError("Continuation changed deterministic training RNG")
    if not torch.equal(rng_before, torch.get_rng_state()): raise AssertionError("Continuation changed caller torch RNG")
    if old.tree_digest(checkpoint) != original_tree or hashlib.sha256(path.read_bytes()).hexdigest() != fork_sha256:
        raise AssertionError("Immutable historical checkpoint changed")
    metadata = {"study": "acs_coalition_strength_v1", "seed": int(seed), "interface": interface, "regime": regime, "beta": float(beta),
        "miniature": miniature, "config": config, "coefficients": coeff, "fit_pool": "representation_fit",
        "reserved_labels_received": False, "final_evaluation_received": False,
        "historical_fork_path": str(path), "historical_fork_sha256": fork_sha256,
        "historical_fork_tree_sha256": original_tree, "historical_training_module_sha256": historical_metadata["module_source_sha256"],
        "fork_hashes": fork_hashes, "final_hashes": historical._hashes(model, observers, optimizer, observer_optimizer),
        "counts": counts, "schedules": historical_metadata["schedules"], "schedule_sha256": schedule_digest,
        "fit_input_sha256": array_digest(xraw), "standardized_fit_sha256": array_digest(x.numpy()),
        "source_label_hashes": historical_metadata["source_label_hashes"], "attribute_label_hashes": historical_metadata["attribute_label_hashes"],
        "source_validation_input_sha256": array_digest(xvraw) if xvraw is not None else None,
        "source_validation_label_hashes": {k: array_digest(y.numpy()) for k, y in validation.items()} if validation is not None else {},
        "preprocessing": historical_metadata["preprocessing"], "prior_entropies": priors,
        "source_fit_coverage": historical_metadata["source_fit_coverage"], "attribute_fit_coverage": historical_metadata["attribute_fit_coverage"],
        "observer_roles": historical_metadata["observer_roles"][interface], "observer_optimization": historical_metadata["observer_optimization"],
        "fork_diagnostic": fork_diagnostic, "final_diagnostic": final_diagnostic, "curve": trajectory,
        "diagnostic_indices_sha256": array_digest(diagnostic_indices), "caller_rng_unchanged": True,
        "historical_fork_unchanged": True, "warmup_refitted": False, "selected_epoch": config["continuation_epochs"],
        "selection": "fixed final iterate; source validation diagnostic only", "mapper_exposure_per_row": config["warm_base_epochs"]+config["continuation_epochs"],
        "observer_exposure_per_row": config["warm_adversary_epochs"]+3*config["continuation_epochs"],
        "source_valid_label_exposures": {target: int((y>=0).sum())*(config["warm_base_epochs"]+config["continuation_epochs"]) for target, y in source.items()},
        "observer_valid_label_exposures": {role: int(({**source, **attributes})[role.split("__")[1]].ge(0).sum())*(config["warm_adversary_epochs"]+3*config["continuation_epochs"]) for role in historical.ROLE_SCHEMA},
        "module_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runtime_seconds": time.perf_counter()-started}
    (out/"training.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    return {"model": model.freeze(), "observers": observers.eval().requires_grad_(False), "metadata": metadata}
