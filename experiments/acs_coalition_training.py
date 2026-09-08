"""Fixed two-purpose feature/prediction coalition experiment.

Only the original three source tasks and two attributes enter this trainer.
Historical training modules are reused without changing their contracts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import acs_bottleneck_training as old
from experiments.acs_transfer_heads import _network
from experiments.acs_transfer_models import array_digest, state_digest

SOURCE_SCHEMA = old.SOURCE_SCHEMA
ATTRIBUTE_SCHEMA = old.ATTRIBUTE_SCHEMA
PURPOSE_TASKS = {"A": ("income_binary", "civilian_at_work"), "B": ("public_coverage",)}
TARGET_ORDER = ("SEX", "RAC1P", "income_binary", "civilian_at_work", "public_coverage", "same_residence", "commute_over20")
VIEW_INDEX = {"A": 0, "B": 1, "AB": 2}
ROLE_TARGETS = {"A": ("public_coverage", "SEX", "RAC1P"),
                "B": ("income_binary", "civilian_at_work", "SEX", "RAC1P"),
                "AB": ("SEX", "RAC1P")}
ROLE_SCHEMA = {view + "__" + target: (9 if target == "RAC1P" else 2)
               for view, targets in ROLE_TARGETS.items() for target in targets}
CONFIG = {**copy.deepcopy(old.TRAIN_CONFIG), "study": "acs_coalition_v1",
          "reconstruction_weight": 0., "preservation_weight": 0.,
          "observer_seed_base": 20260911, "source_reduction": "equal purpose mean"}
COEFFICIENTS = {"I": {"individual": -.1, "extra_local": 0., "coalition": 0.},
                "Iplus": {"individual": -.1, "extra_local": -.1, "coalition": 0.},
                "J": {"individual": -.1, "extra_local": 0., "coalition": -.1}}


class PurposeBranch(nn.Module):
    def __init__(self, original_state, tasks):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.mapper = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))
            self.heads = nn.ModuleDict({name: nn.Linear(16, 1) for name in tasks})
        self.mapper.load_state_dict({name[7:]: value.detach().clone() for name, value in original_state.items()
                                     if name.startswith("mapper.")}, strict=True)
        for name, head in self.heads.items():
            head.load_state_dict({key: original_state[f"heads.{name}.{key}"].detach().clone()
                                  for key in ("weight", "bias")}, strict=True)


class CoalitionModel(nn.Module):
    """One independently parameterized mapper and assigned heads per purpose."""
    def __init__(self, pre, original_state):
        super().__init__()
        mean, scale = (np.asarray(pre[key], np.float64) for key in ("mean", "scale"))
        if mean.shape != (32,) or scale.shape != (32,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Saved original fitting-only PCA32 statistics required")
        for key, expected in (("input_mean", mean), ("input_scale", scale)):
            if not np.array_equal(original_state[key].cpu().numpy(), expected):
                raise ValueError("Original initial checkpoint preprocessing mismatch")
            self.register_buffer(key, torch.from_numpy(expected.copy()))
        self.branches = nn.ModuleDict({purpose: PurposeBranch(original_state, tasks)
                                      for purpose, tasks in PURPOSE_TASKS.items()})

    def standardize(self, raw):
        value = old._features(raw)
        return torch.from_numpy(((value.astype(np.float64) - self.input_mean.numpy()) / self.input_scale.numpy()).astype(np.float32))

    def forward_parts(self, x):
        features, logits = {}, {}
        for purpose, branch in self.branches.items():
            features[purpose] = branch.mapper(x)
            logits[purpose] = {name: head(features[purpose]) for name, head in branch.heads.items()}
        return features, logits

    def tensor_wires(self, x, interface):
        features, logits = self.forward_parts(x)
        return make_wires(features, logits, interface)

    @torch.no_grad()
    def wires(self, raw, interface):
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("Release extraction requires a frozen paired model")
        return {name: value.numpy().astype(np.float32) for name, value in self.tensor_wires(self.standardize(raw), interface).items()}

    @torch.no_grad()
    def native_probabilities(self, raw):
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("Native predictions require a frozen paired model")
        _, logits = self.forward_parts(self.standardize(raw))
        result = {}
        for purpose in PURPOSE_TASKS:
            for name, scores in logits[purpose].items():
                p = torch.sigmoid(scores).reshape(-1)
                result[name] = torch.stack((1. - p, p), 1).numpy()
        return result

    source_probabilities = native_probabilities

    def freeze(self):
        return self.eval().requires_grad_(False)


def build_initial_model(pre, original_state):
    """Frozen, zero-step initial model for protocol identity preparation."""
    return CoalitionModel(pre, original_state).freeze()


def make_wires(features, logits, interface):
    if interface == "F":
        result = dict(features)
    elif interface == "P":
        result = {purpose: torch.cat([torch.sigmoid(logits[purpose][name]) for name in tasks], 1)
                  for purpose, tasks in PURPOSE_TASKS.items()}
    else:
        raise ValueError("Interface must be F or P")
    result["AB"] = torch.cat((result["A"], result["B"]), 1)
    return result


def source_loss(logits, source):
    losses, support = {}, {}
    for purpose, tasks in PURPOSE_TASKS.items():
        _, individual, known = old.masked_source_bce(logits[purpose], {name: source[name] for name in tasks})
        losses.update(individual); support.update(known)
    purposes = {purpose: torch.stack([losses[name] for name in tasks]).mean() for purpose, tasks in PURPOSE_TASKS.items()}
    return torch.stack(list(purposes.values())).mean(), losses, purposes, support


def label_priors(source, attributes):
    result = {}
    for target, size in {**ATTRIBUTE_SCHEMA, **SOURCE_SCHEMA}.items():
        y = ({**source, **attributes})[target].numpy()
        counts = np.bincount(y[y >= 0], minlength=size)
        p = counts / counts.sum() if counts.sum() else np.zeros(size)
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        if not np.isfinite(entropy) or entropy <= 0:
            raise ValueError("Undefined or zero fitting prior entropy for " + target)
        result[target] = {"support": counts.tolist(), "probabilities": p.tolist(), "entropy": entropy}
    return result


def make_observers(interface, seed):
    dims = {"A": 16, "B": 16, "AB": 32} if interface == "F" else {"A": 2, "B": 1, "AB": 3}
    if interface not in ("F", "P"):
        raise ValueError("Interface must be F or P")
    observers = nn.ModuleDict()
    metadata = {}
    for role, size in ROLE_SCHEMA.items():
        view, target = role.split("__")
        role_seed = 20260911 + 100 * int(seed) + 10 * VIEW_INDEX[view] + TARGET_ORDER.index(target)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(role_seed)
            observers[role] = _network(dims[view], [64, 32], size)
        metadata[role] = {"view": view, "target": target, "classes": size, "input_dim": dims[view],
                          "seed": role_seed, "initial_state_sha256": state_digest(observers[role].state_dict()),
                          "parameters": sum(p.numel() for p in observers[role].parameters())}
    return observers, metadata


def observer_losses(observers, wires, labels, priors):
    individual, support = {}, {}
    for role, observer in observers.items():
        view, target = role.split("__")
        scores, y = observer(wires[view]), labels[target]
        known = y >= 0
        individual[role] = F.cross_entropy(scores[known], y[known]) if known.any() else scores.sum() * 0.
        support[role] = int(known.sum())
    normalized = {view: torch.stack([individual[view + "__" + target] / priors[target]["entropy"]
                                    for target in targets]).mean() for view, targets in ROLE_TARGETS.items()}
    sensitive = {view: torch.stack([individual[view + "__" + target] / priors[target]["entropy"]
                                      for target in ATTRIBUTE_SCHEMA]).mean() for view in ("A", "B")}
    return {"individual": (normalized["A"] + normalized["B"]) / 2.,
            "extra_local": sensitive["A"] + sensitive["B"], "coalition": normalized["AB"]}, individual, support


def components(model, observers, x, source, attributes, priors, interface):
    features, logits = model.forward_parts(x)
    base, tasks, purposes, support = source_loss(logits, source)
    protection, observer_ce, observer_support = observer_losses(observers, make_wires(features, logits, interface), {**source, **attributes}, priors)
    return {"source": base, **protection}, {"tasks": tasks, "purposes": purposes, "source_support": support,
                                           "observer_ce": observer_ce, "observer_support": observer_support}


def observer_update(model, observers, optimizer, x, labels, priors, interface):
    model.zero_grad(set_to_none=True); optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        wires = {name: value.detach() for name, value in model.tensor_wires(x, interface).items()}
    _, losses, support = observer_losses(observers, wires, labels, priors)
    loss = torch.stack(list(losses.values())).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite observer objective")
    loss.backward(); optimizer.step()
    return support


def mapper_update(model, observers, optimizer, x, source, attributes, priors, interface, condition):
    optimizer.zero_grad(set_to_none=True); observers.zero_grad(set_to_none=True)
    flags = [p.requires_grad for p in observers.parameters()]
    observers.requires_grad_(False)
    try:
        if condition is None:
            objective, _, _, support = source_loss(model.forward_parts(x)[1], source)
        else:
            losses, detail = components(model, observers, x, source, attributes, priors, interface)
            objective = losses["source"]
            for name, coefficient in COEFFICIENTS[condition].items():
                if coefficient:
                    objective = objective + coefficient * losses[name]
            support = detail["source_support"]
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite forward-model objective")
        objective.backward(); optimizer.step()
    finally:
        for p, flag in zip(observers.parameters(), flags): p.requires_grad_(flag)
    return support


def gradient_diagnostics(model, observers, x, source, attributes, priors, interface, condition):
    before = old.tree_digest((model.state_dict(), observers.state_dict(), [p.grad for p in model.parameters()], [p.grad for p in observers.parameters()]))
    rng = torch.get_rng_state().clone()
    losses, detail = components(model, observers, x, source, attributes, priors, interface)
    coefficients = {"source": 1., **(COEFFICIENTS[condition] if condition else {"individual": 0., "extra_local": 0., "coalition": 0.})}
    groups = {purpose + "_mapper": list(branch.mapper.parameters()) for purpose, branch in model.branches.items()}
    groups.update({purpose + "_heads": list(branch.heads.parameters()) for purpose, branch in model.branches.items()})
    groups["all_mappers"] = groups["A_mapper"] + groups["B_mapper"]
    result = {"coefficients": coefficients, "condition": condition, "interface": interface,
              "losses": {name: float(value.detach()) for name, value in losses.items()},
              "source_task_losses": {name: float(value.detach()) for name, value in detail["tasks"].items()},
              "observer_ce": {name: float(value.detach()) for name, value in detail["observer_ce"].items()},
              "groups": {}}
    for group, params in groups.items():
        vectors = {}
        for name, loss in losses.items():
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grads)])
        applied = {name: coefficients[name] * vector for name, vector in vectors.items()}
        record = {}
        for convention, values in (("raw", vectors), ("applied", applied)):
            for name, vector in values.items(): record[f"{name}_{convention}_l2"] = float(torch.linalg.vector_norm(vector))
            for left, right in (("source", "individual"), ("source", "coalition"), ("source", "extra_local"), ("individual", "extra_local"), ("extra_local", "coalition"), ("individual", "coalition")):
                dot = float(torch.dot(values[left], values[right]))
                denominator = record[f"{left}_{convention}_l2"] * record[f"{right}_{convention}_l2"]
                record[f"{left}_{right}_{convention}_dot"] = dot
                record[f"{left}_{right}_{convention}_cosine"] = dot / denominator if denominator else None
        record["combined_protection_l2"] = float(torch.linalg.vector_norm(applied["individual"] + applied["extra_local"] + applied["coalition"]))
        record["objective_l2"] = float(torch.linalg.vector_norm(sum(applied.values())))
        if not all(value is None or np.isfinite(value) for value in record.values()):
            raise FloatingPointError("Nonfinite coalition gradient diagnostic")
        result["groups"][group] = record
    routing = {}
    for owner in ("A", "B"):
        local = torch.stack([detail["observer_ce"][owner + "__" + target] / priors[target]["entropy"]
                             for target in ROLE_TARGETS[owner]]).mean()
        extra = torch.stack([detail["observer_ce"][owner + "__" + target] / priors[target]["entropy"]
                             for target in ATTRIBUTE_SCHEMA]).mean()
        for destination in ("A", "B"):
            params = groups[destination + "_mapper"]
            for name, value in (("local", local), ("extra", extra)):
                grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
                norm = float(torch.sqrt(sum((g * g).sum() for g in grads if g is not None))) if any(g is not None for g in grads) else 0.
                routing[f"{name}_{owner}_to_{destination}_mapper_l2"] = norm
                if owner != destination and norm != 0.:
                    raise AssertionError("Individual observer gradient crossed the purpose boundary")
    result["routing"] = routing
    after = old.tree_digest((model.state_dict(), observers.state_dict(), [p.grad for p in model.parameters()], [p.grad for p in observers.parameters()]))
    if after != before or not torch.equal(rng, torch.get_rng_state()):
        raise AssertionError("Gradient diagnostics mutated model, gradients or RNG")
    result.update(state_unchanged=True, rng_unchanged=True)
    return result


def _checkpoint(path, model, observers, optimizer, observer_optimizer, counts, stage, *, schedule_state):
    torch.save({"model_state": model.state_dict(), "adversary_state": observers.state_dict(),
                "mapper_optimizer": optimizer.state_dict(),
                "adversary_optimizer": observer_optimizer.state_dict() if observer_optimizer else None,
                "counts": copy.deepcopy(counts), "stage": stage,
                "schedule_state": copy.deepcopy(schedule_state), "torch_rng_state": torch.get_rng_state().clone()}, path)


def _hashes(model, observers, optimizer, observer_optimizer):
    return {"model": state_digest(model.state_dict()), "adversaries": state_digest(observers.state_dict()),
            "mapper_optimizer": old.tree_digest(optimizer.state_dict()),
            "adversary_optimizer": old.tree_digest(observer_optimizer.state_dict()) if observer_optimizer else None}


@torch.no_grad()
def _curve(model, observers, x, source, attributes, priors, interface, condition, epoch, counts, xv=None, validation=None):
    base, task, purpose, _ = source_loss(model.forward_parts(x)[1], source)
    result = {"epoch": epoch, "counts": counts.copy(), "source_loss": float(base),
              "task_losses": {name: float(value) for name, value in task.items()},
              "purpose_losses": {name: float(value) for name, value in purpose.items()},
              "coefficients": {"source": 1., **(COEFFICIENTS[condition] if condition else {"individual": 0., "extra_local": 0., "coalition": 0.})}}
    if len(observers):
        losses, detail = components(model, observers, x, source, attributes, priors, interface)
        result.update(individual_loss=float(losses["individual"]), extra_local_loss=float(losses["extra_local"]), coalition_loss=float(losses["coalition"]),
                      observer_ce={name: float(value) for name, value in detail["observer_ce"].items()},
                      objective=float(sum(result["coefficients"][name] * value for name, value in losses.items())))
    else:
        result.update(individual_loss=None, extra_local_loss=None, coalition_loss=None, observer_ce={}, objective=float(base))
    if xv is not None:
        _, tasks, _, _ = source_loss(model.forward_parts(xv)[1], validation)
        result["source_validation"] = {name: float(value) for name, value in tasks.items()}
    return result


def train_seed(pca_fit, source_y, attr_y, pre, original_state, cfg, seed, out, *, miniature=False, pca_val=None, source_val=None):
    """Fit one common warmup and all six predeclared continuations, no audits."""
    started = time.perf_counter()
    if seed not in (0, 1, 2): raise ValueError("Frozen seeds are 0,1,2")
    xraw = old._features(pca_fit)
    source = old._labels(source_y, len(xraw), SOURCE_SCHEMA, "source fitting")
    attributes = old._labels(attr_y, len(xraw), ATTRIBUTE_SCHEMA, "attribute fitting")
    if (pca_val is None) != (source_val is None): raise ValueError("Both source-validation inputs and labels required")
    xvraw = old._features(pca_val) if pca_val is not None else None
    validation = old._labels(source_val, len(xvraw), SOURCE_SCHEMA, "source validation") if xvraw is not None else None
    config = copy.deepcopy(CONFIG)
    supplied = cfg.get("training", cfg) if cfg else {}
    for key in old.TRAIN_CONFIG:
        if key in supplied and supplied[key] != config[key]: raise ValueError("Frozen training setting differs: " + key)
    aliases = {"learning_rate": "lr", "warmup_epochs": "warm_base_epochs",
               "adversary_warmup_epochs": "warm_adversary_epochs", "adversary_updates_per_mapper": "adversary_updates_per_mapper_step"}
    for key, canonical in aliases.items():
        if key in supplied and supplied[key] != config[canonical]: raise ValueError("Frozen training setting differs: " + key)
    if cfg and "objectives" in cfg and cfg["objectives"] != COEFFICIENTS:
        raise ValueError("Frozen coalition objective coefficients differ")
    if cfg and "observer_roles" in cfg and cfg["observer_roles"] != {key: list(value) for key, value in ROLE_TARGETS.items()}:
        raise ValueError("Frozen observer roles differ")
    if miniature: config.update(warm_base_epochs=1, warm_adversary_epochs=1, continuation_epochs=2, batch_size=16, curve_interval=1)
    raw64 = xraw.astype(np.float64)
    expected = {"mean": raw64.mean(0), "scale": np.where(raw64.std(0) > 1e-12, raw64.std(0), 1.)}
    if any(not np.array_equal(np.asarray(pre[key]), expected[key]) for key in expected):
        raise ValueError("Preprocessing is not the saved fitting-only original statistics")
    model = CoalitionModel(pre, original_state)
    x, xv = model.standardize(xraw), model.standardize(xvraw) if xvraw is not None else None
    priors = label_priors(source, attributes)
    schedules = {}
    for phase, epochs, offset in (("warm_base", config["warm_base_epochs"], 0), ("warm_adversary", config["warm_adversary_epochs"], 100), ("continuation", config["continuation_epochs"], 200)):
        schedule_seed = 1280000 + 100 * int(seed) + offset
        orders, digest = old._orders(len(x), epochs, schedule_seed)
        schedules[phase] = {"seed": schedule_seed, "sha256": digest, "orders": orders}
    diagnostic_indices = schedules["warm_base"]["orders"][0][:config["batch_size"]]
    def save_checkpoint(path, current, observers, optimizer, observer_optimizer, counts, stage):
        positions = {"true initialization": ("warm_base", 0), "common source warmup": ("warm_base", config["warm_base_epochs"]),
                     "before observer warmup": ("warm_adversary", 0), "shared interface fork": ("warm_adversary", config["warm_adversary_epochs"]),
                     "exact interface fork": ("continuation", 0), "fixed final iterate": ("continuation", config["continuation_epochs"])}
        phase, complete_epochs = positions[stage]
        schedule_state = {"recipe": "all epoch permutations materialized before fitting; no live NumPy RNG consumed by updates",
            "schedules": {p: {key: value[key] for key in ("seed", "sha256")} for p, value in schedules.items()},
            "phase": phase, "completed_epochs": complete_epochs, "next_minibatch_index": 0,
            "batch_size": config["batch_size"], "rows": len(x)}
        _checkpoint(path, current, observers, optimizer, observer_optimizer, counts, stage, schedule_state=schedule_state)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    empty = nn.ModuleDict(); optimizer = old._adam(model.parameters())
    counts = {"mapper_optimizer_steps": 0, "adversary_optimizer_steps": 0}
    snapshots = {"I": copy.deepcopy(model).freeze()}
    initial_hash = state_digest(model.state_dict())
    parity = {}
    for pool, raw in (("representation_fit", xraw), ("source_validation", xvraw)):
        if raw is None: continue
        parity[pool] = {}
        for purpose, values in snapshots["I"].wires(raw, "F").items():
            if purpose == "AB": continue
            delta = values.astype(np.float64) - raw[:, :16]
            passed = bool(np.allclose(values, raw[:, :16], atol=1e-5, rtol=1e-5))
            if not passed: raise AssertionError("Original raw PCA16 initialization parity failed")
            parity[pool][purpose] = {"max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta**2))), "passed": passed}
    save_checkpoint(out / "initialization.pt", model, empty, optimizer, None, counts, "true initialization")
    diagnostics = {"I": {}, "W": {}}
    def diagnostic(current, observers, interface, condition, hypothetical=False):
        idx = diagnostic_indices
        result = gradient_diagnostics(current, observers, x[idx], old._batch(source, idx), old._batch(attributes, idx), priors, interface, condition)
        result.update(hypothetical_preobserver_warmup=hypothetical, observer_state_sha256=state_digest(observers.state_dict()))
        return result
    for interface in ("F", "P"):
        hypothetical, _ = make_observers(interface, seed)
        diagnostics["I"][interface] = diagnostic(model, hypothetical, interface, None, True)
    warm_curve = [_curve(model, empty, x, source, attributes, priors, "F", None, 0, counts, xv, validation)]
    for epoch, order in enumerate(schedules["warm_base"]["orders"], 1):
        for start in range(0, len(order), config["batch_size"]):
            ix = order[start:start+config["batch_size"]]
            mapper_update(model, empty, optimizer, x[ix], old._batch(source, ix), old._batch(attributes, ix), priors, "F", None)
            counts["mapper_optimizer_steps"] += 1
        if epoch % config["curve_interval"] == 0 or epoch == config["warm_base_epochs"]:
            warm_curve.append(_curve(model, empty, x, source, attributes, priors, "F", None, epoch, counts, xv, validation))
    save_checkpoint(out / "warm_base.pt", model, empty, optimizer, None, counts, "common source warmup")
    snapshots["W"] = copy.deepcopy(model).freeze()
    warm_hash = state_digest(model.state_dict())
    warm_optimizer_hash = old.tree_digest(optimizer.state_dict())
    arms, arm_metadata, observer_metadata = {}, {}, {}
    n_batch = math.ceil(len(x) / config["batch_size"])
    for interface in ("F", "P"):
        shared = copy.deepcopy(model)
        shared_optimizer = old._adam(shared.parameters()); shared_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        observers, observer_metadata[interface] = make_observers(interface, seed)
        observer_optimizer = old._adam(observers.parameters())
        diagnostics["W"][interface] = diagnostic(shared, observers, interface, None, True)
        local_counts = counts.copy()
        interface_dir = out / interface; interface_dir.mkdir()
        save_checkpoint(interface_dir / "observer_initialization.pt", shared, observers, shared_optimizer, observer_optimizer, local_counts, "before observer warmup")
        observer_curve = [_curve(shared, observers, x, source, attributes, priors, interface, None, 0, local_counts, xv, validation)]
        for epoch, order in enumerate(schedules["warm_adversary"]["orders"], 1):
            for start in range(0, len(order), config["batch_size"]):
                ix = order[start:start+config["batch_size"]]
                observer_update(shared, observers, observer_optimizer, x[ix], {**old._batch(source, ix), **old._batch(attributes, ix)}, priors, interface)
                local_counts["adversary_optimizer_steps"] += 1
            if epoch % config["curve_interval"] == 0 or epoch == config["warm_adversary_epochs"]:
                observer_curve.append(_curve(shared, observers, x, source, attributes, priors, interface, None, epoch, local_counts, xv, validation))
        if state_digest(shared.state_dict()) != warm_hash or old.tree_digest(shared_optimizer.state_dict()) != warm_optimizer_hash:
            raise AssertionError("Observer warmup changed forward model or Adam")
        save_checkpoint(interface_dir / "warm_adversary.pt", shared, observers, shared_optimizer, observer_optimizer, local_counts, "shared interface fork")
        shared_hashes = _hashes(shared, observers, shared_optimizer, observer_optimizer)
        diagnostics[interface + "_fork"] = diagnostic(shared, observers, interface, "J")
        diagnostics[interface + "_fork"]["prospective_J_objective"] = True
        for condition in ("I", "Iplus", "J"):
            arm, adversaries = copy.deepcopy(shared), copy.deepcopy(observers)
            arm_optimizer, adversary_optimizer = old._adam(arm.parameters()), old._adam(adversaries.parameters())
            arm_optimizer.load_state_dict(copy.deepcopy(shared_optimizer.state_dict()))
            adversary_optimizer.load_state_dict(copy.deepcopy(observer_optimizer.state_dict()))
            if _hashes(arm, adversaries, arm_optimizer, adversary_optimizer) != shared_hashes:
                raise AssertionError("Inexact continuation fork")
            name = interface + "_" + condition; directory = out / name; directory.mkdir()
            current_counts = local_counts.copy()
            save_checkpoint(directory / "fork.pt", arm, adversaries, arm_optimizer, adversary_optimizer, current_counts, "exact interface fork")
            fork_diagnostic = diagnostic(arm, adversaries, interface, condition)
            curve = [_curve(arm, adversaries, x, source, attributes, priors, interface, condition, 0, current_counts, xv, validation)]
            for epoch, order in enumerate(schedules["continuation"]["orders"], 1):
                for start in range(0, len(order), config["batch_size"]):
                    ix = order[start:start+config["batch_size"]]
                    ys, ya = old._batch(source, ix), old._batch(attributes, ix)
                    for _ in range(config["adversary_updates_per_mapper_step"]):
                        observer_update(arm, adversaries, adversary_optimizer, x[ix], {**ys, **ya}, priors, interface)
                        current_counts["adversary_optimizer_steps"] += 1
                    mapper_update(arm, adversaries, arm_optimizer, x[ix], ys, ya, priors, interface, condition)
                    current_counts["mapper_optimizer_steps"] += 1
                if epoch % config["curve_interval"] == 0 or epoch == config["continuation_epochs"]:
                    curve.append(_curve(arm, adversaries, x, source, attributes, priors, interface, condition, epoch, current_counts, xv, validation))
            save_checkpoint(directory / "final.pt", arm, adversaries, arm_optimizer, adversary_optimizer, current_counts, "fixed final iterate")
            arm_metadata[name] = {"interface": interface, "condition": condition, "coefficients": COEFFICIENTS[condition],
                "fork_hashes": shared_hashes, "final_hashes": _hashes(arm, adversaries, arm_optimizer, adversary_optimizer),
                "counts": current_counts, "schedule_sha256": schedules["continuation"]["sha256"],
                "fork_diagnostic": fork_diagnostic, "final_diagnostic": diagnostic(arm, adversaries, interface, condition),
                "curve": curve, "observer_warmup_curve": observer_curve,
                "mapper_exposure_per_row": config["warm_base_epochs"] + config["continuation_epochs"],
                "observer_exposure_per_row": config["warm_adversary_epochs"] + 3 * config["continuation_epochs"],
                "source_valid_label_exposures": {target: int((y >= 0).sum()) * (config["warm_base_epochs"] + config["continuation_epochs"]) for target, y in source.items()},
                "observer_valid_label_exposures": {role: int(({**source, **attributes})[role.split("__")[1]].ge(0).sum()) * (config["warm_adversary_epochs"] + 3 * config["continuation_epochs"]) for role in ROLE_SCHEMA},
                "selected_epoch": config["continuation_epochs"], "selection": "fixed final iterate"}
            arms[name] = {"model": arm.freeze(), "observers": adversaries.eval().requires_grad_(False), "adversaries": adversaries}
    for name, expected_hash in (("I", initial_hash), ("W", warm_hash)):
        if state_digest(snapshots[name].state_dict()) != expected_hash: raise AssertionError("Initial/warm snapshot mutated")
    metadata = {"config": config, "seed": int(seed), "miniature": miniature,
        "fit_pool": "representation_fit", "reserved_labels_received": False, "final_evaluation_received": False,
        "preprocessing": {"mean": list(pre["mean"]), "scale": list(pre["scale"]), "fit_rows": len(x)},
        "fit_input_sha256": array_digest(xraw), "standardized_fit_sha256": array_digest(x.numpy()),
        "source_label_hashes": {name: array_digest(y.numpy()) for name, y in source.items()},
        "attribute_label_hashes": {name: array_digest(y.numpy()) for name, y in attributes.items()},
        "source_validation_input_sha256": array_digest(xvraw) if xvraw is not None else None,
        "source_validation_label_hashes": {name: array_digest(y.numpy()) for name, y in validation.items()} if validation is not None else {},
        "source_fit_coverage": old._coverage(source, SOURCE_SCHEMA), "attribute_fit_coverage": old._coverage(attributes, ATTRIBUTE_SCHEMA),
        "prior_entropies": priors, "initialization_parity": parity, "initial_model_sha256": initial_hash,
        "warm_model_sha256": warm_hash, "warm_optimizer_sha256": warm_optimizer_hash,
        "original_model_sha256": state_digest(original_state), "observer_roles": observer_metadata,
        "observer_optimization": "fixed mean of nine ordinary masked role CEs; normalization only in reversed forward losses",
        "schedules": {phase: {key: item[key] for key in ("seed", "sha256")} for phase, item in schedules.items()},
        "diagnostic_indices_sha256": array_digest(diagnostic_indices), "diagnostics": diagnostics,
        "shared_source_curve": warm_curve, "source_validation_role": "diagnostic only; never selects any fitted state",
        "arms": arm_metadata, "forward_parameters": sum(p.numel() for p in model.parameters()),
        "batches_per_epoch": n_batch, "runtime_seconds": time.perf_counter() - started,
        "module_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "training.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return {"arms": arms, "snapshots": snapshots, "metadata": metadata}
