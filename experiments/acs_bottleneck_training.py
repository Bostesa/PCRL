"""Matched fixed-budget ACS bottleneck and adversarial continuation.

Only the three declared source labels and protected labels from representation
fitting enter optimization. Source validation is diagnostic; neither checkpoints
nor hyperparameters are selected from it. Reserved transfer labels are rejected.
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

from experiments.acs_transfer_heads import _network
from experiments.acs_transfer_models import array_digest, state_digest


SOURCE_SCHEMA = {"income_binary": 2, "civilian_at_work": 2, "public_coverage": 2}
ATTRIBUTE_SCHEMA = {"SEX": 2, "RAC1P": 9}
TRAIN_CONFIG = {
    "input_dim": 32, "mapper_hidden": 64, "release_dim": 16,
    "decoder_hidden": 64, "adversary_hidden": [64, 32],
    "warm_base_epochs": 60, "warm_adversary_epochs": 20, "continuation_epochs": 80,
    "batch_size": 256, "adversary_updates_per_mapper_step": 3,
    "lr": .001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.,
    "reconstruction_weight": .1, "protection_weight": .1,
    "curve_interval": 5, "standardizer_std_floor": 1e-12,
    "gradient_clipping": None, "checkpoint_selection": "fixed final iterate",
}


def tree_digest(value):
    """Stable hash of nested Adam/model state, including tensors and counters."""
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor" + array_digest(item.detach().cpu().numpy()).encode())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                update(child)
        else:
            digest.update((type(item).__name__ + ":" + repr(item)).encode())
    update(value)
    return digest.hexdigest()


def _features(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 32 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Nonempty finite PCA32 feature matrices are required")
    return np.ascontiguousarray(array)


def _labels(value, n, schema, name):
    if set(value) != set(schema):
        raise ValueError(f"{name} whitelist is exactly {tuple(schema)}")
    result = {}
    for target, k in schema.items():
        array = value[target]
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        array = np.asarray(array)
        if (array.shape != (n,) or not np.issubdtype(array.dtype, np.integer)
                or ((array < -1) | (array >= k)).any()):
            raise ValueError(f"{name}/{target}: fixed zero-based labels and -1 missing required")
        result[target] = torch.from_numpy(np.ascontiguousarray(array, dtype=np.int64))
    return result


def _coverage(labels, schema):
    result = {}
    for target, k in schema.items():
        y = labels[target].numpy()
        counts = np.bincount(y[y >= 0], minlength=k)
        result[target] = {"class_schema": list(range(k)), "support": counts.tolist(),
                          "known": int((y >= 0).sum()), "missing": int((y < 0).sum()),
                          "valid_class_mask": ((counts > 0) & (counts < counts.sum())).tolist(),
                          "schema_complete": bool((counts > 0).all())}
    return result


def prior_entropies(labels):
    """Empirical, masked fitting priors; no smoothing or validation estimates."""
    result = {}
    for target, k in ATTRIBUTE_SCHEMA.items():
        y = labels[target].numpy()
        counts = np.bincount(y[y >= 0], minlength=k).astype(np.float64)
        if counts.sum() == 0:
            raise ValueError(f"Undefined fitting prior entropy: {target}")
        p = counts / counts.sum()
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        if not np.isfinite(entropy) or entropy <= 0:
            raise ValueError(f"Zero or undefined fitting prior entropy: {target}")
        result[target] = {"support": counts.astype(int).tolist(), "probabilities": p.tolist(),
                          "entropy": entropy, "estimator": "masked representation-fit empirical frequencies"}
    return result


def masked_mean_ce(logits, labels):
    """Equal fixed-head mean; an all-missing minibatch head contributes zero."""
    losses, known = {}, {}
    for target, scores in logits.items():
        valid = labels[target] >= 0
        known[target] = int(valid.sum())
        losses[target] = (F.cross_entropy(scores[valid], labels[target][valid])
                          if known[target] else scores.sum() * 0.)
    return torch.stack(list(losses.values())).mean(), losses, known


def masked_source_bce(logits, labels):
    """Three independent one-logit binary heads, with a fixed /3 mean."""
    losses, known = {}, {}
    for target, scores in logits.items():
        scores = scores.reshape(-1)
        valid = labels[target] >= 0
        known[target] = int(valid.sum())
        losses[target] = (F.binary_cross_entropy_with_logits(scores[valid], labels[target][valid].float())
                          if known[target] else scores.sum() * 0.)
    return torch.stack(list(losses.values())).mean(), losses, known


class BottleneckModel(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(mean, dtype=torch.float64).clone())
        self.register_buffer("input_scale", torch.as_tensor(scale, dtype=torch.float64).clone())
        self.mapper = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))
        self.heads = nn.ModuleDict({name: nn.Linear(16, 1) for name in SOURCE_SCHEMA})
        self.decoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 32))

    def standardize(self, x):
        array = _features(x).astype(np.float64)
        return torch.from_numpy(((array - self.input_mean.cpu().numpy())
                                 / self.input_scale.cpu().numpy()).astype(np.float32))

    @torch.no_grad()
    def release(self, x):
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("Release extraction requires the frozen final model")
        return self.mapper(self.standardize(x)).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def source_probabilities(self, x):
        h = self.mapper(self.standardize(x))
        result = {}
        for name, head in self.heads.items():
            positive = torch.sigmoid(head(h)).reshape(-1)
            result[name] = torch.stack([1 - positive, positive], 1).cpu().numpy()
        return result

    probabilities = source_probabilities

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self


@torch.no_grad()
def initialize_pca16_mapper(model):
    """Overwrite only the mapper to undo standardization and select raw PCA16.

    For z=(x-mean)/scale, ReLU([I;-I]z) and readout
    [S diag(scale), -S diag(scale)] with bias S mean return S x.
    S selects the first16 coordinates. Every parameter remains trainable,
    including the initially zero readout columns for the other16 coordinates.
    """
    before_rng = torch.get_rng_state().clone()
    nonmapper = {k: v for k, v in model.state_dict().items() if not k.startswith("mapper.")}
    before_other = state_digest(nonmapper)
    first, last = model.mapper[0], model.mapper[2]
    identity = torch.eye(32, dtype=first.weight.dtype, device=first.weight.device)
    first.weight.copy_(torch.cat((identity, -identity), 0))
    first.bias.zero_()
    readout = torch.diag(model.input_scale.to(last.weight))[:16]
    last.weight.copy_(torch.cat((readout, -readout), 1))
    last.bias.copy_(model.input_mean[:16].to(last.bias))
    unchanged_other = state_digest(nonmapper) == before_other
    unchanged_rng = torch.equal(before_rng, torch.get_rng_state())
    if not unchanged_other or not unchanged_rng:
        raise AssertionError("PCA16 initialization changed nonmapper state or RNG")
    return {"mode": "pca16", "formula": "W1=[I32;-I32], b1=0; W2=[S diag(scale),-S diag(scale)], b2=S mean; S selects first16",
            "nonmapper_initial_state_sha256": before_other,
            "nonmapper_unchanged_by_overwrite": unchanged_other,
            "torch_rng_unchanged_by_overwrite": unchanged_rng,
            "unused_readout_columns": list(range(16, 32)) + list(range(48, 64)),
            "all_mapper_parameters_trainable": all(p.requires_grad for p in model.mapper.parameters())}


def _initialization_diagnostics(model, xraw, xvraw, xbatch, source_batch):
    """Pre-update parity and a disposable gradient check; no fitting or RNG use."""
    before = state_digest(model.state_dict())
    frozen = copy.deepcopy(model).freeze()
    parity = {}
    for pool, raw in (("representation_fit", xraw), ("source_validation", xvraw)):
        output, target = frozen.release(raw), raw[:, :16]
        delta = output.astype(np.float64) - target.astype(np.float64)
        passed = bool(np.allclose(output, target, atol=1e-5, rtol=1e-5))
        parity[pool] = {"rows": len(raw), "atol": 1e-5, "rtol": 1e-5,
                        "max_absolute_error": float(np.abs(delta).max()),
                        "rms_error": float(np.sqrt(np.mean(delta * delta))), "passed": passed,
                        "input_sha256": array_digest(raw), "output_sha256": array_digest(output),
                        "target_pca16_sha256": array_digest(np.ascontiguousarray(target))}
        if not passed:
            raise AssertionError("Pre-update PCA16 parity failed on " + pool)
    disposable = copy.deepcopy(model)
    base, _, detail = base_loss(disposable, xbatch, source_batch)
    unused = list(range(16, 32)) + list(range(48, 64))
    norms = {}
    for name, loss in (("source", detail["source_loss"]), ("base", base)):
        gradient, = torch.autograd.grad(loss, disposable.mapper[2].weight, retain_graph=True)
        part = gradient[:, unused]
        norms[name + "_unused_readout_gradient_l2"] = float(torch.linalg.vector_norm(part))
        norms[name + "_unused_readout_nonzero_entries"] = int(torch.count_nonzero(part))
    if not np.isfinite(norms["base_unused_readout_gradient_l2"]) or norms["base_unused_readout_gradient_l2"] <= 0:
        raise FloatingPointError("Initially unused PCA16 readout cannot learn")
    if state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("Disposable initialization check changed the actual model")
    return {"parity": parity, "unused_readout_gradient_check": {**norms,
            "disposable_model": True, "actual_model_unchanged": True, "optimizer_steps": 0}}


def base_loss(model, x, source):
    h = model.mapper(x)
    task, task_losses, support = masked_source_bce({k: head(h) for k, head in model.heads.items()}, source)
    reconstruction = F.mse_loss(model.decoder(h), x)
    return task + .1 * reconstruction, h, {
        "source_loss": task, "reconstruction_mse": reconstruction,
        "source_losses": task_losses, "source_support": support,
    }


def protection_loss(adversaries, h, attributes, priors):
    plain, individual, support = masked_mean_ce({k: net(h) for k, net in adversaries.items()}, attributes)
    normalized = torch.stack([individual[k] / priors[k]["entropy"] for k in ATTRIBUTE_SCHEMA]).mean()
    return normalized, {"plain_adversary_loss": plain, "individual": individual, "attribute_support": support}


def _adam(parameters):
    return torch.optim.Adam(parameters, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)


def adversary_update(model, adversaries, optimizer, x, attributes):
    """Detached representation update: only adversary parameters receive gradients."""
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        h = model.mapper(x)
    loss, _, support = masked_mean_ce({k: net(h.detach()) for k, net in adversaries.items()}, attributes)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training adversary loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach()), support


def mapper_update(model, adversaries, optimizer, x, source, attributes, priors, protected):
    optimizer.zero_grad(set_to_none=True)
    adversaries.zero_grad(set_to_none=True)
    states = [p.requires_grad for p in adversaries.parameters()]
    adversaries.requires_grad_(False)
    try:
        base, h, detail = base_loss(model, x, source)
        if protected:
            normalized, _ = protection_loss(adversaries, h, attributes, priors)
            objective = base - .1 * normalized
        else:
            normalized = base.detach() * 0.
            objective = base
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite mapper objective")
        objective.backward()
        optimizer.step()
    finally:
        for parameter, requires_grad in zip(adversaries.parameters(), states):
            parameter.requires_grad_(requires_grad)
    return {"base_loss": float(base.detach()), "normalized_adversary_ce": float(normalized.detach()),
            "objective": float(objective.detach()), "source_support": detail["source_support"]}


def gradient_diagnostics(model, adversaries, x, source, attributes, priors):
    """Measure separate mapper gradients without stepping or changing state."""
    base, h, parts = base_loss(model, x, source)
    normalized, _ = protection_loss(adversaries, h, attributes, priors)
    parameters = list(model.mapper.parameters())
    result = {}
    for name, value in (("source", parts["source_loss"]), ("reconstruction", .1 * parts["reconstruction_mse"]),
                        ("base", base), ("protection", -.1 * normalized)):
        gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        result[name + "_mapper_l2"] = float(torch.sqrt(sum((g * g).sum() for g in gradients if g is not None)))
        result[name + "_first_weight_l2"] = float(torch.linalg.vector_norm(gradients[0]))
    result.update(base_loss=float(base.detach()), normalized_adversary_ce=float(normalized.detach()),
                  protected_objective=float((base - .1 * normalized).detach()))
    other = list(model.heads.parameters()) + list(model.decoder.parameters())
    outside = torch.autograd.grad(-.1 * normalized, other, retain_graph=True, allow_unused=True)
    if any(g is not None and torch.count_nonzero(g) for g in outside):
        raise AssertionError("Protection term unexpectedly updates source heads or decoder")
    result["protection_source_decoder_gradient_l2"] = 0.
    result["protection_source_decoder_gradients_all_absent"] = all(g is None for g in outside)
    if not all(np.isfinite(result[k]) and result[k] > 0 for k in ("base_first_weight_l2", "protection_first_weight_l2")):
        raise FloatingPointError("Broken or nonfinite base/protection gradient path")
    return result


def _batch(labels, indices):
    return {k: value[indices] for k, value in labels.items()}


def _orders(n, epochs, seed):
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(n) for _ in range(epochs)]
    digest = hashlib.sha256()
    for order in orders:
        digest.update(order.tobytes())
    return orders, digest.hexdigest()


@torch.no_grad()
def _curve(model, adversaries, x, source, attributes, xv, source_val, priors, epoch, counts):
    base, h, detail = base_loss(model, x, source)
    if len(adversaries):
        normalized, attribute_detail = protection_loss(adversaries, h, attributes, priors)
        attribute_losses = {k: float(v) for k, v in attribute_detail["individual"].items()}
        normalized = float(normalized)
    else:
        attribute_losses, normalized = None, None
    valbase, _, valdetail = base_loss(model, xv, source_val)
    scalar_tasks = lambda d: {k: float(v) if d["source_support"][k] else None for k, v in d["source_losses"].items()}
    return {"epoch": epoch, **counts, "fit_base_loss": float(base),
            "fit_source_ce": float(detail["source_loss"]), "fit_source_losses": scalar_tasks(detail),
            "fit_reconstruction_mse": float(detail["reconstruction_mse"]),
            "fit_adversary_ce": attribute_losses,
            "fit_normalized_adversary_ce": normalized,
            "source_validation_base_loss": float(valbase), "source_validation_ce": float(valdetail["source_loss"]),
            "source_validation_losses": scalar_tasks(valdetail),
            "source_validation_reconstruction_mse": float(valdetail["reconstruction_mse"]),
            "selection": "none; fixed final iterate"}


def _checkpoint(path, model, adversaries, mapper_optimizer, adversary_optimizer, counters, stage):
    if path.exists():
        raise FileExistsError(path)
    torch.save({"model_state": model.state_dict(), "adversary_state": adversaries.state_dict(),
                "mapper_optimizer_state": mapper_optimizer.state_dict(),
                "adversary_optimizer_state": adversary_optimizer.state_dict() if adversary_optimizer is not None else None,
                "counters": counters, "stage": stage}, path)


def train_pair(xfit, source_y, attr_y, xval, source_val, seed, directory, *, miniature=False,
               fit_pool="representation_fit", initialization="random"):
    if fit_pool != "representation_fit":
        raise ValueError("Training inputs must come from representation_fit")
    if initialization not in ("random", "pca16"):
        raise ValueError("Initialization must be random or pca16")
    xraw, xvraw = _features(xfit), _features(xval)
    source = _labels(source_y, len(xraw), SOURCE_SCHEMA, "source fitting")
    attributes = _labels(attr_y, len(xraw), ATTRIBUTE_SCHEMA, "attribute fitting")
    validation = _labels(source_val, len(xvraw), SOURCE_SCHEMA, "source validation")
    priors = prior_entropies(attributes)
    if any(not (y >= 0).any() for y in source.values()):
        raise ValueError("A source fitting task has no known labels")
    cfg = copy.deepcopy(TRAIN_CONFIG)
    if miniature:
        cfg.update(warm_base_epochs=1, warm_adversary_epochs=1, continuation_epochs=2,
                   batch_size=16, curve_interval=1)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    raw64 = xraw.astype(np.float64)
    mean, std = raw64.mean(0), raw64.std(0)
    scale = np.where(std > cfg["standardizer_std_floor"], std, 1.)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1270000 + 100 * int(seed))
        model = BottleneckModel(mean, scale)
    initialization_metadata = (initialize_pca16_mapper(model) if initialization == "pca16"
                               else {"mode": "random", "recipe": "historical full-model constructor unchanged"})
    snapshots = {"I": copy.deepcopy(model).freeze()} if initialization == "pca16" else {}
    adversaries = nn.ModuleDict()
    x, xv = model.standardize(xraw), model.standardize(xvraw)
    mapper_optimizer, adversary_optimizer = _adam(model.parameters()), None
    schedules = {}
    for phase, epochs, offset in (("warm_base", cfg["warm_base_epochs"], 0),
                                  ("warm_adversary", cfg["warm_adversary_epochs"], 100),
                                  ("continuation", cfg["continuation_epochs"], 200)):
        schedule_seed = 1280000 + 100 * int(seed) + offset
        orders, digest = _orders(len(x), epochs, schedule_seed)
        schedules[phase] = {"orders": orders, "seed": schedule_seed, "sha256": digest}
    if initialization == "pca16":
        first = schedules["warm_base"]["orders"][0][:cfg["batch_size"]]
        initialization_metadata.update(_initialization_diagnostics(model, xraw, xvraw, x[first], _batch(source, first)))
    n_batch = math.ceil(len(x) / cfg["batch_size"])
    counters = {"mapper_optimizer_steps": 0, "adversary_optimizer_steps": 0}
    initial_hashes = {"model": state_digest(model.state_dict()), "adversaries": None}
    _checkpoint(out / "initialization.pt", model, adversaries, mapper_optimizer, adversary_optimizer,
                {**counters, "epoch": 0}, "true initialization")
    curves, missing_batches = {}, {}
    for phase in ("warm_base", "warm_adversary"):
        if phase == "warm_adversary":
            # Literal phase order: adversaries do not exist during common base
            # training; their dedicated RNG starts after the final warm epoch.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1290000 + 100 * int(seed))
                adversaries = nn.ModuleDict({k: _network(16, [64, 32], size) for k, size in ATTRIBUTE_SCHEMA.items()})
            adversary_optimizer = _adam(adversaries.parameters())
            adversary_initial_hash = state_digest(adversaries.state_dict())
        curves[phase] = [_curve(model, adversaries, x, source, attributes, xv, validation, priors, 0, counters)]
        missing_batches[phase] = {k: 0 for k in (SOURCE_SCHEMA if phase == "warm_base" else ATTRIBUTE_SCHEMA)}
        model_before = state_digest(model.state_dict())
        for epoch, order in enumerate(schedules[phase]["orders"], 1):
            for start in range(0, len(order), cfg["batch_size"]):
                indices = order[start:start + cfg["batch_size"]]
                if phase == "warm_base":
                    detail = mapper_update(model, adversaries, mapper_optimizer, x[indices], _batch(source, indices),
                                           _batch(attributes, indices), priors, False)
                    counts = detail["source_support"]
                    counters["mapper_optimizer_steps"] += 1
                else:
                    _, counts = adversary_update(model, adversaries, adversary_optimizer, x[indices], _batch(attributes, indices))
                    counters["adversary_optimizer_steps"] += 1
                for k, count in counts.items():
                    missing_batches[phase][k] += int(count == 0)
            if epoch % cfg["curve_interval"] == 0 or epoch == len(schedules[phase]["orders"]):
                curves[phase].append(_curve(model, adversaries, x, source, attributes, xv, validation, priors, epoch, counters))
        if phase == "warm_adversary" and state_digest(model.state_dict()) != model_before:
            raise AssertionError("Adversary warm-up changed the mapper, decoder or source heads")
        _checkpoint(out / (phase + ".pt"), model, adversaries, mapper_optimizer, adversary_optimizer,
                    {**counters, "epoch": len(schedules[phase]["orders"])}, phase)
        if phase == "warm_base" and initialization == "pca16":
            snapshots["W"] = copy.deepcopy(model).freeze()
    common_hashes = {"model": state_digest(model.state_dict()), "adversaries": state_digest(adversaries.state_dict()),
                     "mapper_optimizer": tree_digest(mapper_optimizer.state_dict()),
                     "adversary_optimizer": tree_digest(adversary_optimizer.state_dict())}
    first = schedules["continuation"]["orders"][0][:cfg["batch_size"]]
    diagnostics = gradient_diagnostics(model, adversaries, x[first], _batch(source, first), _batch(attributes, first), priors)
    arms, arm_metadata = {}, {}
    for name, protected in (("C_bottleneck", False), ("D_protected", True)):
        arm, attackers = copy.deepcopy(model), copy.deepcopy(adversaries)
        optimizer, attacker_optimizer = _adam(arm.parameters()), _adam(attackers.parameters())
        optimizer.load_state_dict(copy.deepcopy(mapper_optimizer.state_dict()))
        attacker_optimizer.load_state_dict(copy.deepcopy(adversary_optimizer.state_dict()))
        cloned = {"model": state_digest(arm.state_dict()), "adversaries": state_digest(attackers.state_dict()),
                  "mapper_optimizer": tree_digest(optimizer.state_dict()),
                  "adversary_optimizer": tree_digest(attacker_optimizer.state_dict())}
        if cloned != common_hashes:
            raise AssertionError("Continuation fork did not clone model and Adam states exactly")
        arm_dir = out / name
        arm_dir.mkdir()
        count = counters.copy()
        _checkpoint(arm_dir / "fork.pt", arm, attackers, optimizer, attacker_optimizer,
                    {**count, "continuation_epoch": 0}, "shared warmed fork")
        curve = [_curve(arm, attackers, x, source, attributes, xv, validation, priors, 0, count)]
        empty = {"source": {k: 0 for k in SOURCE_SCHEMA}, "attribute": {k: 0 for k in ATTRIBUTE_SCHEMA}}
        for epoch, order in enumerate(schedules["continuation"]["orders"], 1):
            for start in range(0, len(order), cfg["batch_size"]):
                indices = order[start:start + cfg["batch_size"]]
                a, y = _batch(attributes, indices), _batch(source, indices)
                for _ in range(cfg["adversary_updates_per_mapper_step"]):
                    _, counts = adversary_update(arm, attackers, attacker_optimizer, x[indices], a)
                    count["adversary_optimizer_steps"] += 1
                    for k, n in counts.items():
                        empty["attribute"][k] += int(n == 0)
                detail = mapper_update(arm, attackers, optimizer, x[indices], y, a, priors, protected)
                count["mapper_optimizer_steps"] += 1
                for k, n in detail["source_support"].items():
                    empty["source"][k] += int(n == 0)
            if epoch % cfg["curve_interval"] == 0 or epoch == cfg["continuation_epochs"]:
                curve.append(_curve(arm, attackers, x, source, attributes, xv, validation, priors, epoch, count))
        _checkpoint(arm_dir / "final.pt", arm, attackers, optimizer, attacker_optimizer,
                    {**count, "continuation_epoch": cfg["continuation_epochs"]}, "fixed final iterate")
        mapper_steps = cfg["continuation_epochs"] * n_batch
        adversary_steps = cfg["adversary_updates_per_mapper_step"] * mapper_steps
        arm_metadata[name] = {
            "protected": protected, "protection_coefficient": -.1 if protected else 0.,
            "fork_hashes": cloned, "final_model_hash": state_digest(arm.state_dict()),
            "final_adversary_hash": state_digest(attackers.state_dict()),
            "final_mapper_optimizer_hash": tree_digest(optimizer.state_dict()),
            "final_adversary_optimizer_hash": tree_digest(attacker_optimizer.state_dict()),
            "schedule_hash": schedules["continuation"]["sha256"], "schedule_seed": schedules["continuation"]["seed"],
            "continuation_mapper_optimizer_steps": mapper_steps,
            "continuation_adversary_optimizer_steps": adversary_steps,
            "optimizer_counts_including_common": count,
            "mapper_row_exposures": len(x) * cfg["continuation_epochs"],
            "adversary_row_exposures": len(x) * cfg["continuation_epochs"] * 3,
            "mapper_exposure_per_row": cfg["continuation_epochs"],
            "adversary_exposure_per_row": cfg["continuation_epochs"] * 3,
            "source_valid_label_exposures": {k: int((v >= 0).sum()) * cfg["continuation_epochs"] for k, v in source.items()},
            "attribute_valid_label_exposures": {k: int((v >= 0).sum()) * cfg["continuation_epochs"] * 3 for k, v in attributes.items()},
            "empty_minibatch_heads": empty, "curve": curve,
            "first_batch_gradient_diagnostics_at_shared_fork": diagnostics,
            "potential_protection_mapper_l2_at_shared_fork": diagnostics["protection_mapper_l2"],
            "applied_protection_mapper_l2_at_shared_fork": diagnostics["protection_mapper_l2"] if protected else 0.,
            "mapper_loss_uses_protection_gradient": protected,
            "selected_epoch": cfg["continuation_epochs"], "selection": "fixed final iterate; no validation selection",
        }
        if count != {"mapper_optimizer_steps": counters["mapper_optimizer_steps"] + mapper_steps,
                      "adversary_optimizer_steps": counters["adversary_optimizer_steps"] + adversary_steps}:
            raise AssertionError("Recorded continuation counts differ from optimizer actions")
        arms[name] = {"model": arm.freeze(), "adversaries": attackers.eval().requires_grad_(False)}
    metadata = {
        "config": cfg, "miniature": miniature, "seed": int(seed), "fit_pool": fit_pool,
        "source_label_keys": list(SOURCE_SCHEMA), "attribute_schema": ATTRIBUTE_SCHEMA,
        "reserved_labels_received": False, "final_evaluation_received": False,
        "source_validation_role": "diagnostics only; no gradients or checkpoint/hyperparameter selection",
        "source_fit_coverage": _coverage(source, SOURCE_SCHEMA), "attribute_fit_coverage": _coverage(attributes, ATTRIBUTE_SCHEMA),
        "source_validation_coverage": _coverage(validation, SOURCE_SCHEMA), "prior_entropies": priors,
        "fit_input_sha256": array_digest(xraw), "source_validation_input_sha256": array_digest(xvraw),
        "standardized_fit_sha256": array_digest(x.numpy()),
        "source_label_hashes": {k: array_digest(v.numpy()) for k, v in source.items()},
        "attribute_label_hashes": {k: array_digest(v.numpy()) for k, v in attributes.items()},
        "source_validation_label_hashes": {k: array_digest(v.numpy()) for k, v in validation.items()},
        "preprocessing": {"fit_rows": len(x), "fit_pool": fit_pool, "mean": mean.tolist(), "scale": scale.tolist(),
                          "fit_computation_dtype": "float64", "standardized_dtype": "float32"},
        "initialization_hashes": initial_hashes, "shared_fork_hashes": common_hashes,
        "initialization": initialization_metadata,
        "snapshots": {
            "I": {"checkpoint": "initialization.pt", "model_hash": initial_hashes["model"], "mapper_optimizer_steps": 0},
            "W": {"checkpoint": "warm_base.pt", "model_hash": common_hashes["model"],
                  "mapper_optimizer_steps": counters["mapper_optimizer_steps"],
                  "unchanged_across_adversary_warmup": True}} if initialization == "pca16" else {},
        "adversary_initialization_hash": adversary_initial_hash,
        "adversary_initialization_phase": "after final common base epoch, before adversary warmup",
        "common_optimizer_counts": counters, "shared_curves": curves, "common_empty_minibatch_heads": missing_batches,
        "schedules": {k: {f: v[f] for f in ("seed", "sha256")} for k, v in schedules.items()},
        "common_base_row_exposures": len(x) * cfg["warm_base_epochs"],
        "common_adversary_row_exposures": len(x) * cfg["warm_adversary_epochs"],
        "common_source_valid_label_exposures": {k: int((v >= 0).sum()) * cfg["warm_base_epochs"] for k, v in source.items()},
        "common_attribute_valid_label_exposures": {k: int((v >= 0).sum()) * cfg["warm_adversary_epochs"] for k, v in attributes.items()},
        "adversary_batch_reuse": "three consecutive detached updates on the current mapper minibatch before each mapper update",
        "source_missing_policy": "masked BCEWithLogits per one-logit head; all-missing minibatch head contributes differentiable zero; fixed mean over three heads",
        "attribute_missing_policy": "masked CE per attribute; all-missing minibatch head contributes differentiable zero; fixed mean over two attributes",
        "objective": "mean three source BCEs + .1 reconstruction MSE; D additionally minus .1 mean(attribute CE / fitting prior entropy)",
        "arms": arm_metadata, "runtime_seconds": time.perf_counter() - started,
        "module_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for name, snapshot in snapshots.items():
        if state_digest(snapshot.state_dict()) != metadata["snapshots"][name]["model_hash"]:
            raise AssertionError("Frozen I/W snapshot changed during training")
    (out / "training.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return {"arms": arms, "snapshots": snapshots, "metadata": metadata}
