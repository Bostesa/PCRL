"""Matched nonlinear upstream adaptation for the purpose-conflict experiment.

This module receives already separated fitting and validation arrays. It never
receives attacker-fitting or final-test arrays. All adaptive choices must be
made from the supplied validation split, and all model updates use fitting data.

The shared backbone is task-pretrained once. The two adaptation arms reuse its
weights, head initialization, and batch order. Erasers are calibrated separately
by the caller after adaptation; this module applies no erasure during training.
The first LoRA adapter precedes a nonlinear activation, allowing actual upstream
representation learning rather than a post-erasure affine change of coordinates.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.training.proxy_lagrangian import Constraint, ProxyLagrangianOptimizer


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Content hash independent of torch serialization container metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def minibatch_schedule(n: int, batch_size: int, steps: int, seed: int) -> list[torch.Tensor]:
    """Deterministic shuffled epochs, identical for every adaptation arm."""
    if n <= 0 or batch_size <= 0 or steps < 0:
        raise ValueError("positive fitting size/batch size and nonnegative steps required")
    generator = torch.Generator().manual_seed(seed)
    batches = []
    while len(batches) < steps:
        order = torch.randperm(n, generator=generator)
        for indices in order.split(batch_size):
            batches.append(indices)
            if len(batches) == steps:
                break
    return batches


def clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Independent CPU snapshot: later optimizer steps cannot mutate it."""
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


TRAIN_CONFIG = {
    "pretrain": {
        "hidden_dim": 32, "repr_dim": 8, "epochs": 80, "batch_size": 256,
        "lr": .002, "eval_every": 5,
    },
    "adaptation": {
        "steps": 400, "batch_size": 256, "lr": .001,
        "rank": 4, "alpha": 4., "dropout": 0.,
        "ridge": .001, "std_floor": 1e-6,
        "per_purpose_threshold": .05, "concat_threshold": .10,
        "dual_init": 1., "dual_lr": .02, "dual_max": 10.,
    },
}
PROHIBITED = ((1, 2), (0, 2))
SIGNALS = ("U", "V", "S")


class SharedEncoder(nn.Module):
    """Small smooth nonlinear backbone; no BatchNorm or dropout state."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, repr_dim: int = 8):
        super().__init__()
        self.repr_dim = repr_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _config(config, section):
    resolved = copy.deepcopy(TRAIN_CONFIG[section])
    overrides = (config or {}).get(section, config or {})
    resolved.update(overrides)
    return resolved


def _tensor(array):
    return torch.as_tensor(array, dtype=torch.float32, device="cpu")


def _dump(path, value):
    import json
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _schedule_hash(batches):
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(len(batch).to_bytes(8, "little"))
        digest.update(batch.numpy().tobytes())
    return digest.hexdigest()


def _model_hash(encoder, heads):
    merged = {"encoder." + k: v for k, v in encoder.state_dict().items()}
    merged.update({"heads." + k: v for k, v in heads.state_dict().items()})
    return state_digest(merged)


@torch.no_grad()
def _validation(encoder, head, x, y):
    encoder.eval()
    head.eval()
    prediction = head(encoder(x))
    mse = (prediction - y).square().mean(0)
    variance = (y - y.mean(0)).square().mean(0)
    return {
        "mean_mse": float(mse.mean()),
        "mse_by_task": mse.tolist(),
        "r2_by_task": (1 - mse / variance).tolist(),
        "n": len(x),
    }


def pretrain(x_train, y_train, x_val, y_val, *, seed, out_dir, config=None):
    """Train U/V jointly; choose weights only by validation mean task MSE.

    Validation occurs at genuine initialization and every configured interval,
    including the final epoch. All training epochs execute irrespective of
    selection. Returned encoder/head contain selected, not final, weights.
    """
    cfg = _config(config, "pretrain")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    x, y = _tensor(x_train), _tensor(y_train)
    vx, vy = _tensor(x_val), _tensor(y_val)
    if y.ndim != 2 or y.shape[1] != 2 or vy.shape[1] != 2:
        raise ValueError("shared pretraining requires exactly the U,V task columns")
    if cfg["epochs"] < 0 or cfg["eval_every"] <= 0:
        raise ValueError("nonnegative epochs and positive validation interval required")
    model_seed, schedule_seed = 610000 + seed, 620000 + seed
    torch.manual_seed(model_seed)
    encoder = SharedEncoder(x.shape[1], cfg["hidden_dim"], cfg["repr_dim"])
    head = nn.Linear(cfg["repr_dim"], 2)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=cfg["lr"])
    steps_per_epoch = (len(x) + cfg["batch_size"] - 1) // cfg["batch_size"]
    batches = minibatch_schedule(len(x), cfg["batch_size"], cfg["epochs"] * steps_per_epoch, schedule_seed)
    initial_encoder, initial_task_head = copy.deepcopy(encoder).eval(), copy.deepcopy(head).eval()
    initial_hash = _model_hash(encoder, head)
    initial_val = _validation(encoder, head, vx, vy)
    history = [{"epoch": 0, "optimizer_steps": 0, **initial_val}]
    selected_epoch, selected_steps = 0, 0
    selected_val = initial_val
    best_encoder, best_head = clone_state(encoder), clone_state(head)
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    torch.save({"encoder": best_encoder, "task_head": best_head,
                "optimizer": best_optimizer, "optimizer_steps": 0, "epoch": 0,
                "state_hash": initial_hash, "initialization": "random before any task-pretraining updates"},
               out / "initialization.pt")
    steps = 0
    for epoch in range(1, cfg["epochs"] + 1):
        encoder.train()
        head.train()
        for indices in batches[(epoch-1)*steps_per_epoch:epoch*steps_per_epoch]:
            optimizer.zero_grad()
            loss = (head(encoder(x[indices])) - y[indices]).square().mean()
            loss.backward()
            optimizer.step()
            steps += 1
        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            score = _validation(encoder, head, vx, vy)
            history.append({"epoch": epoch, "optimizer_steps": steps, **score})
            if score["mean_mse"] < selected_val["mean_mse"]:
                selected_epoch, selected_steps, selected_val = epoch, steps, score
                best_encoder, best_head = clone_state(encoder), clone_state(head)
                best_optimizer = copy.deepcopy(optimizer.state_dict())
    final_val = _validation(encoder, head, vx, vy)
    torch.save({"encoder": clone_state(encoder), "task_head": clone_state(head),
                "optimizer": optimizer.state_dict(), "optimizer_steps": steps,
                "epoch": cfg["epochs"], "state_hash": _model_hash(encoder, head)},
               out / "final.pt")
    encoder.load_state_dict(best_encoder)
    head.load_state_dict(best_head)
    encoder.eval()
    head.eval()
    selected_hash = _model_hash(encoder, head)
    torch.save({"encoder": best_encoder, "task_head": best_head,
                "optimizer": best_optimizer, "optimizer_steps": selected_steps,
                "epoch": selected_epoch, "state_hash": selected_hash,
                "selection": "minimum validation mean U/V task MSE; no attacker or test data"},
               out / "selected.pt")
    metadata = {
        "config": cfg, "seed": seed, "model_seed": model_seed,
        "schedule_seed": schedule_seed, "schedule_hash": _schedule_hash(batches),
        "optimizer_steps": steps, "selected_epoch": selected_epoch,
        "selected_optimizer_steps": selected_steps,
        "initial_state_hash": initial_hash, "selected_state_hash": selected_hash,
        "initial_val_mse": initial_val["mean_mse"],
        "selected_val_mse": selected_val["mean_mse"],
        "final_val_mse": final_val["mean_mse"],
        "initial_validation": initial_val, "selected_validation": selected_val,
        "final_validation": final_val, "validation_history": history,
        "selection": "minimum validation mean U/V MSE; all scheduled epochs execute",
        "dtype": "float32", "device": "cpu", "pretraining_tasks": ["U", "V"],
        "checkpoint_dir": str(out),
    }
    _dump(out / "metadata.json", metadata)
    return {"encoder": encoder, "task_head": head, "metadata": metadata,
            "initial_encoder": initial_encoder, "initial_task_head": initial_task_head}


def continuous_r2(h, z, *, ridge=.001, std_floor=1e-6):
    """Differentiable batch affine ridge R²; empirical training proxy only.

    Center features and target on this fitting batch. Standardize each feature
    by its batch RMS with the declared floor, then solve (H'H/n + ridge I).
    Predictive auditing must fit a separate attacker on its own fitting split.
    """
    hc, zc = h - h.mean(0), z - z.mean(0)
    hc = hc / hc.square().mean(0).clamp_min(std_floor**2).sqrt()
    target_ss = zc.square().sum()
    if not bool(target_ss > 0):
        return h.sum() * 0 + float("nan")
    gram = hc.T @ hc / len(h) + ridge * torch.eye(h.shape[1], dtype=h.dtype, device=h.device)
    weights = torch.linalg.solve(gram, hc.T @ zc / len(h))
    return (1 - (zc - hc @ weights).square().sum() / target_ss).clamp_min(0)


def _constraint_values(views, labels, cfg):
    kwargs = {"ridge": cfg["ridge"], "std_floor": cfg["std_floor"]}
    values = {f"P{p+1}_{SIGNALS[a]}": continuous_r2(views[p], labels[:, a], **kwargs)
              for p in range(2) for a in PROHIBITED[p]}
    values["combined_S"] = continuous_r2(torch.cat(views, 1), labels[:, 2], **kwargs)
    return values


def _losses(encoder, heads, x, y, cfg):
    views = [encoder(x, p) for p in range(2)]
    task = sum((heads[p](views[p]).squeeze(1) - y[:, p]).square().mean()
               for p in range(2)) / 2
    return task, _constraint_values(views, y, cfg)


def gradient_diagnostics(encoder, heads, x_batch, y_batch, config=None):
    """Separate unscaled task/privacy gradients at the true initial state.

    Reports first-layer B gradients before the first GELU. The privacy loss is
    the sum of all five R² values (subtracting fixed thresholds has no effect
    on these gradients), before any strength or one-fifth scaling.
    """
    cfg = _config(config, "adaptation")
    task, values = _losses(encoder, heads, _tensor(x_batch), _tensor(y_batch), cfg)
    first_layer = [encoder.adapters[p][0].B.weight for p in range(2)]
    task_grad = torch.autograd.grad(task, first_layer, retain_graph=True)
    protection = sum(values.values())
    protection_grad = torch.autograd.grad(protection, first_layer)
    task_norms = [float(g.norm()) for g in task_grad]
    protection_norms = [float(g.norm()) for g in protection_grad]
    return {
        "task_first_layer_B_l2": sum(v*v for v in task_norms)**.5,
        "protection_first_layer_B_l2": sum(v*v for v in protection_norms)**.5,
        "task_first_layer_B_l2_by_purpose": task_norms,
        "protection_first_layer_B_l2_by_purpose": protection_norms,
        "constraint_values": {k: float(v.detach()) for k, v in values.items()},
        "task_mse": float(task.detach()), "protection_sum_r2": float(protection.detach()),
        "location": "first Linear LoRA B, upstream of both GELUs; no erasure is applied",
        "before_strength_scaling": True,
    }


def adapt(pretrained, x_train, y_train, *, seed, strength, out_dir, config=None):
    """Matched C/D upstream adaptation, with fixed final-iterate selection.

    C (strength=0) optimizes task loss only and does not update duals. D adds
    strength/5 times the existing proxy-Lagrangian term for the five declared
    pre-erasure leakage constraints. No LEACE is fitted or applied here; the
    caller calibrates final erasers after this model is frozen.
    """
    cfg = _config(config, "adaptation")
    if strength < 0:
        raise ValueError("strength must be nonnegative")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    x, y = _tensor(x_train), _tensor(y_train)
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("adaptation labels must have exactly U,V,S columns")
    model_seed, schedule_seed = 630000 + seed, 640000 + seed
    torch.manual_seed(model_seed)
    encoder = PerPurposeLoRAEncoder(
        copy.deepcopy(pretrained["encoder"]), n_purposes=2,
        rank=cfg["rank"], alpha=cfg["alpha"], dropout=cfg["dropout"],
        lora_target="all_linear",
    )
    width = pretrained["task_head"].in_features
    heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(2)])
    with torch.no_grad():
        for p in range(2):
            heads[p].weight.copy_(pretrained["task_head"].weight[p:p+1])
            heads[p].bias.copy_(pretrained["task_head"].bias[p:p+1])
    optimizer = torch.optim.Adam(list(encoder.trainable_parameters()) + list(heads.parameters()), lr=cfg["lr"])
    constraints = [Constraint(f"P{p+1}_{SIGNALS[a]}", cfg["per_purpose_threshold"],
                              eta_lambda=cfg["dual_lr"], lambda_init=cfg["dual_init"],
                              lambda_max=cfg["dual_max"])
                   for p in range(2) for a in PROHIBITED[p]]
    constraints.append(Constraint("combined_S", cfg["concat_threshold"],
                                  eta_lambda=cfg["dual_lr"], lambda_init=cfg["dual_init"],
                                  lambda_max=cfg["dual_max"]))
    proxy = ProxyLagrangianOptimizer(optimizer, constraints)
    schedule = minibatch_schedule(len(x), cfg["batch_size"], cfg["steps"], schedule_seed)
    initial_encoder, initial_heads = clone_state(encoder), clone_state(heads)
    initial_hash = _model_hash(encoder, heads)
    diagnostic_indices = schedule[0] if schedule else torch.arange(min(len(x), cfg["batch_size"]))
    diagnostics = gradient_diagnostics(encoder, heads, x[diagnostic_indices], y[diagnostic_indices], config)
    for name in ("task_first_layer_B_l2", "protection_first_layer_B_l2"):
        if not math.isfinite(diagnostics[name]) or diagnostics[name] <= 0:
            raise RuntimeError(f"broken upstream gradient path before adaptation: {name}={diagnostics[name]}")
    initial_proxy_term = sum(
        c.lambda_value * c.violation(diagnostics["constraint_values"][name])
        for name, c in proxy.constraints.items()
    )
    initial_scaled_protection = strength / len(constraints) * initial_proxy_term
    # Every initial dual has the same configured value. The protection
    # gradient therefore scales uniformly; its loss value may legitimately
    # be negative when the constraints are already satisfied.
    initial_scaled_gradient_norm = (
        strength / len(constraints) * constraints[0].lambda_value
        * diagnostics["protection_first_layer_B_l2"]
    )
    if not math.isfinite(initial_scaled_protection):
        raise RuntimeError("the initial scaled protection term must be finite")
    if strength and (not math.isfinite(initial_scaled_gradient_norm) or initial_scaled_gradient_norm <= 0):
        raise RuntimeError("positive-strength adaptation requires a finite nonzero scaled upstream protection gradient")
    layer_names = {id(module): name for name, module in encoder.backbone.named_modules()}
    layer_locations = [{
        "layer_index": index, "backbone_module": layer_names[id(module)],
        "before_nonlinearity": index < len(encoder._linear_modules) - 1,
        "in_features": module.in_features, "out_features": module.out_features,
    } for index, module in enumerate(encoder._linear_modules)]
    torch.save({"encoder": initial_encoder, "heads": initial_heads,
                "optimizer": optimizer.state_dict(), "optimizer_steps": 0,
                "state_hash": initial_hash, "pretrained_state_hash": pretrained["metadata"]["selected_state_hash"],
                "duals": {n: c.lambda_value for n, c in proxy.constraints.items()}},
               out / "initialization.pt")
    history = []
    encoder.train()
    heads.train()
    for step, indices in enumerate(schedule, start=1):
        task, values = _losses(encoder, heads, x[indices], y[indices], cfg)
        objective = task
        if strength:
            protection = proxy.lagrangian_loss(task.new_zeros(()), values)
            objective = objective + strength / len(constraints) * protection
        optimizer.zero_grad()
        objective.backward()
        proxy.primal_step()
        scalars = {k: float(v.detach()) for k, v in values.items()}
        if strength:
            proxy.dual_step(scalars)
        history.append({"optimizer_steps": step, "fit_task_mse_before_update": float(task.detach()),
                        "fit_objective_before_update": float(objective.detach()),
                        "fit_r2_before_update": scalars,
                        "duals_after_update": {n: c.lambda_value for n, c in proxy.constraints.items()}})
    encoder.eval()
    heads.eval()
    parameter_delta = {}
    for p, adapters in enumerate(encoder.adapters):
        for layer, adapter in enumerate(adapters):
            prefix = f"adapters.{p}.{layer}."
            delta = sum(float((value - initial_encoder[prefix + name]).square().sum())
                        for name, value in adapter.state_dict().items())**.5
            parameter_delta[f"purpose_{p+1}_layer_{layer}"] = delta
    final_hash = _model_hash(encoder, heads)
    torch.save({"encoder": clone_state(encoder), "heads": clone_state(heads),
                "optimizer": optimizer.state_dict(), "optimizer_steps": len(schedule),
                "state_hash": final_hash, "initial_state_hash": initial_hash,
                "duals": {n: c.lambda_value for n, c in proxy.constraints.items()},
                "strength": strength, "selection": "fixed final iterate, no search"},
               out / "final.pt")
    metadata = {
        "config": cfg, "strength": strength, "optimizer_steps": len(schedule),
        "model_seed": model_seed, "schedule_seed": schedule_seed,
        "schedule_hash": _schedule_hash(schedule),
        "initial_state_hash": initial_hash, "final_state_hash": final_hash,
        "pretrained_state_hash": pretrained["metadata"]["selected_state_hash"],
        "gradient_diagnostics": diagnostics,
        "initial_proxy_lagrangian_term": initial_proxy_term,
        "initial_scaled_protection_term": initial_scaled_protection,
        "initial_scaled_protection_gradient_l2": initial_scaled_gradient_norm,
        "trainable_parameter_count": sum(p.numel() for p in encoder.trainable_parameters()) + sum(p.numel() for p in heads.parameters()),
        "adapter_parameter_count": sum(p.numel() for p in encoder.trainable_parameters()),
        "task_head_parameter_count": sum(p.numel() for p in heads.parameters()),
        "adapter_layer_locations_per_purpose": layer_locations,
        "parameter_delta_l2_by_layer": parameter_delta,
        "first_layer_parameter_delta_l2": sum(v*v for k, v in parameter_delta.items() if k.endswith("layer_0"))**.5,
        "dual_update_count": len(schedule) if strength else 0,
        "final_duals": {n: c.lambda_value for n, c in proxy.constraints.items()},
        "constraint_thresholds": {n: c.threshold for n, c in proxy.constraints.items()},
        "training_history": history,
        "selection": "fixed final iterate; no validation or test selection",
        "erasure_during_training": False,
        "objective": "mean purpose MSE + strength/5 * sum(lambda * (pre-erasure R2 - threshold))",
        "checkpoint_dir": str(out), "dtype": "float32", "device": "cpu",
    }
    _dump(out / "metadata.json", metadata)
    return {"encoder": encoder, "heads": heads, "metadata": metadata}
