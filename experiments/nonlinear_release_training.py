"""Matched nonlinear-adversary training on purpose-specific erased releases.

The caller supplies representation-training data only. A fixed prefix is held
out from every encoder/adversary update and used to refresh detached LEACE
maps. Final calibration, validation, and auditing belong to the caller and
never enter this training API. MSE-based adversary scores are training
surrogates, not universal privacy or classification guarantees.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from concept_erasure import LeaceEraser

from experiments.nonlinear_conflict_training import (
    SharedEncoder, clone_state, continuous_r2, minibatch_schedule,
    state_digest, _model_hash, _schedule_hash,
)
from pcrl.models.lora import PerPurposeLoRAEncoder

CONSTRAINT_NAMES = ("p1_V", "p1_S", "p2_U", "p2_S", "combined_S")
PROHIBITED = ((1, 2), (0, 2))
RELEASE_TRAIN_CONFIG = {
    "steps": 400, "batch_size": 256, "encoder_lr": .001,
    "rank": 4, "alpha": 4., "dropout": 0.,
    "eraser_holdout_size": 1024, "eraser_refresh_every": 25,
    "adversary_hidden": 32, "adversary_lr": .001,
    "adversary_warmup_steps": 100, "adversary_steps_per_encoder": 2,
    "feature_std_floor": 1e-6, "target_std_floor": 1e-6,
    "per_purpose_threshold": .05, "concat_threshold": .10,
    "dual_lr_multiplier": .02, "dual_max": 1.,
    "leace": {"shrinkage": False, "constrain_cov_trace": False, "svd_tol": 1e-10},
}


# Common runner naming; both exports describe the same fixed protocol.
TRAIN_CONFIG = RELEASE_TRAIN_CONFIG

def _dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _hash_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _architecture(state, prefix=""):
    first, last = state[prefix + "network.0.weight"], state[prefix + "network.4.weight"]
    return SharedEncoder(input_dim=first.shape[1], hidden_dim=first.shape[0], repr_dim=last.shape[0]).float()


def load_pretrained(prior_seed_dir):
    """Load a prior selected shared checkpoint without touching prior artifacts."""
    directory = Path(prior_seed_dir)
    source = directory / "pretraining" / "selected.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    encoder = _architecture(checkpoint["encoder"])
    encoder.load_state_dict(checkpoint["encoder"])
    head = nn.Linear(checkpoint["task_head"]["weight"].shape[1], 2).float()
    head.load_state_dict(checkpoint["task_head"])
    encoder.eval()
    head.eval()
    actual_hash = _model_hash(encoder, head)
    if checkpoint.get("state_hash") != actual_hash:
        raise ValueError("prior selected checkpoint content hash does not match its metadata")
    metadata_path = directory / "pretraining" / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({"selected_state_hash": actual_hash, "source_checkpoint": str(source.resolve()),
                     "source_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "selected_optimizer_steps": checkpoint["optimizer_steps"]})
    return {"encoder": encoder, "task_head": head, "metadata": metadata}


def load_reference(prior_seed_dir, key):
    """Reconstruct C or a linear-constraint D reference from its exact checkpoint."""
    aliases = {"C": "C_task_only", "D_0.1": "D_protection_0.1", "D_1": "D_protection_1",
               "D_1.0": "D_protection_1", "initialization": "C_task_only"}
    method = aliases.get(key, key)
    if method not in {"C_task_only", "D_protection_0.1", "D_protection_1"}:
        raise ValueError(f"unknown prior reference {key!r}")
    directory = Path(prior_seed_dir) / method / "adaptation"
    source = directory / ("initialization.pt" if key == "initialization" else "final.pt")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state = checkpoint["encoder"]
    rank = state["adapters.0.0.A.weight"].shape[0]
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    cfg = metadata.get("config", {})
    encoder = PerPurposeLoRAEncoder(_architecture(state, "backbone."), 2, rank=rank,
                                    alpha=cfg.get("alpha", float(rank)),
                                    dropout=cfg.get("dropout", 0.), lora_target="all_linear")
    encoder.load_state_dict(state)
    width = state["backbone.network.4.weight"].shape[0]
    heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(2)])
    heads.load_state_dict(checkpoint["heads"])
    encoder.eval()
    heads.eval()
    actual_hash = _model_hash(encoder, heads)
    if checkpoint.get("state_hash") != actual_hash:
        raise ValueError("prior reference checkpoint content hash does not match its metadata")
    metadata.update({"reference_key": key, "source_checkpoint": str(source.resolve()),
                     "source_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "loaded_state_hash": actual_hash, "loaded_optimizer_steps": checkpoint["optimizer_steps"],
                     "optimizer_steps": checkpoint["optimizer_steps"],
                     "checkpoint_stage": "initialization" if key == "initialization" else "final"})
    return {"encoder": encoder, "heads": heads, "metadata": metadata}


class TrainingRelease(nn.Module):
    """Frozen affine release maps with fixed training-holdout feature scaling."""

    def __init__(self, projections, biases, feature_mean, feature_std, metadata):
        super().__init__()
        for name, value in (("projections", projections), ("biases", biases),
                            ("feature_mean", feature_mean), ("feature_std", feature_std)):
            self.register_buffer(name, value.detach().float().clone())
        self.metadata = metadata

    def forward(self, h, purpose):
        return h @ self.projections[purpose].T + self.biases[purpose]

    def normalize(self, release, purpose):
        return (release - self.feature_mean[purpose]) / self.feature_std[purpose]


@torch.no_grad()
def fit_training_erasers(encoder, x_holdout, y_holdout, *, config=None):
    """Fit detached LEACE in float64; cache actual float32-release normalizers."""
    cfg = {**RELEASE_TRAIN_CONFIG, **(config or {})}
    x = torch.as_tensor(x_holdout, dtype=torch.float32)
    y = torch.as_tensor(y_holdout, dtype=torch.float64)
    was_training = encoder.training
    encoder.eval()
    hs = [encoder(x, p).detach() for p in range(2)]
    projections, biases, means, stds = [], [], [], []
    for p, h in enumerate(hs):
        eraser = LeaceEraser.fit(h.double(), y[:, PROHIBITED[p]], **cfg["leace"])
        matrix = eraser.P.detach().float()
        center = eraser.bias.detach()
        bias = (center - eraser.P @ center).float()
        released = h @ matrix.T + bias
        projections.append(matrix)
        biases.append(bias)
        means.append(released.double().mean(0).float())
        stds.append(released.double().std(0, correction=0).clamp_min(cfg["feature_std_floor"]).float())
    encoder.train(was_training)
    release = TrainingRelease(torch.stack(projections), torch.stack(biases),
                               torch.stack(means), torch.stack(stds), {})
    release.metadata = {
        "fit_n": len(x), "fit_input_sha256": _hash_tensor(x),
        "fit_labels_sha256": _hash_tensor(y),
        "fit_representation_sha256_by_purpose": [_hash_tensor(h) for h in hs],
        "projection_and_normalizer_sha256": state_digest(release.state_dict()),
        "fit_dtype": "float64", "application_dtype": "float32",
        "fit_gradients": False, "application_differentiable": True,
        "feature_stats_fit_split": "representation_train eraser holdout",
    }
    return release


class ReleaseAdversaries(nn.Module):
    """Three matched two-hidden-layer MLPs covering five prohibited targets."""

    def __init__(self, repr_dim=8, hidden=32):
        super().__init__()
        self.networks = nn.ModuleList([
            nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output))
            for width, output in ((repr_dim, 2), (repr_dim, 2), (2*repr_dim, 1))
        ])

    def forward(self, views):
        return [network(view) for network, view in zip(self.networks, views)]


def protection_penalty(mse_per_constraint, weights, thresholds):
    """Encoder minimizes this signed term, thereby increasing adversary MSE."""
    return (weights * (1 - mse_per_constraint - thresholds)).mean()


def _released_views(encoder, release, x):
    hs = [release(encoder(x, p), p) for p in range(2)]
    normalized = [release.normalize(h, p) for p, h in enumerate(hs)]
    return hs, normalized + [torch.cat(normalized, dim=1)]


def _adversary_mse(adversaries, views, normalized_labels):
    prediction = adversaries(views)
    losses = [(prediction[p] - normalized_labels[:, PROHIBITED[p]]).square().mean(0)
              for p in range(2)]
    losses.append((prediction[2].squeeze(1) - normalized_labels[:, 2]).square().mean().reshape(1))
    return torch.cat(losses)


def _task_loss(heads, released, labels):
    return sum((heads[p](released[p]).squeeze(1) - labels[:, p]).square().mean() for p in range(2)) / 2


def _gradient_diagnostics(encoder, heads, release, adversaries, x, y, yn, weights, thresholds):
    released, normalized = _released_views(encoder, release, x)
    task = _task_loss(heads, released, y)
    mse = _adversary_mse(adversaries, normalized, yn)
    first = [encoder.adapters[p][0].B.weight for p in range(2)]
    task_grad = torch.autograd.grad(task, first, retain_graph=True)
    protection_grad = torch.autograd.grad(protection_penalty(mse, torch.ones_like(weights), thresholds),
                                          first, retain_graph=True)
    weighted_grad = torch.autograd.grad(protection_penalty(mse, weights, thresholds), first)
    norm = lambda gs: sum(float(g.square().sum()) for g in gs)**.5
    result = {"task_first_layer_B_l2": norm(task_grad),
              "protection_first_layer_B_l2": norm(protection_grad),
              "weighted_protection_first_layer_B_l2": norm(weighted_grad),
              "task_mse": float(task.detach()),
              "normalized_adversary_mse": dict(zip(CONSTRAINT_NAMES, mse.detach().tolist())),
              "location": "upstream first-layer LoRA B through fixed erased release and frozen adversary"}
    for key in ("task_first_layer_B_l2", "protection_first_layer_B_l2"):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise RuntimeError(f"nonfinite or zero actual-release upstream gradient: {key}")
    if weights.max() > 0 and (not math.isfinite(result["weighted_protection_first_layer_B_l2"])
                              or result["weighted_protection_first_layer_B_l2"] <= 0):
        raise RuntimeError("nonfinite or zero weighted actual-release protection gradient")
    return result


def train_adversarial(pre, x_train, y_train, *, seed, method, weight, out_dir, config=None):
    """Train matched task/fixed/dual arms on actual erased releases only.

    Fixed and dual differ only in their five scalar protection weights. All
    three arms execute the same adversary warmup and update schedule, even
    when task-only encoder updates ignore the protection term.
    """
    cfg = copy.deepcopy(RELEASE_TRAIN_CONFIG)
    cfg.update(config or {})
    if method not in {"task", "fixed", "dual"}:
        raise ValueError("method must be task, fixed, or dual")
    if not math.isfinite(weight) or weight < 0 or (method == "task" and weight != 0):
        raise ValueError("weight must be finite nonnegative; task mode requires zero")
    if method == "dual" and weight > cfg["dual_max"]:
        raise ValueError("initial dual weight exceeds the configured projection cap")
    x = torch.as_tensor(x_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.float32)
    n_holdout = cfg["eraser_holdout_size"]
    if x.ndim != 2 or y.shape != (len(x), 3) or not (2 <= n_holdout < len(x)-1):
        raise ValueError("need U,V,S labels and nonempty distinct eraser-holdout/update pools")
    if cfg["eraser_refresh_every"] <= 0 or cfg["adversary_steps_per_encoder"] < 1:
        raise ValueError("refresh interval and adversary updates per encoder must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    update_x, update_y = x[n_holdout:], y[n_holdout:]
    target_mean = update_y.double().mean(0).float()
    target_std = update_y.double().std(0, correction=0).float()
    if not bool(torch.isfinite(target_std).all()) or bool((target_std < cfg["target_std_floor"]).any()):
        raise ValueError("adversary target normalization needs finite, varying update-pool targets")
    normalized_y = (update_y - target_mean) / target_std
    torch.manual_seed(630000 + seed)
    encoder = PerPurposeLoRAEncoder(copy.deepcopy(pre["encoder"]), 2,
                                    rank=cfg["rank"], alpha=cfg["alpha"],
                                    dropout=cfg["dropout"], lora_target="all_linear")
    width = pre["task_head"].in_features
    heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(2)])
    with torch.no_grad():
        for p in range(2):
            heads[p].weight.copy_(pre["task_head"].weight[p:p+1])
            heads[p].bias.copy_(pre["task_head"].bias[p:p+1])
    encoder_initial = clone_state(encoder)
    heads_initial = clone_state(heads)
    initial_hash = _model_hash(encoder, heads)
    initial_backbone_hash = state_digest(encoder.backbone.state_dict())
    torch.manual_seed(660000 + seed)
    adversaries = ReleaseAdversaries(width, cfg["adversary_hidden"])
    adversary_initial_hash = state_digest(adversaries.state_dict())
    encoder_optimizer = torch.optim.Adam(list(encoder.trainable_parameters()) + list(heads.parameters()),
                                          lr=cfg["encoder_lr"])
    adversary_optimizer = torch.optim.Adam(adversaries.parameters(), lr=cfg["adversary_lr"])
    encoder_schedule = minibatch_schedule(len(update_x), cfg["batch_size"], cfg["steps"], 640000 + seed)
    total_adversary_steps = cfg["adversary_warmup_steps"] + cfg["steps"] * cfg["adversary_steps_per_encoder"]
    adversary_schedule = minibatch_schedule(len(update_x), cfg["batch_size"], total_adversary_steps, 650000 + seed)
    np.savez_compressed(
        out / "training_schedules.npz",
        eraser_fit_indices=np.arange(n_holdout, dtype=np.int64),
        update_pool_indices=np.arange(n_holdout, len(x), dtype=np.int64),
        encoder_indices=(torch.cat(encoder_schedule).numpy()+n_holdout
                         if encoder_schedule else np.array([], dtype=np.int64)),
        encoder_batch_lengths=np.array([len(i) for i in encoder_schedule], dtype=np.int64),
        adversary_indices=(torch.cat(adversary_schedule).numpy()+n_holdout
                           if adversary_schedule else np.array([], dtype=np.int64)),
        adversary_batch_lengths=np.array([len(i) for i in adversary_schedule], dtype=np.int64),
    )
    diagnostic_indices = encoder_schedule[0] if encoder_schedule else torch.arange(min(len(update_x), cfg["batch_size"]))
    thresholds = torch.tensor([cfg["per_purpose_threshold"]]*4 + [cfg["concat_threshold"]])
    weights = torch.full((5,), float(weight), dtype=torch.float32)
    encoder_steps = adversary_steps = dual_steps = 0
    eraser_records, diagnostics, history = [], [], []
    release = None

    def refresh():
        nonlocal release
        release = fit_training_erasers(encoder, x[:n_holdout], y[:n_holdout], config=cfg)
        record = {"encoder_optimizer_steps": encoder_steps, **release.metadata,
                  "fit_indices_first": 0, "fit_indices_last": n_holdout-1,
                  "fit_indices_sha256": _hash_tensor(torch.arange(n_holdout, dtype=torch.int64))}
        eraser_records.append(record)
        torch.save({"release": clone_state(release), "metadata": record},
                   out / f"training_eraser_step_{encoder_steps}.pt")

    def checkpoint(name):
        torch.save({"encoder": clone_state(encoder), "heads": clone_state(heads),
                    "adversaries": clone_state(adversaries), "training_release": clone_state(release),
                    "encoder_optimizer": encoder_optimizer.state_dict(),
                    "adversary_optimizer": adversary_optimizer.state_dict(),
                    "optimizer_steps": {"encoder": encoder_steps, "adversary": adversary_steps, "dual": dual_steps},
                    "weights": weights.clone(), "thresholds": thresholds.clone(),
                    "target_mean": target_mean, "target_std": target_std,
                    "initial_state_hash": initial_hash, "state_hash": _model_hash(encoder, heads),
                    "method": method, "initial_weight": weight}, out / f"{name}.pt")

    @torch.no_grad()
    def diagnostic(stage):
        released, normalized = _released_views(encoder, release, update_x[diagnostic_indices])
        mse = _adversary_mse(adversaries, normalized, normalized_y[diagnostic_indices])
        task_predictions = torch.cat([heads[p](released[p]) for p in range(2)], dim=1)
        labels = update_y[diagnostic_indices]
        task_mse = (task_predictions-labels[:, :2]).square().mean(0)
        task_variance = (labels[:, :2]-labels[:, :2].mean(0)).square().mean(0)
        linear = [float(continuous_r2(released[p], labels[:, a]))
                  for p in range(2) for a in PROHIBITED[p]]
        linear.append(float(continuous_r2(torch.cat(released, 1), labels[:, 2])))
        row = {"stage": stage, "encoder_optimizer_steps": encoder_steps,
               "adversary_optimizer_steps": adversary_steps,
               "task_mse_by_purpose": task_mse.tolist(),
               "task_r2_by_purpose": (1-task_mse/task_variance).tolist(),
               "normalized_adversary_mse": dict(zip(CONSTRAINT_NAMES, mse.tolist())),
               "adversary_training_surrogate_r2": dict(zip(CONSTRAINT_NAMES, (1-mse).tolist())),
               "ridge_linear_r2_on_releases": dict(zip(CONSTRAINT_NAMES, linear)),
               "weights": dict(zip(CONSTRAINT_NAMES, weights.tolist())),
               "release_state_hash": state_digest(release.state_dict()),
               "diagnostic_split": "fixed batch from representation_train update pool; not out-of-sample audit"}
        diagnostics.append(row)
        return row

    def adversary_step():
        nonlocal adversary_steps
        indices = adversary_schedule[adversary_steps]
        for parameter in adversaries.parameters():
            parameter.requires_grad_(True)
        with torch.no_grad():
            _, normalized = _released_views(encoder, release, update_x[indices])
        mse = _adversary_mse(adversaries, [h.detach() for h in normalized], normalized_y[indices])
        loss = mse.mean()
        if not bool(torch.isfinite(loss)):
            _dump(out / "instability.json", {"phase": "adversary", "encoder_steps": encoder_steps,
                                             "adversary_steps": adversary_steps})
            raise FloatingPointError("nonfinite adversary objective; partial evidence retained")
        adversary_optimizer.zero_grad()
        loss.backward()
        adversary_optimizer.step()
        adversary_steps += 1

    encoder.train()
    heads.train()
    adversaries.train()
    refresh()
    checkpoint("initialization")
    diagnostic("initial_before_adversary_warmup")
    for _ in range(cfg["adversary_warmup_steps"]):
        adversary_step()
    diagnostic("after_adversary_warmup")
    for parameter in adversaries.parameters():
        parameter.requires_grad_(False)
    gradient_record = _gradient_diagnostics(
        encoder, heads, release, adversaries, update_x[diagnostic_indices],
        update_y[diagnostic_indices], normalized_y[diagnostic_indices], weights, thresholds)
    checkpoint("after_warmup")
    for indices in encoder_schedule:
        for _ in range(cfg["adversary_steps_per_encoder"]):
            adversary_step()
        for parameter in adversaries.parameters():
            parameter.requires_grad_(False)
        released, normalized = _released_views(encoder, release, update_x[indices])
        mse = _adversary_mse(adversaries, normalized, normalized_y[indices])
        task_loss = _task_loss(heads, released, update_y[indices])
        weights_before = weights.clone()
        privacy = protection_penalty(mse, weights_before, thresholds)
        objective = task_loss + privacy
        if not bool(torch.isfinite(objective)):
            _dump(out / "instability.json", {"phase": "encoder", "encoder_steps": encoder_steps,
                                             "adversary_steps": adversary_steps})
            raise FloatingPointError("nonfinite encoder objective; partial evidence retained")
        encoder_optimizer.zero_grad()
        objective.backward()
        encoder_optimizer.step()
        encoder_steps += 1
        if method == "dual":
            with torch.no_grad():
                weights.add_(cfg["dual_lr_multiplier"] * weight * (1-mse.detach()-thresholds))
                weights.clamp_(0., cfg["dual_max"])
            dual_steps += 1
        history.append({"encoder_optimizer_steps": encoder_steps,
                        "adversary_optimizer_steps": adversary_steps,
                        "task_loss": float(task_loss.detach()), "protection_penalty": float(privacy.detach()),
                        "encoder_objective": float(objective.detach()),
                        "normalized_adversary_mse": mse.detach().tolist(),
                        "weights_before": weights_before.tolist(), "weights_after": weights.tolist(),
                        "thresholds": thresholds.tolist()})
        if encoder_steps % cfg["eraser_refresh_every"] == 0 or encoder_steps == cfg["steps"]:
            refresh()
            diagnostic("after_eraser_refresh")
    encoder.eval()
    heads.eval()
    adversaries.eval()
    final_backbone_hash = state_digest(encoder.backbone.state_dict())
    if initial_backbone_hash != final_backbone_hash:
        raise AssertionError("frozen pretrained backbone changed during adaptation")
    if encoder_steps != cfg["steps"] or adversary_steps != total_adversary_steps:
        raise AssertionError("actual optimizer budget differs from frozen configuration")
    checkpoint("final")
    parameter_delta = {}
    for p, adapters in enumerate(encoder.adapters):
        for layer, adapter in enumerate(adapters):
            prefix = f"adapters.{p}.{layer}."
            parameter_delta[f"purpose_{p+1}_layer_{layer}"] = sum(
                float((value - encoder_initial[prefix+name]).square().sum())
                for name, value in adapter.state_dict().items())**.5
    initial_parameter_count = sum(p.numel() for p in encoder.trainable_parameters()) + sum(p.numel() for p in heads.parameters())
    metadata = {
        "config": cfg, "method": method, "initial_weight": weight,
        "optimizer_steps": encoder_steps, "adversary_optimizer_steps": adversary_steps,
        "adversary_network_update_count": 3*adversary_steps, "dual_update_count": dual_steps,
        "initial_state_hash": initial_hash, "final_state_hash": _model_hash(encoder, heads),
        "pretrained_state_hash": pre["metadata"]["selected_state_hash"],
        "initial_backbone_hash": initial_backbone_hash, "final_backbone_hash": final_backbone_hash,
        "adversary_initial_state_hash": adversary_initial_hash,
        "schedule_hash": _schedule_hash(encoder_schedule),
        "schedule_artifact": "training_schedules.npz",
        "schedule_artifact_coordinates": "absolute representation_train row indices; flattening segmented by batch lengths",
        "adversary_schedule_hash": _schedule_hash(adversary_schedule),
        "encoder_schedule_seed": 640000+seed, "adversary_schedule_seed": 650000+seed,
        "encoder_initialization_seed": 630000+seed, "adversary_initialization_seed": 660000+seed,
        "trainable_parameter_count": initial_parameter_count,
        "adversary_parameter_count": sum(p.numel() for p in adversaries.parameters()),
        "gradient_diagnostics": gradient_record,
        "parameter_delta_l2_by_layer": parameter_delta,
        "first_layer_parameter_delta_l2": sum(v*v for k, v in parameter_delta.items() if k.endswith("layer_0"))**.5,
        "eraser_fit_indices": list(range(n_holdout)),
        "update_indices_first": n_holdout, "update_indices_last": len(x)-1,
        "update_indices_sha256": _hash_tensor(torch.arange(n_holdout, len(x), dtype=torch.int64)),
        "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
        "target_stats_fit_split": "representation_train update pool only",
        "eraser_refresh_history": eraser_records,
        "diagnostics": diagnostics, "training_history": history,
        "constraint_names": list(CONSTRAINT_NAMES), "thresholds": thresholds.tolist(),
        "final_weights": weights.tolist(), "instability": None,
        "selection": "fixed final iterate; no validation or test data enter training",
        "task_head_input": "actual unnormalized erased purpose release",
        "adversary_input": "actual erased release, normalized with detached eraser-holdout statistics",
        "surrogate": "1 - normalized-target adversary MSE; no hinge or clipping",
        "encoder_objective": "raw mean purpose task MSE + mean(weight_c * (1 - normalized_adversary_MSE_c - threshold_c))",
        "adversary_objective": "mean normalized MSE across the five prohibited target constraints",
        "dtype": "float32; LEACE fitting float64", "device": "cpu", "checkpoint_dir": str(out),
    }
    _dump(out / "metadata.json", metadata)
    return {"encoder": encoder, "heads": heads, "adversaries": adversaries,
            "training_release": release, "metadata": metadata}
