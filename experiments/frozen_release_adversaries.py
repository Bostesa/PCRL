"""Matched adversary continuation and fresh fitting on frozen final releases.

B restarts from the saved adversary weights; C initializes the same three
networks afresh. Both reset Adam, keep caller-provided coordinates fixed, and
use identical fitting schedules, budgets and validation-selection opportunities.
Fitting never accepts final-test data and never mutates a supplied model.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments.nonlinear_conflict_probes import predictive_scores
from experiments.nonlinear_conflict_training import clone_state, minibatch_schedule, state_digest, _schedule_hash
from experiments.nonlinear_release_training import ReleaseAdversaries, _adversary_mse


TARGET_COLUMNS = (1, 2, 0, 2, 2)
TARGET_NAMES = ("p1_V", "p1_S", "p2_U", "p2_S", "combined_S")
TARGET_VIEWS = (0, 0, 1, 1, 2)
TARGET_OUTPUTS = (0, 1, 0, 1, 0)
RESTARTS = 2
PREDICTION_BATCH_SIZE = 4096


def _dump(path, value):
    def safe(obj):
        if isinstance(obj, dict):
            return {key: safe(item) for key, item in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [safe(item) for item in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def _matrix(value, name, columns=None):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite matrix")
    if columns is not None and result.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns")
    return result


def _architecture(model):
    state = model.state_dict()
    first = state["networks.0.0.weight"]
    width, hidden = first.shape[1], first.shape[0]
    expected = {"networks.1.0.weight": (hidden, width),
                "networks.2.0.weight": (hidden, 2 * width),
                "networks.0.4.weight": (2, hidden),
                "networks.1.4.weight": (2, hidden),
                "networks.2.4.weight": (1, hidden)}
    if any(tuple(state[key].shape) != shape for key, shape in expected.items()):
        raise ValueError("Expected three matched ReleaseAdversaries networks with outputs 2,2,1")
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("The saved-adversary coordinate protocol requires float32 networks")
    return width, hidden


class FixedCoordinates:
    """Clone and retain provided feature/target preprocessing without refitting."""

    def __init__(self, release_state, target_mean, target_std, width):
        self.release_state = {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                              for key, value in release_state.items()}
        self.feature_mean = torch.as_tensor(release_state["feature_mean"], dtype=torch.float32).detach().cpu().clone()
        self.feature_std = torch.as_tensor(release_state["feature_std"], dtype=torch.float32).detach().cpu().clone()
        self.target_mean = torch.as_tensor(target_mean, dtype=torch.float32).detach().cpu().clone()
        self.target_std = torch.as_tensor(target_std, dtype=torch.float32).detach().cpu().clone()
        if self.feature_mean.shape != (2, width) or self.feature_std.shape != (2, width):
            raise ValueError("Feature means/stds must have shape (2, representation_width)")
        if self.target_mean.shape != (3,) or self.target_std.shape != (3,):
            raise ValueError("Target means/stds must have shape (3,) in U,V,S order")
        if not all(torch.isfinite(value).all() for value in self.state().values()):
            raise ValueError("Fixed preprocessing must be finite")
        if not (self.feature_std > 0).all() or not (self.target_std > 0).all():
            raise ValueError("Fixed standard deviations must be positive")
        self.width = width

    def state(self):
        return {"feature_mean": self.feature_mean.clone(), "feature_std": self.feature_std.clone(),
                "target_mean": self.target_mean.clone(), "target_std": self.target_std.clone()}

    def normalize_views(self, views):
        if len(views) != 3:
            raise ValueError("Supply two frozen purpose releases and their raw concatenation")
        first = [_matrix(views[p], f"view_{p}", self.width) for p in range(2)]
        third = _matrix(views[2], "view_2", 2 * self.width)
        if len(first[0]) != len(first[1]) or len(first[0]) != len(third):
            raise ValueError("All three views must have matching rows")
        if not np.array_equal(third, np.concatenate(first, axis=1)):
            raise ValueError("The combined raw view must exactly concatenate the two purpose releases")
        normalized = [(torch.as_tensor(first[p], dtype=torch.float32) - self.feature_mean[p]) / self.feature_std[p]
                      for p in range(2)]
        return normalized + [torch.cat(normalized, dim=1)]

    def normalize_targets(self, targets):
        targets = _matrix(targets, "targets", 3)
        return (torch.as_tensor(targets, dtype=torch.float32) - self.target_mean) / self.target_std

    def denormalize(self, prediction):
        mean = self.target_mean.double().numpy()[list(TARGET_COLUMNS)]
        std = self.target_std.double().numpy()[list(TARGET_COLUMNS)]
        return np.asarray(prediction, dtype=np.float64) * std + mean


def _model_predictions(model, normalized):
    rows = len(normalized[0])
    prediction = np.empty((rows, len(TARGET_NAMES)), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, rows, PREDICTION_BATCH_SIZE):
            batch = [value[start:start + PREDICTION_BATCH_SIZE] for value in normalized]
            prediction[start:start + len(batch[0])] = torch.cat(model(batch), dim=1).double().numpy()
    return prediction


class SavedAdversaryPredictor:
    """Evaluate a cloned saved adversary in its original coordinates, with no fit."""

    def __init__(self, saved_adversaries, release_state, target_mean, target_std):
        width, _ = _architecture(saved_adversaries)
        self.adversaries = copy.deepcopy(saved_adversaries).cpu().eval()
        for parameter in self.adversaries.parameters():
            parameter.requires_grad_(False)
        self.preprocessing = FixedCoordinates(release_state, target_mean, target_std, width)
        self.release_state = self.preprocessing.release_state
        self.target_mean = self.preprocessing.target_mean
        self.target_std = self.preprocessing.target_std
        self.metadata = {
            "kind": "saved_no_fit", "optimizer_steps": 0,
            "source_state_hash": state_digest(clone_state(saved_adversaries)),
            "preprocessing_hash": state_digest(self.preprocessing.state()),
            "target_names": list(TARGET_NAMES), "target_columns": list(TARGET_COLUMNS),
        }

    def predict(self, views):
        return self.preprocessing.denormalize(_model_predictions(self.adversaries, self.preprocessing.normalize_views(views)))

    def score(self, views, y):
        targets = _matrix(y, "targets", 3)[:, list(TARGET_COLUMNS)]
        return predictive_scores(targets, self.predict(views), TARGET_NAMES)


class SelectedAdversaryPredictor:
    """Each target uses its independently validation-selected complete state."""

    def __init__(self, saved_template, preprocessing, states, metadata):
        self.preprocessing = copy.deepcopy(preprocessing)
        self.release_state = self.preprocessing.release_state
        self.target_mean = self.preprocessing.target_mean
        self.target_std = self.preprocessing.target_std
        self.selected_states = [{key: value.clone() for key, value in state.items()} for state in states]
        self.metadata = metadata
        self.models = []
        for state in self.selected_states:
            model = copy.deepcopy(saved_template).cpu().eval()
            model.load_state_dict(state)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models.append(model)

    def predict(self, views):
        normalized = self.preprocessing.normalize_views(views)
        prediction = np.empty((len(normalized[0]), len(TARGET_NAMES)), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(prediction), PREDICTION_BATCH_SIZE):
                for j, model in enumerate(self.models):
                    features = normalized[TARGET_VIEWS[j]][start:start + PREDICTION_BATCH_SIZE]
                    prediction[start:start + len(features), j] = model.networks[TARGET_VIEWS[j]](features)[:, TARGET_OUTPUTS[j]].double().numpy()
        return self.preprocessing.denormalize(prediction)

    def score(self, views, y):
        return predictive_scores(_matrix(y, "targets", 3)[:, list(TARGET_COLUMNS)], self.predict(views), TARGET_NAMES)


def fit_continuations(
    saved_adversaries: ReleaseAdversaries, release_state: dict,
    target_mean: torch.Tensor, target_std: torch.Tensor,
    views_fit: list[np.ndarray], y_fit: np.ndarray,
    views_val: list[np.ndarray], y_val: np.ndarray,
    *, seed: int, out_dir: Path, epochs: int = 80, batch_size: int = 256,
    validation_interval: int = 5, lr: float = .001, kinds=("B", "C"),
) -> dict:
    """Fit two matched trajectories per requested kind, without any test input.

    B starts each trajectory at the exact saved weights. C starts from fresh
    weights with the same architecture. Both use fresh/reset Adam states and
    all five targets' mean normalized MSE. Selection minimizes each target's
    raw validation MSE; ties favor the earliest candidate, then earliest epoch.
    """
    if epochs < 0 or batch_size < 1 or validation_interval < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid continuation budget")
    kinds = tuple(kinds)
    if not kinds or len(set(kinds)) != len(kinds) or any(kind not in ("B", "C") for kind in kinds):
        raise ValueError("kinds must contain B and/or C exactly once")
    started = time.perf_counter()
    width, hidden = _architecture(saved_adversaries)
    source_state = clone_state(saved_adversaries)
    source_hash = state_digest(source_state)
    coordinates = FixedCoordinates(release_state, target_mean, target_std, width)
    fit_views = coordinates.normalize_views(views_fit)
    val_views = coordinates.normalize_views(views_val)
    fit_targets = _matrix(y_fit, "y_fit", 3)
    val_targets = _matrix(y_val, "y_val", 3)
    if not len(fit_targets) or not len(val_targets) or len(fit_views[0]) != len(fit_targets) or len(val_views[0]) != len(val_targets):
        raise ValueError("Fitting and validation rows must match their nonempty releases")
    normalized_targets = coordinates.normalize_targets(fit_targets)
    validation_target_columns = val_targets[:, list(TARGET_COLUMNS)]
    steps_per_epoch = math.ceil(len(fit_targets) / batch_size)
    steps_per_trajectory = epochs * steps_per_epoch
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.save(coordinates.state(), out / "fixed_preprocessing.pt")
    np.savez_compressed(out / "fixed_preprocessing.npz", **{key: value.numpy() for key, value in coordinates.state().items()})
    schedules, schedule_metadata = [], []
    for restart in range(RESTARTS):
        schedule_seed = 980000 + 100 * seed + restart
        schedule = minibatch_schedule(len(fit_targets), batch_size, steps_per_trajectory, schedule_seed)
        schedules.append(schedule)
        flat = torch.cat(schedule).numpy() if schedule else np.empty(0, dtype=np.int64)
        lengths = np.array([len(indices) for indices in schedule], dtype=np.int64)
        artifact = f"schedule_restart_{restart}.npz"
        np.savez_compressed(out / artifact, indices=flat, batch_lengths=lengths,
                            epoch_by_batch=np.repeat(np.arange(1, epochs + 1), steps_per_epoch),
                            optimizer_steps=np.arange(1, steps_per_trajectory + 1),
                            fitting_row_count=np.array(len(fit_targets)))
        exposure = np.bincount(flat, minlength=len(fit_targets))
        schedule_metadata.append({
            "restart_index": restart, "schedule_seed": schedule_seed,
            "schedule_hash": _schedule_hash(schedule), "artifact": artifact,
            "optimizer_steps": steps_per_trajectory, "fitting_observations_exposed": int(len(flat)),
            "per_row_exposure_min": int(exposure.min()), "per_row_exposure_max": int(exposure.max()),
        })
    result = {}
    common = {
        "seed": int(seed), "n_fit": len(fit_targets), "n_validation": len(val_targets),
        "epochs": epochs, "batch_size": batch_size, "validation_interval": validation_interval,
        "restarts": RESTARTS, "optimizer_steps_per_trajectory": steps_per_trajectory,
        "optimizer_steps_total": RESTARTS * steps_per_trajectory,
        "fitting_observations_exposed_total": RESTARTS * epochs * len(fit_targets),
        "target_names": list(TARGET_NAMES), "target_columns": list(TARGET_COLUMNS),
        "architecture": {"purpose_width": width, "hidden": hidden, "outputs": [2, 2, 1], "activation": "ReLU"},
        "source_state_hash": source_hash, "preprocessing_hash": state_digest(coordinates.state()),
        "preprocessing": "provided fixed float32 feature/target coordinates; no refit inside this API",
        "objective": "mean of all five normalized-target MSEs via original _adversary_mse",
        "optimizer": {"name": "Adam", "lr": lr, "betas": [.9, .999], "eps": 1e-8,
                      "weight_decay": 0., "state": "reset independently for every trajectory in both kinds"},
        "selection_rule": "minimum raw validation MSE per target; earliest candidate then earliest epoch",
        "schedules": schedule_metadata,
    }
    with torch.random.fork_rng(devices=[]):
        for kind in kinds:
            kind_started = time.perf_counter()
            directory = out / kind
            directory.mkdir()
            states = [None] * len(TARGET_NAMES)
            selection = {}
            best_keys = [(float("inf"), float("inf"), float("inf"))] * len(TARGET_NAMES)
            candidates = []
            for restart, schedule in enumerate(schedules):
                initialization_seed = 970000 + 100 * seed + restart
                if kind == "B":
                    model = copy.deepcopy(saved_adversaries).cpu()
                    model.load_state_dict(source_state)
                else:
                    torch.manual_seed(initialization_seed)
                    model = ReleaseAdversaries(repr_dim=width, hidden=hidden).float()
                for parameter in model.parameters():
                    parameter.requires_grad_(True)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                initial_hash = state_digest(clone_state(model))
                initial_optimizer = copy.deepcopy(optimizer.state_dict())
                if initial_optimizer["state"]:
                    raise AssertionError("Continuation Adam state must start empty")
                torch.save({"adversaries": clone_state(model), "preprocessing": coordinates.state(),
                            "optimizer": initial_optimizer, "optimizer_steps": 0,
                            "restart_index": restart, "initial_state_hash": initial_hash},
                           directory / f"initialization_restart_{restart}.pt")
                curve = []
                step = 0
                for epoch in range(epochs + 1):
                    if epoch % validation_interval == 0 or epoch == epochs:
                        raw_prediction = coordinates.denormalize(_model_predictions(model, val_views))
                        scores = predictive_scores(validation_target_columns, raw_prediction, TARGET_NAMES)
                        mse = [scores[name]["mse"] for name in TARGET_NAMES]
                        if not all(math.isfinite(value) for value in mse):
                            raise FloatingPointError("Nonfinite validation MSE; retain partial artifacts")
                        curve.append({"epoch": epoch, "optimizer_steps": step,
                                      "fitting_observations_exposed": epoch * len(fit_targets),
                                      "validation_mse": dict(zip(TARGET_NAMES, mse)),
                                      "validation_r2": {name: scores[name]["r2"] for name in TARGET_NAMES}})
                        changed = [j for j, value in enumerate(mse) if (value, restart, epoch) < best_keys[j]]
                        if changed:
                            state = clone_state(model)
                            selected_hash = state_digest(state)
                            for j in changed:
                                states[j] = state
                                best_keys[j] = (mse[j], restart, epoch)
                                selection[TARGET_NAMES[j]] = {
                                    "restart_index": restart, "epoch": epoch, "optimizer_steps": step,
                                    "validation_mse": mse[j], "validation_r2": scores[TARGET_NAMES[j]]["r2"],
                                    "state_hash": selected_hash,
                                }
                    if epoch == epochs:
                        break
                    model.train()
                    for indices in schedule[epoch * steps_per_epoch:(epoch + 1) * steps_per_epoch]:
                        optimizer.zero_grad(set_to_none=True)
                        loss = _adversary_mse(model, [view[indices] for view in fit_views], normalized_targets[indices]).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite fitting objective; retain partial artifacts")
                        loss.backward()
                        optimizer.step()
                        step += 1
                if step != steps_per_trajectory:
                    raise AssertionError("Actual optimizer steps do not match declared trajectory budget")
                final_state = clone_state(model)
                final_hash = state_digest(final_state)
                torch.save({"adversaries": final_state, "preprocessing": coordinates.state(),
                            "optimizer": optimizer.state_dict(), "optimizer_steps": step,
                            "epoch": epochs, "restart_index": restart, "state_hash": final_hash},
                           directory / f"final_restart_{restart}.pt")
                candidate = {
                    "restart_index": restart,
                    "initialization_seed": initialization_seed if kind == "C" else None,
                    "initialization": "exact saved adversary state" if kind == "B" else "fresh matched architecture",
                    "initial_state_hash": initial_hash, "final_state_hash": final_hash,
                    "schedule_hash": schedule_metadata[restart]["schedule_hash"],
                    "schedule_seed": schedule_metadata[restart]["schedule_seed"],
                    "optimizer_steps": step, "fitting_observations_exposed": epochs * len(fit_targets),
                    "validation_curve": curve,
                }
                candidates.append(candidate)
                _dump(directory / f"trajectory_{restart}.json", candidate)
            metadata = {**copy.deepcopy(common), "kind": kind, "candidates": candidates,
                        "selection": selection,
                        "initial_state_hashes": [candidate["initial_state_hash"] for candidate in candidates],
                        "final_state_hashes": [candidate["final_state_hash"] for candidate in candidates],
                        "schedule_hashes": [candidate["schedule_hash"] for candidate in candidates]}
            predictor = SelectedAdversaryPredictor(saved_adversaries, coordinates, states, metadata)
            metadata["selected_validation_scores"] = predictor.score(views_val, val_targets)
            metadata["fit_runtime_seconds"] = time.perf_counter() - kind_started
            for j, state in enumerate(predictor.selected_states):
                torch.save({"adversaries": state, "preprocessing": coordinates.state(),
                            "target_index": j, "target_name": TARGET_NAMES[j], "target_column": TARGET_COLUMNS[j],
                            "selection": selection[TARGET_NAMES[j]]}, directory / f"selected_target_{j}.pt")
            _dump(directory / "metadata.json", metadata)
            result[kind] = predictor
    if state_digest(clone_state(saved_adversaries)) != source_hash:
        raise AssertionError("Input saved adversary changed")
    result["metadata"] = {**common, "kinds": list(kinds),
                          "initial_state_hashes": {kind: result[kind].metadata["initial_state_hashes"] for kind in kinds},
                          "final_state_hashes": {kind: result[kind].metadata["final_state_hashes"] for kind in kinds},
                          "runtime_seconds": time.perf_counter() - started}
    _dump(out / "metadata.json", result["metadata"])
    return result
