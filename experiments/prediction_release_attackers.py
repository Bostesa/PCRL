"""Independent stronger probes for frozen scalar and paired prediction releases.

Only fitting and validation arrays enter this API. All preprocessing is fit
on fitting rows. MLP checkpoints are selected per target by validation MSE;
affine OLS and histogram boosting each use one fixed configuration. Final-test
prediction belongs to the caller after every model/selection is saved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import importlib.metadata
import math
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.nonlinear_conflict_probes import (
    FitPreprocessing, MLPProbe, ProbeConfig, _fitting_inputs, _matrix,
    _network, _write_metadata, fit_affine_probe, predictive_scores,
)
from experiments.nonlinear_conflict_training import clone_state, state_digest


@dataclass(frozen=True)
class MLPAttackConfig:
    hidden: int = 32
    steps: int = 1800
    batch_size: int = 256
    lr: float = .001
    restarts: int = 2
    validation_interval: int = 40
    prediction_batch_size: int = 4096


ATTACKER_CONFIG = {
    "mlp": asdict(MLPAttackConfig()),
    "histgb": {
        "loss": "squared_error", "max_iter": 200, "max_leaf_nodes": 15,
        "learning_rate": .05, "l2_regularization": 1., "early_stopping": False,
        "min_samples_leaf": 20, "max_bins": 255, "max_depth": None,
        "max_features": 1., "categorical_features": None, "warm_start": False,
    },
    "preprocessing": {"constant_tolerance": 1e-10},
    "ols": {"rcond": 1e-10},
}


def _config(config):
    resolved = copy.deepcopy(ATTACKER_CONFIG)
    for section, updates in (config or {}).items():
        if section not in resolved or not isinstance(updates, dict):
            raise ValueError(f"Unknown or malformed attacker configuration section {section!r}")
        unknown = set(updates) - set(resolved[section])
        if unknown:
            raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
        resolved[section].update(updates)
    return resolved


def array_hash(value):
    """Content hash includes normalized array dtype and shape, then C-order bytes."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def full_batch_schedule(n, steps, batch_size, seed):
    """Full batches from a stream of independently shuffled fitting epochs.

    Batches may cross an epoch boundary; every optimizer update receives
    exactly batch_size rows, even when n is not divisible by batch_size.
    """
    if n < 1 or steps < 0 or batch_size < 1:
        raise ValueError("Require fitting rows, nonnegative steps and positive batch size")
    total = steps * batch_size
    rng = np.random.default_rng(seed)
    flat = np.empty(total, dtype=np.int64)
    position = 0
    while position < total:
        order = rng.permutation(n)
        count = min(n, total - position)
        flat[position:position + count] = order[:count]
        position += count
    return flat.reshape(steps, batch_size)


class HistogramProbe:
    def __init__(self, preprocessing, models, target_names, metadata):
        self.preprocessing = preprocessing
        self.models = models
        self.target_names = list(target_names)
        self.metadata = metadata

    def predict(self, x):
        features = self.preprocessing.transform_x(x)
        prediction = np.column_stack([model.predict(features) for model in self.models])
        return self.preprocessing.inverse_y(prediction)

    def score(self, x, y):
        return predictive_scores(y, self.predict(x), self.target_names)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.preprocessing.save(directory)
        joblib.dump(self.models, directory / "models.joblib", compress=3)
        _write_metadata(directory, self.metadata)


def _fit_histgb(xf, yf, xv, yv, target_names, seed, directory, config, common):
    started = time.perf_counter()
    preprocessing = FitPreprocessing.fit(xf, yf, config["preprocessing"]["constant_tolerance"])
    features, labels = preprocessing.transform_x(xf), preprocessing.transform_y(yf)
    parameters = config["histgb"]
    if parameters["early_stopping"] is not False or parameters["warm_start"] is not False:
        raise ValueError("Histogram probes require early_stopping=False and warm_start=False")
    models, seeds = [], []
    for j in range(yf.shape[1]):
        target_seed = int(seed + 1000 + j)
        model = HistGradientBoostingRegressor(**parameters, random_state=target_seed)
        # Validation is deliberately absent from this fixed-configuration fit.
        model.fit(features, labels[:, j])
        if model.n_iter_ != parameters["max_iter"]:
            raise AssertionError("Histogram boosting did not execute the declared iteration budget")
        models.append(model)
        seeds.append(target_seed)
    metadata = {
        **copy.deepcopy(common), "kind": "histgb", "parameters": copy.deepcopy(parameters),
        "target_seeds": seeds, "target_fit": "independent squared-error model per target",
        "selection": "one fixed configuration; validation selects no trees or parameters",
        "actual_iterations_by_target": {name: int(model.n_iter_) for name, model in zip(target_names, models)},
        "fitting_rows_per_tree": len(xf),
        "tree_fit_row_participations_per_target": {name: int(model.n_iter_) * len(xf) for name, model in zip(target_names, models)},
        "optimizer_exposure_comparability": "Tree fitting row participations are recorded separately from MLP minibatch exposures.",
        "dtype": "float64 preprocessing", "device": "cpu", "sklearn_version": importlib.metadata.version("scikit-learn"),
    }
    probe = HistogramProbe(preprocessing, models, target_names, metadata)
    metadata["validation_scores"] = probe.score(xv, yv)
    metadata["fit_runtime_seconds"] = time.perf_counter() - started
    probe.save(directory)
    return probe


def _fit_mlp(xf, yf, xv, yv, target_names, seed, directory, config, common):
    started = time.perf_counter()
    cfg = MLPAttackConfig(**config["mlp"])
    if (cfg.steps < 0 or cfg.batch_size < 1 or cfg.hidden < 1 or cfg.restarts < 1
            or cfg.validation_interval < 1 or not math.isfinite(cfg.lr) or cfg.lr <= 0):
        raise ValueError("Invalid fixed MLP budget")
    directory.mkdir(parents=True, exist_ok=False)
    preprocessing = FitPreprocessing.fit(xf, yf, config["preprocessing"]["constant_tolerance"])
    preprocessing.save(directory)
    fit_x = torch.from_numpy(preprocessing.transform_x(xf).astype(np.float32))
    fit_y = torch.from_numpy(preprocessing.transform_y(yf).astype(np.float32))
    val_x = torch.from_numpy(preprocessing.transform_x(xv).astype(np.float32))
    states = [None] * len(target_names)
    best = [(float("inf"), float("inf"), float("inf"))] * len(target_names)
    selection, candidates = {}, []
    with torch.random.fork_rng(devices=[]):
        for restart in range(cfg.restarts):
            initialization_seed = int(seed + restart)
            schedule_seed = int(seed + 10000 + restart)
            schedule = full_batch_schedule(len(xf), cfg.steps, cfg.batch_size, schedule_seed)
            schedule_hash = array_hash(schedule)
            exposures = np.bincount(schedule.ravel(), minlength=len(xf))
            np.savez_compressed(directory / f"schedule_restart_{restart}.npz", indices=schedule,
                                optimizer_steps=np.arange(1, cfg.steps + 1), row_exposures=exposures)
            torch.manual_seed(initialization_seed)
            model = _network(xf.shape[1], yf.shape[1], cfg.hidden)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            initial_state = clone_state(model)
            initial_hash = state_digest(initial_state)
            torch.save({"state_dict": initial_state, "optimizer": optimizer.state_dict(),
                        "optimizer_steps": 0, "row_exposures": 0,
                        "initialization_seed": initialization_seed, "state_hash": initial_hash},
                       directory / f"initialization_restart_{restart}.pt")
            curve = []
            for step in range(cfg.steps + 1):
                if step % cfg.validation_interval == 0 or step == cfg.steps:
                    model.eval()
                    with torch.no_grad():
                        prediction = preprocessing.inverse_y(model(val_x).double().numpy())
                    scores = predictive_scores(yv, prediction, target_names)
                    mse = [scores[name]["mse"] for name in target_names]
                    if not all(math.isfinite(value) for value in mse):
                        raise FloatingPointError("Nonfinite MLP validation score; partial fitting artifacts retained")
                    curve.append({"optimizer_steps": step, "row_exposures": step * cfg.batch_size,
                                  "validation_mse": {name: scores[name]["mse"] for name in target_names},
                                  "validation_r2": {name: scores[name]["r2"] for name in target_names}})
                    changed = [j for j, value in enumerate(mse) if (value, restart, step) < best[j]]
                    if changed:
                        state = clone_state(model)
                        digest = state_digest(state)
                        for j in changed:
                            states[j] = state
                            best[j] = (mse[j], restart, step)
                            selection[target_names[j]] = {
                                "restart_index": restart, "initialization_seed": initialization_seed,
                                "optimizer_steps": step, "row_exposures": step * cfg.batch_size,
                                "validation_mse": mse[j], "validation_r2": scores[target_names[j]]["r2"],
                                "state_hash": digest,
                            }
                if step == cfg.steps:
                    break
                indices = schedule[step]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = (model(fit_x[indices]) - fit_y[indices]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite MLP fitting objective; partial fitting artifacts retained")
                loss.backward()
                optimizer.step()
            final_state = clone_state(model)
            final_hash = state_digest(final_state)
            torch.save({"state_dict": final_state, "optimizer": optimizer.state_dict(),
                        "optimizer_steps": cfg.steps, "row_exposures": cfg.steps * cfg.batch_size,
                        "initialization_seed": initialization_seed, "state_hash": final_hash},
                       directory / f"final_restart_{restart}.pt")
            candidate = {
                "restart_index": restart, "initialization_seed": initialization_seed,
                "schedule_seed": schedule_seed, "schedule_hash": schedule_hash,
                "schedule_artifact": f"schedule_restart_{restart}.npz",
                "initial_state_hash": initial_hash, "final_state_hash": final_hash,
                "optimizer_steps": cfg.steps, "row_exposures": int(schedule.size),
                "row_exposure_min": int(exposures.min()), "row_exposure_max": int(exposures.max()),
                "validation_curve": curve,
            }
            candidates.append(candidate)
            _write_metadata(directory, {**common, "kind": "mlp_in_progress", "completed_candidates": candidates})
    metadata = {
        **copy.deepcopy(common), "kind": "mlp", "config": asdict(cfg),
        "architecture": "same prior Linear(d,32)-ReLU-Linear(32,32)-ReLU-Linear(32,targets); hidden size configurable only before freeze",
        "objective": "mean squared standardized-target error across all supplied targets",
        "selection_criterion": "minimum raw validation MSE per target; earliest candidate then earliest update breaks ties",
        "selection": selection, "candidates": candidates,
        "optimizer_steps_per_restart": cfg.steps, "optimizer_steps_total": cfg.steps * cfg.restarts,
        "row_exposures_per_restart": cfg.steps * cfg.batch_size,
        "row_exposures_total": cfg.steps * cfg.batch_size * cfg.restarts,
        "validation_opportunities_per_target": sum(len(candidate["validation_curve"]) for candidate in candidates),
        "optimizer": {"name": "Adam", "lr": cfg.lr, "betas": [.9, .999], "eps": 1e-8,
                      "weight_decay": 0., "state": "fresh/reset for each restart"},
        "dtype": "float32", "device": "cpu",
    }
    probe = MLPProbe(preprocessing, states, target_names, metadata, cfg)
    metadata["validation_scores"] = probe.score(xv, yv)
    metadata["fit_runtime_seconds"] = time.perf_counter() - started
    for j, state in enumerate(probe.selected_states):
        torch.save({"state_dict": state, "target_index": j, "target_name": target_names[j],
                    "selection": selection[target_names[j]], "config": asdict(cfg)},
                   directory / f"selected_target_{j}.pt")
    _write_metadata(directory, metadata)
    return probe


def fit_attackers(
    X_fit, y_fit, X_val, y_val, *, target_names, seed, out_dir,
    kinds=("linear", "mlp", "histgb"), config=None,
):
    """Fit/save only requested probe families using fitting and validation rows.

    Returned dict contains one predictor per requested kind, with ``predict``,
    ``score`` and ``metadata``. Scores map target names to raw MSE, unclipped R²,
    variance, n, and defined status. Utility probes use the same API with the
    appropriate authorized target subset and ``kinds=('mlp',)``.
    """
    started = time.perf_counter()
    kinds = tuple(kinds)
    if not kinds or len(set(kinds)) != len(kinds) or any(kind not in ("linear", "mlp", "histgb") for kind in kinds):
        raise ValueError("Request each of linear, mlp, histgb at most once")
    resolved = _config(config)
    xf, yf, xv, yv = _fitting_inputs(X_fit, y_fit, X_val, y_val, target_names)
    common = {
        "target_names": list(target_names), "n_fit": len(xf), "n_validation": len(xv),
        "input_dim": xf.shape[1], "target_dim": yf.shape[1], "seed": int(seed),
        "fit_array_hashes": {"features": array_hash(xf), "targets": array_hash(yf)},
        "validation_array_hashes": {"features": array_hash(xv), "targets": array_hash(yv)},
        "array_hash_definition": "dtype string then shape string then C-order bytes after float64 input validation",
        "preprocessing_fit_split": "fitting rows only", "intercept_fit_split": "fitting rows only",
        "preprocessing": copy.deepcopy(resolved["preprocessing"]),
        "model_versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "scikit-learn")},
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    probes = {}
    for kind in kinds:
        if kind == "linear":
            cfg = ProbeConfig(ols_rcond=resolved["ols"]["rcond"], constant_tolerance=resolved["preprocessing"]["constant_tolerance"])
            probe = fit_affine_probe(xf, yf, xv, yv, target_names=target_names, config=cfg)
            probe.metadata.update(copy.deepcopy(common))
            probe.metadata["fitting_rows"] = len(xf)
            probe.metadata["selection"] = "one fixed affine OLS fit; no validation selection"
            probe.save(out / kind)
        elif kind == "mlp":
            probe = _fit_mlp(xf, yf, xv, yv, target_names, int(seed), out / kind, resolved, common)
        else:
            probe = _fit_histgb(xf, yf, xv, yv, target_names, int(seed), out / kind, resolved, common)
        probes[kind] = probe
    _write_metadata(out, {**common, "kinds": list(kinds), "config": resolved,
                         "fit_runtime_seconds": time.perf_counter() - started,
                         "fitted_families": {kind: probe.metadata for kind, probe in probes.items()}})
    return probes
