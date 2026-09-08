"""Independent affine and nonlinear regression probes for the conflict pilot.

Only fitting and validation arrays enter fitting. Preprocessing and intercepts
use fitting rows; MLP initialization/epoch selection uses validation MSE for
each target separately. Call ``predict`` or ``score`` after fitting to evaluate
another split. Nothing in this module clips negative predictive R².
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ProbeConfig:
    hidden: int = 32
    epochs: int = 80
    validation_interval: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    initialization_offsets: tuple[int, ...] = (0, 1)
    ols_rcond: float = 1e-10
    constant_tolerance: float = 1e-10
    prediction_batch_size: int = 4096


PROBE_CONFIG = asdict(ProbeConfig())


def _resolve_config(config):
    if config is None:
        return ProbeConfig()
    return ProbeConfig(**config) if isinstance(config, dict) else config


def _matrix(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite matrix with at least one column")
    return values


def predictive_scores(y, predictions, target_names):
    """Per-target out-of-sample scores; SST uses evaluation target means."""
    y = _matrix(y, "targets")
    predictions = _matrix(predictions, "predictions")
    if y.shape != predictions.shape or len(target_names) != y.shape[1] or not len(y):
        raise ValueError("Target/prediction shapes and target names must match")
    mse = np.square(y - predictions).mean(axis=0)
    variance = y.var(axis=0)
    return {
        name: {
            "r2": float(1 - mse[j] / variance[j]) if variance[j] > 0 else float("nan"),
            "mse": float(mse[j]),
            "n": len(y),
            "variance": float(variance[j]),
            "defined": bool(variance[j] > 0),
        }
        for j, name in enumerate(target_names)
    }


@dataclass
class FitPreprocessing:
    x_mean: np.ndarray
    x_scale: np.ndarray
    x_active: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray

    @classmethod
    def fit(cls, x, y, tolerance):
        x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
        x_std, y_std = x.std(axis=0), y.std(axis=0)
        # Avoid amplifying numerical remnants of exact erasure. Thresholds
        # depend only on fitting data and the fixed precommitted tolerance.
        x_active = x_std > tolerance * max(float(x_std.max()), 1.0)
        x_scale = np.where(x_active, x_std, 1.0)
        y_scale = np.where(y_std > 0, y_std, 1.0)
        return cls(x_mean, x_scale, x_active, y_mean, y_scale)

    def transform_x(self, x):
        x = _matrix(x, "features")
        if x.shape[1] != len(self.x_mean):
            raise ValueError("Feature dimension differs from the fitting schema")
        return ((x - self.x_mean) / self.x_scale) * self.x_active

    def transform_y(self, y):
        return (_matrix(y, "targets") - self.y_mean) / self.y_scale

    def inverse_y(self, predictions):
        return predictions * self.y_scale + self.y_mean

    def save(self, directory):
        np.savez_compressed(directory / "preprocessing.npz", **self.__dict__)


class AffineProbe:
    def __init__(self, preprocessing, coefficients, target_names, metadata):
        self.preprocessing = preprocessing
        self.coefficients = coefficients
        self.target_names = list(target_names)
        self.metadata = metadata

    def predict(self, x):
        return self.preprocessing.inverse_y(self.preprocessing.transform_x(x) @ self.coefficients)

    def score(self, x, y):
        return predictive_scores(y, self.predict(x), self.target_names)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.preprocessing.save(directory)
        np.savez_compressed(directory / "model.npz", coefficients=self.coefficients)
        _write_metadata(directory, self.metadata)


def _network(input_dim, output_dim, hidden):
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output_dim),
    )


class MLPProbe:
    def __init__(self, preprocessing, selected_states, target_names, metadata, config):
        self.preprocessing = preprocessing
        self.target_names = list(target_names)
        self.metadata = metadata
        self.config = config
        self.selected_states = selected_states
        self.models = []
        # Reconstruction should not perturb another experiment's RNG state.
        with torch.random.fork_rng(devices=[]):
            for state in selected_states:
                model = _network(len(preprocessing.x_mean), len(target_names), config.hidden)
                model.load_state_dict(state)
                self.models.append(model.eval())

    def predict(self, x):
        features = self.preprocessing.transform_x(x).astype(np.float32)
        prediction = np.empty((len(features), len(self.target_names)), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(features), self.config.prediction_batch_size):
                xb = torch.from_numpy(features[start:start + self.config.prediction_batch_size])
                for j, model in enumerate(self.models):
                    prediction[start:start + len(xb), j] = model(xb)[:, j].numpy()
        return self.preprocessing.inverse_y(prediction)

    def score(self, x, y):
        return predictive_scores(y, self.predict(x), self.target_names)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.preprocessing.save(directory)
        for j, state in enumerate(self.selected_states):
            torch.save({
                "state_dict": state,
                "target_index": j,
                "target_name": self.target_names[j],
                "selection": self.metadata["selection"][self.target_names[j]],
                "config": asdict(self.config),
            }, directory / f"selected_target_{j}.pt")
        _write_metadata(directory, self.metadata)


def _write_metadata(directory, metadata):
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    (directory / "metadata.json").write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False) + "\n")


def _fitting_inputs(x_fit, y_fit, x_val, y_val, target_names):
    xf, yf = _matrix(x_fit, "x_fit"), _matrix(y_fit, "y_fit")
    xv, yv = _matrix(x_val, "x_val"), _matrix(y_val, "y_val")
    if not len(xf) or not len(xv) or len(xf) != len(yf) or len(xv) != len(yv):
        raise ValueError("Fitting and validation splits require matching nonempty rows")
    if xf.shape[1] != xv.shape[1] or yf.shape[1] != yv.shape[1]:
        raise ValueError("Fitting and validation schemas differ")
    if len(target_names) != yf.shape[1] or len(set(target_names)) != len(target_names):
        raise ValueError("Provide one unique target name per target column")
    return xf, yf, xv, yv


def fit_affine_probe(x_fit, y_fit, x_val, y_val, *, target_names, config=None):
    """Fit affine OLS on fitting rows; validation is evaluated but selects nothing."""
    started = time.perf_counter()
    config = _resolve_config(config)
    xf, yf, xv, yv = _fitting_inputs(x_fit, y_fit, x_val, y_val, target_names)
    preprocessing = FitPreprocessing.fit(xf, yf, config.constant_tolerance)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        preprocessing.transform_x(xf), preprocessing.transform_y(yf), rcond=config.ols_rcond,
    )
    metadata = {
        "kind": "affine_ols", "target_names": list(target_names),
        "n_fit": len(xf), "n_validation": len(xv), "input_dim": xf.shape[1],
        "preprocessing_fit_split": "fitting", "intercept_fit_split": "fitting",
        "selection": "single fixed OLS fit; validation selects nothing",
        "rank": int(rank), "singular_values": singular_values.tolist(),
        "config": asdict(config), "dtype": "float64", "device": "cpu",
    }
    probe = AffineProbe(preprocessing, coefficients, target_names, metadata)
    metadata["validation_scores"] = probe.score(xv, yv)
    metadata["fit_runtime_seconds"] = time.perf_counter() - started
    return probe


def fit_mlp_probe(x_fit, y_fit, x_val, y_val, *, target_names, seed, config=None):
    """Choose each target's initialization and epoch by validation MSE only."""
    started = time.perf_counter()
    config = _resolve_config(config)
    if (config.hidden < 1 or config.epochs < 0 or config.batch_size < 1 or config.lr <= 0
            or config.validation_interval < 1 or not config.initialization_offsets):
        raise ValueError("Invalid fixed probe configuration")
    xf, yf, xv, yv = _fitting_inputs(x_fit, y_fit, x_val, y_val, target_names)
    preprocessing = FitPreprocessing.fit(xf, yf, config.constant_tolerance)
    xft = torch.from_numpy(preprocessing.transform_x(xf).astype(np.float32))
    yft = torch.from_numpy(preprocessing.transform_y(yf).astype(np.float32))
    xvt = torch.from_numpy(preprocessing.transform_x(xv).astype(np.float32))
    best = np.full(yf.shape[1], np.inf)
    selected_states = [None] * yf.shape[1]
    selection = {}
    history = []
    candidates = [int(seed + offset) for offset in config.initialization_offsets]
    with torch.random.fork_rng(devices=[]):
        for candidate_index, candidate_seed in enumerate(candidates):
            torch.manual_seed(candidate_seed)
            model = _network(xf.shape[1], yf.shape[1], config.hidden)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            rng = np.random.default_rng(candidate_seed)
            optimizer_steps = 0
            for epoch in range(config.epochs + 1):
                if epoch % config.validation_interval == 0 or epoch == config.epochs:
                    model.eval()
                    with torch.no_grad():
                        prediction = preprocessing.inverse_y(model(xvt).numpy().astype(np.float64))
                    mse = np.square(prediction - yv).mean(axis=0)
                    if not np.isfinite(mse).all():
                        raise FloatingPointError("MLP validation produced a nonfinite score")
                    history.append({"initialization_index": candidate_index,
                                    "initialization_seed": candidate_seed, "epoch": epoch,
                                    "optimizer_steps": optimizer_steps, "validation_mse": mse.tolist()})
                    improved = mse < best
                    if improved.any():
                        state = copy.deepcopy(model.state_dict())
                        for j in np.flatnonzero(improved):
                            selected_states[j] = state
                            best[j] = mse[j]
                            selection[target_names[j]] = {
                                "initialization_index": candidate_index,
                                "initialization_seed": candidate_seed, "epoch": epoch,
                                "optimizer_steps": optimizer_steps, "validation_mse": float(mse[j]),
                            }
                if epoch == config.epochs:
                    break
                model.train()
                order = rng.permutation(len(xft))
                for start in range(0, len(order), config.batch_size):
                    indices = order[start:start + config.batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(xft[indices]) - yft[indices]).square().mean()
                    loss.backward()
                    optimizer.step()
                    optimizer_steps += 1
    metadata = {
        "kind": "mlp_regression", "target_names": list(target_names),
        "n_fit": len(xf), "n_validation": len(xv), "input_dim": xf.shape[1],
        "preprocessing_fit_split": "fitting", "intercept_fit_split": "fitting",
        "selection_criterion": "minimum validation MSE independently per target; earliest candidate/epoch breaks ties",
        "selection": selection, "initialization_seeds": candidates, "history": history,
        "architecture": "Linear(d,hidden)-ReLU-Linear(hidden,hidden)-ReLU-Linear(hidden,targets)",
        "target_training": "shared trunk; mean squared standardized-target error; separate selected full state per target",
        "optimizer": {"name": "Adam", "lr": config.lr, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0.0},
        "config": asdict(config), "dtype": "float32", "device": "cpu",
    }
    probe = MLPProbe(preprocessing, selected_states, target_names, metadata, config)
    metadata["validation_scores"] = probe.score(xv, yv)
    metadata["fit_runtime_seconds"] = time.perf_counter() - started
    return probe


def fit_probes(x_fit, y_fit, x_val, y_val, *, target_names, seed, out_dir, config=None):
    """Fit both fixed probe families and save independently reviewable artifacts."""
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=False)
    probes = {
        "ols": fit_affine_probe(x_fit, y_fit, x_val, y_val, target_names=target_names, config=config),
        "mlp": fit_mlp_probe(x_fit, y_fit, x_val, y_val, target_names=target_names, seed=seed, config=config),
    }
    for name, probe in probes.items():
        probe.save(output / name)
    return probes


def fit_probe(kind, X_fit, y_fit, X_val, y_val, *, seed, artifact_dir, target_names, config=None):
    """Fit one precommitted family, save its fit/selection evidence, and return it.

    ``kind`` is ``linear`` or ``mlp``. The artifact directory must be fresh.
    Returned scores map each target name to ``r2,mse,n,variance,defined``.
    Undefined R² is NaN in memory and JSON null in saved metadata.
    """
    if kind == "linear":
        probe = fit_affine_probe(X_fit, y_fit, X_val, y_val, target_names=target_names, config=config)
    elif kind == "mlp":
        probe = fit_mlp_probe(X_fit, y_fit, X_val, y_val, target_names=target_names, seed=seed, config=config)
    else:
        raise ValueError("Probe kind must be 'linear' or 'mlp'")
    probe.save(artifact_dir)
    return probe
