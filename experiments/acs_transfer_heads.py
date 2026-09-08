"""Fixed-budget downstream heads and auditors for frozen ACS releases.

Fitting and validation are the only data accepted by the fitting API. Input
centering/scaling and all predictor intercepts use fitting rows. A single MLP
trajectory selects an epoch by validation log loss; candidate families are then
selected by that same criterion. No representation is trained here. Labels use
the caller's fixed integer schema 0..K-1, including absent categories.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


HEAD_BUDGET = {
    "logistic": {"C": 1., "max_iter": 500, "solver": "lbfgs", "tol": 1e-4},
    "mlp": {"hidden": [64, 32], "epochs": 40, "batch_size": 256,
            "lr": .001, "validation_interval": 5, "weight_decay": 0.},
    "histgb": {"max_iter": 150, "max_leaf_nodes": 15, "learning_rate": .1,
               "l2_regularization": 1., "min_samples_leaf": 20,
               "early_stopping": False},
    "preprocessing": {"std_floor": 1e-12},
    "prior": {"pseudocount": 1.},
}
PROBABILITY_FLOOR = 1e-12


def _hash_array(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(_hash_array(value.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def _labels(y, n_classes):
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 2:
        raise ValueError("n_classes must declare a fixed schema of at least two classes")
    value = np.asarray(y)
    if value.ndim != 1 or not (np.issubdtype(value.dtype, np.number) or value.dtype == bool):
        raise ValueError("Labels must be one-dimensional numeric class indices")
    if not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
        raise ValueError("Labels must be finite integer class indices")
    value = value.astype(np.int64, copy=False)
    if ((value < 0) | (value >= n_classes)).any():
        raise ValueError("Label outside the fixed class schema")
    return value


def _features(x):
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 1:
        raise ValueError("Features must be a numeric matrix with at least one column")
    return value


def _weights(weights, n):
    value = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if value.shape != (n,) or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Weights must be a finite nonnegative vector aligned with rows")
    if not np.isfinite(value.sum()):
        raise ValueError("Weights must have a finite total")
    return value


def metrics(y, probs, n_classes, weights=None):
    """JSON-safe predictive metrics; missing categories never silently disappear.

    Log loss uses natural logarithms and a fixed 1e-12 numerical probability
    floor. Full-schema balanced accuracy/macro AUROC are undefined when any
    category lacks positive evaluation weight. Observed-category summaries are
    separately named. Weights change scoring only, never the fitted predictor.
    """
    y = _labels(y, n_classes)
    p = np.asarray(probs, dtype=np.float64)
    if (p.shape != (len(y), n_classes) or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1+1e-8).any()
            or not np.allclose(p.sum(1), 1., atol=1e-6, rtol=0)):
        raise ValueError("Probabilities must be finite, normalized and aligned with the full class schema")
    weight = _weights(weights, len(y))
    total = float(weight.sum())
    support = np.bincount(y, minlength=n_classes)
    weighted_support = np.bincount(y, weights=weight, minlength=n_classes)
    prediction = p.argmax(1)
    predicted_support = np.bincount(prediction, minlength=n_classes)
    predicted_weight = np.bincount(prediction, weights=weight, minlength=n_classes)
    coverage = weighted_support > 0
    rows = []
    for k in range(n_classes):
        positive = y == k
        true_positive = float(weight[positive & (prediction == k)].sum())
        recall = true_positive / float(weighted_support[k]) if coverage[k] else None
        precision = true_positive / float(predicted_weight[k]) if predicted_weight[k] > 0 else None
        auc = None
        if weighted_support[k] > 0 and total-weighted_support[k] > 0:
            auc = float(roc_auc_score(positive.astype(int), p[:, k], sample_weight=weight))
        denom = float(weighted_support[k] + predicted_weight[k])
        rows.append({"class_index": k, "support": int(support[k]),
                     "weighted_support": float(weighted_support[k]),
                     "prevalence": float(weighted_support[k]/total) if total > 0 else None,
                     "predicted_support": int(predicted_support[k]),
                     "predicted_weight": float(predicted_weight[k]),
                     "precision": precision, "recall": recall,
                     "f1": 2*true_positive/denom if denom > 0 else None,
                     "auroc": auc, "recall_defined": recall is not None,
                     "auroc_defined": auc is not None})
    recalls = [row["recall"] for row in rows if row["recall"] is not None]
    aucs = [row["auroc"] for row in rows if row["auroc"] is not None]
    clipped = np.clip(p, PROBABILITY_FLOOR, 1.)
    clipped /= clipped.sum(1, keepdims=True)
    loss = float(np.dot(weight, -np.log(clipped[np.arange(len(y)), y]))/total) if total > 0 else None
    complete = bool(coverage.all())
    macro_auc = float(np.mean(aucs)) if len(aucs) == n_classes else None
    return {"n": len(y), "n_classes": int(n_classes), "class_schema": list(range(n_classes)),
            "weighted": weights is not None, "weight_sum": total,
            "support": support.tolist(), "weighted_support": weighted_support.tolist(),
            "prevalence": [row["prevalence"] for row in rows],
            "coverage_complete": complete, "raw_coverage_complete": bool((support > 0).all()),
            "valid_class_mask": coverage.tolist(),
            "log_loss": loss, "log_loss_probability_floor": PROBABILITY_FLOOR,
            "accuracy": float(np.dot(weight, prediction == y)/total) if total > 0 else None,
            "balanced_accuracy": float(np.mean(recalls)) if complete else None,
            "observed_balanced_accuracy": float(np.mean(recalls)) if recalls else None,
            "auroc": rows[1]["auroc"] if n_classes == 2 else macro_auc,
            "macro_auroc": macro_auc,
            "observed_macro_auroc": float(np.mean(aucs)) if aucs else None,
            "per_class": rows}


@dataclass
class InputStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    fitted: bool = True

    @classmethod
    def fit(cls, x, std_floor):
        mean = x.mean(0)
        std = x.std(0)
        return cls(mean.copy(), np.where(std > std_floor, std, 1.))

    def transform(self, x):
        x = _features(x)
        if x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Prediction features must match the fitted finite feature schema")
        return (x-self.mean)/self.scale


class PriorPredictor:
    def __init__(self, probabilities, metadata):
        self.probabilities = np.asarray(probabilities, dtype=np.float64).copy()
        self.n_classes = len(self.probabilities)
        self.metadata = metadata
        self.family = metadata.get("family", "prior")
        self.preprocessing = None

    def predict_proba(self, x):
        return np.broadcast_to(self.probabilities, (len(x), self.n_classes)).copy()

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory/"prior.npz", probabilities=self.probabilities)
        (directory/"metadata.json").write_text(json.dumps(self.metadata, indent=2, allow_nan=False)+"\n")


def fit_prior(yfit, n_classes, weights=None, pseudocount=1.):
    """Smoothed prior fitted only to the designated fitting labels/weights."""
    y = _labels(yfit, n_classes)
    weight = _weights(weights, len(y))
    if (len(y) < 1 or weight.sum() <= 0 or not np.isfinite(pseudocount)
            or pseudocount < 0):
        raise ValueError("Prior requires positive fitting weight and a nonnegative pseudocount")
    support = np.bincount(y, minlength=n_classes)
    counts = np.bincount(y, weights=weight, minlength=n_classes)
    probabilities = (counts+pseudocount)/(counts.sum()+n_classes*pseudocount)
    return PriorPredictor(probabilities, {
        "family": "prior", "n_classes": int(n_classes), "fit_support": support.tolist(),
        "fit_weighted_support": counts.tolist(), "fit_weighted": weights is not None,
        "fit_coverage_complete": bool((counts > 0).all()), "pseudocount_per_class": pseudocount,
        "fit_rows": len(y), "optimizer_steps": 0, "training_row_exposures": len(y),
        "fit_label_hash": _hash_array(y), "probabilities": probabilities.tolist()})


def exposed_probabilities(y, n_classes):
    """One-hot diagnostic features; pass these through ordinary fitted auditors."""
    return np.eye(n_classes, dtype=np.float64)[_labels(y, n_classes)]


class FittedCandidate:
    def __init__(self, family, model, preprocessing, n_classes, metadata):
        self.family, self.model = family, model
        self.preprocessing, self.n_classes = preprocessing, n_classes
        self.metadata = metadata

    def predict_proba(self, x):
        features = self.preprocessing.transform(x)
        if self.family == "mlp":
            self.model.eval()
            outputs = []
            with torch.no_grad():
                for start in range(0, len(features), 4096):
                    logits = self.model(torch.as_tensor(features[start:start+4096], dtype=torch.float32))
                    outputs.append(torch.softmax(logits, dim=1).double().numpy())
            return np.concatenate(outputs) if outputs else np.empty((0, self.n_classes))
        probabilities = np.zeros((len(features), self.n_classes), dtype=np.float64)
        if len(features):
            probabilities[:, np.asarray(self.model.classes_, dtype=int)] = self.model.predict_proba(features)
        return probabilities

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory/"preprocessing.npz", mean=self.preprocessing.mean,
                            scale=self.preprocessing.scale)
        if self.family == "mlp":
            torch.save({"state": self.model.state_dict(), "n_classes": self.n_classes,
                        "input_dim": len(self.preprocessing.mean),
                        "hidden": self.metadata["parameters"]["hidden"]}, directory/"model.pt")
        else:
            joblib.dump(self.model, directory/"model.joblib", compress=3)
        (directory/"metadata.json").write_text(json.dumps(self.metadata, indent=2, allow_nan=False)+"\n")


def _network(input_dim, hidden, n_classes):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden[0]), torch.nn.ReLU(),
                               torch.nn.Linear(hidden[0], hidden[1]), torch.nn.ReLU(),
                               torch.nn.Linear(hidden[1], n_classes))


def load_candidate(directory):
    """Reload saved selected weights with their original fitting standardizer."""
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text())
    if (directory/"prior.npz").exists():
        with np.load(directory/"prior.npz") as arrays:
            return PriorPredictor(arrays["probabilities"], metadata)
    with np.load(directory/"preprocessing.npz") as arrays:
        preprocessing = InputStandardizer(arrays["mean"].copy(), arrays["scale"].copy())
    family = metadata["family"]
    if family == "mlp":
        checkpoint = torch.load(directory/"model.pt", map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            model = _network(checkpoint["input_dim"], checkpoint["hidden"], checkpoint["n_classes"])
        model.load_state_dict(checkpoint["state"])
        model.eval()
        model.requires_grad_(False)
    else:
        model = joblib.load(directory/"model.joblib")
    return FittedCandidate(family, model, preprocessing, metadata["n_classes"], metadata)


def _budget(budget):
    cfg = copy.deepcopy(HEAD_BUDGET)
    for key, updates in (budget or {}).items():
        if key not in cfg or not isinstance(updates, dict) or set(updates)-set(cfg[key]):
            raise ValueError(f"Unknown or malformed fixed head budget: {key}")
        cfg[key].update(updates)
    m = cfg["mlp"]
    if (len(m["hidden"]) != 2 or min(m["hidden"]) < 1 or m["epochs"] < 0
            or m["batch_size"] < 1 or m["validation_interval"] < 1
            or not np.isfinite(m["lr"]) or m["lr"] <= 0):
        raise ValueError("Invalid fixed MLP budget")
    if cfg["histgb"]["early_stopping"] is not False:
        raise ValueError("Histogram early stopping must remain disabled")
    return cfg


def _fit_mlp(x, y, xv, yv, standardizer, n_classes, seed, cfg, common):
    parameters = cfg["mlp"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _network(x.shape[1], parameters["hidden"], n_classes).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["lr"],
                                 weight_decay=parameters["weight_decay"])
    xf = torch.as_tensor(standardizer.transform(x), dtype=torch.float32)
    target = torch.as_tensor(y, dtype=torch.long)
    candidate = FittedCandidate("mlp", model, standardizer, n_classes, {})
    initial_hash = _state_hash(model)
    curve, best_state, best_key = [], None, (float("inf"), -1)
    selected_steps, optimizer_steps = 0, 0
    exposures = np.zeros(len(y), dtype=np.int64)
    schedule_digest = hashlib.sha256()
    rng = np.random.default_rng(seed+700000)

    def numerical_failure(message):
        error = FloatingPointError(message)
        error.fit_accounting = {"optimizer_steps": optimizer_steps,
                                "training_row_exposures": int(exposures.sum())}
        raise error

    def evaluate(epoch):
        nonlocal best_state, best_key, selected_steps
        probability = candidate.predict_proba(xv)
        if not np.isfinite(probability).all():
            numerical_failure("Nonfinite MLP validation probabilities")
        score = metrics(yv, probability, n_classes)
        row = {"epoch": epoch, "optimizer_steps": optimizer_steps,
               "validation_log_loss": score["log_loss"]}
        curve.append(row)
        key = (score["log_loss"], epoch)
        if key < best_key:
            best_key, selected_steps = key, optimizer_steps
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    evaluate(0)
    for epoch in range(1, parameters["epochs"]+1):
        order = rng.permutation(len(y))
        schedule_digest.update(order.tobytes())
        for start in range(0, len(order), parameters["batch_size"]):
            indices = order[start:start+parameters["batch_size"]]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xf[indices]), target[indices])
            if not torch.isfinite(loss):
                numerical_failure("Nonfinite MLP training loss")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            exposures[indices] += 1
        if epoch % parameters["validation_interval"] == 0 or epoch == parameters["epochs"]:
            evaluate(epoch)
    final_hash = _state_hash(model)
    model.load_state_dict(best_state)
    model.eval()
    model.requires_grad_(False)
    candidate.metadata = {
        **common, "family": "mlp", "parameters": copy.deepcopy(parameters),
        "optimizer": {"name": "Adam", "betas": [.9, .999], "eps": 1e-8},
        "initial_state_hash": initial_hash, "final_state_hash": final_hash,
        "selected_state_hash": _state_hash(model), "initialization_seed": seed,
        "schedule_seed": seed+700000, "schedule_hash": schedule_digest.hexdigest(),
        "restarts": 1, "optimizer_steps": optimizer_steps,
        "selected_optimizer_steps": selected_steps, "selected_epoch": best_key[1],
        "training_row_exposures": int(exposures.sum()),
        "row_exposure_min": int(exposures.min()), "row_exposure_max": int(exposures.max()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "validation_curve": curve, "selection": "minimum validation log loss, then earliest epoch"}
    return candidate


def fit_candidates(xfit, yfit, xval, yval, n_classes, seed,
                   families=("logistic", "mlp"), budget=None):
    """Fit one candidate per requested family and select only on validation.

    Fitting is unweighted. Caller-supplied survey weights may be used afterward
    for scoring sensitivity through metrics(..., weights=...). Missing fitting
    categories remain explicit. A one-class fit or nonfinite numerical fit uses
    a labeled fitting-prior fallback; no absent class becomes a protection pass.
    """
    started = time.perf_counter()
    cfg = _budget(budget)
    xf, xv = _features(xfit), _features(xval)
    yf, yv = _labels(yfit, n_classes), _labels(yval, n_classes)
    if (len(xf) != len(yf) or len(xv) != len(yv) or not len(yf) or not len(yv)
            or xf.shape[1] != xv.shape[1]):
        raise ValueError("Aligned nonempty fitting/validation data and matching feature schemas required")
    if (not families or len(set(families)) != len(families)
            or set(families)-{"logistic", "mlp", "histgb"}):
        raise ValueError("Specify unique supported candidate families")
    support = np.bincount(yf, minlength=n_classes)
    prior = fit_prior(yf, n_classes, pseudocount=cfg["prior"]["pseudocount"])
    common = {"n_classes": int(n_classes), "class_schema": list(range(n_classes)),
              "fit_rows": len(yf), "validation_rows": len(yv), "input_dim": xf.shape[1],
              "fit_support": support.tolist(), "fit_coverage_complete": bool((support > 0).all()),
              "fit_weighted": False, "seed": int(seed), "device": "cpu",
              "fit_hashes": {"x": _hash_array(xf), "y": _hash_array(yf)},
              "validation_hashes": {"x": _hash_array(xv), "y": _hash_array(yv)}}
    reason = ("nonfinite_features" if not np.isfinite(xf).all() or not np.isfinite(xv).all()
              else "single_fitting_class" if (support > 0).sum() < 2 else None)
    standardizer = InputStandardizer.fit(xf, cfg["preprocessing"]["std_floor"]) if reason is None else None
    if standardizer is not None and (not np.isfinite(standardizer.mean).all()
                                     or not np.isfinite(standardizer.scale).all()):
        reason = "nonfinite_fitting_standardizer"
    fitted, validation = {}, {}
    for family in families:
        tick = time.perf_counter()
        fallback = reason
        candidate = None
        attempted_accounting = {"optimizer_steps": 0, "training_row_exposures": 0}
        if fallback is None:
            try:
                if family == "mlp":
                    candidate = _fit_mlp(xf, yf, xv, yv, standardizer, n_classes, int(seed), cfg, common)
                else:
                    cls = LogisticRegression if family == "logistic" else HistGradientBoostingClassifier
                    model = cls(**cfg[family], random_state=int(seed))
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model.fit(standardizer.transform(xf), yf)
                    iterations = int(np.max(model.n_iter_))
                    meta = {**common, "family": family, "parameters": copy.deepcopy(cfg[family]),
                            "actual_iterations": iterations,
                            "fit_warnings": [str(w.message) for w in caught],
                            "selection": "one fixed configuration; validation does not alter fitting"}
                    if family == "logistic":
                        meta.update({"optimizer_steps": iterations, "training_row_exposures": None,
                                     "exposure_note": "LBFGS uses line searches; exact row presentations are not exposed by sklearn.",
                                     "parameter_count": int(model.coef_.size+model.intercept_.size)})
                        attempted_accounting = {"optimizer_steps": iterations, "training_row_exposures": None}
                        if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
                            raise FloatingPointError("Nonfinite logistic coefficients")
                    else:
                        meta.update({"optimizer_steps": 0, "training_row_exposures": None,
                                     "boosting_iterations": iterations,
                                     "tree_fit_row_participations": iterations*len(yf),
                                     "trees_per_iteration": int(model.n_trees_per_iteration_),
                                     "exposure_note": "Full fitting rows per boosting iteration; separate from optimizer minibatch exposure."})
                        attempted_accounting = {"optimizer_steps": 0, "training_row_exposures": None,
                                                "boosting_iterations": iterations,
                                                "tree_fit_row_participations": iterations*len(yf)}
                    candidate = FittedCandidate(family, model, standardizer, n_classes, meta)
                probabilities = candidate.predict_proba(xv)
                if not np.isfinite(probabilities).all():
                    raise FloatingPointError("Nonfinite candidate validation probabilities")
            except FloatingPointError as error:
                fallback = str(error)
                attempted_accounting = getattr(error, "fit_accounting", attempted_accounting)
        if fallback is not None:
            candidate = PriorPredictor(prior.probabilities, {
                **common, "family": family, "effective_family": "prior",
                "fallback_reason": fallback, "parameters": copy.deepcopy(cfg[family]),
                **attempted_accounting, "prior_fitting_rows": len(yf),
                "fitting_attempted": reason is None,
                "pseudocount_per_class": cfg["prior"]["pseudocount"],
                "selection": "explicit fitting-prior fallback; any attempted optimization is separately counted"})
            probabilities = candidate.predict_proba(xv)
        validation[family] = metrics(yv, probabilities, n_classes)
        candidate.metadata["validation_scores"] = validation[family]
        candidate.metadata["fit_runtime_seconds"] = time.perf_counter()-tick
        fitted[family] = candidate
    selected = min(fitted, key=lambda name: (validation[name]["log_loss"], name))
    metadata = {**common, "budget": cfg, "families": list(families),
                "validation_scores": validation, "selected_family": selected,
                "selection": "minimum validation log loss, then lexicographic family; no final-test input",
                "fit_runtime_seconds": time.perf_counter()-started,
                "candidates": {name: candidate.metadata for name, candidate in fitted.items()}}
    return {"candidates": fitted, "selected_family": selected, "metadata": metadata}
