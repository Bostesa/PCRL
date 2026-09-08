"""Source-only models for the ACS transfer experiment.

Fitting accepts exactly the five declared source targets, never downstream task
labels. Integer -1 means an unavailable source label. The caller owns row
splitting and fitting-only observation preprocessing. This module neither reads
data files nor fits feature preprocessing nor accesses a final-test split.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier


SOURCE_TARGETS = ("income_binary", "income_bins", "esr", "pubcov", "joint")
FIXED_CLASS_COUNTS = {"income_binary": 2, "esr": 6, "pubcov": 2, "joint": 8}
PROBABILITY_FLOOR = 1e-12


def array_digest(value) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(array_digest(value.detach().cpu().numpy()).encode())
    return h.hexdigest()


def _clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _schema(class_counts):
    if set(class_counts) != set(SOURCE_TARGETS):
        raise ValueError(f"Only the exact source target whitelist is accepted: {SOURCE_TARGETS}")
    result = {}
    for name in SOURCE_TARGETS:
        value = class_counts[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("Class counts must be positive integers")
        result[name] = int(value)
    if any(result[name] != size for name, size in FIXED_CLASS_COUNTS.items()):
        raise ValueError("Source categorical schemas must retain their declared fixed classes")
    if result["income_bins"] > 8:
        raise ValueError("Source income bins are capped at eight")
    return result


def _matrix(value, name, width=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    x = np.asarray(value, dtype=np.float32)
    if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite two-dimensional feature matrix")
    if width is not None and x.shape[1] != width:
        raise ValueError("Feature dimension differs from the fitted source schema")
    return np.ascontiguousarray(x)


def _labels(labels, n, schema, split):
    if set(labels) != set(SOURCE_TARGETS):
        raise ValueError(f"{split} labels must contain only the exact source target whitelist")
    result = {}
    for name, size in schema.items():
        value = labels[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        y = np.asarray(value)
        if y.ndim != 1 or len(y) != n or not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f"{split}/{name} requires one integer label per row; -1 marks missing")
        if ((y < -1) | (y >= size)).any():
            raise ValueError(f"{split}/{name} contains labels outside its fixed schema")
        if not (y >= 0).any():
            raise ValueError(f"{split}/{name} has no known labels; its source loss is undefined")
        result[name] = np.ascontiguousarray(y, dtype=np.int64)
    return result


def _inputs(Xtrain, labels, Xval, val_labels, class_counts):
    schema = _schema(class_counts)
    x, xv = _matrix(Xtrain, "Xtrain"), _matrix(Xval, "Xval")
    if not len(x) or not len(xv) or x.shape[1] != xv.shape[1]:
        raise ValueError("Nonempty fitting and validation splits with matching features required")
    y = _labels(labels, len(x), schema, "source_fit")
    yv = _labels(val_labels, len(xv), schema, "source_validation")
    return x, y, xv, yv, schema


def _coverage(labels, schema):
    return {name: {"known": int((labels[name] >= 0).sum()),
                   "missing": int((labels[name] == -1).sum()),
                   "class_support": np.bincount(labels[name][labels[name] >= 0], minlength=k).tolist()}
            for name, k in schema.items()}


def _provenance(x, y, xv, yv, schema):
    return {
        "source_target_order": list(SOURCE_TARGETS), "class_counts": schema,
        "source_fit_rows": len(x), "source_validation_rows": len(xv),
        "input_dim": x.shape[1], "preprocessing": "already supplied; no feature preprocessing fitted here",
        "coverage": {"source_fit": _coverage(y, schema), "source_validation": _coverage(yv, schema)},
        "accessed_array_hashes": {
            "source_fit_features": array_digest(x), "source_validation_features": array_digest(xv),
            "source_fit_labels": {name: array_digest(y[name]) for name in SOURCE_TARGETS},
            "source_validation_labels": {name: array_digest(yv[name]) for name in SOURCE_TARGETS}},
        "downstream_labels_received": False, "final_test_received": False,
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def source_scores(probabilities, labels, class_counts):
    """Five equally weighted source-head losses, with missing labels excluded.

    The log-loss floor is numerical scoring convention, not evidence that a
    class absent during fitting was learned. Coverage is always returned.
    """
    schema = _schema(class_counts)
    if set(probabilities) != set(schema) or set(labels) != set(schema):
        raise ValueError("Scoring requires the fixed source targets only")
    result = {}
    for name, k in schema.items():
        y = np.asarray(labels[name])
        p = np.asarray(probabilities[name], dtype=np.float64)
        if p.shape != (len(y), k) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("Invalid probabilities or probability schema")
        if not np.allclose(p.sum(1), 1, atol=1e-6):
            raise ValueError("Fixed-schema probabilities must sum to one")
        known = y >= 0
        if not known.any():
            raise ValueError("Source validation loss is undefined without known labels")
        selected = p[known]
        truth = y[known].astype(int)
        clipped = np.clip(selected, PROBABILITY_FLOOR, 1)
        clipped /= clipped.sum(1, keepdims=True)
        result[name] = {
            "log_loss": float(-np.log(clipped[np.arange(len(truth)), truth]).mean()),
            "accuracy": float((selected.argmax(1) == truth).mean()),
            "known": len(truth), "missing": int((~known).sum()),
            "class_support": np.bincount(truth, minlength=k).tolist(),
        }
    return {"mean_source_loss": float(np.mean([v["log_loss"] for v in result.values()])),
            "heads": result, "log_loss_probability_floor": PROBABILITY_FLOOR}


class SourceEncoder(nn.Module):
    """Input→64 ReLU→64 ReLU→32 linear representation and five source heads."""

    def __init__(self, input_dim, class_counts):
        super().__init__()
        self.class_counts = _schema(class_counts)
        self.input_dim = int(input_dim)
        self.repr_dim = 32
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
        self.heads = nn.ModuleDict({name: nn.Linear(32, k) for name, k in self.class_counts.items()})
        self.metadata = {}

    def forward(self, x):
        representation = self.encoder(x)
        return {name: head(representation) for name, head in self.heads.items()}

    @torch.no_grad()
    def release(self, x, batch_size=4096):
        x = _matrix(x, "release features", self.input_dim)
        if batch_size < 1:
            raise ValueError("Positive inference batch size required")
        result = np.empty((len(x), self.repr_dim), dtype=np.float32)
        for start in range(0, len(x), batch_size):
            result[start:start+batch_size] = self.encoder(torch.from_numpy(x[start:start+batch_size])).numpy()
        return result

    @torch.no_grad()
    def probabilities(self, x, batch_size=4096):
        x = _matrix(x, "probability features", self.input_dim)
        if batch_size < 1:
            raise ValueError("Positive inference batch size required")
        result = {name: np.empty((len(x), k), dtype=np.float32) for name, k in self.class_counts.items()}
        for start in range(0, len(x), batch_size):
            logits = self(torch.from_numpy(x[start:start+batch_size]))
            for name, value in logits.items():
                result[name][start:start+batch_size] = value.softmax(1).numpy()
        return result

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self

    def save(self, path, metadata=None):
        torch.save({"format_version": 1, "input_dim": self.input_dim,
                    "class_counts": self.class_counts, "state_dict": _clone_state(self),
                    "metadata": self.metadata if metadata is None else metadata}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, weights_only=True, map_location="cpu")
        with torch.random.fork_rng(devices=[]):
            model = cls(saved["input_dim"], saved["class_counts"])
        model.load_state_dict(saved["state_dict"])
        model.metadata = saved["metadata"]
        return model.freeze()


def masked_source_loss(logits, labels):
    """Average across all five source heads, zero for a wholly missing minibatch head."""
    if set(logits) != set(SOURCE_TARGETS) or set(labels) != set(SOURCE_TARGETS):
        raise ValueError("Training loss accepts source targets only")
    losses = []
    any_known = False
    for name in SOURCE_TARGETS:
        known = labels[name] >= 0
        if known.any():
            any_known = True
            losses.append(nn.functional.cross_entropy(logits[name][known], labels[name][known]))
        else:
            losses.append(logits[name].sum()*0)
    return torch.stack(losses).mean(), any_known


def fit_source_encoder(Xtrain, labels, Xval, val_labels, class_counts, *, seed, lr,
                       epochs=60, batch_size=256, out_dir=None):
    """Fit one declared LR configuration; select its checkpoint by source validation only."""
    started = time.perf_counter()
    x, y, xv, yv, schema = _inputs(Xtrain, labels, Xval, val_labels, class_counts)
    if epochs < 0 or batch_size < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Nonnegative epochs, positive batch size and finite positive LR required")
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    initialization_seed, schedule_seed = 2100000+int(seed), 2200000+int(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        model = SourceEncoder(x.shape[1], schema)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    generator = torch.Generator().manual_seed(schedule_seed)
    xt = torch.from_numpy(x)
    yt = {name: torch.from_numpy(value) for name, value in y.items()}
    initial_state = _clone_state(model)
    initial_validation = source_scores(model.probabilities(xv), yv, schema)
    best_state, best_validation = initial_state, initial_validation
    best_epoch, best_steps, steps, skipped_batches = 0, 0, 0, 0
    schedule_digest = hashlib.sha256()
    history = [{"epoch": 0, "optimizer_steps": 0, "validation": initial_validation}]
    if out is not None:
        model.save(out/"initialization.pt", {"checkpoint_stage": "initialization", "epoch": 0,
                                            "optimizer_steps": 0, "state_hash": state_digest(initial_state)})
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        epoch_losses = []
        for index in order.split(batch_size):
            schedule_digest.update(len(index).to_bytes(8, "little"))
            schedule_digest.update(index.numpy().tobytes())
            loss, observed = masked_source_loss(model(xt[index]), {name: value[index] for name, value in yt.items()})
            if not observed:
                skipped_batches += 1
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite source training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            epoch_losses.append(float(loss.detach()))
        model.eval()
        validation = source_scores(model.probabilities(xv), yv, schema)
        history.append({"epoch": epoch, "optimizer_steps": steps,
                        "mean_training_batch_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                        "validation": validation})
        if validation["mean_source_loss"] < best_validation["mean_source_loss"]:
            best_epoch, best_steps, best_validation = epoch, steps, validation
            best_state = _clone_state(model)
    final_state = _clone_state(model)
    final_validation = source_scores(model.probabilities(xv), yv, schema)
    metadata = {
        **_provenance(x, y, xv, yv, schema), "kind": "source_neural_encoder",
        "config": {"lr": float(lr), "epochs": int(epochs), "batch_size": int(batch_size),
                   "optimizer": "Adam", "weight_decay": 0., "architecture": [x.shape[1], 64, 64, 32],
                   "activation": "ReLU; final representation linear", "batchnorm": False, "dropout": 0.},
        "seed": int(seed), "initialization_seed": initialization_seed, "schedule_seed": schedule_seed,
        "schedule_hash": schedule_digest.hexdigest(), "optimizer_steps": steps,
        "planned_batches": epochs*math.ceil(len(x)/batch_size), "skipped_all_missing_batches": skipped_batches,
        "selected_epoch": best_epoch, "selected_optimizer_steps": best_steps,
        "validation_loss": best_validation["mean_source_loss"], "initial_validation": initial_validation,
        "selected_validation": best_validation, "final_validation": final_validation,
        "validation_history": history, "initial_state_hash": state_digest(initial_state),
        "selected_state_hash": state_digest(best_state), "final_state_hash": state_digest(final_state),
        "selection": "minimum mean source-validation head log loss; true epoch0 and each epoch; strict improvement, earliest tie",
        "training_loss": "mean of five masked-head cross-entropies; empty minibatch head contributes zero",
        "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
        "release_dim": 32, "probability_bank_dim": sum(schema.values()),
        "dtype": "float32", "device": "cpu", "runtime_seconds": time.perf_counter()-started,
    }
    if out is not None:
        model.save(out/"final.pt", {**metadata, "checkpoint_stage": "final", "checkpoint_optimizer_steps": steps})
    model.load_state_dict(best_state)
    model.metadata = metadata
    model.freeze()
    if out is not None:
        model.save(out/"selected.pt", {**metadata, "checkpoint_stage": "selected", "checkpoint_optimizer_steps": best_steps})
        (out/"metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    return model, metadata


class SourceTreeBank:
    """Independent tree classifiers with probability columns fixed to source schemas."""

    def __init__(self, input_dim, class_counts, models, constants):
        self.input_dim = int(input_dim)
        self.class_counts = _schema(class_counts)
        self.models = models
        self.constants = constants
        self.metadata = {}

    def probabilities(self, x):
        x = _matrix(x, "tree probability features", self.input_dim)
        result = {}
        for name, k in self.class_counts.items():
            p = np.zeros((len(x), k), dtype=np.float32)
            if name in self.constants:
                p[:, self.constants[name]] = 1
            elif len(x):
                model = self.models[name]
                p[:, model.classes_.astype(int)] = model.predict_proba(x).astype(np.float32)
            result[name] = p
        return result

    def save(self, path):
        joblib.dump(self, path)


def fit_source_tree_bank(Xtrain, labels, Xval, val_labels, class_counts, *, seed,
                         max_leaf_nodes, max_iter=150, out_dir=None):
    """Fit one fixed tree-capacity configuration; return source validation loss to caller.

    Early stopping is disabled, so no hidden random internal validation split
    consumes source-fitting rows and every nonconstant head executes max_iter.
    """
    started = time.perf_counter()
    x, y, xv, yv, schema = _inputs(Xtrain, labels, Xval, val_labels, class_counts)
    if max_iter < 1 or max_leaf_nodes < 2:
        raise ValueError("Positive tree iteration budget and at least two leaves required")
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    models, constants, training = {}, {}, {}
    for offset, name in enumerate(SOURCE_TARGETS):
        known = y[name] >= 0
        classes = np.unique(y[name][known])
        if len(classes) == 1:
            constants[name] = int(classes[0])
            iterations, predictors = 0, 0
        else:
            model = HistGradientBoostingClassifier(
                max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, learning_rate=.1,
                l2_regularization=1., min_samples_leaf=20, early_stopping=False,
                random_state=2300000+int(seed)*10+offset)
            model.fit(x[known], y[name][known])
            models[name] = model
            iterations = int(model.n_iter_)
            predictors = int(model.n_trees_per_iteration_)*iterations
        training[name] = {"fit_rows": int(known.sum()), "fitted_classes": classes.tolist(),
                          "absent_schema_classes": sorted(set(range(schema[name]))-set(classes.tolist())),
                          "boosting_iterations": iterations, "fitted_trees": predictors,
                          "constant_predictor": name in constants}
    bank = SourceTreeBank(x.shape[1], schema, models, constants)
    validation = source_scores(bank.probabilities(xv), yv, schema)
    metadata = {
        **_provenance(x, y, xv, yv, schema), "kind": "independent_source_tree_bank",
        "seed": int(seed), "training": training,
        "config": {"max_iter": int(max_iter), "max_leaf_nodes": int(max_leaf_nodes),
                   "learning_rate": .1, "l2_regularization": 1., "min_samples_leaf": 20,
                   "early_stopping": False, "internal_validation_split": None},
        "validation_loss": validation["mean_source_loss"], "selected_validation": validation,
        "selection": "none within this capacity configuration; caller compares source-validation loss only",
        "probability_bank_dim": sum(schema.values()),
        "model_content_hash": joblib.hash((models, constants, schema), hash_name="sha1"),
        "dtype": "float32 published probabilities", "runtime_seconds": time.perf_counter()-started,
    }
    bank.metadata = metadata
    if out is not None:
        bank.save(out/"source_tree_bank.joblib")
        (out/"metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    return bank, metadata
