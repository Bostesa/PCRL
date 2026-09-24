"""Dedicated-role, frozen local-input sensitive-risk predictors for Branch B.

The predictors are supervision for *internal split decisions*, never released
scores. They fit only global nuisance_train households and consume X_A,H_A at
deployment. An absent protected class remains present at a declared floor.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from .roles import CLASS_COUNT, role_of

PROBABILITY_FLOOR = 1e-8


@dataclass
class FrozenNuisance:
    mean: np.ndarray
    scale: np.ndarray
    models: dict[str, Any]
    classes: dict[str, tuple[int, ...]]
    training_sha256: str

    def probabilities(self, inputs: RuntimeInputs) -> dict[str, np.ndarray]:
        if not isinstance(inputs, RuntimeInputs):
            raise TypeError("frozen nuisance accepts RuntimeInputs(X_A,H_A) only")
        x = np.asarray(inputs.features(), dtype=np.float64)
        if self.mean.shape != (36,) or self.scale.shape != (36,) or np.any(self.scale <= 0):
            raise ValueError("invalid frozen nuisance standardizer")
        x = (x - self.mean)/self.scale
        out = {}
        for target in ("SEX", "RAC1P"):
            k = CLASS_COUNT[target]
            model = self.models[target]
            p = np.zeros((len(x), k), dtype=np.float64)
            if isinstance(model, int):
                p[:, model] = 1.
            else:
                model_classes = np.asarray(model.classes_, dtype=int)
                if tuple(model_classes) != self.classes[target]:
                    raise ValueError("frozen nuisance class order changed")
                p[:, model_classes] = model.predict_proba(x)
            # This is explicit smoothing, not evidence of learning an absent
            # class. Every class remains in the nine-category audit schema.
            p = (1 - k*PROBABILITY_FLOOR)*p + PROBABILITY_FLOOR
            if not np.isfinite(p).all() or np.any(p <= 0) or not np.allclose(p.sum(1), 1, atol=1e-12):
                raise ValueError("invalid frozen nuisance probabilities")
            out[target] = p
        return out


def _training_hash(rows: dict) -> str:
    digest = hashlib.sha256()
    for household in rows["households"]:
        encoded = str(household).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    for value in (rows["x"], rows["ha"], rows["weights"],
                  rows["labels"]["SEX"], rows["labels"]["RAC1P"]):
        a = np.ascontiguousarray(value)
        digest.update(str(a.shape).encode())
        digest.update(str(a.dtype).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def fit_frozen_nuisance(rows: dict, *, seed: int) -> tuple[FrozenNuisance, dict]:
    """Fit fixed logistic risk models on nuisance_train; no validation tuning."""
    required = {"x", "ha", "households", "weights", "labels"}
    if not required.issubset(rows) or not {"SEX", "RAC1P"}.issubset(rows["labels"]):
        raise ValueError("nuisance_train rows require local features and protected labels")
    x = np.asarray(rows["x"], dtype=np.float64)
    ha = np.asarray(rows["ha"], dtype=np.float64)
    households = np.asarray(rows["households"])
    weights = np.asarray(rows["weights"], dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != 32 or ha.shape != (len(x), 4) or
            households.shape != (len(x),) or weights.shape != (len(x),) or
            len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(ha).all() or
            not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("invalid nuisance_train local inputs or original-person weights")
    if any(role_of(h) != "nuisance_train" for h in households):
        raise ValueError("nuisance predictor may fit nuisance_train households only")
    z = np.column_stack((x, ha))
    mean = z.mean(0)
    scale = z.std(0)
    scale[scale < 1e-12] = 1.
    standardized = (z - mean)/scale
    models = {}
    classes = {}
    support = {}
    excluded = {}
    for target in ("SEX", "RAC1P"):
        k = CLASS_COUNT[target]
        y = np.asarray(rows["labels"][target])
        if y.shape != (len(x),) or y.dtype.kind not in "iu" or np.any((y < -1) | (y >= k)):
            raise ValueError(f"{target} labels violate full {target} schema")
        valid = y >= 0
        if not valid.any() or weights[valid].sum() <= 0:
            raise ValueError(f"no positive-weight {target} nuisance support")
        counts = np.bincount(y[valid], minlength=k)
        observed = tuple(int(v) for v in np.flatnonzero(counts))
        support[target] = counts.tolist()
        excluded[target] = int((~valid).sum())
        classes[target] = observed
        if len(observed) == 1:
            models[target] = observed[0]
        else:
            fit_weight = weights[valid] * (valid.sum()/weights[valid].sum())
            model = LogisticRegression(C=1., max_iter=500, solver="lbfgs",
                                       random_state=int(seed))
            model.fit(standardized[valid], y[valid], sample_weight=fit_weight)
            if tuple(int(v) for v in model.classes_) != observed:
                raise ValueError("fitted nuisance class order differs from support")
            models[target] = model
    training_sha = _training_hash(rows)
    fitted = FrozenNuisance(mean, scale, models, classes, training_sha)
    receipt = {"schema": "pcrl-frozen-nuisance-v1", "training_role": "nuisance_train",
               "training_sha256": training_sha, "people": int(len(x)),
               "households": int(len(np.unique(households))),
               "class_support": support, "missing_label_excluded": excluded,
               "fitted_classes": {key: list(value) for key, value in classes.items()},
               "full_schema": {key: CLASS_COUNT[key] for key in ("SEX", "RAC1P")},
               "probability_floor": PROBABILITY_FLOOR,
               "predictor": "standardized X_A32+H_A4 logistic C=1 lbfgs max_iter=500; fixed recipe, no outcome tuning",
               "deployment_observes": ["X_A", "H_A"],
               "predicted_risks_private": True,
               "unsupported_class_is_not_learned": True}
    return fitted, receipt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_nuisance(directory: str | Path) -> tuple[FrozenNuisance, dict]:
    root = Path(directory)
    receipt = json.loads((root / "NUISANCE.json").read_text())
    path = root / "nuisance.joblib"
    if receipt.get("schema") != "pcrl-frozen-nuisance-v1" or _sha256(path) != receipt.get("model_sha256"):
        raise ValueError("frozen nuisance model/receipt hash mismatch")
    model = joblib.load(path)
    if not isinstance(model, FrozenNuisance) or model.training_sha256 != receipt["training_sha256"]:
        raise ValueError("frozen nuisance source hash differs")
    return model, receipt


def save_frozen_nuisance(directory: str | Path, model: FrozenNuisance,
                         receipt: dict) -> dict:
    """Write once privately; a retry verifies and reuses, never refits here."""
    root = Path(directory)
    if "private" not in root.parts:
        raise ValueError("fitted nuisance must stay under a private output path")
    if root.exists() and any(root.iterdir()):
        restored, existing = load_frozen_nuisance(root)
        if (existing.get("training_sha256") != receipt.get("training_sha256") or
                restored.training_sha256 != model.training_sha256):
            raise ValueError("existing frozen nuisance has different training rows")
        return existing
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    temporary = root / f"nuisance.joblib.tmp.{os.getpid()}"
    joblib.dump(model, temporary, compress=3)
    os.chmod(temporary, 0o600)
    os.replace(temporary, root / "nuisance.joblib")
    stored = {**receipt, "model_sha256": _sha256(root / "nuisance.joblib")}
    text = json.dumps(stored, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary_receipt = root / f"NUISANCE.json.tmp.{os.getpid()}"
    temporary_receipt.write_text(text)
    os.chmod(temporary_receipt, 0o600)
    os.replace(temporary_receipt, root / "NUISANCE.json")
    load_frozen_nuisance(root)
    return stored
