"""Deterministic Team C purpose heads on admitted selected feature values.

Only in-memory OSF_DEFENSE_FIT arrays enter fitting. The caller obtains them
through S2's SafeRoleStore and fits H-GLOBAL-TEMP separately on the admitted
CALIBRATION_HELDOUT role. Checkpoints contain weights and aggregate receipts,
never rows or labels, and must be stored under a caller-declared private root.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prototype import AdultGroupedSchema
from scorecard_builder import group_disjoint_split, preserve_fit_class_support


@dataclass(frozen=True)
class HeadSpec:
    hidden_width: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    batch_size: int = 512
    max_epochs: int = 30
    validation_fraction: float = 0.15
    validation_seed_offset: int = 17_001
    shuffle_seed_offset: int = 20_000
    refit_shuffle_seed_offset: int = 30_000
    patience: int = 5
    min_delta_nll: float = 0.0001
    numerical_threads: int = 1

    def __post_init__(self) -> None:
        if (self.hidden_width < 1 or self.learning_rate <= 0
                or self.weight_decay < 0 or self.batch_size < 1
                or self.max_epochs < 1 or self.patience < 1
                or self.min_delta_nll < 0 or self.numerical_threads != 1
                or not 0 < self.validation_fraction < 1):
            raise ValueError("invalid fixed Team C HeadSpec")


def _torch():
    import torch

    return torch


def _network(input_dim: int, output_dim: int, width: int):
    torch = _torch()
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, width), torch.nn.ReLU(),
        torch.nn.Linear(width, width), torch.nn.ReLU(),
        torch.nn.Linear(width, output_dim),
    )


@dataclass
class FittedTaskHead:
    purpose: str
    model_id: str
    selected_atoms: tuple[str, ...]
    selected_columns: int
    spec: HeadSpec
    receipt: dict[str, Any]
    model: Any

    def predict_proba(self, selected: np.ndarray) -> np.ndarray:
        torch = _torch()
        torch.set_num_threads(self.spec.numerical_threads)
        selected = np.asarray(selected)
        if (selected.ndim != 2 or selected.shape[1] != self.selected_columns
                or not np.isfinite(selected).all()):
            raise ValueError("head needs exactly its finite selected raw columns")
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.ascontiguousarray(selected, dtype=np.float32))
            # The trained network is float32, but S2's fixed calibrator
            # requires every probability row to sum within 1e-8. Casting an
            # already rounded float32 softmax to float64 cannot repair that
            # drift. Softmax the fixed logits in float64 instead.
            logits = self.model(tensor).to(dtype=torch.float64)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        if (not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or np.max(np.abs(probabilities.sum(axis=1) - 1.0), initial=0.0) > 1e-12):
            raise ValueError("head returned probabilities outside the strict simplex")
        return probabilities


def _train_one_epoch(model, optimizer, X, y, train_rows, *, batch_size, shuffle_seed):
    torch = _torch()
    model.train()
    order = np.random.default_rng(shuffle_seed).permutation(train_rows)
    for start in range(0, len(order), batch_size):
        batch_rows = order[start:start + batch_size]
        xx = torch.as_tensor(np.ascontiguousarray(X[batch_rows], dtype=np.float32))
        yy = torch.as_tensor(np.ascontiguousarray(y[batch_rows], dtype=np.int64))
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(xx), yy)
        loss.backward()
        optimizer.step()


def _validation_nll(model, X, y, rows):
    torch = _torch()
    model.eval()
    with torch.no_grad():
        xx = torch.as_tensor(np.ascontiguousarray(X[rows], dtype=np.float32))
        yy = torch.as_tensor(np.ascontiguousarray(y[rows], dtype=np.int64))
        return float(torch.nn.functional.cross_entropy(model(xx), yy).item())


def fit_task_head(
    X: np.ndarray,
    y: np.ndarray,
    unit: np.ndarray,
    feature_names: tuple[str, ...],
    selected_atoms: tuple[str, ...],
    *,
    purpose: str,
    model_id: str,
    seed: int,
    spec: HeadSpec,
) -> FittedTaskHead:
    """Select epochs internally, then refit from scratch on all allowed rows."""
    schema = AdultGroupedSchema()
    schema.validate_feature_names(feature_names)
    if purpose not in ("income", "occupation_group"):
        raise ValueError("purpose must be income or occupation_group")
    if not model_id:
        raise ValueError("model_id is required")
    selected_atoms = schema.normalize_atoms(selected_atoms)
    indices = schema.column_indices(selected_atoms)
    if not indices:
        raise ValueError("task head needs at least one selected atom")
    X = np.asarray(X)
    y = np.asarray(y)
    unit = np.asarray(unit)
    classes = 2 if purpose == "income" else 6
    if X.ndim != 2 or X.shape[1] != 83 or not np.isfinite(X).all():
        raise ValueError("X must be a finite 83-column matrix")
    if (y.shape != (len(X),) or y.dtype.kind not in "iu"
            or np.any(y < 0) or np.any(y >= classes)):
        raise ValueError("task labels must be an aligned integer class vector")
    if unit.shape != (len(X),):
        raise ValueError("group IDs must align with X")
    X_selected = np.ascontiguousarray(X[:, indices], dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.int64)
    fit_rows, validation_rows = group_disjoint_split(
        unit, seed=seed + spec.validation_seed_offset,
        selection_fraction=spec.validation_fraction,
    )
    fit_rows, validation_rows, moved_groups = preserve_fit_class_support(
        unit, fit_rows, validation_rows, {purpose: y}, {purpose: classes}
    )
    if not np.array_equal(np.unique(y[fit_rows]), np.arange(classes)):
        raise ValueError(f"internal fit is missing a {purpose} class")

    torch = _torch()
    torch.set_num_threads(spec.numerical_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    early_model = _network(len(indices), classes, spec.hidden_width)
    early_optimizer = torch.optim.Adam(
        early_model.parameters(), lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    best_epoch = 0
    best_nll = float("inf")
    bad_epochs = 0
    epochs_run = 0
    for epoch in range(1, spec.max_epochs + 1):
        _train_one_epoch(
            early_model, early_optimizer, X_selected, y, fit_rows,
            batch_size=spec.batch_size,
            shuffle_seed=seed + spec.shuffle_seed_offset + epoch,
        )
        nll = _validation_nll(early_model, X_selected, y, validation_rows)
        epochs_run = epoch
        if nll < best_nll - spec.min_delta_nll:
            best_nll = nll
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= spec.patience:
                break
    if best_epoch < 1:
        raise RuntimeError("early stopping did not select an epoch")

    torch.manual_seed(seed)
    final_model = _network(len(indices), classes, spec.hidden_width)
    final_optimizer = torch.optim.Adam(
        final_model.parameters(), lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    all_rows = np.arange(len(X), dtype=np.int64)
    for epoch in range(1, best_epoch + 1):
        _train_one_epoch(
            final_model, final_optimizer, X_selected, y, all_rows,
            batch_size=spec.batch_size,
            shuffle_seed=seed + spec.refit_shuffle_seed_offset + epoch,
        )
    final_model.eval()
    fit_groups = np.unique(unit[fit_rows]).astype("<i8", copy=False)
    validation_groups = np.unique(unit[validation_rows]).astype("<i8", copy=False)
    receipt = {
        "model_id": model_id,
        "purpose": purpose,
        "seed": int(seed),
        "selected_atoms": selected_atoms,
        "selected_columns": len(indices),
        "feature_schema_sha256": hashlib.sha256("\n".join(schema.feature_names).encode()).hexdigest(),
        "head_subfits": 2,
        "early_stop_fit_rows": len(fit_rows),
        "early_stop_validation_rows": len(validation_rows),
        "early_stop_fit_groups": len(fit_groups),
        "early_stop_validation_groups": len(validation_groups),
        "early_stop_group_overlap": len(np.intersect1d(fit_groups, validation_groups)),
        "early_stop_fit_group_sha256": hashlib.sha256(fit_groups.tobytes()).hexdigest(),
        "early_stop_validation_group_sha256": hashlib.sha256(validation_groups.tobytes()).hexdigest(),
        "class_safe_moved_groups": len(moved_groups),
        "class_safe_moved_group_sha256": hashlib.sha256(
            np.asarray(moved_groups, dtype="<i8").tobytes()
        ).hexdigest(),
        "early_stop_epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_validation_nll": best_nll,
        "full_refit_rows": len(X),
        "full_refit_epochs": best_epoch,
        "class_support": np.bincount(y, minlength=classes).tolist(),
        "spec": asdict(spec),
    }
    return FittedTaskHead(
        purpose, model_id, selected_atoms, len(indices), spec, receipt, final_model
    )


def _inside_private(path: Path, private_root: Path) -> bool:
    return path.resolve().is_relative_to(private_root.resolve())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_head(head: FittedTaskHead, path: Path, *, private_root: Path) -> str:
    """Atomically publish a write-once private checkpoint and return its hash."""
    path, private_root = Path(path), Path(private_root)
    if not _inside_private(path, private_root):
        raise ValueError("head checkpoint must stay under its private root")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "team-c-head-checkpoint-v1",
        "purpose": head.purpose,
        "model_id": head.model_id,
        "selected_atoms": head.selected_atoms,
        "selected_columns": head.selected_columns,
        "spec": asdict(head.spec),
        "receipt": head.receipt,
        "state_keys": list(head.model.state_dict()),
    }
    arrays = {"meta": np.array(json.dumps(metadata, sort_keys=True))}
    for i, tensor in enumerate(head.model.state_dict().values()):
        arrays[f"state_{i}"] = tensor.detach().cpu().numpy()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path)  # Fails if a previous unit owns this path.
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return _file_sha256(path)


def restore_head(
    path: Path, *, expected_sha256: str, private_root: Path
) -> FittedTaskHead:
    """Verify and restore a private weights-only checkpoint."""
    path, private_root = Path(path), Path(private_root)
    if not _inside_private(path, private_root):
        raise ValueError("head checkpoint must stay under its private root")
    if _file_sha256(path) != expected_sha256:
        raise ValueError("head checkpoint SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(stored["meta"].item())
        if metadata["schema"] != "team-c-head-checkpoint-v1":
            raise ValueError("unknown head checkpoint schema")
        spec = HeadSpec(**metadata["spec"])
        purpose = metadata["purpose"]
        classes = 2 if purpose == "income" else 6 if purpose == "occupation_group" else None
        if classes is None:
            raise ValueError("unknown checkpoint purpose")
        schema = AdultGroupedSchema()
        atoms = schema.normalize_atoms(tuple(metadata["selected_atoms"]))
        if len(schema.column_indices(atoms)) != metadata["selected_columns"]:
            raise ValueError("checkpoint selected-column count changed")
        expected_schema = hashlib.sha256("\n".join(schema.feature_names).encode()).hexdigest()
        if metadata["receipt"]["feature_schema_sha256"] != expected_schema:
            raise ValueError("checkpoint feature schema changed")
        model = _network(metadata["selected_columns"], classes, spec.hidden_width)
        state = {
            name: _torch().from_numpy(stored[f"state_{i}"].copy())
            for i, name in enumerate(metadata["state_keys"])
        }
        model.load_state_dict(state, strict=True)
        model.eval()
    return FittedTaskHead(
        purpose, metadata["model_id"], atoms, metadata["selected_columns"],
        spec, metadata["receipt"], model,
    )
