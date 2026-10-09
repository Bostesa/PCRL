"""Task-only source architecture on admitted role-only matrices.

This module imports only the historical training kernels. It never invokes
historical data loaders and has no protected-label or assessment input.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jcv import train as jcv_train
from smf import train as smf_train


C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
TASKS = ("income", "occupation_group")
CLASSES = (2, 6)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class TaskOnlyData:
    """The fixed source task kernels' tensor contract, without SEX access.

    `smf.train.train_run` constructs a privacy prior even for its `base=none`
    task-only branch. Its dummy fields are constants and never affect task
    gradients; this is checked against a changed dummy in the synthetic test.
    """

    def __init__(self, x83: np.ndarray, income_y: np.ndarray,
                 occupation_y: np.ndarray, *, dummy_bit: int = 0):
        x = np.asarray(x83)
        labels = (np.asarray(income_y), np.asarray(occupation_y))
        if (x.ndim != 2 or x.shape[1] != 83 or x.dtype != np.float32
                or not np.isfinite(x).all() or len(x) == 0):
            raise ValueError("source task learner needs finite float32 (n,83) input")
        for y, classes in zip(labels, CLASSES):
            if (y.dtype != np.int64 or y.shape != (len(x),)
                    or np.any(y < 0) or np.any(y >= classes)
                    or len(np.unique(y)) != classes):
                raise ValueError("source task learner needs complete aligned classes")
        if dummy_bit not in (0, 1):
            raise ValueError("dummy bit must be 0 or 1")
        self.X = torch.from_numpy(np.ascontiguousarray(x))
        self.Y = {i: torch.from_numpy(np.ascontiguousarray(y))
                  for i, y in enumerate(labels)}
        self.n = len(x)
        # No protected label is accepted or read by this contract.
        self.S = torch.full((self.n,), dummy_bit, dtype=torch.int64)
        self.prior = np.asarray([0.5, 0.5], dtype=np.float64)
        self.cf = self.cv = self.cal = self.ref = np.empty(0, dtype=np.int64)


def train_source_encoder(x83: np.ndarray, income_y: np.ndarray,
                         occupation_y: np.ndarray, seed: int,
                         *, dummy_bit: int = 0):
    """Fresh 20-epoch Adam warm plus 40-epoch task-only source continuation.

    Initialization and minibatch streams are pinned by the original source
    kernels. No saved U0 state or deployed head is loaded here.
    """
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("only the three predeclared paired seeds are allowed")
    torch.set_num_threads(1)
    data = TaskOnlyData(x83, income_y, occupation_y, dummy_bit=dummy_bit)
    warm = jcv_train.warm_start(83, [2, 6], data, seed)
    task_spec = {"base": "none", "schedule": "ONLINE", "update": "task"}
    final, diagnostic, checkpoints, _, _ = smf_train.train_run(
        task_spec, 0.0, warm.state_dict(), data, seed, "A", None,
        n_epochs=40, ckpt_epochs=(40,))
    if diagnostic["encoder_updates"] != 40 * int(np.ceil(data.n / 256)):
        raise RuntimeError("source task-only continuation update count differs")
    if 40 not in checkpoints:
        raise RuntimeError("source task-only epoch-40 checkpoint is absent")
    if not all(torch.equal(final.state_dict()[name], checkpoints[40]["model"][name])
               for name in final.state_dict()):
        raise RuntimeError("source epoch-40 checkpoint differs from returned model")
    final.eval()
    return final


def encode_source(model, x83: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x83)
    if (x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 83
            or not np.isfinite(x).all()):
        raise ValueError("source encoder requires finite float32 (n,83) input")
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(x))
        return tuple(np.asarray(model.encode(i, tensor).double().numpy(),
                                dtype=np.float64) for i in (0, 1))


def _fit_one_head(z: np.ndarray, y: np.ndarray, c: float):
    if c not in C_GRID:
        raise ValueError("source deployed-head C lies outside the frozen grid")
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=c, max_iter=3000)).fit(z, y)


def choose_source_C(table: list[dict]) -> dict:
    """Source head rule: improve NLL by more than 1e-12, else retain lower C."""
    if [row.get("C") for row in table] != list(C_GRID):
        raise ValueError("source head grid order differs")
    best = None
    for row in table:
        loss = float(row["nll"])
        if not np.isfinite(loss):
            raise ValueError("source head validation NLL is nonfinite")
        if best is None or loss < best["nll"] - 1e-12:
            best = row
    return best


def select_source_head_C(model, fit_x83: np.ndarray, fit_income_y: np.ndarray,
                         fit_occupation_y: np.ndarray, val_x83: np.ndarray,
                         val_income_y: np.ndarray,
                         val_occupation_y: np.ndarray) -> dict:
    """Choose source-style C using only the frozen FIT internal fold."""
    fit_z = encode_source(model, fit_x83)
    val_z = encode_source(model, val_x83)
    fit_y = (np.asarray(fit_income_y), np.asarray(fit_occupation_y))
    val_y = (np.asarray(val_income_y), np.asarray(val_occupation_y))
    out = {}
    for i, task in enumerate(TASKS):
        table = []
        for c in C_GRID:
            head = _fit_one_head(fit_z[i], fit_y[i], c)
            loss = float(log_loss(val_y[i], head.predict_proba(val_z[i]),
                                  labels=list(range(CLASSES[i]))))
            table.append({"C": c, "nll": loss})
        best = choose_source_C(table)
        out[task] = {"selected_C": best["C"], "validation_nll": best["nll"],
                     "grid": table}
    return out


def fit_source_heads(model, x83: np.ndarray, income_y: np.ndarray,
                     occupation_y: np.ndarray,
                     c_by_task: dict[str, float]):
    if set(c_by_task) != set(TASKS):
        raise ValueError("both frozen task C values are required")
    z = encode_source(model, x83)
    return tuple(_fit_one_head(z[i], np.asarray(y), c_by_task[task])
                 for i, (task, y) in enumerate(zip(TASKS, (income_y, occupation_y))))


def predict_source(model, heads, x83: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(heads) != 2:
        raise ValueError("source deployment needs both task heads")
    z = encode_source(model, x83)
    result = tuple(np.asarray(heads[i].predict_proba(z[i]), dtype=np.float64)
                   for i in (0, 1))
    for probability, classes in zip(result, CLASSES):
        if (probability.shape != (len(x83), classes)
                or not np.isfinite(probability).all()
                or np.any(probability < 0) or np.any(probability > 1)
                or not np.allclose(probability.sum(1), 1, atol=1e-12, rtol=0)):
            raise ValueError("source deployment produced invalid task probabilities")
    return result


def save_source(model, heads, directory: str | Path) -> dict[str, str]:
    """Write private model/head states; caller owns write-once receipt and path gate."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), root / "encoder.pt")
    for i, head in enumerate(heads):
        joblib.dump(head, root / f"head_{i}.joblib")
    return {name: file_sha256(root / name)
            for name in ("encoder.pt", "head_0.joblib", "head_1.joblib")}


def restore_source(directory: str | Path, seed: int,
                   expected_sha256: dict[str, str]):
    root = Path(directory)
    names = ("encoder.pt", "head_0.joblib", "head_1.joblib")
    if set(expected_sha256) != set(names):
        raise ValueError("source restore requires all private file hashes")
    for name in names:
        if file_sha256(root / name) != expected_sha256[name]:
            raise PermissionError("source checkpoint differs from write-once receipt")
    model = jcv_train.Model(83, [2, 6], seed)
    model.load_state_dict(torch.load(root / "encoder.pt", weights_only=True))
    model.eval()
    heads = tuple(joblib.load(root / f"head_{i}.joblib") for i in (0, 1))
    return model, heads
