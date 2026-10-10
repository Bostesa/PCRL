"""One fixed teacher-guided task-only hard-code recipe and full-role refit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.fit import FitData, make_batch_plan

from .model import FixedScaler, HardCodeModel


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    patience: int = 20
    batch_size: int = 256
    learning_rate: float = 0.001
    distill_weight: float = 0.5
    weight_decay: float = 0.0
    init_seed_offset: int = 370001
    fold_batch_seed_offset: int = 370003
    full_batch_seed_offset: int = 370007

    def __post_init__(self) -> None:
        if (self.epochs < 1 or self.patience < 1 or self.batch_size < 2
                or self.learning_rate <= 0 or not np.isfinite(self.learning_rate)
                or self.distill_weight < 0 or not np.isfinite(self.distill_weight)
                or self.weight_decay != 0
                or len({self.init_seed_offset, self.fold_batch_seed_offset,
                        self.full_batch_seed_offset}) != 3):
            raise ValueError("invalid frozen task-only config")


@dataclass
class FitResult:
    model: HardCodeModel
    scaler: FixedScaler
    selected_epoch: int
    fold_history: list[dict]
    full_history: list[dict]
    fold_updates: int
    full_updates: int
    seed: int
    bits: int
    role: str
    train_rows: int
    validation_rows: int


def task_loss(logits: tuple[torch.Tensor, torch.Tensor],
              y: tuple[torch.Tensor, torch.Tensor],
              teachers: tuple[torch.Tensor, torch.Tensor],
              distill_weight: float) -> torch.Tensor:
    terms = []
    for i in (0, 1):
        ce = F.cross_entropy(logits[i], y[i])
        kl = F.kl_div(F.log_softmax(logits[i], dim=1), teachers[i], reduction="batchmean")
        terms.append(ce + distill_weight * kl)
    return (terms[0] + terms[1]) / 2


def _check_fold(data: FitData, train_idx: np.ndarray,
                val_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train, val = np.asarray(train_idx), np.asarray(val_idx)
    for ix in (train, val):
        if (ix.ndim != 1 or len(ix) < 2 or not np.issubdtype(ix.dtype, np.integer)
                or np.any(ix < 0) or np.any(ix >= data.rows)
                or len(np.unique(ix)) != len(ix)):
            raise ValueError("invalid frozen internal FIT indices")
    if np.intersect1d(train, val).size or len(train) + len(val) != data.rows:
        raise ValueError("internal folds must partition FIT rows")
    return train.astype(np.int64), val.astype(np.int64)


def _new_model(seed: int, offset: int, dim: int, bits: int) -> HardCodeModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + offset)
        return HardCodeModel(dim, (2, 6), bits)


def _epoch(model: HardCodeModel, optimizer: torch.optim.Optimizer,
           x: tuple[torch.Tensor, torch.Tensor], y: tuple[torch.Tensor, torch.Tensor],
           teacher: tuple[torch.Tensor, torch.Tensor],
           batches: tuple[np.ndarray, ...], weight: float) -> float:
    model.train()
    weighted = count = 0
    for chunk in batches:
        ix = torch.from_numpy(chunk)
        optimizer.zero_grad(set_to_none=True)
        logits = tuple(model(i, x[i][ix], True)[1] for i in (0, 1))
        loss = task_loss(logits, (y[0][ix], y[1][ix]),
                         (teacher[0][ix], teacher[1][ix]), weight)
        loss.backward()
        optimizer.step()
        weighted += float(loss.detach()) * len(chunk)
        count += len(chunk)
    return weighted / count


def _validation_nll(model: HardCodeModel,
                    x: tuple[torch.Tensor, torch.Tensor],
                    y: tuple[torch.Tensor, torch.Tensor], val: np.ndarray) -> float:
    model.eval()
    ix = torch.from_numpy(val)
    with torch.no_grad():
        losses = [float(F.cross_entropy(model(i, x[i][ix], False)[1], y[i][ix]))
                  for i in (0, 1)]
    return sum(losses) / 2


def fit_task_only(data: FitData, config: TrainConfig, *, seed: int, bits: int,
                  train_idx: np.ndarray, val_idx: np.ndarray,
                  scaler: FixedScaler | None = None) -> FitResult:
    """Select hard-code epoch on internal FIT, then refit from init on all FIT."""
    if data.sex is not None:
        raise ValueError("task-only trainer must not carry SEX")
    if data.role not in ("OSF_DEFENSE_FIT", "SYNTHETIC"):
        raise PermissionError("task-only fit requires admitted FIT role")
    train, val = _check_fold(data, train_idx, val_idx)
    torch.set_num_threads(1)
    fixed = FixedScaler.fit(data.h0, data.h1) if scaler is None else scaler
    if fixed.dim != data.h0.shape[1]:
        raise ValueError("scaler dimension mismatch")
    x = tuple(torch.from_numpy(fixed.transform(i, (data.h0, data.h1)[i])) for i in (0, 1))
    y = tuple(torch.from_numpy(np.asarray((data.y0, data.y1)[i])) for i in (0, 1))
    teacher = tuple(torch.from_numpy(np.asarray((data.teacher0, data.teacher1)[i]))
                    for i in (0, 1))
    fold = _new_model(seed, config.init_seed_offset, fixed.dim, bits)
    optimizer = torch.optim.Adam(fold.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    fold_plan = make_batch_plan(train, seed=seed + config.fold_batch_seed_offset,
                                epochs=config.epochs, batch_size=config.batch_size)
    fold_history: list[dict] = []
    best = float("inf")
    selected = 0
    stale = 0
    for epoch, batches in enumerate(fold_plan, 1):
        fit_loss = _epoch(fold, optimizer, x, y, teacher, batches,
                          config.distill_weight)
        score = _validation_nll(fold, x, y, val)
        fold_history.append({"epoch": epoch, "fit_task_loss": fit_loss,
                             "validation_nll": score,
                             "optimizer_updates": len(batches)})
        if score < best - 1e-12:
            best, selected, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    full = _new_model(seed, config.init_seed_offset, fixed.dim, bits)
    full_optimizer = torch.optim.Adam(full.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
    full_plan = make_batch_plan(np.arange(data.rows),
                                seed=seed + config.full_batch_seed_offset,
                                epochs=selected, batch_size=config.batch_size)
    full_history = []
    for epoch, batches in enumerate(full_plan, 1):
        fit_loss = _epoch(full, full_optimizer, x, y, teacher, batches,
                          config.distill_weight)
        full_history.append({"epoch": epoch, "fit_task_loss": fit_loss,
                             "optimizer_updates": len(batches)})
    full.eval()
    return FitResult(full, fixed, selected, fold_history, full_history,
                     sum(row["optimizer_updates"] for row in fold_history),
                     sum(row["optimizer_updates"] for row in full_history),
                     seed, bits, data.role, len(train), len(val))
