"""Task-only fitting stages for unchanged continuous-latent diagnostic F."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.nn import functional as F

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.fit import (
    FitData,
    TrainConfig,
    make_batch_plan,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import FixedScaler
from .unchanged import ContinuousHead, UnchangedLatentRelease


@dataclass
class FStageResult:
    heads: tuple[ContinuousHead, ContinuousHead]
    history: list[dict]
    seed: int
    train_rows: int
    optimizer_updates: int
    checkpoint_nll: dict[int, tuple[float, float]]

    @property
    def state_dicts(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return tuple({name: value.detach().cpu().clone()
                      for name, value in head.state_dict().items()}
                     for head in self.heads)


def fit_f_stage(data: FitData, config: TrainConfig, seed: int,
                scaler: FixedScaler, train_idx: np.ndarray,
                init_states: tuple[dict[str, torch.Tensor],
                                   dict[str, torch.Tensor]] | None = None,
                val_idx: np.ndarray | None = None,
                active_purposes: tuple[int, ...] = (0, 1)) -> FStageResult:
    """Fit or continue both 16-D F heads for the caller-fixed epoch count.

    The caller performs the locked split-fold task-NLL checkpoint selection and
    full-FIT refit; this stage never reads a calibration or INNER label.
    """
    train = np.asarray(train_idx)
    if (train.ndim != 1 or len(train) < 2 or not np.issubdtype(train.dtype, np.integer)
            or np.any(train < 0) or np.any(train >= data.rows)
            or len(np.unique(train)) != len(train)):
        raise ValueError("explicit unique permitted fitting indices required")
    if scaler.dim != data.h0.shape[1]:
        raise ValueError("shared scaler and latent dimensions differ")
    if (not active_purposes or set(active_purposes) - {0, 1}
            or len(set(active_purposes)) != len(active_purposes)
            or config.privacy_weight != 0):
        raise ValueError("F accepts active task heads and zero privacy weight only")
    if val_idx is not None:
        val = np.asarray(val_idx)
        if (val.ndim != 1 or len(val) < 2 or not np.issubdtype(val.dtype, np.integer)
                or np.any(val < 0) or np.any(val >= data.rows)
                or len(np.unique(val)) != len(val)
                or np.intersect1d(train, val).size
                or len(train) + len(val) != data.rows):
            raise ValueError("F internal FIT folds must partition admitted rows")
    else:
        val = None
    torch.set_num_threads(1)
    torch.manual_seed(int(seed) + 400009)
    heads = (ContinuousHead(scaler.dim, 2), ContinuousHead(scaler.dim, 6))
    if init_states is not None:
        if len(init_states) != 2:
            raise ValueError("two F task-head states required")
        for head, state in zip(heads, init_states):
            head.load_state_dict(copy.deepcopy(state), strict=True)
    optimizers = tuple(torch.optim.Adam(head.parameters(), lr=config.learning_rate)
                       for head in heads)
    x = tuple(torch.from_numpy(scaler.transform(i, h)) for i, h in enumerate((data.h0, data.h1)))
    y = tuple(torch.from_numpy(v) for v in (data.y0, data.y1))
    teacher = tuple(torch.from_numpy(v) for v in (data.teacher0, data.teacher1))
    batch_plan = make_batch_plan(train, seed=seed, epochs=config.epochs,
                                 batch_size=config.batch_size)
    history: list[dict] = []
    checkpoint_nll: dict[int, tuple[float, float]] = {}
    updates = 0
    for epoch in range(config.epochs):
        total = [0.0, 0.0]
        batches = 0
        for chunk in batch_plan[epoch]:
            ix = torch.from_numpy(chunk)
            for purpose in active_purposes:
                logits = heads[purpose](x[purpose][ix])
                loss = (F.cross_entropy(logits, y[purpose][ix])
                        + config.distill_weight * F.kl_div(
                            F.log_softmax(logits, dim=1), teacher[purpose][ix],
                            reduction="batchmean"))
                optimizers[purpose].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[purpose].step()
                total[purpose] += float(loss.detach())
            batches += 1
            updates += 1
        if val is not None and (epoch + 1) in config.checkpoints:
            with torch.no_grad():
                vi = torch.from_numpy(val)
                checkpoint_nll[epoch + 1] = tuple(float(F.cross_entropy(
                    heads[i](x[i][vi]), y[i][vi])) for i in (0, 1))
        history.append({"epoch": epoch + 1, "updates": updates,
                        "task_loss_by_purpose": tuple(v / batches for v in total),
                        "validation_nll_by_purpose": checkpoint_nll.get(epoch + 1)})
    for head in heads:
        head.eval()
    return FStageResult(heads, history, int(seed), len(train), updates,
                        checkpoint_nll)


@dataclass
class FReferenceResult:
    release: UnchangedLatentRelease
    selected_epochs: tuple[int, int]
    split_checkpoint_nll: dict[int, tuple[float, float]]
    full_fit_updates_by_purpose: tuple[int, int]
    split_updates_per_purpose: int


def fit_f_reference(data: FitData, config: TrainConfig, seed: int,
                    scaler: FixedScaler, train_idx: np.ndarray,
                    val_idx: np.ndarray, source_sha256: str) -> FReferenceResult:
    """Task-specific FIT-fold epoch selection and full-FIT F head refits."""
    if data.role == "OSF_DEFENSE_FIT" and (
            config.epochs != 40 or config.checkpoints != tuple(range(5, 41, 5))):
        raise ValueError("real F diagnostic requires locked 40-epoch checkpoint plan")
    split = fit_f_stage(data, config, seed, scaler, train_idx,
                        val_idx=val_idx)
    if not split.checkpoint_nll:
        raise ValueError("F split fit produced no selected checkpoint")
    epochs = tuple(min(split.checkpoint_nll,
                       key=lambda epoch: (split.checkpoint_nll[epoch][purpose], epoch))
                   for purpose in (0, 1))
    full = tuple(fit_f_stage(data, replace(config, epochs=epochs[purpose]),
                             seed, scaler, np.arange(data.rows),
                             active_purposes=(purpose,))
                 for purpose in (0, 1))
    heads = (full[0].heads[0], full[1].heads[1])
    release = UnchangedLatentRelease(heads, scaler, "F-unchanged-v1",
                                     source_sha256)
    return FReferenceResult(release, epochs, split.checkpoint_nll,
                            (full[0].optimizer_updates, full[1].optimizer_updates),
                            split.optimizer_updates)
