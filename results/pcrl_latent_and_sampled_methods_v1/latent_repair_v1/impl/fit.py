"""Equal-budget task and D-K hard-code fitting on explicitly admitted rows."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import BITS, FixedScaler, HardCodeModel


FIT_ROLES = frozenset({"OSF_DEFENSE_FIT", "SYNTHETIC"})


@dataclass(frozen=True)
class FitData:
    h0: np.ndarray
    h1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    teacher0: np.ndarray
    teacher1: np.ndarray
    sex: np.ndarray | None
    row_ids: np.ndarray
    role: str

    def __post_init__(self) -> None:
        if self.role not in FIT_ROLES:
            raise PermissionError("training requires admitted defense-fit or synthetic role")
        hs = [np.asarray(h, dtype=np.float32) for h in (self.h0, self.h1)]
        if (any(h.ndim != 2 or len(h) < 4 or not np.isfinite(h).all() for h in hs)
                or hs[0].shape != hs[1].shape):
            raise ValueError("aligned finite purpose latents required")
        n = len(hs[0])
        ids = np.asarray(self.row_ids)
        if ids.shape != (n,) or len(np.unique(ids)) != n:
            raise ValueError("one unique aligned row ID per person required")
        for purpose in (0, 1):
            y = np.asarray((self.y0, self.y1)[purpose])
            teacher = np.asarray((self.teacher0, self.teacher1)[purpose], dtype=np.float32)
            if (y.shape != (n,) or not np.issubdtype(y.dtype, np.integer)
                    or teacher.ndim != 2 or teacher.shape[0] != n
                    or teacher.shape[1] != (2, 6)[purpose]
                    or np.any(y < 0) or np.any(y >= teacher.shape[1])
                    or not np.isfinite(teacher).all() or np.any(teacher < 0)
                    or not np.allclose(teacher.sum(axis=1), 1, rtol=0, atol=1e-5)):
                raise ValueError("task labels and private teacher probabilities invalid")
            object.__setattr__(self, f"y{purpose}", y.astype(np.int64, copy=True))
            object.__setattr__(self, f"teacher{purpose}", teacher.copy())
        if self.sex is not None:
            sex = np.asarray(self.sex)
            if sex.shape != (n,) or not np.isin(sex, (0, 1)).all():
                raise ValueError("binary fit-role SEX array required")
            object.__setattr__(self, "sex", sex.astype(np.int64, copy=True))
        object.__setattr__(self, "h0", hs[0].copy())
        object.__setattr__(self, "h1", hs[1].copy())
        object.__setattr__(self, "row_ids", ids.copy())

    @property
    def rows(self) -> int:
        return int(len(self.h0))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 0.001
    distill_weight: float = 0.5
    privacy_weight: float = 1.0
    checkpoints: tuple[int, ...] = (5, 10, 15, 20)

    def __post_init__(self) -> None:
        if (self.epochs < 1 or self.batch_size < 2 or self.learning_rate <= 0
                or not np.isfinite(self.learning_rate)
                or self.distill_weight < 0 or not np.isfinite(self.distill_weight)
                or self.privacy_weight < 0 or not np.isfinite(self.privacy_weight)
                or any(e < 1 for e in self.checkpoints)
                or tuple(sorted(set(self.checkpoints))) != self.checkpoints):
            raise ValueError("invalid fixed training configuration")


@dataclass
class FitResult:
    model: HardCodeModel
    scaler: FixedScaler
    history: list[dict]
    selected_epoch: int
    seed: int
    role: str
    mode: str
    train_rows: int
    validation_rows: int
    optimizer_updates: int

    @property
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


def multiscale_mmd_terms(c0: torch.Tensor, c1: torch.Tensor,
                         sex: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bounded kernel module's independently specified biased MMD terms."""
    from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.kernel.objective import hard_code_mmd
    terms = hard_code_mmd(c0, c1, sex)
    return terms.local0, terms.local1, terms.pair


def _folds(data: FitData, seed: int, train_idx: np.ndarray | None,
           val_idx: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if (train_idx is None) != (val_idx is None):
        raise ValueError("both frozen internal fold arrays must be supplied")
    if train_idx is None:
        if data.role != "SYNTHETIC":
            raise ValueError("real fits require caller-supplied frozen internal fold IDs")
        ix = np.random.default_rng(seed + 7919).permutation(data.rows)
        n_val = max(2, data.rows // 10)
        train_idx, val_idx = ix[n_val:], ix[:n_val]
    train, val = np.asarray(train_idx), np.asarray(val_idx)
    for ix in (train, val):
        if (ix.ndim != 1 or len(ix) < 2 or not np.issubdtype(ix.dtype, np.integer)
                or np.any(ix < 0) or np.any(ix >= data.rows)
                or len(np.unique(ix)) != len(ix)):
            raise ValueError("invalid internal fold indices")
    if np.intersect1d(train, val).size or len(train) + len(val) != data.rows:
        raise ValueError("internal folds must partition defense-fit rows")
    return train.astype(np.int64), val.astype(np.int64)


def _task_loss(logits: tuple[torch.Tensor, torch.Tensor],
               y: tuple[torch.Tensor, torch.Tensor],
               teachers: tuple[torch.Tensor, torch.Tensor],
               distill_weight: float) -> torch.Tensor:
    parts = []
    for i in (0, 1):
        ce = F.cross_entropy(logits[i], y[i])
        kl = F.kl_div(F.log_softmax(logits[i], dim=1), teachers[i], reduction="batchmean")
        parts.append(ce + distill_weight * kl)
    return (parts[0] + parts[1]) / 2


def _task_nll(logits: tuple[torch.Tensor, torch.Tensor],
              y: tuple[torch.Tensor, torch.Tensor]) -> float:
    return float((F.cross_entropy(logits[0], y[0])
                  + F.cross_entropy(logits[1], y[1])) / 2)


def make_batch_plan(train_idx: np.ndarray, *, seed: int, epochs: int,
                    batch_size: int = 256, sex: np.ndarray | None = None
                    ) -> tuple[tuple[np.ndarray, ...], ...]:
    """Fixed-size aligned batches; merge an invalid short final group once.

    A caller may pass the same sealed plan to T without giving its trainer SEX.
    """
    train = np.asarray(train_idx, dtype=np.int64)
    if (train.ndim != 1 or len(train) < 2 or len(np.unique(train)) != len(train)
            or epochs < 1 or batch_size < 2):
        raise ValueError("invalid batch plan inputs")
    groups = None if sex is None else np.asarray(sex)
    if groups is not None and (np.any(train < 0) or np.any(train >= len(groups))
                               or not np.isin(groups[train], (0, 1)).all()):
        raise ValueError("invalid binary batch-plan group array")
    rng = np.random.default_rng(seed + 104729)
    plan = []
    for _ in range(epochs):
        permutation = rng.permutation(train)
        chunks = [permutation[start:start + batch_size]
                  for start in range(0, len(permutation), batch_size)]
        if groups is not None and len(chunks) > 1 and len(chunks[-1]) < batch_size:
            tail = groups[chunks[-1]]
            if not (np.any(tail == 0) and np.any(tail == 1)):
                chunks[-2] = np.concatenate((chunks[-2], chunks[-1]))
                chunks.pop()
        if groups is not None and any(not (np.any(groups[ix] == 0)
                                          and np.any(groups[ix] == 1)) for ix in chunks):
            raise ValueError("non-tail FIT minibatch lacks one SEX group")
        plan.append(tuple(np.asarray(ix, dtype=np.int64) for ix in chunks))
    return tuple(plan)


def fit_arm(
    data: FitData,
    config: TrainConfig,
    seed: int,
    mode: str = "T",
    init_state: dict[str, torch.Tensor] | None = None,
    scaler: FixedScaler | None = None,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    active_purposes: tuple[int, ...] = (0, 1),
    pair_anchor: np.ndarray | None = None,
    full_fit: bool = False,
    batch_plan: tuple[tuple[np.ndarray, ...], ...] | None = None,
) -> FitResult:
    """One fixed-budget stage; T and zero-weight D-K have identical updates.

    `pair_anchor` is the already released hard code for the inactive purpose,
    aligned to `data.row_ids`. Sequential callers supply it after freezing the
    first fitted release. A fresh optimizer is used for every continuation.
    """
    if mode not in ("T", "D-K", "L"):
        raise ValueError("unknown D training mode")
    if not active_purposes or set(active_purposes) - {0, 1} or len(set(active_purposes)) != len(active_purposes):
        raise ValueError("invalid active recipient set")
    if mode == "T" and data.sex is not None:
        raise ValueError("task-only trainer object must not carry SEX")
    if mode != "T" and config.privacy_weight > 0 and data.sex is None:
        raise ValueError("protected arm requires defense-fit SEX labels")
    if data.role == "OSF_DEFENSE_FIT" and mode != "T" and init_state is None:
        raise ValueError("real protected continuation requires cloned selected T initialization")
    if pair_anchor is not None:
        anchor = np.asarray(pair_anchor)
        if (len(active_purposes) != 1 or anchor.shape != (data.rows, BITS)
                or not np.isin(anchor, (0, 1)).all() or mode != "D-K"):
            raise ValueError("pair anchor must be actual aligned ten-bit frozen release")
        anchor = torch.from_numpy(anchor.astype(np.float32, copy=True))
    else:
        anchor = None
    if full_fit:
        if train_idx is not None or val_idx is not None:
            raise ValueError("full FIT refit cannot use internal fold indices")
        train = np.arange(data.rows, dtype=np.int64)
        val = np.empty(0, dtype=np.int64)
    else:
        train, val = _folds(data, seed, train_idx, val_idx)
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    scaler = FixedScaler.fit(data.h0, data.h1) if scaler is None else scaler
    if scaler.dim != data.h0.shape[1]:
        raise ValueError("shared scaler and latents disagree")
    model = HardCodeModel(scaler.dim, (2, 6))
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
    for purpose in (0, 1):
        if purpose not in active_purposes:
            for p in (*model.mappers[purpose].parameters(), *model.heads[purpose].parameters()):
                p.requires_grad_(False)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    x = tuple(torch.from_numpy(scaler.transform(i, (data.h0, data.h1)[i])) for i in (0, 1))
    y = tuple(torch.from_numpy((data.y0, data.y1)[i]) for i in (0, 1))
    teacher = tuple(torch.from_numpy((data.teacher0, data.teacher1)[i]) for i in (0, 1))
    sex = None if data.sex is None else torch.from_numpy(data.sex)
    if batch_plan is None:
        batch_plan = make_batch_plan(train, seed=seed, epochs=config.epochs,
                                     batch_size=config.batch_size, sex=data.sex)
    if len(batch_plan) != config.epochs:
        raise ValueError("batch plan epoch count differs from fixed horizon")
    for epoch_batches in batch_plan:
        concat = np.concatenate(epoch_batches)
        if (len(concat) != len(train) or len(np.unique(concat)) != len(train)
                or not np.array_equal(np.sort(concat), np.sort(train))):
            raise ValueError("batch plan does not partition the selected FIT fold")
    history: list[dict] = []
    best_state = None
    best_val = float("inf")
    selected_epoch = 0
    updates = 0
    for epoch in range(config.epochs):
        model.train()
        task_total = privacy_total = 0.0
        batches = 0
        for chunk in batch_plan[epoch]:
            ix = torch.from_numpy(chunk)
            if len(active_purposes) == 2:
                codes, logits = zip(*(model(i, x[i][ix], True) for i in (0, 1)))
                task = _task_loss(logits, (y[0][ix], y[1][ix]),
                                  (teacher[0][ix], teacher[1][ix]), config.distill_weight)
            else:
                active = active_purposes[0]
                code_active, logits_active = model(active, x[active][ix], True)
                # This is the active purpose's contribution to the joint loss.
                task = (F.cross_entropy(logits_active, y[active][ix])
                        + config.distill_weight * F.kl_div(
                            F.log_softmax(logits_active, dim=1), teacher[active][ix],
                            reduction="batchmean")) / 2
            privacy = task.new_zeros(())
            if mode != "T" and config.privacy_weight > 0 and sex is not None:
                if len(active_purposes) == 2:
                    bits0, bits1 = codes
                else:
                    frozen = 1 - active_purposes[0]
                    if anchor is None:
                        # First freeze-first stage has only the active local term.
                        bits0 = bits1 = code_active
                    else:
                        bits0 = anchor[ix] if frozen == 0 else code_active
                        bits1 = anchor[ix] if frozen == 1 else code_active
                group = sex[ix]
                if not (bool(torch.any(group == 0)) and bool(torch.any(group == 1))):
                    raise ValueError("FIT minibatch lacks one SEX group; no silent privacy skip")
                if len(active_purposes) == 2 or anchor is not None:
                    m0, m1, pair = multiscale_mmd_terms(bits0, bits1, group)
                    if len(active_purposes) == 2:
                        privacy = ((m0 + m1 + pair) / 3 if mode == "D-K"
                                   else (m0 + m1) / 3).to(task.dtype)
                    else:
                        privacy = (((m0 if active_purposes[0] == 0 else m1)
                                    + pair) / 3).to(task.dtype)
                else:
                    # One-purpose local-only stage: same local coefficient as joint D-K.
                    local, _, _ = multiscale_mmd_terms(bits0, bits1, group)
                    privacy = (local / 3).to(task.dtype)
            loss = task + config.privacy_weight * privacy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            task_total += float(task.detach())
            privacy_total += float(privacy.detach())
            batches += 1
            updates += 1
        val_task = None
        if not full_fit and (epoch + 1) in config.checkpoints:
            model.eval()
            with torch.no_grad():
                val_ix = torch.from_numpy(val)
                val_logits = tuple(model(i, x[i][val_ix], False)[1] for i in (0, 1))
                val_task = _task_nll(val_logits, (y[0][val_ix], y[1][val_ix]))
            if val_task < best_val:
                best_val = val_task
                best_state = copy.deepcopy(model.state_dict())
                selected_epoch = epoch + 1
        history.append({"epoch": epoch + 1, "updates": updates,
                        "task_loss": task_total / batches,
                        "privacy_loss": privacy_total / batches,
                        "validation_task_nll": val_task})
    if full_fit:
        selected_epoch = config.epochs
    else:
        if best_state is None:
            raise ValueError("no checkpoint in configured training horizon")
        model.load_state_dict(best_state, strict=True)
    model.eval()
    return FitResult(model, scaler, history, selected_epoch, int(seed), data.role,
                     mode, len(train), len(val), updates)


def calibrate_privacy_weight(data: FitData, warm_state: dict[str, torch.Tensor],
                             scaler: FixedScaler, seed: int,
                             train_idx: np.ndarray, val_idx: np.ndarray,
                             distill_weight: float = 0.5,
                             batch_size: int = 256) -> dict:
    """One FIT-only no-update gradient-ratio receipt for fixed D-K/L/S lambda."""
    if data.sex is None:
        raise ValueError("FIT-only coefficient diagnostic requires SEX")
    train, _ = _folds(data, seed, train_idx, val_idx)
    ix = np.random.default_rng(seed + 104729).permutation(train)[:batch_size]
    if len(ix) != batch_size:
        raise ValueError("coefficient batch must contain 256 aligned people")
    sex = torch.from_numpy(data.sex[ix])
    if not (bool((sex == 0).any()) and bool((sex == 1).any())):
        raise ValueError("fixed coefficient batch lacks a SEX group")
    torch.manual_seed(int(seed))
    model = HardCodeModel(scaler.dim, (2, 6))
    model.load_state_dict(warm_state, strict=True)
    x = (torch.from_numpy(scaler.transform(0, data.h0[ix])),
         torch.from_numpy(scaler.transform(1, data.h1[ix])))
    codes, logits = zip(*(model(i, x[i], True) for i in (0, 1)))
    task = _task_loss(logits,
                      (torch.from_numpy(data.y0[ix]), torch.from_numpy(data.y1[ix])),
                      (torch.from_numpy(data.teacher0[ix]), torch.from_numpy(data.teacher1[ix])),
                      distill_weight)
    m0, m1, pair = multiscale_mmd_terms(codes[0], codes[1], sex)
    privacy = ((m0 + m1 + pair) / 3).to(task.dtype)
    parameters = tuple(model.mappers.parameters())
    g_task = torch.autograd.grad(task, parameters, retain_graph=True)
    g_priv = torch.autograd.grad(privacy, parameters)
    task_norm = float(torch.sqrt(sum((g.detach().double() ** 2).sum() for g in g_task)))
    privacy_norm = float(torch.sqrt(sum((g.detach().double() ** 2).sum() for g in g_priv)))
    raw_ratio = float("inf") if privacy_norm == 0 else 0.5 * task_norm / (privacy_norm + 1e-12)
    coefficient = float(np.clip(raw_ratio, 0.1, 10.0))
    row_hash = hashlib.sha256("\n".join(map(str, data.row_ids[ix])).encode()).hexdigest()
    return {"schema": "d-k-fit-gradient-ratio-v1", "role": data.role,
            "seed": int(seed), "batch_rows": len(ix), "batch_row_order_sha256": row_hash,
            "task_mapper_grad_l2": task_norm, "privacy_mapper_grad_l2": privacy_norm,
            "unclipped_ratio": raw_ratio if np.isfinite(raw_ratio) else None,
            "privacy_weight": coefficient,
            "clip_event": "upper" if raw_ratio > 10 else "lower" if raw_ratio < 0.1 else "none",
            "parameter_updates": 0}
