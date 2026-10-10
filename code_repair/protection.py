"""Matched hard-code continuations, conditional on a sealed task-only witness.

This module reads no data files. A separate queue worker must validate the R0
stage seal before constructing real FitData. Real-role calls here also require
the matching R0 gate and stage-lock hashes, so a direct call fails closed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F

from code_witness.model import FixedScaler, HardCodeModel
from code_witness.learner import FitData, TrainConfig, make_batch_plan, task_loss
from code_repair.kernel import hard_code_mmd
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.adversary import HardCodeCritics


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Canonical tensor-state identity used to bind the immutable T0 start."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = np.ascontiguousarray(state[name].detach().cpu().numpy())
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(str(value.shape).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def calibrate_kernel_strength(data: FitData, *, config: TrainConfig, seed: int,
                              bits: int, init_state: dict[str, torch.Tensor],
                              scaler: FixedScaler, first_batch: np.ndarray,
                              admission: dict | None = None) -> dict:
    """One no-update FIT gradient-ratio receipt at the immutable T0 start."""
    _admit(data, bits, seed, init_state, admission)
    if data.sex is None:
        raise ValueError("kernel gradient ratio requires authorized FIT SEX")
    ix = np.asarray(first_batch, dtype=np.int64)
    if (ix.ndim != 1 or len(ix) < 2 or len(ix) > config.batch_size
            or len(np.unique(ix)) != len(ix) or np.any(ix < 0)
            or np.any(ix >= data.rows)):
        raise ValueError("one valid presealed first FIT batch required")
    group = torch.from_numpy(data.sex[ix])
    if not (bool((group == 0).any()) and bool((group == 1).any())):
        raise ValueError("first FIT batch lacks one SEX group")
    torch.set_num_threads(1)
    model = HardCodeModel(scaler.dim, (2, 6), bits)
    model.load_state_dict(init_state, strict=True)
    x = tuple(torch.from_numpy(scaler.transform(i, h[ix]))
              for i, h in enumerate((data.h0, data.h1)))
    codes, logits = zip(*(model(i, x[i], True) for i in (0, 1)))
    y = tuple(torch.from_numpy(v[ix]) for v in (data.y0, data.y1))
    teachers = tuple(torch.from_numpy(v[ix])
                     for v in (data.teacher0, data.teacher1))
    task = task_loss(logits, y, teachers, config.distill_weight)
    privacy = hard_code_mmd(codes[0], codes[1], group).joint
    parameters = tuple(model.mappers.parameters())
    g_task = torch.autograd.grad(task, parameters, retain_graph=True)
    g_priv = torch.autograd.grad(privacy, parameters)
    task_norm = float(torch.sqrt(sum((g.detach().double() ** 2).sum()
                                     for g in g_task)))
    privacy_norm = float(torch.sqrt(sum((g.detach().double() ** 2).sum()
                                        for g in g_priv)))
    raw = float("inf") if privacy_norm == 0 else 0.5 * task_norm / (privacy_norm + 1e-12)
    coefficient = float(np.clip(raw, 0.1, 10.0))
    row_bytes = np.ascontiguousarray(data.row_ids[ix]).tobytes()
    return {"schema": "pcrl-code-witness-kernel-gradient-ratio-v1",
            "seed": int(seed), "bits": bits, "role": data.role,
            "batch_rows": len(ix),
            "batch_row_order_sha256": hashlib.sha256(row_bytes).hexdigest(),
            "task_mapper_grad_l2": task_norm,
            "privacy_mapper_grad_l2": privacy_norm,
            "unclipped_ratio": raw if np.isfinite(raw) else None,
            "privacy_weight": coefficient,
            "clip_event": "upper" if raw > 10 else "lower" if raw < 0.1 else "none",
            "parameter_updates": 0}


def _admit(data: FitData, bits: int, seed: int, init_state: dict[str, torch.Tensor],
           admission: dict | None) -> None:
    if bits not in (10, 16):
        raise ValueError("only predeclared ten- or sixteen-bit codes are allowed")
    if data.role == "SYNTHETIC":
        return
    if data.role != "OSF_DEFENSE_FIT":
        raise PermissionError("protection only accepts authorized defense fitting")
    if (not isinstance(admission, dict)
            or admission.get("schema") != "pcrl-code-witness-capacity-gate-v1"
            or admission.get("status") != "POINT_GATE_PASS"
            or admission.get("all_two_seed_task_reference_point_caps_pass") is not True
            or admission.get("explicit_r0_go") is not True
            or admission.get("selected_bits") != bits
            or seed not in admission.get("seeds", ())
            or admission.get("t0_state_sha256_by_seed", {}).get(str(seed)) != state_sha256(init_state)
            or not all(isinstance(admission.get(key), str)
                       and len(admission[key]) == 64
                       for key in ("source_lock_sha256", "stage_lock_sha256"))):
        raise PermissionError("real protection fit requires sealed R0 point gate and stage lock")


def _plan(data: FitData, config: TrainConfig, seed: int,
          batch_plan: tuple[tuple[np.ndarray, ...], ...] | None) -> tuple:
    if batch_plan is None:
        batch_plan = make_batch_plan(np.arange(data.rows), seed=seed,
                                     epochs=config.epochs, batch_size=config.batch_size,
                                     sex=data.sex)
    if len(batch_plan) != config.epochs:
        raise ValueError("batch plan does not match sealed epoch count")
    expected = np.arange(data.rows)
    for epoch in batch_plan:
        if not epoch:
            raise ValueError("empty training epoch")
        ix = np.concatenate(epoch)
        if (len(ix) != data.rows or not np.array_equal(np.sort(ix), expected)
                or any(len(chunk) < 2 for chunk in epoch)):
            raise ValueError("batch plan must partition every aligned FIT person")
    return batch_plan


@dataclass
class ContinueResult:
    model: HardCodeModel
    mode: str
    history: list[dict]
    optimizer_updates: int
    seed: int
    bits: int

    @property
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone()
                for name, value in self.model.state_dict().items()}


def fit_continue(data: FitData, *, config: TrainConfig, seed: int, bits: int,
                 init_state: dict[str, torch.Tensor], scaler: FixedScaler,
                 mode: str, privacy_weight: float,
                 batch_plan: tuple[tuple[np.ndarray, ...], ...] | None = None,
                 active_purposes: tuple[int, ...] = (0, 1),
                 first_anchor: np.ndarray | None = None,
                 admission: dict | None = None) -> ContinueResult:
    """One full-FIT continuation with identical mapper/head update budget."""
    _admit(data, bits, seed, init_state, admission)
    if (mode not in ("T+", "L", "J") or not np.isfinite(privacy_weight)
            or privacy_weight < 0 or not active_purposes
            or len(set(active_purposes)) != len(active_purposes)
            or set(active_purposes) - {0, 1}
            or (mode == "T+" and data.sex is not None)
            or (mode != "T+" and data.sex is None)
            or (first_anchor is not None and (mode != "J" or len(active_purposes) != 1))
            or (mode == "J" and len(active_purposes) == 1 and first_anchor is None)):
        raise ValueError("continuation mode, strength, SEX or active-purpose contract invalid")
    if first_anchor is not None:
        anchor = np.asarray(first_anchor)
        if (anchor.shape != (data.rows, bits)
                or not np.isin(anchor, (0, 1)).all()):
            raise ValueError("aligned frozen first hard code required")
        anchor = torch.from_numpy(anchor.astype(np.float32, copy=True))
    else:
        anchor = None
    if scaler.dim != data.h0.shape[1]:
        raise ValueError("T0 scaler and frozen latents differ")
    plan = _plan(data, config, seed, batch_plan)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = HardCodeModel(scaler.dim, (2, 6), bits)
    model.load_state_dict(init_state, strict=True)
    for purpose in (0, 1):
        if purpose not in active_purposes:
            for param in (*model.mappers[purpose].parameters(),
                          *model.heads[purpose].parameters()):
                param.requires_grad_(False)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=config.learning_rate, weight_decay=config.weight_decay)
    x = tuple(torch.from_numpy(scaler.transform(i, h))
              for i, h in enumerate((data.h0, data.h1)))
    y = tuple(torch.from_numpy(v) for v in (data.y0, data.y1))
    teacher = tuple(torch.from_numpy(v) for v in (data.teacher0, data.teacher1))
    sex = None if data.sex is None else torch.from_numpy(data.sex)
    history = []
    updates = 0
    for epoch_batches in plan:
        model.train()
        task_sum = privacy_sum = 0.0
        for chunk in epoch_batches:
            ix = torch.from_numpy(np.asarray(chunk, dtype=np.int64))
            if len(active_purposes) == 2:
                codes, logits = zip(*(model(i, x[i][ix], True) for i in (0, 1)))
                task = task_loss(logits, (y[0][ix], y[1][ix]),
                                 (teacher[0][ix], teacher[1][ix]),
                                 config.distill_weight)
            else:
                i = active_purposes[0]
                code, logit = model(i, x[i][ix], True)
                task = (F.cross_entropy(logit, y[i][ix])
                        + config.distill_weight * F.kl_div(
                            F.log_softmax(logit, dim=1), teacher[i][ix],
                            reduction="batchmean")) / 2
            privacy = task.new_zeros(())
            if mode != "T+" and privacy_weight > 0:
                group = sex[ix]
                if not (bool((group == 0).any()) and bool((group == 1).any())):
                    raise ValueError("protected FIT batch lacks a SEX group")
                if len(active_purposes) == 2:
                    terms = hard_code_mmd(codes[0], codes[1], group)
                    privacy = terms.joint if mode == "J" else terms.local
                elif anchor is not None:
                    frozen = anchor[ix]
                    b0, b1 = (frozen, code) if i == 1 else (code, frozen)
                    terms = hard_code_mmd(b0, b1, group)
                    privacy = ((terms.local0 if i == 0 else terms.local1)
                               + terms.pair) / 3
                else:
                    # A first-stage or single-purpose local term has lambda/3.
                    terms = hard_code_mmd(code, code, group)
                    privacy = terms.local0 / 3
            loss = task + privacy_weight * privacy.to(task.dtype)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            task_sum += float(task.detach())
            privacy_sum += float(privacy.detach())
            updates += 1
        history.append({"epoch": len(history) + 1, "optimizer_updates": updates,
                        "task_loss": task_sum / len(epoch_batches),
                        "privacy_loss": privacy_sum / len(epoch_batches)})
    model.eval()
    return ContinueResult(model, mode, history, updates, seed, bits)


@dataclass
class SequentialResult:
    model: HardCodeModel
    mode: str
    first_calibrator: dict
    first_code: np.ndarray
    first_code_sha256: str
    first_state_sha256: str
    updates_by_purpose: tuple[int, int]
    first_history: list[dict]
    second_history: list[dict]


def fit_sequential(data: FitData, *, config: TrainConfig, seed: int, bits: int,
                   init_state: dict[str, torch.Tensor], scaler: FixedScaler,
                   mode: str, privacy_weight: float,
                   calibrate_first: Callable[[HardCodeModel, int], dict],
                   batch_plan: tuple[tuple[np.ndarray, ...], ...] | None = None,
                   admission: dict | None = None) -> SequentialResult:
    """Fit, calibrate and freeze one recipient before the other is trained."""
    _admit(data, bits, seed, init_state, admission)
    if mode not in ("S12", "S21") or not callable(calibrate_first):
        raise ValueError("a freeze-first order and calibration callback are required")
    first = 0 if mode == "S12" else 1
    second = 1 - first
    one = fit_continue(data, config=config, seed=seed, bits=bits,
                       init_state=init_state, scaler=scaler, mode="L",
                       privacy_weight=privacy_weight, batch_plan=batch_plan,
                       active_purposes=(first,), admission=admission)
    first_calibrator = copy.deepcopy(calibrate_first(one.model, first))
    if not isinstance(first_calibrator, dict) or not first_calibrator:
        raise ValueError("first release calibration did not return a record")
    with torch.no_grad():
        latent = (data.h0, data.h1)[first]
        first_code = one.model(first, torch.from_numpy(scaler.transform(first, latent)),
                               False)[0].to(torch.uint8).numpy().copy()
    if first_code.shape != (data.rows, bits):
        raise AssertionError("first release emitted wrong hard-code width")
    first_state = one.state_dict
    two = fit_continue(data, config=config, seed=seed, bits=bits,
                       init_state=first_state, scaler=scaler, mode="J",
                       privacy_weight=privacy_weight, batch_plan=batch_plan,
                       active_purposes=(second,), first_anchor=first_code,
                       admission=admission)
    for name, value in first_state.items():
        if name.startswith((f"mappers.{first}.", f"heads.{first}.")):
            if not torch.equal(value, two.model.state_dict()[name]):
                raise AssertionError("first complete model changed in second stage")
    digest = hashlib.sha256(np.ascontiguousarray(first_code).tobytes()).hexdigest()
    updates = [0, 0]
    updates[first], updates[second] = one.optimizer_updates, two.optimizer_updates
    return SequentialResult(two.model, mode, first_calibrator, first_code,
                            digest, state_sha256(first_state), tuple(updates),
                            one.history, two.history)


@dataclass
class AdversarialResult:
    model: HardCodeModel
    critics: HardCodeCritics
    history: list[dict]
    optimizer_updates: int
    critic_updates: int


def fit_adversarial(data: FitData, *, config: TrainConfig, seed: int, bits: int,
                    init_state: dict[str, torch.Tensor], scaler: FixedScaler,
                    coefficient: float = 1.0,
                    batch_plan: tuple[tuple[np.ndarray, ...], ...] | None = None,
                    admission: dict | None = None) -> AdversarialResult:
    """Matched ordinary C: two hard-code critic steps per mapper step."""
    _admit(data, bits, seed, init_state, admission)
    if data.sex is None or not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("C needs FIT SEX and finite nonnegative fixed coefficient")
    plan = _plan(data, config, seed, batch_plan)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = HardCodeModel(scaler.dim, (2, 6), bits)
    model.load_state_dict(init_state, strict=True)
    torch.manual_seed(seed + 300007)
    critics = HardCodeCritics(bits=bits, hidden=64)
    mapper_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                        weight_decay=config.weight_decay)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=0.001)
    x = tuple(torch.from_numpy(scaler.transform(i, h))
              for i, h in enumerate((data.h0, data.h1)))
    y = tuple(torch.from_numpy(v) for v in (data.y0, data.y1))
    teacher = tuple(torch.from_numpy(v) for v in (data.teacher0, data.teacher1))
    sex = torch.from_numpy(data.sex)
    history = []
    mapper_updates = critic_updates = 0
    for epoch_batches in plan:
        model.train()
        task_sum = critic_sum = 0.0
        for chunk in epoch_batches:
            ix = torch.from_numpy(np.asarray(chunk, dtype=np.int64))
            group = sex[ix]
            if not (bool((group == 0).any()) and bool((group == 1).any())):
                raise ValueError("C FIT batch lacks a SEX group")
            with torch.no_grad():
                detached = tuple(model(i, x[i][ix], False)[0] for i in (0, 1))
            for _ in range(2):
                receipt = critics.critic_step(detached[0], detached[1], group,
                                               critic_optimizer)
                critic_updates += 1
                critic_sum += sum(receipt.values()) / 3
            codes, logits = zip(*(model(i, x[i][ix], True) for i in (0, 1)))
            task = task_loss(logits, (y[0][ix], y[1][ix]),
                             (teacher[0][ix], teacher[1][ix]),
                             config.distill_weight)
            adversary = critics.mapper_loss(codes[0], codes[1], group, coefficient)
            mapper_optimizer.zero_grad(set_to_none=True)
            (task + adversary).backward()
            mapper_optimizer.step()
            mapper_updates += 1
            task_sum += float(task.detach())
        history.append({"epoch": len(history) + 1,
                        "optimizer_updates": mapper_updates,
                        "critic_updates": critic_updates,
                        "task_loss": task_sum / len(epoch_batches),
                        "critic_balanced_bce": critic_sum / (2 * len(epoch_batches))})
    model.eval()
    critics.eval()
    return AdversarialResult(model, critics, history, mapper_updates, critic_updates)
