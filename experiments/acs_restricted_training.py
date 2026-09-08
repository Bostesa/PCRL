"""Matched teacher/full-input adaptation and teacher-only source banks.

This separately named study leaves historical training contracts untouched.
The restricted boundary accepts only frozen teacher coordinates; its zero raw
block is constructed here. Original source heads/decoder and input scales are
loaded explicitly, while every mapper begins at the same teacher identity.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from experiments import acs_bottleneck_training as old
from experiments.acs_transfer_heads import _network
from experiments.acs_transfer_models import array_digest, state_digest

SOURCE_SCHEMA = old.SOURCE_SCHEMA
ATTRIBUTE_SCHEMA = old.ATTRIBUTE_SCHEMA
CONFIG = {**copy.deepcopy(old.TRAIN_CONFIG), "study": "acs_restricted_inputs_v1",
          "input_dim": 48, "teacher_input_dim": 16, "raw_input_dim": 32,
          "reconstruction_weight": 0., "preservation_beta": 1.,
          "Q_seed_base": 20260909, "initialization": "shifted teacher identity"}


def generate_q(seed):
    """Dedicated float64 Gaussian QR, with nonnegative diagonal R convention."""
    if int(seed) != seed or seed not in (0, 1, 2):
        raise ValueError("Frozen seeds are 0,1,2")
    q, r = np.linalg.qr(np.random.default_rng(20260909 + int(seed)).standard_normal((16, 16)))
    q *= np.where(np.diag(r) < 0., -1., 1.)[None, :]
    if not np.allclose(q.T @ q, np.eye(16), atol=1e-12, rtol=1e-12):
        raise AssertionError("QR did not produce orthogonal Q")
    return q


def _features(value, width, label):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != width or not len(value) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite nonempty float32 n-by-{width} matrix")
    return value


def _statistics(value):
    mean, scale = (np.asarray(value[key], np.float64) for key in ("mean", "scale"))
    if mean.shape != (32,) or scale.shape != (32,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Saved original PCA32 fitting means and positive scales required")
    return mean.copy(), scale.copy()


class RestrictedModel(nn.Module):
    def __init__(self, fitting_statistics, initial_state, orthogonal_q, *, access):
        super().__init__()
        if access not in ("F", "K"):
            raise ValueError("Access must be F or K")
        self.access = access
        mean, scale = _statistics(fitting_statistics)
        q = np.asarray(orthogonal_q, np.float64)
        if q.shape != (16, 16) or not np.isfinite(q).all() or not np.allclose(q.T @ q, np.eye(16), atol=1e-12, rtol=1e-12):
            raise ValueError("Explicit frozen orthogonal16 Q required")
        for key, expected in (("input_mean", mean), ("input_scale", scale)):
            if key not in initial_state or not np.array_equal(initial_state[key].cpu().numpy(), expected):
                raise ValueError("Original checkpoint preprocessing differs from saved statistics")
        self.register_buffer("input_mean", torch.from_numpy(mean))
        self.register_buffer("input_scale", torch.from_numpy(scale))
        self.register_buffer("orthogonal_q", torch.from_numpy(q.copy()))
        # Constructor RNG is private; copied heads/decoder never inherit the
        # changed random draw count of the wider mapper constructor.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.mapper = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 16))
            self.heads = nn.ModuleDict({name: nn.Linear(16, 1) for name in SOURCE_SCHEMA})
            self.decoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 32))
        for prefix, module in (("heads", self.heads), ("decoder", self.decoder)):
            copied = {key[len(prefix)+1:]: value.detach().clone() for key, value in initial_state.items() if key.startswith(prefix + ".")}
            module.load_state_dict(copied, strict=True)
        with torch.no_grad():
            identity, q32 = torch.eye(16), self.orthogonal_q.float()
            self.mapper[0].weight.zero_()
            self.mapper[0].weight[:, :16].copy_(torch.cat((identity, -identity, q32, -q32)))
            self.mapper[0].bias.copy_(torch.cat((torch.ones(16), -torch.ones(16), torch.ones(16), -torch.ones(16))))
            s = self.input_scale[:16].float()
            self.mapper[2].weight.copy_(torch.cat((torch.diag(s), -torch.diag(s), torch.zeros(16, 32)), 1))
            self.mapper[2].bias.copy_(self.input_mean[:16].float() - s)

    def standardize(self, teacher, *, raw=None):
        teacher = _features(teacher, 16, "Frozen teacher")
        u = ((teacher.astype(np.float64) - self.input_mean[:16].numpy()) / self.input_scale[:16].numpy()).astype(np.float32)
        if self.access == "K":
            if raw is not None:
                raise ValueError("Restricted K boundary rejects raw inputs; zero32 is constructed internally")
            v = np.zeros((len(teacher), 32), np.float32)
        else:
            if raw is None:
                raise ValueError("Full F boundary requires aligned original PCA32")
            raw = _features(raw, 32, "Original PCA32")
            if len(raw) != len(teacher):
                raise ValueError("Teacher/raw rows must align")
            v = ((raw.astype(np.float64) - self.input_mean.numpy()) / self.input_scale.numpy()).astype(np.float32)
        return torch.from_numpy(np.concatenate((u, v), axis=1))

    @torch.no_grad()
    def release(self, teacher, *, raw=None):
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("Release extraction requires a frozen model")
        return self.mapper(self.standardize(teacher, raw=raw)).numpy().astype(np.float32)

    @torch.no_grad()
    def source_probabilities(self, teacher, *, raw=None):
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("Native predictions require a frozen model")
        h = self.mapper(self.standardize(teacher, raw=raw))
        result = {}
        for name, head in self.heads.items():
            positive = torch.sigmoid(head(h)).reshape(-1)
            result[name] = torch.stack((1. - positive, positive), 1).numpy()
        return result

    probabilities = source_probabilities

    def freeze(self):
        self.eval().requires_grad_(False)
        return self


def _components(model, x, source, teacher, adversaries, attributes, priors):
    h = model.mapper(x)
    source_loss, individual, support = old.masked_source_bce({name: head(h) for name, head in model.heads.items()}, source)
    losses = {"source": source_loss, "teacher": old.preservation_loss(h, teacher, model.input_scale[:16])}
    if len(adversaries):
        losses["protection"] = old.protection_loss(adversaries, h, attributes, priors)[0]
    return losses, individual, support


def mapper_update(model, observers, optimizer, x, source, teacher, attributes, priors, *, beta, protected):
    optimizer.zero_grad(set_to_none=True)
    observers.zero_grad(set_to_none=True)
    flags = [p.requires_grad for p in observers.parameters()]
    observers.requires_grad_(False)
    try:
        # No decoder call or reconstruction graph exists in this study.
        h = model.mapper(x)
        objective, _, support = old.masked_source_bce({name: head(h) for name, head in model.heads.items()}, source)
        if beta:
            objective = objective + beta * old.preservation_loss(h, teacher, model.input_scale[:16])
        if protected:
            objective = objective - .1 * old.protection_loss(observers, h, attributes, priors)[0]
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite restricted-input objective")
        objective.backward()
        optimizer.step()
    finally:
        for parameter, flag in zip(observers.parameters(), flags):
            parameter.requires_grad_(flag)
    return support


def gradient_diagnostics(model, observers, x, source, teacher, attributes, priors, *, beta, protected):
    before = old.tree_digest((model.state_dict(), observers.state_dict(), [p.grad for p in model.parameters()], [p.grad for p in observers.parameters()]))
    rng = torch.get_rng_state().clone()
    losses, _, _ = _components(model, x, source, teacher, observers, attributes, priors)
    parameters = list(model.mapper.parameters())
    coefficients = {"source": 1., "teacher": beta, "protection": -.1 if protected else 0.}
    result = {"coefficients": coefficients, "reconstruction_coefficient": 0., "decoder_objective_path": "omitted",
              "source_loss": float(losses["source"].detach()), "preservation_loss": float(losses["teacher"].detach()),
              "normalized_adversary_ce": float(losses["protection"].detach()) if "protection" in losses else None,
              "raw_protection_definition": "positive mean prior-entropy-normalized adversary CE",
              "protection_available": "protection" in losses,
              "input_teacher_l2": float(torch.linalg.vector_norm(x[:, :16])),
              "input_raw_l2": float(torch.linalg.vector_norm(x[:, 16:])),
              "teacher_input_weight_l2": float(torch.linalg.vector_norm(model.mapper[0].weight[:, :16]).detach()),
              "raw_input_weight_l2": float(torch.linalg.vector_norm(model.mapper[0].weight[:, 16:]).detach())}
    vectors = {}
    for name, value in losses.items():
        grads = torch.autograd.grad(value, parameters, retain_graph=True)
        vectors[name] = torch.cat([g.reshape(-1) for g in grads])
        for convention, coefficient in (("raw", 1.), ("applied", coefficients[name])):
            result[f"{name}_{convention}_mapper_l2"] = float(torch.linalg.vector_norm(coefficient * vectors[name]))
            result[f"{name}_{convention}_teacher_input_weight_l2"] = float(torch.linalg.vector_norm(coefficient * grads[0][:, :16]))
            result[f"{name}_{convention}_raw_input_weight_l2"] = float(torch.linalg.vector_norm(coefficient * grads[0][:, 16:]))
    for name in ("source", "teacher", "protection"):
        if name not in losses:
            for convention in ("raw", "applied"):
                for block in ("mapper", "teacher_input_weight", "raw_input_weight"):
                    result[f"{name}_{convention}_{block}_l2"] = None
    for left, right in (("teacher", "protection"), ("source", "protection"), ("source", "teacher")):
        for convention in ("raw", "applied"):
            if left not in vectors or right not in vectors:
                dot, cosine = None, None
            else:
                a, b = vectors[left], vectors[right]
                if convention == "applied":
                    a, b = coefficients[left] * a, coefficients[right] * b
                dot = float(torch.dot(a, b))
                denominator = float(torch.linalg.vector_norm(a)) * float(torch.linalg.vector_norm(b))
                cosine = dot / denominator if denominator else None
            result[f"{left}_{right}_{convention}_dot"] = dot
            result[f"{left}_{right}_{convention}_cosine"] = cosine
    outside = list(model.heads.parameters()) + list(model.decoder.parameters()) + list(observers.parameters())
    assert all(g is None for g in torch.autograd.grad(losses["teacher"], outside, allow_unused=True))
    after = old.tree_digest((model.state_dict(), observers.state_dict(), [p.grad for p in model.parameters()], [p.grad for p in observers.parameters()]))
    assert before == after and torch.equal(rng, torch.get_rng_state())
    assert all(np.isfinite(value) for key, value in result.items() if key.endswith("_l2") and value is not None)
    result.update(training_state_unchanged=True, torch_rng_unchanged=True, optimizer_steps=0,
                  preservation_nonmapper_gradients_all_absent=True,
                  cosine_zero_policy="None when either norm is zero or component unavailable")
    return result


@torch.no_grad()
def _curve(model, observers, x, source, teacher, attributes, priors, xv, validation, epoch, counts, *, beta, protected):
    losses, tasks, support = _components(model, x, source, teacher, observers, attributes, priors)
    h = model.mapper(xv)
    val, individual, valid = old.masked_source_bce({name: head(h) for name, head in model.heads.items()}, validation)
    protection = float(losses["protection"]) if "protection" in losses else None
    source_value, preservation = float(losses["source"]), float(losses["teacher"])
    return {"epoch": epoch, **counts, "fit_base_loss": source_value, "fit_source_ce": source_value,
            "fit_source_losses": {k: float(v) if support[k] else None for k, v in tasks.items()},
            "source_validation_ce": float(val), "source_validation_base_loss": float(val),
            "source_validation_losses": {k: float(v) if valid[k] else None for k, v in individual.items()},
            "fit_preservation_loss": preservation, "fit_normalized_adversary_ce": protection,
            "fit_reconstruction_mse": None, "source_coefficient": 1., "preservation_coefficient": beta,
            "reconstruction_coefficient": 0., "protection_coefficient": -.1 if protected else 0.,
            "fit_applied_preservation": beta * preservation, "fit_applied_protection": -.1 * protection if protected else 0.,
            "fit_objective": source_value + beta * preservation - (.1 * protection if protected else 0.),
            "selection": "none; fixed final iterate", "decoder_objective_path": "omitted"}


def _observers(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1290000 + 100 * int(seed))
        return nn.ModuleDict({name: _network(16, [64, 32], size) for name, size in ATTRIBUTE_SCHEMA.items()})


def _nondead_fixture(model):
    """Source gradient on disposable exact-u=0/nonzero-v standardized inputs."""
    before, rng = state_digest(model.state_dict()), torch.get_rng_state().clone()
    disposable = copy.deepcopy(model)
    x = torch.zeros(3, 48)
    x[:, 16:] = torch.linspace(.25, 1.25, 32)[None, :]
    source = {name: torch.zeros(3, dtype=torch.int64) for name in SOURCE_SCHEMA}
    h = disposable.mapper(x)
    loss = old.masked_source_bce({name: head(h) for name, head in disposable.heads.items()}, source)[0]
    first, readout = torch.autograd.grad(loss, (disposable.mapper[0].weight, disposable.mapper[2].weight))
    per_column = torch.linalg.vector_norm(first[:, 16:], dim=0)
    assert bool((per_column > 0).all()) and float(torch.linalg.vector_norm(readout[:, 32:])) > 0.
    assert state_digest(model.state_dict()) == before and torch.equal(rng, torch.get_rng_state())
    return {"disposable_model": True, "optimizer_steps": 0, "actual_model_unchanged": True,
            "torch_rng_unchanged": True, "fixture": "standardized u exactly zero, all32 v coordinates nonzero; synthetic all-zero source labels",
            "all32_full_input_columns_nonzero_source_gradient": True,
            "raw_input_weight_column_gradient_l2": per_column.tolist(),
            "auxiliary_readout_source_gradient_l2": float(torch.linalg.vector_norm(readout[:, 32:]))}


def train_restricted_unit(teacher_fit, source_y, attr_y, teacher_val, source_val, seed, directory, *,
                          teacher_name, access, fitting_statistics, initial_state, orthogonal_q,
                          raw_fit=None, raw_val=None, miniature=False, fit_pool="representation_fit"):
    if access == "K" and (raw_fit is not None or raw_val is not None):
        raise ValueError("K trainer rejects all raw inputs")
    return _train(teacher_fit, source_y, attr_y, teacher_val, source_val, seed, directory,
                  teacher_name=teacher_name, access=access, fitting_statistics=fitting_statistics,
                  initial_state=initial_state, orthogonal_q=orthogonal_q, raw_fit=raw_fit, raw_val=raw_val,
                  miniature=miniature, fit_pool=fit_pool, bank=False)


def train_source_bank(teacher_fit, source_y, teacher_val, source_val, seed, directory, *,
                      teacher_name, fitting_statistics, initial_state, orthogonal_q,
                      miniature=False, fit_pool="representation_fit"):
    """K-only source bank: no raw-input, protected-label, or access argument."""
    return _train(teacher_fit, source_y, None, teacher_val, source_val, seed, directory,
                  teacher_name=teacher_name, access="K", fitting_statistics=fitting_statistics,
                  initial_state=initial_state, orthogonal_q=orthogonal_q, raw_fit=None, raw_val=None,
                  miniature=miniature, fit_pool=fit_pool, bank=True)


def _train(teacher_fit, source_y, attr_y, teacher_val, source_val, seed, directory, *, teacher_name, access,
           fitting_statistics, initial_state, orthogonal_q, raw_fit, raw_val, miniature, fit_pool, bank):
    if fit_pool != "representation_fit" or teacher_name not in ("E", "S") or access not in ("F", "K"):
        raise ValueError("Fixed representation_fit, E/S teacher and F/K access required")
    if not np.array_equal(np.asarray(orthogonal_q), generate_q(seed)):
        raise ValueError("Q must match the prespecified seed and sign-canonicalized QR exactly")
    teacher_fit, teacher_val = _features(teacher_fit, 16, "Fitting teacher"), _features(teacher_val, 16, "Validation teacher")
    source = old._labels(source_y, len(teacher_fit), SOURCE_SCHEMA, "source fitting")
    validation = old._labels(source_val, len(teacher_val), SOURCE_SCHEMA, "source validation")
    if any(not (y >= 0).any() for y in source.values()):
        raise ValueError("A source fitting task has no known labels")
    attributes = {} if bank else old._labels(attr_y, len(teacher_fit), ATTRIBUTE_SCHEMA, "attribute fitting")
    priors = {} if bank else old.prior_entropies(attributes)
    model = RestrictedModel(fitting_statistics, initial_state, orthogonal_q, access=access)
    x, xv = model.standardize(teacher_fit, raw=raw_fit), model.standardize(teacher_val, raw=raw_val)
    if access == "F":
        raw64 = raw_fit.astype(np.float64)
        std = raw64.std(0)
        if not np.array_equal(raw64.mean(0), model.input_mean.numpy()) or not np.array_equal(np.where(std > 1e-12, std, 1.), model.input_scale.numpy()):
            raise ValueError("Full fitting inputs differ from original saved preprocessing")
    teacher = torch.from_numpy(teacher_fit.copy()).detach()
    teacher_hash, initial_hash = array_digest(teacher.numpy()), state_digest(model.state_dict())
    snapshots = {"I": copy.deepcopy(model).freeze()}
    cfg = copy.deepcopy(CONFIG)
    cfg["preservation_beta"] = beta = 0. if bank else 1.
    if miniature:
        cfg.update(warm_base_epochs=1, warm_adversary_epochs=1, continuation_epochs=2, batch_size=16, curve_interval=1)
    schedules = {}
    for phase, epochs, offset in (("warm_base", cfg["warm_base_epochs"], 0), ("warm_adversary", cfg["warm_adversary_epochs"], 100), ("continuation", cfg["continuation_epochs"], 200)):
        schedule_seed = 1280000 + 100 * int(seed) + offset
        orders, digest = old._orders(len(x), epochs, schedule_seed)
        schedules[phase] = {"orders": orders, "seed": schedule_seed, "sha256": digest}
    fixed = schedules["warm_base"]["orders"][0][:cfg["batch_size"]]
    observers, observer_optimizer = nn.ModuleDict(), None
    optimizer = old._adam(model.parameters())
    count = {"mapper_optimizer_steps": 0, "adversary_optimizer_steps": 0}
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parity = {}
    for pool, inp, expected in (("representation_fit", x, teacher_fit), ("source_validation", xv, teacher_val)):
        with torch.no_grad():
            actual = model.mapper(inp).numpy()
        if not np.allclose(actual, expected, atol=1e-5, rtol=1e-5):
            raise AssertionError("Teacher identity parity failed")
        parity[pool] = {"passed": True, "max_absolute_error": float(np.max(np.abs(actual - expected))),
                        "atol": 1e-5, "rtol": 1e-5, "output_sha256": array_digest(actual), "teacher_sha256": array_digest(expected)}
    initialization = {"mode": "shifted teacher identity", "parity": parity, "q_sha256": array_digest(np.asarray(orthogonal_q)),
        "q_seed": 20260909 + int(seed), "Q_convention": "numpy float64 Gaussian QR; columns multiplied by sign(diag(R)), zero sign positive",
        "formula": "W1[:,:16]=[I;-I;Q;-Q],W1[:,16:]=0;b1=[1;-1;1;-1];W2=[diag(s16),-diag(s16),0,0];b2=m16-s16",
        "heads_initial_sha256": state_digest(model.heads.state_dict()), "decoder_initial_sha256": state_digest(model.decoder.state_dict()),
        "nondead_full_input_fixture": _nondead_fixture(model), "all_mapper_parameters_trainable": True}
    def diagnostic(current, actual_observers, protected):
        hypothetical = not bank and not len(actual_observers)
        obs = _observers(seed) if hypothetical else actual_observers
        result = gradient_diagnostics(current, obs, x[fixed], old._batch(source, fixed), teacher[fixed],
            old._batch(attributes, fixed) if attributes else {}, priors, beta=beta, protected=protected)
        result.update(hypothetical_preobserver_warmup=hypothetical,
            protection_observer_basis="not constructed for source-only bank" if bank else ("disposable original initialization" if hypothetical else "actual current observers"),
            diagnostic_observer_state_sha256=state_digest(obs.state_dict()) if len(obs) else None)
        if access == "K":
            assert result["input_raw_l2"] == result["raw_input_weight_l2"] == 0.
            assert all(value == 0. for key, value in result.items() if key.endswith("raw_input_weight_l2") and value is not None)
        return result
    diagnostics = {"initialization": diagnostic(model, observers, False)}
    old._checkpoint(out / "initialization.pt", model, observers, optimizer, observer_optimizer, {**count, "epoch": 0}, "true teacher initialization")
    shared_curves = {}
    for phase in ("warm_base", "warm_adversary"):
        if phase == "warm_adversary":
            if bank:
                break
            observers = _observers(seed)
            observer_optimizer = old._adam(observers.parameters())
            adversary_initial_hash = state_digest(observers.state_dict())
        before_phase = old.tree_digest((model.state_dict(), optimizer.state_dict()))
        phase_beta = beta if phase == "warm_base" else 0.
        shared_curves[phase] = [_curve(model, observers, x, source, teacher, attributes, priors, xv, validation, 0, count, beta=phase_beta, protected=False)]
        for epoch, order in enumerate(schedules[phase]["orders"], 1):
            for start in range(0, len(order), cfg["batch_size"]):
                idx = order[start:start + cfg["batch_size"]]
                if phase == "warm_base":
                    mapper_update(model, observers, optimizer, x[idx], old._batch(source, idx), teacher[idx],
                        old._batch(attributes, idx) if attributes else {}, priors, beta=beta, protected=False)
                    count["mapper_optimizer_steps"] += 1
                else:
                    old.adversary_update(model, observers, observer_optimizer, x[idx], old._batch(attributes, idx))
                    count["adversary_optimizer_steps"] += 1
            if epoch % cfg["curve_interval"] == 0 or epoch == len(schedules[phase]["orders"]):
                shared_curves[phase].append(_curve(model, observers, x, source, teacher, attributes, priors, xv, validation, epoch, count, beta=phase_beta, protected=False))
        if phase == "warm_adversary":
            assert before_phase == old.tree_digest((model.state_dict(), optimizer.state_dict()))
        old._checkpoint(out / (phase + ".pt"), model, observers, optimizer, observer_optimizer, {**count, "epoch": epoch}, phase)
        if phase == "warm_base":
            snapshots["W"] = copy.deepcopy(model).freeze()
            diagnostics["after_base_warmup"] = diagnostic(model, observers, False)
    common = {"model": state_digest(model.state_dict()), "adversaries": state_digest(observers.state_dict()),
              "mapper_optimizer": old.tree_digest(optimizer.state_dict()),
              "adversary_optimizer": old.tree_digest(observer_optimizer.state_dict()) if observer_optimizer else None}
    diagnostics["shared_fork"] = diagnostic(model, observers, not bank)
    arms, arm_metadata = {}, {}
    batches = math.ceil(len(x) / cfg["batch_size"])
    for name in (("B",) if bank else ("C", "D")):
        protected = name == "D"
        arm, attackers = copy.deepcopy(model), copy.deepcopy(observers)
        opt = old._adam(arm.parameters())
        opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        advopt = None if bank else old._adam(attackers.parameters())
        if advopt:
            advopt.load_state_dict(copy.deepcopy(observer_optimizer.state_dict()))
        fork = {"model": state_digest(arm.state_dict()), "adversaries": state_digest(attackers.state_dict()),
                "mapper_optimizer": old.tree_digest(opt.state_dict()), "adversary_optimizer": old.tree_digest(advopt.state_dict()) if advopt else None}
        assert fork == common
        arm_dir = out / name
        arm_dir.mkdir()
        counts = count.copy()
        old._checkpoint(arm_dir / "fork.pt", arm, attackers, opt, advopt, {**counts, "continuation_epoch": 0}, "shared warmed fork")
        fork_diag = diagnostic(arm, attackers, protected)
        curve = [_curve(arm, attackers, x, source, teacher, attributes, priors, xv, validation, 0, counts, beta=beta, protected=protected)]
        for epoch, order in enumerate(schedules["continuation"]["orders"], 1):
            for start in range(0, len(order), cfg["batch_size"]):
                idx = order[start:start + cfg["batch_size"]]
                a, y = (old._batch(attributes, idx) if attributes else {}), old._batch(source, idx)
                if not bank:
                    for _ in range(cfg["adversary_updates_per_mapper_step"]):
                        old.adversary_update(arm, attackers, advopt, x[idx], a)
                        counts["adversary_optimizer_steps"] += 1
                mapper_update(arm, attackers, opt, x[idx], y, teacher[idx], a, priors, beta=beta, protected=protected)
                counts["mapper_optimizer_steps"] += 1
            if epoch % cfg["curve_interval"] == 0 or epoch == cfg["continuation_epochs"]:
                curve.append(_curve(arm, attackers, x, source, teacher, attributes, priors, xv, validation, epoch, counts, beta=beta, protected=protected))
        old._checkpoint(arm_dir / "final.pt", arm, attackers, opt, advopt, {**counts, "continuation_epoch": epoch}, "fixed final iterate")
        mapper_steps = cfg["continuation_epochs"] * batches
        adv_steps = 0 if bank else 3 * mapper_steps
        assert counts == {"mapper_optimizer_steps": count["mapper_optimizer_steps"] + mapper_steps,
                          "adversary_optimizer_steps": count["adversary_optimizer_steps"] + adv_steps}
        assert all(p.grad is None for p in arm.decoder.parameters())
        assert state_digest(arm.decoder.state_dict()) == initialization["decoder_initial_sha256"]
        assert all(i not in opt.state_dict()["state"] for i in (10, 11, 12, 13))
        if access == "K":
            assert torch.count_nonzero(arm.mapper[0].weight[:, 16:]) == 0
            for moment in ("exp_avg", "exp_avg_sq"):
                assert torch.count_nonzero(opt.state_dict()["state"][0][moment][:, 16:]) == 0
        arm_metadata[name] = {"protected": protected, "protection_coefficient": -.1 if protected else 0.,
            "fork_hashes": fork, "final_model_hash": state_digest(arm.state_dict()), "final_adversary_hash": state_digest(attackers.state_dict()),
            "final_mapper_optimizer_hash": old.tree_digest(opt.state_dict()), "final_adversary_optimizer_hash": old.tree_digest(advopt.state_dict()) if advopt else None,
            "schedule_hash": schedules["continuation"]["sha256"], "selected_epoch": epoch,
            "continuation_mapper_optimizer_steps": mapper_steps, "continuation_adversary_optimizer_steps": adv_steps,
            "optimizer_counts_including_common": counts, "mapper_row_exposures": len(x) * cfg["continuation_epochs"],
            "adversary_row_exposures": 0 if bank else len(x) * cfg["continuation_epochs"] * 3,
            "mapper_loss_uses_protection_gradient": protected, "preservation_schedule": "off" if bank else "persistent",
            "continuation_preservation_coefficient": beta, "curve": curve,
            "fixed_batch_gradient_diagnostics_at_shared_fork": fork_diag,
            "fixed_batch_gradient_diagnostics_at_final": diagnostic(arm, attackers, protected),
            "decoder_unchanged": True, "decoder_gradients_absent": True, "decoder_optimizer_state_entries": 0,
            "decoder_optimizer_parameter_indices": [10, 11, 12, 13], "raw_columns_exact_zero": bool(torch.count_nonzero(arm.mapper[0].weight[:, 16:]) == 0),
            "native_source_heads_sha256": state_digest(arm.heads.state_dict())}
        arms[name] = {"model": arm.freeze(), "adversaries": attackers.eval().requires_grad_(False)}
    assert teacher_hash == array_digest(teacher.numpy()) and not teacher.requires_grad
    metadata = {"config": cfg, "miniature": miniature, "seed": int(seed), "teacher_name": teacher_name, "access": access,
        "bank": bank, "fit_pool": fit_pool, "source_label_keys": list(SOURCE_SCHEMA), "attribute_schema": {} if bank else ATTRIBUTE_SCHEMA,
        "reserved_labels_received": False, "final_evaluation_received": False, "raw_inputs_received": access == "F",
        "source_validation_role": "diagnostic only; no checkpoint, stopping or weight selection",
        "source_fit_coverage": old._coverage(source, SOURCE_SCHEMA), "attribute_fit_coverage": {} if bank else old._coverage(attributes, ATTRIBUTE_SCHEMA),
        "source_validation_coverage": old._coverage(validation, SOURCE_SCHEMA), "prior_entropies": priors,
        "fit_input_sha256": array_digest(teacher_fit), "source_validation_input_sha256": array_digest(teacher_val),
        "raw_fit_input_sha256": array_digest(raw_fit) if access == "F" else None,
        "raw_source_validation_input_sha256": array_digest(raw_val) if access == "F" else None,
        "standardized_fit_sha256": array_digest(x.numpy()), "standardized_validation_sha256": array_digest(xv.numpy()),
        "source_label_hashes": {k: array_digest(v.numpy()) for k, v in source.items()},
        "attribute_label_hashes": {k: array_digest(v.numpy()) for k, v in attributes.items()},
        "source_validation_label_hashes": {k: array_digest(v.numpy()) for k, v in validation.items()},
        "preprocessing": {"fit_pool": fit_pool, "fit_rows": len(x), "mean": model.input_mean.tolist(), "scale": model.input_scale.tolist(),
                          "teacher_statistics": "original first16 means/scales, never erased residual statistics", "fit_computation_dtype": "float64", "standardized_dtype": "float32"},
        "teacher": {"sha256": teacher_hash, "detached": True, "immutable_verified": True, "rows": len(teacher),
                    "coordinate_scale": model.input_scale[:16].tolist(), "coordinate_scale_sha256": array_digest(model.input_scale[:16].numpy())},
        "initialization": initialization, "initialization_hashes": {"model": initial_hash, "adversaries": None},
        "shared_fork_hashes": common, "adversary_initialization_hash": None if bank else adversary_initial_hash,
        "adversary_initialization_phase": "omitted for source-only bank" if bank else "after common base warmup",
        "common_optimizer_counts": count, "shared_curves": shared_curves,
        "schedules": {name: {k: v[k] for k in ("seed", "sha256")} for name, v in schedules.items()},
        "skipped_phases": ["warm_adversary", "continuation observer updates"] if bank else [],
        "common_base_row_exposures": len(x) * cfg["warm_base_epochs"],
        "common_adversary_row_exposures": 0 if bank else len(x) * cfg["warm_adversary_epochs"],
        "common_source_valid_label_exposures": {k: int((v >= 0).sum()) * cfg["warm_base_epochs"] for k, v in source.items()},
        "common_attribute_valid_label_exposures": {k: int((v >= 0).sum()) * cfg["warm_adversary_epochs"] for k, v in attributes.items()},
        "native_source_head_training_exposure": {"fit_pool": fit_pool, "mapper_epochs": cfg["warm_base_epochs"] + cfg["continuation_epochs"],
            "valid_label_exposures": {k: int((v >= 0).sum()) * (cfg["warm_base_epochs"] + cfg["continuation_epochs"]) for k, v in source.items()},
            "selection": "fixed final iterate; no head refit or calibration"},
        "snapshots": {name: {"checkpoint": "initialization.pt" if name == "I" else "warm_base.pt", "model_hash": state_digest(snapshot.state_dict()),
            "mapper_optimizer_steps": 0 if name == "I" else count["mapper_optimizer_steps"], "unchanged_across_adversary_warmup": True} for name, snapshot in snapshots.items()},
        "gradient_diagnostic_indices_sha256": array_digest(fixed), "stage_gradient_diagnostics": diagnostics,
        "objective": "mean masked three-source BCE + beta fixed-teacher preservation; D minus .1 mean prior-entropy-normalized real-attribute CE; decoder graph absent",
        "arms": arm_metadata, "runtime_seconds": time.perf_counter() - started,
        "module_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "historical_helper_source_sha256": hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest()}
    assert state_digest(snapshots["I"].state_dict()) == initial_hash
    assert state_digest(snapshots["W"].state_dict()) == common["model"]
    if not bank:
        assert all(diagnostics[name]["diagnostic_observer_state_sha256"] == adversary_initial_hash for name in ("initialization", "after_base_warmup"))
    with (out / "training.json").open("x") as handle:
        handle.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return {"arms": arms, "snapshots": snapshots, "metadata": metadata}
