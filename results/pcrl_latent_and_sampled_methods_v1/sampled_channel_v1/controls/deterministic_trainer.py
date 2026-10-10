"""Bounded, separately trained hard-output gates for deterministic A controls.

Inputs are already admitted FIT rows. Exact public-law replay decides eligibility.
"""

from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class DeterministicBatch:
    features: np.ndarray
    pair_ids: np.ndarray
    valid: np.ndarray
    costs: np.ndarray
    identity_slot: np.ndarray
    bank_sizes: tuple[int, int]
    sex: np.ndarray | None = None

    def __post_init__(self):
        x, p, v = map(np.asarray, (self.features, self.pair_ids, self.valid))
        c, identity = map(np.asarray, (self.costs, self.identity_slot))
        n = len(x)
        if (n < 1 or x.dtype != np.float64 or x.shape != (n, 12) or
                p.dtype != np.int64 or p.shape != (n, 16, 2) or
                v.dtype != np.bool_ or v.shape != (n, 16) or
                c.dtype != np.float64 or c.shape != (n, 16, 4) or
                identity.dtype != np.int64 or identity.shape != (n,) or
                not np.isfinite(x).all() or not np.isfinite(c).all() or
                len(self.bank_sizes) != 2 or min(self.bank_sizes) < 1 or
                np.any(identity < 0) or np.any(identity >= 16) or
                not v.any(axis=1).all() or not v[np.arange(n), identity].all()):
            raise ValueError("invalid fixed-action FIT batch")
        disclosed = p[v]
        if (np.any(disclosed < 0) or np.any(disclosed[:, 0] >= self.bank_sizes[0]) or
                np.any(disclosed[:, 1] >= self.bank_sizes[1]) or
                any(len(np.unique(p[row, v[row]], axis=0)) != v[row].sum()
                    for row in range(n))):
            raise ValueError("invalid or duplicated public actions")
        for name, arr in (("features", x), ("pair_ids", p), ("valid", v),
                          ("costs", c), ("identity_slot", identity)):
            frozen = np.ascontiguousarray(arr).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        if self.sex is not None:
            sex = np.asarray(self.sex)
            if sex.dtype != np.int64 or sex.shape != (n,) or not np.isin(sex, (0, 1)).all():
                raise ValueError("SEX must be aligned binary int64")
            frozen = np.ascontiguousarray(sex).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, "sex", frozen)


@dataclass(frozen=True)
class DeterministicConfig:
    mode: str
    steps: int
    learning_rate: float
    temperature_start: float
    temperature_end: float
    rho: float
    dual_every: int
    replay_every: int
    grad_clip: float
    cpu_limit_seconds: float
    seed: int
    bias_deltas: tuple[float, ...]

    def __post_init__(self):
        if (self.mode not in ("task", "private") or
                any(type(v) is not int or v < 1 for v in
                    (self.steps, self.dual_every, self.replay_every)) or
                type(self.seed) is not int or self.seed < 0 or
                any(not math.isfinite(v) or v <= 0 for v in
                    (self.learning_rate, self.temperature_start,
                     self.temperature_end, self.rho, self.grad_clip,
                     self.cpu_limit_seconds)) or
                tuple(self.bias_deltas) != (-0.5, -0.1, 0.1, 0.5)):
            raise ValueError("invalid fixed deterministic recipe")


@dataclass(frozen=True)
class SelectionMetrics:
    primary: float
    costs: tuple[float, float, float, float]
    feasible: bool
    occupied_pairs: int
    occupied_codes: tuple[int, int]
    recovery: tuple[float, float, float] | None = None


def _bayes(sex, disclosed):
    _, ids = np.unique(disclosed, return_inverse=True, axis=0)
    count = np.zeros((2, ids.max() + 1), dtype=float)
    np.add.at(count, (sex, ids), 1 / len(sex))
    return float(count.max(axis=0).sum())


def evaluate_selection(batch, selected, *, caps, mode, local_caps=None):
    """Replay exact deterministic task losses and empirical public-code Bayes recovery."""
    selected = np.asarray(selected)
    n = len(batch.features)
    if (selected.dtype != np.int64 or selected.shape != (n,) or
            np.any(selected < 0) or np.any(selected >= 16) or
            not batch.valid[np.arange(n), selected].all() or
            len(caps) != 4 or any(not math.isfinite(v) for v in caps)):
        raise ValueError("invalid hard selection or caps")
    public = batch.pair_ids[np.arange(n), selected]
    cost = tuple(map(float, batch.costs[np.arange(n), selected].mean(axis=0)))
    feasible = all(v <= cap for v, cap in zip(cost, caps))
    occupied = (len(np.unique(public[:, 0])), len(np.unique(public[:, 1])))
    pairs = len(np.unique(public, axis=0))
    if mode == "task":
        if batch.sex is not None or local_caps is not None:
            raise ValueError("task-only worker cannot receive SEX or SEX guards")
        _, count = np.unique(public, return_counts=True, axis=0)
        mass = count / n
        primary = float(-(mass * np.log(mass)).sum())
        recovery = None
    elif mode == "private":
        if batch.sex is None or local_caps is None or len(local_caps) != 2:
            raise ValueError("private worker requires SEX and identity guards")
        recovery = (_bayes(batch.sex, public[:, 0]),
                    _bayes(batch.sex, public[:, 1]), _bayes(batch.sex, public))
        primary = recovery[2] + 0.5 * (recovery[0] + recovery[1])
        feasible &= all(v <= cap for v, cap in zip(recovery[:2], local_caps))
    else:
        raise ValueError("unknown deterministic objective")
    return SelectionMetrics(primary, cost, feasible, pairs, occupied, recovery)


@dataclass(frozen=True)
class GateWeights:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def batch_from_prepared(prepared, *, identity_public_pair, sex=None):
    """Bind R12's fixed padded action table to source-checked identity codes.

    The caller must first bind the source row-order and action-map hashes to R01/R11.
    This adapter verifies the per-row identity code and all numeric table shapes.
    """
    identity = np.asarray(identity_public_pair)
    n = len(prepared.features)
    if identity.dtype != np.int64 or identity.shape != (n, 2):
        raise ValueError("canonical identity public IDs must align with FIT rows")
    pair = np.stack((prepared.code1, prepared.code2), axis=2).copy()
    if (prepared.features.shape != (n, 12) or pair.shape != (n, 16, 2) or
            prepared.valid.shape != (n, 16) or
            not prepared.valid[:, 0].all() or
            not np.array_equal(pair[:, 0], identity)):
        raise ValueError("prepared actions disagree with sealed identity map")
    pair[~prepared.valid] = -1
    costs = np.stack((prepared.r1_nll, prepared.r1_brier,
                      prepared.r2_nll, prepared.r2_brier), axis=2)
    return DeterministicBatch(np.asarray(prepared.features), pair,
                              np.asarray(prepared.valid), costs,
                              np.zeros(n, dtype=np.int64),
                              (prepared.z1, prepared.z2), sex)


def as_public_gate(weights: GateWeights, public_gate_type):
    """Construct R12's actual deployable gate type without changing its logits."""
    return public_gate_type(weights.w1, weights.b1, weights.w2, weights.b2)


@dataclass(frozen=True)
class StartReceipt:
    start: str
    optimizer_steps: int
    polish_trials: int
    replay_count: int
    completed: bool
    terminal_metrics: SelectionMetrics | None
    chosen_checkpoint_update: int | None


@dataclass(frozen=True)
class FitResult:
    complete: bool
    success: bool
    weights: GateWeights | None
    metrics: SelectionMetrics | None
    starts: tuple[StartReceipt, ...]
    cpu_seconds: float


def _selection(batch, weights):
    logits = np.tanh(batch.features @ weights.w1 + weights.b1) @ weights.w2 + weights.b2
    return np.argmax(np.where(batch.valid, logits, -np.inf), axis=1).astype(np.int64)


def _rank(m):
    return (m.primary, m.occupied_pairs, sum(m.occupied_codes), sum(m.costs))


def fit_deterministic(batch, config, *, caps, local_caps=None):
    """Two independent STE starts and a deployable 64-trial bias pass per start."""
    if not isinstance(batch, DeterministicBatch) or not isinstance(config, DeterministicConfig):
        raise TypeError("fixed batch and recipe required")
    if config.mode == "task" and (batch.sex is not None or local_caps is not None):
        raise ValueError("task-only fit cannot receive SEX")
    if config.mode == "private":
        if batch.sex is None:
            raise ValueError("protected fit requires SEX")
        identity = evaluate_selection(batch, batch.identity_slot, caps=caps,
                                      mode="private", local_caps=(1.0, 1.0))
        expected = identity.recovery[:2]
        if local_caps is None:
            local_caps = expected
        if any(abs(a - b) > 1e-12 for a, b in zip(local_caps, expected)):
            raise ValueError("local guards must equal exact identity recovery")
    evaluate_selection(batch, batch.identity_slot, caps=caps, mode=config.mode,
                       local_caps=local_caps)
    clock = time.process_time()
    import torch
    torch.set_num_threads(1)
    x, valid = torch.as_tensor(batch.features.copy()), torch.as_tensor(batch.valid.copy())
    public = batch.pair_ids.copy()
    public[~batch.valid] = 0
    id1 = torch.as_tensor(public[:, :, 0].reshape(-1))
    id2 = torch.as_tensor(public[:, :, 1].reshape(-1))
    id_pair = id1 * batch.bank_sizes[1] + id2
    cost_t = torch.as_tensor(batch.costs.copy())
    sex_t = None if batch.sex is None else torch.as_tensor(batch.sex.copy())
    receipts, candidates = [], []
    complete = True
    for start_index, start_name in enumerate(("identity_biased", "uniform")):
        if time.process_time() - clock >= config.cpu_limit_seconds:
            complete = False
            break
        rng = np.random.default_rng(config.seed + start_index)
        arrays = [rng.normal(0, .01, size=shape) for shape in
                  ((12, 16), (16,), (16, 16), (16,))]
        if start_index == 0:
            arrays[3][0] += 4.0
        params = [torch.nn.Parameter(torch.as_tensor(a, dtype=torch.float64))
                  for a in arrays]
        optim = torch.optim.Adam(params, lr=config.learning_rate)
        dual = np.zeros(6 if config.mode == "private" else 4)
        updates = 0
        replay_count = 1
        best_checkpoint = None

        def freeze():
            return GateWeights(*(p.detach().numpy().copy() for p in params))

        def bayes(q, ids, width):
            mass = []
            for s in (0, 1):
                weighted = q * (sex_t == s).to(torch.float64)[:, None] / len(x)
                mass.append(torch.zeros(width, dtype=torch.float64).scatter_add(
                    0, ids, weighted.reshape(-1)))
            return torch.maximum(mass[0], mass[1]).sum()

        initial_weights = freeze()
        initial_metrics = evaluate_selection(batch, _selection(batch, initial_weights),
                                             caps=caps, mode=config.mode,
                                             local_caps=local_caps)
        if initial_metrics.feasible:
            best_checkpoint = (0, initial_weights, initial_metrics)

        for step in range(1, config.steps + 1):
            if time.process_time() - clock >= config.cpu_limit_seconds:
                complete = False
                break
            optim.zero_grad()
            logits = torch.tanh(x @ params[0] + params[1]) @ params[2] + params[3]
            logits = logits.masked_fill(~valid, -torch.inf)
            hard = torch.nn.functional.one_hot(torch.argmax(logits, dim=1), 16).to(torch.float64)
            fraction = (step - 1) / max(1, config.steps - 1)
            temp = config.temperature_start * (config.temperature_end /
                    config.temperature_start) ** fraction
            soft = torch.softmax(logits / temp, dim=1)
            q = hard + soft - soft.detach()
            loss = (q[:, :, None] * cost_t).sum(dim=1).mean(dim=0)
            if config.mode == "task":
                marginal = torch.zeros(batch.bank_sizes[0] * batch.bank_sizes[1],
                                       dtype=torch.float64).scatter_add(
                    0, id_pair, (q / len(x)).reshape(-1))
                objective = -(marginal * torch.log(marginal.clamp_min(1e-12))).sum()
                residual = torch.stack([loss[i] - caps[i] for i in range(4)])
            else:
                g1 = bayes(q, id1, batch.bank_sizes[0])
                g2 = bayes(q, id2, batch.bank_sizes[1])
                gp = bayes(q, id_pair, batch.bank_sizes[0] * batch.bank_sizes[1])
                objective = gp + .5 * (g1 + g2)
                residual = torch.stack([*(loss[i] - caps[i] for i in range(4)),
                                        g1 - local_caps[0], g2 - local_caps[1]])
            positive = torch.clamp(residual, min=0)
            penalty = objective + (torch.as_tensor(dual) * residual).sum() + \
                .5 * config.rho * positive.square().sum()
            penalty.backward()
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optim.step()
            updates += 1
            if step % config.dual_every == 0:
                dual = np.maximum(0, dual + config.rho * residual.detach().numpy())
            if step % config.replay_every == 0 or step == config.steps:
                frozen = freeze()
                checked = evaluate_selection(batch, _selection(batch, frozen), caps=caps,
                                             mode=config.mode, local_caps=local_caps)
                if checked.feasible and (best_checkpoint is None or
                        (_rank(checked), step) < (_rank(best_checkpoint[2]),
                                                  best_checkpoint[0])):
                    best_checkpoint = (step, frozen, checked)
                replay_count += 1
        if not complete:
            receipts.append(StartReceipt(start_name, updates, 0, replay_count, False,
                                         None, None))
            break
        weights = best_checkpoint[1] if best_checkpoint is not None else freeze()
        trials = 0
        for slot in range(16):
            for delta in config.bias_deltas:
                if time.process_time() - clock >= config.cpu_limit_seconds:
                    complete = False
                    break
                candidate_bias = weights.b2.copy()
                candidate_bias[slot] += delta
                candidate = GateWeights(weights.w1, weights.b1, weights.w2, candidate_bias)
                current = evaluate_selection(batch, _selection(batch, weights), caps=caps,
                                             mode=config.mode, local_caps=local_caps)
                proposed = evaluate_selection(batch, _selection(batch, candidate), caps=caps,
                                              mode=config.mode, local_caps=local_caps)
                trials += 1
                if proposed.feasible and (not current.feasible or
                        proposed.primary < current.primary - 1e-10 or
                        (abs(proposed.primary - current.primary) <= 1e-10 and
                         _rank(proposed)[1:] < _rank(current)[1:])):
                    weights = candidate
            if not complete:
                break
        terminal = (evaluate_selection(batch, _selection(batch, weights), caps=caps,
                                       mode=config.mode, local_caps=local_caps)
                    if complete else None)
        receipts.append(StartReceipt(start_name, updates, trials, replay_count,
                                     complete, terminal,
                                     None if best_checkpoint is None else best_checkpoint[0]))
        if not complete:
            break
        if terminal.feasible:
            candidates.append((_rank(terminal), start_index, weights, terminal))
    complete &= len(receipts) == 2 and all(r.completed for r in receipts)
    winner = min(candidates, key=lambda item: (item[0], item[1])) if complete and candidates else None
    return FitResult(complete, bool(winner), None if winner is None else winner[2],
                     None if winner is None else winner[3], tuple(receipts),
                     time.process_time() - clock)
