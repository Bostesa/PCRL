"""Bounded full-row stochastic gate fit on a fixed public score bank.

This pure-array module has no Adult loader. The task-only capability has no
sensitive field; the protected objective requires an explicit protected input.
Numerical recipe defaults are engineering fixtures, not an effective fit lock.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

if __package__:
    from .channel import (GateWeights, JointPolicy, _entropy_margin,
                          expected_cost, bayes_recovery)
else:
    from channel import (GateWeights, JointPolicy, _entropy_margin,
                         expected_cost, bayes_recovery)


TASK_CAP_ORDER = ("r1_nll", "r1_brier", "r2_nll", "r2_brier")
PROTECTED_CAP_ORDER = (*TASK_CAP_ORDER, "r1_bayes", "r2_bayes")
TASK_CAP_KEYS = frozenset(TASK_CAP_ORDER)
PROTECTED_CAP_KEYS = frozenset(PROTECTED_CAP_ORDER)


def _fixed_array(value, *, dtype, ndim, name):
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype) or array.ndim != ndim:
        raise ValueError(f"{name} must be an exact {dtype} rank-{ndim} array")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TaskOnlyData:
    score1: np.ndarray
    score2: np.ndarray
    cell1: np.ndarray
    cell2: np.ndarray
    y1: np.ndarray
    y2: np.ndarray

    def __post_init__(self):
        for key in ("score1", "score2"):
            object.__setattr__(self, key, _fixed_array(getattr(self, key),
                              dtype="float64", ndim=2, name=key))
        for key in ("cell1", "cell2", "y1", "y2"):
            object.__setattr__(self, key, _fixed_array(getattr(self, key),
                              dtype="int64", ndim=1, name=key))
        n = len(self.score1)
        if (n < 1 or self.score2.shape[0] != n or
                any(len(getattr(self, key)) != n for key in
                    ("cell1", "cell2", "y1", "y2")) or
                not np.isfinite(self.score1).all() or
                not np.isfinite(self.score2).all()):
            raise ValueError("task-only rows are misaligned or nonfinite")


@dataclass(frozen=True)
class ProtectedData:
    task: TaskOnlyData
    sex: np.ndarray

    def __post_init__(self):
        if not isinstance(self.task, TaskOnlyData):
            raise TypeError("protected data requires a separate task-only capability")
        sensitive = _fixed_array(self.sex, dtype="int64", ndim=1, name="sex")
        if (len(sensitive) != len(self.task.score1) or
                not np.isin(sensitive, (0, 1)).all()):
            raise ValueError("binary SEX labels must align with task-only rows")
        object.__setattr__(self, "sex", sensitive)


@dataclass(frozen=True)
class FitRecipe:
    updates: int
    checkpoint_every: int
    cpu_seconds: float
    seed: int
    learning_rate: float = .01
    rho: float = 100.0
    dual_every: int = 25
    gradient_norm: float = 5.0
    init_scale: float = .01
    identity_logit_bias: float = 4.0

    def __post_init__(self):
        if (type(self.updates) is not int or self.updates < 1 or
                type(self.checkpoint_every) is not int or self.checkpoint_every < 1 or
                type(self.dual_every) is not int or self.dual_every < 1 or
                type(self.seed) is not int or self.seed < 0 or
                any(not np.isfinite(v) or v <= 0 for v in
                    (self.cpu_seconds, self.learning_rate, self.rho,
                     self.gradient_norm, self.init_scale,
                     self.identity_logit_bias))):
            raise ValueError("optimizer recipe must have explicit finite positive bounds")


@dataclass(frozen=True)
class PreparedLaw:
    features: np.ndarray
    valid: np.ndarray
    public_pair: np.ndarray
    code1: np.ndarray
    code2: np.ndarray
    r1_nll: np.ndarray
    r1_brier: np.ndarray
    r2_nll: np.ndarray
    r2_brier: np.ndarray
    z1: int
    z2: int


def prepare_law(data: TaskOnlyData, policy: JointPolicy) -> PreparedLaw:
    if not isinstance(data, TaskOnlyData) or not isinstance(policy, JointPolicy):
        raise TypeError("prepare_law needs task-only rows and a fixed joint policy")
    n = len(data.y1)
    z1, k1 = policy.bank1.probability.shape
    z2, k2 = policy.bank2.probability.shape
    if data.score1.shape != (n, k1) or data.score2.shape != (n, k2):
        raise ValueError("task-only score shape differs from frozen bank")
    features = np.zeros((n, k1 + k2 + 4), dtype=np.float64)
    valid = np.zeros((n, 16), dtype=bool)
    code1 = np.zeros((n, 16), dtype=np.int64)
    code2 = np.zeros((n, 16), dtype=np.int64)
    costs = {key: np.zeros((n, 16), dtype=np.float64) for key in TASK_CAP_ORDER}
    for row in range(n):
        a, b = data.score1[row], data.score2[row]
        c1, c2 = int(data.cell1[row]), int(data.cell2[row])
        # This validates score probability, fine-cell range and source class.
        policy._pairs_and_probabilities(a, b, c1, c2)
        if (data.y1[row] < 0 or data.y1[row] >= k1 or
                data.y2[row] < 0 or data.y2[row] >= k2):
            raise ValueError("task label is outside the frozen class order")
        features[row] = np.r_[a, b, _entropy_margin(a), _entropy_margin(b)]
        for ai, i in enumerate(policy.actions1[c1]):
            for bj, j in enumerate(policy.actions2[c2]):
                slot = 4 * ai + bj
                valid[row, slot] = True
                code1[row, slot], code2[row, slot] = i, j
                p1, p2 = policy.bank1.probability[i], policy.bank2.probability[j]
                costs["r1_nll"][row, slot] = -np.log(max(p1[data.y1[row]], 1e-12))
                costs["r2_nll"][row, slot] = -np.log(max(p2[data.y2[row]], 1e-12))
                target1, target2 = np.eye(k1)[data.y1[row]], np.eye(k2)[data.y2[row]]
                costs["r1_brier"][row, slot] = np.sum((p1 - target1) ** 2)
                costs["r2_brier"][row, slot] = np.sum((p2 - target2) ** 2)
    result = PreparedLaw(features, valid, code1 * z2 + code2, code1, code2,
                         *(costs[key] for key in
                           ("r1_nll", "r1_brier", "r2_nll", "r2_brier")), z1, z2)
    return result


def _probabilities_numpy(prepared: PreparedLaw, weights: GateWeights) -> np.ndarray:
    logits = np.tanh(prepared.features @ weights.w1 + weights.b1) @ weights.w2 + weights.b2
    logits = np.where(prepared.valid, logits, -np.inf)
    logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return logits / logits.sum(axis=1, keepdims=True)


def _bayes_numpy(prepared: PreparedLaw, q: np.ndarray, sex: np.ndarray):
    n = len(q)
    result = {}
    for name, ids, width in (("r1", prepared.code1, prepared.z1),
                             ("r2", prepared.code2, prepared.z2),
                             ("pair", prepared.public_pair, prepared.z1 * prepared.z2)):
        counts = np.zeros((2, width), dtype=np.float64)
        for sensitive in (0, 1):
            selected = sex == sensitive
            np.add.at(counts[sensitive], ids[selected], q[selected] / n)
        result[f"{name}_bayes"] = float(np.maximum(counts[0], counts[1]).sum())
    return result


def _task_rate_numpy(prepared: PreparedLaw, q: np.ndarray) -> float:
    public = np.zeros(prepared.z1 * prepared.z2, dtype=np.float64)
    np.add.at(public, prepared.public_pair, q / len(q))
    positive = public > 0
    marginal_entropy = -np.sum(public[positive] * np.log(public[positive]))
    positive_row = q > 0
    conditional_entropy = -np.sum(q[positive_row] * np.log(q[positive_row])) / len(q)
    return float(marginal_entropy - conditional_entropy)


def replay_prepared(prepared: PreparedLaw, weights: GateWeights,
                    *, sex: np.ndarray | None = None) -> dict[str, float]:
    q = _probabilities_numpy(prepared, weights)
    result = {key: float(np.mean(np.sum(q * getattr(prepared, key), axis=1)))
              for key in TASK_CAP_ORDER}
    result["task_rate"] = _task_rate_numpy(prepared, q)
    if sex is not None:
        result.update(_bayes_numpy(prepared, q, sex))
        result["protected_objective"] = (result["pair_bayes"] +
                                          .5 * (result["r1_bayes"] +
                                                result["r2_bayes"]))
    result["max_law_residual"] = float(np.max(abs(q.sum(axis=1) - 1)))
    return result


def exact_metrics(data: TaskOnlyData | ProtectedData, policy: JointPolicy) -> dict[str, float]:
    """Replay the policy's ACTUAL public law, including deterministic subclasses."""
    task = data.task if isinstance(data, ProtectedData) else data
    if not isinstance(task, TaskOnlyData) or not isinstance(policy, JointPolicy):
        raise TypeError("exact replay needs a task capability and public policy")
    n = len(task.y1)
    totals = {key: 0.0 for key in TASK_CAP_ORDER}
    public = {}
    conditional_entropy = 0.0
    rows = []
    residual = 0.0
    for index in range(n):
        law = tuple(policy.sparse_law(task.score1[index], task.score2[index],
                                      int(task.cell1[index]), int(task.cell2[index])))
        rows.append(law)
        residual = max(residual, abs(sum(mass for _, _, mass in law) - 1))
        cost = expected_cost(policy.bank1, policy.bank2, law,
                             y1=int(task.y1[index]), y2=int(task.y2[index]))
        for recipient in (1, 2):
            for metric in ("nll", "brier"):
                totals[f"r{recipient}_{metric}"] += cost[f"r{recipient}"][metric] / n
        for i, j, mass in law:
            public[i, j] = public.get((i, j), 0.0) + mass / n
            if mass > 0:
                conditional_entropy -= mass * np.log(mass) / n
    marginal_entropy = -sum(mass * np.log(mass)
                            for mass in public.values() if mass > 0)
    totals["task_rate"] = float(marginal_entropy - conditional_entropy)
    totals["max_law_residual"] = float(residual)
    if isinstance(data, ProtectedData):
        recovered = bayes_recovery(rows, data.sex)
        totals.update({f"{key}_bayes": value for key, value in recovered.items()})
        totals["protected_objective"] = (totals["pair_bayes"] +
                                         .5 * (totals["r1_bayes"] +
                                               totals["r2_bayes"]))
    return totals


def _identity_guards(prepared: PreparedLaw, sex: np.ndarray):
    q = np.zeros_like(prepared.r1_nll)
    q[:, 0] = 1.0
    return _bayes_numpy(prepared, q, sex)


def _torch_metrics(prepared: PreparedLaw, q, *, sex, torch):
    metrics = {key: (q * torch.as_tensor(getattr(prepared, key))).sum(dim=1).mean()
               for key in TASK_CAP_ORDER}
    n = q.shape[0]
    if sex is None:
        flat = torch.zeros(prepared.z1 * prepared.z2, dtype=torch.float64)
        flat = flat.scatter_add(0, torch.as_tensor(prepared.public_pair.ravel()),
                                (q / n).reshape(-1))
        marginal = -(flat * torch.log(torch.clamp(flat, min=1e-300))).sum()
        conditional = -(q * torch.log(torch.clamp(q, min=1e-300))).sum(dim=1).mean()
        metrics["objective"] = marginal - conditional
    else:
        indicator = torch.as_tensor(sex.copy(), dtype=torch.int64)
        for name, ids, width in (("r1", prepared.code1, prepared.z1),
                                 ("r2", prepared.code2, prepared.z2),
                                 ("pair", prepared.public_pair, prepared.z1 * prepared.z2)):
            counts = []
            ids_tensor = torch.as_tensor(ids.ravel().copy())
            for sensitive in (0, 1):
                weighted = q * (indicator == sensitive).to(torch.float64)[:, None] / n
                counts.append(torch.zeros(width, dtype=torch.float64).scatter_add(
                    0, ids_tensor, weighted.reshape(-1)))
            metrics[f"{name}_bayes"] = torch.maximum(*counts).sum()
        metrics["objective"] = (metrics["pair_bayes"] +
                                .5 * (metrics["r1_bayes"] + metrics["r2_bayes"]))
    return metrics


def fit_gate(data: TaskOnlyData | ProtectedData, policy: JointPolicy, *,
             objective_kind: str, caps: dict[str, float], recipe: FitRecipe,
             checkpoint_sink=None) -> dict:
    """Two-start bounded heuristic; only exactly replay-feasible checkpoints select."""
    if objective_kind == "task_rate":
        if type(data) is not TaskOnlyData:
            raise TypeError("task-only fit cannot receive a protected data capability")
        required, sex, task = TASK_CAP_ORDER, None, data
    elif objective_kind == "protected_bayes":
        if type(data) is not ProtectedData:
            raise TypeError("protected fit requires a protected data capability")
        required, sex, task = PROTECTED_CAP_ORDER, data.sex, data.task
    else:
        raise ValueError("unknown registered stochastic gate objective")
    if (not isinstance(policy, JointPolicy) or not isinstance(recipe, FitRecipe) or
            type(caps) is not dict or caps.keys() != TASK_CAP_KEYS or
            any(not np.isfinite(v) for v in caps.values())):
        raise ValueError("fit needs one exact cap for each task loss")
    start_cpu = time.process_time()
    prepared = prepare_law(task, policy)
    identity = _identity_guards(prepared, sex) if sex is not None else {}
    # Local guards are exact fitted-law identity guards, not caller-relaxed caps.
    if sex is not None:
        caps = {**caps, "r1_bayes": identity["r1_bayes"],
                "r2_bayes": identity["r2_bayes"]}
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("frozen fit environment needs PyTorch") from error
    torch.set_num_threads(1)
    feature = torch.as_tensor(prepared.features.copy())
    valid = torch.as_tensor(prepared.valid.copy())
    starts = []
    best = None
    interrupted = False
    for start_index, start_name in enumerate(("identity_biased", "uniform_logit")):
        if time.process_time() - start_cpu >= recipe.cpu_seconds:
            interrupted = True
            break
        rng = np.random.default_rng(recipe.seed + start_index)
        initial = [rng.normal(0, recipe.init_scale, size=(feature.shape[1], 16)),
                   rng.normal(0, recipe.init_scale, size=16),
                   np.zeros((16, 16)), np.zeros(16)]
        if start_index == 0:
            initial[3][0] += recipe.identity_logit_bias
        weights = [torch.nn.Parameter(torch.as_tensor(v.copy(), dtype=torch.float64))
                   for v in initial]
        optimizer = torch.optim.Adam(weights, lr=recipe.learning_rate)
        dual = {key: 0.0 for key in required}
        record = {"name": start_name, "checkpoints": [], "completed": False}

        def checkpoint(update):
            nonlocal best
            frozen = GateWeights(*(v.detach().numpy().copy() for v in weights))
            metrics = replay_prepared(prepared, frozen, sex=sex)
            objective = (metrics["task_rate"] if sex is None else
                         metrics["protected_objective"])
            residual = {key: metrics[key] - caps[key] for key in required}
            feasible = all(value <= 0 for value in residual.values())
            entry = {"update": update, "metrics": metrics,
                     "residual": residual, "feasible": feasible,
                     "objective": objective, "weights": frozen}
            record["checkpoints"].append(entry)
            if checkpoint_sink is not None:
                checkpoint_sink(start_name, entry)
            if feasible and (best is None or (objective, start_index, update) < best[0]):
                best = ((objective, start_index, update), frozen, metrics)
            return metrics

        checkpoint(0)
        for update in range(1, recipe.updates + 1):
            if time.process_time() - start_cpu >= recipe.cpu_seconds:
                interrupted = True
                break
            optimizer.zero_grad()
            logits = torch.tanh(feature @ weights[0] + weights[1]) @ weights[2] + weights[3]
            q = torch.softmax(logits.masked_fill(~valid, -torch.inf), dim=1)
            values = _torch_metrics(prepared, q, sex=sex, torch=torch)
            objective = values["objective"]
            penalty = objective
            for key in required:
                residual = values[key] - caps[key]
                positive = torch.clamp(residual, min=0)
                penalty = penalty + dual[key] * residual + .5 * recipe.rho * positive.square()
            penalty.backward()
            torch.nn.utils.clip_grad_norm_(weights, recipe.gradient_norm)
            optimizer.step()
            if update % recipe.dual_every == 0:
                for key in required:
                    dual[key] = max(0.0, dual[key] + recipe.rho *
                                    (float(values[key].detach()) - caps[key]))
            if update % recipe.checkpoint_every == 0 or update == recipe.updates:
                checkpoint(update)
        if time.process_time() - start_cpu >= recipe.cpu_seconds:
            interrupted = True
        record["completed"] = not interrupted
        record["elapsed_cpu_seconds"] = time.process_time() - start_cpu
        starts.append(record)
        if interrupted:
            break
    if time.process_time() - start_cpu >= recipe.cpu_seconds:
        interrupted = True
    complete = not interrupted and len(starts) == 2 and all(s["completed"] for s in starts)
    # An incumbent from an unfinished search is diagnostic, never selected.
    return {"complete": complete, "success": bool(complete and best is not None),
            "starts": starts, "policy": (JointPolicy(policy.bank1, policy.bank2,
                policy.actions1, policy.actions2, best[1]) if complete and best else None),
            "metrics": best[2] if complete and best else None,
            "diagnostic_best": None if best is None else
                {"objective": best[0][0], "start": best[0][1], "update": best[0][2]},
            "elapsed_cpu_seconds": time.process_time() - start_cpu}
