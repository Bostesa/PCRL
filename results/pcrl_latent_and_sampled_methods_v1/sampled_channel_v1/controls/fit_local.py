"""Two independently fitted purpose-local factors with own-input capabilities."""

from dataclasses import dataclass
import time

import numpy as np

from bank import FixedBank
from channel import GateWeights, _entropy_margin
from fit import FitRecipe, TaskOnlyData, ProtectedData, exact_metrics
from factor_controls import FactorGate, LocalPolicy


@dataclass(frozen=True)
class LocalFactorData:
    score: np.ndarray
    cell: np.ndarray
    y: np.ndarray
    sex: np.ndarray
    bank: FixedBank
    actions: tuple[tuple[int, ...], ...]
    unit_ids: tuple[str, ...]

    def __post_init__(self):
        score, cell, y, sex = map(np.asarray,
                                  (self.score, self.cell, self.y, self.sex))
        n = len(score)
        if (n < 1 or not isinstance(self.bank, FixedBank) or
                score.dtype != np.float64 or score.ndim != 2 or
                score.shape[1] != self.bank.probability.shape[1] or
                any(a.dtype != np.int64 or a.shape != (n,) for a in
                    (cell, y, sex)) or
                len(self.unit_ids) != n or
                any(type(unit) is not str or not unit for unit in self.unit_ids) or
                not np.isfinite(score).all() or np.any(score < 0) or
                not np.allclose(score.sum(axis=1), 1, atol=1e-12, rtol=0) or
                not np.isin(sex, (0, 1)).all() or
                np.any(y < 0) or np.any(y >= score.shape[1]) or
                np.any(cell < 0) or np.any(cell >= len(self.actions))):
            raise ValueError("invalid own-purpose local FIT capability")
        if any(any(type(code) not in (int, np.int64) for code in row)
               for row in self.actions):
            raise ValueError("local action IDs must be exact integers")
        actions = tuple(tuple(int(code) for code in row) for row in self.actions)
        if (any(not 1 <= len(row) <= 4 or len(set(row)) != len(row) or
                any(code < 0 or code >= len(self.bank.probability) for code in row)
                for row in actions) or
                any(any(self.bank.decision[code] != self.bank.decision[row[0]]
                        for code in row) for row in actions) or
                any(int(np.argmax(score[i])) !=
                    int(self.bank.decision[actions[int(cell[i])][0]])
                    for i in range(n))):
            raise ValueError("local score class differs from frozen action map")
        same_unit = {}
        for i, unit in enumerate(self.unit_ids):
            source = (score[i].tobytes(), int(cell[i]), int(y[i]), int(sex[i]))
            if unit in same_unit and same_unit[unit] != source:
                raise ValueError("duplicate local unit has inconsistent input or labels")
            same_unit[unit] = source
        for name, value in (("score", score), ("cell", cell), ("y", y), ("sex", sex)):
            frozen = np.ascontiguousarray(value).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "unit_ids", tuple(self.unit_ids))


def _prepare(data):
    n, k = data.score.shape
    x = np.zeros((n, k + 2), dtype=np.float64)
    valid = np.zeros((n, 4), dtype=bool)
    ids = np.zeros((n, 4), dtype=np.int64)
    costs = np.zeros((n, 4, 2), dtype=np.float64)
    for row in range(n):
        x[row] = np.r_[data.score[row], _entropy_margin(data.score[row])]
        choices = data.actions[int(data.cell[row])]
        for slot, code in enumerate(choices):
            valid[row, slot], ids[row, slot] = True, code
            p = data.bank.probability[code]
            costs[row, slot, 0] = -np.log(max(p[data.y[row]], 1e-12))
            costs[row, slot, 1] = np.sum((p - np.eye(k)[data.y[row]]) ** 2)
    return x, valid, ids, costs


def _bayes(sex, ids, q, z):
    counts = np.zeros((2, z), dtype=np.float64)
    for sensitive in (0, 1):
        rows = sex == sensitive
        np.add.at(counts[sensitive], ids[rows], q[rows] / len(sex))
    return float(counts.max(axis=0).sum())


def _replay(data, prepared, factor, caps, identity_guard):
    x, valid, ids, costs = prepared
    logits = np.tanh(x @ factor.w1 + factor.b1) @ factor.w2 + factor.b2
    logits = np.where(valid, logits, -np.inf)
    q = np.exp(logits - logits.max(axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)
    loss = tuple(float(v) for v in (q[:, :, None] * costs).sum(axis=1).mean(axis=0))
    g = _bayes(data.sex, ids, q, len(data.bank.probability))
    return {"nll": loss[0], "brier": loss[1], "bayes": g,
            "feasible": loss[0] <= caps[0] and loss[1] <= caps[1] and
                        g <= identity_guard}


def _fit_one(data, *, caps, recipe, clock):
    import torch
    torch.set_num_threads(1)
    x, valid, ids, costs = prepared = _prepare(data)
    identity_q = np.zeros((len(x), 4), dtype=np.float64)
    identity_q[:, 0] = 1
    identity_guard = _bayes(data.sex, ids, identity_q, len(data.bank.probability))
    feature = torch.as_tensor(x)
    mask = torch.as_tensor(valid)
    ids_t = torch.as_tensor(ids.reshape(-1))
    costs_t = torch.as_tensor(costs)
    sex_t = torch.as_tensor(data.sex.copy())
    starts, best = [], None
    interrupted = False
    for start_index, start_name in enumerate(("identity_biased", "uniform_logit")):
        if time.process_time() - clock >= recipe.cpu_seconds:
            interrupted = True
            break
        rng = np.random.default_rng(recipe.seed + start_index)
        arrays = [rng.normal(0, recipe.init_scale, size=shape) for shape in
                  ((x.shape[1], 16), (16,), (16, 4), (4,))]
        if start_index == 0:
            arrays[3][0] += recipe.identity_logit_bias
        weights = [torch.nn.Parameter(torch.as_tensor(v, dtype=torch.float64))
                   for v in arrays]
        optim = torch.optim.Adam(weights, lr=recipe.learning_rate)
        dual = np.zeros(3)
        receipt = {"start": start_name, "checkpoints": [], "updates": 0,
                   "completed": False}

        def checkpoint(update):
            nonlocal best
            factor = FactorGate(*(w.detach().numpy().copy() for w in weights))
            metrics = _replay(data, prepared, factor, caps, identity_guard)
            receipt["checkpoints"].append({"update": update, **metrics})
            rank = (metrics["bayes"], start_index, update)
            if metrics["feasible"] and (best is None or rank < best[0]):
                best = (rank, factor, metrics)

        checkpoint(0)
        for update in range(1, recipe.updates + 1):
            if time.process_time() - clock >= recipe.cpu_seconds:
                interrupted = True
                break
            optim.zero_grad()
            logits = (torch.tanh(feature @ weights[0] + weights[1]) @
                      weights[2] + weights[3]).masked_fill(~mask, -torch.inf)
            q = torch.softmax(logits, dim=1)
            loss = (q[:, :, None] * costs_t).sum(dim=1).mean(dim=0)
            counts = []
            for sensitive in (0, 1):
                mass = q * (sex_t == sensitive).to(torch.float64)[:, None] / len(x)
                counts.append(torch.zeros(len(data.bank.probability),
                                           dtype=torch.float64).scatter_add(
                    0, ids_t, mass.reshape(-1)))
            g = torch.maximum(counts[0], counts[1]).sum()
            residuals = (loss[0] - caps[0], loss[1] - caps[1], g - identity_guard)
            penalty = g
            for index, residual in enumerate(residuals):
                penalty = penalty + dual[index] * residual + \
                    .5 * recipe.rho * torch.clamp(residual, min=0).square()
            penalty.backward()
            torch.nn.utils.clip_grad_norm_(weights, recipe.gradient_norm)
            optim.step()
            receipt["updates"] = update
            if update % recipe.dual_every == 0:
                dual = np.maximum(0, dual + recipe.rho *
                                  np.array([float(v.detach()) for v in residuals]))
            if update % recipe.checkpoint_every == 0 or update == recipe.updates:
                checkpoint(update)
        receipt["completed"] = not interrupted
        starts.append(receipt)
        if interrupted:
            break
    complete = not interrupted and len(starts) == 2 and all(s["completed"] for s in starts)
    return {"complete": complete, "success": bool(complete and best is not None),
            "starts": starts, "factor": best[1] if complete and best else None,
            "metrics": best[2] if complete and best else None,
            "identity_guard": identity_guard}


def fit_local_pair(data1: LocalFactorData, data2: LocalFactorData, *,
                   gate: GateWeights, caps1, caps2, recipe: FitRecipe):
    """Fit each purpose alone, then compose and replay the aligned coalition."""
    if (not isinstance(data1, LocalFactorData) or
            not isinstance(data2, LocalFactorData) or
            not isinstance(gate, GateWeights) or not isinstance(recipe, FitRecipe) or
            data1.unit_ids != data2.unit_ids or
            not np.array_equal(data1.sex, data2.sex) or
            len(caps1) != 2 or len(caps2) != 2 or
            any(not np.isfinite(v) for v in (*caps1, *caps2))):
        raise ValueError("local factors require aligned own-purpose FIT capabilities")
    clock = time.process_time()
    first = _fit_one(data1, caps=caps1, recipe=recipe, clock=clock)
    second = (_fit_one(data2, caps=caps2, recipe=recipe, clock=clock)
              if first["complete"] else None)
    complete = bool(first["complete"] and second is not None and second["complete"])
    success = bool(complete and first["success"] and second["success"])
    policy = None
    metrics = None
    if success:
        policy = LocalPolicy(data1.bank, data2.bank, data1.actions,
                             data2.actions, gate, first["factor"], second["factor"])
        task = TaskOnlyData(data1.score, data2.score, data1.cell, data2.cell,
                            data1.y, data2.y)
        metrics = exact_metrics(ProtectedData(task, data1.sex), policy)
    return {"complete": complete, "success": success, "recipient1": first,
            "recipient2": second, "policy": policy, "metrics": metrics,
            "elapsed_cpu_seconds": time.process_time() - clock}
