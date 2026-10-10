"""FIT-only exact identity projection for retained product/local checkpoints."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import _entropy_margin
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.fit import prepare_law
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.factor_controls import (
    LocalPolicy, ProductPolicy,
)
from .r2_fit_core import PROXY_TOLERANCE
from .r2_laws import BoundaryFactorPolicy


def _softmax_rows(logits, valid):
    logits = np.where(valid, logits, -np.inf)
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _factor_marginal(policy, task, recipient):
    n = len(task.y1)
    factor = policy.factor1 if recipient == 1 else policy.factor2
    score = task.score1 if recipient == 1 else task.score2
    cells = task.cell1 if recipient == 1 else task.cell2
    actions = policy.actions1 if recipient == 1 else policy.actions2
    if isinstance(policy, LocalPolicy):
        features = np.stack([np.r_[row, _entropy_margin(row)] for row in score])
    elif isinstance(policy, ProductPolicy):
        features = np.stack([
            np.r_[task.score1[i], task.score2[i],
                  _entropy_margin(task.score1[i]), _entropy_margin(task.score2[i])]
            for i in range(n)])
    else:
        raise TypeError("factor projection requires a product/local source")
    valid = np.zeros((n, 4), dtype=bool)
    for i, cell in enumerate(cells):
        valid[i, :len(actions[int(cell)])] = True
    logits = np.tanh(features @ factor.w1 + factor.b1) @ factor.w2 + factor.b2
    return _softmax_rows(logits, valid), valid


def _counts(sex, ids, q, width):
    result = np.zeros((2, width), dtype=np.float64)
    for value in (0, 1):
        selected = sex == value
        np.add.at(result[value], ids[selected], q[selected] / len(sex))
    return result


def _g(counts):
    return float(np.maximum(counts[0], counts[1]).sum())


def _upper(base, full, cap):
    delta = full - base
    return 1.0 if delta <= 0 else float(np.clip(cap / delta, 0.0, 1.0))


def _guard_upper(c0, c1, initial):
    limit = _g(c0) + 1e-12
    if _g((1-initial)*c0 + initial*c1) <= limit:
        return initial
    left, right = 0.0, initial
    for _ in range(60):
        middle = (left + right) / 2
        if _g((1-middle)*c0 + middle*c1) <= limit:
            left = middle
        else:
            right = middle
    return left


def _marginal_prepared(prepared, valid1, valid2, q1, q2):
    # The source's padded action slot is 4*a1+a2. Invalid slots stay zero.
    if q1.shape != valid1.shape or q2.shape != valid2.shape:
        raise ValueError("factor checkpoint shape differs from fixed actions")
    if (np.any(q1[~valid1]) or np.any(q2[~valid2]) or
            not np.allclose(q1.sum(axis=1), 1, rtol=0, atol=1e-12) or
            not np.allclose(q2.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("factor checkpoint law is not normalized")
    q00 = np.zeros_like(prepared.r1_nll)
    q00[:, 0] = 1.0
    e1 = np.zeros_like(q1); e1[:, 0] = 1.0
    e2 = np.zeros_like(q2); e2[:, 0] = 1.0
    q10 = (q1[:, :, None] * e2[:, None, :]).reshape(-1, 16)
    q01 = (e1[:, :, None] * q2[:, None, :]).reshape(-1, 16)
    q11 = (q1[:, :, None] * q2[:, None, :]).reshape(-1, 16)
    return q00, q10, q01, q11


def project_product_checkpoint(data, source_policy):
    """Select eta1,eta2 from one fixed P checkpoint, with FIT-only quantities."""
    if type(source_policy) is not ProductPolicy:
        raise TypeError("central product checkpoint required")
    task = data.task
    prepared = prepare_law(task, source_policy)
    q1, valid1 = _factor_marginal(source_policy, task, 1)
    q2, valid2 = _factor_marginal(source_policy, task, 2)
    q00, q10, q01, q11 = _marginal_prepared(prepared, valid1, valid2, q1, q2)
    # Each task's expected cost depends only on its own marginal eta.
    upper = [1.0, 1.0]
    for r, changed in ((1, q10), (2, q01)):
        for metric, allowance in (("nll", .001), ("brier", .0005)):
            key = f"r{r}_{metric}"
            before = float(np.mean(np.sum(q00 * getattr(prepared, key), axis=1)))
            after = float(np.mean(np.sum(changed * getattr(prepared, key), axis=1)))
            upper[r-1] = min(upper[r-1], _upper(before, after, allowance))
    c1_0 = _counts(data.sex, prepared.code1, q00, prepared.z1)
    c1_1 = _counts(data.sex, prepared.code1, q10, prepared.z1)
    c2_0 = _counts(data.sex, prepared.code2, q00, prepared.z2)
    c2_1 = _counts(data.sex, prepared.code2, q01, prepared.z2)
    upper[0] = _guard_upper(c1_0, c1_1, upper[0])
    upper[1] = _guard_upper(c2_0, c2_1, upper[1])
    pair_ids = prepared.public_pair
    c00 = _counts(data.sex, pair_ids, q00, prepared.z1 * prepared.z2)
    c10 = _counts(data.sex, pair_ids, q10, prepared.z1 * prepared.z2)
    c01 = _counts(data.sex, pair_ids, q01, prepared.z1 * prepared.z2)
    c11 = _counts(data.sex, pair_ids, q11, prepared.z1 * prepared.z2)

    def objective(pair):
        a, b = pair
        counts = ((1-a)*(1-b)*c00 + a*(1-b)*c10 +
                  (1-a)*b*c01 + a*b*c11)
        local1 = (1-a)*c1_0 + a*c1_1
        local2 = (1-b)*c2_0 + b*c2_1
        return _g(counts) + .5*(_g(local1) + _g(local2))

    bounds = [(0.0, upper[0]), (0.0, upper[1])]
    candidates = [(0.0, 0.0), (upper[0], 0.0),
                  (0.0, upper[1]), tuple(upper)]
    if min(upper) > 0:
        result = minimize(objective, np.asarray(upper)/2, method="Powell",
                          bounds=bounds, options={"xtol": 1e-9, "ftol": 1e-11,
                                                  "maxiter": 80})
        if result.success:
            candidates.append(tuple(float(v) for v in result.x))
    rank = sorted((objective(x), x[0], x[1]) for x in candidates)
    for value, a, b in rank:
        selected = BoundaryFactorPolicy(source_policy, a, b)
        # Confirm the actual constructed joint public law, not a surrogate.
        q = ((1-a)*(1-b)*q00 + a*(1-b)*q10 +
             (1-a)*b*q01 + a*b*q11)
        costs = {key: float(np.mean(np.sum(q*getattr(prepared, key), axis=1)))
                 for key in PROXY_TOLERANCE}
        b0 = {key: float(np.mean(np.sum(q00*getattr(prepared, key), axis=1)))
              for key in PROXY_TOLERANCE}
        if all(costs[key] - b0[key] <= PROXY_TOLERANCE[key] + 1e-13
               for key in PROXY_TOLERANCE) and all((a, b)[r-1] <= upper[r-1]+1e-13
                                                   for r in (1, 2)):
            return selected, {"objective": value, "eta1": a, "eta2": b,
                              "fit_costs": costs, "b0_fit_costs": b0,
                              "fit_excess": {k: costs[k]-b0[k] for k in costs},
                              "local_bayes": {"r1": _g((1-a)*c1_0+a*c1_1),
                                              "r2": _g((1-b)*c2_0+b*c2_1)},
                              "identity_alias": a == b == 0.0}
    raise RuntimeError("exact product identity was excluded from feasible set")


def project_local_checkpoint(data, factor, *, recipient):
    """Select one own-purpose eta without reading other recipient inputs."""
    if recipient not in (1, 2):
        raise ValueError("recipient must be 1 or 2")
    from .r2_source_fit_local import _prepare, _bayes
    x, valid, ids, costs = _prepare(data)
    logits = np.tanh(x @ factor.w1 + factor.b1) @ factor.w2 + factor.b2
    q1 = _softmax_rows(logits, valid)
    q0 = np.zeros_like(q1); q0[:, 0] = 1.0
    upper = 1.0
    for index, allowance in enumerate((.001, .0005)):
        before = float(np.mean(np.sum(q0*costs[:, :, index], axis=1)))
        after = float(np.mean(np.sum(q1*costs[:, :, index], axis=1)))
        upper = min(upper, _upper(before, after, allowance))
    c0 = _counts(data.sex, ids, q0, len(data.bank.probability))
    c1 = _counts(data.sex, ids, q1, len(data.bank.probability))
    upper = _guard_upper(c0, c1, upper)
    objective = lambda eta: _g((1-eta)*c0+eta*c1)
    candidates = [0.0, upper]
    if upper > 0:
        result = minimize_scalar(objective, bounds=(0, upper), method="bounded",
                                 options={"xatol": 1e-10, "maxiter": 80})
        if result.success:
            candidates.append(float(result.x))
    value, eta = min((objective(v), v) for v in candidates)
    q = (1-eta)*q0 + eta*q1
    metrics = tuple(float(np.mean(np.sum(q*costs[:, :, j], axis=1)))
                    for j in (0, 1))
    b0 = tuple(float(np.mean(np.sum(q0*costs[:, :, j], axis=1)))
               for j in (0, 1))
    if (metrics[0]-b0[0] > .001+1e-13 or metrics[1]-b0[1] > .0005+1e-13
            or value > _g(c0)+1e-12):
        raise RuntimeError("projected local checkpoint violates B0 FIT caps")
    return {"recipient": recipient, "eta": eta, "objective": value,
            "fit_nll": metrics[0], "fit_brier": metrics[1],
            "b0_fit_nll": b0[0], "b0_fit_brier": b0[1],
            "fit_nll_excess": metrics[0]-b0[0],
            "fit_brier_excess": metrics[1]-b0[1],
            "identity_alias": eta == 0.0}
