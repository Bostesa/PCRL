"""Prefit bank-relative selection around the inherited central gate optimizer.

This reuses the old two-start Adam training path and searches only the exact
identity-to-checkpoint segment introduced in the bank-anchored design. All
selection uses FIT quantities. No role loader or queue bypass lives here.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize_scalar

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.fit import (
    FitRecipe, ProtectedData, TaskOnlyData, TASK_CAP_ORDER,
    _bayes_numpy, _probabilities_numpy, _task_rate_numpy,
    fit_gate, prepare_law,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import JointPolicy

from .r2_laws import BoundaryJointPolicy, IdentityPolicy


PROXY_TOLERANCE = {"r1_nll": .001, "r1_brier": .0005,
                   "r2_nll": .001, "r2_brier": .0005}
LOCAL_NUMERIC_TOLERANCE = 1e-12


def _identity_q(prepared):
    q = np.zeros_like(prepared.r1_nll)
    q[:, 0] = 1.0
    if not np.all(prepared.valid[:, 0]):
        raise ValueError("source action zero is not exact bank identity")
    return q


def _task_means(prepared, q):
    return {key: float(np.mean(np.sum(q * getattr(prepared, key), axis=1)))
            for key in TASK_CAP_ORDER}


def _accuracy_check(prepared, task, policy, q):
    if q.shape != prepared.valid.shape:
        raise ValueError("accuracy law shape differs from fixed actions")
    correct1 = (policy.bank1.decision[prepared.code1] == task.y1[:, None])
    correct2 = (policy.bank2.decision[prepared.code2] == task.y2[:, None])
    out = {"r1": float(np.mean(np.sum(q * correct1, axis=1))),
           "r2": float(np.mean(np.sum(q * correct2, axis=1)))}
    return out


def _metric(prepared, q, sex, objective_kind):
    if objective_kind == "task_rate":
        return _task_rate_numpy(prepared, q)
    bayes = _bayes_numpy(prepared, q, sex)
    return bayes["pair_bayes"] + .5 * (bayes["r1_bayes"] + bayes["r2_bayes"])


def _local_feasible(prepared, q0, q1, sex, eta, baseline):
    if sex is None:
        return True
    q = (1.0 - eta) * q0 + eta * q1
    bayes = _bayes_numpy(prepared, q, sex)
    return all(bayes[key] <= baseline[key] + LOCAL_NUMERIC_TOLERANCE
               for key in ("r1_bayes", "r2_bayes"))


def project_joint_segment(prepared, source_q, *, sex=None,
                          objective_kind: str):
    """Select eta on one locked checkpoint segment by FIT utility/objective.

    Identity is exact eta=0. Utility is affine in eta. Local fitted-law G caps
    cut a convex interval containing zero. One bounded scalar minimization
    searches this segment; endpoint and returned point are exactly replayed.
    """
    if objective_kind not in ("task_rate", "protected_bayes"):
        raise ValueError("unregistered source objective")
    if (objective_kind == "protected_bayes") != (sex is not None):
        raise ValueError("SEX capability differs from objective kind")
    q0 = _identity_q(prepared)
    q1 = np.asarray(source_q, dtype=np.float64)
    if (q1.shape != q0.shape or not np.isfinite(q1).all() or
            np.any(q1 < 0) or not np.allclose(q1.sum(axis=1), 1,
                                            rtol=0, atol=1e-12) or
            np.any(q1[~prepared.valid] != 0)):
        raise ValueError("fitted source law has invalid action mass")
    base = _task_means(prepared, q0)
    changed = _task_means(prepared, q1)
    upper = 1.0
    for key, tolerance in PROXY_TOLERANCE.items():
        increase = changed[key] - base[key]
        if increase > 0:
            upper = min(upper, tolerance / increase)
    upper = max(0.0, min(1.0, upper))
    guards = _bayes_numpy(prepared, q0, sex) if sex is not None else None
    if sex is not None and not _local_feasible(prepared, q0, q1, sex, upper, guards):
        left, right = 0.0, upper
        for _ in range(60):
            middle = (left + right) / 2.0
            if _local_feasible(prepared, q0, q1, sex, middle, guards):
                left = middle
            else:
                right = middle
        upper = left

    def objective(eta):
        q = (1.0 - eta) * q0 + eta * q1
        return _metric(prepared, q, sex, objective_kind)

    candidates = [0.0, upper]
    if upper > 0:
        solve = minimize_scalar(objective, bounds=(0.0, upper), method="bounded",
                                options={"xatol": 1e-10, "maxiter": 80})
        if solve.success:
            candidates.append(float(solve.x))
    # Exact endpoints win ties. The selected eta is checked after construction;
    # no numerical grace can turn an infeasible proxy into a feasible return.
    ranked = sorted((objective(eta), eta) for eta in candidates)
    for value, eta in ranked:
        q = (1.0 - eta) * q0 + eta * q1
        means = _task_means(prepared, q)
        feasible = all(means[key] - base[key] <= PROXY_TOLERANCE[key] + 1e-13
                       for key in PROXY_TOLERANCE)
        if feasible and (sex is None or _local_feasible(prepared, q0, q1, sex,
                                                       eta, guards)):
            return {"eta": float(eta), "objective": float(value),
                    "fit_costs": means,
                    "b0_fit_costs": base,
                    "fit_excess": {k: means[k] - base[k] for k in base},
                    "local_bayes": (_bayes_numpy(prepared, q, sex) if sex is not None
                                    else None),
                    "identity_alias": eta == 0.0}
    raise RuntimeError("exact identity was incorrectly excluded from feasible set")


def fit_central_boundary(data: TaskOnlyData | ProtectedData,
                         base_policy: JointPolicy, *, objective_kind: str,
                         recipe: FitRecipe):
    """One J or T fit; select from all source checkpoints plus exact B0.

    The caller must already own a queued/admitted FIT unit. This function is
    deliberately loader-free and cannot open INNER or assessment labels.
    """
    if objective_kind == "task_rate" and type(data) is not TaskOnlyData:
        raise TypeError("task-only channel must not receive SEX")
    if objective_kind == "protected_bayes" and type(data) is not ProtectedData:
        raise TypeError("protected channel needs an admitted SEX capability")
    task = data if type(data) is TaskOnlyData else data.task
    prepared = prepare_law(task, base_policy)
    q0 = _identity_q(prepared)
    base = _task_means(prepared, q0)
    caps = {key: base[key] + PROXY_TOLERANCE[key] for key in TASK_CAP_ORDER}
    source = fit_gate(data, base_policy, objective_kind=objective_kind,
                      caps=caps, recipe=recipe)
    if not source["complete"]:
        return {"complete": False, "source": source, "selection": None,
                "policy": None}
    sex = None if objective_kind == "task_rate" else data.sex
    candidates = []
    for start_index, start in enumerate(source["starts"]):
        for entry in start["checkpoints"]:
            q1 = _probabilities_numpy(prepared, entry["weights"])
            projected = project_joint_segment(prepared, q1, sex=sex,
                                              objective_kind=objective_kind)
            candidates.append((projected["objective"], start_index,
                               entry["update"], projected["eta"],
                               entry["weights"], projected))
    # This ordering uses only fitting quantities and cannot choose on INNER.
    objective, start_index, update, eta, weights, selected = min(candidates,
        key=lambda row: (row[0], row[1], row[2], row[3]))
    source_policy = JointPolicy(base_policy.bank1, base_policy.bank2,
                                base_policy.actions1, base_policy.actions2, weights)
    policy = BoundaryJointPolicy(source_policy, eta)
    q_selected = (1.0 - eta) * q0 + eta * _probabilities_numpy(prepared, weights)
    accuracy_b0 = _accuracy_check(prepared, task, base_policy, q0)
    accuracy_selected = _accuracy_check(prepared, task, policy, q_selected)
    accuracy_loss = {r: accuracy_b0[r] - accuracy_selected[r] for r in ("r1", "r2")}
    if any(loss > .002 + 1e-13 for loss in accuracy_loss.values()):
        raise RuntimeError("selected channel violates explicit B0 accuracy proxy")
    selected.update({"b0_fit_accuracy": accuracy_b0,
                     "fit_accuracy": accuracy_selected,
                     "fit_accuracy_loss": accuracy_loss})
    return {"complete": True, "source": source,
            "selection": {"start": source["starts"][start_index]["name"],
                          "update": int(update), **selected},
            "policy": policy}
