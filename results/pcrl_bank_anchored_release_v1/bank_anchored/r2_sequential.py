"""Frozen-first sequential bank law with exact identity and second boundaries."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import (
    JointPolicy, aggregate_public_pairs,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.controls.factor_controls import (
    SequentialPolicy, _central_features, _factor_law,
)
from .r2_laws import identity_pair, _eta
from .r2_fit_core import PROXY_TOLERANCE, _identity_q, _task_means
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.fit import (
    prepare_law, _bayes_numpy,
)


class BoundarySequentialPolicy(JointPolicy):
    """Exact B0 or source first law, then conditional second-law mixture.

    The first law is frozen before the second is fitted. `eta_first` mixes
    the original B0 code with a retained first-stage source factor. `eta_second` is a
    fitted scalar chosen only on FIT; each second output is an actual draw.
    """

    law_kind = "bank_boundary_sequential_v1"

    def __init__(self, base: SequentialPolicy, eta_first: float,
                 eta_second: float):
        if type(base) is not SequentialPolicy:
            raise TypeError("frozen-first sequential source/identity flag differs")
        super().__init__(base.bank1, base.bank2, base.actions1, base.actions2,
                         base.gate)
        self.base = base
        self.eta_first = _eta(eta_first)
        self.eta_second = _eta(eta_second)

    def sparse_law(self, score1, score2, cell1, cell2):
        identity = identity_pair(self, score1, score2, cell1, cell2)
        features = _central_features(score1, score2)
        if self.base.order == "12":
            first_actions = self.actions1[cell1]
            second_actions = self.actions2[cell2]
            first_b0, second_b0 = identity
        else:
            first_actions = self.actions2[cell2]
            second_actions = self.actions1[cell1]
            first_b0, second_b0 = identity[1], identity[0]
        if self.eta_first == 0:
            first = ((first_b0, 1.0),)
        else:
            source_first = _factor_law(self.base.first, features, first_actions)
            masses = {int(code): self.eta_first*float(mass)
                      for code, mass in source_first}
            masses[first_b0] = masses.get(first_b0, 0.0) + 1-self.eta_first
            first = tuple(sorted(masses.items()))
        pairs, probabilities = [], []
        for first_code, first_mass in first:
            source_second = _factor_law(self.base.second, features,
                                        second_actions, first_code=first_code)
            second = {int(code): self.eta_second * float(mass)
                      for code, mass in source_second}
            second[second_b0] = second.get(second_b0, 0.0) + 1.0 - self.eta_second
            for second_code, second_mass in second.items():
                pair = ((first_code, second_code) if self.base.order == "12"
                        else (second_code, first_code))
                pairs.append(pair)
                probabilities.append(first_mass * second_mass)
        return aggregate_public_pairs(pairs, np.asarray(probabilities,
                                                       dtype=np.float64))

    def mechanism_descriptor(self):
        return {"law_kind": self.law_kind, "order": self.base.order,
                "eta_first": self.eta_first, "eta_second": self.eta_second,
                "first_checkpoint_sha256": self.base.first_checkpoint_sha256}


def _prepared_q(prepared, task, policy):
    q = np.zeros_like(prepared.r1_nll)
    for row in range(len(task.y1)):
        slots = {(int(prepared.code1[row, slot]), int(prepared.code2[row, slot])): slot
                 for slot in np.flatnonzero(prepared.valid[row])}
        law = policy.sparse_law(task.score1[row], task.score2[row],
                                int(task.cell1[row]), int(task.cell2[row]))
        for z1, z2, mass in law:
            q[row, slots[(z1, z2)]] += mass
    if not np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("frozen sequential public law is not normalized")
    return q


def project_second_checkpoint(data, source_policy, *, eta_first):
    """Optimize only second-stage eta after first release is frozen."""
    if type(source_policy) is not SequentialPolicy or not 0 <= eta_first <= 1:
        raise TypeError("frozen sequential checkpoint required")
    task = data.task
    prepared = prepare_law(task, source_policy)
    b0q = _identity_q(prepared)
    q0 = _prepared_q(prepared, task,
                     BoundarySequentialPolicy(source_policy, eta_first, 0.0))
    q1 = _prepared_q(prepared, task,
                     BoundarySequentialPolicy(source_policy, eta_first, 1.0))
    b0 = _task_means(prepared, b0q)
    start = _task_means(prepared, q0)
    end = _task_means(prepared, q1)
    upper = 1.0
    for key, cap in PROXY_TOLERANCE.items():
        room = b0[key] + cap - start[key]
        if room < -1e-12:
            raise RuntimeError("frozen first sequential release violates FIT proxy")
        delta = end[key] - start[key]
        if delta > 0:
            upper = min(upper, max(0.0, room / delta))
    guards = _bayes_numpy(prepared, b0q, data.sex)

    def metrics(eta):
        q = (1-eta)*q0 + eta*q1
        bayes = _bayes_numpy(prepared, q, data.sex)
        objective = bayes["pair_bayes"] + .5*(bayes["r1_bayes"] + bayes["r2_bayes"])
        return objective, bayes, _task_means(prepared, q)

    def feasible(eta):
        _, bayes, cost = metrics(eta)
        return (all(bayes[f"r{r}_bayes"] <= guards[f"r{r}_bayes"] + 1e-12
                    for r in (1, 2)) and
                all(cost[k] - b0[k] <= cap + 1e-12
                    for k, cap in PROXY_TOLERANCE.items()))

    upper = float(np.clip(upper, 0, 1))
    if not feasible(upper):
        left, right = 0.0, upper
        if not feasible(0.0):
            raise RuntimeError("frozen first sequential law is not locally feasible")
        for _ in range(60):
            middle = (left+right)/2
            if feasible(middle):
                left = middle
            else:
                right = middle
        upper = left
    candidates = [0.0, upper]
    if upper > 0:
        result = minimize_scalar(lambda x: metrics(x)[0], bounds=(0, upper),
                                 method="bounded",
                                 options={"xatol": 1e-10, "maxiter": 80})
        if result.success:
            candidates.append(float(result.x))
    for objective, eta in sorted((metrics(eta)[0], eta) for eta in candidates):
        if feasible(eta):
            _, bayes, cost = metrics(eta)
            return BoundarySequentialPolicy(source_policy, eta_first, eta), {
                "objective": objective, "eta_first": eta_first,
                "eta_second": eta, "fit_costs": cost,
                "b0_fit_costs": b0,
                "fit_excess": {k: cost[k]-b0[k] for k in b0},
                "local_bayes": bayes,
                "identity_alias": eta_first == 0 and eta == 0.0}
    raise RuntimeError("sequential identity completion excluded from feasible set")
