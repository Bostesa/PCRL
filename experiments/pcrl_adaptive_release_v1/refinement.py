"""Nested, deployable internal-state refinement for 17-token PCRL releases.

The state is private mechanism computation, never a wire feature. All split
features must be functions of local X_A/H_A and frozen predictors of them.
This module does not fit a nuisance model or inspect a person's Y/S labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Mapping, Sequence

import numpy as np


ALLOWED_FEATURES = frozenset(
    ["residual", "task_posterior"]
    + [f"h_a_{i}" for i in range(4)]
    + [f"sex_risk_{i}" for i in range(2)]
    + [f"race_risk_{i}" for i in range(9)]
)


def _binary_mi(joint: Mapping[tuple[int, int], Fraction]) -> float:
    """Mutual information of a finite binary Y and finite token Z law."""
    py = {y: sum(p for (yy, _), p in joint.items() if yy == y) for y in (0, 1)}
    pz = {z: sum(p for (_, zz), p in joint.items() if zz == z)
          for z in {k[1] for k in joint}}
    return sum(float(p) * math.log(float(p / (py[y] * pz[z])))
               for (y, z), p in joint.items() if p)


def _task_information(masses: Sequence[Fraction], task_p: Sequence[Fraction],
                      tokens: Sequence[int] | Sequence[Fraction]) -> float:
    """Exact rational joint law followed by a floating-point logarithm."""
    joint: dict[tuple[int, int], Fraction] = {}
    for mass, p_y, token in zip(masses, task_p, tokens):
        if isinstance(token, Fraction):
            outcomes = ((0, 1-token), (1, token))
        else:
            outcomes = ((int(token), Fraction(1)),)
        for z, p_z in outcomes:
            for y, p in ((0, 1-p_y), (1, p_y)):
                joint[y, z] = joint.get((y, z), Fraction(0)) + mass*p_z*p
    return _binary_mi(joint)


def _partitions(n: int):
    """Canonical restricted-growth strings, one per set partition."""
    if n < 1:
        return
    def walk(prefix: tuple[int, ...], highest: int):
        if len(prefix) == n:
            yield prefix
        else:
            for label in range(highest + 2):
                yield from walk(prefix + (label,), max(highest, label))
    yield from walk((0,), 0)


def fixture_b1() -> dict:
    """Four-state perfect-privacy example, with exact independence checks.

    The task posterior T is .1 on X={0,1} and .9 on X={2,3}, hence exactly
    sufficient for Y. For T-only token rows a,b, S-independence says
    (4a+b)/5 = (a+4b)/5, so a=b. The richer stochastic X channel below
    preserves S-independence yet retains task information.
    """
    masses = tuple(map(Fraction, ("2/5", "1/10", "1/10", "2/5")))
    s = (0, 1, 0, 1)
    task_p = tuple(map(Fraction, ("1/10", "1/10", "9/10", "9/10")))
    q_z1 = tuple(map(Fraction, ("0", "0", "1", "1/4")))
    ps = tuple(sum(masses[i] for i in range(4) if s[i] == a) for a in (0, 1))
    p_z1_s = tuple(sum(masses[i]*q_z1[i] for i in range(4) if s[i] == a)/ps[a]
                     for a in (0, 1))
    pz1 = sum(m*q for m, q in zip(masses, q_z1))
    py_z1 = sum(m*q*p for m, q, p in zip(masses, q_z1, task_p))/pz1
    py_z0 = sum(m*(1-q)*p for m, q, p in zip(masses, q_z1, task_p))/(1-pz1)

    partitions = list(_partitions(4))
    private = []
    for partition in partitions:
        # P(Z=z|S=0)==P(Z=z|S=1), tested exactly in Q.
        if all(sum(masses[i] for i in range(4) if s[i] == 0 and partition[i] == z)/ps[0]
               == sum(masses[i] for i in range(4) if s[i] == 1 and partition[i] == z)/ps[1]
               for z in set(partition)):
            private.append(partition)
    return {
        "input_masses": masses,
        "private_stochastic_z1": q_z1,
        "p_z1_given_s": p_z1_s,
        "p_y1_given_z": (py_z0, py_z1),
        "t_only_private_implies_row_equality": Fraction(4, 5)-Fraction(1, 5) != 0,
        "partitions_enumerated": len(partitions),
        "deterministic_private_partitions": len(private),
        "private_partitions": tuple(private),
        "max_deterministic_private_task_information": max(
            _task_information(masses, task_p, p) for p in private),
        "stochastic_task_information_nats": _task_information(masses, task_p, q_z1),
    }


def fixture_b2() -> dict:
    """Fair H,T example: context selects a deterministic private release."""
    # Each row is (h,t,s,y,z,mass). H,T are independent fair bits.
    rows = tuple(
        (0, t, s, t, t, Fraction(1, 8)) for t in (0, 1) for s in (0, 1)
    ) + tuple((1, t, t, 0, 0, Fraction(1, 4)) for t in (0, 1))
    task_information = 0.
    sensitive_information = 0.
    exact_sensitive_independence = True
    for h in (0, 1):
        yz: dict[tuple[int, int], Fraction] = {}
        sz: dict[tuple[int, int], Fraction] = {}
        for hh, _, s, y, z, mass in rows:
            if hh != h:
                continue
            yz[y, z] = yz.get((y, z), Fraction(0)) + 2*mass
            sz[s, z] = sz.get((s, z), Fraction(0)) + 2*mass
        task_information += 0.5*_binary_mi(yz)
        sensitive_information += 0.5*_binary_mi(sz)
        exact_sensitive_independence &= all(
            p == sum(v for (ss, _), v in sz.items() if ss == s)
                 * sum(v for (_, zz), v in sz.items() if zz == z)
            for (s, z), p in sz.items()
        )
    return {"conditional_sensitive_information_nats": sensitive_information,
            "conditional_task_information_nats": task_information,
            "exact_sensitive_independence": exact_sensitive_independence,
            # At H=1 S=T, so a context-blind channel hides S only if its
            # two T rows agree, which then removes Y information at H=0.
            "context_blind_private_implies_constant": True,
            "contextual_map": ((0, 0), (0, 1), (1, 0), (1, 0))}


@dataclass(frozen=True)
class SplitRule:
    leaf: int
    feature_name: str
    threshold: float
    new_leaf: int


@dataclass(frozen=True)
class NestedPartition:
    """Private child IDs; a split retains the old leaf for <= threshold."""
    parents: tuple[int, ...]
    rules: tuple[SplitRule, ...] = ()

    @classmethod
    def base(cls, n_parent: int = 32) -> "NestedPartition":
        if n_parent < 1:
            raise ValueError("positive parent count required")
        return cls(tuple(range(n_parent)))

    @property
    def parent_of_leaf(self) -> np.ndarray:
        return np.asarray(self.parents, dtype=np.int64)

    def to_record(self) -> dict:
        """JSON-safe, reviewable routing rule record."""
        return {"schema": "pcrl-nested-state-v1", "n_parent": self.n_parent,
                "parents": list(self.parents),
                "rules": [{"leaf": r.leaf, "feature_name": r.feature_name,
                           "threshold": r.threshold, "new_leaf": r.new_leaf}
                          for r in self.rules]}

    @classmethod
    def from_record(cls, record: Mapping) -> "NestedPartition":
        if set(record) != {"schema", "n_parent", "parents", "rules"} or record["schema"] != "pcrl-nested-state-v1":
            raise ValueError("invalid nested-state record schema")
        result = cls.base(int(record["n_parent"]))
        for rule in record["rules"]:
            if set(rule) != {"leaf", "feature_name", "threshold", "new_leaf"}:
                raise ValueError("invalid nested-state split rule")
            if rule["new_leaf"] != len(result.parents):
                raise ValueError("nonsequential child ID")
            result = result.split(int(rule["leaf"]), rule["feature_name"],
                                  float(rule["threshold"]))
        if list(result.parents) != record["parents"]:
            raise ValueError("nested-state parent map mismatch")
        return result

    def split(self, leaf: int, feature_name: str, threshold: float) -> "NestedPartition":
        if not 0 <= leaf < len(self.parents):
            raise ValueError("unknown leaf")
        if len(self.parents) >= 128:
            raise ValueError("registered maximum is 128 internal cells")
        if feature_name not in ALLOWED_FEATURES:
            raise ValueError("forbidden or unknown split feature")
        if not math.isfinite(threshold):
            raise ValueError("finite split threshold required")
        new_leaf = len(self.parents)
        return NestedPartition(self.parents + (self.parents[leaf],),
                               self.rules + (SplitRule(leaf, feature_name,
                                                       float(threshold), new_leaf),))

    def route(self, parent_codes: np.ndarray,
              features: Mapping[str, np.ndarray]) -> np.ndarray:
        parent = np.asarray(parent_codes)
        if parent.ndim != 1 or parent.dtype.kind not in "iu" or np.any(parent < 0) or np.any(parent >= self.n_parent):
            raise ValueError("invalid parent codes")
        leaf = parent.astype(np.int64, copy=True)
        for rule in self.rules:
            if rule.feature_name not in features:
                raise ValueError(f"missing deployable feature {rule.feature_name}")
            values = np.asarray(features[rule.feature_name], dtype=np.float64)
            if values.shape != parent.shape or not np.isfinite(values).all():
                raise ValueError("invalid deployable feature array")
            leaf[(leaf == rule.leaf) & (values > rule.threshold)] = rule.new_leaf
        if not np.array_equal(self.parent_of_leaf[leaf], parent):
            raise AssertionError("child left its T32 parent")
        return leaf

    def route_runtime(self, inputs, frozen_encoder, frozen_nuisance) -> np.ndarray:
        """Deployment path: derive every split feature from local inputs."""
        parent, features = build_deployable_features(
            inputs, frozen_encoder, frozen_nuisance)
        return self.route(parent, features)

    @property
    def n_parent(self) -> int:
        return max(self.parents) + 1


def copy_parent_kernel(parent_q: np.ndarray, parent_of_leaf: np.ndarray) -> np.ndarray:
    """Embed every parent release exactly; no new token or altered row."""
    q = np.asarray(parent_q)
    parent = np.asarray(parent_of_leaf)
    if q.ndim != 2 or q.shape[1] != 17 or parent.ndim != 1 or parent.dtype.kind not in "iu":
        raise ValueError("expected parent Q with 17 tokens and integer parent IDs")
    if np.any(parent < 0) or np.any(parent >= len(q)):
        raise ValueError("invalid child-parent map")
    if not np.isfinite(q).all() or np.any(q < 0) or not np.allclose(q.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError("invalid parent channel simplex")
    return q[parent].copy()


def priced_contributions(task_rows: np.ndarray, attack_rows: Sequence[np.ndarray],
                         multipliers: Sequence[float]) -> np.ndarray:
    """g_i(z)=u_i(z)-sum_j lambda_j a_ij(z), with masses already inside rows."""
    task = np.asarray(task_rows, dtype=np.float64)
    if task.ndim != 2 or not np.isfinite(task).all() or len(attack_rows) != len(multipliers):
        raise ValueError("invalid priced contributions")
    priced = task.copy()
    for attack, multiplier in zip(attack_rows, multipliers):
        a = np.asarray(attack, dtype=np.float64)
        lam = float(multiplier)
        if a.shape != task.shape or not np.isfinite(a).all() or not math.isfinite(lam) or lam < 0:
            raise ValueError("invalid nonnegative privacy multiplier or attack rows")
        priced -= lam*a
    return priced


def fixed_price_gain(priced_rows: np.ndarray, left_mask: Sequence[bool]) -> float:
    """Freedom gain for a binary split at fixed decoder, bank and dual prices."""
    g = np.asarray(priced_rows, dtype=np.float64)
    left = np.asarray(left_mask, dtype=bool)
    if g.ndim != 2 or g.shape[0] != len(left) or not np.isfinite(g).all() or not left.any() or left.all():
        raise ValueError("nonempty aligned children and finite priced rows required")
    gain = float(g.sum(axis=0).min() - g[left].sum(axis=0).min()
                 - g[~left].sum(axis=0).min())
    if gain < -1e-10:
        raise AssertionError("fixed-price split gain has wrong sign")
    return max(gain, 0.)


@dataclass(frozen=True)
class SplitCandidate:
    leaf: int
    feature_name: str
    threshold: float
    gain: float
    children_households: tuple[int, int]
    children_effective_weight: tuple[float, float]
    checking_gain: float | None = None


def _effective_count(weights: np.ndarray) -> float:
    total = float(weights.sum())
    return total*total/float(np.square(weights).sum()) if total > 0 else 0.


def _effective_households(households: np.ndarray, weights: np.ndarray) -> float:
    """Kish count after aggregating all original-person mass by household."""
    _, inverse = np.unique(households, return_inverse=True)
    return _effective_count(np.bincount(inverse, weights=weights))


def rank_splits(*, leaf_ids: np.ndarray, leaf: int,
                features: Mapping[str, np.ndarray], priced_rows: np.ndarray,
                households: np.ndarray, weights: np.ndarray,
                min_households: int = 100, min_effective_weight: float = 100,
                feature_names: Sequence[str] = (),
                quantiles: Sequence[float] = (.25, .5, .75),
                max_candidates_per_feature: int = 3,
                checking: tuple[np.ndarray, Mapping[str, np.ndarray], np.ndarray, np.ndarray] | None = None
                ) -> list[SplitCandidate]:
    """Rank bounded training-quantile splits; optional checking scores do not set cuts.

    `priced_rows` already carries original-person U/PWGTP normalization. The
    caller must supply a separate household-disjoint checking resource when
    using `checking`; it may use that score to reject unstable split gains.
    """
    states = np.asarray(leaf_ids)
    g = np.asarray(priced_rows, dtype=np.float64)
    hh = np.asarray(households)
    w = np.asarray(weights, dtype=np.float64)
    if (states.ndim != 1 or g.ndim != 2 or len(states) != len(g) or
            hh.shape != states.shape or w.shape != states.shape or
            not np.isfinite(g).all() or not np.isfinite(w).all() or np.any(w < 0)):
        raise ValueError("unaligned split resources")
    if min_households < 1 or min_effective_weight <= 0 or max_candidates_per_feature < 1:
        raise ValueError("invalid preregistered support/candidate limits")
    if any(not 0 < float(q) < 1 for q in quantiles):
        raise ValueError("quantiles must be internal")
    active = states == leaf
    out = []
    for name in feature_names:
        if name not in ALLOWED_FEATURES:
            raise ValueError("forbidden split feature")
        if name not in features:
            raise ValueError("missing deployable split feature")
        x = np.asarray(features[name], dtype=np.float64)
        if x.shape != states.shape or not np.isfinite(x).all():
            raise ValueError("invalid split feature")
        if active.sum() < 2:
            continue
        thresholds = np.unique(np.quantile(x[active], quantiles))[:max_candidates_per_feature]
        for threshold in thresholds:
            left = active & (x <= threshold)
            right = active & (x > threshold)
            if not left.any() or not right.any():
                continue
            support = (len(np.unique(hh[left])), len(np.unique(hh[right])))
            neff = (_effective_households(hh[left], w[left]),
                    _effective_households(hh[right], w[right]))
            if min(support) < min_households or min(neff) < min_effective_weight:
                continue
            gain = fixed_price_gain(g[active], left[active])
            check_gain = None
            if checking is not None:
                check_states, check_features, check_g, check_households = checking
                check_states = np.asarray(check_states)
                check_g = np.asarray(check_g, dtype=float)
                check_households = np.asarray(check_households)
                if name not in check_features:
                    raise ValueError("missing checking feature")
                check_x = np.asarray(check_features[name], dtype=float)
                if (check_states.ndim != 1 or check_g.ndim != 2 or
                        len(check_states) != len(check_g) or
                        check_x.shape != check_states.shape or
                        check_households.shape != check_states.shape):
                    raise ValueError("unaligned checking rows")
                if np.intersect1d(hh, check_households).size:
                    raise ValueError("checking households overlap split-fitting households")
                c = check_states == leaf
                c_left = c & (check_x <= threshold)
                c_right = c & (check_x > threshold)
                if c_left.any() and c_right.any():
                    check_gain = fixed_price_gain(check_g[c], c_left[c])
            out.append(SplitCandidate(leaf, name, float(threshold), gain, support,
                                      neff, check_gain))
    return sorted(out, key=lambda c: (-c.gain, c.feature_name, c.threshold))


def build_deployable_features(inputs, frozen_encoder, frozen_nuisance):
    """Apply frozen local models only; returns T0 and legal split features.

    Nuisance fitting, hashes, split roles and timing are the caller's separate
    responsibility. Requiring the historical RuntimeInputs type excludes H_B,
    observed labels, row losses, and household IDs from this API.
    """
    from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
    if not isinstance(inputs, RuntimeInputs):
        raise TypeError("deployable refinement accepts RuntimeInputs(X_A,H_A) only")
    if frozen_nuisance is None:
        raise ValueError("frozen nuisance predictor required")
    encoded = frozen_encoder.encode(inputs)
    risk = frozen_nuisance.probabilities(inputs)
    t = np.asarray(encoded["codes"]["T0"], dtype=np.int64)
    features = {"residual": np.asarray(encoded["r"], dtype=float),
                "task_posterior": np.asarray(encoded["p"], dtype=float)}
    features.update({f"h_a_{i}": np.asarray(inputs.h_a)[:, i] for i in range(4)})
    for key, prefix, classes in (("SEX", "sex_risk", 2), ("RAC1P", "race_risk", 9)):
        p = np.asarray(risk[key], dtype=float)
        if p.shape != (len(t), classes) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-8):
            raise ValueError("frozen nuisance probabilities violate full class schema")
        features.update({f"{prefix}_{i}": p[:, i] for i in range(classes)})
    if any(v.shape != t.shape or not np.isfinite(v).all() for v in features.values()):
        raise ValueError("invalid frozen deployable feature")
    return t, features
