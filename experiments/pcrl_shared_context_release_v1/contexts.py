"""Hard shared contexts (K in {1, 2, 4}) from legal frozen features.

K=4 is a 2x2 split at PWGTP-weighted medians, fitted on nuisance_train only,
of (i) the stored residual r and (ii) the frozen local SEX-risk
max-probability (`AR/nuisance.fit_frozen_nuisance`).  K=2 (registered
fallback) splits on the SEX-risk coordinate only.  A context id is private
mechanism computation and never a wire field.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from .policies import check_legal

MIN_CONTEXT_HOUSEHOLDS = 300
MIN_CONTEXT_HH_ESS = 150.
REGISTERED_K = (1, 2, 4)


def weighted_median(values, weights):
    """Smallest value whose cumulative weight reaches half the total."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (v.ndim != 1 or w.shape != v.shape or len(v) == 0 or not np.isfinite(v).all()
            or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0):
        raise ValueError("finite aligned values and positive weights required")
    order = np.argsort(v, kind="mergesort")
    cumulative = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")])


def sex_risk_max(nuisance, inputs) -> np.ndarray:
    legal = check_legal(inputs)
    local = RuntimeInputs(np.asarray(legal["x"]), np.asarray(legal["ha"]))
    p = nuisance.probabilities(local)["SEX"]
    return np.asarray(p, dtype=np.float64).max(1)


@dataclass(frozen=True)
class ContextRule:
    """Frozen hard context map; thresholds and nuisance fitted on nuisance_train."""
    K: int
    residual_median: float
    sex_risk_median: float
    nuisance: object
    nuisance_sha256: str

    def assign(self, inputs) -> np.ndarray:
        legal = check_legal(inputs)
        n = len(np.asarray(legal["x"]))
        if self.K == 1:
            return np.zeros(n, dtype=np.int64)
        s = (sex_risk_max(self.nuisance, legal) > self.sex_risk_median).astype(np.int64)
        if self.K == 2:
            return s
        r = np.asarray(legal["residual"], dtype=np.float64)
        if r.shape != (n,) or not np.isfinite(r).all():
            raise ValueError("stored residual must be finite and aligned")
        return 2 * (r > self.residual_median).astype(np.int64) + s


def fit_context_rules(nuisance_rows, nuisance, nuisance_sha256: str) -> dict:
    """Fit the three registered hard rules on nuisance_train (PWGTP medians)."""
    w = np.asarray(nuisance_rows["weights"], dtype=np.float64)
    legal = {key: nuisance_rows[key] for key in ("x", "ha", "residual")}
    r_med = weighted_median(np.asarray(nuisance_rows["residual"], dtype=np.float64), w)
    s_med = weighted_median(sex_risk_max(nuisance, legal), w)
    return {k: ContextRule(k, r_med, s_med, nuisance, nuisance_sha256) for k in REGISTERED_K}


def _ess(masses):
    m = np.asarray(masses, dtype=np.float64)
    s2 = float((m ** 2).sum())
    return float(m.sum() ** 2 / s2) if s2 > 0 else 0.


def support_census(contexts, households, pwgtp, K) -> dict:
    """Unique households and household-weight ESS per hard context."""
    k = np.asarray(contexts, dtype=np.int64)
    hh = np.asarray(households).astype(str)
    w = np.asarray(pwgtp, dtype=np.float64)
    rows = []
    for c in range(K):
        member = k == c
        if member.any():
            _, inverse = np.unique(hh[member], return_inverse=True)
            mass = np.bincount(inverse, weights=w[member])
            rows.append({"context": c, "people": int(member.sum()),
                         "unique_households": int(len(mass)),
                         "household_weight_ess": _ess(mass),
                         "weight_share": float(w[member].sum() / w.sum())})
        else:
            rows.append({"context": c, "people": 0, "unique_households": 0,
                         "household_weight_ess": 0., "weight_share": 0.})
    supported = all(r["unique_households"] >= MIN_CONTEXT_HOUSEHOLDS and
                    r["household_weight_ess"] >= MIN_CONTEXT_HH_ESS for r in rows)
    return {"K": int(K), "per_context": rows, "meets_registered_floor": bool(supported),
            "floor": {"unique_households": MIN_CONTEXT_HOUSEHOLDS,
                      "household_weight_ess": MIN_CONTEXT_HH_ESS}}


def fallback_decision(k4_census_by_anchor: dict) -> dict:
    """Registered rule: any anchor with an under-supported K=4 context -> K=2."""
    if set(map(str, k4_census_by_anchor)) != {"0", "1", "2"}:
        raise ValueError("the K fallback decision requires all three anchors")
    failing = sorted(str(a) for a, c in k4_census_by_anchor.items()
                     if int(c["K"]) != 4 or not c["meets_registered_floor"])
    return {"nm4_K": 2 if failing else 4, "failing_anchors": failing,
            "rule": "K=2 (SEX risk only) if any K=4 context has < 300 unique coefficient "
                    "households or household-weight ESS < 150 on any anchor"}


def save_rules(directory, rules: dict) -> dict:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "contexts.joblib"
    if path.exists():
        raise FileExistsError("frozen context rules are write-once")
    temporary = root / "contexts.joblib.tmp"
    joblib.dump(rules, temporary, compress=3)
    temporary.replace(path)
    return {"file": "contexts.joblib", "sha256": _sha(path),
            "thresholds": {str(k): {"residual_median": r.residual_median,
                                    "sex_risk_median": r.sex_risk_median}
                           for k, r in rules.items()}}


def load_rules(directory, expected_sha256: str) -> dict:
    path = Path(directory) / "contexts.joblib"
    if _sha(path) != expected_sha256:
        raise ValueError("frozen context rule hash mismatch")
    rules = joblib.load(path)
    if set(rules) != set(REGISTERED_K) or any(not isinstance(r, ContextRule) for r in rules.values()):
        raise ValueError("unexpected frozen context rule schema")
    return rules


def _sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def census_json(census) -> str:
    return json.dumps(census, sort_keys=True, indent=2, allow_nan=False)
