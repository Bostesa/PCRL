"""Access records, release contracts and the surface / attacker / metric decomposition table.

Access tags (EVALUATION_PROTOCOL_DRAFT.md section 3):
  A1 release         : (release, S) pairs from the attacker-fit role; one target release
  A2 defense-aware   : A1 + mechanism code + public parameters; simulates the defense on data it holds
  A3(N) repeated     : A2 + N releases of the same target; ONLY where fresh randomness is issued per query
  A4 white-box pop.  : A2 + clean (pre-protection) representations of the attacker population
  A5 insider         : the target's pre-protection representation (stress test; never refutes a guarantee
                       that excludes it)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

ACCESS_TAGS = {
    "A1": "release only: (release, S) pairs from attacker-fit rows; one target release",
    "A2": "defense-aware: A1 + mechanism code + public params; simulates releases on its own population",
    "A3": "repeated query: A2 + N releases of the target; valid only with fresh randomness per query",
    "A4": "white-box population: A2 + clean pre-protection representations of the attacker population",
    "A5": "insider: target's pre-protection representation (stress test only)",
}

CONTRACTS = ("none", "fresh_per_query", "persistent_token")


@dataclass(frozen=True)
class ReleaseContract:
    noise: str = "none"            # none | fresh_per_query | persistent_token
    sigma: float | None = None     # isotropic noise scale if Sigma not given
    Sigma: np.ndarray | None = field(default=None, compare=False)

    def __post_init__(self):
        if self.noise not in CONTRACTS:
            raise ValueError(f"noise contract must be one of {CONTRACTS}")

    def effective_queries(self, N: int) -> int:
        """Number of independent noise draws N queries actually yield."""
        if self.noise == "fresh_per_query":
            return int(N)
        return 1

    def to_json(self):
        return {"noise": self.noise, "sigma": self.sigma,
                "Sigma": None if self.Sigma is None else "declared (matrix)"}


@dataclass
class AccessRecord:
    tag: str
    attacker: str
    surface: str
    fit_inputs: list
    eval_inputs: list
    requires: list = field(default_factory=list)
    contract: dict | None = None
    valid: bool = True
    invalid_reason: str | None = None
    queries: int = 1
    effective_queries: int = 1

    def to_json(self):
        return dict(self.__dict__)


# --------------------------------------------------------------------------------------------------
# decomposition table
# --------------------------------------------------------------------------------------------------

AXES = ("surface", "attacker", "metric")


def decompose(values: dict, baseline: tuple, target: tuple, scales: dict | None = None) -> dict:
    """Attribute value(target) - value(baseline) to surface, attacker and metric changes SEPARATELY.

    values: {(surface, attacker, metric): float} on a common scale (e.g. all AUCs).
    scales: optional {metric: scale-name}; changing between metrics on different scales (R2 vs AUC) is
            refused because the difference would not be meaningful.
    Returns the sequential path surface -> attacker -> metric, all 6 orderings, and Shapley shares
    (average over orderings). Both the sequential steps and the Shapley shares sum exactly to the total.
    """
    if scales is not None and scales.get(baseline[2]) != scales.get(target[2]):
        raise ValueError(f"metric change {baseline[2]} -> {target[2]} crosses scales "
                         f"({scales.get(baseline[2])} vs {scales.get(target[2])}); not decomposable")

    def val(state):
        if state not in values:
            raise KeyError(f"decomposition needs value for {state}")
        return float(values[state])

    total = val(target) - val(baseline)
    orderings = {}
    for order in permutations(range(3)):
        state = list(baseline)
        steps = {}
        prev = val(tuple(state))
        for ax in order:
            state[ax] = target[ax]
            cur = val(tuple(state))
            steps[AXES[ax]] = cur - prev
            prev = cur
        orderings["->".join(AXES[a] for a in order)] = steps
    shap = {ax: float(np.mean([o[ax] for o in orderings.values()])) for ax in AXES}
    seq = orderings["surface->attacker->metric"]
    spread = {ax: (min(o[ax] for o in orderings.values()), max(o[ax] for o in orderings.values()))
              for ax in AXES}
    return {"baseline": list(baseline), "target": list(target), "total": total,
            "sequential_surface_attacker_metric": seq, "shapley": shap,
            "order_dependence_range": spread, "all_orderings": orderings,
            "sum_check": {"sequential": sum(seq.values()) - total, "shapley": sum(shap.values()) - total}}


def decomposition_rows(dec: dict) -> list[dict]:
    """Flat table: one row per axis, reported separately (never folded into one 'improvement')."""
    return [{"axis": ax, "sequential": dec["sequential_surface_attacker_metric"][ax],
             "shapley": dec["shapley"][ax], "range_over_orderings": list(dec["order_dependence_range"][ax])}
            for ax in AXES] + [{"axis": "total", "sequential": dec["total"], "shapley": dec["total"],
                                "range_over_orderings": [dec["total"], dec["total"]]}]
