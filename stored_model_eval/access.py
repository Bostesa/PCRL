"""Access records, release contracts and the surface / attacker / metric decomposition table.

Access tags (PILOT_PROTOCOL.md / ATTACKER_ACCESS_TABLE.csv; executed set in effective.py):
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
    """What a recipient receives and how often fresh randomness is issued.

    noise:
      none              deterministic release (untreated arms); no randomness
      fresh_per_query   r = h + N(0, sigma^2 I) with a fresh draw on every query
      persistent_token  r = h + N(0, sigma^2 I) drawn once per row and returned on every query
    The pilot derives the contract from each manifest's "release" block (`from_manifest`): a gaussian_noise
    release with release_count "one" is one persistent draw per row per release seed -> persistent_token
    (Addendum D1 #3). Applying the protocol-wide default noise="none" to a noise arm is a bug
    (tests/test_17_pilot_units.py::test_noise_contract_reaches_access_records).
    """
    noise: str = "none"
    sigma: float | None = None     # isotropic noise scale if Sigma not given
    Sigma: np.ndarray | None = field(default=None, compare=False)
    release_count: str | None = None
    seed: int | None = None

    def __post_init__(self):
        if self.noise not in CONTRACTS:
            raise ValueError(f"noise contract must be one of {CONTRACTS}")

    @classmethod
    def from_manifest(cls, release: dict | None) -> "ReleaseContract":
        """Contract from a manifest "release" block (None = untreated deterministic release)."""
        if not release:
            return cls(noise="none", release_count="one")
        if release.get("kind") != "gaussian_noise":
            raise ValueError(f"unsupported release kind {release.get('kind')!r}")
        if release.get("release_count") != "one":
            raise ValueError(f"release_count {release.get('release_count')!r}: only one persistent draw per row "
                             "is supported by this contract")
        sigma = float(release["sigma_abs"])
        if not sigma > 0:
            raise ValueError("gaussian_noise release needs sigma_abs > 0")
        return cls(noise="persistent_token", sigma=sigma, release_count="one",
                   seed=None if release.get("seed") is None else int(release["seed"]))

    @property
    def persistent(self) -> bool:
        return self.noise != "fresh_per_query"

    def issues_fresh_noise(self) -> bool:
        return self.noise == "fresh_per_query"

    def effective_queries(self, N: int) -> int:
        """Number of independent noise draws N queries actually yield."""
        return int(N) if self.issues_fresh_noise() else 1

    def to_json(self):
        return {"noise": self.noise, "sigma": self.sigma,
                "Sigma": None if self.Sigma is None else "declared (matrix)",
                "persistent": self.persistent, "release_count": self.release_count, "seed": self.seed}


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
    status: str = "EXECUTED"       # EXECUTED | REUSED | STAGED_NOT_RUN
    note: str | None = None

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
