"""Cluster bootstrap over declared grouping units and the interval decision rule.

* Sampling unit = the declared linkage unit (household / patient / identity). Rows that are duplicated
  records (same record key) are merged into ONE unit even if they carry different unit ids (union-find),
  so duplication can never inflate the number of independent units.
* Seeds are refit replicates over the same evaluation units: each bootstrap replicate draws units once
  and evaluates every seed's statistic on the same draw (paired); the headline is the seed-mean and the
  between-seed spread is reported separately. Seeds never add sampling units.
* Selection statistics (worst pair, worst class) are re-selected inside every replicate (bootstrap of the
  max), never by bootstrapping a pair chosen on the full sample.
* Decision vs a bar: UCB < bar -> ESTABLISHED_BELOW; LCB > bar -> ESTABLISHED_ABOVE; else UNRESOLVED;
  a non-estimable statistic -> NOT_ESTIMABLE. There is deliberately no "pass" outcome.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .metrics import NotEstimable, is_estimable, pairwise_auc, worst_class_auc, macro_ovr_auc

DECISIONS = ("ESTABLISHED_BELOW", "ESTABLISHED_ABOVE", "UNRESOLVED", "NOT_ESTIMABLE")


def resolve_units(unit_ids: Sequence, record_keys: Sequence | None = None) -> dict:
    """Map rows to independent units. Rows sharing a unit id OR a record key are one unit."""
    unit_ids = np.asarray(unit_ids)
    n = len(unit_ids)
    _, u_inv = np.unique(unit_ids, return_inverse=True)
    parent = list(range(int(u_inv.max()) + 1 if n else 0))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    merges = 0
    if record_keys is not None:
        record_keys = np.asarray(record_keys)
        first = {}
        for r in range(n):
            k = record_keys[r].item() if hasattr(record_keys[r], "item") else record_keys[r]
            u = int(u_inv[r])
            if k in first:
                a, b = find(first[k]), find(u)
                if a != b:
                    parent[b] = a
                    merges += 1
            else:
                first[k] = u
    roots = np.array([find(int(u)) for u in u_inv]) if n else np.zeros(0, int)
    _, unit_index = np.unique(roots, return_inverse=True)
    n_units = int(unit_index.max()) + 1 if n else 0
    return {"unit_index": unit_index, "n_units": n_units, "n_rows": n,
            "n_declared_units": int(len(parent)), "merged_by_record_key": merges,
            "n_duplicate_rows": int(n - len(np.unique(record_keys))) if record_keys is not None else None}


def _draw_weights(rng, unit_index, n_units):
    counts = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units))
    return counts[unit_index].astype(np.float64)


def cluster_bootstrap(stat_fns: Callable | Sequence[Callable], unit_index: np.ndarray, n_boot: int = 2000,
                      seed: int = 0, alpha: float = 0.05, max_not_estimable_frac: float | None = None) -> dict:
    """Percentile cluster bootstrap. stat_fns: one callable (weights -> float|NotEstimable) or a list of
    callables, one per refit seed (headline = mean over seeds on the same unit draw)."""
    fns = [stat_fns] if callable(stat_fns) else list(stat_fns)
    unit_index = np.asarray(unit_index)
    n_units = int(unit_index.max()) + 1
    ones = np.ones(len(unit_index))
    point_seeds = [f(ones) for f in fns]
    if not all(is_estimable(p) for p in point_seeds):
        bad = [p.to_json() for p in point_seeds if not is_estimable(p)]
        return {"point": NotEstimable("point estimate not estimable", counts={"per_seed": bad}),
                "interval": None, "n_units": n_units}
    point = float(np.mean(point_seeds))
    rng = np.random.default_rng(seed)
    reps, n_ne = [], 0
    for _ in range(n_boot):
        w = _draw_weights(rng, unit_index, n_units)
        vals = [f(w) for f in fns]
        if all(is_estimable(v) for v in vals):
            reps.append(float(np.mean(vals)))
        else:
            n_ne += 1
    limit = alpha / 2 if max_not_estimable_frac is None else max_not_estimable_frac
    out = {"point": point, "n_units": n_units, "n_rows": int(len(unit_index)), "n_boot": n_boot,
           "alpha": alpha, "replicates_not_estimable": n_ne, "seed": seed,
           "n_refit_seeds": len(fns)}
    if len(fns) > 1:
        out["refit_replicates"] = {"per_seed_point": [float(p) for p in point_seeds],
                                   "between_seed_sd": float(np.std(point_seeds, ddof=1)),
                                   "note": "seeds are refit replicates on shared units, not sample units"}
    if n_ne / n_boot > limit or not reps:
        out["interval"] = NotEstimable("too many non-estimable bootstrap replicates",
                                       counts={"n_not_estimable": n_ne, "n_boot": n_boot})
        return out
    reps = np.asarray(reps)
    out["interval"] = (float(np.quantile(reps, alpha / 2)), float(np.quantile(reps, 1 - alpha / 2)))
    out["boot_sd"] = float(reps.std(ddof=1))
    return out


def decide(interval, bar: float) -> str:
    if interval is None or not is_estimable(interval):
        return "NOT_ESTIMABLE"
    lo, hi = interval
    if hi < bar:
        return "ESTABLISHED_BELOW"
    if lo > bar:
        return "ESTABLISHED_ABOVE"
    return "UNRESOLVED"


def decide_all(result: dict, bars: Sequence[float]) -> dict:
    if not is_estimable(result.get("point")):
        return {f"{b:.2f}": "NOT_ESTIMABLE" for b in bars}
    return {f"{b:.2f}": decide(result.get("interval"), b) for b in bars}


# ---- statistic factories (closures over fixed arrays, weights supplied by the bootstrap) ----------

def macro_auc_stat(y, P, min_support: int):
    return lambda w: macro_ovr_auc(y, P, min_support, w)


def worst_pair_stat(y, P, min_support: int):
    """Bootstrap-of-max: the set of supported pairs is fixed on the full sample (support rule), the
    argmax is re-selected in every replicate."""
    return lambda w: pairwise_auc(y, P, min_support, w)["max"]


def worst_class_stat(y, P, min_support: int):
    return lambda w: worst_class_auc(y, P, min_support, w)["value"]


def naive_row_bootstrap_sd(stat_fn: Callable, n_rows: int, n_boot: int = 500, seed: int = 0) -> float:
    """Row-level bootstrap SD (treats every row as independent). Used only to demonstrate what
    duplicated records do to a naive interval; never used for decisions."""
    return float(cluster_bootstrap(stat_fn, np.arange(n_rows), n_boot=n_boot, seed=seed)["boot_sd"])


def permutation_null(stat_of_labels: Callable, y: np.ndarray, unit_index: np.ndarray, n_perm: int = 200,
                     seed: int = 0) -> dict:
    """Permutation null with attackers held fixed: labels are permuted at the UNIT level (all rows of a unit
    move together; a unit carrying more than one label raises, because unit-level permutation is undefined), the
    same statistic (incl. any max over pairs/classes) is recomputed, and the one-sided p-value
    (1 + #{null >= observed}) / (1 + n_perm) is returned with null quantiles."""
    y = np.asarray(y)
    unit_index = np.asarray(unit_index)
    n_units = int(unit_index.max()) + 1
    first = np.full(n_units, -1)
    for r, u in enumerate(unit_index):
        if first[u] < 0:
            first[u] = r
    unit_label = y[first]
    if not np.array_equal(unit_label[unit_index], y):
        raise ValueError("units carry more than one label; unit-level permutation undefined")
    obs = stat_of_labels(y)
    if not is_estimable(obs):
        return {"observed": obs, "p_value": NotEstimable("observed statistic not estimable")}
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        v = stat_of_labels(rng.permutation(unit_label)[unit_index])
        if is_estimable(v):
            null.append(float(v))
    null = np.asarray(null)
    return {"observed": float(obs), "n_perm": n_perm, "n_estimable": int(len(null)),
            "p_value": float((1 + (null >= obs).sum()) / (1 + len(null))),
            "null_q95": float(np.quantile(null, 0.95)), "null_mean": float(null.mean())}
