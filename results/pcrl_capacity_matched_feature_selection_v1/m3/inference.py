"""Selection-aware, development-only paired-group simultaneous inference.

M4 supplies frozen aligned INNER arrays and FIT-selected rule statuses. This
module opens no role file and fits no reader. Every resample recomputes the
fixed-slate oriented reader envelope; all possible strongest-control clauses
share one max-|t| family, so point-selected controls inherit simultaneous
coverage without a bootstrap argmin refit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import isfinite

import numpy as np
from sklearn.metrics import roc_auc_score


SCHEMA = "pcrl-focused-simultaneous-inference-v1"
RULES = ("J", "T", "I", "S12", "S21", "SHARED", "T_U", "I_U", "S12_U", "S21_U")
CONTROLS = RULES[1:]
TASKS = ("income", "occupation_group")
REFERENCES = ("U0", "Ucal", "F")
VIEWS = ("income", "occupation_group", "coalition")
METRICS = ("accuracy", "nll", "brier")
REPLICATES = 2048
INNER_ROWS = 2235
INNER_GROUPS = 2234
STUDENTIZATION_FLOOR = 1e-12


def margin_names():
    utility = tuple(f"{rule}|{task}|{ref}|{metric}"
                    for rule in RULES for task in TASKS
                    for ref in REFERENCES for metric in METRICS)
    privacy = tuple(f"J|vs|{control}|{clause}"
                    for control in CONTROLS
                    for clause in ("coalition_benefit", "income_local_cap",
                                   "occupation_local_cap"))
    return utility + privacy


def _freeze(raw, *, dtype, shape=None):
    observed = np.asarray(raw)
    if observed.dtype != np.dtype(dtype):
        raise ValueError("inference wire array dtype differs from frozen contract")
    value = np.array(observed, copy=True)
    if shape is not None and value.shape != shape:
        raise ValueError("inference array shape differs from registered role")
    if dtype == np.float64 and not np.all(np.isfinite(value)):
        raise ValueError("inference array contains nonfinite values")
    value.setflags(write=False)
    return value


def _digest(array):
    contiguous = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(contiguous.dtype).encode())
    h.update(str(contiguous.shape).encode())
    h.update(contiguous.tobytes())
    return h.hexdigest()


def _probability(raw, n, width):
    value = _freeze(raw, dtype=np.float64, shape=(n, width))
    if np.any(value < 0) or not np.allclose(value.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("inference probability row is invalid")
    return value


@dataclass(frozen=True)
class Prepared:
    names: tuple[str, ...]
    active: np.ndarray
    seeds: dict[int, dict]
    rows: int
    groups: np.ndarray
    reader_count_by_view: dict[str, int]
    invalid_rule_ids: tuple[str, ...]


def prepare(seed_data, plans_by_seed, *, expected_seeds=(0, 1),
            expected_rows=INNER_ROWS, expected_groups=INNER_GROUPS):
    """Bind complete fixed views to FIT rules; intern exact identical arrays."""
    seeds = tuple(expected_seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or set(seed_data) != set(seeds) or set(plans_by_seed) != set(seeds)):
        raise ValueError("paired inference seed records are incomplete")
    names = margin_names()
    frozen = {}
    reference_ids = None
    reference_labels = None
    common_reader_ids = {}
    valid_by_rule = {rule: True for rule in RULES}
    for seed in seeds:
        data = seed_data[seed]
        plan = plans_by_seed[seed]
        if (plan.get("schema") != "pcrl-focused-m3-control-plan-v1"
                or set(plan.get("rules", {})) != set(RULES)
                or set(data) != {"truth", "references", "rules"}
                or set(data["rules"]) != set(RULES)):
            raise ValueError("inference FIT rule/data registry differs from frozen family")
        truth = data["truth"]
        if set(truth) != {"row_id", "unit", "income", "occupation", "sex"}:
            raise ValueError("inference truth identity/label fields differ")
        rid = _freeze(truth["row_id"], dtype=np.int64, shape=(expected_rows,))
        unit = _freeze(truth["unit"], dtype=np.int64, shape=(expected_rows,))
        if np.unique(rid).size != expected_rows or np.unique(unit).size != expected_groups:
            raise ValueError("inference ordered row/group geometry differs")
        if reference_ids is None:
            reference_ids = (rid, unit)
        elif not np.array_equal(reference_ids[0], rid) or not np.array_equal(reference_ids[1], unit):
            raise ValueError("paired seeds are not aligned by row and group")
        labels = {
            name: _freeze(truth[name], dtype=np.int64, shape=(expected_rows,))
            for name in ("income", "occupation", "sex")
        }
        if (np.any((labels["income"] < 0) | (labels["income"] > 1))
                or np.any((labels["occupation"] < 0) | (labels["occupation"] > 5))
                or np.any((labels["sex"] < 0) | (labels["sex"] > 1))):
            raise ValueError("inference task/SEX label range differs")
        if reference_labels is None:
            reference_labels = labels
        elif any(not np.array_equal(reference_labels[name], labels[name])
                 for name in ("income", "occupation", "sex")):
            raise ValueError("paired labels differ across model seeds")
        unique = {}

        def intern(array):
            digest = _digest(array)
            if digest in unique and not np.array_equal(unique[digest], array):
                raise ValueError("inference array hash collision")
            if digest not in unique:
                unique[digest] = array
            return digest

        refs = data["references"]
        if set(refs) != set(TASKS):
            raise ValueError("inference task reference names differ")
        reference_bank = {}
        for task, width in (("income", 2), ("occupation_group", 6)):
            if set(refs[task]) != set(REFERENCES):
                raise ValueError("inference U0/Ucal/F references incomplete")
            reference_bank[task] = {
                ref: intern(_probability(refs[task][ref], expected_rows, width))
                for ref in REFERENCES
            }
        rule_bank = {}
        for rule_id in RULES:
            fitted = plan["rules"][rule_id]
            record = data["rules"][rule_id]
            status = fitted["status"]
            if record.get("status") != status or record.get("pair_id") != fitted["pair_id"]:
                raise ValueError("inference saved rule differs from FIT selection")
            if status != "SELECTED":
                if set(record) != {"status", "pair_id"}:
                    raise ValueError("invalid FIT rule has unregistered outcome arrays")
                valid_by_rule[rule_id] = False
                continue
            if set(record) != {"status", "pair_id", "prob_income", "prob_occupation",
                               "reader_predictions"}:
                raise ValueError("selected rule lacks complete frozen arrays")
            prob = {
                "income": intern(_probability(record["prob_income"], expected_rows, 2)),
                "occupation_group": intern(_probability(
                    record["prob_occupation"], expected_rows, 6)),
            }
            predictions = record["reader_predictions"]
            if set(predictions) != set(VIEWS):
                raise ValueError("inference fixed reader views are incomplete")
            readers = {}
            for view in VIEWS:
                bank = predictions[view]
                if not isinstance(bank, dict) or not bank:
                    raise ValueError("inference reader bank is empty")
                names_here = tuple(sorted(bank))
                if view in common_reader_ids and common_reader_ids[view] != names_here:
                    raise ValueError("inference frozen reader slate narrowed or changed")
                common_reader_ids[view] = names_here
                readers[view] = tuple(intern(_freeze(bank[name], dtype=np.float64,
                                                    shape=(expected_rows,)))
                                      for name in names_here)
            rule_bank[rule_id] = {"prob": prob, "readers": readers}
        frozen[seed] = {"row_id": rid, "unit": unit, "labels": labels,
                        "refs": reference_bank, "rules": rule_bank,
                        "unique": unique}
    if not valid_by_rule["J"]:
        raise ValueError("joint rule invalid on a paired seed; no J inference")
    active = np.ones(len(names), dtype=bool)
    offset = 0
    for rule_id in RULES:
        if not valid_by_rule[rule_id]:
            active[offset:offset + 18] = False
        offset += 18
    for control in CONTROLS:
        if not valid_by_rule[control]:
            active[offset:offset + 3] = False
        offset += 3
    active.setflags(write=False)
    return Prepared(names, active, frozen, expected_rows, reference_ids[1],
                    {view: len(ids) for view, ids in common_reader_ids.items()},
                    tuple(rule for rule in RULES if not valid_by_rule[rule]))


def _task_metrics(prob, labels, task):
    width = 2 if task == "income" else 6
    y = labels
    p = prob
    accuracy = float(np.mean(np.argmax(p, axis=1) == y))
    nll = float(np.mean(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))))
    onehot = np.eye(width, dtype=np.float64)[y]
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    return {"accuracy": accuracy, "nll": nll, "brier": brier}


def _oriented_auc(labels, prediction):
    if np.unique(labels).size != 2:
        raise ValueError("inference resample lacks both SEX classes")
    auc = float(roc_auc_score(labels, prediction))
    return max(auc, 1.0 - auc)


def evaluate(prepared, positions):
    """One paired resample: utility and every fixed-reader envelope."""
    pos = np.asarray(positions, dtype=np.int64)
    if pos.ndim != 1 or not len(pos) or np.any((pos < 0) | (pos >= prepared.rows)):
        raise ValueError("paired resample positions invalid")
    across_seeds = []
    for seed in sorted(prepared.seeds):
        data = prepared.seeds[seed]
        arrays = data["unique"]
        labels = data["labels"]
        metrics_cache = {}
        auc_cache = {}

        def metric(digest, task):
            key = (digest, task)
            if key not in metrics_cache:
                y = labels["income" if task == "income" else "occupation"][pos]
                metrics_cache[key] = _task_metrics(arrays[digest][pos], y, task)
            return metrics_cache[key]

        def auc(digest):
            if digest not in auc_cache:
                auc_cache[digest] = _oriented_auc(labels["sex"][pos], arrays[digest][pos])
            return auc_cache[digest]

        refs = {task: {ref: metric(digest, task)
                       for ref, digest in bank.items()}
                for task, bank in data["refs"].items()}
        stats = {}
        for rule_id, rule in data["rules"].items():
            stats[rule_id] = {
                "task": {task: metric(digest, task)
                         for task, digest in rule["prob"].items()},
                "auc": {view: max(auc(digest) for digest in rule["readers"][view])
                        for view in VIEWS},
            }
        margins = []
        for rule_id in RULES:
            if rule_id not in stats:
                margins.extend((np.nan,) * 18)
                continue
            for task in TASKS:
                candidate = stats[rule_id]["task"][task]
                for ref in REFERENCES:
                    useful = refs[task][ref]
                    margins.extend((
                        candidate["accuracy"] - useful["accuracy"] + 0.010,
                        useful["nll"] + 0.010 - candidate["nll"],
                        useful["brier"] + 0.005 - candidate["brier"],
                    ))
        joint = stats["J"]["auc"]
        for control in CONTROLS:
            if control not in stats:
                margins.extend((np.nan,) * 3)
                continue
            other = stats[control]["auc"]
            margins.extend((
                other["coalition"] - joint["coalition"] - 0.020,
                other["income"] + 0.010 - joint["income"],
                other["occupation_group"] + 0.010 - joint["occupation_group"],
            ))
        across_seeds.append(np.asarray(margins, dtype=np.float64))
    values = np.full(len(prepared.names), np.nan, dtype=np.float64)
    values[prepared.active] = np.mean(np.stack(across_seeds)[:, prepared.active], axis=0)
    if values.shape != (len(prepared.names),) or not np.all(np.isfinite(values[prepared.active])):
        raise ValueError("paired margin vector is incomplete or nonfinite")
    return values


def point_endpoints(prepared, plans_by_seed):
    """Independently recompute M4's point utility and full-reader envelopes."""
    if set(plans_by_seed) != set(prepared.seeds):
        raise ValueError("point endpoint seeds differ from prepared inference")
    records = {}
    for seed in sorted(prepared.seeds):
        data = prepared.seeds[seed]
        plan = plans_by_seed[seed]
        labels = data["labels"]
        arrays = data["unique"]
        references = {
            task: {name: _task_metrics(arrays[digest],
                                       labels["income" if task == "income" else "occupation"],
                                       task)
                   for name, digest in bank.items()}
            for task, bank in data["refs"].items()
        }
        by_rule = {}
        for rule_id in RULES:
            selection = plan["rules"][rule_id]
            if selection["status"] != "SELECTED":
                by_rule[rule_id] = {"status": selection["status"], "pair_id": None}
                continue
            rule = data["rules"][rule_id]
            task_records = {}
            for task, digest in rule["prob"].items():
                candidate = _task_metrics(
                    arrays[digest],
                    labels["income" if task == "income" else "occupation"], task,
                )
                comparisons = {}
                for name, useful in references[task].items():
                    diff = {
                        "accuracy_difference": candidate["accuracy"] - useful["accuracy"],
                        "nll_excess": candidate["nll"] - useful["nll"],
                        "brier_excess": candidate["brier"] - useful["brier"],
                    }
                    comparisons[name] = {
                        **useful, **diff,
                        "point_eligible": (diff["accuracy_difference"] >= -0.010 and
                                           diff["nll_excess"] <= 0.010 and
                                           diff["brier_excess"] <= 0.005),
                    }
                task_records[task] = {"candidate": candidate, "references": comparisons}
            recovery = {
                view: {"primary_envelope_auc": max(
                    _oriented_auc(labels["sex"], arrays[digest])
                    for digest in rule["readers"][view]
                )}
                for view in VIEWS
            }
            by_rule[rule_id] = {
                "seed": seed, "rule_id": rule_id,
                "pair_id": selection["pair_id"],
                "income_mask_id": selection["income_mask_id"],
                "occupation_mask_id": selection["occupation_mask_id"],
                "union_sha256": selection["union_sha256"],
                "recipient_capacity_signatures":
                selection["recipient_capacity_signatures"],
                "union_capacity_signature": selection["union_capacity_signature"],
                "task": task_records, "recovery": recovery,
            }
        records[seed] = by_rule
    return records


def _pcg_seed_integer(private_seed):
    if not isinstance(private_seed, bytes) or len(private_seed) != 32:
        raise ValueError("locked PCG64DXSM seed must contain all 32 bytes")
    return int.from_bytes(private_seed, "big")


def _bootstrap(prepared, private_seed, *, replicates):
    if len(private_seed) != 32 or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("private seed or replicate count invalid")
    ordered = np.unique(prepared.groups)
    positions = {int(group): np.flatnonzero(prepared.groups == group) for group in ordered}
    rng = np.random.Generator(np.random.PCG64DXSM(_pcg_seed_integer(private_seed)))
    point = evaluate(prepared, np.arange(prepared.rows, dtype=np.int64))
    active_indices = np.flatnonzero(prepared.active)
    boot = np.empty((replicates, len(active_indices)), dtype=np.float64)
    for replicate in range(replicates):
        draw = rng.choice(ordered, size=len(ordered), replace=True)
        idx = np.concatenate([positions[int(group)] for group in draw])
        try:
            boot[replicate] = evaluate(prepared, idx)[active_indices]
        except ValueError as error:
            if "both SEX classes" not in str(error):
                raise
            return {"schema": SCHEMA, "status": "INCOMPLETE_UNDEFINED_REPLICATE",
                    "bootstrap_replicates": replicates, "undefined_replicate": replicate,
                    "clauses": [{"name": name,
                                 "status": ("UNRESOLVED_INVALID_FIT_RULE" if not valid
                                            else "UNRESOLVED_UNDEFINED_REPLICATE")}
                                for name, valid in zip(prepared.names, prepared.active)]}
    se = boot.std(axis=0, ddof=1)
    point_active = point[active_indices]
    zero = np.max(np.abs(boot - point_active), axis=0) <= STUDENTIZATION_FLOOR
    se[zero] = 0.0
    scale = np.maximum(se, STUDENTIZATION_FLOOR)
    scaled = np.zeros_like(boot)
    scaled[:, ~zero] = np.abs((boot[:, ~zero] - point_active[~zero]) / scale[~zero])
    critical = float(np.quantile(scaled.max(axis=1), 0.95, method="higher"))
    width = critical * np.where(zero, 0.0, scale)
    clauses = []
    by_index = {int(index): j for j, index in enumerate(active_indices)}
    for i, name in enumerate(prepared.names):
        if i not in by_index:
            clauses.append({"name": name, "status": "UNRESOLVED_INVALID_FIT_RULE"})
            continue
        j = by_index[i]
        lower, upper = point_active[j] - width[j], point_active[j] + width[j]
        clauses.append({
            "name": name, "point_margin": float(point_active[j]),
            "se": float(se[j]), "lower": float(lower), "upper": float(upper),
            "status": ("PASS" if lower >= 0 else "VIOLATION" if upper < 0
                       else "UNRESOLVED"),
            "numerical_se_status": ("NUMERICALLY_CONSTANT" if zero[j] else "VARIABLE"),
        })
    return {
        "schema": SCHEMA, "status": "DEVELOPMENT_SIMULTANEOUS_INTERVALS",
        "bootstrap_replicates": replicates, "inner_groups": len(ordered),
        "critical_family": "two-sided-maximum-absolute-studentized",
        "critical_95": critical, "studentization_floor": STUDENTIZATION_FLOOR,
        "private_seed_sha256": hashlib.sha256(private_seed).hexdigest(),
        "numpy_version": np.__version__, "registered_clause_count": len(prepared.names),
        "active_clause_count": int(prepared.active.sum()),
        "invalid_fit_rule_ids": list(prepared.invalid_rule_ids), "clauses": clauses,
        "interpretation": "development inference conditional on fitted models and exposed rows",
    }


def simultaneous_inference(seed_data, plans_by_seed, private_seed, *, expected_seeds=(0, 1)):
    """Production gate: 2,235 rows, 2,234 groups, 2,048 shared resamples."""
    prepared = prepare(seed_data, plans_by_seed, expected_seeds=expected_seeds)
    return _bootstrap(prepared, private_seed, replicates=REPLICATES)


def selected_control_clauses(inference_result, strongest_registry):
    """Locate point-selected rules inside the already simultaneous full family."""
    if (inference_result.get("schema") != SCHEMA or
            strongest_registry.get("schema") !=
            "pcrl-focused-m3-strongest-controls-v1"):
        raise ValueError("selected-control inference/registry schemas differ")
    clauses = inference_result.get("clauses")
    if (not isinstance(clauses, list) or len(clauses) != len(margin_names()) or
            tuple(item.get("name") for item in clauses) != margin_names()):
        raise ValueError("simultaneous family omitted or reordered a clause")
    by_name = {item["name"]: item for item in clauses}
    result = {"schema": "pcrl-focused-selected-control-intervals-v1",
              "selection_adjustment": "FULL_207_CLAUSE_SIMULTANEOUS_FAMILY"}
    for pool in ("main", "attribution"):
        selected = strongest_registry.get(pool)
        if not isinstance(selected, dict):
            raise ValueError("strongest-control pool missing")
        if selected.get("status") != "SELECTED":
            result[pool] = {"status": selected.get("status", "INCOMPLETE")}
            continue
        rule_id = selected.get("rule_id")
        if rule_id not in CONTROLS:
            raise ValueError("selected strongest rule is outside frozen control slate")
        result[pool] = {
            clause: by_name[f"J|vs|{rule_id}|{clause}"]
            for clause in ("coalition_benefit", "income_local_cap",
                           "occupation_local_cap")
        }
    return result
