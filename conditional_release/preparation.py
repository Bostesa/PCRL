"""Common exact task costs, inherited guards and budget translations.

Functions consume supplied allowed-role arrays only. No forbidden-role loader.
"""
from __future__ import annotations
import json
import numpy as np
from .source import pinned, read_exact

FLOOR = 1e-12
NUMERIC = 1e-10
STRICT_CAPS = {"accuracy": .002, "nll": .001, "brier": .0005}
ORIGINAL_CAPS = {"accuracy": .010, "nll": .010, "brier": .005}


def probability_costs(probability, labels):
    p, y = np.asarray(probability), np.asarray(labels)
    if p.dtype != np.float64 or y.dtype != np.int64 or p.ndim not in (2, 3):
        raise ValueError("frozen probability/label dtype differs")
    if p.ndim == 3:
        if y.shape != (len(p),): raise ValueError("action labels not paired")
        yy = np.broadcast_to(y[:, None], p.shape[:-1])
    else:
        if y.shape != (len(p),): raise ValueError("reference labels not paired")
        yy = y
    if np.any(yy < 0) or np.any(yy >= p.shape[-1]) or not np.isfinite(p).all():
        raise ValueError("invalid task cost inputs")
    return {"accuracy": -(p.argmax(-1) == yy).astype(np.float64),
            "nll": -np.log(np.maximum(np.take_along_axis(p, yy[..., None], -1)[..., 0], FLOOR)),
            "brier": ((p-np.eye(p.shape[-1])[yy])**2).sum(-1)}


def task_costs(space, prepared, labels):
    return {f"r{r}_{metric}": value for r, y in enumerate(labels, 1)
            for metric, value in probability_costs(getattr(space, f"probability{r}")[prepared[f"code_r{r}"]], y).items()}


def reference_costs(references, labels):
    return {reference: {f"r{r}_{metric}": value for r, y in enumerate(labels, 1)
                       for metric, value in probability_costs(references[f"{reference}_{r}"], y).items()}
            for reference in ("U0", "Ucal")}


def restore_local_reader_losses(manifest, seed):
    record = manifest["seeds"][str(seed)]["fixed_readers"]
    index = json.loads(pinned(record["index"]).read_text())
    # The archive contains allowed FIT OOF reader losses/model coefficients only.
    # Read every expected key before selecting local guards: no hidden role arrays.
    from zipfile import ZipFile
    with ZipFile(pinned(record["arrays"])) as z:
        keys = {name[:-4] for name in z.namelist() if name.endswith(".npy")}
        if len(keys) != len(z.namelist()): raise PermissionError("non-array reader member")
    arrays = read_exact(record["arrays"], keys)
    from incumbent_view.r2_package import array_sha
    result = {}; catalog = []
    for i, reader in enumerate(index["readers"]):
        loss = arrays[f"r{i}_loss"]
        if array_sha(loss) != reader["loss_sha256"]:
            raise PermissionError("restored reader loss binding differs")
        if reader["view"] in ("r1", "r2"):
            result[reader["name"]] = loss
            catalog.append({"name": reader["name"], "view": reader["view"],
                            "loss_sha256": reader["loss_sha256"], "shape": list(loss.shape)})
    return result, {"source_index_sha256": record["index"]["sha256"],
                    "source_reader_archive_sha256": record["arrays"]["sha256"],
                    "retained_local_readers": catalog, "new_reader_fit_count": 0}


def views(folds):
    f = np.asarray(folds)
    if f.dtype != np.int64 or f.ndim != 1 or set(np.unique(f)) != {0, 1, 2}:
        raise ValueError("three inherited group folds required")
    return [("pooled", np.ones(len(f), bool)), *((f"fold{k}", f == k) for k in range(3))]


def constraint_rows(law, costs_by_key, folds, reference_row_costs, reader_losses, budget):
    """Return generic affine constraints plus public translations.

    Accuracy costs are negative accuracy. Thus all constraints are <= RHS.
    Fixed-reader CE differences must be >= -NUMERIC, retaining source safeguard.
    """
    from .lp import ConstraintRows
    if budget not in ("STRICT", "ORIGINAL"): raise ValueError("budget identity differs")
    names = []; rows = []; rhs = []; translations = []
    for view, mask in views(folds):
        for key, loss in costs_by_key.items():
            metric = key.rsplit("_", 1)[1]
            baseline = np.asarray(loss[:, 0], np.float64)
            intercept, coefficient = law.expected_coefficients(loss-baseline[:, None], mask)
            references = ("P0",) if budget == "STRICT" else ("U0", "Ucal")
            for reference in references:
                cap = (STRICT_CAPS if budget == "STRICT" else ORIGINAL_CAPS)[metric]
                p0_excess = 0. if reference == "P0" else float(np.mean(baseline[mask]-reference_row_costs[reference][key][mask]))
                remaining = cap-p0_excess
                name = f"{budget}/{view}/{key}/{reference}"
                names.append(name); rows.append(coefficient); rhs.append(remaining-intercept)
                translations.append({"budget": budget, "view": view, "task_metric": key,
                                     "reference": reference, "rows": int(mask.sum()),
                                     "cap": cap, "p0_excess": p0_excess, "remaining_rhs": remaining,
                                     "affine_intercept": float(intercept), "identity_pass": bool(p0_excess <= cap),
                                     "negative_rhs": bool(remaining < 0)})
        for name, loss in reader_losses.items():
            intercept, coefficient = law.expected_coefficients(loss-loss[:, 0, None], mask)
            names.append(f"CE_GUARD/{view}/{name}")
            rows.append(-coefficient); rhs.append(NUMERIC+intercept)
    matrix = np.asarray(rows, np.float64)
    return ConstraintRows(tuple(names), matrix, np.asarray(rhs, np.float64)), translations


def expected_rows(prepared, mass, space, labels, references):
    costs = task_costs(space, prepared, labels); refs = reference_costs(references, labels)
    expected = {key: (mass*loss).sum(1) for key, loss in costs.items()}
    return {"expected": expected, "reference": refs,
            "excess": {ref: {key: expected[key]-value for key, value in costs_by_key.items()}
                       for ref, costs_by_key in refs.items()}}
