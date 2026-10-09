"""Independent arithmetic over frozen ATT and INNER reader predictions.

This module never imports the fitted-reader runner, selection or scoring code.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from results.pcrl_utility_reference_repair_v1.r3.privacy_math import (
    alias_auc, independent_group_shuffle,
)


def reader_book(y_tune: np.ndarray, tune: Mapping[str, np.ndarray],
                y_inner: np.ndarray, inner: Mapping[str, np.ndarray],
                order: tuple[str, ...]) -> dict:
    if (not order or len(set(order)) != len(order)
            or set(order) != set(tune) or set(order) != set(inner)):
        raise ValueError("frozen reader inventory differs")
    validation = {}
    recovery = {}
    winner = None
    best = -1.0
    for name in order:
        tune_auc = alias_auc(y_tune, tune[name])
        tune_auc["selected_orientation"] = (
            "positive" if tune_auc["positive_alias_auc"] >=
            tune_auc["reverse_alias_auc"] else "reverse")
        validation[name] = tune_auc
        recovery[name] = alias_auc(y_inner, inner[name])
        strength = tune_auc["fixed_alias_max_auc"]
        if strength > best + 1e-12:
            winner, best = name, strength
    inner_selected = recovery[winner]["positive_alias_auc"]
    if validation[winner]["selected_orientation"] == "reverse":
        inner_selected = 1 - inner_selected
    return {
        "validation": validation,
        "readers": recovery,
        "validation_winner": winner,
        "validation_winner_orientation": validation[winner]["selected_orientation"],
        "validation_winner_inner_auc": float(inner_selected),
        "primary_envelope_auc": max(value["fixed_alias_max_auc"]
                                    for value in recovery.values()),
    }


def check_ignore_aliases(coalition: Mapping[str, np.ndarray],
                         local: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    """A coalition alias must carry the exact local prediction, not a refit."""
    if set(local) != {"income", "occupation_group"}:
        raise PermissionError("coalition lacks two local sources")
    for purpose, ignored in (("income", "R2"),
                             ("occupation_group", "R1")):
        for name, values in local[purpose].items():
            alias = f"IGNORE_{ignored}__{name}"
            if (alias not in coalition or
                    not np.array_equal(coalition[alias], values)):
                raise PermissionError("ignore-recipient alias differs from local")


def control_book(y_inner: np.ndarray, group: np.ndarray,
                 shuffled_prediction: np.ndarray,
                 planted_prediction: np.ndarray) -> dict:
    label = np.asarray(y_inner, dtype=np.int64)
    unit = np.asarray(group, dtype=np.int64)
    if label.shape != unit.shape:
        raise ValueError("control label/group alignment differs")
    null_auc = alias_auc(label, np.full(len(label), .5))["fixed_alias_max_auc"]
    shuffled_true = alias_auc(label, shuffled_prediction)["fixed_alias_max_auc"]
    shuffled_independent = alias_auc(
        independent_group_shuffle(label, unit),
        shuffled_prediction)["fixed_alias_max_auc"]
    planted = alias_auc(label, planted_prediction)["fixed_alias_max_auc"]
    gates = {
        "null_exact": null_auc == .5,
        "shuffled_independent_inner_at_most_0.60": shuffled_independent <= .60,
        "planted_at_least_0.99": planted >= .99,
    }
    return {
        "null_auc": null_auc,
        "shuffled_att_true_inner_auc": shuffled_true,
        "shuffled_att_independent_inner_auc": shuffled_independent,
        "planted_auc": planted,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
