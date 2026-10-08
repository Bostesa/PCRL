"""Predeclared engineering checks for the shared complete-view attack path.

The two logistic fits here are numerical jobs. A caller must hold the S3
semaphore and S1's effective science seal before passing Adult role records.
Only aggregates leave private custody. The planted SEX column exists solely
on a separate engineering copy and is never a candidate release.
"""

from __future__ import annotations

import hashlib

import numpy as np

from attack_eval import AttackView, _factory, _label_record, attack_fit_tune_split, oriented_auc


SCHEMA = "pcrl-portfolio-attack-control-probes-v1"
SHUFFLE_SALT = "pcrl-method-portfolio-v1|shuffled-sex-probe|20261008"
NULL_EXPECTED = 0.5
SHUFFLE_CAP = 0.60
PLANTED_FLOOR = 0.99


def groupwise_shuffled_labels(y, unit) -> np.ndarray:
    """Permute whole group label vectors within same-sized group buckets."""
    labels = np.asarray(y, dtype=np.int64)
    groups = np.asarray(unit, dtype=np.int64)
    if labels.shape != groups.shape or labels.ndim != 1:
        raise ValueError("group shuffle row shape mismatch")
    seed = int.from_bytes(hashlib.sha256(SHUFFLE_SALT.encode()).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    by_size = {}
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        by_size.setdefault(len(idx), []).append(idx)
    result = np.empty_like(labels)
    for size in sorted(by_size):
        buckets = by_size[size]
        for target, source_index in zip(buckets, rng.permutation(len(buckets))):
            result[target] = labels[buckets[int(source_index)]]
    return result


def run_control_probes(att_view: AttackView, sex_att: dict,
                       inner_view: AttackView, sex_inner: dict) -> dict:
    att_view.validate()
    inner_view.validate()
    if att_view.components != inner_view.components or \
            att_view.wire_components != inner_view.wire_components or \
            att_view.component_columns != inner_view.component_columns or \
            att_view.wire_constants != inner_view.wire_constants:
        raise ValueError("engineering control view schemas differ")
    y_att = _label_record(sex_att, att_view)
    y_inner = _label_record(sex_inner, inner_view)
    split = attack_fit_tune_split(att_view.unit)
    fit = split["fit"]
    x_att = np.asarray(att_view.x, dtype=np.float64)
    x_inner = np.asarray(inner_view.x, dtype=np.float64)
    if np.unique(y_att[fit]).size != 2 or np.unique(y_inner).size != 2:
        raise ValueError("engineering control needs both SEX classes")

    null = oriented_auc(y_inner, np.full(len(y_inner), 0.5))
    shuffled_y = groupwise_shuffled_labels(y_att[fit], att_view.unit[fit])
    if np.unique(shuffled_y).size != 2:
        raise ValueError("shuffled control lost a SEX class")
    shuffled_model = _factory("LR_C1", 0)
    shuffled_model.fit(x_att[fit], shuffled_y)
    shuffled = oriented_auc(y_inner, shuffled_model.predict_proba(x_inner)[:, 1])

    planted_model = _factory("LR_C1", 0)
    planted_model.fit(np.column_stack((x_att[fit], y_att[fit])), y_att[fit])
    planted = oriented_auc(y_inner, planted_model.predict_proba(np.column_stack((x_inner, y_inner)))[:, 1])
    gates = {"null_exact": null["fixed_alias_max_auc"] == NULL_EXPECTED,
             "shuffled_at_most_0.60": shuffled["fixed_alias_max_auc"] <= SHUFFLE_CAP,
             "planted_at_least_0.99": planted["fixed_alias_max_auc"] >= PLANTED_FLOOR}
    return {"schema": SCHEMA, "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
            "score_role": "INNER_SELECTION", "additional_reader_fits": 2,
            "null_auc": null["fixed_alias_max_auc"],
            "shuffled_auc": shuffled["fixed_alias_max_auc"],
            "planted_auc": planted["fixed_alias_max_auc"],
            "gates": gates, "status": "PASS" if all(gates.values()) else "FAIL"}
