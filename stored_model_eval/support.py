"""Class / pair support frozen from counts in ALL three roles before any fit or inference.

Rule (FROZEN_DESIGN): class k is supported iff it has >= 100 rows in attacker_fit, >= 30 in attacker_val and
>= 100 in assessment. A pair is supported iff both classes are supported. Fewer than 2 supported classes makes
the unit NE (not estimable). Unsupported classes keep reason codes; coverage is reported. NE is distinct from
UNRESOLVED (an estimable quantity whose interval straddles the bar).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

ROLE_KEYS = (("attacker_fit", "min_attacker_fit"), ("attacker_val", "min_attacker_val"),
             ("assessment", "min_assessment"))


def freeze_support(y, roles, rule, n_classes: int | None = None, what: str = "sensitive") -> dict:
    """y: labels for all rows; roles: role name per row; rule: mapping with min_attacker_fit/val/assessment and
    min_supported_classes. Returns the JSON-ready support record."""
    y = np.asarray(y).astype(np.int64)
    roles = np.asarray(roles).astype(str)
    K = int(n_classes if n_classes is not None else y.max() + 1)
    counts = {r: np.bincount(y[roles == r], minlength=K)[:K].tolist() for r, _ in ROLE_KEYS}
    thresholds = {r: int(rule[k]) for r, k in ROLE_KEYS}
    classes, reasons = [], {}
    for k in range(K):
        why = [f"{r}<{thresholds[r]} (n={counts[r][k]})" for r, _ in ROLE_KEYS if counts[r][k] < thresholds[r]]
        if why:
            reasons[str(k)] = {"code": "class_below_support", "detail": why,
                               "counts": {r: counts[r][k] for r, _ in ROLE_KEYS}}
        else:
            classes.append(k)
    pairs = [list(p) for p in combinations(classes, 2)]
    all_pairs = list(combinations(range(K), 2))
    ne = len(classes) < int(rule["min_supported_classes"])
    n_assess = int((roles == "assessment").sum())
    row_cov = float(np.isin(y[roles == "assessment"], classes).mean()) if n_assess else 0.0
    return {"what": what, "n_classes": K, "thresholds": thresholds, "counts_per_role": counts,
            "supported_classes": classes, "supported_pairs": pairs,
            "unsupported_classes": reasons,
            "unsupported_pairs": [list(p) for p in all_pairs if list(p) not in pairs],
            "status": "NE" if ne else "ESTIMABLE",
            "ne_reason": "fewer_than_2_supported_classes" if ne else None,
            "coverage": {"classes": [len(classes), K], "pairs": [len(pairs), len(all_pairs)],
                         "assessment_row_fraction_in_supported_classes": row_cov}}
