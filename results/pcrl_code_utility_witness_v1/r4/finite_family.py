"""Prospective signed-clause order spanning both hard-code capacities.

This module enumerates IDs only. It reads no role, result, label or endpoint.
R0 decides admission and seals these exact bytes before any INNER-scored fit.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
import json


CAPACITIES = (10, 16)
CAPACITY_ARM_BASE = ("T0", "TPLUS", "L", "J", "S12", "S21", "C")
ARMS = tuple(f"{arm}_B{bits}" for bits in CAPACITIES for arm in CAPACITY_ARM_BASE) + (
    "W_Q", "W_TSTAR",
)
REFERENCES = ("F", "U0", "Ucal")
SUMMARIES = ("s0", "s1", "mean")
TASKS = ("income", "occupation")
METRICS = ("accuracy", "nll", "brier")
VIEWS = ("r1", "r2", "pair")


def ordered_clause_ids() -> tuple[str, ...]:
    """Every arm/reference utility and signed complete-view recovery contrast.

    Recovery values are raw first-minus-second AUCs. A coalition benefit is
    control-minus-candidate and a local worsening is candidate-minus-control;
    both are derived by sign from these registered contrasts. F/U anchors are
    diagnostic when their disclosure contracts differ, never eligible matched
    controls solely because they appear in this family.
    """
    base = []
    for arm in ARMS:
        for task in TASKS:
            for ref in REFERENCES:
                for metric in METRICS:
                    base.append(f"utility:{arm}:{task}:{ref}:{metric}")
    for first, second in combinations(ARMS, 2):
        for view in VIEWS:
            base.append(f"arm_pair:{first}:{second}:{view}:auc")
    for arm in ARMS:
        for ref in REFERENCES:
            for view in VIEWS:
                base.append(f"ref_pair:{arm}:{ref}:{view}:auc")
    names = tuple(f"{summary}:{clause}" for summary in SUMMARIES for clause in base)
    if len(names) != 2376 or len(set(names)) != len(names):
        raise AssertionError("prospective finite clause dimension differs")
    return names


def family_manifest() -> dict:
    names = ordered_clause_ids()
    encoded = json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "schema": "pcrl-code-witness-finite-family-v1",
        "status": "DATA_FREE_PROPOSAL_FOR_R0_SEAL",
        "capacities": CAPACITIES,
        "capacity_arm_base": CAPACITY_ARM_BASE,
        "arms": ARMS,
        "references": REFERENCES,
        "summaries": SUMMARIES,
        "tasks": TASKS,
        "metrics": METRICS,
        "views": VIEWS,
        "clause_count": len(names),
        "per_summary_counts": {
            "utility": len(ARMS) * len(TASKS) * len(REFERENCES) * len(METRICS),
            "arm_pair_auc": len(ARMS) * (len(ARMS) - 1) // 2 * len(VIEWS),
            "ref_pair_auc": len(ARMS) * len(REFERENCES) * len(VIEWS),
        },
        "ordered_ids_sha256": hashlib.sha256(encoded).hexdigest(),
        "contrast_convention": "raw first-minus-second; apply frozen direction and caps",
        "missing_slot_convention": "NOT_ESTIMABLE_DEPENDENCY with null point and bounds; retain ID",
        "ordered_ids": names,
    }


if __name__ == "__main__":
    print(json.dumps(family_manifest(), sort_keys=True, indent=2) + "\n", end="")
