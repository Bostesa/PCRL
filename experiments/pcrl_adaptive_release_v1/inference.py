"""Locked arm-specific endpoint enumeration and paired household inference.

All endpoints describe repeatedly used 2018 development data. Sensitive
recovery difference is CE(comparator) - CE(candidate); lower is favorable.
The family must be enumerated and pinned before assessment rows are opened.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from experiments.pcrl_task_aligned_cuts_v1 import inference as paired
from .audit import ROLES


TASK_ROLE = "utility:A/same_residence"
WEIGHTINGS = ("U", "PWGTP")
THRESHOLDS = {
    "U": {"task": -.003, "target": None, "guard": .001},
    "P": {"task": .001, "target": -.002, "guard": .001},
}
P_TARGET = "attack:AB/SEX"


def enumerate_endpoints(slots: Sequence[Mapping], *,
                        alias_of: Mapping[str, str] | None = None) -> list[dict]:
    """Derive all declared comparisons, roles, weights and signs from slots."""
    if not 1 <= len(slots) <= 2:
        raise ValueError("one or two locked candidate slots required")
    aliases = dict(alias_of or {})
    endpoints: list[dict] = []
    candidates: set[str] = set()
    for slot in slots:
        candidate, arm = slot["id"], slot["arm"]
        if not isinstance(candidate, str) or not candidate or candidate in candidates:
            raise ValueError("candidate IDs must be unique nonempty strings")
        if arm not in THRESHOLDS:
            raise ValueError("unknown adaptive-release arm")
        candidates.add(candidate)
        named = list(slot["comparators"])
        if not named or any(not isinstance(name, str) or not name for name in named):
            raise ValueError("nonempty registered comparator names required")
        groups: dict[str, list[str]] = {}
        for name in named:
            canonical = aliases.get(name, name)
            if not isinstance(canonical, str) or not canonical:
                raise ValueError("invalid comparator alias")
            groups.setdefault(canonical, []).append(name)
        for comparator, names in groups.items():
            for role in ROLES:
                clause = ("task" if role == TASK_ROLE else
                          "target" if arm == "P" and role == P_TARGET else "guard")
                threshold = THRESHOLDS[arm][clause]
                for weighting in WEIGHTINGS:
                    endpoints.append({
                        "id": f"{candidate}|{comparator}|{role}|{weighting}",
                        "candidate": candidate, "arm": arm,
                        "comparator": comparator,
                        "comparator_names": sorted(set(names)),
                        "role": role, "weighting": weighting,
                        "clause": clause, "threshold": threshold,
                        "plus": candidate if clause == "task" else comparator,
                        "minus": comparator if clause == "task" else candidate,
                        "orientation": ("CE_candidate - CE_comparator" if clause == "task" else
                                        "CE_comparator - CE_candidate = recovery_candidate - recovery_comparator"),
                    })
    if len({item["id"] for item in endpoints}) != len(endpoints):
        raise ValueError("endpoint IDs collided")
    return endpoints


def family_manifest(slots: Sequence[Mapping], *,
                    alias_of: Mapping[str, str] | None = None) -> dict:
    endpoints = enumerate_endpoints(slots, alias_of=alias_of)
    return {
        "schema": 1, "n_endpoints": len(endpoints), "endpoints": endpoints,
        "multiplicity": "two-sided Bonferroni over exact frozen endpoint IDs",
        "sampling_unit": "household; common resample across roles, releases and anchors",
        "anchor_aggregation": "equal mean of anchor-specific weighted ratios; anchors share households",
        "scope": "2018 development; conditional on frozen predictor and candidate selection",
    }


capability_endpoints = paired.capability_endpoints
contrasts_from_scores = paired.contrasts_from_scores
evaluate_family = paired.evaluate_family
