"""Predeclared inner-development comparator and route selection.

Inputs are validation-selected predictor losses from the three global-household
2018 anchor panels. This module has no loader for outer-assessment labels.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from .inference import P_TARGET, TASK_ROLE, WEIGHTINGS
from .audit import ROLES

SENSITIVE_ROLES = tuple(role for role in ROLES if role != TASK_ROLE)
ANCHORS = (0, 1, 2)
GUARD = .001
U_TASK = -.003
P_TARGET_MARGIN = -.002


def inner_validation_from_report(report: Mapping) -> dict:
    """Extract selected-route *validation* scores without opening check losses.

    The caller must verify the saved report against its COMPLETE inventory.
    This deliberately never reads private per-person contributions or the
    `candidate` field, which holds the separate inner-check outcome.
    """
    if (report.get("schema") != "pcrl-adaptive-inner-audit-v1" or
            report.get("selection_role") != "inner_selection" or
            report.get("score_role") != "inner_check" or
            report.get("outer_pool_opened") is not False):
        raise ValueError("registered inner audit report required")
    releases = report.get("releases")
    if not isinstance(releases, Mapping) or not releases:
        raise ValueError("inner report has no releases")
    scores: dict[str, dict] = {}
    h_scores: dict[str, dict[str, float]] = {}
    for release, payload in releases.items():
        role_records = payload.get("roles", {})
        if set(role_records) != set(ROLES):
            raise ValueError("inner report lacks registered role endpoints")
        scores[release] = {}
        for role in ROLES:
            record = role_records[role]
            selected = record["selected_candidate"]
            h_selected = record["H_selected_candidate"]
            candidate = record["candidate_validation_scores"][selected]
            h = record["H_validation_scores"][h_selected]
            scores[release][role] = {weighting: float(candidate[weighting])
                                     for weighting in WEIGHTINGS}
            current_h = {weighting: float(h[weighting]) for weighting in WEIGHTINGS}
            if role in h_scores and h_scores[role] != current_h:
                raise ValueError("shared H validation reference differs across releases")
            h_scores[role] = current_h
    scores["H"] = h_scores
    return scores


def _loss(scores: Mapping, anchor: int, release: str, role: str, weighting: str) -> float:
    value = float(scores[anchor][release][role][weighting])
    if not math.isfinite(value):
        raise ValueError("nonfinite inner validation loss")
    return value


def aggregate_differences(scores: Mapping, candidate: str,
                          comparator: str = "D17") -> dict:
    """Equal-anchor mean of task and recovery contrasts, with fixed signs."""
    if candidate == comparator:
        # Identical release names must score exactly zero independent of
        # floating ratio accumulation or duplicated anchor representations.
        return {"task": {w: 0. for w in WEIGHTINGS},
                "sensitive": {role: {w: 0. for w in WEIGHTINGS}
                              for role in SENSITIVE_ROLES}}
    result = {"task": {}, "sensitive": {role: {} for role in SENSITIVE_ROLES}}
    for weighting in WEIGHTINGS:
        result["task"][weighting] = sum(
            _loss(scores, anchor, candidate, TASK_ROLE, weighting) -
            _loss(scores, anchor, comparator, TASK_ROLE, weighting)
            for anchor in ANCHORS) / len(ANCHORS)
        for role in SENSITIVE_ROLES:
            result["sensitive"][role][weighting] = sum(
                _loss(scores, anchor, comparator, role, weighting) -
                _loss(scores, anchor, candidate, role, weighting)
                for anchor in ANCHORS) / len(ANCHORS)
    return result


def point_eligible(differences: Mapping, route: str) -> bool:
    """Apply the registered point screen; inferential bounds remain separate."""
    task = [differences["task"][w] for w in WEIGHTINGS]
    sensitive = [differences["sensitive"][role][w]
                 for role in SENSITIVE_ROLES for w in WEIGHTINGS]
    if route == "U":
        return max(task) <= U_TASK and max(sensitive) <= GUARD
    if route == "P":
        target = [differences["sensitive"][P_TARGET][w] for w in WEIGHTINGS]
        return (max(task) <= GUARD and max(sensitive) <= GUARD and
                max(target) <= P_TARGET_MARGIN)
    raise ValueError("route must be U or P")


def _ranking(differences: Mapping, route: str, stable_id: str) -> tuple:
    task = [differences["task"][w] for w in WEIGHTINGS]
    sensitive = [differences["sensitive"][role][w]
                 for role in SENSITIVE_ROLES for w in WEIGHTINGS]
    if route == "U":
        return (max(task), sum(task)/2, max(sensitive), stable_id)
    if route == "P":
        target = [differences["sensitive"][P_TARGET][w] for w in WEIGHTINGS]
        return (max(target), sum(target)/2, max(task), stable_id)
    raise ValueError("route must be U or P")


def select_family_representative(scores: Mapping, family: str,
                                 release_ids: Sequence[str], route: str,
                                 *, alias_of: Mapping[str, str] | None = None) -> dict:
    """Lock one independently constructed family choice, including D17 fallback.

    The P-family feasibility screen is weaker than the P *success* target:
    it requires no measured harm above +.001, then ranks by AB/SEX recovery.
    """
    if not family or not isinstance(family, str):
        raise ValueError("baseline family name required")
    for anchor in ANCHORS:
        for role in ROLES:
            for weighting in WEIGHTINGS:
                _loss(scores, anchor, "D17", role, weighting)
    aliases = dict(alias_of or {})
    named = ["D17", *release_ids]
    canonical_to_names: dict[str, list[str]] = {}
    for name in named:
        canonical = aliases.get(name, name)
        if not name or not canonical:
            raise ValueError("nonempty release and canonical names required")
        canonical_to_names.setdefault(canonical, []).append(name)
    candidates = []
    for canonical, names in canonical_to_names.items():
        differences = aggregate_differences(scores, canonical)
        task = [differences["task"][w] for w in WEIGHTINGS]
        sensitive = [differences["sensitive"][role][w]
                     for role in SENSITIVE_ROLES for w in WEIGHTINGS]
        eligible = (max(sensitive) <= GUARD if route == "U" else
                    max(task) <= GUARD and max(sensitive) <= GUARD)
        candidates.append({"id": canonical, "names": sorted(set(names)),
                           "eligible": eligible,
                           "differences_vs_D17": differences,
                           "rank": _ranking(differences, route, canonical)})
    eligible = [item for item in candidates if item["eligible"]]
    if not eligible:
        raise ValueError("D17 fallback contradiction: no eligible family choice")
    chosen = min(eligible, key=lambda item: item["rank"])
    return {"family": family, "route": route,
            "selected": chosen["id"], "selected_names": chosen["names"],
            "ranking_rule": ("max/mean task, max sensitive, ID" if route == "U"
                             else "max/mean AB/SEX recovery, max task, ID"),
            "points": candidates,
            "anchor_aggregation": "equal mean of three anchor-specific differences",
            "score_resource": "inner_selection validation-selected fitted predictors",
            "D17_fallback_included": True}


def select_route_candidate(scores: Mapping, release_ids: Sequence[str],
                           route: str) -> dict:
    """Choose a point-screen nominee, or one explicitly diagnostic best point."""
    if not release_ids:
        raise ValueError("candidate family is empty")
    points = []
    for release in sorted(set(release_ids)):
        differences = aggregate_differences(scores, release)
        benefit = {weighting: sum(
            _loss(scores, anchor, "H", TASK_ROLE, weighting) -
            _loss(scores, anchor, release, TASK_ROLE, weighting)
            for anchor in ANCHORS) / len(ANCHORS)
            for weighting in WEIGHTINGS}
        capability = min(benefit.values()) >= .01
        points.append({"id": release, "point_eligible_vs_D17":
                       point_eligible(differences, route) and capability,
                       "residence_benefit_over_H": benefit,
                       "capability_point_eligible": capability,
                       "differences_vs_D17": differences,
                       "rank": _ranking(differences, route, release)})
    nominees = [item for item in points if item["point_eligible_vs_D17"]]
    selected = min(nominees or points, key=lambda item: item["rank"])
    return {"route": route, "selected": selected["id"],
            "status": "NOMINATED_POINT_SCREEN" if nominees else "DIAGNOSTIC_ONLY",
            "points": points,
            "score_resource": "inner_selection validation-selected fitted predictors"}
