"""PROTOCOL section 6: nomination on inner data only.

Contrasts are equal three-anchor means of **inner_check** scores of the frozen
validation-selected routes (the `candidate` / `H` fields of each role record in
`audit_panel` reports), not validation minima as in AR. The contrast, screen and
ranking arithmetic is AR's (`AR/selection.aggregate_differences`,
`point_eligible`, `_ranking`), which matches the protocol:

* U: task <= -.003 in both weightings, all 8 recovery <= +.001, H capability
  (CE_Y(H) - CE_Y(cand)) >= .01 in both; key (max task, mean task, max recovery, id).
* P: task <= +.001, AB/SEX recovery <= -.002 (both), other 6 <= +.001 (AR's
  all-8 <= .001 check is implied), H capability >= .01; key (max AB/SEX,
  mean AB/SEX, max task, id).

Both slots choose among the four NM units; if none passes, the rank-minimum of
all four is nominated with label DIAGNOSTIC_ONLY. The ADV representative is
the rank-minimum of ADV_B1/ADV_B2 (U; task-selected checkpoints) or
ADV_B1_P/ADV_B2_P (P; privacy-selected checkpoints, amendment M3) under the slot key among eligible ones (U:
max recovery <= .001; P: max task <= .001), else the rank-minimum overall.
Aliases collapse only when identical on all three anchors (AR rule).
No outer row or outer score is read here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from experiments.pcrl_adaptive_release_v1 import selection as ar_selection
from experiments.pcrl_adaptive_release_v1.inference import P_TARGET, TASK_ROLE, WEIGHTINGS

from . import audit_panel

ANCHORS = (0, 1, 2)
ROLES = audit_panel.ar_audit.ROLES
SENSITIVE_ROLES = ar_selection.SENSITIVE_ROLES
NM_CANDIDATES = ("NM1_U", "NM1_P", "NM4_U", "NM4_P")
# Amendment M3: route-specific ADV checkpoint rules.
ADV_BY_ROUTE = {"U": ("ADV_B1", "ADV_B2"), "P": ("ADV_B1_P", "ADV_B2_P")}
ADV_NAMES = (*ADV_BY_ROUTE["U"], *ADV_BY_ROUTE["P"])
FIXED_COMPARATORS = ("D17", "RD_TASK", "RD_PRIV")
REQUIRED_NAMES = (*NM_CANDIDATES, *FIXED_COMPARATORS, *ADV_NAMES)
GUARD = ar_selection.GUARD
CAPABILITY = .01
SCHEMA = "pcrl-sc-inner-selection-v1"
SCORE_RESOURCE = "inner_check scores of the frozen inner_selection-validated routes"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_scores_from_report(report: Mapping) -> dict:
    """{logical name: {role: {w: CE}}} plus "H", from one anchor's inner report."""
    if (report.get("schema") != audit_panel.REPORT_SCHEMA or
            report.get("selection_role") != "inner_selection" or
            report.get("score_role") != "inner_check" or
            report.get("outer_pool_opened") is not False):
        raise ValueError("registered shared-context inner audit report required")
    releases, mapping = report.get("releases"), report.get("logical_to_canonical")
    if not isinstance(releases, Mapping) or not releases or not isinstance(mapping, Mapping):
        raise ValueError("inner report lacks releases or the logical alias map")
    canonical, h_scores = {}, {}
    for release, payload in releases.items():
        records = payload.get("roles", {})
        if set(records) != set(ROLES):
            raise ValueError(f"{release}: inner report lacks registered roles")
        canonical[release] = {}
        for role in ROLES:
            cand = {w: float(records[role]["candidate"][w]) for w in WEIGHTINGS}
            h = {w: float(records[role]["H"][w]) for w in WEIGHTINGS}
            if not all(math.isfinite(v) for v in (*cand.values(), *h.values())):
                raise ValueError("nonfinite inner_check score")
            if role in h_scores and h_scores[role] != h:
                raise ValueError("shared H inner_check reference differs across releases")
            canonical[release][role], h_scores[role] = cand, h
    named = {name: canonical[target] for name, target in mapping.items()
             if target in canonical}
    if set(mapping) - set(named):
        raise ValueError("a logical name maps to an unscored canonical release")
    named["H"] = h_scores
    return named


def global_aliases(mappings: Mapping[int, Mapping[str, str]], names: Sequence[str]) -> dict[str, str]:
    """AR rule: collapse names only if their canonical law agrees on all three anchors."""
    if set(mappings) != set(ANCHORS):
        raise ValueError("three anchor alias maps required")
    groups: dict[tuple, list[str]] = {}
    for name in sorted(set(names)):
        vector = tuple(mappings[a].get(name) for a in ANCHORS)
        if any(v is None for v in vector):
            raise ValueError(f"logical release {name} absent on an anchor")
        groups.setdefault(vector, []).append(name)
    aliases = {}
    for group in groups.values():
        representative = min(group, key=lambda n: (n != "D17", n))
        aliases.update({n: representative for n in group if n != representative})
    return aliases


def _max_recovery(differences: Mapping) -> float:
    return max(differences["sensitive"][r][w] for r in SENSITIVE_ROLES for w in WEIGHTINGS)


def adv_representative(scores: Mapping, route: str) -> dict:
    points = []
    for name in ADV_BY_ROUTE[route]:
        differences = ar_selection.aggregate_differences(scores, name)
        task = max(differences["task"][w] for w in WEIGHTINGS)
        eligible = (_max_recovery(differences) <= GUARD if route == "U" else task <= GUARD)
        points.append({"id": name, "eligible": eligible, "differences_vs_D17": differences,
                       "rank": list(ar_selection._ranking(differences, route, name))})
    pool = [p for p in points if p["eligible"]] or points
    chosen = min(pool, key=lambda p: tuple(p["rank"]))
    return {"route": route, "selected": chosen["id"],
            "status": "ELIGIBLE_RANK_MIN" if any(p["eligible"] for p in points) else "RANK_MIN_OVERALL",
            "eligibility": ("max recovery <= .001" if route == "U" else "max task <= .001"),
            "points": points}


def nominate(scores: Mapping, route: str) -> dict:
    """One nominee per route among the four NM units (AR select_route_candidate)."""
    chosen = ar_selection.select_route_candidate(scores, NM_CANDIDATES, route)
    for point in chosen["points"]:
        point["rank"] = list(point["rank"])
        # The protocol's P screen spelled out (equivalent to AR's check).
        d = point["differences_vs_D17"]
        point["screen_detail"] = {
            "max_task": max(d["task"][w] for w in WEIGHTINGS),
            "max_recovery": _max_recovery(d),
            "max_AB_SEX_recovery": max(d["sensitive"][P_TARGET][w] for w in WEIGHTINGS),
            "min_H_capability": min(point["residence_benefit_over_H"].values())}
    chosen["score_resource"] = SCORE_RESOURCE
    return chosen


def select_inner(reports: Mapping[int, Mapping]) -> dict:
    """Both routes from three verified anchor reports; inner data only."""
    if set(reports) != set(ANCHORS):
        raise ValueError("three anchor inner reports required")
    scores = {a: check_scores_from_report(reports[a]) for a in ANCHORS}
    for a in ANCHORS:
        missing = sorted(set(REQUIRED_NAMES) - set(scores[a]))
        if missing:
            raise ValueError(f"anchor {a} lacks required releases {missing}")
    mappings = {a: dict(reports[a]["logical_to_canonical"]) for a in ANCHORS}
    aliases = global_aliases(mappings, sorted(set.intersection(*(set(m) for m in mappings.values()))))
    routes = {}
    for route in ("U", "P"):
        nominee = nominate(scores, route)
        adv = adv_representative(scores, route)
        routes[route] = {"nominee": nominee["selected"], "label": nominee["status"],
                         "nomination": nominee, "adv_representative": adv["selected"],
                         "adv_selection": adv,
                         "comparators": [*FIXED_COMPARATORS, adv["selected"]]}
    return {"schema": SCHEMA, "study": audit_panel.STUDY,
            "outer_labels_accessed": False, "score_resource": SCORE_RESOURCE,
            "anchor_aggregation": "equal mean of three anchor-specific differences",
            "candidates": list(NM_CANDIDATES), "routes": routes, "alias_of": aliases,
            "logical_to_canonical": {str(a): mappings[a] for a in ANCHORS}}


def load_panel(panel_dir: str | Path) -> tuple[dict, dict]:
    """Verified completed inner panel -> (report, pins)."""
    root = Path(panel_dir)
    report = audit_panel.verify_complete(root)
    return report, {"panel_dir": str(root.resolve()),
                    "inner_panel_complete_sha256": _sha(root / "COMPLETE.json"),
                    "inner_audit_sha256": _sha(root / "INNER_AUDIT.json"),
                    "index_sha256": report["index_sha256"]}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    inner = sub.add_parser("inner")
    inner.add_argument("--reports", nargs=3, required=True, help="completed inner panel dirs a0 a1 a2")
    inner.add_argument("--out", required=True, help="INNER_SELECTION.json (write-once)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("INNER_SELECTION.json is write-once")
    reports, pins = {}, {}
    for anchor, directory in zip(ANCHORS, args.reports):
        reports[anchor], pins[str(anchor)] = load_panel(directory)
        if reports[anchor]["anchor"] != anchor:
            raise ValueError(f"panel {directory} is not anchor {anchor}")
    if len({p["index_sha256"] for p in pins.values()}) != 1:
        raise ValueError("inner panels used different input indexes")
    result = select_inner(reports)
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), panels=pins,
                  source_sha256={"selection.py": _sha(__file__),
                                 "AR/selection.py": _sha(ar_selection.__file__)})
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(out)
    print(json.dumps({route: {"nominee": r["nominee"], "label": r["label"],
                              "adv": r["adv_representative"]}
                      for route, r in result["routes"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
