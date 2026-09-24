"""PROTOCOL section 7: generate, count and hash the endpoint families; write the lock.

`SELECTION_LOCK.json` (write-once) holds the two nominee slots with their
labels, the comparator representatives, three code-generated families (primary,
secondary, H-capability) with a SHA-256 of each canonical manifest, the code
commit, the inner report hashes, per-anchor logical->canonical maps and the
per-anchor J pin (`external_context.J`), and the six positive-control receipts
with their `detected` flags (amendment M5.4; refuses to lock if any is missing). The outer role is opened only after
the coordinator commits and pushes this file and writes `private/OUTER_UNLOCK.json`
(schema `pcrl-sc-outer-unlock-v1`) with the remote-verified commit; see
`audit_panel.verify_outer_gate`.

Clause orientation and thresholds are AR's (`AR/inference.THRESHOLDS`):
task = CE_cand - CE_comp; recovery = CE_comp - CE_cand. U: task <= -.003,
guards <= +.001. P: task guard <= +.001, AB/SEX target <= -.002, guards <= +.001.
Comparators that are the same law on all three anchors are grouped into one
endpoint (AR rule), so the count is whatever the generator produces (80 when no
comparator aliases). Endpoints whose two releases are the same law, or chose the
same H-only route, on every anchor are labelled `exact_zero_same_route`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from scipy.stats import norm

from experiments.pcrl_adaptive_release_v1.inference import P_TARGET, TASK_ROLE, THRESHOLDS, WEIGHTINGS

from . import audit_panel, selection

ANCHORS = selection.ANCHORS
ROLES = selection.ROLES
SCHEMA = audit_panel.LOCK_SCHEMA
ALPHA = .05
BOOTSTRAP = {"draws": 10000, "seed": 20260923, "alpha": ALPHA,
             "resampling": "one multinomial over the union of household IDs across the three anchors",
             "estimator": "equal mean of per-anchor weighted ratios",
             "bounds": "normal approximation, two-sided Bonferroni within each family"}
SECONDARY_THRESHOLDS = THRESHOLDS  # descriptive only; never part of a pass
CAPABILITY_THRESHOLD = -.01        # CE_cand - CE_H <= -.01  <=>  benefit over H >= .01


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_sha256(manifest: Mapping) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _clause(arm: str, role: str) -> str:
    return "task" if role == TASK_ROLE else ("target" if arm == "P" and role == P_TARGET else "guard")


def enumerate_family(slots: Sequence[Mapping], *, alias_of: Mapping[str, str],
                     thresholds: Mapping = THRESHOLDS, family: str = "primary") -> list[dict]:
    """Every (slot, comparator group, role, weighting) endpoint with AR clauses."""
    endpoints = []
    for slot in slots:
        slot_id, arm, source = slot["id"], slot["arm"], slot["source_name"]
        if arm not in thresholds:
            raise ValueError("slot arm must be U or P")
        groups: dict[str, list[str]] = {}
        for name in slot["comparators"]:
            groups.setdefault(alias_of.get(name, name), []).append(name)
        for comparator, names in groups.items():
            for role in ROLES:
                clause = _clause(arm, role)
                for weighting in WEIGHTINGS:
                    endpoints.append({
                        "id": f"{family}|{slot_id}|{comparator}|{role}|{weighting}",
                        "family": family, "candidate": slot_id, "candidate_release": source,
                        "arm": arm, "comparator": comparator,
                        "comparator_names": sorted(set(names)), "role": role,
                        "weighting": weighting, "clause": clause,
                        "threshold": thresholds[arm][clause],
                        "plus": source if clause == "task" else comparator,
                        "minus": comparator if clause == "task" else source,
                        "orientation": ("CE_candidate - CE_comparator" if clause == "task" else
                                        "CE_comparator - CE_candidate = recovery_candidate - recovery_comparator")})
    if len({e["id"] for e in endpoints}) != len(endpoints):
        raise ValueError("endpoint IDs collided")
    return endpoints


def capability_family(slots: Sequence[Mapping]) -> list[dict]:
    return [{"id": f"capability|{slot['id']}|H|{TASK_ROLE}|{w}", "family": "capability",
             "candidate": slot["id"], "candidate_release": slot["source_name"], "arm": slot["arm"],
             "comparator": "H", "comparator_names": ["H"], "role": TASK_ROLE, "weighting": w,
             "clause": "capability", "threshold": CAPABILITY_THRESHOLD,
             "plus": slot["source_name"], "minus": "H", "orientation": "CE_candidate - CE_H"}
            for slot in slots for w in WEIGHTINGS]


def secondary_slots(slots: Sequence[Mapping]) -> list[dict]:
    out = []
    for slot in slots:
        nominee = slot["source_name"]           # e.g. NM4_P
        family, form = nominee.split("_")       # NM4, P
        k = family[-1]
        other = f"NM{'1' if k == '4' else '4'}_{form}"
        out.append({"id": slot["id"], "arm": slot["arm"], "source_name": nominee,
                    "comparators": [f"T32_{form}", f"DET_SEL{k}", other, "Q_HIST", "J", "H"]})
    out.append({"id": "RD_TASK_preflight", "arm": "U", "source_name": "RD_TASK",
                "comparators": ["D17"]})
    return out


def _resolve(name: str, anchor: int, report: Mapping) -> str:
    if name in ("H", "J"):
        return name
    return report["logical_to_canonical"][name]


def exact_zero_flags(endpoints: list[dict], reports: Mapping[int, Mapping]) -> None:
    """Label endpoints identical on every anchor (same law, or same H-only route)."""
    for endpoint in endpoints:
        same = []
        for anchor in ANCHORS:
            report = reports[anchor]
            if "J" in (endpoint["plus"], endpoint["minus"]):
                same.append(False)
                continue
            left, right = (_resolve(endpoint["plus"], anchor, report),
                           _resolve(endpoint["minus"], anchor, report))
            if left == right:
                same.append(True)
                continue
            role = endpoint["role"]

            def route(canonical):
                if canonical == "H":
                    rec = next(iter(report["releases"].values()))["roles"][role]
                    return rec["H_selected_route"]
                return report["releases"][canonical]["roles"][role]["selected_route"]
            a, b = route(left), route(right)
            same.append(a == b and a.get("wire") == "H")
        endpoint["exact_zero_same_route"] = all(same)


def _manifest(endpoints: list[dict], scope: str) -> dict:
    m = len(endpoints)
    return {"schema": 1, "n_endpoints": m, "endpoints": endpoints,
            "critical_value_two_sided": float(norm.isf(ALPHA / (2 * m))) if m else None,
            "multiplicity": "two-sided Bonferroni over exactly this generated list",
            "exact_zero_same_route_count": sum(e["exact_zero_same_route"] for e in endpoints),
            "scope": scope}


POS_ROLES = ("AB/SEX", "AB/RAC1P")


def pos_key(anchor: int, role: str) -> str:
    return f"{anchor}|attack:{role}"


def load_positive_controls(units_root: str | Path) -> dict:
    """Amendment M5.4: pin all six POS receipts and `detected` flags; refuse if any is missing."""
    root = Path(units_root)
    out = {}
    for anchor in ANCHORS:
        for role in POS_ROLES:
            uid = f"a{anchor}_POS_{role.replace('/', '_')}"
            receipt_path = root / "_receipts" / f"{uid}.json"
            summary_name = f"a{anchor}_{role.replace('/', '_')}_SUMMARY.json"
            summary_path = root / uid / summary_name
            if not receipt_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(f"positive control {uid} has no completed receipt/summary; refusing to lock")
            receipt = json.loads(receipt_path.read_text())
            summary = json.loads(summary_path.read_text())
            if (receipt.get("outputs_sha256", {}).get(summary_name) != _sha(summary_path) or
                    summary.get("anchor") != anchor or summary.get("role") != role or
                    summary.get("smoke") is not False or summary.get("outer_labels_accessed") is not False or
                    not isinstance(summary.get("detected"), bool)):
                raise ValueError(f"positive control {uid} receipt or summary differs")
            out[pos_key(anchor, role)] = {
                "unit_id": uid, "runner_receipt_sha256": _sha(receipt_path),
                "summary_sha256": _sha(summary_path), "detected": summary["detected"],
                "improvement_nats": summary["improvement_nats"],
                "threshold_nats": summary["threshold_nats"]}
    return out


def build_lock(inner_selection: Mapping, reports: Mapping[int, Mapping], pins: Mapping[str, Mapping],
               *, j_pins: Mapping[str, Mapping] | None, code_commit: str,
               protocol_sha256: str, inner_selection_sha256: str,
               positive_controls: Mapping[str, Mapping]) -> dict:
    expected_pos = {pos_key(a, r) for a in ANCHORS for r in POS_ROLES}
    if set(positive_controls) != expected_pos:
        raise ValueError("all six positive-control receipts are required before the lock")
    routes = inner_selection["routes"]
    aliases = dict(inner_selection["alias_of"])
    slots = [{"id": f"{route}_nominee", "arm": route, "source_name": routes[route]["nominee"],
              "label": routes[route]["label"], "adv_representative": routes[route]["adv_representative"],
              "comparators": list(routes[route]["comparators"])} for route in ("U", "P")]
    primary = enumerate_family(slots, alias_of=aliases)
    secondary_def = secondary_slots(slots)
    if j_pins is None:
        for slot in secondary_def:
            slot["comparators"] = [c for c in slot["comparators"] if c != "J"]
    secondary = enumerate_family(secondary_def, alias_of=aliases,
                                 thresholds=SECONDARY_THRESHOLDS, family="secondary")
    capability = capability_family(slots)
    for family in (primary, secondary, capability):
        exact_zero_flags(family, reports)
    manifests = {"family_manifest": _manifest(primary, "primary: every clause of a slot must pass"),
                 "secondary_manifest": _manifest(secondary, "descriptive, separately corrected; never part of a pass"),
                 "capability_manifest": _manifest(capability, "separate H-capability family; not a primary rescue")}
    anchors = {}
    for anchor in ANCHORS:
        report, pin = reports[anchor], pins[str(anchor)]
        anchors[str(anchor)] = {
            "inner_panel_complete_sha256": pin["inner_panel_complete_sha256"],
            "inner_audit_sha256": pin["inner_audit_sha256"],
            "logical_to_canonical": dict(report["logical_to_canonical"]),
            "releases": {rid: payload["source"] for rid, payload in report["releases"].items()}}
    index_shas = {pins[str(a)]["index_sha256"] for a in ANCHORS}
    if len(index_shas) != 1:
        raise ValueError("inner panels used different input indexes")
    lock = {"schema": SCHEMA, "status": "LOCKED", "study": audit_panel.STUDY,
            "assessment_year": 2018, "outer_assessment_authorized": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": code_commit, "protocol_sha256": protocol_sha256,
            "inner_selection_sha256": inner_selection_sha256,
            "input_index_sha256": index_shas.pop(),
            "slots": slots, "alias_of": aliases, **manifests,
            "manifest_sha256": {name: manifest_sha256(value) for name, value in manifests.items()},
            "bootstrap": BOOTSTRAP, "anchors": anchors,
            "positive_controls": dict(positive_controls),
            "audit_uninformative": sorted(k for k, v in positive_controls.items() if not v["detected"]),
            "external_context": {"J": dict(j_pins)} if j_pins is not None else None,
            "outer_access": ("only via audit_panel.verify_outer_gate after this file is committed, "
                             "pushed and remote-verified and private/OUTER_UNLOCK.json is written")}
    return lock


def verify_lock_structure(lock: Mapping) -> None:
    """Regenerate every family from the slots and compare counts and hashes."""
    if lock.get("schema") != SCHEMA or lock.get("status") != "LOCKED":
        raise PermissionError("not a locked shared-context selection")
    pos = lock.get("positive_controls") or {}
    if (set(pos) != {pos_key(a, r) for a in ANCHORS for r in POS_ROLES} or
            lock.get("audit_uninformative") != sorted(k for k, v in pos.items() if not v["detected"])):
        raise PermissionError("positive-control receipts are not pinned in the lock")
    aliases = lock["alias_of"]
    primary = enumerate_family(lock["slots"], alias_of=aliases)
    secondary_def = secondary_slots(lock["slots"])
    if lock.get("external_context") is None:
        for slot in secondary_def:
            slot["comparators"] = [c for c in slot["comparators"] if c != "J"]
    secondary = enumerate_family(secondary_def, alias_of=aliases,
                                 thresholds=SECONDARY_THRESHOLDS, family="secondary")
    capability = capability_family(lock["slots"])
    for name, generated in (("family_manifest", primary), ("secondary_manifest", secondary),
                            ("capability_manifest", capability)):
        stored = lock[name]
        strip = [{k: v for k, v in e.items() if k != "exact_zero_same_route"} for e in stored["endpoints"]]
        if strip != generated or stored["n_endpoints"] != len(generated):
            raise PermissionError(f"{name} differs from the family generated from the slots")
        if lock["manifest_sha256"][name] != manifest_sha256(stored):
            raise PermissionError(f"{name} hash differs")


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=audit_panel.ROOT, text=True).strip()


def write_once(path: Path, value: Mapping) -> str:
    if path.exists():
        raise FileExistsError(f"{path.name} is write-once")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    return _sha(path)


def _j_pin(directory: str | Path, anchor: int, index_sha: str) -> dict:
    root = Path(directory)
    complete = json.loads((root / "COMPLETE.json").read_text())
    report = json.loads((root / "INNER_AUDIT.json").read_text())
    if (complete.get("schema") != "pcrl-adaptive-continuous-inner-complete-v1" or
            complete.get("artifacts") != audit_panel.evaluate._inventory(root) or
            report.get("anchor") != anchor or report.get("outer_pool_opened") is not False or
            "J" not in report.get("releases", {}) or report.get("input_index_sha256") != index_sha):
        raise ValueError(f"J inner panel for anchor {anchor} is incomplete or differs")
    return {"panel_dir": str(root.resolve()), "inner_complete_sha256": _sha(root / "COMPLETE.json"),
            "inner_audit_sha256": _sha(root / "INNER_AUDIT.json")}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inner-selection", required=True)
    parser.add_argument("--reports", nargs=3, required=True, help="inner panel dirs a0 a1 a2")
    parser.add_argument("--j-reports", nargs=3, help="J inner panel dirs a0 a1 a2")
    parser.add_argument("--units-root", required=True, help="runner units root holding the six POS units")
    parser.add_argument("--protocol", default=str(audit_panel.RESULTS / "PROTOCOL.md"))
    parser.add_argument("--out", default=str(audit_panel.LOCK_PATH))
    args = parser.parse_args(argv)
    inner_path = Path(args.inner_selection)
    inner = json.loads(inner_path.read_text())
    reports, pins = {}, {}
    for anchor, directory in zip(ANCHORS, args.reports):
        reports[anchor], pins[str(anchor)] = selection.load_panel(directory)
        if pins[str(anchor)]["inner_audit_sha256"] != inner["panels"][str(anchor)]["inner_audit_sha256"]:
            raise ValueError("inner panel differs from the one INNER_SELECTION was made from")
    if selection.select_inner(reports)["routes"] != inner["routes"]:
        raise ValueError("INNER_SELECTION does not replay from the pinned panels")
    j_pins = None
    if args.j_reports:
        index_sha = pins["0"]["index_sha256"]
        j_pins = {str(a): _j_pin(d, a, index_sha) for a, d in zip(ANCHORS, args.j_reports)}
    lock = build_lock(inner, reports, pins, j_pins=j_pins, code_commit=_git_commit(),
                      protocol_sha256=_sha(args.protocol), inner_selection_sha256=_sha(inner_path),
                      positive_controls=load_positive_controls(args.units_root))
    verify_lock_structure(lock)
    digest = write_once(Path(args.out), lock)
    print(json.dumps({"lock_sha256": digest,
                      "primary": lock["family_manifest"]["n_endpoints"],
                      "secondary": lock["secondary_manifest"]["n_endpoints"],
                      "capability": lock["capability_manifest"]["n_endpoints"],
                      "z_primary": lock["family_manifest"]["critical_value_two_sided"]}, sort_keys=True))


if __name__ == "__main__":
    main()
