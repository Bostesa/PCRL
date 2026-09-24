"""Code-generated endpoint families and the write-once lock (PROTOCOL section 7)."""
from __future__ import annotations

import copy
import json

import pytest
from scipy.stats import norm

from experiments.pcrl_shared_context_release_v1 import lock, selection

from tests.pcrl_shared_context_release_v1.test_selection import ABSEX, TASK, route, three

PINS = {str(a): {"inner_panel_complete_sha256": f"{a}" * 64, "inner_audit_sha256": f"{a + 3}" * 64,
                 "index_sha256": "e" * 64} for a in (0, 1, 2)}
J_PINS = {str(a): {"inner_complete_sha256": "c" * 64, "inner_audit_sha256": "d" * 64} for a in (0, 1, 2)}
POS = {lock.pos_key(a, r): {"detected": not (a == 2 and r == "AB/RAC1P"), "improvement_nats": {}}
       for a in (0, 1, 2) for r in lock.POS_ROLES}


def build(reports=None, j=True):
    reports = reports or three(deltas={"NM4_U": {TASK: -.006}, "NM1_P": {ABSEX: .003}})
    inner = selection.select_inner(reports)
    return lock.build_lock(inner, reports, PINS, j_pins=J_PINS if j else None, code_commit="f" * 40,
                           protocol_sha256="1" * 64, inner_selection_sha256="2" * 64,
                           positive_controls=POS), reports


def test_primary_family_is_80_with_registered_clauses_and_z():
    value, _ = build()
    manifest = value["family_manifest"]
    assert manifest["n_endpoints"] == 80 == 2 * 4 * 5 * 2
    assert manifest["critical_value_two_sided"] == pytest.approx(norm.isf(.05 / 160))
    rows = manifest["endpoints"]
    u = [r for r in rows if r["candidate"] == "U_nominee"]
    p = [r for r in rows if r["candidate"] == "P_nominee"]
    assert len(u) == len(p) == 40
    assert {r["threshold"] for r in u if r["clause"] == "task"} == {-.003}
    assert {r["threshold"] for r in u if r["clause"] == "guard"} == {.001}
    assert {r["threshold"] for r in p if r["clause"] == "task"} == {.001}
    assert {(r["role"], r["threshold"]) for r in p if r["clause"] == "target"} == {(ABSEX, -.002)}
    assert sum(r["clause"] == "guard" for r in p) == 4 * 3 * 2  # 3 non-target recovery roles
    assert {r["comparator"] for r in u} == {"D17", "RD_TASK", "RD_PRIV", value["slots"][0]["adv_representative"]}
    task = next(r for r in u if r["clause"] == "task")
    assert (task["plus"], task["minus"]) == ("NM4_U", "D17")
    guard = next(r for r in u if r["clause"] == "guard")
    assert guard["minus"] == "NM4_U"  # recovery = CE_comp - CE_cand
    assert value["manifest_sha256"]["family_manifest"] == lock.manifest_sha256(manifest)


def test_secondary_and_capability_families():
    value, _ = build()
    sec = value["secondary_manifest"]
    assert sec["n_endpoints"] == 2 * 6 * 10 + 10
    u_comps = {r["comparator"] for r in sec["endpoints"] if r["candidate"] == "U_nominee"}
    assert u_comps == {"T32_U", "DET_SEL4", "NM1_U", "Q_HIST", "J", "H"}
    p_comps = {r["comparator"] for r in sec["endpoints"] if r["candidate"] == "P_nominee"}
    assert p_comps == {"T32_P", "DET_SEL1", "NM4_P", "Q_HIST", "J", "H"}
    assert any(r["candidate"] == "RD_TASK_preflight" and r["comparator"] == "D17" for r in sec["endpoints"])
    cap = value["capability_manifest"]
    assert cap["n_endpoints"] == 4 and {r["threshold"] for r in cap["endpoints"]} == {-.01}
    assert value["external_context"]["J"]["0"]["inner_complete_sha256"] == "c" * 64
    no_j, _ = build(j=False)
    assert no_j["secondary_manifest"]["n_endpoints"] == 2 * 5 * 10 + 10
    assert no_j["external_context"] is None


def test_comparator_aliases_on_all_anchors_merge_endpoints():
    reports = three(deltas={"NM4_U": {TASK: -.006}}, aliases={"RD_TASK": "D17"})
    value, _ = build(reports)
    assert value["family_manifest"]["n_endpoints"] == 2 * 3 * 10
    d17 = next(r for r in value["family_manifest"]["endpoints"] if r["comparator"] == "D17")
    assert d17["comparator_names"] == ["D17", "RD_TASK"]


def test_exact_zero_same_route_label():
    h_routes = {("NM4_U", "attack:A/RAC1P"): route("H", "h" * 64),
                ("D17", "attack:A/RAC1P"): route("H", "h" * 64)}
    reports = three(deltas={"NM4_U": {TASK: -.006}}, h_routes=h_routes)
    value, _ = build(reports)
    flagged = [r for r in value["family_manifest"]["endpoints"] if r["exact_zero_same_route"]]
    assert {(r["candidate"], r["comparator"], r["role"]) for r in flagged} >= {
        ("U_nominee", "D17", "attack:A/RAC1P")}
    assert all(r["role"] == "attack:A/RAC1P" for r in flagged if r["candidate"] == "U_nominee")


def test_verify_structure_detects_tampering_and_lock_is_write_once(tmp_path):
    value, _ = build()
    lock.verify_lock_structure(value)
    bad = copy.deepcopy(value)
    bad["family_manifest"]["endpoints"][0]["threshold"] = .5
    with pytest.raises(PermissionError):
        lock.verify_lock_structure(bad)
    bad = copy.deepcopy(value)
    bad["slots"][0]["comparators"].pop()
    with pytest.raises(PermissionError):
        lock.verify_lock_structure(bad)
    path = tmp_path / "SELECTION_LOCK.json"
    lock.write_once(path, value)
    assert json.loads(path.read_text())["status"] == "LOCKED"
    with pytest.raises(FileExistsError):
        lock.write_once(path, value)


def test_lock_requires_all_six_positive_controls(tmp_path):
    value, reports = build()
    assert value["audit_uninformative"] == ["2|attack:AB/RAC1P"]
    inner = selection.select_inner(reports)
    partial = {k: v for k, v in POS.items() if k != "0|attack:AB/SEX"}
    with pytest.raises(ValueError, match="six positive-control"):
        lock.build_lock(inner, reports, PINS, j_pins=None, code_commit="f" * 40,
                        protocol_sha256="1" * 64, inner_selection_sha256="2" * 64,
                        positive_controls=partial)
    with pytest.raises(FileNotFoundError, match="refusing to lock"):
        lock.load_positive_controls(tmp_path / "private" / "units")


def test_load_positive_controls_pins_receipts(tmp_path):
    root = tmp_path / "private" / "units"
    for a in (0, 1, 2):
        for r in lock.POS_ROLES:
            uid = f"a{a}_POS_{r.replace('/', '_')}"
            name = f"a{a}_{r.replace('/', '_')}_SUMMARY.json"
            (root / uid).mkdir(parents=True)
            summary = {"anchor": a, "role": r, "smoke": False, "outer_labels_accessed": False,
                       "detected": True, "improvement_nats": {"U": 1., "PWGTP": 1.}, "threshold_nats": .01}
            (root / uid / name).write_text(json.dumps(summary))
            (root / "_receipts").mkdir(exist_ok=True)
            (root / "_receipts" / f"{uid}.json").write_text(json.dumps(
                {"outputs_sha256": {name: lock._sha(root / uid / name)}}))
    pinned = lock.load_positive_controls(root)
    assert len(pinned) == 6 and all(v["detected"] for v in pinned.values())
    (root / "a1_POS_AB_SEX" / "a1_AB_SEX_SUMMARY.json").write_text("{}")
    with pytest.raises(ValueError):
        lock.load_positive_controls(root)
