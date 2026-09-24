"""Nomination on synthetic inner_check reports (PROTOCOL section 6)."""
from __future__ import annotations

import pytest

from experiments.pcrl_shared_context_release_v1 import audit_panel, selection

ROLES = selection.ROLES
TASK, ABSEX = "utility:A/same_residence", "attack:AB/SEX"
NAMES = ("D17", "Q_HIST", "NM1_U", "NM1_P", "NM4_U", "NM4_P", "T32_U", "T32_P",
         "DET_SEL1", "DET_SEL4", "RD_TASK", "RD_PRIV", "ADV_B1", "ADV_B2", "ADV_B1_P", "ADV_B2_P")
BASE = {TASK: .60, "attack:A/SEX": .69, "attack:A/RAC1P": 1.2, ABSEX: .65, "attack:AB/RAC1P": 1.1}
H = {TASK: .65, "attack:A/SEX": .69, "attack:A/RAC1P": 1.2, ABSEX: .66, "attack:AB/RAC1P": 1.1}


def route(wire="release", sha="a" * 64):
    return {"source_view": "A", "wire": wire, "model_sha256": sha}


def report(anchor, deltas=None, aliases=None, h_routes=None):
    """deltas[name][role] = CE shift vs D17 (negative task = better; positive recovery role = less recovery)."""
    deltas = deltas or {}
    aliases = dict(aliases or {})
    releases, mapping = {}, {}
    for name in NAMES:
        mapping[name] = aliases.get(name, name)
        if mapping[name] != name:
            continue
        roles = {}
        for role in ROLES:
            value = BASE[role] + deltas.get(name, {}).get(role, 0.)
            roles[role] = {"candidate": {"U": value, "PWGTP": value + 1e-4},
                           "H": {"U": H[role], "PWGTP": H[role] + 1e-4},
                           "selected_route": (h_routes or {}).get((name, role), route(sha=name[:1] * 64)),
                           "H_selected_route": route("H", "h" * 64)}
        releases[name] = {"roles": roles, "source": {"kind": "synthetic", "release_id": name}}
    return {"schema": audit_panel.REPORT_SCHEMA, "anchor": anchor, "selection_role": "inner_selection",
            "score_role": "inner_check", "outer_pool_opened": False, "releases": releases,
            "logical_to_canonical": mapping}


def three(**kw):
    return {a: report(a, **kw) for a in (0, 1, 2)}


def test_u_nominee_passes_screen_and_ranks_by_task():
    deltas = {"NM1_U": {TASK: -.004}, "NM4_U": {TASK: -.006, ABSEX: .002},  # NM4_U best task, AB/SEX ok
              "NM4_P": {TASK: -.010, "attack:A/SEX": -.005}}               # more recovery: fails guard
    out = selection.select_inner(three(deltas=deltas))
    assert out["routes"]["U"]["nominee"] == "NM4_U"
    assert out["routes"]["U"]["label"] == "NOMINATED_POINT_SCREEN"
    assert out["routes"]["U"]["comparators"][:3] == ["D17", "RD_TASK", "RD_PRIV"]
    assert out["score_resource"].startswith("inner_check")


def test_p_screen_needs_ab_sex_target_and_fallback_is_diagnostic_only():
    deltas = {"NM1_P": {ABSEX: .003}, "NM4_P": {ABSEX: .0025, TASK: .0005}}
    out = selection.select_inner(three(deltas=deltas))
    assert out["routes"]["P"]["nominee"] == "NM1_P"   # max AB/SEX recovery lowest (-.003)
    assert out["routes"]["P"]["label"] == "NOMINATED_POINT_SCREEN"
    none = selection.select_inner(three(deltas={"NM1_P": {ABSEX: .001}}))
    assert none["routes"]["P"]["label"] == "DIAGNOSTIC_ONLY"
    assert none["routes"]["P"]["nominee"] == "NM1_P"  # rank-min over all four
    assert none["routes"]["U"]["label"] == "DIAGNOSTIC_ONLY"


def test_capability_screen_uses_h_on_inner_check():
    deltas = {"NM1_U": {TASK: -.004}}
    reports = three(deltas=deltas)
    for rep in reports.values():
        for payload in rep["releases"].values():
            payload["roles"][TASK]["H"] = {"U": .59, "PWGTP": .5901}  # H better than candidates
    out = selection.select_inner(reports)
    assert out["routes"]["U"]["label"] == "DIAGNOSTIC_ONLY"


def test_adv_representative_eligibility_then_rank():
    deltas = {"ADV_B1": {TASK: -.02, "attack:A/SEX": -.01},   # best task but recovers: ineligible for U
              "ADV_B2": {TASK: -.001}}
    out = selection.select_inner(three(deltas=deltas))
    assert out["routes"]["U"]["adv_representative"] == "ADV_B2"
    assert out["routes"]["U"]["adv_selection"]["status"] == "ELIGIBLE_RANK_MIN"
    assert {p["id"] for p in out["routes"]["U"]["adv_selection"]["points"]} == {"ADV_B1", "ADV_B2"}
    both_bad = {"ADV_B1_P": {TASK: .01, ABSEX: -.001}, "ADV_B2_P": {TASK: .02, ABSEX: -.002},
                "ADV_B1": {ABSEX: .01}}  # task-selected units never enter the P route
    out = selection.select_inner(three(deltas=both_bad))
    assert out["routes"]["P"]["adv_selection"]["status"] == "RANK_MIN_OVERALL"
    assert out["routes"]["P"]["adv_representative"] == "ADV_B1_P"  # max AB/SEX recovery lower
    assert out["routes"]["P"]["comparators"][-1] == "ADV_B1_P"


def test_aliases_collapse_only_when_identical_on_all_anchors():
    reports = {0: report(0, aliases={"RD_TASK": "D17", "T32_U": "D17"}),
               1: report(1, aliases={"RD_TASK": "D17", "T32_U": "D17"}),
               2: report(2, aliases={"RD_TASK": "D17"})}
    out = selection.select_inner(reports)
    assert out["alias_of"]["RD_TASK"] == "D17"
    assert "T32_U" not in out["alias_of"]


def test_refuses_non_inner_reports_and_missing_names():
    bad = three()
    bad[1]["outer_pool_opened"] = True
    with pytest.raises(ValueError):
        selection.select_inner(bad)
    missing = three()
    for rep in missing.values():
        del rep["logical_to_canonical"]["ADV_B2"]
    with pytest.raises(ValueError, match="ADV_B2"):
        selection.select_inner(missing)


def test_cli_inner_selection_from_three_completed_panels(tmp_path, monkeypatch):
    import json
    from tests.pcrl_shared_context_release_v1 import test_audit_panel as tap

    tap.mock_slate(monkeypatch)
    d17_spec, q = tap.d17_file(tmp_path)
    panels = []
    for anchor in (0, 1, 2):
        sources = {"D17": d17_spec}
        for name in NAMES[1:]:
            sources[name] = tap.nested_unit(tmp_path / f"a{anchor}", monkeypatch, name,
                                            lambda legal: q[legal["token_codes"]].copy()) \
                if (tmp_path / f"a{anchor}").mkdir(exist_ok=True) is None else None
        out = tmp_path / "private" / f"panel_a{anchor}"
        audit_panel.run_panel(anchor, tap.synthetic_pools(seed=anchor), sources, out,
                              index_sha256="e" * 64)
        panels.append(str(out))
    target = tmp_path / "INNER_SELECTION.json"
    selection.main(["inner", "--reports", *panels, "--out", str(target)])
    value = json.loads(target.read_text())
    assert value["routes"]["U"]["label"] == "DIAGNOSTIC_ONLY"   # every NM unit collapsed to D17
    assert value["alias_of"]["NM1_U"] == "D17"
    assert set(value["panels"]) == {"0", "1", "2"}
    with pytest.raises(FileExistsError):
        selection.main(["inner", "--reports", *panels, "--out", str(target)])
