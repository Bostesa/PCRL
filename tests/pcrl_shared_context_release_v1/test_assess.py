"""Locked assessment on synthetic outer archives (no ACS rows)."""
from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import evaluate
from experiments.pcrl_shared_context_release_v1 import assess, audit_panel, lock, selection

from tests.pcrl_shared_context_release_v1.test_lock import J_PINS, PINS, POS
from tests.pcrl_shared_context_release_v1.test_selection import ABSEX, TASK, route, three

N = 60


def people(anchor):
    rng = np.random.default_rng(anchor)
    # Anchors overlap in households, as the real global roles do.
    return {"ids": np.array([f"p{anchor}-{i}" for i in range(N)]),
            "households": np.array([f"hh{(i + 7 * anchor) // 2}" for i in range(N)]),
            "weights": rng.uniform(.5, 2., N)}


def write_outer(root, anchor, lock_sha, per_release, *, j=False, confirmed=None, breaks=None):
    """Mimic audit_panel.score_outer / score_outer_j output files."""
    root.mkdir(parents=True)
    rows = people(anchor)
    rng = np.random.default_rng(100 + anchor)
    noise = {role: rng.normal(0, .01, N) for role in selection.ROLES}
    contributions, releases = {}, {}
    for release, shift in per_release.items():
        prefix = hashlib.sha256(release.encode()).hexdigest()[:12]
        roles = {}
        for role in selection.ROLES:
            h_loss = .7 + noise[role] + .05
            loss = .7 + noise[role] + shift.get(role, 0.)
            score = {**rows, "loss": loss, "scores": {}}
            h_score = {**rows, "loss": h_loss, "scores": {}}
            contributions.update(evaluate._private_contributions(prefix, role, score, h_score))
            roles[role] = {}
        releases[release] = {"roles": roles, "private_contribution_prefix": prefix}
    np.savez_compressed(root / "OUTER_CONTRIBUTIONS.npz", **contributions)
    report = {"schema": "x", "anchor": anchor, "no_outer_fit_or_selection": True,
              "selection_lock_sha256": lock_sha, "releases": releases,
              "outer_alias_confirmed": confirmed or {}, "outer_alias_breaks": breaks or {},
              "contributions_sha256": evaluate._sha(root / "OUTER_CONTRIBUTIONS.npz")}
    (root / "OUTER_AUDIT.json").write_text(json.dumps(report))
    schema = "pcrl-sc-outer-J-complete-v1" if j else "pcrl-sc-outer-audit-complete-v1"
    (root / "COMPLETE.json").write_text(json.dumps(
        {"schema": schema, "anchor": anchor, "artifacts": evaluate._inventory(root)}))


def setup(tmp_path, aliases=None, break_alias=False):
    h_routes = {(n, "attack:A/RAC1P"): route("H", "h" * 64) for n in ("NM4_U", "D17")}
    reports = three(deltas={"NM4_U": {TASK: -.006}, "NM1_P": {ABSEX: .003}}, h_routes=h_routes,
                    aliases=aliases)
    inner = selection.select_inner(reports)
    value = lock.build_lock(inner, reports, PINS, j_pins=J_PINS, code_commit="f" * 40,
                            protocol_sha256="1" * 64, inner_selection_sha256="2" * 64,
                            positive_controls=POS)
    path = tmp_path / "SELECTION_LOCK.json"
    sha = lock.write_once(path, value)
    shifts = {name: {} for name in reports[0]["releases"]}
    shifts["NM4_U"] = {TASK: -.05}
    # identical person losses for the exact-zero endpoint pair on A/RAC1P
    outer, outer_j = {}, {}
    for anchor in (0, 1, 2):
        outer[anchor] = tmp_path / "private" / f"outer_a{anchor}"
        per = dict(shifts)
        confirmed, breaks = {}, {}
        for name, canonical in (aliases or {}).items():
            if break_alias and anchor == 1:
                per[name] = {TASK: .03}
                breaks[name] = {"canonical": canonical, "outer_people_differing": 3}
            else:
                confirmed[name] = canonical
        write_outer(outer[anchor], anchor, sha, per, confirmed=confirmed, breaks=breaks)
        outer_j[anchor] = tmp_path / "private" / f"outer_j_a{anchor}"
        write_outer(outer_j[anchor], anchor, sha, {"J": {TASK: .02}}, j=True)
    return value, path, sha, outer, outer_j


def test_assessment_writes_all_three_families(tmp_path):
    value, path, sha, outer, outer_j = setup(tmp_path)
    out = tmp_path / "results"
    report = assess.assess(path, sha, outer, out, j_outer_dirs=outer_j, n_boot=200,
                           _gate=lambda p, s: json.loads(path.read_text()))
    families = report["families"]
    assert families["family_manifest"]["family_size"] == 80
    assert families["secondary_manifest"]["family_size"] == 130
    assert families["capability_manifest"]["family_size"] == 4
    u = report["decisions"]["U_nominee"]
    assert u["nominee"] == "NM4_U" and u["clauses_total"] == 40
    assert u["route_passed_all_primary_clauses"] is True
    assert report["decisions"]["P_nominee"]["route_passed_all_primary_clauses"] is False
    table = json.loads((out / "ENDPOINT_TABLE.json").read_text())["rows"]
    zero = [r for r in table if r["exact_zero_same_route"]]
    assert zero and all(r["estimate"] == 0. for r in zero if r["family"] == "primary")
    with (out / "FULL_RESULTS.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 80 + 130 + 4
    assert all(report["j_H_identical_to_main_H"].values())
    assert report["audit_uninformative"] == ["2|attack:AB/RAC1P"]
    rac = next(r for r in table if r["role"] == "attack:AB/RAC1P")
    assert rac["audit_uninformative_anchors"] == [2]
    assert next(r for r in table if r["role"] == ABSEX)["audit_uninformative_anchors"] == []
    with pytest.raises(FileExistsError):
        assess.assess(path, sha, outer, out, j_outer_dirs=outer_j, n_boot=200,
                      _gate=lambda p, s: json.loads(path.read_text()))


def test_assessment_refuses_without_coordinator_unlock(tmp_path):
    value, path, sha, outer, _ = setup(tmp_path)
    with pytest.raises(PermissionError):
        assess.assess(path, sha, outer, tmp_path / "results")  # real gate: wrong path, no unlock
    assert not (tmp_path / "results").exists()


def test_archive_from_another_lock_is_refused(tmp_path):
    value, path, sha, outer, outer_j = setup(tmp_path)
    with pytest.raises(ValueError, match="not from this lock"):
        assess.assess(path, "0" * 64, outer, tmp_path / "results", n_boot=50, j_outer_dirs=outer_j,
                      _gate=lambda p, s: json.loads(path.read_text()))


def test_locked_j_endpoints_require_outer_j(tmp_path):
    value, path, sha, outer, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="outer-j"):
        assess.assess(path, sha, outer, tmp_path / "results", n_boot=50,
                      _gate=lambda p, s: json.loads(path.read_text()))


def test_outer_alias_confirmed_or_scored_separately_never_copied(tmp_path):
    value, path, sha, outer, outer_j = setup(tmp_path / "ok", aliases={"RD_TASK": "D17"})
    assert value["family_manifest"]["n_endpoints"] == 60
    report = assess.assess(path, sha, outer, tmp_path / "ok" / "res", j_outer_dirs=outer_j, n_boot=50,
                           _gate=lambda p, s: json.loads(path.read_text()))
    assert report["outer_alias_breaks"] == {} and report["outer_merge_recomputed"] == []
    value, path, sha, outer, outer_j = setup(tmp_path / "br", aliases={"RD_TASK": "D17"}, break_alias=True)
    report = assess.assess(path, sha, outer, tmp_path / "br" / "res", j_outer_dirs=outer_j, n_boot=50,
                           _gate=lambda p, s: json.loads(path.read_text()))
    assert set(report["outer_alias_breaks"]["RD_TASK"]) == {"1"}
    assert report["families"]["outer_alias_supplement"]["family_size"] == 2 * 10  # D17|RD_TASK group in both slots
    assert report["families"]["family_manifest"]["family_size"] == 60  # locked family unchanged
    # an alias that is neither confirmed nor broken on outer rows is refused
    value, path, sha, outer, outer_j = setup(tmp_path / "un", aliases={"RD_TASK": "D17"})
    report_path = outer[0] / "OUTER_AUDIT.json"
    record = json.loads(report_path.read_text())
    record["outer_alias_confirmed"] = {}
    report_path.write_text(json.dumps(record))
    (outer[0] / "COMPLETE.json").write_text(json.dumps(
        {"schema": "pcrl-sc-outer-audit-complete-v1", "anchor": 0,
         "artifacts": evaluate._inventory(outer[0])}))
    with pytest.raises(ValueError, match="not verified on outer rows"):
        assess.assess(path, sha, outer, tmp_path / "un" / "res", j_outer_dirs=outer_j, n_boot=50,
                      _gate=lambda p, s: json.loads(path.read_text()))
