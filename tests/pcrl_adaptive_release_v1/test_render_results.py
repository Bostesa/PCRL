"""Aggregate-only renderer fixtures; no outer person records."""
from __future__ import annotations

import hashlib
import json

import pytest

from experiments.pcrl_adaptive_release_v1 import audit, inference, inference_run
from experiments.pcrl_adaptive_release_v1 import render_results


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    slot = {"id": "P_candidate", "arm": "P", "source_name": "A_selected",
            "comparators": ["D17"]}
    family = inference.family_manifest([slot])
    capability = inference_run.expected_capability_manifest([slot])
    choice = {"P": {"selected": "A_selected", "status": "DIAGNOSTIC_ONLY"}}
    families = {"simple": {"P": {"selected": "D17"}}}
    lock = {"schema": "pcrl-adaptive-selection-lock-v1", "status": "LOCKED",
            "assessment_year": 2018, "slots": [slot], "alias_of": {},
            "family_manifest": family, "capability_manifest": capability,
            "selection_source": {"candidate_choices": choice,
                                 "family_choices": families},
            "anchors": {str(a): {"logical_to_canonical": {
                "P_candidate": "A_selected", "D17": "D17"},
                "releases": {"A_selected": {"toy": "A"}, "D17": {"toy": "D"}}}
                for a in range(3)}}
    lock_path = tmp_path / "SELECTION_LOCK.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True))
    lock_sha = _sha(lock_path)
    inner = {"schema": "pcrl-adaptive-inner-selection-v1",
             "source_role": "inner_selection", "outer_pool_opened": False,
             "candidate_choices": choice, "family_choices": families,
             "named_validation_scores": {}}
    inner_path = tmp_path / "INNER_SELECTION.json"
    inner_path.write_text(json.dumps(inner, sort_keys=True))
    rows = []
    for endpoint in family["endpoints"]:
        role = endpoint["role"]
        estimate = (.0005 if role == inference.TASK_ROLE else
                    -.0025 if role == inference.P_TARGET else -.0002)
        lower, upper = estimate-.001, estimate+.001
        rows.append({**endpoint, "estimate": estimate, "lower": lower,
                     "upper": upper, "bootstrap_se": .0002,
                     "anchor_estimates": [estimate]*3,
                     "point_screen_passed": estimate <= endpoint["threshold"],
                     "passed_upper_bound": upper <= endpoint["threshold"],
                     "demonstrated_adverse": lower > endpoint["threshold"]})
    cap_rows = []
    for endpoint in capability["endpoints"]:
        estimate, lower, upper = -.02, -.025, -.015
        cap_rows.append({**endpoint, "estimate": estimate, "lower": lower,
                         "upper": upper, "bootstrap_se": .001,
                         "anchor_estimates": [estimate]*3,
                         "point_screen_passed": True,
                         "passed_upper_bound": True,
                         "demonstrated_adverse": False,
                         "point_benefit_over_H": .02,
                         "benefit_over_H_interval": [.015, .025],
                         "point_capability_screen_passed": True,
                         "interval_supports_registered_threshold": True})
    decision = {"P_candidate": {"all_primary_clauses_passed": False,
                                "passed": sum(r["passed_upper_bound"] for r in rows),
                                "total": len(rows)}}
    result = {"schema": "pcrl-adaptive-outer-inference-v1",
              "selection_lock_sha256": lock_sha,
              "assessment_year": 2018, "development_only": True,
              "family_manifest": family, "capability_manifest": capability,
              "family_size": len(rows), "rows": rows, "decisions": decision,
              "capability": {"family_size": len(cap_rows), "rows": cap_rows,
                             "primary_clause_rescue": False},
              "anchor_resolution": {str(a): {"P_candidate": "A_selected",
                                             "D17": "D17"} for a in range(3)}}
    inference_path = tmp_path / "INFERENCE.json"
    inference_path.write_text(json.dumps(result, sort_keys=True))
    outer = {}
    for anchor in range(3):
        releases = {}
        for name in ("A_selected", "D17"):
            roles = {}
            for role in audit.ROLES:
                candidate = (.5005 if role == inference.TASK_ROLE and name == "A_selected" else
                             .5205 if role == inference.TASK_ROLE and name == "D17" else
                             .5025 if role == inference.P_TARGET and name == "A_selected" else
                             .5002 if name == "A_selected" else .5)
                # D17 task .5 for primary; H task .5205 for capability.
                if role == inference.TASK_ROLE and name == "D17":
                    candidate = .5
                h = .5205 if role == inference.TASK_ROLE else .6
                roles[role] = {"candidate": {"U": candidate, "PWGTP": candidate},
                               "H": {"U": h, "PWGTP": h}}
            releases[name] = {"source": lock["anchors"][str(anchor)]["releases"][name],
                              "roles": roles}
        report = {"schema": "pcrl-adaptive-outer-audit-v1", "anchor": anchor,
                  "assessment_year": 2018, "development_only": True,
                  "no_outer_fit_or_selection": True,
                  "selection_lock_sha256": lock_sha, "releases": releases}
        path = tmp_path / f"OUTER_AUDIT_a{anchor}.json"
        path.write_text(json.dumps(report, sort_keys=True))
        outer[anchor] = path
    return lock_path, inner_path, inference_path, outer


def test_renderer_keeps_primary_and_capability_separate_and_bound_based(tmp_path):
    lock, inner, inference_path, outer = _fixture(tmp_path)
    csv = tmp_path / "FULL_RESULTS.csv"
    table = tmp_path / "ENDPOINT_TABLE.json"
    plot = tmp_path / "PLOT_DATA.json"
    summary = render_results.render_results(
        lock, _sha(lock), inner, _sha(inner), inference_path, _sha(inference_path),
        {a: (path, _sha(path)) for a, path in outer.items()}, csv, table, plot)
    assert summary["primary_all_passed"] is False
    assert summary["primary_endpoint_count"] == 10
    assert summary["capability_endpoint_count"] == 2
    assert "UNRESOLVED" in csv.read_text()
    assert "person-" not in csv.read_text()
    endpoint_table = json.loads(table.read_text())
    assert endpoint_table["outer_aggregate_crosscheck"] == "PASS_ALL_THREE_ANCHORS"
    assert endpoint_table["primary"]["P_candidate"]["all_primary_clauses_passed"] is False
    cap = json.loads(plot.read_text())["capability"][0]
    assert cap["axis"] == "CE_H_minus_CE_candidate"
    assert cap["estimate"] == pytest.approx(.02)
    assert cap["primary_clause_rescue"] is False
    with pytest.raises(FileExistsError):
        render_results.render_results(
            lock, _sha(lock), inner, _sha(inner), inference_path, _sha(inference_path),
            {a: (path, _sha(path)) for a, path in outer.items()}, csv, table, plot)


def test_renderer_rejects_changed_sign_or_unpinned_outer_report(tmp_path):
    lock, inner, inference_path, outer = _fixture(tmp_path)
    result = json.loads(inference_path.read_text())
    result["rows"][0]["passed_upper_bound"] = True
    inference_path.write_text(json.dumps(result, sort_keys=True))
    with pytest.raises(ValueError, match="bound|pass"):
        render_results.render_results(
            lock, _sha(lock), inner, _sha(inner), inference_path, _sha(inference_path),
            {a: (path, _sha(path)) for a, path in outer.items()},
            tmp_path / "FULL_RESULTS.csv", tmp_path / "ENDPOINT_TABLE.json",
            tmp_path / "PLOT_DATA.json")
    result["rows"][0]["passed_upper_bound"] = False
    inference_path.write_text(json.dumps(result, sort_keys=True))
    outer[1].write_text(outer[1].read_text() + " ")
    with pytest.raises(ValueError, match="SHA|hash|pin"):
        render_results.render_results(
            lock, _sha(lock), inner, _sha(inner), inference_path, _sha(inference_path),
            {a: (path, "0"*64 if a == 1 else _sha(path)) for a, path in outer.items()},
            tmp_path / "FULL_RESULTS.csv", tmp_path / "ENDPOINT_TABLE.json",
            tmp_path / "PLOT_DATA.json")


def test_renderer_rejects_falsified_anchor_aggregate_and_capability_rescue(tmp_path):
    lock, inner, inference_path, outer = _fixture(tmp_path)
    report = json.loads(outer[2].read_text())
    report["releases"]["A_selected"]["roles"][inference.TASK_ROLE]["candidate"]["U"] += .01
    outer[2].write_text(json.dumps(report, sort_keys=True))
    with pytest.raises(ValueError, match="outer aggregate contrast"):
        render_results.render_results(
            lock, _sha(lock), inner, _sha(inner), inference_path, _sha(inference_path),
            {a: (path, _sha(path)) for a, path in outer.items()},
            tmp_path / "FULL_RESULTS.csv", tmp_path / "ENDPOINT_TABLE.json",
            tmp_path / "PLOT_DATA.json")
    report = json.loads(inference_path.read_text())
    report["capability"]["primary_clause_rescue"] = True
    inference_path.write_text(json.dumps(report, sort_keys=True))
    with pytest.raises(ValueError, match="rescue"):
        render_results.render_results(
            lock, _sha(lock), inner, _sha(inner), inference_path, _sha(inference_path),
            None, tmp_path / "FULL_RESULTS.csv", tmp_path / "ENDPOINT_TABLE.json",
            tmp_path / "PLOT_DATA.json")


def test_renderer_cli_accepts_exact_three_anchor_pins(tmp_path):
    lock, inner, inference_path, outer = _fixture(tmp_path)
    args = ["--selection-lock", str(lock), "--lock-sha256", _sha(lock),
            "--inner-selection", str(inner), "--inner-selection-sha256", _sha(inner),
            "--inference", str(inference_path), "--inference-sha256", _sha(inference_path),
            "--output-csv", str(tmp_path / "FULL_RESULTS.csv"),
            "--output-table", str(tmp_path / "ENDPOINT_TABLE.json"),
            "--output-plot", str(tmp_path / "PLOT_DATA.json")]
    for anchor, path in outer.items():
        args += ["--outer-report", f"{anchor}={path}",
                 "--outer-sha256", f"{anchor}={_sha(path)}"]
    result = render_results.main(args)
    assert result["outer_aggregate_crosscheck"] == "PASS_ALL_THREE_ANCHORS"
    assert (tmp_path / "FULL_RESULTS.csv").is_file()
