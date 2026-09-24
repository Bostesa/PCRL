"""Independent fixture for validation-only selection through completed-panel pins."""

import hashlib
import json

from experiments.pcrl_adaptive_release_v1 import evaluate, selection_lock
from experiments.pcrl_adaptive_release_v1.audit import ROLES


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def _anchor_fixture(root, *, check_loss):
    panel = root / "a0_inner_panel_d001"
    panel.mkdir(parents=True)
    spec_path = root / "a0_INNER_SPECS.json"
    index_sha = "f" * 64
    releases = {}
    for release, validation_loss in (("A_selected", .4),
                                     ("A_control_D17", .5)):
        roles = {}
        for role in ROLES:
            roles[role] = {
                "selected_candidate": "own/selected",
                "candidate_validation_scores": {
                    "own/selected": {"U": validation_loss,
                                     "PWGTP": validation_loss}},
                "H_selected_candidate": "H/selected",
                "H_validation_scores": {
                    "H/selected": {"U": .6, "PWGTP": .6}},
                # The separately scored inner-check result must never choose
                # the release or move its validation score.
                "candidate": {"U": check_loss, "PWGTP": check_loss},
                "H": {"U": 1 - check_loss, "PWGTP": 1 - check_loss},
            }
        releases[release] = {"roles": roles}
    report = {"schema": "pcrl-adaptive-inner-audit-v1", "anchor": 0,
              "selection_role": "inner_selection", "score_role": "inner_check",
              "outer_pool_opened": False, "index_sha256": index_sha,
              "releases": releases}
    report_path = panel / "INNER_AUDIT.json"
    _write(report_path, report)
    complete_path = panel / "COMPLETE.json"
    _write(complete_path, {"schema": "pcrl-adaptive-inner-audit-complete-v1",
                           "anchor": 0,
                           "release_ids": sorted(releases),
                           "artifacts": evaluate._inventory(panel)})
    spec = {"schema": "pcrl-inner-audit-spec-index-v1", "anchor": 0,
            "delta": .001, "slate": "standard",
            "input_index_sha256": index_sha,
            "canonical_ids": sorted(releases),
            "aliases": {name: name for name in releases}}
    _write(spec_path, spec)
    binding_path = spec_path.with_suffix(".binding.json")
    _write(binding_path, {"schema": "pcrl-inner-panel-binding-v1",
                          "anchor": 0, "canonical_ids": spec["canonical_ids"],
                          "spec_index_sha256": _sha(spec_path),
                          "inner_complete_sha256": _sha(complete_path),
                          "inner_audit_sha256": _sha(report_path)})
    return index_sha


def test_completed_panel_inner_check_reversal_cannot_change_lock_input(tmp_path):
    first = tmp_path / "first" / "private"
    second = tmp_path / "second" / "private"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    index_sha = _anchor_fixture(first, check_loss=.001)
    _anchor_fixture(second, check_loss=99.)
    _, _, a = selection_lock._verified_anchor(first, 0, index_sha)
    _, _, b = selection_lock._verified_anchor(second, 0, index_sha)
    assert a["scores"] == b["scores"]
    assert a["scores"]["A_selected"][ROLES[0]]["U"] == .4
