"""Proposal planning must not open role arrays before their sealed stage."""

from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r2 import build_downstream_unit_proposal as proposal


def test_role_input_uses_manifest_sha_without_opening_role_file(tmp_path, monkeypatch):
    monkeypatch.setattr(proposal, "STUDY", tmp_path)
    monkeypatch.setattr(proposal, "REPO", tmp_path)
    key = "INNER_SELECTION:sex"
    manifest = {
        "role_file_relative_path": {key: ".private/role_export_v1/INNER_SELECTION/sex.npz"},
        "role_file_sha256": {key: "a" * 64},
    }
    inputs = {}
    proposal.role_input(inputs, manifest, key)
    assert inputs == {
        ".private/role_export_v1/INNER_SELECTION/sex.npz": {
            "sha256": "a" * 64,
            "source": "FROZEN_MANIFEST_ROLE_METADATA",
        }}
