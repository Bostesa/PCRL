"""Proposal planning must not open role arrays before their sealed stage."""

import json
import re
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


def test_all_frozen_unit_ids_and_dependencies_are_queue_safe():
    frozen = json.loads(proposal.OUTPUT.read_text())
    units = frozen["unit_map"]
    pattern = re.compile(r"[A-Z0-9][A-Z0-9_-]{2,119}\Z")
    assert all(pattern.fullmatch(uid) for uid in units)
    assert all(record["unit_id"] == uid for uid, record in units.items())
    assert all(set(record["deps"]) <= set(units) for record in units.values())
    assert all(pattern.fullmatch(input_record["producer_unit"])
               for record in units.values()
               for input_record in record["inputs"].values()
               if "producer_unit" in input_record)
