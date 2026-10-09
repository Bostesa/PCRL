"""No-role-read admission checks for R2's conditional branch."""

from __future__ import annotations

import hashlib
import json

import pytest

from results.pcrl_utility_reference_repair_v1.r2.admission import (
    admit_conditional_gate,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path):
    private = tmp_path / ".private"
    private.mkdir()
    packages = {}
    for seed in (0, 1):
        packages[str(seed)] = {}
        for task in ("income", "occupation_group"):
            relative = f"r1/F_{seed}_{task}.json"
            path = private / relative
            path.parent.mkdir(exist_ok=True)
            content = f"{seed}:{task}:weights".encode()
            path.write_bytes(content)
            packages[str(seed)][task] = {
                "relative_path": relative, "sha256": _sha(content),
            }
    gate = {
        "schema": "pcrl-utility-full-input-gate-v1",
        "status": "PASS_FULL_INPUT_POINT_GATE",
        "selected_recipe": "B",
        "common_role": "INNER_SELECTION",
        "common_row_unit_pair_sha256": "b" * 64,
        "source_science_lock_sha256": "a" * 64,
        "per_seed_task": {
            str(seed): {
                task: {"U0": True, "Ucal": True}
                for task in ("income", "occupation_group")
            } for seed in (0, 1)
        },
        "f_packages": packages,
    }
    path = tmp_path / "FULL_INPUT_GATE.json"
    raw = (json.dumps(gate, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return path, private, gate, _sha(raw)


def test_conditional_gate_admits_only_a_complete_hashed_f_receipt(tmp_path):
    gate_path, private, _, gate_sha = _fixture(tmp_path)
    admitted = admit_conditional_gate(
        gate_path, expected_gate_sha256=gate_sha,
        expected_source_lock_sha256="a" * 64,
        expected_common_row_unit_pair_sha256="b" * 64,
        private_root=private,
    )
    assert admitted.selected_recipe == "B"
    assert len(admitted.f_package_paths) == 4
    assert admitted.gate_sha256 == gate_sha


def test_conditional_gate_rejects_failed_task_and_changed_package(tmp_path):
    gate_path, private, gate, _ = _fixture(tmp_path)
    gate["per_seed_task"]["1"]["occupation_group"]["Ucal"] = False
    raw = (json.dumps(gate, sort_keys=True) + "\n").encode()
    gate_path.write_bytes(raw)
    with pytest.raises(PermissionError, match="point eligibility"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256=_sha(raw),
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )
    gate["per_seed_task"]["1"]["occupation_group"]["Ucal"] = True
    raw = (json.dumps(gate, sort_keys=True) + "\n").encode()
    gate_path.write_bytes(raw)
    (private / "r1/F_0_income.json").write_text("changed")
    with pytest.raises(PermissionError, match="F package"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256=_sha(raw),
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )


def test_conditional_gate_rejects_escape_and_wrong_receipt_sha(tmp_path):
    gate_path, private, gate, gate_sha = _fixture(tmp_path)
    with pytest.raises(PermissionError, match="gate SHA"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256="c" * 64,
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )
    gate["f_packages"]["0"]["income"]["relative_path"] = "../outside"
    raw = (json.dumps(gate, sort_keys=True) + "\n").encode()
    gate_path.write_bytes(raw)
    with pytest.raises(PermissionError, match="private root"):
        admit_conditional_gate(
            gate_path, expected_gate_sha256=_sha(raw),
            expected_source_lock_sha256="a" * 64,
            expected_common_row_unit_pair_sha256="b" * 64,
            private_root=private,
        )

