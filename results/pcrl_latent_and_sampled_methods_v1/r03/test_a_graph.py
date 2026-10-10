"""Data-free admission checks for the future sampled-channel unit graph."""

import hashlib
import json
from pathlib import Path

import pytest

from results.pcrl_latent_and_sampled_methods_v1.r03.a_graph import compile_graph


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    source = tmp_path / "worker.py"
    source.write_text("print('synthetic fixture')\n")
    seal = tmp_path / "CAMPAIGN_LOCK.json"
    seal.write_text('{"status":"LOCALLY_SEALED"}\n')
    amendment = tmp_path / "A_AMENDMENT.json"
    plan = {
        "A-FIT-X-S0": {
            "kind": "fit", "arm_id": "X", "seed": 0,
            "output_id": "A-FIT-X-S0", "cpu_seconds": 10,
            "wall_seconds": 10,
        },
        "A-Q-X-S0": {
            "kind": "q_preprocess", "arm_id": "X", "seed": 0,
            "output_id": "A-Q-X-S0", "fit_unit_id": "A-FIT-X-S0",
            "deps": ["A-FIT-X-S0"], "cpu_seconds": 5,
            "wall_seconds": 5,
        },
        "A-RELEASE-X-S0-ATT-R0": {
            "kind": "release", "arm_id": "X", "seed": 0,
            "role": "ATTACK_FIT_NEW", "realization": 0,
            "output_id": "A-RELEASE-X-S0-ATT-R0",
            "fit_unit_id": "A-FIT-X-S0", "deps": ["A-FIT-X-S0"],
            "cpu_seconds": 5, "wall_seconds": 5,
        },
        "A-RELEASE-X-S0-INNER-R0": {
            "kind": "release", "arm_id": "X", "seed": 0,
            "role": "INNER_SELECTION", "realization": 0,
            "output_id": "A-RELEASE-X-S0-INNER-R0",
            "fit_unit_id": "A-FIT-X-S0", "deps": ["A-FIT-X-S0"],
            "cpu_seconds": 5, "wall_seconds": 5,
        },
        "A-ATTACK-X-S0-R0": {
            "kind": "attack", "arm_id": "X", "seed": 0,
            "realization": 0, "output_id": "A-ATTACK-X-S0-R0",
            "fit_unit_id": "A-FIT-X-S0", "q_unit_id": "A-Q-X-S0",
            "release_unit_ids": {
                "ATT": "A-RELEASE-X-S0-ATT-R0",
                "INNER": "A-RELEASE-X-S0-INNER-R0",
            },
            "deps": ["A-FIT-X-S0", "A-RELEASE-X-S0-ATT-R0",
                     "A-RELEASE-X-S0-INNER-R0", "A-Q-X-S0"],
            "cpu_seconds": 10, "wall_seconds": 10,
        },
    }
    amendment.write_text(json.dumps({
        "status": "LOCALLY_SEALED_A",
        "original_lock_sha256": _digest(seal),
        "implementation_source_sha256": {"worker.py": _digest(source)},
        "unit_plan": plan,
    }))
    return seal, amendment, source, plan


def test_graph_is_deterministic_and_dependency_first(tmp_path):
    seal, amendment, _, plan = _fixture(tmp_path)
    first = compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                          source_root=tmp_path, expected_units=5)
    second = compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                           source_root=tmp_path, expected_units=5)
    assert first == second
    assert first["unit_count"] == len(plan)
    assert first["topological_order"][0] == "A-FIT-X-S0"
    assert first["topological_order"][-1] == "A-ATTACK-X-S0-R0"


def test_graph_rejects_cycle_without_partial_index(tmp_path):
    seal, amendment, _, plan = _fixture(tmp_path)
    plan["A-FIT-X-S0"]["deps"] = ["A-ATTACK-X-S0-R0"]
    document = json.loads(amendment.read_text())
    document["unit_plan"] = plan
    amendment.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="cycle"):
        compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                      source_root=tmp_path, expected_units=5)


def test_graph_rejects_missing_cap_and_source_drift(tmp_path):
    seal, amendment, source, plan = _fixture(tmp_path)
    plan["A-FIT-X-S0"]["wall_seconds"] = None
    document = json.loads(amendment.read_text())
    document["unit_plan"] = plan
    amendment.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="wall_seconds"):
        compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                      source_root=tmp_path, expected_units=5)
    plan["A-FIT-X-S0"]["wall_seconds"] = 10
    document["unit_plan"] = plan
    amendment.write_text(json.dumps(document))
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="implementation source .* hash differs"):
        compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                      source_root=tmp_path, expected_units=5)


def test_graph_rejects_attack_without_inner_release_dependency(tmp_path):
    seal, amendment, _, plan = _fixture(tmp_path)
    plan["A-ATTACK-X-S0-R0"]["deps"].remove("A-RELEASE-X-S0-INNER-R0")
    document = json.loads(amendment.read_text())
    document["unit_plan"] = plan
    amendment.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="attack deps differ"):
        compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                      source_root=tmp_path, expected_units=5)


def test_continuous_reference_has_exact_two_role_parents_and_no_q(tmp_path):
    seal, amendment, _, plan = _fixture(tmp_path)
    for role, suffix in (("ATTACK_FIT_NEW", "ATT"),
                         ("INNER_SELECTION", "INNER")):
        unit_id = f"A-REFERENCE-RELEASE-U0-S0-{suffix}"
        plan[unit_id] = {"kind": "reference_release", "reference_id": "U0",
                         "seed": 0, "role": role, "output_id": unit_id,
                         "cpu_seconds": 5, "wall_seconds": 5}
    attack_id = "A-REFERENCE-ATTACK-U0-S0"
    plan[attack_id] = {
        "kind": "reference_attack", "reference_id": "U0", "seed": 0,
        "q_unit_id": None, "output_id": attack_id,
        "release_unit_ids": {"ATT": "A-REFERENCE-RELEASE-U0-S0-ATT",
                             "INNER": "A-REFERENCE-RELEASE-U0-S0-INNER"},
        "deps": ["A-REFERENCE-RELEASE-U0-S0-ATT",
                 "A-REFERENCE-RELEASE-U0-S0-INNER"],
        "trained_reader_and_probe_fits": 24,
        "cpu_seconds": 10, "wall_seconds": 10,
    }
    document = json.loads(amendment.read_text())
    document["unit_plan"] = plan
    amendment.write_text(json.dumps(document))
    index = compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                          source_root=tmp_path, expected_units=8)
    assert index["kind_counts"]["reference_attack"] == 1
    plan[attack_id]["release_unit_ids"]["INNER"] = "A-RELEASE-X-S0-INNER-R0"
    plan[attack_id]["deps"][1] = "A-RELEASE-X-S0-INNER-R0"
    amendment.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="reference attack INNER release identity"):
        compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                      source_root=tmp_path, expected_units=8)


def test_graph_rejects_pending_numeric_cap_but_allows_only_continuous_null_q(tmp_path):
    seal, amendment, _, plan = _fixture(tmp_path)
    plan["A-ATTACK-X-S0-R0"]["q_unit_id"] = None
    document = json.loads(amendment.read_text())
    document["unit_plan"] = plan
    amendment.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="pending null"):
        compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                      source_root=tmp_path, expected_units=5)
