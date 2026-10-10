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
        "A-ATTACK-X-S0-R0": {
            "kind": "attack", "arm_id": "X", "seed": 0,
            "realization": 0, "output_id": "A-ATTACK-X-S0-R0",
            "deps": ["A-FIT-X-S0", "A-Q-X-S0"],
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
                          source_root=tmp_path, expected_units=3)
    second = compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                           source_root=tmp_path, expected_units=3)
    assert first == second
    assert first["unit_count"] == len(plan)
    assert first["topological_order"] == [
        "A-FIT-X-S0", "A-Q-X-S0", "A-ATTACK-X-S0-R0"
    ]


def test_graph_rejects_cycle_without_partial_index(tmp_path):
    seal, amendment, _, plan = _fixture(tmp_path)
    plan["A-FIT-X-S0"]["deps"] = ["A-ATTACK-X-S0-R0"]
    document = json.loads(amendment.read_text())
    document["unit_plan"] = plan
    amendment.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="cycle"):
        compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                      source_root=tmp_path, expected_units=3)


def test_graph_rejects_missing_cap_and_source_drift(tmp_path):
    seal, amendment, source, plan = _fixture(tmp_path)
    plan["A-FIT-X-S0"]["wall_seconds"] = None
    document = json.loads(amendment.read_text())
    document["unit_plan"] = plan
    amendment.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="wall_seconds"):
        compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                      source_root=tmp_path, expected_units=3)
    plan["A-FIT-X-S0"]["wall_seconds"] = 10
    document["unit_plan"] = plan
    amendment.write_text(json.dumps(document))
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="implementation source .* hash differs"):
        compile_graph(amendment, _digest(amendment), seal, _digest(seal),
                      source_root=tmp_path, expected_units=3)
