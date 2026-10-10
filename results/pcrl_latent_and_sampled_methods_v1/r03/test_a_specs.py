"""Data-free FIT spec admission fixture for the future effective A seal."""

import hashlib
import json
from pathlib import Path

import pytest

from results.pcrl_latent_and_sampled_methods_v1.r03.a_specs import fit_spec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def _fixture(tmp_path: Path):
    campaign = tmp_path / "results/pcrl_latent_and_sampled_methods_v1"
    portfolio = tmp_path / "results/pcrl_method_portfolio_v1"
    seal = _write(campaign / "CAMPAIGN_LOCK.json", '{"status":"sealed"}\n')
    source = _write(campaign / "worker.py", "# invented worker\n")
    audit = _write(campaign / "r01/SOURCE_AUDIT.json", "{}\n")
    repair = _write(tmp_path / "repair/m1/DATA_ACCESS_MANIFEST.json", "{}\n")
    bank = _write(portfolio / "shared/.private/tstar_score_bank_s0_v1/"
                  "BANK_ADMISSION_RECEIPT.json", "{}\n")
    private_stream = _write(campaign / "sampled_channel_v1/design/.private/"
                            "stream_roots_v1.json", "{}\n")
    stream = _write(campaign / "sampled_channel_v1/design/"
                    "STREAM_ROOT_HASH_MANIFEST.json",
                    json.dumps({"private_file_sha256": _sha(private_stream)}))
    plan = {f"A-FIT-X-{number:03d}": {
        "kind": "fit", "arm_id": "A_JOINT_STOCH", "seed": 0,
        "output_id": f"A-FIT-X-{number:03d}",
        "cpu_seconds": 300, "wall_seconds": 300,
    } for number in range(220)}
    amendment = _write(campaign / "A_EFFECTIVE.json", json.dumps({
        "status": "LOCALLY_SEALED_A_STAGE",
        "original_lock_sha256": _sha(seal),
        "implementation_source_sha256": {"worker.py": _sha(source)},
        "source_sha256": {"r01/SOURCE_AUDIT.json": _sha(audit)},
        "role_bindings": {
            "role_store_id": "UTILITY_REPAIR_ROLE_EXPORT_V1",
            "repair_manifest_sha256": _sha(repair),
            "source_audit_sha256": _sha(audit),
            "fit_row_unit_pair_sha256": "a" * 64,
            "source_binding_sha256_by_seed": {"s0": "b" * 64},
            "tstar_bank_receipt_sha256_by_seed": {"s0": _sha(bank)},
        },
        "realizations": {"stream_root_hash_manifest_sha256": _sha(stream)},
        "unit_plan": plan,
    }))
    queue_state = _write(campaign / "r03/.private/compute/state.json", json.dumps({
        "seal": {"sha256": _sha(seal), "path": str(seal)}, "occupancy": None,
        "amendments": [{"sha256": _sha(amendment), "stages": ["a"],
                        "path": str(amendment)}],
    }))
    private_parent = campaign / "r03/.private/a_units"
    private_parent.mkdir(parents=True, mode=0o700)
    private_parent.chmod(0o700)
    return campaign, portfolio, seal, amendment, repair, queue_state, private_parent


def test_fit_spec_pins_seal_role_source_bank_and_new_receipt(tmp_path):
    campaign, portfolio, seal, amendment, repair, queue_state, parent = _fixture(tmp_path)
    spec = fit_spec(
        unit_id="A-FIT-X-000", amendment_path=amendment,
        amendment_sha256=_sha(amendment), seal_path=seal,
        seal_sha256=_sha(seal), repair_manifest=repair,
        portfolio_package=portfolio, queue_state=queue_state,
        private_parent=parent)
    assert spec["outputs"] == [str(parent / "A-FIT-X-000/FIT_RECEIPT.json")]
    assert spec["cpu_reserve_s"] == spec["max_wall_s"] == 300
    assert spec["argv"][2].endswith("impl.run_admitted")
    pinned = {row["path"] for row in spec["sources"]}
    assert str(repair) in pinned
    assert str(campaign / "r01/SOURCE_AUDIT.json") in pinned
    assert any(path.endswith("BANK_ADMISSION_RECEIPT.json") for path in pinned)
    assert not (parent / "A-FIT-X-000").exists()
    state = json.loads(queue_state.read_text())
    state["amendments"].append({"sha256": "f" * 64, "stages": ["a"],
                                "path": str(amendment)})
    queue_state.write_text(json.dumps(state))
    with pytest.raises(PermissionError, match="not latest"):
        fit_spec(unit_id="A-FIT-X-000", amendment_path=amendment,
                 amendment_sha256=_sha(amendment), seal_path=seal,
                 seal_sha256=_sha(seal), repair_manifest=repair,
                 portfolio_package=portfolio, queue_state=queue_state,
                 private_parent=parent)
