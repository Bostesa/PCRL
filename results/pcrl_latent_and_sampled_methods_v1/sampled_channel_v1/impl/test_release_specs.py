"""Data-free custody checks for selected A release queue specifications."""

import hashlib
import json
from pathlib import Path

import pytest

from results.pcrl_latent_and_sampled_methods_v1.r03.a_specs import fit_spec
from results.pcrl_latent_and_sampled_methods_v1.r03.test_a_specs import _fixture

from .release_specs import selected_release_spec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _selected_fixture(tmp_path, arm_id="A_JOINT_STOCH"):
    campaign, portfolio, seal, amendment_path, repair, queue_state, parent = _fixture(tmp_path)
    seed, role = 0, "ATTACK_FIT_NEW"
    token = portfolio / f"shared/.private/tstar_score_bank_s{seed}_v1/tokens_{role}.npz"
    token.write_bytes(b"synthetic cells only")
    bank_receipt = token.parent / "BANK_ADMISSION_RECEIPT.json"
    _write_json(bank_receipt, {"private_output_sha256": {token.name: _sha(token)}})
    reference = repair.parent.parent / ".private/references_s0.npz"
    reference.parent.mkdir(mode=0o700)
    reference.write_bytes(b"synthetic reference only")
    _write_json(repair, {
        "role_file_relative_path": {f"{role}:references_s0": ".private/references_s0.npz"},
        "role_file_sha256": {f"{role}:references_s0": _sha(reference)},
    })
    stream_private = campaign / "sampled_channel_v1/design/.private/stream_roots_v1.json"
    stream_private.parent.chmod(0o700)
    stream_private.chmod(0o600)
    root_hashes = {str(i): hashlib.sha256(bytes([i + 1]) * 32).hexdigest()
                   for i in range(3)}
    seq_hash = hashlib.sha256(b"\x04" * 32).hexdigest()
    stream_manifest = campaign / "sampled_channel_v1/design/STREAM_ROOT_HASH_MANIFEST.json"
    _write_json(stream_manifest, {
        "private_file_sha256": _sha(stream_private),
        "release_realization_root_sha256": root_hashes,
        "seq_fit_root_sha256": seq_hash,
    })
    amendment = json.loads(amendment_path.read_text())
    amendment["role_bindings"]["repair_manifest_sha256"] = _sha(repair)
    amendment["role_bindings"]["tstar_bank_receipt_sha256_by_seed"]["s0"] = _sha(bank_receipt)
    bank_hashes = {"bank_r1": "1" * 64, "bank_r2": "2" * 64,
                   "action_r1": "3" * 64, "action_r2": "4" * 64}
    amendment["bank_action_sha256_by_seed"] = {"s0": bank_hashes}
    amendment["realizations"] = {
        "ids": [0, 1, 2], "release_root_sha256_by_realization": root_hashes,
        "seq_fit_root_sha256": seq_hash,
        "stream_root_hash_manifest_sha256": _sha(stream_manifest),
    }
    release_id = f"A-RELEASE-{arm_id}-S0-ATT-R0"
    amendment["unit_plan"]["A-FIT-X-000"]["arm_id"] = arm_id
    amendment["unit_plan"].pop("A-FIT-X-001")
    amendment["unit_plan"][release_id] = {
        "kind": "release", "arm_id": arm_id, "seed": 0,
        "role": role, "realization": 0, "fit_unit_id": "A-FIT-X-000",
        "deps": ["A-FIT-X-000"], "output_id": release_id,
        "cpu_seconds": 30, "wall_seconds": 30,
    }
    _write_json(amendment_path, amendment)
    state = json.loads(queue_state.read_text())
    state["amendments"][-1]["sha256"] = _sha(amendment_path)
    _write_json(queue_state, state)
    parent_spec = fit_spec(
        unit_id="A-FIT-X-000", amendment_path=amendment_path,
        amendment_sha256=_sha(amendment_path), seal_path=seal,
        seal_sha256=_sha(seal), repair_manifest=repair,
        portfolio_package=portfolio, queue_state=queue_state,
        private_parent=parent)
    fit_dir = parent / "A-FIT-X-000"
    fit_dir.mkdir(mode=0o700)
    fit_receipt = fit_dir / "FIT_RECEIPT.json"
    policy_receipt = fit_dir / "POLICY_RECEIPT.json"
    policy_file = fit_dir / "policy.npz"
    policy_file.write_bytes(b"synthetic selected package")
    common = {
        "arm_id": arm_id, "source_seed": 0,
        "source_binding_sha256": "b" * 64,
        "row_unit_pair_sha256": "a" * 64,
        "bank_sha256_by_recipient": {"r1": "1" * 64, "r2": "2" * 64},
        "action_sha256_by_recipient": {"r1": "3" * 64, "r2": "4" * 64},
    }
    stochastic = arm_id in {"A_JOINT_STOCH", "A_TASK_STOCH"}
    fit = {
        "schema": ("SCV1-private-stochastic-fit-v1" if stochastic else
                   "SCV1-private-control-fit-v1"),
        "status": "SELECTED_FIT_POLICY", "unit_id": "A-FIT-X-000",
        "a_amendment_sha256": _sha(amendment_path), **common,
    }
    if stochastic:
        fit.update({"selected": {"objective": 0.7, "start": 0, "update": 25},
                    "starts": {
            "identity_biased": {"completed": True,
                                "checkpoints": [{"update": 25, "feasible": True}]},
            "uniform_logit": {"completed": True, "checkpoints": []},
        }})
    else:
        detail = {"complete": True, "success": True}
        if arm_id in {"A_TASK_DET", "A_JOINT_DET"}:
            detail.update({"selection": {"objective": 0.7},
                           "starts": [{"start": "identity_biased",
                                       "terminal_metrics": {"objective": 0.7},
                                       "chosen_checkpoint_update": 25}]})
            selected_start, selected_update = "identity_biased", 25
        elif arm_id == "A_CENTRAL_PRODUCT":
            detail["diagnostic_best"] = [0.7, 0, 25]
            selected_start, selected_update = "identity_biased", 25
        else:
            selected_start, selected_update = "composite", 0
        fit["fit"] = detail
    _write_json(fit_receipt, fit)
    policy = {
        "schema": ("SCV1-selected-stochastic-policy-v1" if stochastic else
                   "SCV1-selected-control-policy-v1"),
        "status": "SELECTED_FIT_POLICY", "fit_unit_id": "A-FIT-X-000",
        "fit_receipt_sha256": _sha(fit_receipt),
        "policy_file_sha256": _sha(policy_file),
        "mechanism_sha256": "5" * 64,
        "selected_start": ("identity_biased" if stochastic else selected_start),
        "selected_update": (25 if stochastic else selected_update),
        **common,
    }
    _write_json(policy_receipt, policy)
    state["units"] = {"A-FIT-X-000": {
        "status": "done", "spec": parent_spec,
        "output_sha256": {str(fit_receipt): _sha(fit_receipt)},
    }}
    _write_json(queue_state, state)
    kwargs = dict(unit_id=release_id, amendment_path=amendment_path,
                  amendment_sha256=_sha(amendment_path), seal_path=seal,
                  seal_sha256=_sha(seal), queue_state=queue_state,
                  private_parent=parent)
    return kwargs, fit_receipt, policy_receipt, queue_state


@pytest.mark.parametrize("arm_id", [
    "A_JOINT_STOCH", "A_TASK_STOCH", "A_JOINT_DET", "A_TASK_DET",
    "A_CENTRAL_PRODUCT", "A_LOCAL", "A_SEQ12", "A_SEQ21",
])
def test_selected_release_spec_pins_parent_stream_role_and_file_output(tmp_path, arm_id):
    kwargs, fit_receipt, policy_receipt, _ = _selected_fixture(tmp_path, arm_id)
    spec = selected_release_spec(**kwargs)
    assert spec["deps"] == ["A-FIT-X-000"]
    assert spec["outputs"] == [str(kwargs["private_parent"] / kwargs["unit_id"] /
                                    "RELEASE_RECEIPT.json")]
    assert spec["cpu_reserve_s"] == spec["max_wall_s"] == 30
    assert spec["argv"][2].endswith(
        "impl.run_release" if arm_id in {"A_JOINT_STOCH", "A_TASK_STOCH"}
        else "controls.run_control_release")
    sources = {entry["path"]: entry["sha256"] for entry in spec["sources"]}
    assert sources[str(fit_receipt)] == _sha(fit_receipt)
    assert sources[str(policy_receipt)] == _sha(policy_receipt)
    assert any(path.endswith("release_specs.py") for path in sources)
    assert not (kwargs["private_parent"] / kwargs["unit_id"]).exists()


@pytest.mark.parametrize("change", ["no_selection", "infeasible_checkpoint",
                                    "swapped_policy", "stale_parent_hash"])
def test_selected_release_refuses_unselected_or_mismatched_parent(tmp_path, change):
    kwargs, fit_path, policy_path, state_path = _selected_fixture(tmp_path)
    if change in ("no_selection", "infeasible_checkpoint"):
        fit = json.loads(fit_path.read_text())
        if change == "no_selection":
            fit["status"] = "COMPLETE_NO_FEASIBLE_POLICY"
            fit["selected"] = None
            policy_path.unlink()
        else:
            fit["starts"]["identity_biased"]["checkpoints"][0]["feasible"] = False
        _write_json(fit_path, fit)
        state = json.loads(state_path.read_text())
        state["units"]["A-FIT-X-000"]["output_sha256"][str(fit_path)] = _sha(fit_path)
        _write_json(state_path, state)
    elif change == "swapped_policy":
        policy = json.loads(policy_path.read_text())
        policy["source_binding_sha256"] = "f" * 64
        _write_json(policy_path, policy)
    else:
        state = json.loads(state_path.read_text())
        state["units"]["A-FIT-X-000"]["output_sha256"][str(fit_path)] = "0" * 64
        _write_json(state_path, state)
    with pytest.raises((PermissionError, ValueError)):
        selected_release_spec(**kwargs)


@pytest.mark.parametrize("arm_id", ["A_TASK_DET", "A_JOINT_DET",
                                    "A_CENTRAL_PRODUCT", "A_LOCAL", "A_SEQ12"])
def test_control_release_refuses_swapped_selected_checkpoint(tmp_path, arm_id):
    kwargs, _, policy_path, _ = _selected_fixture(tmp_path, arm_id)
    policy = json.loads(policy_path.read_text())
    policy["selected_update"] += 25
    _write_json(policy_path, policy)
    with pytest.raises(PermissionError, match="selected checkpoint"):
        selected_release_spec(**kwargs)
