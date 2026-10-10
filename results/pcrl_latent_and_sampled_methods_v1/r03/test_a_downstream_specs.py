"""Data-free selected FIT to release queue custody fixture."""

import hashlib
import json
from pathlib import Path

from results.pcrl_latent_and_sampled_methods_v1.r03.a_specs import fit_spec
from results.pcrl_latent_and_sampled_methods_v1.r03.a_downstream_specs import release_spec
from results.pcrl_latent_and_sampled_methods_v1.r03.test_a_specs import _fixture


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_spec_requires_selected_parent_and_pins_role_sources(tmp_path):
    campaign, portfolio, seal, amendment, repair, queue_state, parent = _fixture(tmp_path)
    token = portfolio / "shared/.private/tstar_score_bank_s0_v1/tokens_ATTACK_FIT_NEW.npz"
    token.write_bytes(b"invented tokens")
    bank_receipt = token.parent / "BANK_ADMISSION_RECEIPT.json"
    bank_receipt.write_text(json.dumps({"private_output_sha256": {token.name: _sha(token)}}))
    reference = repair.parent.parent / ".private/references_s0.npz"
    reference.parent.mkdir(mode=0o700)
    reference.write_bytes(b"invented references")
    repair.write_text(json.dumps({
        "role_file_relative_path": {"ATTACK_FIT_NEW:references_s0":
                                    ".private/references_s0.npz"},
        "role_file_sha256": {"ATTACK_FIT_NEW:references_s0": _sha(reference)},
    }))
    document = json.loads(amendment.read_text())
    document["role_bindings"]["repair_manifest_sha256"] = _sha(repair)
    document["role_bindings"]["tstar_bank_receipt_sha256_by_seed"]["s0"] = _sha(bank_receipt)
    document["unit_plan"].pop("A-FIT-X-001")
    release_id = "A-RELEASE-X-S0-ATT-R0"
    document["unit_plan"][release_id] = {
        "kind": "release", "arm_id": "A_JOINT_STOCH", "seed": 0,
        "role": "ATTACK_FIT_NEW", "realization": 0,
        "fit_unit_id": "A-FIT-X-000", "deps": ["A-FIT-X-000"],
        "output_id": release_id, "cpu_seconds": 30, "wall_seconds": 30,
    }
    amendment.write_text(json.dumps(document))
    state = json.loads(queue_state.read_text())
    state["amendments"][-1]["sha256"] = _sha(amendment)
    queue_state.write_text(json.dumps(state))
    selected = fit_spec(unit_id="A-FIT-X-000", amendment_path=amendment,
                        amendment_sha256=_sha(amendment), seal_path=seal,
                        seal_sha256=_sha(seal), repair_manifest=repair,
                        portfolio_package=portfolio, queue_state=queue_state,
                        private_parent=parent)
    fit_dir = parent / "A-FIT-X-000"
    fit_dir.mkdir(mode=0o700)
    fit_receipt = fit_dir / "FIT_RECEIPT.json"
    fit_receipt.write_text('{"status":"SELECTED_FIT_POLICY"}\n')
    (fit_dir / "POLICY_RECEIPT.json").write_text("{}\n")
    (fit_dir / "policy.npz").write_bytes(b"invented policy")
    state["units"] = {"A-FIT-X-000": {
        "status": "done", "spec": selected,
        "output_sha256": {str(fit_receipt): _sha(fit_receipt)},
    }}
    queue_state.write_text(json.dumps(state))
    spec = release_spec(unit_id=release_id, amendment_path=amendment,
                        amendment_sha256=_sha(amendment), seal_path=seal,
                        seal_sha256=_sha(seal), queue_state=queue_state,
                        private_parent=parent)
    assert spec["deps"] == ["A-FIT-X-000"]
    assert spec["outputs"] == [str(parent / release_id / "RELEASE_RECEIPT.json")]
    assert spec["argv"][2].endswith("impl.run_release")
    pins = {entry["path"] for entry in spec["sources"]}
    assert str(token) in pins and str(reference) in pins
    assert str(fit_receipt) in pins and str(fit_dir / "policy.npz") in pins
