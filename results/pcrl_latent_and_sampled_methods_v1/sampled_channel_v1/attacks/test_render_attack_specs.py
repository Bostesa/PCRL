"""Data-free parent-custody and selection checks for A attack spec rendering."""

import hashlib
import json

import pytest

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks import (
    render_attack_specs as renderer,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_requires_completed_exact_queue_json_receipt(tmp_path):
    private = tmp_path / ".private"
    private.mkdir(mode=0o700)
    output = private / "fit"
    output.mkdir(mode=0o700)
    result = output / "FIT_RECEIPT.json"
    result.write_text('{"status":"SELECTED_FIT_POLICY"}\n')
    queue = tmp_path / "queue"
    receipt_dir = queue / "units" / "A-FIT-TEST"
    receipt_dir.mkdir(parents=True)
    spec = {"unit_id": "A-FIT-TEST", "outputs": [str(result)]}
    unit = {"status": "done", "spec_sha256": "a" * 64,
            "spec": spec, "output_sha256": {str(result): _sha(result)}}
    receipt = {"status": "done", "returncode": 0,
               "spec_sha256": unit["spec_sha256"],
               "output_sha256": unit["output_sha256"]}
    (receipt_dir / "RECEIPT.json").write_text(json.dumps(receipt))
    parent, _, root, docs = renderer._parent(
        {"units": {"A-FIT-TEST": unit}}, queue, "A-FIT-TEST",
        ("FIT_RECEIPT.json",))
    assert parent is spec and root == output
    assert docs["FIT_RECEIPT.json"]["status"] == "SELECTED_FIT_POLICY"
    result.write_text('{"status":"COMPLETE_NO_FEASIBLE_POLICY"}\n')
    with pytest.raises(ValueError, match="changed after R03 completion"):
        renderer._parent({"units": {"A-FIT-TEST": unit}}, queue,
                         "A-FIT-TEST", ("FIT_RECEIPT.json",))
    result.unlink()
    with pytest.raises(ValueError, match="absent or redirected"):
        renderer._parent({"units": {"A-FIT-TEST": unit}}, queue,
                         "A-FIT-TEST", ("FIT_RECEIPT.json",))


@pytest.mark.parametrize("schema,arm,fit,selected,passes", [
    ("SCV1-private-stochastic-fit-v1", "A_JOINT_STOCH", None,
     {"start": 0, "update": 300}, True),
    ("SCV1-private-stochastic-fit-v1", "A_JOINT_STOCH", None,
     None, False),
    ("SCV1-private-control-fit-v1", "A_JOINT_DET",
     {"complete": True, "success": True}, None, True),
    ("SCV1-private-control-fit-v1", "A_JOINT_DET",
     {"complete": True, "success": False}, None, False),
    ("SCV1-private-control-fit-v1", "A_JOINT_DET",
     {"complete": False, "success": True}, None, False),
])
def test_selected_fit_recognizes_exact_stochastic_and_control_schemas(
        tmp_path, schema, arm, fit, selected, passes):
    record = {"schema": schema, "status": "SELECTED_FIT_POLICY",
              "unit_id": "A-FIT-TEST", "arm_id": arm, "source_seed": 0,
              "selected": selected, "fit": fit}
    if passes:
        renderer._selected_fit(record, arm=arm, seed=0,
                               fit_id="A-FIT-TEST", directory=tmp_path)
    else:
        with pytest.raises(ValueError, match="frozen feasible policy"):
            renderer._selected_fit(record, arm=arm, seed=0,
                                   fit_id="A-FIT-TEST", directory=tmp_path)
    (tmp_path / "INCOMPLETE_RESOURCE.json").write_text("{}\n")
    with pytest.raises(ValueError, match="eligible frozen policy"):
        renderer._selected_fit(record, arm=arm, seed=0,
                               fit_id="A-FIT-TEST", directory=tmp_path)


def test_selected_null_fit_refused_before_role_parent_access(tmp_path, monkeypatch):
    plan = {"kind": "attack", "arm_id": "A_JOINT_STOCH", "seed": 0,
            "realization": 0, "fit_unit_id": "A-FIT-TEST",
            "release_unit_ids": {"ATT": "ATT", "INNER": "INNER"},
            "q_unit_id": "Q", "deps": ["A-FIT-TEST", "ATT", "INNER", "Q"],
            "trained_reader_and_probe_fits": 33, "output_id": "A-ATTACK-TEST",
            "cpu_seconds": 90, "wall_seconds": 90}
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    amendment = {"unit_plan": {"A-ATTACK-TEST": plan}}
    output = tmp_path / ".private" / "A-ATTACK-TEST"
    monkeypatch.setattr(renderer, "_base", lambda **kwargs: (
        campaign, amendment, plan, {}, tmp_path, output, {}))
    monkeypatch.setattr(renderer, "_parent", lambda *args, **kwargs: (
        {"stage": "a", "mode": "real_data", "amendment_sha256": "a" * 64,
         "arm_id": "A_JOINT_STOCH", "seed": 0}, {}, tmp_path,
        {"FIT_RECEIPT.json": {
            "schema": "SCV1-private-stochastic-fit-v1",
            "status": "COMPLETE_NO_FEASIBLE_POLICY", "unit_id": "A-FIT-TEST",
            "arm_id": "A_JOINT_STOCH", "source_seed": 0, "selected": None}}))
    monkeypatch.setattr(renderer, "_role_parent", lambda *args, **kwargs:
                        pytest.fail("role release was opened after null FIT"))
    with pytest.raises(ValueError, match="eligible frozen policy"):
        renderer.attack_spec(
            unit_id="A-ATTACK-TEST", amendment_path=tmp_path,
            amendment_sha256="a" * 64, seal_path=tmp_path,
            seal_sha256="b" * 64, role_manifest=tmp_path,
            queue_state=tmp_path, private_parent=tmp_path,
            downstream_manifest_path=tmp_path,
            downstream_manifest_sha256="c" * 64,
            downstream_registration_path=tmp_path,
            downstream_registration_sha256="d" * 64)


def test_inherited_reader_sources_require_sealed_hashes(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    pins = {}
    for name in renderer.SHARED:
        path = shared / name
        path.write_text(f"frozen {name}\n")
        pins[name] = _sha(path)
    amendment = {"attack_plan": {"inherited_shared_source_sha256": pins}}
    sources = {}
    renderer._locked_shared_sources(amendment, shared, sources)
    assert sources == {str((shared / name).resolve()): digest
                       for name, digest in pins.items()}
    (shared / renderer.SHARED[0]).write_text("postlock change\n")
    with pytest.raises(ValueError, match="changed after the A source freeze"):
        renderer._locked_shared_sources(amendment, shared, {})
    amendment["attack_plan"]["inherited_shared_source_sha256"].pop(
        renderer.SHARED[0])
    with pytest.raises(ValueError, match="exact inherited reader source freeze"):
        renderer._locked_shared_sources(amendment, shared, {})


def test_reference_parent_uses_reference_id_and_rejects_other_arm():
    spec = {"stage": "a", "mode": "real_data", "amendment_sha256": "a" * 64,
            "reference_id": "D0_SEQ12", "seed": 1,
            "role": "ATTACK_FIT_NEW", "realization": 0}
    renderer._expect_parent(spec, amendment_sha256="a" * 64,
                            arm="D0_SEQ12", seed=1, role="ATTACK_FIT_NEW",
                            draw=0)
    with pytest.raises(ValueError, match="arm/seed/role/draw"):
        renderer._expect_parent(spec, amendment_sha256="a" * 64,
                                arm="D0_SEQ21", seed=1,
                                role="ATTACK_FIT_NEW", draw=0)


def test_control_fit_builds_selected_wrapper_without_stochastic_selected_field(
        tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "A_DOWNSTREAM_SOURCE_MANIFEST.json").write_text("{}\n")
    (campaign / "REGISTRATION.json").write_text("{}\n")
    design = campaign / "sampled_channel_v1" / "design"
    design.mkdir(parents=True)
    (design / "STREAM_ROOT_HASH_MANIFEST.json").write_text(json.dumps({
        "release_realization_root_sha256": {"0": "d" * 64}}))
    fit_dir = tmp_path / ".private" / "fit"
    fit_dir.mkdir(parents=True, mode=0o700)
    fit_id, arm, unit_id = "A-FIT-CONTROL", "A_JOINT_DET", "A-ATTACK-CONTROL"
    policy = {"status": "SELECTED_FIT_POLICY", "fit_unit_id": fit_id,
              "fit_receipt_sha256": "f" * 64, "arm_id": arm,
              "source_seed": 0, "source_binding_sha256": "a" * 64,
              "release_id": "frozen-control", "mechanism_sha256": "b" * 64,
              "bank_sha256_by_recipient": {"r1": "1" * 64, "r2": "2" * 64},
              "policy_file_sha256": "c" * 64}
    (fit_dir / "POLICY_RECEIPT.json").write_text(json.dumps(policy))
    (fit_dir / "policy.npz").write_bytes(b"data-free policy placeholder")
    fit_record = {"schema": "SCV1-private-control-fit-v1",
                  "status": "SELECTED_FIT_POLICY", "unit_id": fit_id,
                  "arm_id": arm, "source_seed": 0,
                  "fit": {"complete": True, "success": True}}
    (fit_dir / "FIT_RECEIPT.json").write_text(json.dumps(fit_record))
    fit_digest = _sha(fit_dir / "FIT_RECEIPT.json")
    policy["fit_receipt_sha256"] = fit_digest
    (fit_dir / "POLICY_RECEIPT.json").write_text(json.dumps(policy))
    fit_spec = {"stage": "a", "mode": "real_data",
                "amendment_sha256": "e" * 64, "arm_id": arm,
                "seed": 0, "sources": []}
    fit_parent = (fit_spec,
                  {"output_sha256": {str(fit_dir / "FIT_RECEIPT.json"): fit_digest}},
                  fit_dir, {"FIT_RECEIPT.json": fit_record})
    plan = {"kind": "attack", "arm_id": arm, "seed": 0,
            "realization": 0, "fit_unit_id": fit_id,
            "release_unit_ids": {"ATT": "ATT", "INNER": "INNER"},
            "q_unit_id": "Q", "deps": [fit_id, "ATT", "INNER", "Q"],
            "trained_reader_and_probe_fits": 33, "output_id": unit_id,
            "cpu_seconds": 90, "wall_seconds": 90}
    monkeypatch.setattr(renderer, "_base", lambda **kwargs: (
        campaign, {"unit_plan": {unit_id: plan}}, plan, {}, tmp_path,
        tmp_path / ".private" / unit_id, {}))
    monkeypatch.setattr(renderer, "_parent", lambda *args, **kwargs: fit_parent)
    role = {"release_id": "frozen-control", "source_binding_sha256": "a" * 64,
            "mechanism_sha256": "b" * 64,
            "canonical_bank_sha256_by_recipient": policy[
                "bank_sha256_by_recipient"],
            "private_stream_root_sha256": "d" * 64}
    def role_parent(*args, **kwargs):
        value = dict(role)
        value["row_unit_pair_sha256"] = (
            "3" * 64 if kwargs["role"] == "ATTACK_FIT_NEW" else "4" * 64)
        return fit_dir, "7" * 64, value
    monkeypatch.setattr(renderer, "_role_parent", role_parent)
    monkeypatch.setattr(renderer, "_q_parent", lambda *args, **kwargs: (
        fit_dir, {"receipt_sha256": "8" * 64, "content_sha256": "9" * 64}))
    role_manifest = tmp_path / "role.json"
    role_manifest.write_text("{}\n")
    result = renderer.attack_spec(
        unit_id=unit_id, amendment_path=campaign / "A_STAGE_AMENDMENT.json",
        amendment_sha256="e" * 64, seal_path=campaign / "CAMPAIGN_LOCK.json",
        seal_sha256="6" * 64, role_manifest=role_manifest,
        queue_state=tmp_path / "state.json", private_parent=fit_dir,
        downstream_manifest_path=campaign / "A_DOWNSTREAM_SOURCE_MANIFEST.json",
        downstream_manifest_sha256="5" * 64,
        downstream_registration_path=campaign / "REGISTRATION.json",
        downstream_registration_sha256="4" * 64)
    assert result["attack_identity"]["policy_receipt_sha256"] == _sha(
        fit_dir / "POLICY_RECEIPT.json")
    assert result["argv"][2].endswith(".attacks.run_downstream_attack")
    assert result["deps"] == plan["deps"]
