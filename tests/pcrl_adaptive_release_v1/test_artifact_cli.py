"""Synthetic archived T32 packaging tests; no ACS data are loaded."""
import hashlib
import json

import joblib
import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import artifact_cli, evaluate, fit_a, fit_controls, privacy_first_fit
from experiments.pcrl_task_aligned_cuts_v1.synthetic_fixture import SyntheticEncoder


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def fixture(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir(parents=True)
    a, p, c = [private / name for name in ("a0_A_center_d001", "a0_privacy_first_d001", "a0_A_controls_d001")]
    paths = {"U": a / "round_r01/channel/Q.npz", "P": p / "channel/Q.npz", "D17": c / "channels/D17/Q.npz"}
    for i, path in enumerate(paths.values()):
        path.parent.mkdir(parents=True)
        np.savez_compressed(path, Q=np.eye(17)[(np.arange(32) + i) % 17])
    encoder = private / "encoder.joblib"
    joblib.dump(SyntheticEncoder(), encoder)
    write(a / "COMPLETE.json", {"schema": "pcrl-adaptive-A-center-v1", "status": "COMPLETE", "anchor": 0, "delta": .001,
        "selected_round": 1, "encoder_sha256": sha(encoder), "artifact_sha256": fit_a._inventory(a)})
    write(p / "RELEASE_SPEC.json", {"schema": "pcrl-adaptive-privacy-first-T0-spec-v1", "router": "T0",
        "channel_relative_path": "channel/Q.npz", "channel_file_sha256": sha(paths["P"]),
        "channel_array_sha256": fit_controls._array_sha(np.load(paths["P"])["Q"])})
    write(p / "COMPLETE.json", {"schema": "pcrl-adaptive-privacy-first-fit-v1", "status": "COMPLETE",
        "artifact_sha256": privacy_first_fit._inventory(p)})
    write(c / "CONTROLS.json", {"schema": 1, "controls": {"D17": {"relative_path": "channels/D17/Q.npz",
        "Q_file_sha256": sha(paths["D17"])}}})
    write(c / "COMPLETE.json", {"status": "COMPLETE", "artifact_sha256": fit_controls._inventory(c)})
    canonical = {"U": "A_selected", "P": "PrivacyFirst_selected", "D17": "A_control_D17"}
    sources = {canonical[m]: {"router_kind": "T0", "channel_artifact": {"sha256": sha(path)},
        "channel_array_sha256": evaluate._array_sha(np.load(path)["Q"])} for m, path in paths.items()}
    lock = private / "SELECTION_LOCK.json"
    write(lock, {"schema": "pcrl-adaptive-selection-lock-v1", "status": "LOCKED", "assessment_year": 2018,
        "anchors": {"0": {"logical_to_canonical": {"U_candidate": canonical["U"], "P_candidate": canonical["P"], "D17": canonical["D17"]},
        "releases": sources}}})
    manifest = private / "MODEL_MANIFEST.json"
    verified = {"readback_sha256_verified": True, "restored_inventory_verified": True}
    write(manifest, {"schema": "pcrl-adaptive-trained-model-manifest-v1", "status": "EXPERIMENTAL_NO_ADVANTAGE",
        "assessment_year": 2018, "selection_lock_sha256": sha(lock), "anchors": {"0": {
        "encoder_sha256": sha(encoder),
        "A_center_d001": {**verified, "selected_round": 1, "channel_file_sha256": sha(paths["U"])},
        "privacy_first_d001": {**verified, "channel_file_sha256": sha(paths["P"])},
        "A_controls_d001": {**verified, "D17_channel_relative_path": "channels/D17/Q.npz",
                             "D17_channel_file_sha256": sha(paths["D17"])}}}})
    inputs = private / "inputs.npz"
    h = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.savez_compressed(inputs, x_a=np.zeros((2, 32), np.float32), h_a=h, release_ids=np.array(["id0", "id1"]))
    key = private / "replay.key"
    key.write_bytes(bytes(range(32)))
    key.chmod(0o600)
    monkeypatch.setattr(artifact_cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(artifact_cli.platform, "machine", lambda: "x86_64")
    return locals()


def emit(f, mode, path):
    unit = {"U": "a", "P": "p", "D17": "c"}[mode]
    return artifact_cli.emit(f["manifest"], sha(f["manifest"]), f["lock"], sha(f["lock"]), 0,
        mode, f["a"], f[unit], f["encoder"], f["inputs"], f["key"], path)


@pytest.mark.parametrize("mode,status", [("U", "EXPERIMENTAL_NO_ADVANTAGE"), ("P", "EXPERIMENTAL_NO_ADVANTAGE"), ("D17", "BASELINE_CONTROL")])
def test_locked_modes_emit_immutable_private_wire(tmp_path, monkeypatch, mode, status):
    f = fixture(tmp_path, monkeypatch)
    one = emit(f, mode, f["private"] / "one.npz")
    two = emit(f, mode, f["private"] / "two.npz")
    with np.load(one["wire_path"]) as x, np.load(two["wire_path"]) as y:
        assert set(x.files) == {"h_a", "token"}
        assert x["h_a"].dtype == f["h"].dtype and x["h_a"].tobytes() == f["h"].tobytes()
        assert np.array_equal(x["token"], y["token"])
        assert np.all((x["token"] >= 0) & (x["token"] < 17))
    assert one["model_status"] == status
    with pytest.raises(FileExistsError):
        emit(f, mode, f["private"] / "one.npz")


def test_reject_hidden_input_bad_key_tampered_channel_and_encoder(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    wire = f["private"] / "wire.npz"
    np.savez_compressed(f["inputs"], x_a=np.zeros((2, 32), np.float32), h_a=f["h"],
        release_ids=np.array(["id0", "id1"]), y=np.zeros(2))
    with pytest.raises(ValueError, match="allowed|field"):
        emit(f, "U", wire)
    np.savez_compressed(f["inputs"], x_a=np.zeros((2, 32), np.float32), h_a=f["h"], release_ids=np.array(["id0", "id1"]))
    f["key"].chmod(0o644)
    with pytest.raises(PermissionError, match="key"):
        emit(f, "U", wire)
    f["key"].chmod(0o600)
    f["paths"]["U"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="channel|SHA|inventory"):
        emit(f, "U", wire)
    assert not wire.exists()
    g = fixture(tmp_path / "second", monkeypatch)
    g["encoder"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="encoder|SHA"):
        emit(g, "P", g["private"] / "wire.npz")


def test_aggregate_delegates_to_renderer(monkeypatch, tmp_path):
    monkeypatch.setattr(artifact_cli.render_results, "render_results", lambda *a: {"primary_all_passed": False})
    result = artifact_cli.reproduce_aggregates("lock", "a"*64, "inner", "b"*64, "inf", "c"*64,
        None, tmp_path/"full.csv", tmp_path/"table.json", tmp_path/"plot.json")
    assert result["primary_all_passed"] is False


def test_verified_channel_rejects_hash_pinned_wrong_t32_shape(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    np.savez_compressed(f["paths"]["U"], Q=np.eye(17)[np.arange(16)])
    complete = json.loads((f["a"] / "COMPLETE.json").read_text())
    complete["artifact_sha256"] = fit_a._inventory(f["a"])
    write(f["a"] / "COMPLETE.json", complete)
    record = {"selected_round": 1, "channel_file_sha256": sha(f["paths"]["U"])}
    with pytest.raises(ValueError, match="32 states and 17 tokens"):
        artifact_cli._verified_channel("U", f["a"], record, 0, sha(f["encoder"]))


def test_emit_cli_uses_archived_t32_mode_and_lock(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    result = artifact_cli.main(["emit", "--model-manifest", str(f["manifest"]),
        "--manifest-sha256", sha(f["manifest"]), "--selection-lock", str(f["lock"]),
        "--lock-sha256", sha(f["lock"]), "--anchor", "0", "--mode", "P",
        "--a-center-dir", str(f["a"]), "--artifact-dir", str(f["p"]),
        "--encoder", str(f["encoder"]), "--input-npz", str(f["inputs"]),
        "--replay-key", str(f["key"]), "--wire-out", str(f["private"] / "wire.npz")])
    assert result["model_status"] == "EXPERIMENTAL_NO_ADVANTAGE"
    f["lock"].write_text(f["lock"].read_text() + " ")
    with pytest.raises(ValueError, match="pin|lock|SHA"):
        emit(f, "P", f["private"] / "another.npz")
