"""Privacy-first inner runner reuses, rather than refits, complete H slates."""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import evaluate, run_privacy_inner
from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited_audit


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir(parents=True)
    source = private / "a0_inner_panel_d001"
    source.mkdir()
    index = private / "INPUT_INDEX.json"
    index.write_text("{}\n")
    fit = private / "a0_privacy_first_d001"
    fit.mkdir()
    (fit / "COMPLETE.json").write_text(json.dumps({"status": "COMPLETE",
        "inputs": {"anchor": 0, "branch": "A", "delta": .001}}))
    (fit / "RELEASE_SPEC.json").write_text('{"router":"T0"}\n')
    q_path = fit / "Q.npz"
    q = np.zeros((32, 17)); q[:, 0] = 1
    np.savez_compressed(q_path, Q=q)
    spec = {"Q": q, "router": "T0", "channel_artifact_path": str(q_path),
            "channel_artifact_sha256": _sha(q_path)}
    monkeypatch.setattr(run_privacy_inner.privacy_first_fit,
                        "load_release_spec", lambda path: spec)
    for ordinal, role in enumerate(evaluate.H_ROLES):
        slug = role.replace(":", "_").replace("/", "_")
        root = source / "H" / slug
        model = root / "models" / "m"
        model.mkdir(parents=True)
        (model / "weights.bin").write_bytes(bytes([ordinal]))
        registry = {"schema": 1, "role": role, "release_id": "H",
                    "seed": 26000 + ordinal, "slate": "standard",
                    "own_selection": "m", "models": {"m": {
                        "model_relative_directory": "models/m",
                        "model_sha256": inherited_audit.model_directory_hash(model)}},
                    "artifacts_sha256": inherited_audit._artifact_inventory(root)}
        (root / "own_registry.json").write_text(json.dumps(registry))
    (source / "PANEL_CONTRIBUTIONS.npz").write_bytes(b"private")
    report = {"schema": "pcrl-adaptive-inner-audit-v1", "anchor": 0,
              "slate": "standard", "outer_pool_opened": False,
              "index_sha256": _sha(index), "releases": {"A_selected": {}},
              "contributions_sha256": _sha(source / "PANEL_CONTRIBUTIONS.npz")}
    (source / "INNER_AUDIT.json").write_text(json.dumps(report))
    complete = {"schema": "pcrl-adaptive-inner-audit-complete-v1",
                "anchor": 0, "release_ids": ["A_selected"],
                "index_sha256": _sha(index),
                "artifacts": evaluate._inventory(source)}
    (source / "COMPLETE.json").write_text(json.dumps(complete))
    output = private / "a0_privacy_first_inner_d001"
    spec_index = private / "a0_privacy_first_specs.json"
    return source, fit, index, output, spec_index, spec


def test_reuses_byte_identical_h_slates_and_verifies_complete(tmp_path, monkeypatch):
    source, fit, index, output, spec_index, spec = _fixture(tmp_path, monkeypatch)
    called = []

    def fake_audit(anchor, releases, input_index, target, *, resume, slate):
        assert anchor == 0 and resume is True and slate == "standard"
        assert list(releases) == ["PrivacyFirst_selected"]
        for role in evaluate.H_ROLES:
            slug = role.replace(":", "_").replace("/", "_")
            assert inherited_audit._artifact_inventory(target / "H" / slug) == \
                   inherited_audit._artifact_inventory(source / "H" / slug)
            assert _sha(target / "H" / slug / "own_registry.json") == \
                   _sha(source / "H" / slug / "own_registry.json")
        called.append(True)
        report = {"schema": "pcrl-adaptive-inner-audit-v1", "anchor": anchor,
                  "slate": slate, "outer_pool_opened": False,
                  "index_sha256": _sha(index), "releases": {
                      "PrivacyFirst_selected": {"source": {"channel_artifact": {
                          "sha256": spec["channel_artifact_sha256"]}}}},
                  "contributions_sha256": "a" * 64}
        (target / "PANEL_CONTRIBUTIONS.npz").write_bytes(b"private")
        report["contributions_sha256"] = _sha(target / "PANEL_CONTRIBUTIONS.npz")
        (target / "INNER_AUDIT.json").write_text(json.dumps(report))
        receipt = {"schema": "pcrl-adaptive-inner-audit-complete-v1",
                   "anchor": anchor, "release_ids": ["PrivacyFirst_selected"],
                   "index_sha256": _sha(index),
                   "artifacts": evaluate._inventory(target)}
        (target / "COMPLETE.json").write_text(json.dumps(receipt))
        return report

    monkeypatch.setattr(run_privacy_inner.evaluate, "audit_panel", fake_audit)
    args = (0, index, fit, source, output, spec_index)
    result = run_privacy_inner.dispatch_privacy_inner(*args)
    assert len(called) == 1
    assert result["releases"]["PrivacyFirst_selected"]
    assert json.loads(spec_index.read_text())["main_inner_complete_sha256"] == \
           _sha(source / "COMPLETE.json")
    assert json.loads(spec_index.with_suffix(".binding.json").read_text())[
        "inner_complete_sha256"] == _sha(output / "COMPLETE.json")
    monkeypatch.setattr(run_privacy_inner.evaluate, "audit_panel",
                        lambda *a, **k: pytest.fail("completed panel must not refit"))
    assert run_privacy_inner.dispatch_privacy_inner(*args) == result


def test_refuses_corrupt_source_h_and_partial_destination(tmp_path, monkeypatch):
    source, fit, index, output, spec_index, _ = _fixture(tmp_path, monkeypatch)
    args = (0, index, fit, source, output, spec_index)
    first = evaluate.H_ROLES[0].replace(":", "_").replace("/", "_")
    (source / "H" / first / "models" / "m" / "weights.bin").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="inventory|hash|slate"):
        run_privacy_inner.dispatch_privacy_inner(*args)
    assert not output.exists()

    source, fit, index, output, spec_index, _ = _fixture(tmp_path / "other", monkeypatch)
    args = (0, index, fit, source, output, spec_index)
    (output / "H" / first).mkdir(parents=True)
    with pytest.raises((ValueError, FileExistsError), match="partial|mismatch|differs"):
        run_privacy_inner.dispatch_privacy_inner(*args)


def test_refuses_resealed_bad_h_registry_and_changed_index(tmp_path, monkeypatch):
    source, fit, index, output, spec_index, _ = _fixture(tmp_path, monkeypatch)
    role = evaluate.H_ROLES[0]
    slug = role.replace(":", "_").replace("/", "_")
    registry_path = source / "H" / slug / "own_registry.json"
    registry = json.loads(registry_path.read_text())
    registry["models"]["m"]["model_sha256"] = "0" * 64
    registry_path.write_text(json.dumps(registry))
    complete_path = source / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["artifacts"] = evaluate._inventory(source)
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="model hash"):
        run_privacy_inner.dispatch_privacy_inner(0, index, fit, source,
                                                 output, spec_index)
    assert not output.exists()

    source, fit, index, output, spec_index, _ = _fixture(tmp_path / "other", monkeypatch)
    index.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="main common inner panel"):
        run_privacy_inner.dispatch_privacy_inner(0, index, fit, source,
                                                 output, spec_index)


def test_cli_exposes_exact_sources():
    result = subprocess.run([sys.executable, "-m",
        "experiments.pcrl_adaptive_release_v1.run_privacy_inner", "--help"],
        capture_output=True, text=True, check=True)
    for option in ("--anchor", "--index", "--privacy-fit-dir",
                   "--main-inner-dir", "--output-dir", "--spec-index"):
        assert option in result.stdout
