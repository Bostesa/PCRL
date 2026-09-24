"""The optional LP fit unit is immutable and exposes the standard T0 law."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import evaluate, privacy_first_fit


def _tiny32():
    q = np.zeros((32, 17)); q[:, 0] = 1.
    u = np.zeros_like(q); w = np.zeros_like(q)
    u[0, 1] = 1.; w[0, 1] = 2.
    cuts = []
    for role in ("A/SEX", "A/RAC1P", "AB/SEX", "AB/RAC1P"):
        for weighting in ("U", "W"):
            coeff = np.zeros_like(q)
            if role == "AB/SEX":
                coeff[0, 1] = 1.
            cuts.append({"id": f"{role}/{weighting}/attack", "role": role,
                         "weighting": weighting, "coeff": coeff,
                         "rho": 0., "delta": .001, "floor": -.001})
    source = {"branch": "A", "anchor": 0, "delta": .001,
              "center_sha256": "a"*64, "selected_bank_sha256": "b"*64}
    return {"U": u, "W": w}, cuts, q, source


def _receipt_paths(tmp_path, source):
    center = tmp_path/"private/center"
    controls = tmp_path/"private/controls"
    center.mkdir(parents=True); controls.mkdir(parents=True)
    (center/"COMPLETE.json").write_text('{"synthetic_center":true}\n')
    (controls/"COMPLETE.json").write_text('{"synthetic_controls":true}\n')
    source["center_sha256"] = hashlib.sha256((center/"COMPLETE.json").read_bytes()).hexdigest()
    return center, controls


def test_fit_unit_saves_full_channel_and_reuses_only_verified_complete(tmp_path, monkeypatch):
    pair, cuts, d17, source = _tiny32()
    center, controls = _receipt_paths(tmp_path, source)
    monkeypatch.setattr(privacy_first_fit.privacy_first, "load_frozen_inputs",
                        lambda *args, **kwargs: (pair, cuts, d17, source))
    output = tmp_path/"private/privacy_first_a0"
    kwargs = dict(branch="A", anchor=0, delta=.001,
                  center_dir=center, controls_dir=controls,
                  output_dir=output, time_limit_seconds=5.)
    receipt = privacy_first_fit.run_fit_unit(**kwargs)
    assert receipt["status"] == "COMPLETE"
    assert receipt["tau"] == pytest.approx(.0005, abs=1e-8)
    assert set(receipt["artifact_sha256"]) == {"INPUTS.json", "FIT.json",
                                         "RELEASE_SPEC.json", "channel/Q.npz"}
    report = json.loads((output/"FIT.json").read_text())
    assert report["phase_one"]["minimum_common_violation"] == pytest.approx(0)
    assert report["replay"]["feasible"] is True
    assert report["dual_gap"] <= 1e-7
    spec = privacy_first_fit.load_release_spec(output)
    law = evaluate.token_law_for_release(spec, {"token_codes": np.array([0, 1])})
    assert np.array_equal(law, spec["Q"][[0, 1]])
    saved = (output/"channel/Q.npz").read_bytes()
    assert privacy_first_fit.run_fit_unit(**kwargs) == receipt
    assert (output/"channel/Q.npz").read_bytes() == saved
    (output/"channel/Q.npz").write_bytes(saved+b"tamper")
    with pytest.raises(ValueError, match="artifact|hash|inventory"):
        privacy_first_fit.run_fit_unit(**kwargs)


def test_fit_unit_rejects_changed_source_and_partial_unit(tmp_path, monkeypatch):
    pair, cuts, d17, source = _tiny32()
    center, controls = _receipt_paths(tmp_path, source)
    current = [source]
    monkeypatch.setattr(privacy_first_fit.privacy_first, "load_frozen_inputs",
                        lambda *args, **kwargs: (pair, cuts, d17, current[0]))
    output = tmp_path/"private/privacy_first_a0"
    kwargs = dict(branch="A", anchor=0, delta=.001,
                  center_dir=center, controls_dir=controls,
                  output_dir=output, time_limit_seconds=5.)
    privacy_first_fit.run_fit_unit(**kwargs)
    current[0] = {**source, "selected_bank_sha256": "c"*64}
    with pytest.raises(ValueError, match="source|input"):
        privacy_first_fit.run_fit_unit(**kwargs)
    partial = tmp_path/"private/partial"
    partial.mkdir(); (partial/"INPUTS.json").write_text("partial")
    with pytest.raises(FileExistsError, match="partial"):
        privacy_first_fit.run_fit_unit(**{**kwargs, "output_dir": partial})


def test_fit_unit_cli_declares_only_frozen_inputs_and_private_output():
    run = subprocess.run([sys.executable, "-m",
                          "experiments.pcrl_adaptive_release_v1.privacy_first_fit", "--help"],
                         capture_output=True, text=True, check=True)
    for option in ("--branch", "--anchor", "--delta", "--center-dir",
                   "--controls-dir", "--output-dir", "--time-limit-seconds"):
        assert option in run.stdout
