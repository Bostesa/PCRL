"""Data-free custody checks for the queued M4 endpoint worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import endpoint_worker as subject


def test_all_phase_outputs_refuse_wrong_path_before_role_read(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setattr(subject, "STUDY", study)
    monkeypatch.setattr(subject, "assert_running_unit", lambda *a: (_ for _ in ()).throw(
        AssertionError("queue should not be reached for a wrong endpoint path")))
    assert subject.PHASE_SEEDS == {"FIRST": (0,), "SCREEN": (0, 1),
                                   "FINALIST": (0, 1, 2)}
    for phase in subject.PHASE_SEEDS:
        wrong = argparse.Namespace(
            phase=phase, unit_id=f"M4-ENDPOINTS-{phase}",
            output_index=str(study / ".private/m4/endpoints/OTHER/INFERENCE_INDEX.json"),
            queue_root=str(study / ".private/queue"),
        )
        with pytest.raises(PermissionError, match="not frozen"):
            subject._pre_role_gate(wrong)


def test_done_child_must_match_presealed_spec_and_saved_output(tmp_path):
    queue = tmp_path / ".private/queue"
    specs = queue / "specs"
    specs.mkdir(parents=True)
    child_id = "M4-ATT-S0-PAIR-INCOME"
    output = tmp_path / ".private/child/ATTACK_INDEX.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n")
    sealed = {"unit_id": child_id, "outputs": [str(output)],
              "seal_sha256": "$LOCAL_SEAL_SHA256"}
    observed = {**sealed, "seal_sha256": "a" * 64}
    spec_path = specs / (child_id + ".json")
    spec_path.write_text(json.dumps(observed))
    state = {"gate": {"seal_sha256": "a" * 64}, "units": {
        child_id: {"status": "DONE", "spec_path": str(spec_path),
                   "spec_sha256": subject._sha(spec_path),
                   "output_sha256": {str(output): subject._sha(output)}}}}
    registry = {"unit_specs": {child_id: sealed}}
    _, digest = subject._done_output(child_id, output, queue, state, registry)
    assert digest == subject._sha(output)
    registry["unit_specs"][child_id] = {**sealed, "outputs": ["other"]}
    with pytest.raises(PermissionError, match="sealed potential spec"):
        subject._done_output(child_id, output, queue, state, registry)


def test_actual_probability_slice_needs_exact_component_and_width():
    view = SimpleNamespace(
        row_id=np.arange(3, dtype=np.int64),
        x=np.asarray([[10, .8, .2], [20, .2, .8], [30, .6, .4]],
                     dtype=np.float64),
        component_columns=(("selected_values", 0, 1), ("probabilities", 1, 3)),
    )
    assert np.array_equal(subject._wire_probability(view, "probabilities", 2),
                          view.x[:, 1:3])
    with pytest.raises(PermissionError, match="omits exact"):
        subject._wire_probability(view, "probabilities", 6)


def test_coalition_inherits_both_complete_local_reader_banks_exactly():
    local_income = {f"R{i}": np.asarray([.1, .9], dtype=np.float64)
                    for i in range(6)}
    local_occupation = {f"R{i}": np.asarray([.2, .8], dtype=np.float64)
                        for i in range(6)}
    coalition = {f"R{i}": np.asarray([.3, .7], dtype=np.float64)
                 for i in range(6)}
    coalition.update({f"IGNORE_R2__{name}": value.copy()
                      for name, value in local_income.items()})
    coalition.update({f"IGNORE_R1__{name}": value.copy()
                      for name, value in local_occupation.items()})
    bank = {"income": local_income, "occupation_group": local_occupation,
            "coalition": coalition}
    subject._check_reader_inheritance(bank)
    bank["coalition"]["IGNORE_R2__R0"][0] += 1e-7
    with pytest.raises(PermissionError, match="omits or changes"):
        subject._check_reader_inheritance(bank)
