"""Data-free ATT-only engineering probe contract."""

from pathlib import Path

import numpy as np

from results.pcrl_utility_reference_repair_v1.r2 import control_fit
from results.pcrl_utility_reference_repair_v1.r2.complete_views import (
    build_conditional_attack_views,
)
from results.pcrl_utility_reference_repair_v1.r2.test_complete_views import fixture


def test_control_fit_reads_only_att_sex_and_saves_inner_shuffled_prediction(
        tmp_path, monkeypatch):
    source, public, wire = fixture()
    views, _ = build_conditional_attack_views(source, wire, public)
    view = views["income"]
    monkeypatch.setattr(control_fit, "attack_fit_tune_split", lambda unit: {
        "fit": np.array([0, 1, 2, 3]), "tune": np.array([], dtype=np.int64)})
    sex_att = {"row_id": view.row_id.copy(), "unit": view.unit.copy(),
               "sex": np.array([0, 1, 0, 1], dtype=np.int64)}
    out = tmp_path / ".private" / "controls"
    report = control_fit.fit_control_models(
        view, sex_att, view, seed=0, output_dir=out)
    assert report["fresh_fit_count"] == 2
    assert report["INNER_SEX_read"] is False
    assert set(path.name for path in out.iterdir()) == set(control_fit.FILES)
    score = np.load(out / "shuffled_inner.npy", allow_pickle=False)
    assert score.shape == (4,)


def test_control_fit_receives_no_inner_sex_parameter():
    import inspect
    assert "sex_inner" not in inspect.signature(
        control_fit.fit_control_models).parameters
