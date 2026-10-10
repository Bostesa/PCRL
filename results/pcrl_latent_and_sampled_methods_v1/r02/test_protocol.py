import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.protocol import (
    reader_plan, unit_fit_forecast, score_release_tasks, fit_probability_specialists,
    run_three_view_attack,
)


def test_reader_plan_freezes_required_complete_and_specialist_readers():
    d = reader_plan(family="D", code_bits=True)
    a = reader_plan(family="A", code_bits=True)
    assert d.core == ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32", "CELL_A1", "CODE_BIT_LR_C1")
    assert a.core == d.core
    assert d.probability == a.probability == ("PROB_LR_C1", "PROB_HGB_15x200")
    assert unit_fit_forecast(d)["mandatory_fits"] == 30
    assert unit_fit_forecast(a)["mandatory_fits"] == 30


def test_task_endpoint_preserves_both_original_references_and_recall():
    truth = {"row_id": np.array([1, 2], dtype=np.int64), "unit": np.array([1, 2], dtype=np.int64),
             "income": np.array([0, 1]), "occupation": np.array([0, 1])}
    p_income = np.array([[.8, .2], [.3, .7]])
    p_occ = np.eye(6)[[0, 1]] * .9 + .1 / 6
    candidate = {**truth, "income": p_income, "occupation": p_occ}
    refs = {"U0": candidate, "Ucal": candidate}
    score = score_release_tasks(candidate, truth, refs)
    assert score["income"]["metrics"]["per_class_recall"] == [1.0, 1.0]
    assert score["income"]["references"]["U0"]["point_eligible"]
    assert score["occupation"]["references"]["Ucal"]["delta"]["brier"] == 0


def test_task_endpoint_rejects_row_swap_and_missing_reference():
    truth = {"row_id": np.array([1], dtype=np.int64), "unit": np.array([1], dtype=np.int64),
             "income": np.array([0]), "occupation": np.array([0])}
    candidate = {"row_id": np.array([2], dtype=np.int64), "unit": np.array([1], dtype=np.int64),
                 "income": np.array([[1., 0.]]), "occupation": np.array([[1., 0., 0., 0., 0., 0.]])}
    with pytest.raises(ValueError, match="identity"):
        score_release_tasks(candidate, truth, {"U0": candidate, "Ucal": candidate})


def test_probability_fit_requires_a_sealed_lock_before_any_numerical_work(tmp_path):
    with pytest.raises(ValueError, match="lock"):
        fit_probability_specialists(None, None, seed=0,
                                    private_dir=tmp_path / ".private" / "readers",
                                    lock_path=tmp_path / "missing.json",
                                    lock_sha256="0" * 64, queue_guard=lambda: None)


def test_attack_runner_requires_effective_lock_before_any_fit(tmp_path):
    with pytest.raises(ValueError, match="lock"):
        run_three_view_attack({}, {}, {}, family="D", seed=0,
                              private_root=tmp_path / ".private" / "unit",
                              lock_path=tmp_path / "missing.json", lock_sha256="0" * 64,
                              queue_guard=lambda: None)
