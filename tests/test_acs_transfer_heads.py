"""Regression checks for fixed-schema ACS transfer heads and auditors."""
import inspect
import json

import numpy as np
import pytest
import torch
from sklearn.metrics import log_loss, roc_auc_score

from experiments.acs_transfer_heads import (
    exposed_probabilities, fit_candidates, fit_prior, load_candidate, metrics,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(892)
    x = rng.normal(size=(57, 4))
    x[:, 3] = 7.
    y = (x[:, 0] + .4*x[:, 1] > 0).astype(int)
    xv = rng.normal(size=(23, 4)) + 2
    xv[:, 3] = 7.
    yv = (xv[:, 0] - xv[:, 1] > 0).astype(int)
    return x, y, xv, yv


@pytest.fixture
def tiny_budget():
    return {"mlp": {"hidden": [8, 4], "epochs": 3, "batch_size": 16,
                    "validation_interval": 1},
            "histgb": {"max_iter": 3, "min_samples_leaf": 3}}


def test_fit_api_cannot_receive_final_test_and_fits_scaler_on_fit_only(classification_data, tiny_budget):
    x, y, xv, yv = classification_data
    assert not any("test" in name for name in inspect.signature(fit_candidates).parameters)
    result = fit_candidates(x, y, xv, yv, 2, 23, families=("logistic", "mlp", "histgb"), budget=tiny_budget)
    for candidate in result["candidates"].values():
        np.testing.assert_allclose(candidate.preprocessing.mean, x.mean(0))
        expected = x.std(0)
        expected[3] = 1.
        np.testing.assert_allclose(candidate.preprocessing.scale, expected)
        assert np.max(np.abs(candidate.preprocessing.mean-np.vstack([x, xv]).mean(0))) > .1
        assert candidate.metadata["fit_rows"] == len(x)
    selection = min(result["metadata"]["validation_scores"],
                    key=lambda k: (result["metadata"]["validation_scores"][k]["log_loss"], k))
    assert result["selected_family"] == selection
    json.dumps(result["metadata"], allow_nan=False)


def test_mlp_records_real_initialization_budget_and_validation_minimum(classification_data, tiny_budget):
    x, y, xv, yv = classification_data
    before = torch.get_rng_state().clone()
    result = fit_candidates(x, y, xv, yv, 2, 58, families=("mlp",), budget=tiny_budget)
    assert torch.equal(before, torch.get_rng_state())
    model = result["candidates"]["mlp"]
    meta = model.metadata
    assert meta["optimizer_steps"] == 3*4
    assert meta["training_row_exposures"] == 3*len(x)
    assert meta["row_exposure_min"] == meta["row_exposure_max"] == 3
    assert meta["validation_curve"][0]["epoch"] == 0
    assert meta["validation_curve"][0]["optimizer_steps"] == 0
    best = min(meta["validation_curve"], key=lambda row: (row["validation_log_loss"], row["epoch"]))
    assert meta["selected_epoch"] == best["epoch"]
    assert meta["selected_optimizer_steps"] == best["optimizer_steps"]
    assert metrics(yv, model.predict_proba(xv), 2)["log_loss"] == pytest.approx(best["validation_log_loss"])
    assert meta["initial_state_hash"] != meta["final_state_hash"]
    assert not any(p.requires_grad for p in model.model.parameters())


def test_validation_labels_do_not_change_fixed_fits_or_mlp_training(classification_data, tiny_budget):
    x, y, xv, yv = classification_data
    original = fit_candidates(x, y, xv, yv, 2, 90, families=("logistic", "mlp", "histgb"), budget=tiny_budget)
    changed = fit_candidates(x, y, xv, 1-yv, 2, 90, families=("logistic", "mlp", "histgb"), budget=tiny_budget)
    for family in ("logistic", "histgb"):
        np.testing.assert_array_equal(original["candidates"][family].predict_proba(xv),
                                      changed["candidates"][family].predict_proba(xv))
    first, second = (r["candidates"]["mlp"].metadata for r in (original, changed))
    for key in ("initial_state_hash", "final_state_hash", "schedule_hash", "optimizer_steps"):
        assert first[key] == second[key]
    assert first["validation_hashes"]["y"] != second["validation_hashes"]["y"]
    assert original["candidates"]["histgb"].model.early_stopping is False
    assert original["candidates"]["histgb"].metadata["actual_iterations"] == 3


@pytest.mark.parametrize("family", ["logistic", "mlp", "histgb"])
def test_saved_selected_weights_preserve_original_preprocessing(classification_data, tiny_budget, family, tmp_path):
    x, y, xv, yv = classification_data
    candidate = fit_candidates(x, y, xv, yv, 2, 76, families=(family,), budget=tiny_budget)["candidates"][family]
    candidate.save(tmp_path/family)
    loaded = load_candidate(tmp_path/family)
    np.testing.assert_array_equal(candidate.predict_proba(xv), loaded.predict_proba(xv))
    np.testing.assert_array_equal(loaded.preprocessing.mean, x.mean(0))
    assert loaded.metadata == candidate.metadata
    if family == "mlp":
        assert not any(p.requires_grad for p in loaded.model.parameters())


def test_absent_fitting_race_categories_preserve_nine_column_probabilities(tiny_budget):
    y = np.tile([0, 3, 8], 15)
    x = np.column_stack([y, np.arange(len(y)) % 4]).astype(float)
    yv = np.tile(np.arange(9), 2)
    xv = np.column_stack([yv, np.zeros(len(yv))]).astype(float)
    result = fit_candidates(x, y, xv, yv, 9, 36, families=("logistic", "mlp", "histgb"), budget=tiny_budget)
    absent = [1, 2, 4, 5, 6, 7]
    for family, candidate in result["candidates"].items():
        probability = candidate.predict_proba(xv)
        assert probability.shape == (len(xv), 9)
        np.testing.assert_allclose(probability.sum(1), 1., atol=1e-6)
        assert candidate.metadata["fit_coverage_complete"] is False
        assert candidate.metadata["fit_support"] == [15, 0, 0, 15, 0, 0, 0, 0, 15]
        if family != "mlp":
            np.testing.assert_array_equal(probability[:, absent], 0.)
        score = result["metadata"]["validation_scores"][family]
        assert score["coverage_complete"] is True
        assert len(score["per_class"]) == 9
        assert np.isfinite(score["log_loss"])


def test_missing_evaluation_categories_are_explicit_and_do_not_define_macro_scores():
    y = np.array([0, 0, 2, 2])
    p = np.array([[.8, .1, .1], [.7, .2, .1], [.1, .2, .7], [.1, .1, .8]])
    result = metrics(y, p, 3)
    assert result["support"] == [2, 0, 2]
    assert result["coverage_complete"] is False
    assert result["balanced_accuracy"] is None
    assert result["observed_balanced_accuracy"] == 1.
    assert result["macro_auroc"] is None
    assert result["observed_macro_auroc"] == 1.
    assert result["per_class"][1]["recall"] is None
    assert result["per_class"][1]["auroc"] is None
    json.dumps(result, allow_nan=False)


def test_weighted_sensitivity_matches_explicit_metrics_and_effective_support():
    y = np.array([0, 0, 1, 1])
    p = np.array([[.9, .1], [.3, .7], [.6, .4], [.2, .8]])
    w = np.array([1., 2., 4., 1.])
    result = metrics(y, p, 2, weights=w)
    assert result["log_loss"] == pytest.approx(log_loss(y, p, sample_weight=w, labels=[0, 1]))
    assert result["auroc"] == pytest.approx(roc_auc_score(y, p[:, 1], sample_weight=w))
    assert result["accuracy"] == pytest.approx(2/8)
    assert result["balanced_accuracy"] == pytest.approx((1/3+1/5)/2)
    assert result["prevalence"] == pytest.approx([3/8, 5/8])
    assert result["per_class"][1]["precision"] == pytest.approx(1/3)
    omitted = metrics(y, p, 2, weights=[1, 1, 0, 0])
    assert omitted["raw_coverage_complete"] is True
    assert omitted["coverage_complete"] is False
    assert omitted["auroc"] is None
    assert omitted["balanced_accuracy"] is None
    empty = metrics(y, p, 2, weights=np.zeros(4))
    assert empty["log_loss"] is None
    assert empty["accuracy"] is None
    json.dumps(empty, allow_nan=False)


def test_prior_and_single_class_fallback_use_only_fitting_labels(tmp_path, tiny_budget):
    prior = fit_prior(np.array([0, 0, 2]), 3)
    np.testing.assert_allclose(prior.predict_proba(np.zeros((2, 4))), [[.5, 1/6, 1/3]]*2)
    x, xv = np.arange(36).reshape(12, 3), np.zeros((9, 3))
    result = fit_candidates(x, np.ones(12, dtype=int), xv, np.arange(9), 9, 83,
                            families=("logistic", "mlp", "histgb"), budget=tiny_budget)
    for family, candidate in result["candidates"].items():
        assert candidate.metadata["effective_family"] == "prior"
        assert candidate.metadata["fallback_reason"] == "single_fitting_class"
        assert candidate.metadata["optimizer_steps"] == 0
        np.testing.assert_allclose(candidate.predict_proba(xv)[:, 1], 13/21)
        candidate.save(tmp_path/family)
        np.testing.assert_array_equal(load_candidate(tmp_path/family).predict_proba(xv), candidate.predict_proba(xv))


def test_nonfinite_feature_fallback_is_explicit_and_invalid_schema_is_rejected(tiny_budget):
    x = np.array([[0., np.nan], [1., 2.]])
    result = fit_candidates(x, np.array([0, 1]), np.zeros((2, 2)), np.array([0, 1]), 2, 1,
                            families=("logistic",), budget=tiny_budget)
    assert result["candidates"]["logistic"].metadata["fallback_reason"] == "nonfinite_features"
    with pytest.raises(ValueError, match="outside the fixed class schema"):
        fit_candidates(np.ones((2, 2)), np.array([0, 2]), np.ones((2, 2)), np.array([0, 1]), 2, 1)
    with pytest.raises(ValueError, match="Probabilities"):
        metrics(np.array([0, 1]), np.array([[np.nan, .1], [.2, .8]]), 2)
    with pytest.raises(ValueError, match="Weights"):
        metrics(np.array([0, 1]), np.full((2, 2), .5), 2, weights=[1., -1.])


def test_numerical_fallback_retains_completed_optimizer_steps(classification_data, tiny_budget, monkeypatch):
    x, y, xv, yv = classification_data
    original = torch.nn.functional.cross_entropy
    calls = 0

    def fail_after_two_steps(*args, **kwargs):
        nonlocal calls
        calls += 1
        return torch.tensor(float("nan")) if calls == 3 else original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", fail_after_two_steps)
    candidate = fit_candidates(x, y.astype(bool), xv, yv.astype(bool), 2, 38,
                               families=("mlp",), budget=tiny_budget)["candidates"]["mlp"]
    assert candidate.metadata["effective_family"] == "prior"
    assert candidate.metadata["fitting_attempted"] is True
    assert candidate.metadata["optimizer_steps"] == 2
    assert candidate.metadata["training_row_exposures"] == 32
    assert candidate.metadata["prior_fitting_rows"] == len(y)


def test_exposed_target_control_is_actually_fit_through_all_auditor_families(tiny_budget):
    y = np.tile(np.arange(3), 25)
    yv = np.tile(np.arange(3), 7)
    x, xv = exposed_probabilities(y, 3), exposed_probabilities(yv, 3)
    np.testing.assert_array_equal(x.argmax(1), y)
    result = fit_candidates(x, y, xv, yv, 3, 66, families=("logistic", "mlp", "histgb"), budget=tiny_budget)
    for family, candidate in result["candidates"].items():
        assert candidate.metadata["family"] == family
        assert "fallback_reason" not in candidate.metadata
        if family != "mlp":
            assert result["metadata"]["validation_scores"][family]["accuracy"] == 1.
        else:
            assert candidate.metadata["optimizer_steps"] > 0
            assert candidate.metadata["initial_state_hash"] != candidate.metadata["final_state_hash"]
