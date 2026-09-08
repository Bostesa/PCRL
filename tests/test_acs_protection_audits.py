"""Fixed-candidate, validation-selection and isolation checks for ACS audits."""
import copy
import inspect
import json

import numpy as np
import pytest
import torch

from experiments.acs_protection_audits import (
    AUDIT_BUDGET, CANDIDATE_IDS, fit_primary_auditors, save_auditors, select_candidates,
)
from experiments.acs_transfer_heads import HEAD_BUDGET, load_candidate


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def arrays():
    rng = np.random.default_rng(166)
    x = rng.normal(size=(49, 4))
    xv = rng.normal(size=(21, 4)) + 1.5
    x[:, 3], xv[:, 3] = 4., 4.
    y = (x[:, 0]-.6*x[:, 1] > 0).astype(int)
    yv = (xv[:, 0]-xv[:, 1] > 0).astype(int)
    return x, y, xv, yv


@pytest.fixture
def tiny_budget():
    return {"mlp": {"hidden": [8, 4], "epochs": 2, "batch_size": 16, "validation_interval": 1},
            "histgb": {"max_iter": 3}}


def test_default_candidate_set_strengthens_only_declared_budgets():
    assert CANDIDATE_IDS == ("logistic", "mlp_0", "mlp_1", "hist_gb_20", "hist_gb_5")
    assert AUDIT_BUDGET["mlp"]["epochs"] == 120
    assert AUDIT_BUDGET["mlp"]["hidden"] == [64, 32]
    assert AUDIT_BUDGET["mlp"]["lr"] == .001
    assert AUDIT_BUDGET["mlp"]["validation_interval"] == 5
    assert AUDIT_BUDGET["histgb"]["max_iter"] == 150
    assert AUDIT_BUDGET["histgb"]["max_leaf_nodes"] == 15
    assert AUDIT_BUDGET["histgb"]["l2_regularization"] == 1.
    assert AUDIT_BUDGET["histgb"]["early_stopping"] is False
    assert HEAD_BUDGET["mlp"]["epochs"] == 40
    assert not any("test" in name for name in inspect.signature(fit_primary_auditors).parameters)


def test_all_candidates_share_fit_only_preprocessing_and_real_budgets(arrays, tiny_budget):
    x, y, xv, yv = arrays
    originals = [a.copy() for a in arrays]
    budget_before = copy.deepcopy(AUDIT_BUDGET)
    rng_before = torch.get_rng_state().clone()
    result = fit_primary_auditors(x, y, xv, yv, 2, 809, budget=tiny_budget)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert AUDIT_BUDGET == budget_before
    for actual, original in zip(arrays, originals):
        np.testing.assert_array_equal(actual, original)
    assert tuple(result["candidates"]) == CANDIDATE_IDS
    for key, candidate in result["candidates"].items():
        assert candidate.metadata["candidate_id"] == key
        np.testing.assert_allclose(candidate.preprocessing.mean, x.mean(0))
        np.testing.assert_allclose(candidate.preprocessing.scale, [*x.std(0)[:3], 1.])
        assert candidate.metadata["fit_hashes"] == result["metadata"]["fit_hashes"]
        assert candidate.metadata["validation_hashes"] == result["metadata"]["validation_hashes"]
    mlps = [result["candidates"][key] for key in ("mlp_0", "mlp_1")]
    for index, candidate in enumerate(mlps):
        meta = candidate.metadata
        assert meta["family"] == "mlp"
        assert meta["restart_index"] == index
        assert meta["initialization_seed"] == 809+index*10000
        assert meta["schedule_seed"] == 700809+index*10000
        assert meta["optimizer_steps"] == 8
        assert meta["training_row_exposures"] == 98
        assert [r["optimizer_steps"] for r in meta["validation_curve"]] == [0, 4, 8]
        assert not any(p.requires_grad for p in candidate.model.parameters())
    assert mlps[0].metadata["initial_state_hash"] != mlps[1].metadata["initial_state_hash"]
    assert result["metadata"]["total_mlp_optimizer_steps"] == 16
    assert result["metadata"]["total_mlp_training_row_exposures"] == 196
    for minimum in (20, 5):
        candidate = result["candidates"][f"hist_gb_{minimum}"]
        assert candidate.model.min_samples_leaf == minimum
        assert candidate.model.max_leaf_nodes == 15
        assert candidate.model.early_stopping is False
        assert candidate.metadata["actual_iterations"] == 3
        assert candidate.metadata["tree_fit_row_participations"] == 147
    json.dumps(result["metadata"], allow_nan=False)


def test_primary_and_auroc_choice_use_validation_of_same_ll_selected_candidates(arrays, tiny_budget):
    result = fit_primary_auditors(*arrays, 2, 77, budget=tiny_budget)
    scores = result["metadata"]["validation_scores"]
    loss_choice = min(scores, key=lambda k: (scores[k]["log_loss"], k))
    auc_choice = min(scores, key=lambda k: (-scores[k]["auroc"], k))
    assert result["selected_family"] == result["selected_log_loss"] == loss_choice
    assert result["selected_auroc"] == auc_choice
    assert "not AUROC" in result["metadata"]["auroc_scope"]
    for key in ("mlp_0", "mlp_1"):
        meta = result["candidates"][key].metadata
        best = min(meta["validation_curve"], key=lambda r: (r["validation_log_loss"], r["epoch"]))
        assert meta["selected_epoch"] == best["epoch"]
        assert scores[key]["log_loss"] == best["validation_log_loss"]


def test_selection_handles_ties_and_undefined_scores_without_silent_auc_pass():
    scores = {"mlp_1": {"log_loss": .2, "auroc": .9}, "mlp_0": {"log_loss": .2, "auroc": .8},
              "logistic": {"log_loss": .3, "auroc": .9}, "hist_gb_5": {"log_loss": None, "auroc": None}}
    selected = select_candidates(scores)
    assert selected["selected_log_loss"] == "mlp_0"
    assert selected["selected_auroc"] == "logistic"
    assert selected["undefined_auroc_candidates"] == ["hist_gb_5"]
    undefined = select_candidates({"x": {"log_loss": None, "auroc": float("nan")}})
    assert undefined["selected_log_loss"] is None
    assert undefined["selected_auroc"] is None


def test_equal_seed_and_fit_rows_match_schedules_across_releases(arrays, tiny_budget):
    x, y, xv, yv = arrays
    first = fit_primary_auditors(x, y, xv, yv, 2, 184, budget=tiny_budget)
    second = fit_primary_auditors(x[:, :2], y, xv[:, :2], yv, 2, 184, budget=tiny_budget)
    for key in ("mlp_0", "mlp_1"):
        a, b = (r["candidates"][key].metadata for r in (first, second))
        for field in ("schedule_hash", "optimizer_steps", "training_row_exposures", "schedule_seed"):
            assert a[field] == b[field]
        assert a["fit_hashes"]["y"] == b["fit_hashes"]["y"]
        assert a["parameter_count"] != b["parameter_count"]


def test_saved_suite_preserves_each_candidate_id_and_fitted_scaler(arrays, tiny_budget, tmp_path):
    result = fit_primary_auditors(*arrays, 2, 37, budget=tiny_budget)
    save_auditors(result, tmp_path/"audits")
    stored = json.loads((tmp_path/"audits/selection.json").read_text())
    assert stored == result["metadata"]
    for key, candidate in result["candidates"].items():
        reloaded = load_candidate(tmp_path/"audits"/key)
        assert reloaded.metadata["candidate_id"] == key
        np.testing.assert_array_equal(reloaded.predict_proba(arrays[2]), candidate.predict_proba(arrays[2]))
    with pytest.raises(FileExistsError):
        save_auditors(result, tmp_path/"audits")


def test_race_schema_and_missing_fit_evaluation_coverage_remain_explicit(tiny_budget):
    y = np.tile([0, 2, 8], 12)
    yv = np.tile([0, 8], 6)
    x = np.column_stack([y, np.arange(len(y)) % 4]).astype(float)
    xv = np.column_stack([yv, np.zeros(len(yv))]).astype(float)
    result = fit_primary_auditors(x, y, xv, yv, 9, 21, budget=tiny_budget)
    assert result["metadata"]["all_candidates_fit_coverage_complete"] is False
    assert result["metadata"]["validation_coverage_complete"] is False
    assert result["selected_auroc"] is None
    for candidate in result["candidates"].values():
        assert candidate.predict_proba(xv).shape == (len(yv), 9)
        score = candidate.metadata["validation_scores"]
        assert score["balanced_accuracy"] is None
        assert score["macro_auroc"] is None
        assert len(score["per_class"]) == 9
    json.dumps(result["metadata"], allow_nan=False)


def test_prior_fallback_is_auditable_and_not_numerical_fit_success(tiny_budget):
    x = np.arange(30).reshape(10, 3)
    result = fit_primary_auditors(x, np.zeros(10, dtype=int), x[:4], np.array([0, 1, 0, 1]),
                                  2, 52, budget=tiny_budget)
    assert result["metadata"]["all_candidates_numerically_fit"] is False
    assert result["metadata"]["all_candidates_fit_coverage_complete"] is False
    assert result["metadata"]["fallbacks"] == {key: "single_fitting_class" for key in CANDIDATE_IDS}
    assert result["metadata"]["total_mlp_optimizer_steps"] == 0
    for candidate in result["candidates"].values():
        assert candidate.metadata["effective_family"] == "prior"
