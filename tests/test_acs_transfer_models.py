"""Small source-access and model-freezing regressions; no real ACS data."""
import inspect

import joblib
import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from experiments import acs_transfer_models as source


SCHEMA = {"income_binary": 2, "income_bins": 3, "esr": 6, "pubcov": 2, "joint": 8}


def fixture_arrays(n=65, offset=0):
    rng = np.random.default_rng(90+offset)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    labels = {name: np.arange(n, dtype=np.int64) % k for name, k in SCHEMA.items()}
    labels["income_binary"] = (x[:, 0] > 0).astype(np.int64)
    # Joint availability represents complete source-label observations.
    labels["joint"][::3] = -1
    labels["pubcov"][::7] = -1
    return x, labels


def test_source_whitelist_rejects_downstream_access_before_fit():
    x, labels = fixture_arrays()
    for fitter, options in [(source.fit_source_encoder, dict(lr=.001, epochs=0)),
                             (source.fit_source_tree_bank, dict(max_leaf_nodes=15, max_iter=1))]:
        injected = {**labels, "transportation": np.zeros(len(x), dtype=np.int64)}
        with pytest.raises(ValueError, match="whitelist"):
            fitter(x, injected, x, labels, SCHEMA, seed=0, **options)
        with pytest.raises(ValueError, match="whitelist"):
            fitter(x, injected, x, injected, {**SCHEMA, "transportation": 2}, seed=0, **options)
        parameters = inspect.signature(fitter).parameters
        assert not any("test" in name or "downstream" in name for name in parameters)


def test_missing_heads_contribute_zero_without_renormalizing_remaining_heads():
    logits = {name: torch.zeros(4, k, requires_grad=True) for name, k in SCHEMA.items()}
    labels = {name: torch.full((4,), -1, dtype=torch.long) for name in SCHEMA}
    labels["income_binary"] = torch.tensor([0, 1, -1, -1])
    loss, known = source.masked_source_loss(logits, labels)
    assert known
    assert float(loss.detach()) == pytest.approx(np.log(2)/5)
    loss.backward()
    assert torch.count_nonzero(logits["income_binary"].grad[:2]) > 0
    assert torch.count_nonzero(logits["income_binary"].grad[2:]) == 0
    assert all(torch.count_nonzero(logits[k].grad) == 0 for k in SCHEMA if k != "income_binary")


def test_source_fit_saves_true_initialization_selected_freeze_and_counts(tmp_path):
    x, y = fixture_arrays()
    xv, yv = fixture_arrays(31, offset=1)
    before = torch.random.get_rng_state().clone()
    with threadpool_limits(limits=1):
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            model, meta = source.fit_source_encoder(x, y, xv, yv, SCHEMA, seed=4, lr=.001,
                                                    epochs=2, batch_size=16, out_dir=tmp_path/"fit")
        finally:
            torch.set_num_threads(previous_threads)
    assert torch.equal(before, torch.random.get_rng_state())
    assert meta["optimizer_steps"] == 2*5
    assert meta["planned_batches"] == 10
    assert meta["selected_optimizer_steps"] <= meta["optimizer_steps"]
    assert meta["validation_history"][0]["optimizer_steps"] == 0
    assert meta["validation_loss"] == min(r["validation"]["mean_source_loss"] for r in meta["validation_history"])
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    digest = source.state_digest(model.state_dict())
    assert digest == meta["selected_state_hash"]
    assert model.release(xv).shape == (31, 32)
    for key, p in model.probabilities(xv).items():
        assert p.shape == (31, SCHEMA[key])
        assert p.dtype == np.float32
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    assert source.state_digest(model.state_dict()) == digest
    restored = source.SourceEncoder.load(tmp_path/"fit/selected.pt")
    np.testing.assert_array_equal(restored.release(xv), model.release(xv))
    initial = torch.load(tmp_path/"fit/initialization.pt", weights_only=True)
    assert initial["metadata"]["optimizer_steps"] == 0
    assert source.state_digest(initial["state_dict"]) == meta["initial_state_hash"]
    assert meta["coverage"]["source_fit"]["joint"]["missing"] == 22
    assert not meta["downstream_labels_received"] and not meta["final_test_received"]


def test_initialization_and_schedule_match_learning_rate_configs():
    x, y = fixture_arrays(17)
    a, ma = source.fit_source_encoder(x, y, x, y, SCHEMA, seed=7, lr=.001, epochs=1, batch_size=8)
    b, mb = source.fit_source_encoder(x, y, x, y, SCHEMA, seed=7, lr=.003, epochs=1, batch_size=8)
    assert ma["initial_state_hash"] == mb["initial_state_hash"]
    assert ma["schedule_hash"] == mb["schedule_hash"]
    assert ma["optimizer_steps"] == mb["optimizer_steps"] == 3
    assert ma["final_state_hash"] != mb["final_state_hash"]


def test_validation_labels_cannot_change_gradient_updates():
    x, y = fixture_arrays(25)
    xv, yv = fixture_arrays(17, offset=1)
    changed = {name: np.where(value >= 0, (value+1) % SCHEMA[name], -1)
               for name, value in yv.items()}
    _, first = source.fit_source_encoder(x, y, xv, yv, SCHEMA, seed=9, lr=.001, epochs=1, batch_size=8)
    _, second = source.fit_source_encoder(x, y, xv, changed, SCHEMA, seed=9, lr=.001, epochs=1, batch_size=8)
    assert first["final_state_hash"] == second["final_state_hash"]
    assert first["schedule_hash"] == second["schedule_hash"]
    assert first["final_validation"] != second["final_validation"]


def test_tree_preserves_absent_class_columns_and_masked_rows(tmp_path):
    x, y = fixture_arrays(65)
    xv, yv = fixture_arrays(31, offset=1)
    y["esr"] = np.where(y["esr"] == 5, -1, y["esr"])
    y["pubcov"][:] = 1
    with threadpool_limits(limits=1):
        bank, meta = source.fit_source_tree_bank(x, y, xv, yv, SCHEMA, seed=3,
                                                max_leaf_nodes=15, max_iter=2, out_dir=tmp_path/"tree")
    p = bank.probabilities(xv)
    assert p["esr"].shape == (31, 6)
    np.testing.assert_array_equal(p["esr"][:, 5], 0)
    np.testing.assert_array_equal(p["pubcov"][:, 1], 1)
    assert meta["training"]["esr"]["absent_schema_classes"] == [5]
    assert meta["training"]["esr"]["fit_rows"] == 55
    assert meta["training"]["esr"]["boosting_iterations"] == 2
    assert meta["training"]["pubcov"]["boosting_iterations"] == 0
    assert np.isfinite(meta["validation_loss"])
    restored = joblib.load(tmp_path/"tree/source_tree_bank.joblib")
    np.testing.assert_array_equal(restored.probabilities(xv)["esr"], p["esr"])


def test_entire_source_head_missing_is_not_a_finite_validation_score():
    x, y = fixture_arrays(17)
    invalid = {k: v.copy() for k, v in y.items()}
    invalid["joint"][:] = -1
    with pytest.raises(ValueError, match="undefined"):
        source.fit_source_encoder(x, y, x, invalid, SCHEMA, seed=0, lr=.001, epochs=0)
