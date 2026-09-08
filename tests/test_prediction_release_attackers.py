"""Small isolation and budget checks for stronger frozen-prediction probes."""
import json

import numpy as np
import pytest
import torch

from experiments.prediction_release_attackers import fit_attackers, full_batch_schedule


def test_full_batches_preserve_exact_exposure_even_at_epoch_boundaries():
    schedule = full_batch_schedule(n=11, steps=7, batch_size=8, seed=5)
    assert schedule.shape == (7, 8)
    assert schedule.size == 56
    assert np.array_equal(schedule, full_batch_schedule(11, 7, 8, 5))
    counts = np.bincount(schedule.ravel(), minlength=11)
    assert counts.min() == 5
    assert counts.max() == 6


def test_mlp_saves_actual_update_budget_and_per_target_validation_minima(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(19, 1))
    y = np.column_stack([2 * x[:, 0] + 3, x[:, 0] ** 2 - 2])
    xv = rng.normal(size=(13, 1)) + 4
    yv = np.column_stack([2 * xv[:, 0] + 3, xv[:, 0] ** 2 - 2])
    initial_rng = torch.get_rng_state().clone()
    [probe] = fit_attackers(
        x, y, xv, yv, target_names=["U", "S"], seed=11, out_dir=tmp_path / "fit",
        kinds=("mlp",), config={"mlp": {"steps": 3, "batch_size": 8, "hidden": 4, "validation_interval": 1}},
    ).values()
    assert torch.equal(initial_rng, torch.get_rng_state())
    np.testing.assert_allclose(probe.preprocessing.x_mean, x.mean(0))
    np.testing.assert_allclose(probe.preprocessing.y_mean, y.mean(0))
    scores = probe.score(xv, yv)
    for restart, candidate in enumerate(probe.metadata["candidates"]):
        saved = np.load(tmp_path / "fit" / "mlp" / f"schedule_restart_{restart}.npz")
        assert saved["indices"].shape == (3, 8)
        assert saved["row_exposures"].sum() == 24
        final = torch.load(tmp_path / "fit" / "mlp" / f"final_restart_{restart}.pt", weights_only=True)
        assert all(float(state["step"]) == 3 for state in final["optimizer"]["state"].values())
        assert [row["optimizer_steps"] for row in candidate["validation_curve"]] == [0, 1, 2, 3]
    for target in ("U", "S"):
        best = min(
            (row["validation_mse"][target], candidate["restart_index"], row["optimizer_steps"])
            for candidate in probe.metadata["candidates"] for row in candidate["validation_curve"]
        )
        selected = probe.metadata["selection"][target]
        assert (selected["validation_mse"], selected["restart_index"], selected["optimizer_steps"]) == best
        assert scores[target]["mse"] == pytest.approx(best[0], abs=1e-10)
    assert probe.metadata["row_exposures_total"] == 48


def test_fixed_ols_and_boosting_are_unchanged_by_validation_labels(tmp_path):
    rng = np.random.default_rng(6)
    x = rng.normal(size=(48, 1))
    y = 3 * x + 7
    xv = rng.normal(size=(10, 1)) + 8
    yv = 3 * xv + 7
    settings = {"histgb": {"max_iter": 3, "min_samples_leaf": 4}}
    original = fit_attackers(x, y, xv, yv, target_names=["U"], seed=8,
                            out_dir=tmp_path / "original", kinds=("linear", "histgb"), config=settings)
    changed = fit_attackers(x, y, xv, yv + 1000, target_names=["U"], seed=8,
                           out_dir=tmp_path / "changed", kinds=("linear", "histgb"), config=settings)
    for kind in original:
        np.testing.assert_array_equal(original[kind].predict(xv), changed[kind].predict(xv))
        np.testing.assert_allclose(original[kind].preprocessing.y_mean, y.mean(0))
    np.testing.assert_allclose(original["linear"].predict(xv), yv, atol=1e-12)
    assert original["histgb"].models[0].n_iter_ == 3
    assert original["histgb"].models[0].early_stopping is False
    assert changed["histgb"].metadata["validation_scores"]["U"]["r2"] < 0
    metadata = json.loads((tmp_path / "original" / "metadata.json").read_text())
    assert metadata["fit_array_hashes"] == changed["linear"].metadata["fit_array_hashes"]
    assert metadata["validation_array_hashes"] != changed["linear"].metadata["validation_array_hashes"]
