"""Affine fitting holdouts, original-scale residuals and undefined ratios."""
import inspect
import json

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from experiments import acs_selective_diagnostics as geometry
from experiments.acs_selective_teachers import build_teachers
from tests.test_acs_selective_teachers import example


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        yield
    torch.set_num_threads(before)


def test_affine_recovers_moved_coordinates_and_fits_intercept_only_on_fit():
    rng = np.random.default_rng(144)
    raw = rng.normal(size=(140, 16))
    release = raw*3+17
    decoder = geometry.fit_affine(release[:100], raw[:100])
    assert decoder["metadata"]["fit_rows"] == 100
    evaluated = geometry.evaluate_affine(decoder, release[100:], raw[100:])
    assert evaluated["mean_mse"] < 1e-24
    assert evaluated["mse_over_prior"] < 1e-24
    assert decoder["metadata"]["rank"] == 17
    before = decoder["coefficient"].copy()
    geometry.evaluate_affine(decoder, release[100:]+100, raw[100:]-50)
    np.testing.assert_array_equal(before, decoder["coefficient"])
    with pytest.raises(ValueError):
        decoder["coefficient"].setflags(write=True)
    with pytest.raises(ValueError, match="representation_fit"):
        geometry.fit_affine(release, raw, fit_pool="source_validation")
    assert set(inspect.signature(geometry.fit_affine).parameters) == {"release", "standardized_target", "fit_pool", "rcond"}


def test_negligible_targets_and_shifted_constants_have_undefined_ratios():
    rng = np.random.default_rng(619)
    release = rng.normal(size=(61, 16))
    target = np.zeros((61, 16))
    target[:, 0] = rng.normal(size=61)*1e-8
    fitted = geometry.fit_affine(release, target)
    result = geometry.evaluate_affine(fitted, release, target)
    assert result["mse_over_prior"] is None
    assert result["per_coordinate_mse_over_prior"] == [None]*16
    assert result["ratio_defined"] is False
    shifted = geometry.evaluate_affine(fitted, release, target+500)
    assert shifted["prior_mean_mse"] > 1e4
    assert shifted["mse_over_prior"] is None
    json.dumps(result, allow_nan=False)


def test_original_scale_retained_removed_components_and_sham_scope(tmp_path):
    raw, labels, mean, scale = example()
    bundle = build_teachers(raw, labels, 0, tmp_path/"teachers", original_mean=mean, original_scale=scale)
    pre = {"mean": mean.tolist(), "scale": scale.tolist()}
    components = geometry.components(raw, bundle["maps"], pre, include_sham=True)
    np.testing.assert_allclose(components["E"]+components["qE"], components["raw"], atol=1e-15)
    np.testing.assert_allclose(components["S"]+components["qS"], components["raw"], atol=1e-15)
    expected = (raw.astype(np.float64)-bundle["fit_targets"]["E"].astype(np.float64))/scale[:16]
    np.testing.assert_array_equal(components["qE"], expected)
    raw_val = raw[:19].copy()+.25
    raw_val[:, 4] = raw[0, 4]  # Keep the deliberately constant fitting coordinate constant.
    pca = {"representation_fit": raw, "source_validation": raw_val}
    releases = {name: {pool: values.copy() for pool, values in pca.items()} for name in ("E_condition", "S_condition")}
    fitted, report = geometry.fit_snapshots(releases, pca, pre, bundle["maps"], tmp_path/"geometry",
                                           {"E_condition": "E", "S_condition": "S"})
    assert set(report["snapshots"]["E_condition"]["targets"]) == {"raw", "E", "qE"}
    assert set(report["snapshots"]["S_condition"]["targets"]) == {"raw", "E", "qE", "S", "qS"}
    for name in releases:
        for target, record in report["snapshots"][name]["targets"].items():
            assert record["source_validation"]["mean_mse"] < 1e-10
            assert (tmp_path/"geometry"/record["artifact"]).exists()
    geometry.evaluate_snapshots(fitted, report, {name: raw_val for name in releases}, raw_val, pre, bundle["maps"])
    assert report["snapshots"]["E_condition"]["targets"]["raw"]["development_evaluation"]["mean_mse"] < 1e-20
    json.dumps(report, allow_nan=False)
