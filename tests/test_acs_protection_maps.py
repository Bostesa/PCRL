"""Synthetic numerical and schema regressions; no ACS task/attacker fitting."""
from dataclasses import FrozenInstanceError
import inspect

from concept_erasure import LeaceEraser
import numpy as np
import pytest
import torch

from experiments import acs_protection_maps as maps


def fixture(n=360):
    rng = np.random.default_rng(7400)
    labels = {"SEX": np.arange(n) % 2, "RAC1P": (np.arange(n) // 2) % 9}
    z = np.column_stack([np.eye(2)[labels["SEX"]], np.eye(9)[labels["RAC1P"]]])
    x = rng.normal(size=(n, 14)) + z @ rng.normal(size=(11, 14)) + 4.5
    return x, labels, z


def test_fixed_solve_matches_documented_leace_in_full_rank_fixture():
    x, labels, z = fixture()
    fitted = maps.fit_joint_leace(x, labels)
    direct = LeaceEraser.fit(torch.tensor(x), torch.tensor(z), **maps.LEACE_OPTIONS)
    expected = direct(torch.tensor(x)).numpy()
    actual = fitted.apply(x, dtype=np.float64)
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    assert fitted.metadata["input_retained_rank"] == 14
    assert fitted.metadata["projection_retained_rank"] == 5
    assert fitted.metadata["after_float64"]["empirical_covariance_within_tolerance"]
    assert fitted.metadata["after_float64"]["cross_covariance_max_abs"] < 1e-12
    np.testing.assert_allclose(actual.mean(0), x.mean(0), atol=2e-13)
    assert fitted.metadata["unmodified_library_comparison"]["max_absolute_output_difference"] < 2e-11
    assert fitted.metadata["after_float32"]["dtype"] == "float64"
    assert fitted.apply(x).dtype == np.float32


def test_full_removal_is_exact_constant_even_for_new_rows():
    _, labels, z = fixture()
    # All three feature directions have genuine nonzero protected covariance.
    x = z[:, [0, 2, 3]] @ np.array([[1., .3, .2], [.2, 2., .1], [.1, .4, 3.]]) + .37
    fitted = maps.fit_joint_leace(x.astype(np.float32), labels)
    assert fitted.metadata["projection_retained_rank"] == 0
    assert fitted.metadata["exact_constant_output"]
    assert not np.any(fitted.matrix)
    unseen = np.random.default_rng(0).normal(size=(37, 3)) * 1e6
    for dtype in (np.float32, np.float64):
        released = fitted.apply(unseen, dtype=dtype)
        np.testing.assert_array_equal(released, np.tile(fitted.mean.astype(dtype), (37, 1)))
    assert fitted.metadata["after_float64"]["feature_covariance_rank"]["rank"] == 0
    assert fitted.metadata["after_float32"]["feature_covariance_rank"]["rank"] == 0


def test_float32_probability_simplex_noise_is_not_retained_as_a_direction():
    _, labels, _ = fixture()
    rng = np.random.default_rng(124)
    logits = rng.normal(size=(360, 5)) + (labels["SEX"] * .4)[:, None] * np.arange(5)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p = (p / p.sum(1, keepdims=True)).astype(np.float32)
    assert np.any(p.astype(np.float64).sum(1) != 1)
    fitted = maps.fit_joint_leace(p, labels)
    assert fitted.metadata["input_retained_rank"] == 4
    assert fitted.metadata["input_discarded_rank"] == 1
    assert fitted.metadata["discarded_covariance_trace"] < 1e-14
    assert fitted.metadata["exact_constant_output"]
    assert fitted.metadata["unmodified_library_comparison"]["status"] == "ok"


def test_fixed_schema_missing_masks_and_unsupported_columns_never_pass():
    x, labels, _ = fixture()
    labels = {k: v.copy() for k, v in labels.items()}
    labels["SEX"][:10] = -1
    labels["RAC1P"][10:20] = -1
    labels["RAC1P"][labels["RAC1P"] == 8] = -1
    fitted = maps.fit_joint_leace(x, labels)
    complete = (labels["SEX"] >= 0) & (labels["RAC1P"] >= 0)
    np.testing.assert_array_equal(fitted.fit_mask, complete)
    np.testing.assert_array_equal(fitted.known_masks, np.stack([labels[k] >= 0 for k in maps.SCHEMA]))
    metadata = fitted.metadata
    assert len(metadata["concept_columns"]) == 11
    assert metadata["coverage"]["RAC1P"]["support_complete_cases"][8] == 0
    assert not metadata["after_float64"]["valid_column_mask"][-1]
    assert not metadata["after_float64"]["schema_complete"]
    assert not metadata["after_float64"]["empirical_covariance_within_tolerance"]
    altered = x.copy()
    altered[~complete] = 1e9
    refit = maps.fit_joint_leace(altered, labels)
    np.testing.assert_array_equal(fitted.matrix, refit.matrix)
    np.testing.assert_array_equal(fitted.mean, refit.mean)
    diagnostics = maps.covariance_diagnostics(fitted.apply(x), labels)
    assert diagnostics["complete_rows"] == complete.sum()
    assert diagnostics["excluded_rows"] == (~complete).sum()
    assert not diagnostics["schema_complete"]


def test_constant_inputs_are_handled_without_an_undefined_whitening_solve():
    _, labels, _ = fixture()
    x = np.tile(np.array([.1, .3, -1.2]), (360, 1))
    fitted = maps.fit_joint_leace(x, labels)
    assert fitted.metadata["input_retained_rank"] == 0
    assert fitted.metadata["projection_retained_rank"] == 0
    assert fitted.metadata["before"]["feature_covariance_rank"]["rank"] == 0
    np.testing.assert_array_equal(fitted.apply(x, dtype=np.float64), x)


def test_map_arrays_and_metadata_are_immutable_and_save_is_exclusive(tmp_path):
    x, labels, _ = fixture()
    fitted = maps.fit_joint_leace(x, labels)
    fingerprint = fitted.fingerprint()
    for name in ("matrix", "mean", "fit_mask", "known_masks"):
        with pytest.raises(ValueError):
            getattr(fitted, name).setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        fitted.mean = np.zeros(14)
    copied = fitted.metadata
    copied["numerical_policy"]["input_covariance_rtol"] = 1.
    assert fitted.fingerprint() == fingerprint
    np.testing.assert_array_equal(fitted.apply(x), fitted.apply(x))
    path = tmp_path / "map.npz"
    fitted.save(path)
    restored = maps.FrozenAffineMap.load(path)
    assert restored.fingerprint() == fingerprint
    np.testing.assert_array_equal(restored.apply(x), fitted.apply(x))
    with pytest.raises(FileExistsError):
        fitted.save(path)
    assert fitted.fingerprint() == fingerprint


def test_source_labels_wrong_split_and_invalid_schema_are_rejected():
    x, labels, _ = fixture()
    with pytest.raises(ValueError, match="whitelist"):
        maps.fit_joint_leace(x, {**labels, "income_binary": labels["SEX"]})
    with pytest.raises(ValueError, match="representation_fit"):
        maps.fit_joint_leace(x, labels, fit_split="attacker_fit")
    for value in (2, np.nan, .5):
        invalid = {**labels, "SEX": labels["SEX"].astype(float)}
        invalid["SEX"][0] = value
        with pytest.raises(ValueError, match="zero-based"):
            maps.fit_joint_leace(x, invalid)
    with pytest.raises(ValueError, match="two complete"):
        maps.fit_joint_leace(x, {"SEX": np.full(len(x), -1), "RAC1P": labels["RAC1P"]})
    assert set(inspect.signature(maps.fit_joint_leace).parameters) == {"x_repr_fit", "labels", "fit_split"}
