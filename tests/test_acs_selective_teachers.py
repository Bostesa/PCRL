"""Toy-only teacher provenance, complete-case permutation and immutability."""
import copy
import inspect
import json

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from experiments import acs_selective_teachers as teachers
from experiments.acs_protection_maps import fit_joint_leace


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        yield
    torch.set_num_threads(before)


def example(n=181):
    rng = np.random.default_rng(909)
    labels = {"SEX": np.arange(n) % 2, "RAC1P": (np.arange(n)//2) % 9}
    labels["RAC1P"][labels["RAC1P"] == 3] = 0
    labels["SEX"][:3] = -1
    labels["RAC1P"][3:8] = -1
    raw = (rng.normal(size=(n, 32))*np.arange(1, 33)+3).astype(np.float32)
    raw[:, 4] = 7
    mean, std = raw.astype(np.float64).mean(0), raw.astype(np.float64).std(0)
    return raw[:, :16], labels, mean, np.where(std > 1e-12, std, 1.)


def test_joint_permutation_preserves_pairs_masks_and_fixed_seed():
    raw, labels, _, _ = example()
    original = copy.deepcopy(labels)
    result = teachers.paired_permutation(labels, 2, len(raw))
    rows = np.flatnonzero((labels["SEX"] >= 0) & (labels["RAC1P"] >= 0))
    expected = np.random.default_rng(20260910).permutation(len(rows))
    np.testing.assert_array_equal(result["permutation"], expected)
    for key in labels:
        np.testing.assert_array_equal(labels[key], original[key])
        np.testing.assert_array_equal(result["protected"][key][rows], labels[key][rows[expected]])
        np.testing.assert_array_equal(result["protected"][key][~result["complete_mask"]], labels[key][~result["complete_mask"]])
    assert result["metadata"]["joint_counts_preserved"]
    assert result["metadata"]["coverage"]["RAC1P"]["support_complete_cases"][3] == 0
    with pytest.raises(ValueError):
        result["permutation"].setflags(write=True)


def test_permutation_is_frozen_before_each_exact_historical_leace_fit(tmp_path, monkeypatch):
    raw, labels, mean, scale = example()
    directory = tmp_path/"teachers"
    called = []
    original = teachers.fit_joint_leace
    def observed(x, protected, *, fit_split):
        prefit = json.loads((directory/"teacher_prefit_freeze.json").read_text())
        assert prefit["any_teacher_fit_started"] is False
        assert (directory/"paired_permutation.npz").exists()
        called.append(copy.deepcopy(dict(protected)))
        return original(x, protected, fit_split=fit_split)
    monkeypatch.setattr(teachers, "fit_joint_leace", observed)
    bundle = teachers.build_teachers(raw, labels, 0, directory, original_mean=mean, original_scale=scale,
                                    fit_raw_rows=np.arange(len(raw), dtype=np.int64))
    assert len(called) == 2
    expected = fit_joint_leace(raw, labels)
    np.testing.assert_array_equal(expected.matrix, bundle["maps"]["E"].matrix)
    np.testing.assert_array_equal(expected.mean, bundle["maps"]["E"].mean)
    np.testing.assert_array_equal(bundle["fit_targets"]["R"], raw)
    np.testing.assert_array_equal(bundle["fit_targets"]["E"], expected.apply(raw))
    assert bundle["metadata"]["diagnostics"]["E"]["movement_from_raw_mean_mse"] > 0
    assert bundle["metadata"]["maps"]["E"]["after_float64"]["schema_complete"] is False
    assert bundle["metadata"]["maps"]["E"]["coverage"] == bundle["metadata"]["maps"]["S"]["coverage"]
    restored = teachers.load_teachers(directory)
    for name in ("E", "S"):
        assert restored["maps"][name].fingerprint() == bundle["maps"][name].fingerprint()
    for name in ("R", "E", "S"):
        np.testing.assert_array_equal(restored["fit_targets"][name], bundle["fit_targets"][name])
        with pytest.raises(ValueError):
            restored["fit_targets"][name].setflags(write=True)
    with pytest.raises(FileExistsError):
        teachers.build_teachers(raw, labels, 0, directory, original_mean=mean, original_scale=scale)
    json.dumps(bundle["metadata"], allow_nan=False)


def test_wrong_split_outcome_schema_and_reestimated_scale_rejected(tmp_path):
    raw, labels, mean, scale = example()
    with pytest.raises(ValueError, match="representation_fit"):
        teachers.build_teachers(raw, labels, 0, tmp_path/"bad", original_mean=mean, original_scale=scale, fit_pool="attacker_fit")
    with pytest.raises(ValueError, match="whitelist"):
        teachers.build_teachers(raw, {**labels, "same_residence": labels["SEX"]}, 0,
                                tmp_path/"bad", original_mean=mean, original_scale=scale)
    with pytest.raises(ValueError, match="original raw"):
        teachers.build_teachers(raw, labels, 0, tmp_path/"bad", original_mean=mean, original_scale=scale*2)
    with pytest.raises(ValueError, match="float32"):
        teachers.build_teachers(raw.astype(np.float64), labels, 0, tmp_path/"bad", original_mean=mean, original_scale=scale)
    assert not any("val" in key or "test" in key for key in inspect.signature(teachers.build_teachers).parameters)


def test_teacher_application_uses_no_labels_and_does_not_refit(tmp_path, monkeypatch):
    raw, labels, mean, scale = example()
    bundle = teachers.build_teachers(raw, labels, 1, tmp_path/"teachers", original_mean=mean, original_scale=scale)
    def forbidden(*args, **kwargs):
        raise AssertionError("No map fitting allowed during application")
    monkeypatch.setattr(teachers, "fit_joint_leace", forbidden)
    probe = raw[:1].copy()+7
    targets = teachers.apply_teachers(bundle["maps"], probe)
    assert set(targets) == {"R", "E", "S"}
    for key in ("E", "S"):
        np.testing.assert_array_equal(targets[key], bundle["maps"][key].apply(probe))
    assert set(inspect.signature(teachers.apply_teachers).parameters) == {"maps", "raw"}
