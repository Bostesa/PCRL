"""Artificial matched-training regressions; never load or fit ACS data."""
import copy
import inspect

import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from experiments import acs_bottleneck_training as training
from experiments.acs_transfer_heads import _network
from experiments.acs_transfer_models import state_digest


def fixture(n=37, offset=0):
    rng = np.random.default_rng(408 + offset)
    x = (rng.normal(size=(n, 32)) + np.arange(32) / 3).astype(np.float32)
    source = {name: ((x[:, j] - j / 3) > 0).astype(np.int64)
              for j, name in enumerate(training.SOURCE_SCHEMA)}
    attributes = {"SEX": np.arange(n) % 2, "RAC1P": np.arange(n) % 9}
    source["income_binary"][::7] = -1
    attributes["RAC1P"][::13] = -1
    return x, source, attributes


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    x, source, attributes = fixture()
    xv, yv, _ = fixture(23, 1)
    out = tmp_path_factory.mktemp("bottleneck") / "pair"
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            result = training.train_pair(x, source, attributes, xv, yv, 2, out, miniature=True)
    finally:
        torch.set_num_threads(old)
    return result, out, (x, source, attributes, xv, yv)


def test_exact_warmed_clone_includes_both_adam_states_and_final_fixed_budget(pair):
    result, out, _ = pair
    metadata = result["metadata"]
    assert metadata["config"]["warm_base_epochs"] == 1
    assert metadata["config"]["warm_adversary_epochs"] == 1
    initial = torch.load(out / "initialization.pt", weights_only=True)
    assert initial["counters"] == {"epoch": 0, "mapper_optimizer_steps": 0, "adversary_optimizer_steps": 0}
    assert not initial["mapper_optimizer_state"]["state"]
    assert initial["adversary_optimizer_state"] is None
    assert initial["adversary_state"] == {}
    warm_base = torch.load(out / "warm_base.pt", weights_only=True)
    warm_adversary = torch.load(out / "warm_adversary.pt", weights_only=True)
    assert state_digest(warm_base["model_state"]) == state_digest(warm_adversary["model_state"])
    assert state_digest(warm_base["adversary_state"]) != state_digest(warm_adversary["adversary_state"])
    forks = {name: torch.load(out / name / "fork.pt", weights_only=True) for name in result["arms"]}
    assert training.tree_digest(forks["C_bottleneck"]) == training.tree_digest(forks["D_protected"])
    for name in result["arms"]:
        fork, meta = forks[name], metadata["arms"][name]
        assert meta["fork_hashes"] == metadata["shared_fork_hashes"]
        assert fork["mapper_optimizer_state"]["state"] and fork["adversary_optimizer_state"]["state"]
        assert meta["continuation_mapper_optimizer_steps"] == 6
        assert meta["continuation_adversary_optimizer_steps"] == 18
        assert meta["optimizer_counts_including_common"] == {"mapper_optimizer_steps": 9, "adversary_optimizer_steps": 21}
        assert meta["mapper_row_exposures"] == 37 * 2
        assert meta["adversary_row_exposures"] == 37 * 2 * 3
        for target, value in metadata["source_fit_coverage"].items():
            assert meta["source_valid_label_exposures"][target] == value["known"] * 2
            assert metadata["common_source_valid_label_exposures"][target] == value["known"]
        for target, value in metadata["attribute_fit_coverage"].items():
            assert meta["attribute_valid_label_exposures"][target] == value["known"] * 6
            assert metadata["common_attribute_valid_label_exposures"][target] == value["known"]
        assert meta["selected_epoch"] == 2
        assert meta["curve"][-1]["epoch"] == 2
        final = torch.load(out / name / "final.pt", weights_only=True)
        assert state_digest(final["model_state"]) == meta["final_model_hash"]
        assert all(int(v["step"]) == 9 for v in final["mapper_optimizer_state"]["state"].values())
        assert all(int(v["step"]) == 21 for v in final["adversary_optimizer_state"]["state"].values())
    c, d = (metadata["arms"][name] for name in ("C_bottleneck", "D_protected"))
    assert c["schedule_hash"] == d["schedule_hash"]
    assert c["final_model_hash"] != d["final_model_hash"]
    assert c["final_adversary_hash"] != d["final_adversary_hash"]
    assert c["potential_protection_mapper_l2_at_shared_fork"] > 0
    assert c["applied_protection_mapper_l2_at_shared_fork"] == 0
    assert d["applied_protection_mapper_l2_at_shared_fork"] > 0


def test_mapper_and_adversary_gradient_isolation_and_loss_sign(pair):
    result, _, (x, source, attributes, _, _) = pair
    model = copy.deepcopy(result["arms"]["D_protected"]["model"]).requires_grad_(True)
    adversaries = copy.deepcopy(result["arms"]["D_protected"]["adversaries"]).requires_grad_(True)
    xf = model.standardize(x[:16])
    ys = {k: torch.tensor(v[:16]) for k, v in source.items()}
    ya = {k: torch.tensor(v[:16]) for k, v in attributes.items()}
    priors = result["metadata"]["prior_entropies"]
    before = state_digest(model.state_dict())
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    training.adversary_update(model, adversaries, training._adam(adversaries.parameters()), xf, ya)
    assert state_digest(model.state_dict()) == before
    assert all(p.grad is None for p in model.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in adversaries.parameters())
    for parameter in adversaries.parameters():
        parameter.grad = torch.ones_like(parameter)
    before_adversary = state_digest(adversaries.state_dict())
    detail = training.mapper_update(model, adversaries, training._adam(model.parameters()), xf, ys, ya, priors, True)
    assert detail["objective"] == pytest.approx(detail["base_loss"] - .1 * detail["normalized_adversary_ce"], abs=1e-7)
    assert state_digest(adversaries.state_dict()) == before_adversary
    assert all(p.grad is None for p in adversaries.parameters())
    diag = training.gradient_diagnostics(model, adversaries, xf, ys, ya, priors)
    assert diag["base_first_weight_l2"] > 0 and diag["protection_first_weight_l2"] > 0
    assert diag["protection_source_decoder_gradient_l2"] == 0
    assert diag["protection_source_decoder_gradients_all_absent"]
    # C's mapper update is independent of the protected labels and adversary
    # weights when the current mapper/head/decoder and Adam states match.
    left, right = copy.deepcopy(model), copy.deepcopy(model)
    changed_attributes = {k: torch.zeros_like(v) for k, v in ya.items()}
    training.mapper_update(left, adversaries, training._adam(left.parameters()), xf, ys, ya, priors, False)
    training.mapper_update(right, adversaries, training._adam(right.parameters()), xf, ys, changed_attributes, priors, False)
    assert state_digest(left.state_dict()) == state_digest(right.state_dict())


def test_fixed_head_mask_mean_and_prior_entropy_are_explicit():
    logits = {name: torch.zeros(4, 1, requires_grad=True) for name in training.SOURCE_SCHEMA}
    labels = {name: torch.full((4,), -1, dtype=torch.long) for name in training.SOURCE_SCHEMA}
    labels["income_binary"] = torch.tensor([0, 1, -1, -1])
    mean, _, support = training.masked_source_bce(logits, labels)
    assert float(mean.detach()) == pytest.approx(np.log(2) / 3)
    assert support == {"income_binary": 2, "civilian_at_work": 0, "public_coverage": 0}
    mean.backward()
    assert torch.count_nonzero(logits["income_binary"].grad[:2]) > 0
    assert torch.count_nonzero(logits["income_binary"].grad[2:]) == 0
    for name in ("civilian_at_work", "public_coverage"):
        assert torch.count_nonzero(logits[name].grad) == 0
    priors = training.prior_entropies({"SEX": torch.tensor([0, 0, 1, -1]), "RAC1P": torch.tensor([0, 1, 1, -1])})
    assert priors["SEX"]["entropy"] == pytest.approx(-(2 / 3 * np.log(2 / 3) + 1 / 3 * np.log(1 / 3)))
    assert priors["RAC1P"]["support"] == [1, 2, 0, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="entropy"):
        training.prior_entropies({"SEX": torch.tensor([0, 0]), "RAC1P": torch.tensor([0, 1])})


def test_release_is_frozen_and_standardization_uses_only_fitting_features(pair):
    result, _, (x, _, _, xv, _) = pair
    mean = x.astype(np.float64).mean(0)
    scale = x.astype(np.float64).std(0)
    for arm in result["arms"].values():
        model = arm["model"]
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert all(head.out_features == 1 for head in model.heads.values())
        assert not arm["adversaries"].training and not any(p.requires_grad for p in arm["adversaries"].parameters())
        np.testing.assert_array_equal(model.input_mean.numpy(), mean)
        np.testing.assert_array_equal(model.input_scale.numpy(), scale)
        before = state_digest(model.state_dict())
        release = model.release(xv)
        assert release.shape == (23, 16) and release.dtype == np.float32
        assert state_digest(model.state_dict()) == before
        assert set(model.source_probabilities(xv)) == set(training.SOURCE_SCHEMA)
        for target, adversary in arm["adversaries"].items():
            expected = _network(16, [64, 32], training.ATTRIBUTE_SCHEMA[target])
            assert tuple(adversary.state_dict()) == tuple(expected.state_dict())
            assert not any(isinstance(m, (nn.Dropout, nn.BatchNorm1d)) for m in adversary.modules())


def test_validation_changes_do_not_change_any_fitted_weights_or_optimizer_states(pair, tmp_path):
    result, _, (x, source, attributes, xv, yv) = pair
    changed = {k: np.where(v < 0, -1, 1 - v) for k, v in yv.items()}
    with threadpool_limits(limits=1):
        repeated = training.train_pair(x, source, attributes, xv * 3 + 100, changed, 2,
                                      tmp_path / "different_validation", miniature=True)
    for name in result["arms"]:
        for field in ("final_model_hash", "final_adversary_hash", "final_mapper_optimizer_hash", "final_adversary_optimizer_hash"):
            assert result["metadata"]["arms"][name][field] == repeated["metadata"]["arms"][name][field]


def test_reserved_labels_and_wrong_fitting_pool_are_rejected_before_writing(tmp_path):
    x, source, attributes = fixture()
    for injected in ({**source, "same_residence": source["income_binary"]},
                     {"income_binary": source["income_binary"]}):
        with pytest.raises(ValueError, match="whitelist"):
            training.train_pair(x, injected, attributes, x, source, 0, tmp_path / "invalid", miniature=True)
    with pytest.raises(ValueError, match="representation_fit"):
        training.train_pair(x, source, attributes, x, source, 0, tmp_path / "invalid", fit_pool="attacker_fit")
    assert not (tmp_path / "invalid").exists()
    assert not any("test" in key or "reserved" in key for key in inspect.signature(training.train_pair).parameters)
