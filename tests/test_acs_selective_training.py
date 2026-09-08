"""Artificial selective-teacher, omitted-decoder, and exact-fork regressions."""
import copy

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from experiments import acs_bottleneck_training as t
from scripts.verify_acs_selective_training_regression import artificial_data, fitting_statistics, run as regression


@pytest.fixture(scope="module")
def units(tmp_path_factory):
    x, source, attributes = artificial_data(37, 45)
    xv, validation, _ = artificial_data(23, 46)
    teachers = {"R": x[:, :16].copy(), "E": x[:, :16] * .7, "S": x[:, :16] + .1}
    directory = tmp_path_factory.mktemp("selective")
    results = {}
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            for name in teachers:
                for rho in (.1, 0.):
                    results[name, rho] = t.train_selective_unit(x, source, attributes, xv, validation, 2,
                        directory / f"{name}_{rho}", teacher_targets=teachers[name], teacher_name=name,
                        reconstruction_weight=rho, fitting_statistics=fitting_statistics(x), miniature=True)
    finally:
        torch.set_num_threads(old)
    return results, directory, (x, source, attributes, xv, validation), teachers


def test_reviewed_e169_raw_rho_point1_update_path_is_bitwise_exact():
    result = regression()
    assert result["passed"] and len(result["checkpoints"]) == 7
    assert all(v["bitwise_exact"] for v in result["checkpoints"].values())


def test_same_raw_initialization_fixed_teacher_and_exact_forks(units):
    results, directory, (x, _, _, xv, _), teachers = units
    initial_hashes, schedules = set(), []
    for (name, rho), result in results.items():
        meta = result["metadata"]
        initial_hashes.add(meta["initialization_hashes"]["model"])
        schedules.append(meta["schedules"])
        assert list(result["arms"]) == ["C", "D"]
        assert meta["teacher"]["sha256"] == t.array_digest(teachers[name])
        assert meta["teacher"]["immutable_verified"] and meta["teacher"]["detached"]
        assert meta["teacher"]["attribute_label_informed"] == (name != "R")
        np.testing.assert_array_equal(meta["teacher"]["coordinate_scale"], fitting_statistics(x)["scale"][:16])
        assert meta["teacher"]["coordinate_scale"][12] == 1.
        for raw in (x, xv):
            np.testing.assert_allclose(result["snapshots"]["I"].release(raw), raw[:, :16], atol=1e-5, rtol=1e-5)
        initial_loss = meta["stage_gradient_diagnostics"]["initialization"]["preservation_loss"]
        assert (initial_loss < 1e-10) if name == "R" else (initial_loss > 1e-5)
        root = directory / f"{name}_{rho}"
        forks = [torch.load(root / arm / "fork.pt", weights_only=True) for arm in ("C", "D")]
        assert t.tree_digest(forks[0]) == t.tree_digest(forks[1])
        warm_base, warm_adv = (torch.load(root / filename, weights_only=True) for filename in ("warm_base.pt", "warm_adversary.pt"))
        assert t.state_digest(warm_base["model_state"]) == t.state_digest(warm_adv["model_state"])
        assert t.tree_digest(warm_base["mapper_optimizer_state"]) == t.tree_digest(warm_adv["mapper_optimizer_state"])
        for arm, record in meta["arms"].items():
            assert record["fork_hashes"] == meta["shared_fork_hashes"]
            assert record["optimizer_counts_including_common"] == {"mapper_optimizer_steps": 9, "adversary_optimizer_steps": 21}
            assert record["continuation_mapper_optimizer_steps"] == 6 and record["continuation_adversary_optimizer_steps"] == 18
            assert record["continuation_preservation_coefficient"] == 1.
            assert record["preservation_schedule"] == "persistent"
            for row in record["curve"]:
                assert row["preservation_coefficient"] == 1. and row["reconstruction_coefficient"] == rho
                assert row["protection_coefficient"] == (-.1 if arm == "D" else 0.)
        for snapshot, model in result["snapshots"].items():
            assert not model.training and not any(p.requires_grad for p in model.parameters())
            assert t.state_digest(model.state_dict()) == meta["snapshots"][snapshot]["model_hash"]
    assert len(initial_hashes) == 1 and all(schedule == schedules[0] for schedule in schedules)


def test_rho_zero_omits_decoder_gradients_and_all_adam_state(units):
    results, directory, _, _ = units
    for (name, rho), result in results.items():
        initial = result["snapshots"]["I"].decoder.state_dict()
        for arm, record in result["metadata"]["arms"].items():
            final = torch.load(directory / f"{name}_{rho}" / arm / "final.pt", weights_only=True)
            indices = record["decoder_optimizer_parameter_indices"]
            if rho == 0.:
                assert record["decoder_unchanged"] and record["decoder_gradients_absent"]
                assert record["decoder_optimizer_state_entries"] == 0
                assert all(i not in final["mapper_optimizer_state"]["state"] for i in indices)
                for key, value in initial.items():
                    assert torch.equal(final["model_state"]["decoder." + key], value)
            else:
                assert record["decoder_optimizer_state_entries"] == 4 and not record["decoder_unchanged"]
                assert all(int(final["mapper_optimizer_state"]["state"][i]["step"]) == 9 for i in indices)


def test_raw_applied_gradient_norms_cosines_and_disposable_observers(units):
    results, _, _, _ = units
    for (_, rho), result in results.items():
        meta = result["metadata"]
        for point in ("initialization", "after_base_warmup"):
            d = meta["stage_gradient_diagnostics"][point]
            assert d["hypothetical_preobserver_warmup"] and d["protection_raw_mapper_l2"] > 0
            assert d["diagnostic_observer_state_sha256"] == meta["adversary_initialization_hash"]
            assert d["protection_applied_mapper_l2"] == 0. and d["teacher_protection_applied_cosine"] is None
        diagnostics = list(meta["stage_gradient_diagnostics"].values())
        for arm in ("C", "D"):
            for point in ("shared_fork", "final"):
                d = meta["arms"][arm]["fixed_batch_gradient_diagnostics_at_" + point]
                assert not d["hypothetical_preobserver_warmup"]
                assert d["coefficients"]["protection"] == (-.1 if arm == "D" else 0.)
                diagnostics.append(d)
        for d in diagnostics:
            assert d["training_state_unchanged"] and d["torch_rng_unchanged"] and d["optimizer_steps"] == 0
            for name, coefficient in d["coefficients"].items():
                assert d[f"{name}_applied_mapper_l2"] == pytest.approx(abs(coefficient) * d[f"{name}_raw_mapper_l2"], rel=2e-6, abs=1e-12)
            if rho == 0.:
                assert d["reconstruction_raw_mapper_l2"] > 0 and d["reconstruction_applied_mapper_l2"] == 0
                assert d["teacher_reconstruction_applied_cosine"] is None
            for pair in ("teacher_reconstruction", "teacher_protection", "source_protection"):
                for convention in ("raw", "applied"):
                    value = d[f"{pair}_{convention}_cosine"]
                    assert value is None or -1.000001 <= value <= 1.000001


def test_teacher_term_isolation_and_fixed_input_scale(units):
    results, _, (x, source, attrs, _, _), teachers = units
    base = copy.deepcopy(results["E", .1]["snapshots"]["W"]).requires_grad_(True)
    left, right = copy.deepcopy(base), copy.deepcopy(base)
    source = {k: torch.tensor(v) for k, v in source.items()}
    attrs = {k: torch.tensor(v) for k, v in attrs.items()}
    teacher = torch.tensor(teachers["E"], requires_grad=True)
    observers = torch.nn.ModuleDict()
    priors = results["E", .1]["metadata"]["prior_entropies"]
    for model, beta in ((left, 0.), (right, 1.)):
        t.mapper_update(model, observers, t._adam(model.parameters()), model.standardize(x), source, attrs,
                        priors, False, teacher=teacher, preservation_beta=beta, reconstruction_weight=.1)
    assert teacher.grad is None
    assert t.state_digest(left.heads.state_dict()) == t.state_digest(right.heads.state_dict())
    assert t.state_digest(left.decoder.state_dict()) == t.state_digest(right.decoder.state_dict())
    assert t.state_digest(left.mapper.state_dict()) != t.state_digest(right.mapper.state_dict())


def test_validation_and_reserved_label_boundaries(units, tmp_path):
    results, _, (x, source, attrs, xv, validation), teachers = units
    changed = {k: np.where(v < 0, -1, 1-v) for k, v in validation.items()}
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            repeated = t.train_selective_unit(x, source, attrs, xv * 3 + 100, changed, 2, tmp_path / "changed",
                teacher_targets=teachers["E"], teacher_name="E", reconstruction_weight=0., fitting_statistics=fitting_statistics(x), miniature=True)
    finally:
        torch.set_num_threads(old)
    for arm in ("C", "D"):
        for field in ("final_model_hash", "final_adversary_hash", "final_mapper_optimizer_hash", "final_adversary_optimizer_hash"):
            assert repeated["metadata"]["arms"][arm][field] == results["E", 0.]["metadata"]["arms"][arm][field]
    default = dict(teacher_targets=teachers["E"], teacher_name="E", reconstruction_weight=0., fitting_statistics=fitting_statistics(x), miniature=True)
    for change in ({"teacher_name": "X"}, {"reconstruction_weight": .2}, {"fit_pool": "attacker_fit"},
                   {"teacher_name": "R"}, {"teacher_targets": np.zeros((37, 15))}, {"fitting_statistics": None}):
        with pytest.raises(ValueError):
            t.train_selective_unit(x, source, attrs, xv, validation, 2, tmp_path / "invalid", **{**default, **change})
        assert not (tmp_path / "invalid").exists()
    with pytest.raises(ValueError, match="whitelist"):
        t.train_selective_unit(x, {**source, "same_residence": source["income_binary"]}, attrs, xv, validation, 2,
                              tmp_path / "invalid", **default)
