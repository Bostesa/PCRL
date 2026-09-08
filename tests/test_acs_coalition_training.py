"""Artificial checks of coalition objectives, access, exact forks and budgets."""
import copy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from experiments import acs_bottleneck_training as old
from experiments import acs_coalition_training as t


def fixture_data():
    rng = np.random.default_rng(743)
    raw = (rng.normal(size=(37, 32)) * np.linspace(.2, 2., 32) + np.linspace(-2., 1., 32)).astype(np.float32)
    raw[:, 12] = 3.
    val = rng.normal(size=(19, 32)).astype(np.float32)
    sources = {name: (np.arange(len(raw)) % 2).astype(np.int64) for name in t.SOURCE_SCHEMA}
    sources["income_binary"][::7] = -1
    attributes = {"SEX": np.arange(len(raw)) % 2, "RAC1P": np.arange(len(raw)) % 9}
    attributes["RAC1P"][::11] = -1
    validation = {name: (np.arange(len(val)) % 2).astype(np.int64) for name in t.SOURCE_SCHEMA}
    raw64 = raw.astype(np.float64)
    pre = {"mean": raw64.mean(0), "scale": np.where(raw64.std(0) > 1e-12, raw64.std(0), 1.)}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1270000)
        original = old.BottleneckModel(pre["mean"], pre["scale"])
    old.initialize_pca16_mapper(original)
    return raw, sources, attributes, pre, original.state_dict(), val, validation


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    data = fixture_data(); raw, sources, attrs, pre, state, val, validation = data
    out = tmp_path_factory.mktemp("coalition") / "seed_0"
    before = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            result = t.train_seed(raw, sources, attrs, pre, state, {}, 0, out, miniature=True, pca_val=val, source_val=validation)
    finally:
        torch.set_num_threads(before)
    return result, out, data


def test_initial_identity_genuine_heads_wire_and_native_boundary(trained):
    result, _, data = trained
    raw, _, _, _, genuine, val, _ = data
    initial = result["snapshots"]["I"]
    assert sum(p.numel() for p in initial.parameters()) == 6355
    assert not any("decoder" in name for name, _ in initial.named_parameters())
    for purpose, branch in initial.branches.items():
        for name, value in branch.mapper.state_dict().items(): assert torch.equal(value, genuine["mapper." + name])
        for task, head in branch.heads.items():
            for name, value in head.state_dict().items(): assert torch.equal(value, genuine[f"heads.{task}.{name}"])
    for x in (raw, val):
        f, p, native = initial.wires(x, "F"), initial.wires(x, "P"), initial.native_probabilities(x)
        assert [f[k].shape[1] for k in ("A", "B", "AB")] == [16, 16, 32]
        assert [p[k].shape[1] for k in ("A", "B", "AB")] == [2, 1, 3]
        np.testing.assert_allclose(f["A"], x[:, :16], atol=1e-5, rtol=1e-5)
        assert np.array_equal(f["A"], f["B"])
        assert np.array_equal(p["A"], np.stack([native[name][:, 1] for name in t.PURPOSE_TASKS["A"]], 1))
        assert np.array_equal(p["B"], native["public_coverage"][:, 1:2])
        assert np.array_equal(p["AB"], np.concatenate((p["A"], p["B"]), 1))
    with pytest.raises(ValueError): initial.wires(raw, "hidden")


def test_equal_purpose_source_loss_masks_and_prior_entropy():
    a = torch.tensor([[.2], [-.3], [1.]], requires_grad=True)
    logits = {"A": {"income_binary": a, "civilian_at_work": 2*a}, "B": {"public_coverage": -a}}
    labels = {"income_binary": torch.tensor([1, -1, 0]), "civilian_at_work": torch.tensor([-1, -1, -1]), "public_coverage": torch.tensor([0, 1, 0])}
    actual, tasks, purposes, counts = t.source_loss(logits, labels)
    expected = .25 * F.binary_cross_entropy_with_logits(a[[0, 2]].flatten(), torch.tensor([1., 0.])) + .5 * F.binary_cross_entropy_with_logits(-a.flatten(), torch.tensor([0., 1., 0.]))
    assert torch.equal(actual, expected)
    assert float(tasks["civilian_at_work"].detach()) == 0. and counts["civilian_at_work"] == 0
    torch.autograd.grad(actual, a)
    raw, source, attrs, _, _, _, _ = fixture_data()
    ys, ya = old._labels(source, len(raw), t.SOURCE_SCHEMA, "source"), old._labels(attrs, len(raw), t.ATTRIBUTE_SCHEMA, "attribute")
    priors = t.label_priors(ys, ya)
    for name, labels in {**source, **attrs}.items():
        counts = np.bincount(labels[labels >= 0], minlength=9 if name == "RAC1P" else 2)
        p = counts / counts.sum()
        assert priors[name]["entropy"] == float(-(p[p > 0]*np.log(p[p > 0])).sum())


def test_exact_targeted_extra_local_sum_algebra_and_role_seeds():
    raw, source, attrs, pre, state, _, _ = fixture_data()
    model = t.CoalitionModel(pre, state)
    ys, ya = old._labels(source, len(raw), t.SOURCE_SCHEMA, "source"), old._labels(attrs, len(raw), t.ATTRIBUTE_SCHEMA, "attribute")
    priors = t.label_priors(ys, ya)
    rng = torch.get_rng_state().clone()
    observers, meta = t.make_observers("P", 0)
    assert torch.equal(rng, torch.get_rng_state()) and len(observers) == 9
    losses, detail = t.components(model, observers, model.standardize(raw), ys, ya, priors, "P")
    ell = {role: ce / priors[role.split("__")[1]]["entropy"] for role, ce in detail["observer_ce"].items()}
    la = torch.stack([ell["A__"+target] for target in t.ROLE_TARGETS["A"]]).mean()
    lb = torch.stack([ell["B__"+target] for target in t.ROLE_TARGETS["B"]]).mean()
    ma = torch.stack([ell["A__SEX"], ell["A__RAC1P"]]).mean()
    mb = torch.stack([ell["B__SEX"], ell["B__RAC1P"]]).mean()
    assert torch.equal(losses["individual"], (la+lb)/2)
    assert torch.equal(losses["extra_local"], ma+mb)
    assert not torch.equal(losses["extra_local"], (ma+mb)/2)
    assert t.COEFFICIENTS["Iplus"] == {"individual": -.1, "extra_local": -.1, "coalition": 0.}
    assert t.COEFFICIENTS["I"]["individual"] == t.COEFFICIENTS["J"]["individual"] == t.COEFFICIENTS["Iplus"]["individual"]
    for role, record in meta.items():
        view, target = role.split("__")
        assert record["seed"] == 20260911 + 10*t.VIEW_INDEX[view] + t.TARGET_ORDER.index(target)
        assert record["input_dim"] == {"A": 2, "B": 1, "AB": 3}[view]


def test_local_coalition_head_gradient_paths_and_diagnostic_immutability(trained):
    result, _, _ = trained
    for interface in ("F", "P"):
        for regime in ("I", "Iplus", "J"):
            record = result["metadata"]["arms"][interface+"_"+regime]["fork_diagnostic"]
            for owner, other in (("A", "B"), ("B", "A")):
                assert record["routing"][f"local_{owner}_to_{other}_mapper_l2"] == 0.
                assert record["routing"][f"extra_{owner}_to_{other}_mapper_l2"] == 0.
                assert record["groups"][owner+"_mapper"]["coalition_raw_l2"] > 0.
                for loss in ("individual", "extra_local", "coalition"):
                    norm = record["groups"][owner+"_heads"][loss+"_raw_l2"]
                    assert norm == 0. if interface == "F" else norm > 0.
            assert record["state_unchanged"] and record["rng_unchanged"]
    raw, source, attrs, pre, state, _, _ = fixture_data()
    model = t.CoalitionModel(pre, state)
    x = model.standardize(raw)
    y = old._labels(source, len(raw), t.SOURCE_SCHEMA, "source")
    source_value = t.source_loss(model.forward_parts(x)[1], y)[0]
    unused = list(range(16, 32)) + list(range(48, 64))
    for branch in model.branches.values():
        grad, = torch.autograd.grad(source_value, branch.mapper[2].weight, retain_graph=True)
        assert torch.linalg.vector_norm(grad[:, unused]) > 0


def test_detached_observers_and_frozen_observers_on_mapper_update():
    raw, source, attrs, pre, state, _, _ = fixture_data()
    model = t.CoalitionModel(pre, state); observers, _ = t.make_observers("P", 0)
    ys, ya = old._labels(source, len(raw), t.SOURCE_SCHEMA, "source"), old._labels(attrs, len(raw), t.ATTRIBUTE_SCHEMA, "attribute")
    priors = t.label_priors(ys, ya); x = model.standardize(raw)
    model_hash = old.state_digest(model.state_dict())
    t.observer_update(model, observers, old._adam(observers.parameters()), x, {**ys, **ya}, priors, "P")
    assert old.state_digest(model.state_dict()) == model_hash
    assert all(p.grad is None for p in model.parameters())
    observer_hash = old.state_digest(observers.state_dict())
    t.mapper_update(model, observers, old._adam(model.parameters()), x, ys, ya, priors, "P", "J")
    assert old.state_digest(observers.state_dict()) == observer_hash
    assert all(p.grad is None for p in observers.parameters())


def test_full_adam_forks_schedules_exposure_and_snapshot_immutability(trained):
    result, out, _ = trained
    load = lambda path: torch.load(path, map_location="cpu", weights_only=False)
    initial, warm = load(out/"initialization.pt"), load(out/"warm_base.pt")
    metadata = result["metadata"]
    assert old.state_digest(result["snapshots"]["I"].state_dict()) == old.state_digest(initial["model_state"])
    assert old.state_digest(result["snapshots"]["W"].state_dict()) == old.state_digest(warm["model_state"])
    assert warm["counts"] == {"mapper_optimizer_steps": 3, "adversary_optimizer_steps": 0}
    for interface in ("F", "P"):
        common = load(out/interface/"warm_adversary.pt")
        assert old.tree_digest(common["model_state"]) == old.tree_digest(warm["model_state"])
        assert old.tree_digest(common["mapper_optimizer"]) == old.tree_digest(warm["mapper_optimizer"])
        for regime in ("I", "Iplus", "J"):
            name = interface+"_"+regime
            fork, final = load(out/name/"fork.pt"), load(out/name/"final.pt")
            for key in ("model_state", "adversary_state", "mapper_optimizer", "adversary_optimizer", "counts"):
                assert old.tree_digest(fork[key]) == old.tree_digest(common[key])
            assert torch.equal(fork['torch_rng_state'], common['torch_rng_state'])
            assert fork['schedule_state']['schedules'] == common['schedule_state']['schedules'] == metadata['schedules']
            assert fork['schedule_state']['phase'] == 'continuation' and fork['schedule_state']['completed_epochs'] == 0
            assert final['schedule_state']['completed_epochs'] == 2 and final['schedule_state']['next_minibatch_index'] == 0
            assert final["counts"] == {"mapper_optimizer_steps": 9, "adversary_optimizer_steps": 21}
            assert {int(v["step"]) for v in final["mapper_optimizer"]["state"].values()} == {9}
            assert {int(v["step"]) for v in final["adversary_optimizer"]["state"].values()} == {21}
            assert metadata["arms"][name]["schedule_sha256"] == metadata["schedules"]["continuation"]["sha256"]
            assert metadata["arms"][name]["observer_exposure_per_row"] == 7
            assert metadata["arms"][name]["mapper_exposure_per_row"] == 3
    assert metadata["reserved_labels_received"] is False and metadata["final_evaluation_received"] is False
    json.dumps(metadata, allow_nan=False)


def test_reserved_label_and_configuration_boundaries(tmp_path):
    raw, sources, attrs, pre, state, val, validation = fixture_data()
    bad = {**sources, "same_residence": np.zeros(len(raw), np.int64)}
    with pytest.raises(ValueError, match="whitelist"):
        t.train_seed(raw, bad, attrs, pre, state, {}, 0, tmp_path/"reserved", miniature=True)
    with pytest.raises(ValueError, match="differs"):
        t.train_seed(raw, sources, attrs, pre, state, {"training": {"warmup_epochs": 61}}, 0, tmp_path/"schedule", miniature=True)
    with pytest.raises(ValueError, match="coefficients"):
        t.train_seed(raw, sources, attrs, pre, state, {"objectives": {"Iplus": {"individual": -.2}}}, 0, tmp_path/"algebra", miniature=True)


def test_independent_literal_diagnostic_replay(trained):
    from scripts.verify_acs_coalition_training import literal_diagnostic, compare
    result, out, data = trained
    raw, source, attrs, pre, _, _, _ = data
    standardized = ((raw.astype(np.float64)-pre['mean'])/pre['scale']).astype(np.float32)
    idx = np.random.default_rng(1280000).permutation(len(raw))[:16]
    labels = {name: y[idx] for name, y in {**source, **attrs}.items()}
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            for interface in ('F', 'P'):
                for regime in ('I', 'Iplus', 'J'):
                    name = interface+'_'+regime
                    checkpoint = torch.load(out/name/'fork.pt', map_location='cpu', weights_only=True)
                    actual = literal_diagnostic(checkpoint['model_state'], checkpoint['adversary_state'], standardized[idx], labels,
                        result['metadata']['prior_entropies'], interface, regime)
                    assert compare(actual, result['metadata']['arms'][name]['fork_diagnostic']) == 0.
    finally:
        torch.set_num_threads(previous)
