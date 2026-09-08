"""Artificial input-boundary, matched identity/fork and source-bank checks."""
import copy
import inspect

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from experiments import acs_bottleneck_training as old
from experiments import acs_restricted_training as t


def data(n, seed):
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(n, 32)) * np.linspace(.2, 3., 32) + np.linspace(-3., 5., 32)).astype(np.float32)
    x[:, 12] = 7.
    source = {k: (np.arange(n) % 2).astype(np.int64) for k in t.SOURCE_SCHEMA}
    source["income_binary"][::9] = -1
    attr = {"SEX": np.arange(n) % 2, "RAC1P": np.arange(n) % 9}
    attr["RAC1P"][::11] = -1
    return x, source, attr


def setup():
    x, source, attr = data(37, 43)
    xv, val, _ = data(23, 44)
    raw64 = x.astype(np.float64)
    stats = {"mean": raw64.mean(0).tolist(), "scale": np.where(raw64.std(0) > 1e-12, raw64.std(0), 1.).tolist()}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1270200)
        original = old.BottleneckModel(stats["mean"], stats["scale"])
    old.initialize_pca16_mapper(original)
    teachers = {"E": (x[:, :16] * .7).astype(np.float32), "S": (x[:, :16] + .1).astype(np.float32)}
    tv = {"E": (xv[:, :16] * .7).astype(np.float32), "S": (xv[:, :16] + .1).astype(np.float32)}
    return x, source, attr, xv, val, stats, original.state_dict(), teachers, tv


@pytest.fixture(scope="module")
def units(tmp_path_factory):
    args = setup()
    x, source, attr, xv, val, stats, state, teachers, tv = args
    directory = tmp_path_factory.mktemp("restricted_training")
    result = {}
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            for name in ("E", "S"):
                for access in ("F", "K"):
                    result[name, access] = t.train_restricted_unit(teachers[name], source, attr, tv[name], val, 2,
                        directory / (name + access), teacher_name=name, access=access, fitting_statistics=stats,
                        initial_state=state, orthogonal_q=t.generate_q(2), miniature=True,
                        **({"raw_fit": x, "raw_val": xv} if access == "F" else {}))
                result[name, "B"] = t.train_source_bank(teachers[name], source, tv[name], val, 2,
                    directory / (name + "B"), teacher_name=name, fitting_statistics=stats,
                    initial_state=state, orthogonal_q=t.generate_q(2), miniature=True)
    finally:
        torch.set_num_threads(previous)
    return result, directory, args


def test_sign_canonical_Q_is_dedicated_and_original_rng_untouched():
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    q = t.generate_q(2)
    rng = np.random.default_rng(20260911)
    raw = rng.standard_normal((16, 16))
    np.testing.assert_allclose(q.T @ q, np.eye(16), atol=1e-12, rtol=1e-12)
    assert (np.diag(q.T @ raw) >= 0).all()
    assert np.array_equal(q, t.generate_q(2)) and not np.array_equal(q, t.generate_q(1))
    assert torch.equal(torch_before, torch.get_rng_state())
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(numpy_before, after))
    with pytest.raises(ValueError): t.generate_q(3)


def test_exact_all_condition_initial_tensors_genuine_heads_and_identity(units):
    results, _, args = units
    x, _, _, xv, _, stats, genuine, teachers, tv = args
    states = []
    for (name, access), result in results.items():
        model = result["snapshots"]["I"]
        states.append(old.state_digest(model.state_dict()))
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in genuine.items() if key.startswith(("heads.", "decoder.")))
        assert model.mapper[0].weight.shape == (64, 48) and model.mapper[2].weight.shape == (16, 64)
        assert torch.count_nonzero(model.mapper[0].weight[:, 16:]) == 0
        for target, raw in ((teachers[name], x), (tv[name], xv)):
            np.testing.assert_allclose(model.release(target, **({"raw": raw} if access == "F" else {})), target, atol=1e-5, rtol=1e-5)
        fixture = result["metadata"]["initialization"]["nondead_full_input_fixture"]
        assert fixture["all32_full_input_columns_nonzero_source_gradient"] and fixture["optimizer_steps"] == 0
        assert min(fixture["raw_input_weight_column_gradient_l2"]) > 0
        assert fixture["auxiliary_readout_source_gradient_l2"] > 0
        assert model.input_scale[12] == 1.
    assert len(set(states)) == 1


def test_restricted_boundary_and_native_probabilities(units, tmp_path):
    results, _, args = units
    x, source, attr, xv, val, stats, state, teachers, tv = args
    model = results["E", "K"]["arms"]["C"]["model"]
    for raw in (x, np.zeros_like(x), np.full_like(x, np.nan)):
        with pytest.raises(ValueError): model.release(teachers["E"], raw=raw)
        with pytest.raises(ValueError): model.source_probabilities(teachers["E"], raw=raw)
        with pytest.raises(ValueError):
            t.train_restricted_unit(teachers["E"], source, attr, tv["E"], val, 2, tmp_path / "forbidden",
                teacher_name="E", access="K", fitting_statistics=stats, initial_state=state,
                orthogonal_q=t.generate_q(2), raw_fit=raw, miniature=True)
    full = results["E", "F"]["arms"]["C"]["model"]
    with pytest.raises(ValueError): full.release(teachers["E"])
    with pytest.raises(ValueError): full.release(teachers["E"], raw=x[:1])
    assert not any(name in inspect.signature(t.train_source_bank).parameters for name in ("raw_fit", "raw_val", "attr_y", "access"))
    bank = results["E", "B"]["arms"]["B"]["model"]
    probs = bank.source_probabilities(teachers["E"])
    assert list(probs) == list(t.SOURCE_SCHEMA)
    for name, p in probs.items():
        assert p.shape == (len(x), 2) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-7)
        with torch.no_grad():
            expected = torch.sigmoid(bank.heads[name](torch.from_numpy(bank.release(teachers["E"])))).numpy().reshape(-1)
        np.testing.assert_array_equal(p[:, 1], expected)


def test_exact_full_forks_Adam_zero_raw_columns_decoder_and_bank_schedule(units):
    results, directory, _ = units
    schedules = []
    for (name, access), result in results.items():
        meta = result["metadata"]
        schedules.append(meta["schedules"])
        root = directory / (name + access)
        initial = torch.load(root / "initialization.pt", weights_only=True)
        warm = torch.load(root / "warm_base.pt", weights_only=True)
        assert not initial["mapper_optimizer_state"]["state"]
        bank = access == "B"
        assert set(result["arms"]) == ({"B"} if bank else {"C", "D"})
        if bank:
            assert not (root / "warm_adversary.pt").exists()
            assert meta["attribute_label_hashes"] == {} and meta["prior_entropies"] == {}
            assert meta["common_optimizer_counts"]["adversary_optimizer_steps"] == 0
        else:
            aw = torch.load(root / "warm_adversary.pt", weights_only=True)
            assert old.tree_digest((warm["model_state"], warm["mapper_optimizer_state"])) == old.tree_digest((aw["model_state"], aw["mapper_optimizer_state"]))
            assert old.tree_digest(torch.load(root / "C/fork.pt", weights_only=True)) == old.tree_digest(torch.load(root / "D/fork.pt", weights_only=True))
        for arm, record in meta["arms"].items():
            final = torch.load(root / arm / "final.pt", weights_only=True)
            assert final["counters"] == {"mapper_optimizer_steps": 9, "adversary_optimizer_steps": 0 if bank else 21, "continuation_epoch": 2}
            assert all(int(v["step"]) == 9 for v in final["mapper_optimizer_state"]["state"].values())
            assert all(i not in final["mapper_optimizer_state"]["state"] for i in (10, 11, 12, 13))
            assert all(torch.equal(value, final["model_state"][key]) for key, value in initial["model_state"].items() if key.startswith("decoder."))
            assert record["decoder_unchanged"] and record["decoder_gradients_absent"]
            if access != "F":
                assert torch.count_nonzero(final["model_state"]["mapper.0.weight"][:, 16:]) == 0
                for key in ("exp_avg", "exp_avg_sq"):
                    assert torch.count_nonzero(final["mapper_optimizer_state"]["state"][0][key][:, 16:]) == 0
            else:
                assert torch.count_nonzero(final["model_state"]["mapper.0.weight"][:, 16:]) > 0
            assert record["continuation_preservation_coefficient"] == (0. if bank else 1.)
        assert old.state_digest(result["snapshots"]["I"].state_dict()) == meta["snapshots"]["I"]["model_hash"]
        assert old.state_digest(result["snapshots"]["W"].state_dict()) == meta["snapshots"]["W"]["model_hash"]
    assert all(s == schedules[0] for s in schedules)


def test_component_gradients_block_boundaries_and_isolation(units):
    results, _, _ = units
    for (_, access), result in results.items():
        meta = result["metadata"]
        diagnostics = list(meta["stage_gradient_diagnostics"].values())
        for arm in meta["arms"].values():
            diagnostics += [arm["fixed_batch_gradient_diagnostics_at_shared_fork"], arm["fixed_batch_gradient_diagnostics_at_final"]]
        for d in diagnostics:
            assert d["training_state_unchanged"] and d["torch_rng_unchanged"] and d["optimizer_steps"] == 0
            assert d["preservation_nonmapper_gradients_all_absent"]
            if access != "F":
                assert d["input_raw_l2"] == 0.
                assert all(value == 0. for key, value in d.items() if key.endswith("raw_input_weight_l2") and value is not None)
            for name, coefficient in d["coefficients"].items():
                raw = d[f"{name}_raw_mapper_l2"]
                if raw is not None:
                    assert d[f"{name}_applied_mapper_l2"] == pytest.approx(abs(coefficient) * raw, rel=2e-6, abs=1e-12)
            if access == "B":
                assert d["teacher_applied_mapper_l2"] == 0. and d["protection_raw_mapper_l2"] is None
                assert not d["hypothetical_preobserver_warmup"]
        assert meta["stage_gradient_diagnostics"]["initialization"]["preservation_loss"] < 1e-10


def test_validation_labels_do_not_choose_mapper_and_reject_bad_inputs(units, tmp_path):
    results, _, args = units
    x, source, attr, xv, val, stats, state, teachers, tv = args
    changed = {k: 1 - v for k, v in val.items()}
    changed["income_binary"][val["income_binary"] < 0] = -1
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            new = t.train_restricted_unit(teachers["E"], source, attr, tv["E"], changed, 2, tmp_path / "changedval",
                teacher_name="E", access="K", fitting_statistics=stats, initial_state=state,
                orthogonal_q=t.generate_q(2), miniature=True)
    finally:
        torch.set_num_threads(previous)
    for arm in ("C", "D"):
        assert new["metadata"]["arms"][arm]["final_model_hash"] == results["E", "K"]["metadata"]["arms"][arm]["final_model_hash"]
        assert new["metadata"]["arms"][arm]["final_adversary_hash"] == results["E", "K"]["metadata"]["arms"][arm]["final_adversary_hash"]
    kwargs = dict(teacher_name="E", access="K", fitting_statistics=stats, initial_state=state, orthogonal_q=t.generate_q(2), miniature=True)
    for patch in ({"teacher_name": "R"}, {"fit_pool": "test"}, {"orthogonal_q": t.generate_q(1)}):
        with pytest.raises(ValueError):
            t.train_restricted_unit(teachers["E"], source, attr, tv["E"], val, 2, tmp_path / "bad", **{**kwargs, **patch})
    with pytest.raises(ValueError):
        t.train_restricted_unit(teachers["E"], {**source, "same_residence": source["income_binary"]}, attr,
            tv["E"], val, 2, tmp_path / "reserved", **kwargs)
