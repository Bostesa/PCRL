"""Task-only and matched nested-partition controls on synthetic roles."""
import json
import numpy as np
import pytest

from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from experiments.pcrl_adaptive_release_v1 import refinement, roles, task_baselines


def _assigned(role, n, salt):
    found = []
    i = 0
    while len(found) < n:
        household = f"task-baseline-{salt}-{i}"
        if roles.role_of(household) == role:
            found.append(household)
        i += 1
    return np.asarray(found)


def _role_rows(role, n, salt):
    ha = np.zeros((n, 4), dtype=np.float32)
    ha[:, 0] = np.linspace(-1, 1, n)
    x = np.zeros((n, 32), dtype=np.float32)
    x[:, 0] = np.linspace(-1, 1, n)
    return {"x": x, "ha": ha, "token_codes": np.zeros(n, dtype=np.int64),
            "teacher_p": np.full(n, .5), "residual": np.zeros(n),
            "households": _assigned(role, n, salt), "weights": np.ones(n),
            "labels": {"same_residence": (ha[:, 0] > 0).astype(np.int64)}}


class FixedNuisance:
    def probabilities(self, inputs: RuntimeInputs):
        z = np.asarray(inputs.h_a)[:, 0]
        sex1 = .25 + .5*(z+1)/2
        sex = np.column_stack((1-sex1, sex1))
        race = np.full((len(z), 9), .01)
        race[:, 0] = .92 - .4*(z+1)/2
        race[:, 1] = .01 + .4*(z+1)/2
        return {"SEX": sex, "RAC1P": race}


def _task_rows(rows):
    n = len(rows["ha"])
    cost = np.full((n, 17), 2/n)
    cost[rows["ha"][:, 0] <= 0, 0] = 0
    cost[rows["ha"][:, 0] > 0, 1] = 0
    return cost


def test_task_only_fits_nuisance_role_and_releases_one_legal_token(tmp_path):
    rows = _role_rows("nuisance_train", 60, "fit")
    model, receipt = task_baselines.fit_task_only(rows, seed=31)
    assert receipt["training_role"] == "nuisance_train"
    assert receipt["fitted_people"] == 60
    assert receipt["quantile_thresholds_from"] == "nuisance_train predictions only"
    inputs = RuntimeInputs(rows["x"], rows["ha"])
    proba = model.predict_probability(inputs)
    assert proba.shape == (60,)
    assert proba[-1] > proba[0]
    release = model.release(inputs)
    assert set(release) == {"h_a", "token"}
    assert release["h_a"].tobytes() == rows["ha"].tobytes()
    assert release["token"].shape == (60,)
    assert np.all((0 <= release["token"]) & (release["token"] < 17))
    assert np.all(np.diff(release["token"]) >= 0)
    directory = tmp_path / "private" / "task_only"
    saved = task_baselines.save_task_only(directory, model, receipt)
    from experiments.pcrl_adaptive_release_v1.archive_unit import inventory
    assert set(inventory(directory)) == {"task_only.joblib", "TASK_ONLY.json", "COMPLETE.json"}
    loaded, verified = task_baselines.load_task_only(directory)
    assert saved == verified
    assert np.array_equal(loaded.release(inputs)["token"], release["token"])
    with (directory / "task_only.joblib").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        task_baselines.load_task_only(directory)
    with pytest.raises(ValueError, match="hash|inventory|artifact"):
        task_baselines.save_task_only(directory, model, receipt)


def test_task_only_resume_rejects_changed_scientific_source_pin(tmp_path):
    rows = _role_rows("nuisance_train", 60, "source-pin")
    model, receipt = task_baselines.fit_task_only(rows, seed=31)
    directory = tmp_path / "private" / "task_only"
    task_baselines.save_task_only(directory, model,
                                  {**receipt, "module_sha256": "a"*64})
    with pytest.raises(ValueError, match="different unit"):
        task_baselines.save_task_only(directory, model,
                                      {**receipt, "module_sha256": "b"*64})


def test_task_only_loader_recovers_verified_module_cli_pickle(tmp_path):
    """The historical -m fit serialized its class as __main__.TaskOnlyPredictor."""
    import __main__
    import joblib

    rows = _role_rows("nuisance_train", 60, "module-cli")
    model, receipt = task_baselines.fit_task_only(rows, seed=31)
    root = tmp_path / "private" / "task_only_cli"
    root.mkdir(parents=True)
    path = root / "task_only.joblib"
    sentinel = object()
    prior = getattr(__main__, "TaskOnlyPredictor", sentinel)
    prior_module = task_baselines.TaskOnlyPredictor.__module__
    try:
        setattr(__main__, "TaskOnlyPredictor", task_baselines.TaskOnlyPredictor)
        task_baselines.TaskOnlyPredictor.__module__ = "__main__"
        joblib.dump(model, path, compress=3)
    finally:
        task_baselines.TaskOnlyPredictor.__module__ = prior_module
        if prior is sentinel:
            delattr(__main__, "TaskOnlyPredictor")
        else:
            setattr(__main__, "TaskOnlyPredictor", prior)
    (root / "TASK_ONLY.json").write_text(json.dumps({**receipt,
        "model_sha256": task_baselines._sha_file(path)}))
    loaded, verified = task_baselines.load_task_only(root)
    assert verified["model_sha256"] == task_baselines._sha_file(path)
    assert isinstance(loaded, task_baselines.TaskOnlyPredictor)
    assert np.array_equal(loaded.token_codes(RuntimeInputs(rows["x"], rows["ha"])),
                          model.token_codes(RuntimeInputs(rows["x"], rows["ha"])))


def test_task_only_rejects_labels_from_any_other_household_role():
    rows = _role_rows("coefficient_split", 60, "wrong")
    with pytest.raises(ValueError, match="nuisance_train"):
        task_baselines.fit_task_only(rows, seed=31)


def test_task_only_private_token_laws_are_actual_17_token_mixtures():
    rows = _role_rows("nuisance_train", 60, "law")
    model, _ = task_baselines.fit_task_only(rows, seed=31)
    inputs = RuntimeInputs(rows["x"], rows["ha"])
    direct = task_baselines.task_only_control_law(model, inputs,
                                                  mode="unmodified", publish=.5)
    token = model.token_codes(inputs)
    assert direct.shape == (60, 17)
    assert np.array_equal(direct, np.eye(17)[token])
    constant = task_baselines.task_only_control_law(
        model, inputs, mode="constant_replacement", publish=.5, constant_token=0)
    assert np.allclose(constant, .5*direct + .5*np.eye(17)[np.zeros(60, dtype=int)])
    response = task_baselines.task_only_control_law(
        model, inputs, mode="randomized_response", publish=.5)
    assert np.allclose(response, .5*direct + .5/17)
    assert np.allclose(response.sum(axis=1), 1)
    with pytest.raises(ValueError, match="publish"):
        task_baselines.task_only_control_law(model, inputs,
                                             mode="randomized_response", publish=1.1)


@pytest.mark.parametrize("mode", ["constant_replacement", "randomized_response"])
def test_sampled_task_only_release_has_private_stable_one_token_wire(tmp_path, mode):
    rows = _role_rows("nuisance_train", 60, f"sample-{mode}")
    model, _ = task_baselines.fit_task_only(rows, seed=31)
    inputs = RuntimeInputs(rows["x"][:5], rows["ha"][:5])
    ids = [f"person-{j}" for j in range(5)]
    cache_path = tmp_path / "private" / f"{mode}_cache.json"
    kwargs = dict(mode=mode, publish=.5, replay_key=b"k"*32,
                  release_id="synthetic-2018-development", cache_path=cache_path)
    first_session = task_baselines.TaskOnlyReleaseSession(model, **kwargs)
    first = first_session.emit(inputs, ids)
    assert set(first) == {"h_a", "token"}
    assert first["h_a"].tobytes() == inputs.h_a.tobytes()
    assert first["h_a"].dtype == inputs.h_a.dtype
    assert first["token"].shape == (5,)
    assert np.all((0 <= first["token"]) & (first["token"] < 17))
    assert np.array_equal(first_session.emit(inputs, ids)["token"], first["token"])
    assert cache_path.is_file()
    assert "person-0" not in cache_path.read_text()
    second_session = task_baselines.TaskOnlyReleaseSession(model, **kwargs)
    assert np.array_equal(second_session.emit(inputs, ids)["token"], first["token"])
    wrong_key = task_baselines.TaskOnlyReleaseSession(
        model, **{**kwargs, "replay_key": b"z"*32})
    with pytest.raises(ValueError, match="different model or release"):
        wrong_key.emit(inputs, ids)
    changed_ha = inputs.h_a.copy(); changed_ha[0, 0] += 1
    with pytest.raises(ValueError, match="changed"):
        second_session.emit(RuntimeInputs(inputs.x_a, changed_ha), ids)
    changed_x = inputs.x_a.copy(); changed_x[0, 0] += 1
    with pytest.raises(ValueError, match="changed"):
        second_session.emit(RuntimeInputs(changed_x, inputs.h_a), ids)
    with pytest.raises(TypeError, match="RuntimeInputs"):
        second_session.emit({"x": inputs.x_a, "ha": inputs.h_a,
                             "labels": rows["labels"]}, ids)
    with pytest.raises(ValueError, match="duplicate"):
        second_session.emit(inputs, [ids[0]]*5)
    saved_cache = json.loads(cache_path.read_text())
    record = next(iter(saved_cache["records"].values()))
    record["token"] = (record["token"]+1) % 17
    cache_path.write_text(json.dumps(saved_cache))
    with pytest.raises(ValueError, match="cache token"):
        second_session.emit(inputs, ids)


def test_sampled_task_only_control_matches_exact_registered_law_and_pins_model():
    rows = _role_rows("nuisance_train", 60, "sample-law")
    model, _ = task_baselines.fit_task_only(rows, seed=31)
    inputs = RuntimeInputs(rows["x"], rows["ha"])
    session = task_baselines.TaskOnlyReleaseSession(
        model, mode="randomized_response", publish=.75,
        replay_key=b"s"*32, release_id="development-replay")
    ids = list(range(60))
    law = task_baselines.task_only_control_law(
        model, inputs, mode="randomized_response", publish=.75)
    observed = session.emit(inputs, ids)["token"]
    assert observed.shape == (60,)
    for i, identifier in enumerate(ids):
        digest = task_baselines._runtime_row_digest(inputs.x_a[i], inputs.h_a[i])
        u = session._uniform(("int", identifier), digest)
        expected = int(np.searchsorted(np.cumsum(law[i]), u, side="right"))
        assert observed[i] == expected
    model.cuts[0] -= .01
    with pytest.raises(ValueError, match="model"):
        session.emit(inputs, ids)


def test_sampled_task_only_requires_private_key_and_registered_control_point():
    rows = _role_rows("nuisance_train", 60, "sample-invalid")
    model, _ = task_baselines.fit_task_only(rows, seed=31)
    with pytest.raises(ValueError, match="key"):
        task_baselines.TaskOnlyReleaseSession(
            model, mode="randomized_response", publish=.5,
            replay_key=b"short", release_id="r1")
    with pytest.raises(ValueError, match="publish"):
        task_baselines.TaskOnlyReleaseSession(
            model, mode="randomized_response", publish=.25,
            replay_key=b"k"*32, release_id="r1")
    with pytest.raises(ValueError, match="private"):
        task_baselines.TaskOnlyReleaseSession(
            model, mode="constant_replacement", publish=.5,
            replay_key=b"k"*32, release_id="r1", cache_path="/tmp/public.json")


@pytest.mark.parametrize("policy", ["task_only", "joint_risk", "random_eligible"])
def test_matched_partition_controls_keep_t32_parent_support_and_leaf_budget(policy):
    coefficient = _role_rows("coefficient_split", 220, "coefficient")
    selection = _role_rows("inner_selection", 220, "selection")
    result = task_baselines.select_partition_control(
        policy, coefficient, selection, FixedNuisance(),
        task_fit_rows=_task_rows(coefficient),
        task_selection_rows=_task_rows(selection),
        target_states=33, feature_names=("h_a_0",), seed=41)
    partition = result["partition"]
    assert len(partition.parents) == 33
    assert len(result["history"]) == 1
    assert result["history"][0]["children_unique_households"] == [110, 110]
    assert min(result["history"][0]["children_effective_households"]) >= 100
    t = coefficient["token_codes"]
    features = {"h_a_0": coefficient["ha"][:, 0]}
    child = partition.route(t, features)
    assert np.array_equal(partition.parent_of_leaf[child], t)
    assert partition.rules[0].feature_name == "h_a_0"


def test_random_partition_replays_seed_and_does_not_use_task_cost_or_labels():
    coefficient = _role_rows("coefficient_split", 220, "coefficient2")
    selection = _role_rows("inner_selection", 220, "selection2")
    coefficient.pop("labels")
    selection.pop("labels")
    kwargs = dict(policy="random_eligible", coefficient_rows=coefficient,
                  selection_rows=selection, frozen_nuisance=FixedNuisance(),
                  target_states=33, feature_names=("h_a_0",), seed=19)
    first = task_baselines.select_partition_control(**kwargs)
    second = task_baselines.select_partition_control(**kwargs)
    assert first["partition"].to_record() == second["partition"].to_record()
    assert len(first["partition"].parents) == 33


def test_fixed_support_floor_stops_all_partition_controls_without_weakening():
    coefficient = _role_rows("coefficient_split", 80, "small-fit")
    selection = _role_rows("inner_selection", 80, "small-selection")
    for policy in ("task_only", "joint_risk", "random_eligible"):
        result = task_baselines.select_partition_control(
            policy, coefficient, selection, FixedNuisance(),
            task_fit_rows=_task_rows(coefficient),
            task_selection_rows=_task_rows(selection),
            target_states=64, feature_names=("h_a_0",), seed=17)
        assert len(result["partition"].parents) == 32
        assert result["status"] == "support_limited"


def test_task_only_fills_matched_leaf_budget_with_zero_task_gain_when_supported():
    coefficient = _role_rows("coefficient_split", 220, "zero-task-fit")
    selection = _role_rows("inner_selection", 220, "zero-task-selection")
    zero_fit = np.zeros((220, 17))
    zero_selection = np.zeros((220, 17))
    result = task_baselines.select_partition_control(
        "task_only", coefficient, selection, FixedNuisance(),
        task_fit_rows=zero_fit, task_selection_rows=zero_selection,
        target_states=33, feature_names=("h_a_0",), seed=17)
    assert len(result["partition"].parents) == 33
    assert result["history"][0]["fitting_gain"] == pytest.approx(0)


def test_task_contributions_are_half_U_half_weighted_original_person_loss():
    class FixedDecoder:
        def predict_token_proba(self, h, n_tokens):
            p = np.tile([.7, .3], (len(h), n_tokens, 1))
            p[:, 1, :] = [.2, .8]
            return p

    rows = _role_rows("coefficient_split", 3, "cost")
    rows["weights"] = np.array([1., 2., 3.])
    task = task_baselines.task_contributions(FixedDecoder(), rows)
    q = np.zeros((32, 17)); q[:, 0] = 1; q[0, :] = 0; q[0, 1] = 1
    y = rows["labels"]["same_residence"]
    person = -np.log(np.where(y == 1, .8, .2))
    expected = .5*person.mean() + .5*np.dot(rows["weights"]/6, person)
    assert np.sum(task*q[rows["token_codes"]]) == pytest.approx(expected)


def test_partition_control_bundle_saves_and_resumes_matched_leaf_artifacts(tmp_path):
    class FixedDecoder:
        def predict_token_proba(self, h, n_tokens):
            p = np.tile([.5, .5], (len(h), n_tokens, 1))
            p[:, 0, :] = [.9, .1]
            p[:, 1, :] = [.1, .9]
            return p

    coefficient = _role_rows("coefficient_split", 220, "bundle-fit")
    selection = _role_rows("inner_selection", 220, "bundle-select")
    target = refinement.NestedPartition.base(32).split(0, "h_a_0", 0)
    root = tmp_path / "private" / "partition_controls"
    pins = {"frozen_decoder_sha256": "a"*64,
            "frozen_nuisance_sha256": "b"*64,
            "b_complete_sha256": "c"*64}
    args = (coefficient, selection, FixedNuisance(), FixedDecoder(), target, root)
    first = task_baselines.fit_partition_controls_from_roles(
        *args, seed=31, source_pins=pins, feature_names=("h_a_0",))
    assert first["status"] == "COMPLETE"
    assert set(first["controls"]) == set(task_baselines.PARTITION_POLICIES)
    assert all(item["realized_states"] == 33 for item in first["controls"].values())
    for policy in task_baselines.PARTITION_POLICIES:
        assert (root / "policies" / f"{policy}.json").is_file()
    before = (root / "COMPLETE.json").read_bytes()
    again = task_baselines.fit_partition_controls_from_roles(
        *args, seed=31, source_pins=pins, feature_names=("h_a_0",))
    assert again == first
    assert (root / "COMPLETE.json").read_bytes() == before
    with pytest.raises(ValueError, match="different scientific unit"):
        task_baselines.fit_partition_controls_from_roles(
            *args, seed=31,
            source_pins={**pins, "b_complete_sha256": "d"*64},
            feature_names=("h_a_0",))
    changed_target = refinement.NestedPartition.base(32).split(0, "h_a_0", .1)
    with pytest.raises(ValueError, match="different scientific unit"):
        task_baselines.fit_partition_controls_from_roles(
            coefficient, selection, FixedNuisance(), FixedDecoder(),
            changed_target, root, seed=31, source_pins=pins,
            feature_names=("h_a_0",))


def test_partition_runner_refuses_unsanitized_archive_before_deserialization(tmp_path, monkeypatch):
    from experiments.pcrl_task_aligned_cuts_v1 import data

    b_root = tmp_path / "private" / "B"
    b_root.mkdir(parents=True)
    (b_root / "COMPLETE.json").write_text(json.dumps({
        "anchor": 0, "delta": .001, "status": "COMPLETE", "artifact_sha256": {}}))
    fake_file = tmp_path / "repo" / "experiments" / "pcrl_adaptive_release_v1" / "task_baselines.py"
    monkeypatch.setattr(task_baselines, "__file__", str(fake_file))
    monkeypatch.setattr(data, "index", lambda path: {})

    def forbidden(*args, **kwargs):
        raise AssertionError("raw prepared archive was deserialized")

    monkeypatch.setattr(data, "load_prepared", forbidden)
    with pytest.raises(RuntimeError, match="sanitized"):
        task_baselines.run_partition_controls(
            0, .001, "unused-index", "unused-A", b_root,
            tmp_path / "private" / "controls", seed=1)
