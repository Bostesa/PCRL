"""Branch B orchestration primitives without accessing ACS outcomes."""
import json
import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import fit_a, fit_b, refinement, roles


class FrozenRisk:
    def probabilities(self, inputs):
        n = len(inputs.h_a)
        return {"SEX": np.tile([0.4, 0.6], (n, 1)),
                "RAC1P": np.tile(np.ones(9)/9, (n, 1))}


def _rows(n=8):
    x = np.zeros((n, 32))
    ha = np.zeros((n, 4))
    ha[:, 0] = np.arange(n) % 2
    return {"x": x, "ha": ha,
            "token_codes": np.zeros(n, dtype=int),
            "teacher_p": np.full(n, 0.5),
            "residual": np.zeros(n),
            "households": np.arange(n), "weights": np.ones(n)}


def test_stored_features_use_archived_t0_and_frozen_nuisance_without_reencoding():
    rows = _rows()
    t, features = fit_b.stored_deployable_features(rows, FrozenRisk())
    assert np.array_equal(t, rows["token_codes"])
    assert set(features) == refinement.ALLOWED_FEATURES
    assert np.array_equal(features["h_a_0"], rows["ha"][:, 0])
    bad = {**rows, "token_codes": np.full(8, 32)}
    with pytest.raises(ValueError, match="T32"):
        fit_b.stored_deployable_features(bad, FrozenRisk())


def test_synthetic_supported_split_preserves_17_token_reference_and_person_law():
    rows = _rows()
    check = _rows()
    check["households"] = np.arange(100, 108)
    task = np.tile([0., 1.] + [2.]*15, (8, 1))
    task[rows["ha"][:, 0] == 1, :2] = [1., 0.]
    selected = fit_b.select_nested_partition(
        rows, check, FrozenRisk(), task, task,
        max_states=33, min_households=4,
        min_effective_households=4, feature_names=("h_a_0",))
    partition = selected["partition"]
    assert len(partition.parents) == 33
    assert len(selected["split_history"]) == 1
    q32 = np.eye(17)[np.arange(32) % 17]
    q_child = refinement.copy_parent_kernel(q32, partition.parent_of_leaf)
    assert np.array_equal(q_child[selected["coefficient_leaves"]], q32[rows["token_codes"]])
    assert selected["split_history"][0]["checking_gain"] >= 0


def test_unsupported_parent_is_retained_instead_of_weakening_floor():
    rows = _rows()
    check = _rows()
    check["households"] = np.arange(100, 108)
    task = np.zeros((8, 17))
    selected = fit_b.select_nested_partition(
        rows, check, FrozenRisk(), task, task,
        max_states=64, min_households=100,
        min_effective_households=100, feature_names=("h_a_0",))
    assert len(selected["partition"].parents) == 32
    assert selected["split_history"] == []
    assert selected["status"] == "support_limited"


def test_split_with_fit_gain_but_no_checking_gain_is_rejected_before_channel_fit():
    rows = _rows()
    check = _rows()
    check["households"] = np.arange(100, 108)
    fitting = np.tile([0., 1.] + [2.]*15, (8, 1))
    fitting[rows["ha"][:, 0] == 1, :2] = [1., 0.]
    checking = np.zeros((8, 17))
    selected = fit_b.select_nested_partition(
        rows, check, FrozenRisk(), fitting, checking,
        max_states=33, min_households=4,
        min_effective_households=4, feature_names=("h_a_0",))
    assert len(selected["partition"].parents) == 32
    assert selected["status"] == "checking_unstable"
    assert selected["rejected_candidates"][0]["reason"] == "checking_gain_below_fixed_ratio"


def test_dual_price_sign_matches_a_known_one_row_privacy_floor():
    cost = np.full((32, 17), 10., dtype=float)
    cost[:, 0] = 0
    cost[:, 1] = 1
    attack = np.zeros_like(cost)
    attack[0, 1] = 1
    certificate = fit_b.fixed_bank_dual_prices(
        cost, [{"id": "attack", "coeff": attack, "floor": 0.5}])
    assert certificate["status"] == "optimal"
    assert certificate["multipliers"]["attack"] == pytest.approx(1)
    assert certificate["objective"] == pytest.approx(0.5)
    assert certificate["dual_lower_bound"] == pytest.approx(0.5)
    assert certificate["Q"][0, 1] == pytest.approx(0.5)
    priced = refinement.priced_contributions(cost, [attack], [1.])
    assert priced[0, 0] == pytest.approx(priced[0, 1])


def test_original_person_priced_rows_keep_distinct_role_and_weight_normalization():
    rows = {"weights": np.array([1., 2., 1.])}
    task_mask = np.array([True, True, True])
    task_losses = np.tile(np.arange(17, dtype=float), (3, 1))
    attack_mask = np.array([True, False, True])
    attack_losses = np.ones((2, 17))
    g = fit_b.priced_original_person_rows(
        rows, task_mask, task_losses,
        [{"id": "s_U", "weighting": "U", "valid_mask": attack_mask,
          "losses": attack_losses},
         {"id": "s_W", "weighting": "W", "valid_mask": attack_mask,
          "losses": attack_losses}],
        {"s_U": 0.25, "s_W": 0.5})
    expected_task_mass = .5/3 + .5*np.array([1, 2, 1])/4
    expected_attack = np.array([.25/2 + .5/2, 0, .25/2 + .5/2])
    assert np.allclose(g, expected_task_mass[:, None]*task_losses - expected_attack[:, None])
    assert g.shape == (3, 17)  # no token expansion of the person count


def test_child_coefficients_replay_exact_person_token_expectations_for_both_weights():
    rows = {"weights": np.array([1., 3., 2.])}
    children = np.array([0, 32, 0])
    task_mask = np.array([True, True, True])
    task_losses = np.tile(np.arange(17, dtype=float)[None, :]+1, (3, 1))
    attack_mask = np.array([True, True, False])
    attack_losses = np.tile(np.arange(17, dtype=float)[None, :]+2, (2, 1))
    problem = fit_b.aggregate_child_problem(
        rows, children, 33, task_mask, task_losses,
        [{"id": "fixed_A_SEX", "role": "A/SEX", "target": "SEX",
          "valid_mask": attack_mask, "losses": attack_losses,
          "class_order": [0, 1], "coefficient_pool_sha256": "fixture-pool"}])
    q = np.eye(17)[np.arange(33) % 17]
    selected_task = np.sum(q[children]*task_losses, axis=1)
    for weighting, expected in (("U", selected_task.mean()),
                                ("W", np.dot(rows["weights"]/6, selected_task))):
        assert np.sum(problem["task_cost_pair"][weighting]*q) == pytest.approx(expected)
        cut = next(c for c in problem["bank"] if c["weighting"] == weighting)
        expected_attack = np.sum(q[children[attack_mask]]*attack_losses, axis=1)
        score = (expected_attack.mean() if weighting == "U" else
                 np.dot(np.array([1., 3.])/4, expected_attack))
        assert np.sum(cut["coeff"]*q) == pytest.approx(score)
    assert len(problem["bank"]) == 2


def test_refined_release_emits_only_unchanged_h_and_cached_one_token():
    from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs

    class Encoder:
        def encode(self, inputs):
            return {"codes": {"T0": np.array([0, 0])},
                    "r": np.zeros(2), "p": np.full(2, .5)}

    h = np.array([[.1, .2, .3, .4], [.9, .8, .7, .6]], dtype=np.float32)
    inputs = RuntimeInputs(np.zeros((2, 32), dtype=np.float32), h)
    partition = refinement.NestedPartition.base(32).split(0, "h_a_0", .5)
    q = np.eye(17)[np.arange(33) % 17]
    session = fit_b.RefinedReleaseSession(q, partition, Encoder(), FrozenRisk(),
                                          synthetic_fixture=True,
                                          rng=np.random.default_rng(11))
    released = session.release(inputs, person_ids=["p1", "p2"])
    assert set(released) == {"h_a", "token"}
    assert released["token"].tolist() == [0, 15]  # child 32 maps to token 15
    assert released["h_a"].dtype == h.dtype
    assert released["h_a"].tobytes() == h.tobytes()
    assert session.release(inputs, person_ids=["p1", "p2"])["token"].tolist() == [0, 15]
    moved = RuntimeInputs(inputs.x_a.copy(), h.copy())
    moved.h_a[0, 0] = .3
    with pytest.raises(ValueError, match="immutable"):
        session.release(moved, person_ids=["p1", "p2"])


def test_refined_release_requires_verified_linux_runtime_outside_synthetic_fixture():
    partition = refinement.NestedPartition.base(32)
    q = np.eye(17)[np.arange(32) % 17]
    with pytest.raises(ValueError, match="Linux x86 parity"):
        fit_b.RefinedReleaseSession(q, partition, object(), FrozenRisk())


def test_private_keyed_replay_survives_a_fresh_release_process_without_wire_secrets():
    from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs

    class Encoder:
        def encode(self, inputs):
            return {"codes": {"T0": np.zeros(len(inputs.h_a), dtype=int)},
                    "r": np.zeros(len(inputs.h_a)),
                    "p": np.full(len(inputs.h_a), .5)}

    inputs = RuntimeInputs(np.zeros((1, 32)), np.zeros((1, 4)))
    q = np.eye(17)[np.arange(32) % 17].astype(float)
    q[0] = 1/17
    first = fit_b.RefinedReleaseSession(
        q, refinement.NestedPartition.base(32), Encoder(), FrozenRisk(),
        synthetic_fixture=True, replay_key=b"x"*32)
    second = fit_b.RefinedReleaseSession(
        q, refinement.NestedPartition.base(32), Encoder(), FrozenRisk(),
        synthetic_fixture=True, replay_key=b"x"*32)
    a = first.release(inputs, person_ids=["same"])
    b = second.release(inputs, person_ids=["same"])
    assert a["token"].tolist() == b["token"].tolist()
    assert set(a) == {"h_a", "token"}


def test_runtime_parity_receipt_detects_stored_code_or_child_drift():
    from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs

    class Encoder:
        def encode(self, inputs):
            return {"codes": {"T0": np.array([0, 0])},
                    "r": np.zeros(2), "p": np.full(2, .5)}

    partition = refinement.NestedPartition.base(32).split(0, "h_a_0", .5)
    inputs = RuntimeInputs(np.zeros((2, 32)),
                           np.array([[0., 0., 0., 0.], [1., 0., 0., 0.]]))
    rows = {"x": inputs.x_a, "ha": inputs.h_a,
            "token_codes": np.array([0, 0]), "teacher_p": np.full(2, .5),
            "residual": np.zeros(2)}
    receipt = fit_b.runtime_parity_receipt(rows, partition, Encoder(), FrozenRisk(),
                                            encoder_sha256="a"*64,
                                            nuisance_sha256="b"*64)
    assert receipt["t32_bitwise_equal"] is True
    assert receipt["child_bitwise_equal"] is True
    assert receipt["service_byte_equal"] is True
    drift = {**rows, "token_codes": np.array([0, 1])}
    bad = fit_b.runtime_parity_receipt(drift, partition, Encoder(), FrozenRisk(),
                                        encoder_sha256="a"*64,
                                        nuisance_sha256="b"*64)
    assert bad["linux_x86_parity_verified"] is False
    assert bad["t32_bitwise_equal"] is False


def test_b_final_selection_rejects_earlier_q_that_violates_later_union_cut():
    q0 = np.eye(17)[np.zeros(32, dtype=int)]
    q1 = np.eye(17)[np.ones(32, dtype=int)]
    cost = np.zeros((32, 17))
    coeff = np.zeros_like(cost)
    coeff[:, 1] = 1/32
    rounds = [
        {"round": 0, "inner_selection_fixed_decoder_task": {"U": .1, "W": .1}},
        {"round": 1, "inner_selection_fixed_decoder_task": {"U": .5, "W": .5}},
    ]
    selected = fit_b.select_b_final_bank(
        [q0, q1], rounds, cost, [{"id": "later", "coeff": coeff, "floor": .5}])
    assert selected["selected_round"] == 1
    assert selected["final_bank_checks"][0]["feasible"] is False
    assert np.array_equal(selected["selected_channel"], q1)


class ForbiddenLabels(dict):
    def __getitem__(self, key):
        raise AssertionError("Branch B fitting accessed independent inner-check labels")


def _assigned_households(role, n):
    found = []
    j = 0
    while len(found) < n:
        value = f"synthetic-B-{role}-{j}"
        if roles.role_of(value) == role:
            found.append(value)
        j += 1
    return np.asarray(found)


def _synthetic_role(role, n, seed):
    rng = np.random.default_rng(seed)
    ha = rng.normal(size=(n, 4))
    x = rng.normal(size=(n, 32))
    return {"x": x, "ha": ha, "hb": rng.normal(size=(n, 2)),
            "token_codes": rng.integers(0, 2, size=n, dtype=np.int64),
            "teacher_p": np.full(n, .5), "residual": np.zeros(n),
            "labels": {"SEX": np.arange(n) % 2,
                       "RAC1P": np.arange(n) % 9,
                       "same_residence": (ha[:, 0] > 0).astype(int)},
            "weights": np.ones(n), "ids": np.arange(seed*1000, seed*1000+n),
            "households": _assigned_households(role, n)}


def test_b_full_synthetic_support_alias_receipt_resume_without_inner_check_labels(tmp_path):
    role_dict = {name: _synthetic_role(name, count, j+11)
                 for j, (name, count) in enumerate((("nuisance_train", 60),
                    ("audit_fit", 40), ("coefficient_split", 30),
                    ("inner_selection", 20), ("inner_check", 20)))}
    d17 = np.zeros((32, 17)); d17[:, 0] = 1
    old_q = np.zeros((32, 17)); old_q[:, 1] = 1
    a_dir = tmp_path / "private" / "A"
    fit_a.run_center_from_roles(0, .001, role_dict, d17, old_q, "a"*64,
                               a_dir, 0, target_roles=("A/SEX",),
                               initial_sources=("H", "D17"))
    selected_a = fit_b.load_a_selected(a_dir, anchor=0, delta=.001)
    role_dict["inner_check"] = {**role_dict["inner_check"],
                                 "labels": ForbiddenLabels()}
    b_dir = tmp_path / "private" / "B"
    args = (0, .001, role_dict, d17, old_q, "a"*64, selected_a, b_dir)
    first = fit_b.run_center_from_roles(*args, max_states=64, max_rounds=0)
    assert first["status"] == "SUPPORT_LIMITED_ALIAS_A"
    assert np.array_equal(first["selected_channel"], selected_a["Q"])
    complete = (b_dir / "COMPLETE.json").read_bytes()
    receipt = json.loads(complete)
    assert receipt["amendment_01_sha256"] == fit_b.AMENDMENT_01_SHA256
    assert len(receipt["source_commit"]) == 40
    assert receipt["linux_runtime_parity"] == "REQUIRED_BEFORE_PRODUCTION"
    second = fit_b.run_center_from_roles(*args, max_states=64, max_rounds=0)
    assert np.array_equal(second["selected_channel"], first["selected_channel"])
    assert (b_dir / "COMPLETE.json").read_bytes() == complete


def test_b_supported_child_rounds_restore_the_exact_selected_artifact(tmp_path, monkeypatch):
    role_dict = {name: _synthetic_role(name, count, j+31)
                 for j, (name, count) in enumerate((("nuisance_train", 60),
                    ("audit_fit", 40), ("coefficient_split", 220),
                    ("inner_selection", 220), ("inner_check", 20)))}
    for name in ("coefficient_split", "inner_selection"):
        rows = role_dict[name]
        rows["token_codes"][:] = 0
        rows["ha"][:, 0] = np.linspace(-1, 1, len(rows["ha"]))
        rows["labels"]["same_residence"] = (rows["ha"][:, 0] > 0).astype(int)
    d17 = np.zeros((32, 17)); d17[:, 0] = 1
    old_q = np.zeros((32, 17)); old_q[:, 1] = 1
    a_dir = tmp_path / "private" / "A"
    fit_a.run_center_from_roles(0, .001, role_dict, d17, old_q, "b"*64,
                               a_dir, 0, target_roles=("A/SEX",),
                               initial_sources=("H", "D17"))
    selected_a = fit_b.load_a_selected(a_dir, anchor=0, delta=.001)
    role_dict["inner_check"] = {**role_dict["inner_check"],
                                 "labels": ForbiddenLabels()}

    # Only the split-ranking result is fixed in this orchestration fixture.
    # Both real development resources meet the frozen 100-household floor.
    def supported_fixture_split(fit_rows, selection_rows, nuisance_model,
                                g_fit, g_selection, **kwargs):
        partition = refinement.NestedPartition.base(32).split(0, "h_a_0", 0.)
        leaves = []
        for rows in (fit_rows, selection_rows):
            t, features = fit_b.stored_deployable_features(rows, nuisance_model)
            child = partition.route(t, features)
            assert all(np.count_nonzero(child == leaf) >= 100 for leaf in (0, 32))
            leaves.append(child)
        return {"partition": partition, "coefficient_leaves": leaves[0],
                "checking_leaves": leaves[1], "status": "synthetic_supported_split",
                "split_history": [{"feature": "h_a_0", "threshold": 0.}],
                "rejected_candidates": [], "candidate_history": []}

    monkeypatch.setattr(fit_b, "select_nested_partition", supported_fixture_split)
    original_bank = fit_a.build_attack_bank

    def small_fresh_bank(*args, **kwargs):
        return original_bank(*args, **{**kwargs, "target_roles": ("A/SEX",)})

    monkeypatch.setattr(fit_a, "build_attack_bank", small_fresh_bank)
    b_dir = tmp_path / "private" / "B"
    args = (0, .001, role_dict, d17, old_q, "b"*64, selected_a, b_dir)
    first = fit_b.run_center_from_roles(*args, max_states=33, max_rounds=1)
    assert first["status"] == "COMPLETE"
    assert len(first["rounds"]) == 2
    assert all("inner_check_fixed_decoder_task" not in record
               for record in first["rounds"])
    selected = json.loads((b_dir / "SELECTED.json").read_text())
    with np.load(b_dir / selected["selected_channel_relative"],
                 allow_pickle=False) as archive:
        assert np.array_equal(first["selected_channel"], archive["Q"])
    complete = (b_dir / "COMPLETE.json").read_bytes()
    second = fit_b.run_center_from_roles(*args, max_states=33, max_rounds=1)
    assert np.array_equal(second["selected_channel"], first["selected_channel"])
    assert (b_dir / "COMPLETE.json").read_bytes() == complete
