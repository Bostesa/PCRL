"""Hash-pinned read-only adapters from fitted units to common audit specs."""
import json

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import (
    evaluate, fit_a, fit_b, fit_controls, nuisance, refinement,
    release_specs, roles, task_baselines,
)
from experiments.pcrl_task_aligned_cuts_v1 import data


def _rows(seed, n, offset):
    rng = np.random.default_rng(seed)
    ha = rng.normal(size=(n, 4))
    return {"x": rng.normal(size=(n, 32)), "ha": ha,
            "hb": rng.normal(size=(n, 2)),
            "token_codes": rng.integers(0, 2, size=n, dtype=np.int64),
            "teacher_p": np.full(n, .5), "residual": np.zeros(n),
            "labels": {"same_residence": (ha[:, 0] > 0).astype(int),
                       "SEX": np.arange(n) % 2,
                       "RAC1P": np.arange(n) % 9},
            "weights": np.ones(n), "ids": np.arange(offset, offset+n),
            "households": np.asarray([f"release-spec-{offset+i}" for i in range(n)])}


def _nuisance_rows(n):
    result = _rows(60, n, 90000)
    households = []
    index = 0
    while len(households) < n:
        label = f"release-spec-nuisance-{index}"
        if roles.role_of(label) == "nuisance_train":
            households.append(label)
        index += 1
    result["households"] = np.asarray(households)
    return result


def _a_and_controls(tmp_path):
    role_dict = {"nuisance_train": _rows(1, 60, 0),
                 "audit_fit": _rows(2, 40, 1000),
                 "coefficient_split": _rows(3, 30, 2000),
                 "inner_selection": _rows(4, 20, 3000),
                 "inner_check": _rows(5, 20, 4000)}
    d17 = np.zeros((32, 17)); d17[:, 0] = 1
    old_q = np.zeros((32, 17)); old_q[:, 1] = 1
    a_dir = tmp_path / "private" / "A"
    fit_a.run_center_from_roles(0, .001, role_dict, d17, old_q,
                               "a"*64, a_dir, 0,
                               target_roles=("A/SEX",),
                               initial_sources=("H", "D17"))
    selected = fit_b.load_a_selected(a_dir, anchor=0, delta=.001)
    c_dir = tmp_path / "private" / "controls_A"
    fit_controls.fit_controls_from_frozen(
        {w: selected["cost"][w] for w in ("U", "W")},
        selected["cuts"], d17, old_q, c_dir,
        source={"branch": "A", "anchor": 0, "delta": .001,
                "center_sha256": selected["complete_receipt_sha256"]},
        milp_seconds=1., heuristic_seconds=.01, gradient_steps=2,
        gradient_penalties=(1.,), gradient_seeds=(17,))
    return a_dir, c_dir, role_dict["coefficient_split"]


def test_a_controls_and_task_only_specs_deduplicate_exact_named_aliases(tmp_path):
    a_dir, c_dir, rows = _a_and_controls(tmp_path)
    task_rows = _nuisance_rows(60)
    model, receipt = task_baselines.fit_task_only(task_rows, seed=31)
    task_dir = tmp_path / "private" / "task"
    task_baselines.save_task_only(task_dir, model, receipt)
    bundle = release_specs.build_release_specs(
        0, .001, a_dir, a_controls_dir=c_dir, task_only_dir=task_dir)
    assert bundle["aliases"]["TaskOnly_constant_100"] == bundle["aliases"]["TaskOnly"]
    assert bundle["aliases"]["TaskOnly_rr_100"] == bundle["aliases"]["TaskOnly"]
    assert "A_selected" in bundle["aliases"]
    assert "A_control_D17" in bundle["aliases"]
    assert len(bundle["aliases"]) > len(bundle["releases"])
    for canonical, spec in bundle["releases"].items():
        law = evaluate.token_law_for_release(spec, rows)
        assert law.shape == (len(rows["ha"]), 17), canonical
        assert np.allclose(law.sum(axis=1), 1)
    assert bundle["aliases"]["A_control_D17"] in bundle["releases"]
    assert bundle["aliases"]["TaskOnly_rr_075"] in bundle["releases"]
    complete_path = c_dir / "COMPLETE.json"
    original_complete = complete_path.read_bytes()
    wrong_cost = json.loads(original_complete)
    wrong_cost["cost_pair_sha256"]["U"] = "b"*64
    complete_path.write_text(json.dumps(wrong_cost))
    with pytest.raises(ValueError, match="cost|bank"):
        release_specs.build_release_specs(
            0, .001, a_dir, a_controls_dir=c_dir, task_only_dir=task_dir)
    wrong_bank = json.loads(original_complete)
    wrong_bank["fixed_bank_sha256"] = "c"*64
    complete_path.write_text(json.dumps(wrong_bank))
    with pytest.raises(ValueError, match="cost|bank"):
        release_specs.build_release_specs(
            0, .001, a_dir, a_controls_dir=c_dir, task_only_dir=task_dir)
    complete_path.write_bytes(original_complete)
    control = c_dir / "channels" / "D17" / "Q.npz"
    control.write_bytes(control.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash|inventory|artifact"):
        release_specs.build_release_specs(
            0, .001, a_dir, a_controls_dir=c_dir, task_only_dir=task_dir)


def test_refined_b_router_uses_only_legal_stored_fields_and_checks_runtime_pins(tmp_path):
    role_dict = {"nuisance_train": _nuisance_rows(60),
                 "audit_fit": _rows(2, 40, 1000),
                 "coefficient_split": _rows(3, 30, 2000),
                 "inner_selection": _rows(4, 20, 3000),
                 "inner_check": _rows(5, 20, 4000)}
    d17 = np.zeros((32, 17)); d17[:, 0] = 1
    old_q = np.zeros((32, 17)); old_q[:, 1] = 1
    a_dir = tmp_path / "private" / "A"
    fit_a.run_center_from_roles(0, .001, role_dict, d17, old_q,
                               "a"*64, a_dir, 0,
                               target_roles=("A/SEX",),
                               initial_sources=("H", "D17"))
    selected_a = fit_b.load_a_selected(a_dir, anchor=0, delta=.001)
    b_dir = tmp_path / "private" / "B"
    fit_b.run_center_from_roles(0, .001, role_dict, d17, old_q,
                               "a"*64, selected_a, b_dir,
                               max_states=64, max_rounds=0)
    complete = json.loads((b_dir / "COMPLETE.json").read_text())
    assert complete["status"] == "SUPPORT_LIMITED_ALIAS_A"
    partition = refinement.NestedPartition.from_record(
        json.loads((b_dir / "PARTITION.json").read_text()))
    with np.load(b_dir / complete["selected_channel_relative"],
                 allow_pickle=False) as archive:
        q = archive["Q"].copy()
    nuisance_record = json.loads((b_dir / "nuisance/NUISANCE.json").read_text())
    parity = {"schema": "pcrl-refined-runtime-parity-v1",
              "system": "Linux", "machine": "x86_64",
              "linux_x86_parity_verified": True,
              "center_complete_sha256": data.sha256_file(b_dir / "COMPLETE.json"),
              "selected_channel_relative": complete["selected_channel_relative"],
              "selected_channel_array_sha256": fit_b._array_sha256(q),
              "channel_array_sha256": fit_b._array_sha256(q),
              "partition_record": partition.to_record(),
              "encoder_sha256": "a"*64,
              "nuisance_sha256": nuisance_record["model_sha256"],
              "fixture_input_sha256": "b"*64,
              "fixture_h_a_sha256": "c"*64,
              "task_posterior_max_absolute_difference": 0.,
              "residual_max_absolute_difference": 0.,
              "t32_bitwise_equal": True, "child_bitwise_equal": True,
              "service_byte_equal": True, "checked_original_people": 60}
    parity_path = tmp_path / "private" / "B_PARITY.json"
    parity_path.write_text(json.dumps(parity))
    spec = release_specs.load_refined_b_spec(
        b_dir, parity_path, anchor=0, delta=.001,
        a_center_dir=a_dir, require_current_linux=False)
    legal = {"x": np.zeros((2, 32)),
             "ha": np.array([[-1., 0, 0, 0], [1., 0, 0, 0]]),
             "token_codes": np.zeros(2, dtype=np.int64),
             "teacher_p": np.full(2, .5), "residual": np.zeros(2)}
    law = evaluate.token_law_for_release(spec, legal)
    assert np.array_equal(law, q[[0, 0]])
    with pytest.raises(ValueError, match="legal|forbidden"):
        spec["router"]({**legal, "labels": {"SEX": np.array([0, 1])}})
    (b_dir / "PARTITION.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="hash|inventory|artifact"):
        evaluate.token_law_for_release(spec, legal)


def test_non_alias_refined_b_selected_artifact_and_child_routing(tmp_path, monkeypatch):
    role_dict = {"nuisance_train": _nuisance_rows(60),
                 "audit_fit": _rows(12, 40, 12000),
                 "coefficient_split": _rows(13, 220, 13000),
                 "inner_selection": _rows(14, 220, 14000),
                 "inner_check": _rows(15, 20, 15000)}
    for role in ("coefficient_split", "inner_selection"):
        rows = role_dict[role]
        rows["token_codes"][:] = 0
        rows["ha"][:, 0] = np.linspace(-1., 1., len(rows["ha"]))
        rows["labels"]["same_residence"] = (rows["ha"][:, 0] > 0).astype(int)
    d17 = np.zeros((32, 17)); d17[:, 0] = 1
    old_q = np.zeros((32, 17)); old_q[:, 1] = 1
    a_dir = tmp_path / "private" / "A"
    fit_a.run_center_from_roles(0, .001, role_dict, d17, old_q,
                               "a"*64, a_dir, 0,
                               target_roles=("A/SEX",),
                               initial_sources=("H", "D17"))
    selected_a = fit_b.load_a_selected(a_dir, anchor=0, delta=.001)

    def supported_split(fit_rows, selection_rows, frozen_nuisance,
                        g_fit, g_selection, **kwargs):
        partition = refinement.NestedPartition.base(32).split(0, "h_a_0", 0.)
        leaves = []
        for rows in (fit_rows, selection_rows):
            t, features = fit_b.stored_deployable_features(rows, frozen_nuisance)
            child = partition.route(t, features)
            assert all(np.count_nonzero(child == leaf) >= 100
                       for leaf in (0, 32))
            leaves.append(child)
        return {"partition": partition, "coefficient_leaves": leaves[0],
                "checking_leaves": leaves[1],
                "status": "synthetic_supported_split",
                "split_history": [{"feature": "h_a_0", "threshold": 0.}],
                "rejected_candidates": [], "candidate_history": []}

    monkeypatch.setattr(fit_b, "select_nested_partition", supported_split)
    b_dir = tmp_path / "private" / "B"
    fit_b.run_center_from_roles(0, .001, role_dict, d17, old_q,
                               "a"*64, selected_a, b_dir,
                               max_states=33, max_rounds=0)
    complete = json.loads((b_dir / "COMPLETE.json").read_text())
    assert complete["status"] == "COMPLETE"
    partition = refinement.NestedPartition.from_record(
        json.loads((b_dir / "PARTITION.json").read_text()))
    with np.load(b_dir / complete["selected_channel_relative"],
                 allow_pickle=False) as archive:
        q = archive["Q"].copy()
    nuisance_record = json.loads((b_dir / "nuisance/NUISANCE.json").read_text())
    parity = {"schema": "pcrl-refined-runtime-parity-v1",
              "system": "Linux", "machine": "x86_64",
              "linux_x86_parity_verified": True,
              "center_complete_sha256": data.sha256_file(b_dir / "COMPLETE.json"),
              "selected_channel_relative": complete["selected_channel_relative"],
              "selected_channel_array_sha256": fit_b._array_sha256(q),
              "channel_array_sha256": fit_b._array_sha256(q),
              "partition_record": partition.to_record(),
              "encoder_sha256": "a"*64,
              "nuisance_sha256": nuisance_record["model_sha256"],
              "fixture_input_sha256": "b"*64,
              "fixture_h_a_sha256": "c"*64,
              "task_posterior_max_absolute_difference": 0.,
              "residual_max_absolute_difference": 0.,
              "t32_bitwise_equal": True, "child_bitwise_equal": True,
              "service_byte_equal": True, "checked_original_people": 220}
    parity_path = tmp_path / "private" / "B_PARITY.json"
    parity_path.write_text(json.dumps(parity))
    spec = release_specs.load_refined_b_spec(
        b_dir, parity_path, anchor=0, delta=.001,
        require_current_linux=False)
    legal = {"x": np.zeros((2, 32)),
             "ha": np.array([[-1., 0, 0, 0], [1., 0, 0, 0]]),
             "token_codes": np.zeros(2, dtype=np.int64),
             "teacher_p": np.full(2, .5), "residual": np.zeros(2)}
    assert np.array_equal(spec["router"](legal), np.array([0, 32]))
    assert np.array_equal(evaluate.token_law_for_release(spec, legal), q[[0, 32]])
    parity_path.write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="hash|parity"):
        evaluate.token_law_for_release(spec, legal)
