"""Synthetic C view replay checks. No Adult roles or Team C imports."""

import hashlib
import json

import numpy as np
import pytest

from replay_c_values_v1 import (atom_indices, head_probabilities,
                                load_head_checkpoint, load_saved_bundle,
                                rebuild_views, sha_array, temperature_apply,
                                view_content_hashes)


def _names():
    return (
        "age", "education-num", "capital-gain", "capital-loss", "hours-per-week",
        *(f"workclass=w{i}" for i in range(8)),
        *(f"education=e{i}" for i in range(16)),
        *(f"marital-status=m{i}" for i in range(7)),
        *(f"relationship=r{i}" for i in range(6)),
        *(f"native-country=n{i}" for i in range(41)),
    )


def _fixture():
    names = _names()
    x = np.arange(3 * 83, dtype=np.float64).reshape(3, 83) / 100
    indices = atom_indices(names)
    income_groups = ("age", "workclass")
    occupation_groups = ("age", "education_fact")
    union = ("age", "education_fact", "workclass")
    def columns(groups):
        return tuple(sorted(i for group in groups for i in indices[group]))
    income = np.array([[.8, .2], [.3, .7], [.6, .4]])
    occupation = np.tile(np.array([.1, .2, .1, .3, .1, .2]), (3, 1))
    raw_income = np.array([[.7, .3], [.4, .6], [.55, .45]])
    raw_occupation = np.tile(np.array([.2, .2, .1, .2, .1, .2]), (3, 1))
    wires = {
        "income": {
            "selected_values": x[:, columns(income_groups)].copy(),
            "feature_names": tuple(names[i] for i in columns(income_groups)),
            "selected_groups": income_groups, "mask_id": "age+workclass",
            "head_id": "head-i", "calibrator_id": "cal-i", "flags": (),
            "probabilities": income, "decisions": income.argmax(axis=1),
        },
        "occupation_group": {
            "selected_values": x[:, columns(occupation_groups)].copy(),
            "feature_names": tuple(names[i] for i in columns(occupation_groups)),
            "selected_groups": occupation_groups, "mask_id": "age+education_fact",
            "head_id": "head-o", "calibrator_id": "cal-o", "flags": (),
            "probabilities": occupation, "decisions": occupation.argmax(axis=1),
        },
        "coalition": {
            "selected_values": x[:, columns(union)].copy(),
            "feature_names": tuple(names[i] for i in columns(union)),
            "selected_groups": union, "income_groups": income_groups,
            "occupation_group_groups": occupation_groups,
            "income_mask_id": "age+workclass",
            "occupation_group_mask_id": "age+education_fact",
            "income_head_id": "head-i", "occupation_group_head_id": "head-o",
            "income_calibrator_id": "cal-i", "occupation_group_calibrator_id": "cal-o",
            "flags": (), "income_probabilities": income.copy(),
            "occupation_group_probabilities": occupation.copy(),
            "income_decisions": income.argmax(axis=1),
            "occupation_group_decisions": occupation.argmax(axis=1),
        },
    }
    return x, names, wires, {"income": raw_income,
                             "occupation_group": raw_occupation}


def test_rebuilds_all_transmitted_raw_and_defense_aware_columns():
    x, names, wires, raw = _fixture()
    views = rebuild_views(x, names, wires, raw)
    assert set(views) == {"income", "occupation_group", "coalition"}
    for name in views:
        assert np.array_equal(views[name]["x_raw_only"], wires[name]["selected_values"])
        assert views[name]["x"].shape[1] == (views[name]["x_raw_only"].shape[1]
                                               + (16 if name == "coalition" else
                                                  4 if name == "income" else 12))
    assert np.array_equal(views["income"]["x_defense_aware"][:, -2:], raw["income"])
    assert np.array_equal(views["coalition"]["x_defense_aware"][:, -6:],
                          raw["occupation_group"])
    hashes = view_content_hashes(views["coalition"])
    assert hashes["x"] == sha_array(views["coalition"]["x"])
    assert hashes["x_raw_only"] == sha_array(
        views["coalition"]["x_raw_only"])
    assert hashes["x_metadata"] is None and hashes["cell_key"] is None


def test_refuses_selected_value_or_coalition_union_mutation():
    x, names, wires, raw = _fixture()
    wires["income"]["selected_values"][0, 0] += 1
    with pytest.raises(ValueError, match="selected raw"):
        rebuild_views(x, names, wires, raw)
    x, names, wires, raw = _fixture()
    wires["coalition"]["selected_values"][0, 0] += 1
    with pytest.raises(ValueError, match="selected raw"):
        rebuild_views(x, names, wires, raw)


def test_refuses_missing_public_field_and_wrong_decision():
    x, names, wires, raw = _fixture()
    del wires["income"]["head_id"]
    with pytest.raises(ValueError, match="wire fields"):
        rebuild_views(x, names, wires, raw)
    x, names, wires, raw = _fixture()
    wires["occupation_group"]["decisions"][0] = 5
    with pytest.raises(ValueError, match="decision"):
        rebuild_views(x, names, wires, raw)


def test_replays_hash_bound_head_and_temperature_without_candidate_import(tmp_path):
    names = _names()
    meta = {
        "schema": "team-c-head-checkpoint-v1", "purpose": "income",
        "model_id": "synthetic-income", "selected_atoms": ["age"],
        "selected_columns": 1,
        "spec": {"hidden_width": 3},
        "receipt": {"feature_schema_sha256":
                    hashlib.sha256("\n".join(names).encode()).hexdigest()},
        "state_keys": ["0.weight", "0.bias", "2.weight", "2.bias",
                       "4.weight", "4.bias"],
    }
    weights = (
        np.zeros((3, 1), dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32), np.array([0., 1.], dtype=np.float32),
    )
    path = tmp_path / "head.npz"
    np.savez_compressed(path, meta=np.array(json.dumps(meta)),
                        **{f"state_{i}": arr for i, arr in enumerate(weights)})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    head = load_head_checkpoint(path, digest, names)
    raw = head_probabilities(head, np.array([[1.], [2.]], dtype=np.float64))
    expected = np.exp([0., 1.]) / np.exp([0., 1.]).sum()
    assert np.allclose(raw, [expected, expected], rtol=0, atol=1e-14)
    calibrated = temperature_apply(raw, {"schema": "pcrl-portfolio-global-temp-v1",
                                          "task": "income", "alpha": 2.0})
    assert np.allclose(calibrated[0], expected ** 2 / (expected ** 2).sum(),
                       rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match="checkpoint SHA"):
        load_head_checkpoint(path, "0" * 64, names)


def test_saved_complete_view_bundle_matches_independent_rebuild(tmp_path):
    x, names, wires, raw = _fixture()
    views = rebuild_views(x, names, wires, raw)
    arrays = {}
    constants = {}
    schemas = {}
    for name in ("income", "occupation_group", "coalition"):
        constants[name] = {key: value for key, value in wires[name].items()
                           if not isinstance(value, np.ndarray)}
        for key, value in wires[name].items():
            if isinstance(value, np.ndarray):
                arrays[f"{name}__wire__{key}"] = value
        for key in ("x", "x_defense_aware", "x_raw_only"):
            arrays[f"{name}__view__{key}"] = views[name][key]
        for key in ("row_id", "unit"):
            arrays[f"{name}__view__{key}"] = np.arange(3, dtype=np.int64)
        schemas[name] = {key: views[name][key] for key in
                         ("components", "wire_components", "component_columns",
                          "wire_constants")}
    npz = tmp_path / "view.npz"
    np.savez_compressed(npz, **arrays)
    receipt = {
        "schema": "team-c-private-complete-view-v1",
        "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "array_sha256": {key: sha_array(value) for key, value in arrays.items()},
        "wire_constants": constants, "view_schemas": schemas,
    }
    receipt_path = tmp_path / "view.json"
    receipt_path.write_text(json.dumps(receipt))
    row_id = np.arange(3, dtype=np.int64)
    unit = np.arange(3, dtype=np.int64)
    recovered = load_saved_bundle(receipt_path, npz, x, names, raw,
                                  row_id, unit)
    assert set(recovered["views"]) == set(views)
    assert np.array_equal(recovered["views"]["coalition"]["x"],
                          views["coalition"]["x"])
    receipt["array_sha256"]["income__view__x"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="array SHA"):
        load_saved_bundle(receipt_path, npz, x, names, raw, row_id, unit)
    receipt["array_sha256"]["income__view__x"] = sha_array(
        arrays["income__view__x"])
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="ordered row/unit"):
        load_saved_bundle(receipt_path, npz, x, names, raw,
                          row_id + 1, unit)
