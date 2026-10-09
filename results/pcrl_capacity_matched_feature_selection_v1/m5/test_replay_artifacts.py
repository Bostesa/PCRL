"""Tamper checks for independently decoded, hash-bound private releases."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from results.pcrl_capacity_matched_feature_selection_v1.m5.replay_artifacts import (
    read_frozen_release,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arr_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _fixture(tmp_path):
    private = tmp_path / ".private"
    release = private / "release"
    release.mkdir(parents=True)
    x = np.array([[0.0, -0.0, 3.0], [1.0, 2.0, 4.0]], dtype=np.float32)
    row = np.array([4, 5], dtype=np.int64)
    unit = np.array([7, 8], dtype=np.int64)
    arrays = {
        "income__wire__selected_values": x[:, [0, 1]].copy(),
        "occupation_group__wire__selected_values": x[:, [2]].copy(),
        "coalition__wire__selected_values": x.copy(),
        "income__wire__probabilities": np.array([[.8, .2], [.2, .8]], dtype=np.float64),
        "occupation_group__wire__probabilities": np.full((2, 6), 1 / 6, dtype=np.float64),
        "coalition__wire__income_probabilities": np.array([[.8, .2], [.2, .8]], dtype=np.float64),
        "coalition__wire__occupation_group_probabilities": np.full((2, 6), 1 / 6, dtype=np.float64),
    }
    for view in ("income", "occupation_group", "coalition"):
        arrays[f"{view}__view__row_id"] = row.copy()
        arrays[f"{view}__view__unit"] = unit.copy()
    npz = release / "INNER_SELECTION.npz"
    np.savez_compressed(npz, **arrays)
    view_receipt = {
        "schema": "team-c-private-complete-view-v1", "role": "INNER_SELECTION",
        "arm_id": "pair-a", "seed": 0, "campaign_lock_sha256": "a" * 64,
        "data_manifest_sha256": "b" * 64, "role_feature_sha256": "c" * 64,
        "package_sha256_by_purpose": {"income": "d" * 64,
                                      "occupation_group": "e" * 64},
        "npz_sha256": _sha(npz),
        "array_sha256": {key: _arr_sha(value) for key, value in arrays.items()},
    }
    view_path = release / "INNER_SELECTION.json"
    view_path.write_text(json.dumps(view_receipt))
    index = {
        "schema": "pcrl-focused-complete-release-index-v1", "status": "FROZEN",
        "seed": 0, "pair_id": "pair-a", "lock_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "package_sha256_by_purpose": view_receipt["package_sha256_by_purpose"],
        "roles": {"INNER_SELECTION": {
            "view_receipt_file": view_path.name,
            "view_receipt_sha256": _sha(view_path),
            "view_npz_sha256": _sha(npz),
            "raw_union_columns": [0, 1, 2],
        }},
    }
    index_path = release / "RELEASE_RECEIPT.json"
    index_path.write_text(json.dumps(index))
    kwargs = dict(index_path=index_path, index_sha256=_sha(index_path),
                  private_root=private, role="INNER_SELECTION", seed=0,
                  pair_id="pair-a", lock_sha256="a" * 64,
                  manifest_sha256="b" * 64,
                  role_feature_sha256="c" * 64,
                  source_x=x, source_row_id=row, source_unit=unit,
                  selected_columns={"income": (0, 1),
                                    "occupation_group": (2,),
                                    "coalition": (0, 1, 2)})
    return kwargs, npz, view_path


def test_frozen_release_replays_all_three_raw_views_and_local_probability_wires(tmp_path):
    kwargs, _, _ = _fixture(tmp_path)
    result = read_frozen_release(**kwargs)
    assert set(result["candidate_probabilities"]) == {"income", "occupation_group"}
    assert result["candidate_probabilities"]["income"][1, 1] == .8
    assert result["source_row_unit_count"] == 2


def test_signed_zero_source_change_and_npz_tamper_are_rejected(tmp_path):
    kwargs, npz, _ = _fixture(tmp_path)
    changed = kwargs["source_x"].copy()
    changed[0, 1] = +0.0
    kwargs["source_x"] = changed
    with pytest.raises(PermissionError, match="source"):
        read_frozen_release(**kwargs)
    kwargs["source_x"][0, 1] = -0.0
    npz.write_bytes(npz.read_bytes() + b"tamper")
    with pytest.raises(PermissionError, match="NPZ"):
        read_frozen_release(**kwargs)
