"""Data-free complete reader and real-control output accounting."""

import json
from pathlib import Path

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r2.attack_fit import (
    CORE, RAW, _validated_local_aliases, expected_outputs,
)
from results.pcrl_utility_reference_repair_v1.r1.role_store import sha_file


def test_six_reader_and_two_probe_artifact_inventory():
    output = Path("/tmp/.private/attacks/seed0/pair/income/ATTACK_FIT_INDEX.json")
    files = expected_outputs(output)
    assert len(files) == 16
    assert len(CORE) == 4 and len(RAW) == 2
    assert str(output.parent / "controls/planted_model.joblib") in files
    assert str(output.parent / "core/model__DA_MLP_64x32.joblib") in files
    assert str(output.parent / "ALL_READER_PREDICTIONS.npz") in files


def test_coalition_alias_accepts_sealed_prior_local_lock(tmp_path):
    class View:
        def __init__(self):
            self.row_id = np.array([1, 2, 3], dtype=np.int64)
            self.unit = self.row_id.copy()

        def content_hashes(self):
            return {"wire_schema": "a" * 64}

    local_lock, coalition_lock = "a" * 64, "b" * 64
    view = View()
    index_path = tmp_path / "ATTACK_FIT_INDEX.json"
    vectors = {reader: np.array([.1, .5, .9], dtype=np.float64)
               for reader in CORE + RAW}
    tune_path = tmp_path / "ALL_READER_TUNE_PREDICTIONS.npz"
    inner_path = tmp_path / "ALL_READER_PREDICTIONS.npz"
    np.savez_compressed(tune_path, **vectors)
    np.savez_compressed(inner_path, **vectors)
    from results.pcrl_utility_reference_repair_v1.r2.attack_fit import _pair_sha
    index = {
        "schema": "pcrl-utility-complete-att-fit-v1",
        "view_name": "income", "seed": 0, "pair_id": "PAIR",
        "release_index_sha256": "c" * 64,
        "lock_sha256": local_lock, "manifest_sha256": "d" * 64,
        "att_view_content_sha256": view.content_hashes(),
        "inner_view_content_sha256": view.content_hashes(),
        "att_row_unit_sha256": _pair_sha(view),
        "inner_row_unit_sha256": _pair_sha(view),
        "reader_order": list(CORE + RAW),
        "all_prediction_sha256": sha_file(inner_path),
        "all_tune_sha256": sha_file(tune_path),
    }
    index_path.write_text(json.dumps(index))
    _, _, custody = _validated_local_aliases(
        "income", index_path, sha_file(index_path), seed=0, pair_id="PAIR",
        release_sha="c" * 64, att_view=view, inner_view=view,
        coalition_att=view, coalition_inner=view,
        lock_sha=local_lock, manifest_sha="d" * 64)
    assert custody["new_fit_count"] == 0
    with pytest.raises(PermissionError, match="local reader source differs"):
        _validated_local_aliases(
            "income", index_path, sha_file(index_path), seed=0,
            pair_id="PAIR", release_sha="c" * 64,
            att_view=view, inner_view=view, coalition_att=view,
            coalition_inner=view, lock_sha=coalition_lock,
            manifest_sha="d" * 64)
