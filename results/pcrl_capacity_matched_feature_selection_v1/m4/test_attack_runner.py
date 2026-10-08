"""Synthetic local-reader coalition import, with no new fit or role read."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "results/pcrl_method_portfolio_v1/shared"))

import attack_runner
from attack_runner import ALL_READER_IDS, load_local_reader_aliases
from safe_data_guard import sha_arr


class View:
    def __init__(self, first, content):
        self.row_id = np.asarray([first, first + 1], dtype=np.int64)
        self.unit = np.asarray([first + 100, first + 101], dtype=np.int64)
        self.content = content

    def content_hashes(self):
        return {"x": self.content, "wire_schema": "fixed"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(attack_runner, "attack_fit_tune_split",
                        lambda unit: {"tune": np.asarray([1]),
                                      "tune_group_id_sha256": "a" * 64})
    root = tmp_path / ".private" / "one"
    core = root / "core_readers"
    core.mkdir(parents=True)
    (core / "READERS_RECEIPT.json").write_text(json.dumps({"files": {}}))
    att, inner = View(1, "att-content"), View(101, "inner-content")
    for role, length, file in (("inner", 2, "ALL_READER_PREDICTIONS.npz"),
                               ("tune", 1, "ALL_READER_TUNE_PREDICTIONS.npz")):
        np.savez_compressed(root / file, **{
            name: np.full(length, 0.25 + index * 0.01)
            for index, name in enumerate(ALL_READER_IDS)
        })
    record = {
        "schema": "pcrl-focused-complete-reader-index-v1",
        "status": "FROZEN", "view_name": "income",
        "pair_id": "PAIR", "seed": 0,
        "lock_sha256": "b" * 64, "manifest_sha256": "c" * 64,
        "att_view_content_sha256": att.content_hashes(),
        "inner_view_content_sha256": inner.content_hashes(),
        "att_row_unit_sha256": sha_arr(np.column_stack((att.row_id, att.unit))),
        "inner_row_unit_sha256": sha_arr(np.column_stack((inner.row_id, inner.unit))),
        "attack_split_sha256": "a" * 64,
        "reader_order": list(ALL_READER_IDS),
        "core_readers_receipt_sha256": _sha(core / "READERS_RECEIPT.json"),
        "all_reader_predictions_sha256": _sha(root / "ALL_READER_PREDICTIONS.npz"),
        "all_reader_tune_predictions_sha256": _sha(
            root / "ALL_READER_TUNE_PREDICTIONS.npz"),
    }
    path = root / "ATTACK_INDEX.json"
    path.write_text(json.dumps(record))
    return path, _sha(path), att, inner


def test_local_complete_readers_import_with_exact_wire_and_people(tmp_path, monkeypatch):
    path, digest, att, inner = _fixture(tmp_path, monkeypatch)
    scores, receipt = load_local_reader_aliases(
        local_index_path=path, expected_index_sha256=digest,
        expected_view_name="income", expected_pair_id="PAIR",
        att_local_view=att, inner_local_view=inner, seed=0,
        lock_sha256="b" * 64, manifest_sha256="c" * 64)
    assert set(scores) == {"tune", "inner"}
    assert set(scores["inner"]) == set(ALL_READER_IDS)
    assert receipt["reader_count"] == 6


def test_local_reader_import_rejects_other_wire_or_modified_score(tmp_path, monkeypatch):
    path, digest, att, inner = _fixture(tmp_path, monkeypatch)
    bad_att = View(1, "changed")
    with pytest.raises(PermissionError, match="recipient wire"):
        load_local_reader_aliases(
            local_index_path=path, expected_index_sha256=digest,
            expected_view_name="income", expected_pair_id="PAIR",
            att_local_view=bad_att, inner_local_view=inner, seed=0,
            lock_sha256="b" * 64, manifest_sha256="c" * 64)
    with np.load(path.parent / "ALL_READER_PREDICTIONS.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays[ALL_READER_IDS[0]][0] = 0.99
    np.savez_compressed(path.parent / "ALL_READER_PREDICTIONS.npz", **arrays)
    with pytest.raises(PermissionError, match="scores changed"):
        load_local_reader_aliases(
            local_index_path=path, expected_index_sha256=digest,
            expected_view_name="income", expected_pair_id="PAIR",
            att_local_view=att, inner_local_view=inner, seed=0,
            lock_sha256="b" * 64, manifest_sha256="c" * 64)
