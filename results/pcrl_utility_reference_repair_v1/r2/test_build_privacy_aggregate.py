"""Data-free checks for post-replay aggregate publication."""

import hashlib
import json

import pytest

from results.pcrl_utility_reference_repair_v1.r2.build_privacy_aggregate import (
    _view_rows, replay_admitted,
)


def test_replay_admission_requires_done_matching_output_and_pass(tmp_path):
    path = tmp_path / "PRIVACY_REPLAY.json"
    path.write_text(json.dumps({"test": True}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    replay = {
        "schema": "pcrl-utility-r3-independent-privacy-replay-v1",
        "status": "REPLAYED", "all_control_gates_pass": True,
        "max_abs_reported_metric_difference": 0.0,
    }
    queue = {"units": {"R3_PRIVACY_REPLAY": {
        "status": "DONE", "output_sha256": {str(path): digest},
    }}}
    replay_admitted(replay, queue, path, "R3_PRIVACY_REPLAY")
    with pytest.raises(PermissionError, match="did not pass"):
        replay_admitted({**replay, "status": "CONTROL_FAILED_REPLAYED"},
                        queue, path, "R3_PRIVACY_REPLAY")
    with pytest.raises(PermissionError, match="did not pass"):
        replay_admitted(replay, {"units": {"R3_PRIVACY_REPLAY": {
            "status": "FAILED", "output_sha256": {str(path): digest}}}},
                        path, "R3_PRIVACY_REPLAY")
    with pytest.raises(PermissionError, match="did not pass"):
        replay_admitted(replay, {"units": {"R3_PRIVACY_REPLAY": {
            "status": "DONE", "output_sha256": {str(path): "0" * 64}}}},
                        path, "R3_PRIVACY_REPLAY")


def test_view_rows_preserve_all_readers_and_reject_replay_difference():
    score = {
        "schema": "pcrl-utility-complete-privacy-score-v1", "seed": 0,
        "pair_id": "PAIR", "view_name": "coalition", "reader_order": ["A", "B"],
        "readers": {
            "A": {"positive_alias_auc": .8, "reverse_alias_auc": .2,
                  "fixed_alias_max_auc": .8},
            "B": {"positive_alias_auc": .7, "reverse_alias_auc": .3,
                  "fixed_alias_max_auc": .7},
        },
        "primary_envelope_auc": .8, "validation_winner": "B",
        "validation_winner_inner_auc": .7,
        "controls": {"status": "PASS"},
    }
    replay = {"controls_status": "PASS", "primary_envelope_auc": .8,
              "validation_winner": "B",
              "validation_winner_inner_auc": .7}
    rows = _view_rows(0, "PAIR", "coalition", ["J", "T"], score, replay,
                      "a" * 64, "b" * 64, "c" * 64)
    assert len(rows) == 2
    assert [row["reader_id"] for row in rows] == ["A", "B"]
    assert all(row["rule_ids"] == "J,T" for row in rows)
    with pytest.raises(PermissionError, match="independent privacy endpoint"):
        _view_rows(0, "PAIR", "coalition", ["J"], score,
                   {**replay, "primary_envelope_auc": .9},
                   "a" * 64, "b" * 64, "c" * 64)
