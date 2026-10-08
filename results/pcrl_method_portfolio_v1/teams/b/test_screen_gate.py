"""Preseal Team B startup smoke against the admitted S3 queue record.

S3 supplies the real queue root and preserved R1 unit ID. This test reads only
lock/config/manifest metadata, queue records and the S2 synthetic receipt. It
does not request any Adult role archive or label value. Only the independently
checked local-seal/occupancy flags are simulated because the science gate is
intentionally closed until this correction receives its new seal.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from compute_queue import Queue
from fit_input_bridge import build_fit_input
from safe_data_guard import SafeRoleStore
from screen_driver import _check_gate, _sha


def _fixture():
    queue_root = os.environ["PCRL_B_GATE_QUEUE_ROOT"]
    prior_unit_id = os.environ["PCRL_B_GATE_PRIOR_UNIT_ID"]
    queue = Queue(queue_root)
    state = queue.status()
    prior = state["units"][prior_unit_id]
    if prior["spec"].get("team") != "B" or prior["spec"].get("stage") != "screen":
        raise AssertionError("Expected the preserved Team B screen R1 queue record")
    argv = prior["spec"]["argv"]
    values = {argv[index][2:].replace("-", "_"): argv[index + 1]
              for index in range(len(argv) - 1) if argv[index].startswith("--")}
    values["seed"] = int(values["seed"])
    # R1's preserved argv records the old lock SHA. The amended lock is
    # separately sealed before this queued gate replay runs.
    for field, path_field in (("lock_sha256", "campaign_lock"),
                              ("config_sha256", "screen_config"),
                              ("manifest_sha256", "data_manifest")):
        values[field] = _sha(Path(values[path_field]))
    synthetic_current_id = "B-SYNTHETIC-GATE-REPLAY"
    values["unit_id"] = synthetic_current_id
    args = SimpleNamespace(**values)
    mocked = copy.deepcopy(state)
    mocked["gate"]["safe_inputs"] = True
    mocked["gate"]["occupancy_confirmed"] = True
    current = copy.deepcopy(prior)
    current["unit_id"] = synthetic_current_id
    current["status"] = "running"
    current["child_pgid"] = os.getpgrp()
    current["spec"]["unit_id"] = synthetic_current_id
    current["spec"]["argv"] = [synthetic_current_id if item == prior_unit_id else item
                                for item in argv]
    mocked["units"][synthetic_current_id] = current
    return args, mocked


def test_actual_s2_smoke_record_passes_full_b_startup_gate_without_role_values():
    args, state = _fixture()
    with patch.object(Queue, "status", return_value=state), \
            patch.object(Queue, "_local_seal_ok", return_value=True), \
            patch.object(SafeRoleStore, "read", side_effect=AssertionError("role value read")) as read:
        lock, config = _check_gate(args)
        store = SafeRoleStore(args.data_manifest)
        with pytest.raises(AssertionError, match="role value read"):
            build_fit_input(store, arm=args.arm, seed=args.seed)
        read.assert_called_once_with("OSF_DEFENSE_FIT", "features")
    assert lock["fit_admission"] == "B_ONLY"
    assert config["status"] == "EFFECTIVE"
    assert store.manifest["schema"] == "pcrl-method-portfolio-data-access-v1"
    smoke = state["units"][lock["synthetic_attack_plumbing"]["s3_unit_id"]]
    assert lock["synthetic_attack_plumbing"]["s3_unit_mode"] == "preseal_smoke"
    assert lock["synthetic_attack_plumbing"]["s3_unit_team"] == "S2_PLUMBING_027"
    assert smoke["spec"]["mode"] == lock["synthetic_attack_plumbing"]["s3_unit_mode"]
    assert smoke["spec"]["team"] == lock["synthetic_attack_plumbing"]["s3_unit_team"]


@pytest.mark.parametrize("field,wrong", [
    ("mode", "prelock_synthetic"),
    ("team", "S2"),
])
def test_b_gate_refuses_mutated_s2_smoke_unit_metadata(field, wrong):
    args, state = _fixture()
    smoke_id = json.loads(Path(args.campaign_lock).read_text())["synthetic_attack_plumbing"]["s3_unit_id"]
    assert state["units"][smoke_id]["spec"][field] != wrong
    state["units"][smoke_id]["spec"][field] = wrong
    with patch.object(Queue, "status", return_value=state), \
            patch.object(Queue, "_local_seal_ok", return_value=True), \
            patch.object(SafeRoleStore, "read", side_effect=AssertionError("role value read")):
        with pytest.raises(PermissionError, match="synthetic attack-plumbing smoke"):
            _check_gate(args)
