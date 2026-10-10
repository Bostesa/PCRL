"""Follow-up worker admission and warm-state custody checks only."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.run_followup import (
    _admit,
    _amendment_guard,
    _warm_state,
)
from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.impl.model import HardCodeModel
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_file


def test_followup_rejects_no_queue_before_any_role_or_seal_path(monkeypatch):
    for key in tuple(__import__("os").environ):
        if key.startswith("PCRL_QUEUE_"):
            monkeypatch.delenv(key)
    with pytest.raises(RuntimeError, match="queue slot environment missing"):
        _admit(Namespace(unit_id="d-followup-test"))


def test_warm_state_checks_hash_inventory_and_float32(tmp_path):
    torch.manual_seed(2)
    model = HardCodeModel(16, (2, 6))
    arrays = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    path = tmp_path / "warm.npz"
    np.savez_compressed(path, **arrays)
    restored = _warm_state(path, sha_file(path), 16)
    assert set(restored) == set(model.state_dict())
    assert all(torch.equal(restored[name], value)
               for name, value in model.state_dict().items())
    with pytest.raises(PermissionError, match="byte hash mismatch"):
        _warm_state(path, "0" * 64, 16)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **{**arrays, "extra": np.zeros(1, dtype=np.float32)})
    with pytest.raises(ValueError, match="inventory"):
        _warm_state(bad, sha_file(bad), 16)


def test_followup_requires_latest_registered_a2_and_exact_pins(tmp_path, monkeypatch):
    seal = tmp_path / "CAMPAIGN_LOCK.json"
    seal.write_text("{}")
    amendment = tmp_path / "A2.json"
    worker = Path(__import__(
        "results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.run_followup",
        fromlist=["__file__"]).__file__)
    payload = {
        "schema": "pcrl-d-followup-amendment-v1",
        "status": "LOCALLY_SEALED_D_FOLLOWUP",
        "effective_seal_sha256": "a" * 64,
        "controls_worker_sha256": sha_file(worker),
        "first_receipt_sha256_by_seed": {"0": "b" * 64, "1": "c" * 64},
        "role_manifest_sha256": "d" * 64,
        "source_seed_sha256_by_seed": {"0": "e" * 64, "1": "f" * 64},
    }
    amendment.write_text(json.dumps(payload))
    digest = hashlib.sha256(amendment.read_bytes()).hexdigest()
    root = tmp_path / "queue"
    root.mkdir()
    state = {"amendments": [{"path": str(amendment), "sha256": digest,
                             "stages": ["d_followup"]}],
             "units": {"ctl": {"spec": {"amendment_sha256": digest}}}}
    (root / "state.json").write_text(json.dumps(state))
    monkeypatch.setenv("PCRL_QUEUE_ROOT", str(root))
    args = Namespace(amendment=str(amendment), amendment_sha256=digest,
                     unit_id="ctl", seal_sha256="a" * 64, seed=0,
                     first_receipt_sha256="b" * 64,
                     role_manifest_sha256="d" * 64,
                     source_sha256="e" * 64)
    _amendment_guard(args, {"seal_sha256": "a" * 64}, seal)
    state["amendments"].append({"path": str(tmp_path / "A3.json"),
                                "sha256": "0" * 64, "stages": ["d_followup"]})
    (root / "state.json").write_text(json.dumps(state))
    with pytest.raises(PermissionError, match="latest"):
        _amendment_guard(args, {"seal_sha256": "a" * 64}, seal)
