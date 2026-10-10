"""Follow-up worker admission and warm-state custody checks only."""

from argparse import Namespace

import numpy as np
import pytest
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.run_followup import (
    _admit,
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
