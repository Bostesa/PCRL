"""Tiny data-free checks for the independent P* class-temperature replay."""

from __future__ import annotations

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_source import sha_array
from results.pcrl_latent_and_sampled_methods_v1.r04.independent_pstar import check_pstar_bank


def _fixture():
    proto = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    cls = np.array([0, 1], dtype=np.int64)
    alpha = np.array([0.5, 1.3], dtype=np.float64)
    logits = alpha[cls, None] * np.log(proto)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    bank = {"probability": p, "token_id": np.arange(2, dtype=np.int64),
            "token_class": cls, "input_class": cls.copy(),
            "fine_to_hcal_token": np.arange(2, dtype=np.int64)}
    fine = {"K": 2, "cell_class": cls.tolist(), "centroid": proto.tolist()}
    policy = {"cell_token": [0, 1], "token_class": cls.tolist(),
              "token_proto": proto.tolist(), "fine": fine}
    decoder = {"kind": "H-CLASS-TEMP", "target": "tokens", "T": 2, "K": 2,
               "q": p.tolist(), "token_class": cls.tolist(), "alphas": alpha.tolist()}
    binding = {name: sha_array(bank[field]) for field, name in
               (("probability", "bank_probability_sha256"),
                ("token_class", "token_class_sha256"),
                ("input_class", "input_class_sha256"),
                ("fine_to_hcal_token", "fine_to_hcal_token_sha256"))}
    return bank, policy, fine, decoder, binding


def test_pstar_per_class_temperature_matches_public_bank():
    result = check_pstar_bank(*_fixture())
    assert result["public_actions"] == 2
    assert result["max_temperature_difference"] < 1e-15


def test_pstar_temperature_and_id_collision_refused():
    bank, policy, fine, decoder, binding = _fixture()
    decoder["alphas"][0] = 1.0
    with pytest.raises(ValueError):
        check_pstar_bank(bank, policy, fine, decoder, binding)
    bank, policy, fine, decoder, binding = _fixture()
    bank["probability"][1] = bank["probability"][0]
    with pytest.raises(ValueError):
        check_pstar_bank(bank, policy, fine, decoder, binding)
