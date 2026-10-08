"""Synthetic S4 anchor replay checks with no Adult source or role files."""

from __future__ import annotations

import hashlib

import joblib
import numpy as np
import pytest
import torch

import replay_feature_anchor as anchor


class FixedHead:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=np.float64)

    def predict_proba(self, x):
        return np.tile(self.probabilities, (len(x), 1))


def _source_state():
    result = {}
    for r, width in ((0, 2), (1, 6)):
        for layer, own, output in ((0, 83, 64), (2, 64, 64), (4, 64, 16)):
            result[f"enc.{r}.{layer}.weight"] = torch.zeros((output, own))
            result[f"enc.{r}.{layer}.bias"] = torch.zeros(output)
        result[f"head.{r}.weight"] = torch.zeros((width, 16))
        result[f"head.{r}.bias"] = torch.zeros(width)
    return result


def test_pinned_u_source_forward_uses_model_and_both_public_heads(tmp_path):
    model = tmp_path / "model.pt"
    heads = (tmp_path / "head_0.joblib", tmp_path / "head_1.joblib")
    torch.save(_source_state(), model)
    p1 = np.array([0.25, 0.75])
    p2 = np.arange(1, 7, dtype=np.float64) / 21.0
    joblib.dump(FixedHead(p1), heads[0])
    joblib.dump(FixedHead(p2), heads[1])
    paths = {"model.pt": model, "head_0.joblib": heads[0],
             "head_1.joblib": heads[1]}
    digests = {name: anchor.sha_file(path) for name, path in paths.items()}
    x = np.zeros((2, 83), dtype=np.float32)
    refs = {"U0_1": np.tile(p1, (2, 1)), "U0_2": np.tile(p2, (2, 1))}
    assert anchor.source_forward(model, heads, digests, x, refs) == {
        "U0_1_max_abs_error": 0.0, "U0_2_max_abs_error": 0.0}
    wrong = dict(refs)
    wrong["U0_1"] = np.tile(p1[::-1], (2, 1))
    with pytest.raises(ValueError, match="U0 probability"):
        anchor.source_forward(model, heads, digests, x, wrong)


def test_raw83_and_public_score_wire_contains_every_recipient_field(monkeypatch):
    names = tuple(f"f{i}" for i in range(83))
    monkeypatch.setattr(anchor, "FEATURE_NAMES_SHA256",
                        hashlib.sha256("\n".join(names).encode()).hexdigest())
    x = np.arange(166, dtype=np.float32).reshape(2, 83)
    features = {"row_id": np.array([11, 22]), "unit": np.array([31, 32]),
                "X": x, "feature_names": np.asarray(names)}
    q1 = np.tile([0.2, 0.8], (2, 1))
    q2 = np.tile(np.arange(1, 7, dtype=np.float64) / 21.0, (2, 1))
    refs = {"row_id": features["row_id"], "unit": features["unit"],
            "U0_1": q1.copy(), "U0_2": q2.copy(),
            "Ucal_1": q1, "Ucal_2": q2,
            "d_1": np.ones(2, dtype=np.int64),
            "d_2": np.full(2, 5, dtype=np.int64)}
    views = anchor.build_views(features, refs, "a" * 64)
    assert {name: view["x"].shape for name, view in views.items()} == {
        "r1": (2, 87), "r2": (2, 95), "pair": (2, 182)}
    assert {name: view["x_defense_aware"].shape for name, view in views.items()} == {
        "r1": (2, 89), "r2": (2, 101), "pair": (2, 190)}
    for view in views.values():
        np.testing.assert_array_equal(view["x_raw_only"], x)
        assert len(view["wire_constants"]) in (4, 8)
        assert "r1.selected_values" in view["wire_components"] or \
            "r2.selected_values" in view["wire_components"]
    wrong = dict(refs, d_2=np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError, match="Ucal/U0/decision"):
        anchor.build_views(features, wrong, "a" * 64)
