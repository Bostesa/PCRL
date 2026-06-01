"""Tests for run_eval_multi.py lora_target detection + checkpoint loading.

Yesterday's AB run crashed in eval because run_eval_multi.load_encoder()
always rebuilt the encoder with the default ``lora_target="all_linear"``,
while training used ``--lora-target=repr_proj_only``. The state_dict
shapes mismatched and load_state_dict raised RuntimeError.

These tests verify:
  1. ``_detect_lora_target`` returns ``"all_linear"`` for a state_dict that
     has adapter indices ``.1.`` or ``.2.`` per purpose (3 adapters
     per purpose — one per Linear in the backbone).
  2. Returns ``"repr_proj_only"`` when only index ``.0.`` is present
     (one adapter per purpose — only the final repr_proj).
  3. End-to-end: ``load_encoder`` loads a real repr_proj_only checkpoint
     from the 2026-05-19 CPU smoke without raising. This is the integration
     check — on the unpatched code this raises the exact error we saw
     in production yesterday (size mismatch + 18 missing keys).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.crosspurp import run_eval_multi as rem  # noqa: E402


def _t(*shape):
    return torch.zeros(*shape)


def test_detect_all_linear_keys_with_dot1_and_dot2():
    sd = {
        "0.0.bias": _t(64), "0.0.A.weight": _t(8, 128), "0.0.B.weight": _t(64, 8),
        "0.1.bias": _t(128), "0.1.A.weight": _t(8, 128), "0.1.B.weight": _t(128, 8),
        "0.2.bias": _t(128), "0.2.A.weight": _t(8, 105), "0.2.B.weight": _t(128, 8),
        "1.0.bias": _t(64), "1.0.A.weight": _t(8, 128), "1.0.B.weight": _t(64, 8),
        "1.1.bias": _t(128), "1.1.A.weight": _t(8, 128), "1.1.B.weight": _t(128, 8),
    }
    assert rem._detect_lora_target(sd) == "all_linear"


def test_detect_repr_proj_only_when_only_index_zero():
    sd = {
        "0.0.bias": _t(64), "0.0.A.weight": _t(8, 128), "0.0.B.weight": _t(64, 8),
        "1.0.bias": _t(64), "1.0.A.weight": _t(8, 128), "1.0.B.weight": _t(64, 8),
        "2.0.bias": _t(64), "2.0.A.weight": _t(8, 128), "2.0.B.weight": _t(64, 8),
    }
    assert rem._detect_lora_target(sd) == "repr_proj_only"


def test_detect_handles_single_purpose_repr_proj_only():
    sd = {
        "0.0.bias": _t(64), "0.0.A.weight": _t(8, 128), "0.0.B.weight": _t(64, 8),
    }
    assert rem._detect_lora_target(sd) == "repr_proj_only"


def test_detect_index_10_does_not_false_positive_as_all_linear():
    # purpose_idx=1, linear_idx=0 → key "1.0.bias" — the regex anchors on
    # linear_idx ∈ {1, 2}, so purpose_idx digits like "10." or "1." that
    # are followed by ".0." must NOT trigger all_linear.
    sd = {
        "1.0.A.weight": _t(8, 128), "10.0.A.weight": _t(8, 128),
    }
    assert rem._detect_lora_target(sd) == "repr_proj_only"


_HMDA_PROCESSED = REPO_ROOT / "data" / "hmda_processed" / "train.npz"
_DIABETES_PROCESSED = REPO_ROOT / "data" / "diabetes_processed" / "train.npz"


@pytest.mark.skipif(not _HMDA_PROCESSED.exists(), reason="HMDA preprocessed npz absent")
def test_build_loaders_hmda_does_not_propagate_norm_stats():
    """HMDADataset bakes normalization into the preprocessed npz and does
    NOT expose ``.norm_stats``. Yesterday's eval crashed on
    ``train_ds.norm_stats``. After the fix, build_loaders must skip the
    propagation for HMDA."""
    purposes, train_ds, test_ds, _, _, cfg = rem.build_loaders("hmda")
    assert not hasattr(train_ds, "norm_stats") or train_ds.norm_stats == {}, \
        "HMDADataset is expected to have no norm_stats; if it does now, " \
        "update the DATASET_CONFIG flag"
    assert len(test_ds) > 0


@pytest.mark.skipif(not _DIABETES_PROCESSED.exists(), reason="Diabetes preprocessed npz absent")
def test_build_loaders_diabetes_data_root_points_at_processed():
    """DiabetesDataset looks at ``<root>/<split>.npz`` directly, so the
    DATASET_CONFIG ``data_root`` must be ``data/diabetes_processed``.
    Yesterday's eval crashed because ``data_root`` was ``data`` and the
    dataset looked for ``data/train.npz`` instead of
    ``data/diabetes_processed/train.npz``."""
    purposes, train_ds, test_ds, _, _, cfg = rem.build_loaders("diabetes")
    assert cfg["data_root"] == "data/diabetes_processed"
    assert len(test_ds) > 0


@pytest.mark.skipif(
    not (REPO_ROOT / "checkpoints/v2_adult_CROSSPURP_ERASE_SMOKE_s0/best.pt").exists(),
    reason="Requires the 2026-05-19 repr_proj_only smoke checkpoint",
)
def test_load_encoder_round_trip_against_real_repr_proj_only_ckpt():
    """End-to-end: load a real repr_proj_only checkpoint via the patched
    load_encoder. On unpatched code this raises RuntimeError with size
    mismatch + missing keys (the exact error from the AB eval failure)."""
    ckpt_path = REPO_ROOT / "checkpoints/v2_adult_CROSSPURP_ERASE_SMOKE_s0/best.pt"
    # Adult dims: input_dim=105, n_purposes=3, lora_rank=8, alpha=16.0
    cfg = {"lora_rank": 8, "lora_alpha": 16.0}
    encoder = rem.load_encoder(ckpt_path, input_dim=105, n_purposes=3, cfg=cfg)
    # The encoder constructed against a repr_proj_only checkpoint must
    # have one LoRA adapter per purpose (index 0 only).
    for p_idx in range(3):
        adapters = encoder.adapters[p_idx]
        assert len(adapters) == 1, (
            f"purpose {p_idx} should have exactly 1 adapter (repr_proj_only), "
            f"got {len(adapters)}"
        )
    # Smoke a forward to confirm shape compatibility end-to-end.
    x = torch.zeros(2, 105)
    with torch.no_grad():
        h = encoder(x, 0)
    assert h.shape == (2, 64), f"expected output (2,64), got {h.shape}"
