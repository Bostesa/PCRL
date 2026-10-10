"""Data-free fixed Q custody inventory regression."""

import pytest

from .replay_a_fixed_attacks_v3 import _fixed_external_q


def _row(name, digest):
    return {"name": name, "custody_file": f"model__{name}.bin",
            "model_sha256": digest, "new_fit": True,
            "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN"}


def test_fixed_finite_q_is_one_of_three_external_readers():
    digest = "a" * 64
    rows = [_row("PROB_LR_C1", "b" * 64),
            _row("PROB_HGB_15x200", "c" * 64),
            _row("Q_POST_LR_C1", digest)]
    assert _fixed_external_q(rows, digest)["custody_file"] == "model__Q_POST_LR_C1.bin"


def test_missing_or_duplicate_q_is_rejected():
    digest = "a" * 64
    rows = [_row("PROB_LR_C1", "b" * 64),
            _row("PROB_HGB_15x200", "c" * 64),
            _row("Q_POST_LR_C1", digest)]
    for wrong in (rows[:2], rows + [rows[-1]], rows[:2] + [_row("Q_POST_LR_C1", "d" * 64)]):
        with pytest.raises(ValueError):
            _fixed_external_q(wrong, digest)
