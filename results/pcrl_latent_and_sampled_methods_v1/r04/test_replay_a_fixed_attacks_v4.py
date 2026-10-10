"""Data-free fixed attack reader-family receipt regression."""

import pytest

from .replay_a_fixed_attacks_v4 import _expected_freeze_family


def test_finite_and_continuous_inherited_reader_families_are_distinct():
    assert _expected_freeze_family("finite") == "A"
    assert _expected_freeze_family("continuous") == "D"
    with pytest.raises(ValueError):
        _expected_freeze_family("latent")
