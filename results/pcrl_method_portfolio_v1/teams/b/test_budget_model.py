"""Data-free checks that all mandatory Team B costs enter the budget."""

import pytest

from budget_model import estimate, required_smoke_keys


def test_missing_smoke_unit_blocks_estimate():
    with pytest.raises(ValueError, match="Missing required"):
        estimate({}, steps_per_encoder=10, finalist=False)


def test_known_unit_times_scale_to_exact_screen_counts():
    smoke = {key: 1.0 for key in required_smoke_keys(10)}
    result = estimate(smoke, steps_per_encoder=10, finalist=False)
    assert result["unit_counts"]["training_process_jobs"] == 22
    assert result["unit_counts"]["fresh_attacker_model_fits"] == 192
    assert result["defense_cpu_s"] == 22
    assert result["attack_cpu_s"] == 192
    assert result["calibration_cpu_s"] == 32
    assert result["forward_cpu_s"] == 72
    assert result["control_cpu_s"] == 144
    assert result["total_cpu_s"] == 462
