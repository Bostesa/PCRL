"""Meaningful mixed-dtype boundary fixtures; ROOT queue execution only."""
import numpy as np
import pytest
from occupation_bottleneck.gradient_worker import finite_step, parameter_vector


@pytest.fixture(autouse=True)
def require_engineering_queue():
    from occupation_bottleneck.scheduler import assert_active_from_env
    assert_active_from_env(stage='admission')


@pytest.mark.parametrize('guarded', [False, True])
def test_float32_boundary_repaired_without_zeroing_a_valid_step(guarded):
    import torch
    # .125 is exactly representable. .125+.01 rounds upward to
    # .13500000536441803, so the uncorrected displacement exceeds .01+1e-10.
    encoder = torch.nn.Parameter(torch.tensor([.125], dtype=torch.float32))
    head = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float64))
    parameters = [encoder, head]; before = parameter_vector(parameters)
    result = finite_step(parameters, np.array([.02, 0.]), np.zeros((2, 2)),
                         lambda: {'nll': 0., 'brier': 0.}, .01, guarded=guarded)
    after = parameter_vector(parameters); actual = after - before
    assert result['status'] == 'ACCEPTED'
    assert 0 < np.linalg.norm(actual) <= .01
    assert result['attempts'][0]['norm_repair_attempts'][0]['bounded'] is False
    assert len(result['attempts'][0]['norm_repair_attempts']) > 1
    assert 0 < result['attempts'][0]['norm_repair_scale'] < 1
    assert result['actual_norm_repair']['task_metrics_used'] is False
    assert after[1] == before[1]


def test_ordinary_norm_repair_preserves_safeguard_off():
    import torch
    encoder = torch.nn.Parameter(torch.tensor([.125], dtype=torch.float32))
    def increasing_task():
        return {'nll': float(encoder.sum()), 'brier': float(encoder.sum())}
    result = finite_step([encoder], np.array([.02]), np.zeros((2, 1)),
                         increasing_task, .01, guarded=False)
    assert result['status'] == 'ACCEPTED'
    assert result['attempts'][0]['finite_step_check'] is False
    assert result['actual_displacement_norm'] <= .01
    assert float(encoder.sum()) > .125


@pytest.mark.parametrize('guarded', [False, True])
def test_full_parameter_count_mixed_dtype_clipped_step_stays_nonzero(guarded):
    import torch
    # Source architecture: 10576 encoder parameters float32; 102 occupation
    # head parameters float64. A valid clipped model direction must not be
    # blanket-stalled by accumulation of encoder rounding error.
    encoder = torch.nn.Parameter(torch.full((10576,), .125, dtype=torch.float32))
    head = torch.nn.Parameter(torch.full((102,), .125, dtype=torch.float64))
    parameters = [encoder, head]; before = parameter_vector(parameters)
    direction = np.linspace(.5, 1.5, len(before)); direction /= np.linalg.norm(direction)
    result = finite_step(parameters, .02 * direction, np.zeros((2, len(before))),
                         lambda: {'nll': 0., 'brier': 0.}, .01, guarded=guarded)
    actual = parameter_vector(parameters) - before
    assert result['status'] == 'ACCEPTED'
    assert 0 < np.linalg.norm(actual) <= .01
    assert float(actual @ direction) > 0
    assert result['actual_displacement_norm'] == float(np.linalg.norm(actual))


def test_constrained_norm_repair_still_restores_task_unsafe_steps():
    import torch
    encoder = torch.nn.Parameter(torch.tensor([.125], dtype=torch.float32))
    before = parameter_vector([encoder])
    def increasing_task():
        return {'nll': float(encoder.sum()), 'brier': float(encoder.sum())}
    result = finite_step([encoder], np.array([.02]), np.zeros((2, 1)),
                         increasing_task, .01, guarded=True)
    assert result['status'] == 'REJECTED_ZERO_STEP'
    np.testing.assert_array_equal(parameter_vector([encoder]), before)
    assert all(attempt['finite_step_check'] is False for attempt in result['attempts'])


def test_norm_repair_logs_real_rounding_zero_without_fake_acceptance():
    import torch
    encoder = torch.nn.Parameter(torch.tensor([1e8], dtype=torch.float32))
    before = parameter_vector([encoder])
    result = finite_step([encoder], np.array([.02]), np.zeros((2, 1)),
                         lambda: {'nll': 0., 'brier': 0.}, .01, guarded=False)
    assert result['status'] == 'ROUNDING_ZERO_STEP'
    assert result['actual_displacement_norm'] == 0.
    np.testing.assert_array_equal(parameter_vector([encoder]), before)
