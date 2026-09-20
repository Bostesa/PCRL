"""The constant-channel diagnostic for the training protection penalty.

`extension.role_gains` reports `CE(p0J) - CE(p0J + correction)` where the correction network reads
`[H view, Z_J, R]`. The frozen `p0J` baseline is a finite-capacity MLP fitted on one fold, so a
correction on the SAME inputs has headroom even when `R` carries nothing. These fixtures demonstrate
that the reported gain is therefore not zero under the null, and pin the diagnostic that measures
the null level. No data is needed.
"""
import torch

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_stochastic_channel_v1 import diagnostics as diag


def test_constant_extension_still_produces_a_positive_reported_gain():
    """The defect, executed: R is exactly constant, so I(S;R | H, Z_J) = 0, yet the gain is > 0."""
    out = diag.constant_channel_gain(seed=0, steps=120, r_width=2, rows=512)
    assert out['extension_is_constant'] is True
    sensitive = [out['gains'][role] for role in ('A/SEX', 'A/RAC1P', 'AB/SEX', 'AB/RAC1P')]
    assert max(sensitive) > 1e-3, out['gains']
    # The extension receives no gradient signal it could exploit: its column variance is zero.
    assert out['extension_std'] == [0.0] * 2


def test_the_gain_floor_makes_the_statistic_one_sided():
    """`clamp(min=0)` on the argmax means baseline slack can only push the reported gain up."""
    out = diag.constant_channel_gain(seed=1, steps=120, r_width=2, rows=512)
    assert all(v >= 0.0 for v in out['gains'].values())
    assert out['clamped_at_zero_roles'] + out['positive_roles'] == len(dax.ROLE_ORDER)


def test_gain_is_zero_before_the_first_optimizer_step():
    """METHOD section 3's 'zero at initialisation by construction' is true, and only at init."""
    out = diag.constant_channel_gain(seed=0, steps=0, r_width=2, rows=256)
    assert all(abs(v) < 1e-9 for v in out['gains'].values()), out['gains']


def test_null_correction_removes_the_baseline_slack_on_a_constant_channel():
    """Subtracting the matched constant-channel run leaves ~0 when the channel really is constant."""
    out = diag.null_corrected_gain(seed=0, steps=120, r_width=2, rows=512, informative=False)
    for role, v in out['corrected'].items():
        assert abs(v) < 1e-6, (role, v)


def test_null_correction_keeps_a_real_signal():
    """An extension that genuinely carries the label must survive the same correction."""
    out = diag.null_corrected_gain(seed=0, steps=120, r_width=2, rows=512, informative=True)
    assert out['corrected']['A/SEX'] > 1e-2, out['corrected']
    assert out['raw']['A/SEX'] > out['corrected']['A/SEX']   # slack was actually subtracted


def test_null_level_decays_as_the_frozen_baseline_is_better_fitted():
    """Scoping the finding honestly.

    The inflation is baseline-approximation slack, so it shrinks as `p0J` approaches its own
    optimum. A deliberately under-fitted baseline shows the mechanism at ~1 nat; a well-fitted one
    leaves a residue near the screen's own scale. The real study fits `p0J` for `p0_epochs = 60`
    minibatch epochs, which is far more optimiser steps than the top of this ladder, so the
    magnitude here must NOT be read as the magnitude in the study -- only the mechanism transfers.
    """
    ladder = [max(diag.constant_channel_gain(seed=0, steps=120, baseline_steps=bs)['gains'].values())
              for bs in (10, 200, 800)]
    assert ladder[0] > ladder[1] > ladder[2]
    assert ladder[0] > 0.5            # under-fitted baseline: the mechanism is unmistakable
    assert ladder[2] < 0.05           # well-fitted baseline: a small residue, not zero


def test_diagnostic_uses_the_shipped_role_gains_not_a_reimplementation():
    """Guard: the fixture must exercise experiments.pcrl_utility_extension_v1.extension.role_gains."""
    from experiments.pcrl_utility_extension_v1 import extension as ext
    assert diag.ROLE_GAINS is ext.role_gains


def test_constant_extension_detector_matches_the_pilot_field():
    """`fit_record['extension_constant']` is std == 0 on representation_fit; same rule here."""
    assert diag.is_constant(torch.zeros(10, 3)) is True
    assert diag.is_constant(torch.ones(10, 3) * 4.5) is True     # constant, not zero
    x = torch.zeros(10, 3)
    x[0, 1] = 1e-9
    assert diag.is_constant(x) is False
