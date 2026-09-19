"""Fixture tests for the extension mechanism (no data needed)."""
import numpy as np
import torch

from experiments.pcrl_utility_extension_v1 import extension as ext
from experiments.pcrl_utility_extension_v1 import program as pg


def test_step_zero_is_zero_extension_and_d1_equals_d0():
    m = ext.Extension(2, 64, 123)
    x = torch.randn(50, 32)
    r = m.encode(x)
    assert torch.count_nonzero(r) == 0                      # constant zero extension at step 0
    dec_in = torch.randn(50, ext.HA + ext.Z_WIDTH)
    assert torch.count_nonzero(m.decode(dec_in, r)) == 0    # D1 = D0 + 0 exactly


def test_extension_receives_gradient_after_first_decoder_step():
    torch.manual_seed(0)
    m = ext.Extension(2, 16, 7)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x, d, t = torch.randn(64, 32), torch.randn(64, 20), torch.randn(64, 16)
    for _ in range(3):
        opt.zero_grad()
        ext.recon_loss(m.decode(d, m.encode(x)), t, 1.0).backward()
        opt.step()
    assert torch.count_nonzero(m.encode(x)) > 0


def test_inclusion_fact_on_finite_least_squares():
    """Adding a column can never raise the in-sample optimum of least squares (fixture of the
    Bayes-risk inclusion argument; finite held-out learners may still move either way)."""
    rng = np.random.default_rng(0)
    v0, r = rng.normal(size=(200, 3)), rng.normal(size=(200, 1))
    y = v0 @ [1., -2., .5] + r[:, 0] * .3 + rng.normal(size=200)
    def rss(x):
        x = np.column_stack([np.ones(len(x)), x])
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        return float(((y - x @ b) ** 2).sum())
    assert rss(np.column_stack([v0, r])) <= rss(v0) + 1e-9


def test_unit_names_and_counts():
    assert len(pg.units(pg.PILOT_R)) * 3 == 42
    assert len(pg.units(pg.FULL_R, leace=True)) * 3 == 135
    assert pg.unit_spec('X_r4_C1_b030') == {'kind': 'protected', 'r': 4, 'policy': 'C1', 'beta': 30.0}
    assert pg.unit_spec('P_r8')['kind'] == 'pca'


def test_selection_rule_tiebreak():
    rows = {'X_r4_C1_b003': {'config': 'X_r4_C1_b003', 'pass': True, 'recon_mean': .2},
            'X_r2_C1_b010': {'config': 'X_r2_C1_b010', 'pass': True, 'recon_mean': .2},
            'X_r8_C1_b001': {'config': 'X_r8_C1_b001', 'pass': True, 'recon_mean': .3},
            'X_r2_L2_b001': {'config': 'X_r2_L2_b001', 'pass': True, 'recon_mean': .9},
            'X_r2_C1_b001': {'config': 'X_r2_C1_b001', 'pass': False, 'recon_mean': .9}}
    assert pg.select_tier3(rows) == ['X_r8_C1_b001', 'X_r2_C1_b010']
