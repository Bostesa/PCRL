"""Engineering tests for the confirmation runner and power code (synthetic only)."""
import json
import numpy as np
import pytest


def test_execute_fresh_is_refused(tmp_path):
    from tree_completion.confirmation_runner import main
    assert main(['--mode', 'execute-fresh', '--protocol', str(tmp_path / 'x.json'), '--protocol-sha256', '0' * 64]) == 2


def test_protocol_hash_mismatch_refused(tmp_path):
    from tree_completion.confirmation_runner import main
    p = tmp_path / 'p.json'; p.write_text('{}')
    assert main(['--mode', 'dry-run', '--protocol', str(p), '--protocol-sha256', '0' * 64]) == 2


def test_immutable_report_refuses_existing_dir(tmp_path):
    from tree_completion.confirmation_runner import Immutable
    r = Immutable(tmp_path / 'rep'); r.write_json('a.json', {'x': 1}); sha = r.close()
    assert len(sha) == 64
    with pytest.raises(PermissionError):
        Immutable(tmp_path / 'rep')
    with pytest.raises(FileExistsError):
        open(tmp_path / 'rep' / 'a.json', 'x')


def test_role_overlap_refused():
    from tree_completion.confirmation_runner import check_roles
    names = np.array([f'f{i}' for i in range(83)])
    a = {'feature_names': names, 'row_id': np.arange(10), 'unit': np.arange(10)}
    b = {'feature_names': names, 'row_id': np.arange(9, 20), 'unit': np.arange(9, 20)}
    with pytest.raises(PermissionError):
        check_roles(a, b)
    c = {'feature_names': names, 'row_id': np.arange(10, 20), 'unit': np.arange(10, 20)}
    assert check_roles(a, c)['overlap'] == 0


def test_schema_mismatch_refused():
    from tree_completion.confirmation_runner import check_roles
    a = {'feature_names': np.array(['a', 'b']), 'row_id': np.arange(3), 'unit': np.arange(3)}
    b = {'feature_names': np.array(['b', 'a']), 'row_id': np.arange(3, 6), 'unit': np.arange(3, 6)}
    with pytest.raises(PermissionError):
        check_roles(a, b)


def test_auc_influence_matches_reference_and_bootstrap():
    from tree_completion.feasibility_worker import auc_if
    from task_conditioned_representation.inference import auc
    rng = np.random.default_rng(0); y = (rng.random(3000) < 0.33).astype(int); s = rng.normal(size=3000) + 0.8 * y
    s[:50] = 0.25  # ties
    A, IF = auc_if(y, s)
    assert abs(A - auc(y, s)) < 1e-12 and abs(IF.mean()) < 1e-9
    se_if = IF.std(ddof=1) / np.sqrt(len(y))
    bs = [auc(y[i], s[i]) for i in [rng.integers(0, 3000, 3000) for _ in range(400)]]
    assert abs(se_if / np.std(bs, ddof=1) - 1) < 0.15


def test_power_gstar_matches_analytic_single_clause():
    from tree_completion.power import gstar
    from scipy import stats
    mu = np.array([0.004]); Sig = np.array([[0.16 ** 2]]); dirs = np.array([1.0]); bounds = np.array([0.005]); c = stats.norm.ppf(0.975)
    gs, slack, sd = gstar(mu, Sig, dirs, bounds, c, draws=200000)
    analytic = ((c + stats.norm.ppf(0.8)) * 0.16 / 0.001) ** 2
    assert abs(np.quantile(gs, 0.8) / analytic - 1) < 0.02


def test_power_boundary_not_zero_but_small():
    from tree_completion.power import gstar
    from scipy import stats
    mu = np.array([0.005]); Sig = np.array([[0.16 ** 2]]); c = stats.norm.ppf(0.975)
    gs, _, _ = gstar(mu, Sig, np.array([1.0]), np.array([0.005]), c, draws=100000, grid=[10000, 1000000])
    assert np.isinf(gs).all()
    assert abs(gstar.direct[1000000] - 0.025) < 0.003   # power stays at alpha at the boundary
