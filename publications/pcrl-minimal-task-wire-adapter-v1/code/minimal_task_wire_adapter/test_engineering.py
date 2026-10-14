"""Finite engineering checks for the residual adapter; synthetic inputs only; queued."""
import json
import random
import pytest


@pytest.fixture(autouse=True)
def queued():
    from .scheduler import assert_active_from_env
    assert_active_from_env(stage='admission')


def fixture(seed=913):
    import torch
    from task_conditioned_representation.useful_service import UsefulModel
    from .wires import bind_geometry, prepare_geometry
    torch.manual_seed(seed)
    model = UsefulModel([f'x{i}' for i in range(83)]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    g = prepare_geometry(bind_geometry(model, source_seed=0, checkpoint_sha256='a' * 64))
    return model, g


def test_zero_residual_equals_raw_proj_and_raw_path():
    import numpy as np
    import torch
    from .adapter import Adapter, forward_raw, hidden64, z_from_hidden
    from .wires import forward_role
    model, g = fixture()
    x = np.random.default_rng(0).normal(size=(2500, 83)).astype(np.float32)
    a = Adapter(model, g, 1.0)
    assert a.rank == 5 and a.n_parameters() == 325
    ours = forward_raw(a, x, model.names); ref = forward_role(model, x, model.names, g, variant='RAW_PROJ')
    assert all(np.array_equal(ours[i]['wire'], ref[i]['wire']) for i in range(2))
    xt = torch.from_numpy(x[:300])
    assert torch.equal(z_from_hidden(model, hidden64(model, xt)), model.encode(1, xt))
    hb, q0, inc = a.frozen_features(xt)
    assert torch.equal(a.wire(hb, q0)['full'], torch.from_numpy(ref[1]['wire'][:300]))


def test_schema_refusals_and_class_order():
    import numpy as np
    from .adapter import Adapter, forward_raw
    model, g = fixture()
    a = Adapter(model, g, 1.0)
    x = np.zeros((4, 83), np.float32)
    for bad in (lambda: forward_raw(a, x[:, ::-1], model.names[::-1]), lambda: forward_raw(a, x[:, :82], model.names[:82]),
                lambda: forward_raw(a, x.astype(np.int64), model.names), lambda: forward_raw(a, np.full((2, 83), np.nan, np.float32), model.names)):
        with pytest.raises(ValueError):
            bad()
    assert model.class_orders == [[0, 1], list(range(6))]
    with pytest.raises(ValueError):
        Adapter(model, g, 0.5)


def test_scale_never_scales_up():
    import torch
    from .adapter import fit_scale
    model, _ = fixture()
    s, rec = fit_scale(model, torch.zeros(10, 83))
    assert s == 1.0
    with torch.no_grad():
        model.enc[1][2].weight.mul_(50.0)
    s2, rec2 = fit_scale(model, torch.randn(500, 83))
    assert s2 > 1 and abs(s2 - rec2['rms_per_coordinate']) < 1e-12


def test_planted_residual_bypass_is_caught_by_wire_check():
    import torch
    from .adapter import Adapter, wire_contract_check
    from task_conditioned_representation.models import own_wire
    model, g = fixture()
    a = Adapter(model, g, 1.0)
    a.R = torch.randn(5, 64, dtype=torch.float64) * 1e-2
    x = torch.randn(200, 83)
    hb, q0, _ = a.frozen_features(x)
    good = a.wire(hb, q0)['full']
    assert wire_contract_check(a, hb, q0, good) == []
    z0 = model.encode(1, x).double()
    leaked = a.q(hb, q0) @ a.B + 1e-3 * (z0 - (z0 @ a.B.T) @ a.B)   # discarded residual added back
    bad = own_wire(leaked, a.head, a.order)['full']
    fails = wire_contract_check(a, hb, q0, bad)
    assert any('row space' in f for f in fails)
    teacher = good.clone(); teacher[:, 22:28] = own_wire(z0, a.head, a.order)['probabilities']  # unprotected teacher output bypass
    assert wire_contract_check(a, hb, q0, teacher)


def test_slack_projection_three_halfspaces_matches_reference_solver():
    import numpy as np
    from scipy.optimize import minimize
    from .guard import project_slack
    rng = np.random.default_rng(7)
    for trial in range(400):
        n = int(rng.integers(2, 8)); p = rng.normal(size=n) * rng.choice([1e-3, 1e-2, 5e-2])
        m = int(rng.integers(1, 4)); a = rng.normal(size=(m, n)); b = np.abs(rng.normal(size=m)) * rng.choice([0, 1e-4, 1e-2], size=m)
        if trial % 5 == 0:
            a[-1] = 0.0                                # zero KL gradient row
        d, info = project_slack(p, a, b, 0.01)
        assert (a @ d <= b + 1e-9).all() and np.linalg.norm(d) <= 0.01 + 1e-12 and not info['conservative_zero_fallback']
        cons = [{'type': 'ineq', 'fun': lambda x, j=j: b[j] - a[j] @ x} for j in range(m)] + [{'type': 'ineq', 'fun': lambda x: 1e-4 - x @ x}]
        ref = minimize(lambda x: .5 * np.sum((x - p) ** 2), np.zeros(n), jac=lambda x: x - p, constraints=cons,
                       method='SLSQP', options={'ftol': 1e-16, 'maxiter': 500})
        if (a @ ref.x <= b + 1e-9).all() and np.linalg.norm(ref.x) <= 0.01 + 1e-9:
            assert .5 * np.sum((d - p) ** 2) <= .5 * np.sum((ref.x - p) ** 2) + 1e-9


def test_degenerate_rows_duplicate_antiparallel_nearparallel_and_ball():
    import numpy as np
    from scipy.optimize import minimize
    from .guard import project_slack
    rng = np.random.default_rng(11)
    def ref(p, a, b):
        cons = [{'type': 'ineq', 'fun': lambda x, j=j: b[j] - a[j] @ x} for j in range(len(a))] + [{'type': 'ineq', 'fun': lambda x: 1e-4 - x @ x}]
        return minimize(lambda x: .5 * np.sum((x - p) ** 2), np.zeros(len(p)), jac=lambda x: x - p, constraints=cons, method='SLSQP', options={'ftol': 1e-16, 'maxiter': 1000}).x
    for kind in ('duplicate', 'antiparallel', 'nearparallel', 'two_active_with_ball'):
        for _ in range(50):
            n = 6; p = rng.normal(size=n) * 0.02; a0 = rng.normal(size=n)
            if kind == 'duplicate':
                a = np.vstack((a0, a0, rng.normal(size=n))); b = np.array([1e-3, 5e-4, 1e-3])
            elif kind == 'antiparallel':
                a = np.vstack((a0, -a0)); b = np.array([1e-3, 1e-3])
            elif kind == 'nearparallel':
                a = np.vstack((a0, a0 + 1e-9 * rng.normal(size=n), rng.normal(size=n))); b = np.array([1e-3, 2e-3, 1e-3])
            else:
                a = rng.normal(size=(2, n)); p = -(a[0] + a[1]) * 0.05; b = np.array([1e-4, 1e-4])
            d, info = project_slack(p, a, b, 0.01)
            assert (a @ d <= b + 1e-9).all() and np.linalg.norm(d) <= 0.01 + 1e-12
            x = ref(p, a, b)
            if (a @ x <= b + 1e-9).all() and np.linalg.norm(x) <= 0.01 + 1e-9:
                assert .5 * np.sum((d - p) ** 2) <= .5 * np.sum((x - p) ** 2) + 1e-8, (kind, info)


def test_zero_kl_gradient_contributes_no_direction_and_zero_feasible():
    import numpy as np
    from .guard import project_slack
    p = np.array([0.004, -0.003]); a = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    d, info = project_slack(p, a, np.array([0.001, 0.0, 0.0005]), 0.01)
    assert info['zero_gradient_rows'] == 1 and np.allclose(d, [0.001, -0.003])
    noisy = a.copy(); noisy[2] = np.array([3e-20, -7e-20])      # round-off KL gradient, positive slack
    dn, _ = project_slack(p, noisy, np.array([0.001, 0.0, 0.0005]), 0.01)
    assert np.allclose(dn, d, atol=1e-15, rtol=0)
    d0, _ = project_slack(p, a, np.zeros(3), 0.01)
    assert np.all(a @ d0 <= 1e-12)
    with pytest.raises(ValueError):
        project_slack(p, a, np.array([-1e-6, 0, 0]), 0.01)


def _pool(seed=3, n=2300):
    import torch
    torch.manual_seed(seed)
    return torch.randn(n, 83), torch.randint(0, 6, (n,))


def test_full_pool_chunks_equal_single_pass_and_kl_zero_at_start():
    import torch
    from . import guard
    from .adapter import Adapter
    model, g = fixture(); a = Adapter(model, g, 1.0)
    x, y = _pool(); hb, q0, _ = a.frozen_features(x)
    lp0 = torch.log_softmax(a.wire(hb, q0)['centered_logits'], 1)
    c = guard.pool_pass(a, hb, q0, y, lp0, params=True)
    # Analytically zero at the start; float64 round-off leaves ~1e-19 entries.
    assert c['kl'] == 0.0 and abs(c['gradients']['kl']).max() <= 1e-15 and c['rows'] == 2300
    old = guard.CHUNK; guard.CHUNK = 10 ** 9
    try:
        s = guard.pool_pass(a, hb, q0, y, lp0, params=True)
    finally:
        guard.CHUNK = old
    for k in ('nll', 'brier', 'ce'):
        assert abs(c[k] - s[k]) <= 1e-12 * max(1, abs(s[k]))
        assert abs(c['gradients'][k] - s['gradients'][k]).max() <= 1e-12 * max(1, abs(s['gradients'][k]).max())
    a.R = torch.randn(5, 64, dtype=torch.float64) * 0.05
    m = guard.pool_pass(a, hb, q0, y, lp0, params=True)
    assert m['kl'] > 0 and abs(m['gradients']['kl']).max() > 0
    # Finite-difference check of the analytic KL/NLL gradients w.r.t. b_res.
    eps = 1e-6; e = torch.zeros(5, dtype=torch.float64); e[2] = eps
    a.b = a.b + e; up = guard.pool_pass(a, hb, q0, y, lp0); a.b = a.b - 2 * e; dn = guard.pool_pass(a, hb, q0, y, lp0); a.b = a.b + e
    for k in ('kl', 'ce', 'brier'):
        fd = (up[k] - dn[k]) / (2 * eps)
        assert abs(fd - m['gradients'][k][320 + 2]) <= 1e-6 * max(1, abs(fd))


def test_anchored_budgets_never_reset_and_kl_only_for_trust():
    from .guard import Anchor
    start = {'nll': 1.0, 'brier': 0.5, 'correct': 1000, 'rows': 10223, 'accuracy': .1}
    t = Anchor(start, trust=True); f = Anchor(start, trust=False)
    assert t.nll_limit == 1.004 and t.brier_limit == 0.502 and t.correct_floor == 1000 - 51
    m = {'nll': 1.003, 'brier': 0.501, 'correct': 960, 'kl': 0.002}
    assert not t.within(m) and f.within(m)
    assert len(t.slacks(m)) == 3 and len(f.slacks(m)) == 2


def test_guarded_step_backtracks_rolls_back_and_respects_norm():
    import numpy as np
    import torch
    from . import guard
    from .adapter import Adapter
    model, g = fixture(); a = Adapter(model, g, 1.0)
    x, y = _pool(); hb, q0, _ = a.frozen_features(x)
    lp0 = torch.log_softmax(a.wire(hb, q0)['centered_logits'], 1)
    start = guard.pool_pass(a, hb, q0, y, lp0, params=True)
    anchor = guard.Anchor(start, trust=True)
    before = guard.flat([a.R, a.b])
    huge = -start['gradients']['ce'] * 1e4               # projected to the norm ball
    r = guard.guarded_step(a, huge, start, anchor, lambda: guard.pool_pass(a, hb, q0, y, lp0), 0.01)
    after = guard.flat([a.R, a.b])
    assert np.linalg.norm(after - before) <= 0.01 + 1e-15
    if r['status'] == 'ACCEPTED':
        post = guard.pool_pass(a, hb, q0, y, lp0)
        assert anchor.within(post)
    # Impossible budget: everything rejected and exact rollback.
    a.R = torch.zeros(5, 64, dtype=torch.float64); a.b = torch.zeros(5, dtype=torch.float64)
    tight = guard.Anchor(start, trust=True); tight.kl_limit = 0.0
    tight.nll_limit = start['nll'] - 1.0
    r2 = guard.guarded_step(a, huge, start, tight, lambda: guard.pool_pass(a, hb, q0, y, lp0), 0.01)
    assert r2['status'] == 'INFEASIBLE_CURRENT_STATE' and np.array_equal(guard.flat([a.R, a.b]), before)
    tight2 = guard.Anchor(start, trust=True); tight2.kl_limit = 1e-14
    r3 = guard.guarded_step(a, huge, start, tight2, lambda: guard.pool_pass(a, hb, q0, y, lp0), 0.01)
    assert r3['status'] == 'REJECTED_ZERO_STEP' and np.array_equal(guard.flat([a.R, a.b]), before)
    assert [x['fraction'] for x in r3['attempts']] == [1.0, .5, .25, .125]


def test_trainer_refuses_heldout_exports():
    from .train_worker import load_export
    for path in ('/x/.private/roles/VALIDATION/features.npz', '/x/ATTACK_FIT_NEW/features.npz', '/x/INNER_SELECTION/features.npz'):
        with pytest.raises(PermissionError):
            load_export({'path': path, 'sha256': '0' * 64}, ('X',), lambda p, s: p)


def test_task_weights_population():
    from .train_worker import Task
    t = Task(8180, 2043)
    assert abs(t.w_update - 8180 / 10223) < 1e-15 and abs(t.w_guard - 2043 / 10223) < 1e-15


def test_frozen_source_and_income_unchanged_by_arm_and_bank_end_to_end(tmp_path):
    """Every arm runs real proposals, critics, checkpoints; source tensors untouched; J_TRUST counterfactual scan."""
    import numpy as np
    import torch
    from . import train_worker as T
    from .adapter import Adapter, tensor_sha
    from .guard import pool_pass
    from .recipe import ARMS, RECIPE
    model, g = fixture()
    before = {k: tensor_sha(v) for k, v in model.state_dict().items()}
    torch.manual_seed(5)
    xU = torch.randn(600, 83); yU = torch.randint(0, 6, (600,)); sU = torch.randint(0, 2, (600,))
    xG = torch.randn(150, 83); yG = torch.randint(0, 6, (150,))
    a0 = Adapter(model, g, 1.0)
    hU, qU, iU = a0.frozen_features(xU); hG, qG, _ = a0.frozen_features(xG)
    with torch.no_grad():
        lU = torch.log_softmax(a0.wire(hU, qU)['centered_logits'], 1); lG = torch.log_softmax(a0.wire(hG, qG)['centered_logits'], 1)
    hP, qP, yP, lP = torch.cat((hU, hG)), torch.cat((qU, qG)), torch.cat((yU, yG)), torch.cat((lU, lG))
    ctx = dict(seed=0, geometry=g, scale=1.0, task=T.Task(600, 150), source_model=model, hU=hU, qU=qU, iU=iU, yU=yU, sU=sU,
               hG=hG, qG=qG, yG=yG, hP=hP, qP=qP, yP=yP, lp0P=lP, lp0U=lU, lp0G=lG,
               reference_rows=T._draw(random.Random(1), 600), probe_h=hU[:64], probe_q=qU[:64], out=tmp_path,
               start_pool=pool_pass(a0, hP, qP, yP, lP),
               wire_tolerance={'representation': 1e-10, 'centered_logits': 1e-10, 'probabilities': 1e-10},
               income_guard=lambda: all(tensor_sha(v) == before[k] for k, v in model.state_dict().items()),
               checkpoint_sha256='a' * 64, adapter_binding_sha256='b' * 64)
    norm = T.normalization(ctx, tmp_path)
    assert norm['coefficients']['joint']['status'] == 'FIXED' and norm['coefficients']['local']['status'] == 'FIXED'
    ctx['coefficients'] = {'J_TRUST': norm['coefficients']['joint']['coefficient'], 'J_FREE': norm['coefficients']['joint']['coefficient'],
                           'L_TRUST': norm['coefficients']['local']['coefficient'], 'L_FREE': norm['coefficients']['local']['coefficient'], 'TASK_TRUST': None}
    ctx['common'] = {'source_seed': 0}
    old = dict(RECIPE)
    RECIPE['epochs'] = 2; RECIPE['proposals_per_epoch'] = 3; RECIPE['checkpoint_epochs'] = [0, 1, 2]
    try:
        out = {arm: T.run_arm(arm, ctx) for arm in ARMS}
    finally:
        RECIPE.clear(); RECIPE.update(old)
    assert len({s['warm_critic_state_sha256'] for s in out.values()}) == 1
    assert all(s['model_updates'] == 6 and s['critic_optimizer_steps'] == (80 + 5 * 6) * 6 for s in out.values())
    assert all(len(s['checkpoints']) == 3 for s in out.values())
    assert all(tensor_sha(v) == before[k] for k, v in model.state_dict().items())
    js = out['J_TRUST']['counterfactual_scan']
    assert all('privacy_minus_counterfactual_max_abs' in r for r in js)
    for arm in ('J_TRUST', 'L_TRUST', 'TASK_TRUST'):
        rows = json.loads((tmp_path / arm / 'BUDGET_TRACE.json').read_text())
        assert all(r['post_kl'] is None or r['post_kl'] <= 0.001 + 1e-10 for r in rows)
        assert len({r['nll_limit'] for r in rows}) == 1
    for arm in ARMS:
        rows = json.loads((tmp_path / arm / 'BUDGET_TRACE.json').read_text())
        assert all(r['post_nll'] is None or r['post_nll'] <= r['nll_limit'] + 1e-10 for r in rows)
