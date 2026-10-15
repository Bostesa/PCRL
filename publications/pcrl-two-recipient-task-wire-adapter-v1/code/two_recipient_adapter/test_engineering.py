"""Finite engineering checks for the two-recipient adapters; synthetic inputs only; queued."""
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


def test_zero_residual_equals_raw_proj_both_recipients_and_ranks():
    import numpy as np
    import torch
    from .adapter import RecipientAdapter, forward_raw, hidden64, z_from_hidden, head_record
    from .wires import forward_role
    model, g = fixture()
    x = np.random.default_rng(0).normal(size=(2500, 83)).astype(np.float32)
    ads = [RecipientAdapter(model, g, i, 1.0) for i in (0, 1)]
    assert [a.rank for a in ads] == [1, 5] and [a.n_parameters() for a in ads] == [65, 325]
    ours = forward_raw(model, g, [1.0, 1.0], [None, None], x, model.names); ref = forward_role(model, x, model.names, g, variant='RAW_PROJ')
    assert all(np.array_equal(ours[i]['wire'], ref[i]['wire']) for i in range(2))
    xt = torch.from_numpy(x[:300])
    for i, a in enumerate(ads):
        assert torch.equal(z_from_hidden(model, i, hidden64(model, i, xt)), model.encode(i, xt))
        hb, q0 = a.frozen_features(xt)
        assert torch.equal(a.wire(hb, q0)['full'], torch.from_numpy(ref[i]['wire'][:300]))
        h = head_record(model, g, i)
        assert h['A_Bt_B_minus_A_max_abs'] <= 1e-10 and h['measured_centered_rank'] == h['basis_rank']


def test_schema_refusals_and_class_order():
    import numpy as np
    from .adapter import RecipientAdapter, forward_raw
    model, g = fixture()
    x = np.zeros((4, 83), np.float32)
    f = lambda xx, nn: forward_raw(model, g, [1.0, 1.0], [None, None], xx, nn)
    for bad in (lambda: f(x[:, ::-1], model.names[::-1]), lambda: f(x[:, :82], model.names[:82]),
                lambda: f(x.astype(np.int64), model.names), lambda: f(np.full((2, 83), np.nan, np.float32), model.names)):
        with pytest.raises(ValueError):
            bad()
    assert model.class_orders == [[0, 1], list(range(6))]
    with pytest.raises(ValueError):
        RecipientAdapter(model, g, 0, 0.5)
    with pytest.raises(ValueError):
        RecipientAdapter(model, g, 2, 1.0)


def test_scale_never_scales_up_per_recipient():
    import torch
    from .adapter import fit_scale
    model, _ = fixture()
    for i in (0, 1):
        s, rec = fit_scale(model, i, torch.zeros(10, 83))
        assert s == 1.0
    with torch.no_grad():
        model.enc[0][2].weight.mul_(50.0)
    s0, rec0 = fit_scale(model, 0, torch.randn(500, 83)); s1, _ = fit_scale(model, 1, torch.zeros(10, 83))
    assert s0 > 1 and abs(s0 - rec0['rms_per_coordinate']) < 1e-12 and s1 == 1.0


def test_planted_residual_bypass_is_caught_for_both_recipients():
    import torch
    from .adapter import RecipientAdapter, wire_contract_check
    from task_conditioned_representation.models import own_wire
    model, g = fixture()
    x = torch.randn(200, 83)
    for i in (0, 1):
        a = RecipientAdapter(model, g, i, 1.0)
        a.R = torch.randn(a.rank, 64, dtype=torch.float64) * 1e-2
        hb, q0 = a.frozen_features(x)
        good = a.wire(hb, q0)['full']
        assert wire_contract_check(a, hb, q0, good) == []
        z0 = model.encode(i, x).double()
        leaked = a.q(hb, q0) @ a.B + 1e-3 * (z0 - (z0 @ a.B.T) @ a.B)
        fails = wire_contract_check(a, hb, q0, own_wire(leaked, a.head, a.order)['full'])
        assert any('row space' in f for f in fails)
        teacher = good.clone(); K = a.K; teacher[:, 16 + K:16 + 2 * K] = own_wire(z0, a.head, a.order)['probabilities']
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



def test_anchor_limits_and_kl_only_for_trust():
    from .guard import Anchor
    start = {'nll': 1.0, 'brier': 0.5, 'correct': 1000, 'rows': 10223, 'accuracy': .1}
    t = Anchor(start, trust=True); f = Anchor(start, trust=False)
    assert t.nll_limit == 1.004 and t.brier_limit == 0.502 and t.correct_floor == 1000 - 51
    m = {'nll': 1.003, 'brier': 0.501, 'correct': 960, 'kl': 0.002}
    assert not t.within(m) and f.within(m)
    assert len(t.slacks(m)) == 3 and len(f.slacks(m)) == 2



def _pool(seed=3, n=2300):
    import torch
    torch.manual_seed(seed)
    return torch.randn(n, 83), [torch.randint(0, 2, (n,)), torch.randint(0, 6, (n,))]


def test_full_pool_chunks_equal_single_pass_and_kl_zero_at_start_both_recipients():
    import torch
    from . import guard
    from .adapter import RecipientAdapter
    model, g = fixture()
    x, ys = _pool()
    for i in (0, 1):
        a = RecipientAdapter(model, g, i, 1.0); y = ys[i]
        hb, q0 = a.frozen_features(x)
        lp0 = torch.log_softmax(a.wire(hb, q0)['centered_logits'], 1)
        c = guard.pool_pass(a, hb, q0, y, lp0, params=True)
        assert c['kl'] == 0.0 and abs(c['gradients']['kl']).max() <= 1e-15 and c['rows'] == 2300
        old = guard.CHUNK; guard.CHUNK = 10 ** 9
        try:
            s = guard.pool_pass(a, hb, q0, y, lp0, params=True)
        finally:
            guard.CHUNK = old
        for k in ('nll', 'brier', 'ce'):
            assert abs(c[k] - s[k]) <= 1e-12 * max(1, abs(s[k]))
            assert abs(c['gradients'][k] - s['gradients'][k]).max() <= 1e-12 * max(1, abs(s['gradients'][k]).max())
        a.R = torch.randn(a.rank, 64, dtype=torch.float64) * 0.05
        m = guard.pool_pass(a, hb, q0, y, lp0, params=True)
        assert m['kl'] > 0 and abs(m['gradients']['kl']).max() > 0
        eps = 1e-6; e = torch.zeros(a.rank, dtype=torch.float64); e[0] = eps
        a.b = a.b + e; up = guard.pool_pass(a, hb, q0, y, lp0); a.b = a.b - 2 * e; dn = guard.pool_pass(a, hb, q0, y, lp0); a.b = a.b + e
        for k in ('kl', 'ce', 'brier'):
            fd = (up[k] - dn[k]) / (2 * eps)
            assert abs(fd - m['gradients'][k][a.rank * 64]) <= 1e-6 * max(1, abs(fd))


def test_guarded_step_backtracks_rolls_back_and_respects_norm_income():
    import numpy as np
    import torch
    from . import guard
    from .adapter import RecipientAdapter
    model, g = fixture(); a = RecipientAdapter(model, g, 0, 1.0)
    x, ys = _pool(); y = ys[0]; hb, q0 = a.frozen_features(x)
    lp0 = torch.log_softmax(a.wire(hb, q0)['centered_logits'], 1)
    start = guard.pool_pass(a, hb, q0, y, lp0, params=True)
    anchor = guard.Anchor(start, trust=True)
    before = guard.flat([a.R, a.b])
    huge = -start['gradients']['ce'] * 1e4
    r = guard.guarded_step(a, huge, start, anchor, lambda: guard.pool_pass(a, hb, q0, y, lp0), 0.01)
    assert np.linalg.norm(guard.flat([a.R, a.b]) - before) <= 0.01 + 1e-15
    if r['status'] == 'ACCEPTED':
        assert anchor.within(guard.pool_pass(a, hb, q0, y, lp0))
    a.R = torch.zeros(1, 64, dtype=torch.float64); a.b = torch.zeros(1, dtype=torch.float64)
    tight = guard.Anchor(start, trust=True); tight.kl_limit = 1e-14
    r3 = guard.guarded_step(a, huge, start, tight, lambda: guard.pool_pass(a, hb, q0, y, lp0), 0.01)
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


def _synthetic_ctx(tmp_path):
    import torch
    from . import train_worker as T
    from .adapter import RecipientAdapter, tensor_sha
    from .guard import pool_pass, Anchor
    model, g = fixture()
    before = {k: tensor_sha(v) for k, v in model.state_dict().items()}
    torch.manual_seed(5)
    xU = torch.randn(600, 83); xG = torch.randn(150, 83); sU = torch.randint(0, 2, (600,))
    labels = {0: (torch.randint(0, 2, (600,)), torch.randint(0, 2, (150,))), 1: (torch.randint(0, 6, (600,)), torch.randint(0, 6, (150,)))}
    feats = []; anchors = []
    for i in (0, 1):
        a0 = RecipientAdapter(model, g, i, 1.0)
        hU, qU = a0.frozen_features(xU); hG, qG = a0.frozen_features(xG)
        yU, yG = labels[i]
        with torch.no_grad():
            lU = torch.log_softmax(a0.wire(hU, qU)['centered_logits'], 1); lG = torch.log_softmax(a0.wire(hG, qG)['centered_logits'], 1)
        f = dict(hU=hU, qU=qU, yU=yU, hG=hG, qG=qG, yG=yG, hP=torch.cat((hU, hG)), qP=torch.cat((qU, qG)), yP=torch.cat((yU, yG)),
                 lp0U=lU, lp0G=lG, lp0P=torch.cat((lU, lG)), probe_h=hU[:64], probe_q=qU[:64])
        feats.append(f); anchors.append(Anchor(pool_pass(a0, f['hP'], f['qP'], f['yP'], f['lp0P']), trust=True))
    tol = {'representation': 1e-10, 'centered_logits': 1e-10, 'probabilities': 1e-10}
    ctx = dict(seed=0, geometry=g, scales=[1.0, 1.0], task=T.Task(600, 150), source_model=model, feats=feats, anchors=anchors, sU=sU,
               reference_rows=T._draw(random.Random(1), 600), out=tmp_path, wire_tolerance=[tol, tol],
               source_guard=lambda: all(tensor_sha(v) == before[k] for k, v in model.state_dict().items()) or (_ for _ in ()).throw(AssertionError('source changed')),
               checkpoint_sha256='a' * 64, adapter_binding_sha256='b' * 64)
    return ctx, model, before


def test_bank_end_to_end_simultaneous_sequential_phase_freezing_and_counterfactuals(tmp_path):
    """Every arm runs real proposals/critics/checkpoints; inactive residual frozen; sequential pairs saved; J_BOTH counterfactuals for both recipients."""
    import numpy as np
    import torch
    from . import train_worker as T
    from .adapter import tensor_sha
    from .recipe import ARMS, RECIPE
    ctx, model, before = _synthetic_ctx(tmp_path)
    norm = T.normalization(ctx, tmp_path)
    assert all(v['status'] == 'FIXED' for v in norm['coefficients'].values())
    cj = {r: norm['coefficients'][f'joint|{r}']['coefficient'] for r in ('income', 'occupation')}
    cl = {r: norm['coefficients'][f'local|{r}']['coefficient'] for r in ('income', 'occupation')}
    ctx['coefficients'] = {'T_BOTH': None, 'L_BOTH': cl, 'J_BOTH': cj, 'S12': cj, 'S21': cj, 'I_ONLY': cj}
    ctx['common'] = {'source_seed': 0}
    old = dict(RECIPE)
    RECIPE['epochs_per_active_recipient'] = 2; RECIPE['proposals_per_epoch'] = 3; RECIPE['checkpoint_epochs'] = [0, 1, 2]
    try:
        out = {arm: T.run_arm(arm, ctx) for arm in ARMS}
    finally:
        RECIPE.clear(); RECIPE.update(old)
    assert len({s['warm_critic_state_sha256'] for s in out.values()}) == 1
    assert out['J_BOTH']['opportunities'] == {'income': 6, 'occupation': 6} and out['J_BOTH']['macro_steps'] == 6
    assert out['S12']['opportunities'] == {'income': 6, 'occupation': 6} and out['S12']['macro_steps'] == 12
    assert out['I_ONLY']['opportunities'] == {'income': 6, 'occupation': 0}
    assert out['J_BOTH']['critic_rounds'] == 80 + 10 * 6 and out['S21']['critic_rounds'] == 80 + 5 * 12 and out['I_ONLY']['critic_rounds'] == 80 + 5 * 6
    assert all(tensor_sha(v) == before[k] for k, v in model.state_dict().items())
    # Sequential phase freezing: P2E0 pair equals P1E-last pair; first recipient bitwise fixed through phase 2.
    for arm, first, second in (('S12', 0, 1), ('S21', 1, 0)):
        ck = {c['label']: torch.load(tmp_path / arm / c['path_name'], weights_only=False) for c in out[arm]['checkpoints']}
        assert [c['label'] for c in out[arm]['checkpoints']] == ['P1E0', 'P1E1', 'P1E2', 'P2E0', 'P2E1', 'P2E2']
        for k in ('R', 'b'):
            assert torch.equal(ck['P1E2']['states'][first][k], ck['P2E2']['states'][first][k])
            assert torch.equal(ck['P1E2']['states'][second][k], torch.zeros_like(ck['P1E2']['states'][second][k]))
            assert torch.equal(ck['P2E0']['states'][second][k], ck['P1E2']['states'][second][k])
    ck = torch.load(tmp_path / 'I_ONLY' / 'E2.pt', weights_only=False)
    assert torch.equal(ck['states'][1]['R'], torch.zeros(5, 64, dtype=torch.float64))
    sc = out['J_BOTH']['counterfactual_scan']
    assert len(sc['income']) >= 1 and len(sc['occupation']) >= 1
    assert all('privacy_minus_counterfactual_max_abs' in r for r in sc['income'] + sc['occupation'])
    for arm in ARMS:
        rows = json.loads((tmp_path / arm / 'BUDGET_TRACE.json').read_text())
        assert all(r['post_kl'] is None or r['post_kl'] <= 0.001 + 1e-10 for r in rows)
        assert all(r['post_nll'] is None or r['post_nll'] <= r['nll_limit'] + 1e-10 for r in rows)
        assert len({(r['recipient'], r['nll_limit']) for r in rows}) == len({r['recipient'] for r in rows})


def test_simultaneous_macro_step_is_order_independent_and_uses_one_pre_state(tmp_path):
    """Both proposals from one pre-update state; applying income then occupation equals the reverse order bitwise."""
    import numpy as np
    from . import train_worker as T
    from .guard import flat
    ctx, model, _ = _synthetic_ctx(tmp_path)
    critics = T.warm_critics(ctx, T.fresh_adapters(ctx))
    ctx['coefficients'] = {'J_BOTH': {'income': 0.7, 'occupation': 0.9}}
    rows = T._draw(random.Random(3), 600)
    results = []
    for order in ((0, 1), (1, 0)):
        ads = T.fresh_adapters(ctx)
        import torch
        for i, ad in enumerate(ads):
            ad.R = torch.full_like(ad.R, 1e-3 * (i + 1)); ad.b = torch.full_like(ad.b, -1e-3)
        g, _ = T.privacy_gradients(ads, critics, ctx['feats'], rows, ctx['sU'][rows], 'joint', (0, 1))
        preps = {i: T.recipient_step(i, ads, ctx, rows, g[i], ctx['coefficients']['J_BOTH'][('income', 'occupation')[i]]) for i in (0, 1)}
        statuses = [T.apply_step(i, ads, ctx, preps[i])[0]['status'] for i in order]
        assert statuses == ['ACCEPTED', 'ACCEPTED']
        results.append([flat([a.R, a.b]) for a in ads])
    assert all(np.array_equal(a, b) for a, b in zip(*results))
    assert all(np.any(r) for r in results[0])


def test_per_recipient_privacy_gradients(tmp_path):
    """Active (0,) equals the income block of (0,1); local uses only the own view; joint gradient matches finite differences."""
    import numpy as np
    import torch
    from . import train_worker as T
    ctx, model, _ = _synthetic_ctx(tmp_path)
    ads = T.fresh_adapters(ctx)
    for i, ad in enumerate(ads):
        ad.R = torch.full_like(ad.R, 2e-3 * (i + 1)); ad.b = torch.full_like(ad.b, 1e-3)
    critics = T.warm_critics(ctx, ads)
    rows = T._draw(random.Random(4), 600); s = ctx['sU'][rows]; F = ctx['feats']
    both, _ = T.privacy_gradients(ads, critics, F, rows, s, 'joint', (0, 1))
    inc, _ = T.privacy_gradients(ads, critics, F, rows, s, 'joint', (0,))
    occ, _ = T.privacy_gradients(ads, critics, F, rows, s, 'joint', (1,))
    assert np.allclose(both[0], inc[0], atol=1e-14, rtol=0) and np.allclose(both[1], occ[1], atol=1e-14, rtol=0)
    def terms_at(vec, i):
        n = ads[i].R.numel(); old = (ads[i].R, ads[i].b)
        ads[i].R = torch.from_numpy(vec[:n].reshape(ads[i].R.shape).copy()); ads[i].b = torch.from_numpy(vec[n:].copy())
        with torch.no_grad():
            w = [a.wire(F[k]['hU'][rows], F[k]['qU'][rows])['full'] for k, a in enumerate(ads)]
            t = critics.terms(w[0], w[1], s)
        ads[i].R, ads[i].b = old
        return t
    from .guard import flat
    for i in (0, 1):
        base = flat([ads[i].R, ads[i].b]); g = both[i]
        # local gradient equals gradient of own-view risk only
        loc, _ = T.privacy_gradients(ads, critics, F, rows, s, 'local', (i,))
        own = 'income' if i == 0 else 'occupation'; other = 'occupation' if i == 0 else 'income'
        k = int(np.argmax(np.abs(g))); eps = 1e-6; e = np.zeros_like(base); e[k] = eps
        up, dn = terms_at(base + e, i), terms_at(base - e, i)
        fd_joint = float(sum(up[v] - dn[v] for v in ('income', 'occupation', 'pair'))) / (2 * eps)
        fd_own = float(up[own] - dn[own]) / (2 * eps)
        assert abs(float(up[other] - dn[other])) == 0.0
        assert abs(fd_joint - g[k]) <= 1e-5 * max(1.0, abs(fd_joint))
        kl = int(np.argmax(np.abs(loc[i]))); e = np.zeros_like(base); e[kl] = eps
        up, dn = terms_at(base + e, i), terms_at(base - e, i)
        assert abs(float(up[own] - dn[own]) / (2 * eps) - loc[i][kl]) <= 1e-5 * max(1.0, abs(loc[i][kl]))


def test_counterfactual_scans_do_not_disturb_trajectory(tmp_path):
    import numpy as np
    from . import train_worker as T
    from .recipe import RECIPE
    out = {}
    for tag in ('with', 'without'):
        d = tmp_path / tag; d.mkdir()
        ctx, model, _ = _synthetic_ctx(d)
        ctx['coefficients'] = {'J_BOTH': {'income': 0.8, 'occupation': 0.8}}; ctx['common'] = {'source_seed': 0}
        old = dict(RECIPE); real = T.counterfactual
        RECIPE['epochs_per_active_recipient'] = 1; RECIPE['proposals_per_epoch'] = 4; RECIPE['checkpoint_epochs'] = [0, 1]
        if tag == 'without':
            T.counterfactual = lambda *a, **k: ({}, {})
        try:
            s = T.run_arm('J_BOTH', ctx)
        finally:
            RECIPE.clear(); RECIPE.update(old); T.counterfactual = real
        with np.load(d / 'J_BOTH' / 'STATE_TRACE.npz') as z:
            out[tag] = (z['income'].copy(), z['occupation'].copy(), s['checkpoints'][-1]['R_sha256'])
    assert np.array_equal(out['with'][0], out['without'][0]) and np.array_equal(out['with'][1], out['without'][1])
    assert out['with'][2] == out['without'][2]
