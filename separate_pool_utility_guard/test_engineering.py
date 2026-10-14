"""Finite engineering checks; synthetic inputs only; run through the queue."""
import copy
import json
import pytest


@pytest.fixture(autouse=True)
def queued():
    from .scheduler import assert_active_from_env
    assert_active_from_env(stage='admission')


def fixture_model(seed=913):
    import torch
    from task_conditioned_representation.useful_service import UsefulModel
    torch.manual_seed(seed)
    model = UsefulModel([f'x{i}' for i in range(83)]); model.eval()
    return model


def test_slack_projection_matches_reference_solver():
    import numpy as np
    from scipy.optimize import minimize
    from .guard import project_slack
    rng = np.random.default_rng(7)
    for _ in range(400):
        n = int(rng.integers(2, 7)); p = rng.normal(size=n) * rng.choice([1e-3, 1e-2, 5e-2])
        a = rng.normal(size=(2, n)); b = np.abs(rng.normal(size=2)) * rng.choice([0, 1e-4, 1e-2], size=2)
        d, info = project_slack(p, a, b, 0.01)
        assert (a @ d <= b + 1e-9).all() and np.linalg.norm(d) <= 0.01 + 1e-12 and not info['conservative_zero_fallback']
        cons = [{'type': 'ineq', 'fun': lambda x, j=j: b[j] - a[j] @ x} for j in range(2)] + [{'type': 'ineq', 'fun': lambda x: 1e-4 - x @ x}]
        ref = minimize(lambda x: .5 * np.sum((x - p) ** 2), np.zeros(n), jac=lambda x: x - p, constraints=cons,
                       method='SLSQP', options={'ftol': 1e-16, 'maxiter': 500})
        if (a @ ref.x <= b + 1e-9).all() and np.linalg.norm(ref.x) <= 0.01 + 1e-9:
            assert .5 * np.sum((d - p) ** 2) <= .5 * np.sum((ref.x - p) ** 2) + 1e-9


def test_zero_remains_feasible_and_unused_slack_is_spent():
    import numpy as np
    from .guard import project_slack
    a = np.array([[1.0, 0.0], [0.0, 1.0]]); p = np.array([0.02, 0.02])
    d0, _ = project_slack(p, a, np.zeros(2), 0.01)
    assert np.all(a @ d0 <= 1e-12)
    d1, info = project_slack(p, a, np.array([0.003, 0.004]), 0.01)
    assert np.allclose(d1, [0.003, 0.004]) and info['active_set'] == [0, 1]
    with pytest.raises(ValueError):
        project_slack(p, a, np.array([-1e-6, 0.0]), 0.01)


class _Curved:
    """f(theta)=1000 theta^2 - theta: linear model predicts descent; curvature fails large steps."""
    def __init__(self, theta):
        self.theta = theta

    def metrics(self):
        t = float(self.theta[0])
        return {'nll': 1000 * t * t - t, 'brier': 0.0, 'correct': 10, 'rows': 10, 'accuracy': 1.0}


def test_curved_loss_fails_first_order_check_and_backtracks():
    import numpy as np
    import torch
    from .guard import Anchor, global_step
    theta = torch.zeros(1, dtype=torch.float32)
    c = _Curved(theta); start = c.metrics()
    anchor = Anchor(start); anchor.nll_limit = start['nll'] + 0.001
    current = dict(start, gradients={'nll': np.array([-1.0]), 'brier': np.array([0.0])})
    r = global_step([theta], np.array([0.01]), current, anchor, c.metrics, 0.01)
    assert r['status'] == 'ACCEPTED' and r['accepted_fraction'] == 0.125
    assert [a['accepted'] for a in r['attempts']] == [False, False, False, True]
    assert r['attempts'][0]['linearized_check'] is True and r['attempts'][0]['within_absolute_budget'] is False


def test_failed_steps_restore_exact_mixed_dtype_weights():
    import numpy as np
    import torch
    from .guard import Anchor, global_step
    from .strict import parameter_vector
    model = fixture_model(); params = list(model.enc[1].parameters()) + list(model.head[1].parameters())
    snap = [p.detach().clone() for p in params]
    start = {'nll': 1.0, 'brier': 0.5, 'correct': 100, 'rows': 100, 'accuracy': 1.0}
    anchor = Anchor(start)
    worse = lambda: {'nll': 2.0, 'brier': 0.5, 'correct': 100, 'rows': 100, 'accuracy': 1.0}
    n = len(parameter_vector(params))
    current = dict(start, gradients={'nll': np.zeros(n), 'brier': np.zeros(n)})
    r = global_step(params, np.full(n, 1e-4), current, anchor, worse, 0.01)
    assert r['status'] == 'REJECTED_ZERO_STEP'
    assert all(torch.equal(a, b) and a.dtype == b.dtype for a, b in zip(params, snap))
    assert {p.dtype for p in params} == {torch.float32, torch.float64}


def test_budgets_are_absolute_and_never_reset():
    from .guard import Anchor
    start = {'nll': 1.0, 'brier': 0.5, 'correct': 1000, 'rows': 2000, 'accuracy': .5}
    anchor = Anchor(start)
    assert anchor.nll_limit == 1.010 and anchor.brier_limit == 0.505 and anchor.correct_floor == 980
    # A sequence of small accepted increases cannot exceed the start-anchored budget.
    state = dict(start)
    for _ in range(50):
        state = dict(state, nll=state['nll'] + 0.0009)
        if not anchor.within(state):
            break
    assert state['nll'] <= 1.010 + 0.0009 and not anchor.within(state)
    assert anchor.slacks({'nll': 1.004, 'brier': 0.501})[0] == pytest.approx(0.006)


def test_chunked_full_pool_equals_single_pass_and_sampled_pool_differs():
    import numpy as np
    import torch
    from . import guard
    from .wires import bind_geometry, prepare_geometry
    model = fixture_model(); g = prepare_geometry(bind_geometry(model, source_seed=0, checkpoint_sha256='a' * 64))
    for _, p in model.named_parameters():
        p.requires_grad_(True)
    params = list(model.enc[1].parameters()) + list(model.head[1].parameters())
    torch.manual_seed(3); x = torch.randn(2500, 83); y = torch.randint(0, 6, (2500,))
    chunked = guard.pool_pass(model, x, y, g, params)
    old = guard.CHUNK; guard.CHUNK = 10 ** 9
    try:
        single = guard.pool_pass(model, x, y, g, params)
    finally:
        guard.CHUNK = old
    for k in ('nll', 'brier', 'ce'):
        assert abs(chunked[k] - single[k]) <= 1e-12 * max(1, abs(single[k]))
        # Encoder backprop is float32: chunk sums agree to float32 rounding, and the
        # fixed chunk rule (not chunk invariance) makes every pass deterministic.
        rel = np.max(np.abs(chunked['gradients'][k] - single['gradients'][k])) / np.max(np.abs(single['gradients'][k]))
        assert rel <= 1e-5
    assert chunked['correct'] == single['correct'] and chunked['rows'] == 2500
    again = guard.pool_pass(model, x, y, g, params)
    assert all(np.array_equal(again['gradients'][k], chunked['gradients'][k]) for k in ('nll', 'brier', 'ce'))
    sampled = guard.pool_pass(model, x[:256], y[:256], g)
    assert sampled['rows'] != chunked['rows']


def test_shared_task_weights_population():
    import numpy as np
    from .train_worker import Task
    t = Task(8000, 2000)
    assert t.w_update == 0.8 and t.w_guard == 0.2


def test_new_queue_schema_rejects_old_closed_queue():
    from pathlib import Path
    from .scheduler import QUEUE_SCHEMA
    old = json.loads((Path(__file__).resolve().parents[2] / 'pcrl-compressed-income-safeguard-v1/.private/compressed_income_safeguard/compute/state.json').read_text())
    assert old['schema'] != QUEUE_SCHEMA and old['closed'] is True


def test_five_arm_end_to_end_synthetic(tmp_path):
    """Every arm runs real proposals, checkpoints, counterfactual scan and matched critics."""
    import random
    import torch
    from .train_worker import run_arm, Task, _freeze, OccupationCritics, _draw, _seed, _params, normalization
    from .guard import pool_pass
    from .recipe import ARMS, RECIPE
    from .wires import bind_geometry, prepare_geometry
    model = fixture_model(); g = prepare_geometry(bind_geometry(model, source_seed=0, checkpoint_sha256='a' * 64))
    _freeze(model, g)
    torch.manual_seed(5)
    xU = torch.randn(600, 83); yU = torch.randint(0, 6, (600,)); sU = torch.randint(0, 2, (600,))
    xG = torch.randn(150, 83); yG = torch.randint(0, 6, (150,))
    task = Task(600, 150)
    nm = copy.deepcopy(model); nf = _freeze(nm, g); nc = OccupationCritics(0, sU, g, nf)
    ref = _draw(random.Random(_seed(0, 'reference')), 600); nc.refresh(nm, xU[ref])
    wr = random.Random(_seed(0, 'warm'))
    for _ in range(80):
        rows = _draw(wr, 600); nc.update(nm, xU[rows], sU[rows])
    coef, _ = normalization(nm, nc, task, xU, yU, sU, lambda: pool_pass(nm, xG, yG, g, _params(nm), want=('ce',)), 0, tmp_path)
    ctx = dict(seed=0, geometry=g, task=task, xU=xU, yU=yU, sU=sU, xG=xG, yG=yG, source_model=model,
               source_payload={'schema': 'TCR1-trainable-own-head-checkpoint-v1', 'feature_names': list(model.names),
                               'class_orders': model.class_orders, 'source_bindings': {}},
               reference_rows=ref, out=tmp_path, coefficient=coef, probe_x=xU[:64],
               wire_tolerance={'representation': 1e-10, 'centered_logits': 1e-10, 'probabilities': 1e-10},
               anchors={'GUARD': pool_pass(model, xG, yG, g), 'UPDATE': pool_pass(model, xU, yU, g)}, budget_rows=[],
               common={'source_seed': 0, 'proposals_per_epoch': 2, 'coefficient': coef})
    out = {arm: run_arm(arm, ctx) for arm in ARMS}
    assert len({s['warm_critic_state_sha256'] for s in out.values()}) == 1
    assert all(s['model_updates'] == 10 and s['critic_optimizer_steps'] == (80 + 50) * 6 for s in out.values())
    assert all(len(s['checkpoints']) == 4 for s in out.values())
    assert out['TASK_ONLY']['update_status_counts'].get('ACCEPTED', 0) == 10
    for arm in ('SEPARATE_GUARD', 'GLOBAL_UPDATE_GUARD'):
        rows = [r for r in ctx['budget_rows'] if r['arm'] == arm]
        assert len(rows) == 10 and all(r['post_nll'] is None or r['post_nll'] <= r['nll_limit'] + 1e-10 for r in rows)
        assert len({r['nll_limit'] for r in rows}) == 1
    assert out['SEPARATE_GUARD']['guard_pass_accounting']['shared_task_GUARD_full_passes'] == 10
