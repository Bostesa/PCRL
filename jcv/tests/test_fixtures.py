"""Registered executable fixtures (synthetic). Real-data success rules are not touched by these tests."""
import numpy as np
import torch

from jcv import audit as A
from jcv import fixtures as FX
from jcv import train as T


def test_xor_attacker_positive_control():
    r = FX.xor_attacker_control()
    assert max(r["A_alone"].values()) < 0.55 and max(r["B_alone"].values()) < 0.55
    assert r["pair"]["LR_C1"] < 0.55                     # linear attacker cannot read XOR
    assert max(r["pair"]["MLP_64x64"], r["pair"]["HGB_0.1_31"]) > 0.95


def test_xor_training_positive_control_J_beats_L():
    X, y1, y2, s, a, b = FX.xor_data(30000, 0)
    r = FX.run_arms(X, y1, y2, s, ["L", "J"], beta=10.0, masks=FX.XOR_MASKS)
    assert r["L"]["recovery_max_inner"]["v1"] < 0.55 and r["L"]["recovery_max_inner"]["v2"] < 0.55
    assert r["L"]["recovery_max_inner"]["pair"] > 0.95                    # local training cannot see the clue
    assert r["J"]["recovery_max_inner"]["pair"] < r["L"]["recovery_max_inner"]["pair"] - 0.2
    assert min(r["J"]["task_acc"].values()) > 0.95                         # tasks preserved


def test_task_equal_to_S_is_reported_as_conflict():
    X, y1, y2, s, a, b = FX.xor_data(9000, 1, task_eq_s=True)
    r = FX.run_arms(X, y1, y2, s, ["U", "J"], beta=10.0, warm=8, prot=6)
    assert r["U"]["task_acc"][0] > 0.99 and r["U"]["recovery_max_inner"]["v1"] > 0.95
    assert r["J"]["task_acc"][0] < 0.6          # cannot keep task while hiding S: utility gate must fail, not be repaired


def test_constant_release_is_withholding():
    r = FX.constant_release_control()
    assert abs(r["recovery"] - 0.5) < 1e-9 and r["useful_gain_over_constant"] == 0.0


def test_privacy_direction_sign_and_seed_aliasing():
    X, y1, y2, s, a, b = FX.xor_data(3000, 2)
    data = FX._engine_data(X, y1, y2, s, np.arange(3000), 0)
    m0 = T.Model(X.shape[1], [2, 2], 0)
    m1 = T.Model(X.shape[1], [2, 2], 0)
    m2 = T.Model(X.shape[1], [2, 2], 1)
    sd0, sd1, sd2 = m0.state_dict(), m1.state_dict(), m2.state_dict()
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0)                 # same seed -> bitwise identical
    assert not all(torch.equal(sd0[k], sd2[k]) for k in sd0)             # different seed -> different
    # descent sign: a small step along -grad(task) lowers the task loss; along +grad raises it
    params = [p for p in m0.enc[0].parameters()] + [p for p in m0.head[0].parameters()]
    L = T.task_losses(m0, data.X, data.Y, [None, None], (0,))[0]
    g = T.flat_grad(L, params)
    snap = T.snapshot(params)
    T.assign_add(params, -g, 1e-3)
    down = float(T.task_losses(m0, data.X, data.Y, [None, None], (0,))[0])
    T.restore(params, snap)
    T.assign_add(params, g, 1e-3)
    up = float(T.task_losses(m0, data.X, data.Y, [None, None], (0,))[0])
    assert down < float(L) < up


def test_centring_and_affine_logits():
    from jcv.finalize import outputs
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    R = rng.normal(size=(500, 5))
    y = rng.integers(0, 3, 500)
    head = LogisticRegression(max_iter=2000).fit(R, y)
    cen, P, hard = outputs(head, R)
    assert np.allclose(cen.sum(1), 0, atol=1e-10)
    assert np.allclose(np.exp(cen) / np.exp(cen).sum(1, keepdims=True), P, atol=1e-10)
    assert np.array_equal(hard, P.argmax(1))
    # centred logits are affine in R: second differences vanish along a line
    r0, dlt = R[:1], rng.normal(size=(1, 5))
    c = [outputs(head, r0 + t * dlt)[0] for t in (0.0, 1.0, 2.0)]
    assert np.allclose(c[2] - 2 * c[1] + c[0], 0, atol=1e-9)
