"""Design-review fixtures for the strength-matched feedback study (owned by the design reviewer).

Small synthetic tensors and synthetic DEFENSE_FIT-shaped arrays only: no study row is read and there is no scientific
fit. Every test is written so that it can FAIL on a real defect (wrong denominator, float32 norm overflow, a cap or
zero rule that invents strength, a coalition gradient in a local arm, a controller with the wrong sign, clip or
selectivity, a broken common-weight symmetry, a matched-count template that is not honoured per bank, a probe that is
blind to a rotated low-scale clue or flips orientation on the calibration rows, a selection rule that drops a feasible
control). See results/pcrl_strength_matched_feedback_v1/MATH_REVIEW.md for the failing receipts.

    env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest smf/tests/test_math_review.py -q

Selection fixtures monkeypatch BOTH the private run directory (R.RUN) and the public results directory (R.PKG) to
tmp_path, and a guard asserts that the real results folder is byte-for-byte untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.stats import norm

try:
    from smf import train as T
except Exception:  # pragma: no cover - engine not yet importable
    T = None

needs_T = pytest.mark.skipif(T is None, reason="smf.train not importable")
torch.set_num_threads(1)

WT = Path(__file__).resolve().parents[2]
REAL_PKG = WT / "results" / "pcrl_strength_matched_feedback_v1"


def _fn(name):
    f = getattr(T, name, None) if T is not None else None
    if f is None:
        pytest.skip(f"smf.train.{name} not available")
    return f


# ------------------------------------------------------------------ synthetic helpers (no study rows)
class SynthData:
    """DEFENSE_FIT-shaped synthetic data (83 inputs, SEX prior ~0.67, binary income and 6-class occupation that depend
    on the inputs and on SEX) with fixed CRITIC_FIT / CRITIC_VAL / CONTROLLER_CALIB sets and the online reference."""

    def __init__(self, n=768, d=83, seed=0, sex_strength=1.0):
        rng = np.random.default_rng(seed)
        S = (rng.random(n) < 0.67).astype(np.int64)
        X = rng.normal(size=(n, d)).astype(np.float32)
        X[:, :6] += sex_strength * (2 * S[:, None] - 1) * np.linspace(0.8, 0.2, 6)[None, :]
        lin = X[:, 6:20] @ rng.normal(size=14) * 0.4 + 0.5 * (2 * S - 1)
        yi = (lin + rng.normal(size=n) > 0.6).astype(np.int64)
        lo = X[:, 20:40] @ rng.normal(size=(20, 6)) * 0.4
        lo[:, 0] += 0.8 * S
        yo = np.argmax(lo + rng.gumbel(size=lo.shape), 1).astype(np.int64)
        self.X = torch.from_numpy(X)
        self.Y = {0: torch.from_numpy(yi), 1: torch.from_numpy(yo)}
        self.S = torch.from_numpy(S)
        self.n = n
        self.prior = np.bincount(S, minlength=2) / n
        u = rng.random(n)
        self.cf = np.flatnonzero(u < 0.70)
        self.cv = np.flatnonzero((u >= 0.70) & (u < 0.85))
        self.cal = np.flatnonzero(u >= 0.85)
        self.ref = np.sort(np.random.default_rng(20261004).choice(self.cf, min(4096, len(self.cf)), replace=False))


def warm_state(data, seed=0, epochs=3):
    m = T.Model(T.D_IN, T.KS, seed)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = np.random.default_rng([seed, 999, ep]).permutation(data.n)
        for s in range(0, data.n, 256):
            b = torch.from_numpy(perm[s:s + 256])
            L = T.task_losses(m, data.X[b], {i: data.Y[i][b] for i in (0, 1)}, [None, None])
            opt.zero_grad()
            (L[0] + L[1]).backward()
            opt.step()
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


@pytest.fixture(scope="module")
def world():
    if T is None:
        pytest.skip("smf.train not importable")
    data = SynthData()
    st = warm_state(data)
    return {"data": data, "warm": st, "head": T.head_of(st), "steps": math.ceil(data.n / 256)}


def run(world, spec, rho, n_epochs=1, **kw):
    kw.setdefault("ckpt_epochs", ())
    return T.train_run(spec, rho, world["warm"], world["data"], 0, kw.pop("stage", "A"), world["head"],
                       n_epochs=n_epochs, **kw)


def same_state(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def task_spec():
    return {"base": "none", "schedule": "ONLINE", "update": "task"}


def fresh_init_critics(seed=0, perturb_pair=False):
    from rgj import train as RT
    crit = {v: {} for v in RT.VIEWS}
    for v in RT.VIEWS:
        for k in RT.KINDS:
            c = RT.new_critic(seed, v, k, "init")
            if perturb_pair and v == "pair":
                g = torch.Generator().manual_seed(7)
                with torch.no_grad():
                    for p in c.parameters():
                        p.mul_(3.0).add_(torch.randn(p.shape, generator=g))
            crit[v][k] = copy.deepcopy(c.state_dict())
    return {"critics": crit, "transforms": {v: None for v in RT.VIEWS}}


def rand_vecs(n=10576, seed=0, t_scale=0.01, p_scale=0.1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g) * t_scale, torch.randn(n, generator=g) * p_scale


def nrm64(x):
    return float(x.double().norm())


# ------------------------------------------------------------------ 1. per-encoder norm matching (section 7)
@needs_T
@pytest.mark.parametrize("k", [-20, -3, 1, 5, 20, 70])
def test_positive_rescaling_of_p_leaves_q_bitwise_unchanged_power_of_two(k):
    """c = 2^k is exact in binary floating point, so q(c p) must equal q(p) BITWISE when neither the cap nor the zero
    threshold is active. k = 70 makes sum(p^2) overflow float32 (~1e39): a float32 norm returns inf and fails here."""
    nd = _fn("normalized_direction")
    t, p = rand_vecs(t_scale=1e-8)          # ||t|| << ||p||: a ~ 1e-7, far from the cap for every k used
    q0, i0 = nd(t, p, 0.75)
    q1, i1 = nd(t, p * (2.0 ** k), 0.75)
    assert not i0["cap"] and not i1["cap"] and i0["zero"] is None and i1["zero"] is None
    assert torch.equal(q0, q1)
    assert abs(i1["ratio"] - 0.75) < 1e-6


@needs_T
@pytest.mark.parametrize("c", [0.37, 3.7, 1234.5, 1e-4])
def test_positive_rescaling_of_p_leaves_q_unchanged_general_c(c):
    nd = _fn("normalized_direction")
    t, p = rand_vecs(seed=1, t_scale=1e-6)
    q0, _ = nd(t, p, 1.5)
    q1, i1 = nd(t, p * c, 1.5)
    assert not i1["cap"] and i1["zero"] is None
    assert torch.allclose(q0, q1, rtol=2e-6, atol=0.0)


@needs_T
def test_achieved_ratio_and_direction_are_exact():
    nd = _fn("normalized_direction")
    t, p = rand_vecs(seed=2)
    for rho in (0.25, 0.75, 1.5):
        q, info = nd(t, p, rho)
        assert abs(nrm64(q) / nrm64(t) - rho) < 1e-6 and abs(info["ratio"] - rho) < 1e-6
        cos_qp = float(torch.dot(q.double(), p.double()) / (nrm64(q) * nrm64(p)))
        assert cos_qp > 1 - 1e-6                                   # same direction as p, positive multiple
        assert abs(info["t_norm"] - nrm64(t)) < 1e-12 * nrm64(t)   # float64 norm of t, not of p or t+p


@needs_T
def test_scale_factor_is_stop_gradient():
    nd = _fn("normalized_direction")
    t, p = rand_vecs(seed=3)
    p = p.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    q, _ = nd(t, p, 0.75)
    assert not q.requires_grad and q.grad_fn is None


@needs_T
def test_zero_p_and_zero_t_give_q_zero_and_log_the_event():
    nd = _fn("normalized_direction")
    t, p = rand_vecs(seed=4)
    q, info = nd(t, torch.zeros_like(p), 0.75)
    assert torch.equal(q, torch.zeros_like(p)) and info["zero"] == "p"
    assert info.get("ratio") in (None, 0.0)                         # never rho: no invented strength
    q, info = nd(torch.zeros_like(t), p, 0.75)
    assert torch.equal(q, torch.zeros_like(p)) and info["zero"] == "t"
    assert info.get("ratio") is None                                # 0/0: undefined
    tol = T.HP["ZERO_TOL"]
    q, info = nd(t, p / nrm64(p) * tol * 0.5, 0.75)                 # below the fixed threshold
    assert torch.equal(q, torch.zeros_like(p)) and info["zero"] == "p"


@needs_T
def test_cap_binds_and_reports_achieved_ratio():
    nd = _fn("normalized_direction")
    t, p = rand_vecs(seed=5)
    p = p / nrm64(p) * (1e-4 * nrm64(t))                            # ||p|| / ||t|| = 1e-4 -> a = 1.5e4 > 100
    q, info = nd(t, p, 1.5)
    assert info["cap"] is True and info["zero"] is None
    assert abs(info["a"] - T.HP["A_MAX"]) == 0.0
    assert abs(info["ratio"] - 100 * 1e-4) < 1e-9                   # achieved 0.01, not the declared 1.5
    assert torch.allclose(q, p * 100.0, rtol=1e-6, atol=0.0)
    # just above the boundary the cap is inactive and the ratio is the declared one
    p2 = p / nrm64(p) * (1.5 * nrm64(t) / 99.0)
    q2, info2 = nd(t, p2, 1.5)
    assert not info2["cap"] and abs(info2["ratio"] - 1.5) < 1e-6


@needs_T
def test_rho_zero_returns_exact_zero():
    nd = _fn("normalized_direction")
    t, p = rand_vecs(seed=6)
    q, _ = nd(t, p, 0.0)
    assert torch.equal(q, torch.zeros_like(p))


@needs_T
def test_denominator_is_per_encoder_task_gradient_not_head_or_both_encoders(world, monkeypatch):
    """Spy on normalized_direction inside train_run: on step 1 the t passed for encoder i must be exactly
    d L_i / d enc_i (length = encoder-i parameters; norm = independent recomputation), and must differ materially from
    the head-including and both-encoder alternatives (so the check discriminates)."""
    calls = []
    orig = T.normalized_direction

    def spy(t, p, rho_i, *a, **k):
        calls.append((t.detach().clone(), p.detach().clone(), rho_i))
        return orig(t, p, rho_i, *a, **k)

    monkeypatch.setattr(T, "normalized_direction", spy)
    d = world["data"]
    run(world, {"base": "joint", "schedule": "ONLINE"}, 0.75)
    m = T.Model(T.D_IN, T.KS, 0)
    m.load_state_dict(world["warm"])
    b = torch.from_numpy(np.random.default_rng([0, T.SALT["A"], 0]).permutation(d.n)[:256])
    ind, head_incl = [], []
    for i in (0, 1):
        L = F.cross_entropy(m.head[i](m.encode(i, d.X[b])), d.Y[i][b])
        ge = torch.autograd.grad(L, list(m.enc[i].parameters()), retain_graph=True)
        gh = torch.autograd.grad(L, list(m.head[i].parameters()))
        ge = torch.cat([g.reshape(-1) for g in ge])
        gh = torch.cat([g.reshape(-1) for g in gh])
        ind.append(ge)
        head_incl.append(nrm64(torch.cat([ge, gh])))
    n_enc = sum(p.numel() for p in m.enc[0].parameters())
    for i in (0, 1):
        t_i = calls[i][0]
        assert t_i.numel() == n_enc
        assert abs(nrm64(t_i) - nrm64(ind[i])) < 1e-5 * nrm64(ind[i])
        assert abs(head_incl[i] - nrm64(ind[i])) > 1e-3 * nrm64(ind[i])       # the check can tell them apart
    both = math.sqrt(nrm64(ind[0]) ** 2 + nrm64(ind[1]) ** 2)
    assert abs(both - nrm64(ind[0])) > 1e-3 * both
    assert calls[0][2] == 0.75 and calls[1][2] == 0.75                    # Phase A: s_1 = s_2 = 1


# ------------------------------------------------------------------ 2. Phase B allocation (section 10)
@needs_T
def test_rms_identity_before_caps_and_clipping():
    alloc, nd = _fn("allocation"), _fn("normalized_direction")
    rng = np.random.default_rng(0)
    for _ in range(50):
        w1, w2 = rng.uniform(0.25, 8.0, 2)
        s1, s2 = alloc(w1, w2)
        assert abs(math.sqrt((s1 * s1 + s2 * s2) / 2) - 1.0) < 1e-14
        t1, p1 = rand_vecs(seed=int(rng.integers(1e6)))
        t2, p2 = rand_vecs(seed=int(rng.integers(1e6)), t_scale=0.3)
        for rho in (0.25, 0.75, 1.5):
            _, i1 = nd(t1, p1, rho * s1)
            _, i2 = nd(t2, p2, rho * s2)
            rms = math.sqrt((i1["ratio"] ** 2 + i2["ratio"] ** 2) / 2)
            assert abs(rms - rho) < 2e-6 * rho


@needs_T
def test_allocation_range_is_bounded_by_sqrt2():
    alloc = _fn("allocation")
    s_hi, s_lo = alloc(8.0, 0.25)
    assert abs(s_hi - 8 / math.sqrt((64 + 0.0625) / 2)) < 1e-15 and s_hi < math.sqrt(2)
    assert abs(s_lo - 0.25 / math.sqrt((64 + 0.0625) / 2)) < 1e-15
    # feedback can raise one recipient's relative strength by at most 41% (s <= sqrt 2), only by taking from the other


@needs_T
def test_asymmetric_weights_change_actual_per_encoder_update_norms():
    alloc, nd = _fn("allocation"), _fn("normalized_direction")
    t1, p1 = rand_vecs(seed=10)
    t2, p2 = rand_vecs(seed=11, t_scale=0.05)
    s = alloc(3.0, 1.0)
    q1, _ = nd(t1, p1, 0.75 * s[0])
    q2, _ = nd(t2, p2, 0.75 * s[1])
    assert abs(nrm64(q1) / nrm64(t1) - 0.75 * 3 / math.sqrt(5)) < 1e-6
    assert abs(nrm64(q2) / nrm64(t2) - 0.75 * 1 / math.sqrt(5)) < 1e-6
    q1n, _ = nd(t1, p1, 0.75)
    assert nrm64(q1) > 1.3 * nrm64(q1n)


@needs_T
@pytest.mark.parametrize("w", [0.25, 0.5, 2.0, 3.0, 5.5, 8.0])
def test_equal_weight_increase_is_exact_symmetry_for_local(w):
    coef, alloc, nd = _fn("coefficients"), _fn("allocation"), _fn("normalized_direction")
    c1, c0 = coef("local", (w, w)), coef("local", (1.0, 1.0))
    assert c1 == c0 and c1["pair"] == 0.0
    assert alloc(w, w) == (1.0, 1.0)
    g = rand_vecs(seed=12)
    p_w = c1["v1"] * g[1]
    p_1 = c0["v1"] * g[1]
    assert torch.equal(nd(g[0], p_w, 0.75 * alloc(w, w)[0])[0], nd(g[0], p_1, 0.75)[0])


@needs_T
def test_equal_weight_increase_changes_joint_local_pair_direction():
    """In joint arms a common weight leaves the allocation (hence ||q_i||) unchanged but changes the local/pair mix of
    p_i: the logged weight change acts on the update direction, not on its norm."""
    coef, alloc, nd = _fn("coefficients"), _fn("allocation"), _fn("normalized_direction")
    t, g_loc = rand_vecs(seed=13)
    _, g_pair = rand_vecs(seed=14)
    out = {}
    for w in (1.0, 3.0, 8.0):
        c = coef("joint", (w, w))
        assert abs(c["v1"] + c["v2"] + c["pair"] - 1.0) < 1e-15 and abs(c["pair"] - 1 / (2 * w + 1)) < 1e-15
        q, info = nd(t, c["v1"] * g_loc + c["pair"] * g_pair, 0.75 * alloc(w, w)[0])
        out[w] = q
        assert abs(info["ratio"] - 0.75) < 1e-6
    cosine = float(torch.dot(out[1.0].double(), out[3.0].double()) / (nrm64(out[1.0]) * nrm64(out[3.0])))
    assert cosine < 0.95
    assert all(coef("local", (a, b))["pair"] == 0.0 for a in (0.25, 1, 8) for b in (0.25, 1, 8))


# ------------------------------------------------------------------ 3. controller (section 10)
# Registered rule (additive):  w <- clip(w + c, 0.25, 8),       c = clip((AUC - b) / 0.01, -1, 1).
# Review repair A1 (log form): w <- clip(w * 2**c, 0.25, 8)      (gain ln 2 per unit c in log w; identical first step
#                              from w = 1; same bounds, inner clip, target and frequency).  Engine flag:
#                              smf.train.HP["ctrl_mode"] in {"additive" (default), "log"}.
def _c(auc, b):
    return max(-1.0, min(1.0, (auc - b) / 0.01))


def spec_update(mode, w, auc, b):
    c = _c(auc, b)
    if mode == "log":
        return float(min(8.0, max(0.25, w * 2.0 ** c)))
    return float(min(8.0, max(0.25, w + c)))


def _mode():
    return (T.HP.get("ctrl_mode") or "additive") if T is not None else "additive"


def _ratio(w):
    return w[0] / w[1]


@needs_T
def test_engine_controller_matches_the_registered_mode_spec():
    """The engine's controller_update must equal the registered spec of its declared mode on a grid (catches a sign,
    gain, clip or bound error, and a mode switched without the declared flag)."""
    cu, mode = _fn("controller_update"), _mode()
    assert mode in ("additive", "log")
    for w in (0.25, 0.3, 1.0, 1.7, 2.0, 4.0, 7.5, 8.0):
        for b in (0.5, 0.6, 0.79):
            for d in (-0.05, -0.01, -0.004, 0.0, 0.003, 0.0099, 0.01, 0.02, 0.2):
                got, v = cu(w, b + d, b)
                assert abs(got - spec_update(mode, w, b + d, b)) < 1e-12, (mode, w, b, d)
                assert v == pytest.approx(d, abs=1e-12)


@needs_T
def test_controller_sign_bounds_and_on_target():
    cu = _fn("controller_update")
    assert cu(1.0, 0.803, 0.80)[0] > 1.0 and cu(1.0, 0.797, 0.80)[0] < 1.0        # violation raises, slack lowers
    assert cu(8.0, 0.95, 0.80)[0] == 8.0 and cu(0.25, 0.70, 0.80)[0] == 0.25      # outer bounds [0.25, 8]
    assert cu(1.0, 0.95, 0.80)[0] == 2.0                                          # inner clip: one unit at w = 1
    assert cu(2.0, 0.80, 0.80)[0] == 2.0                                          # exactly on target: unchanged


@needs_T
def test_targets_floor_and_epoch0_unit_step():
    tf, cu, mode = _fn("targets_from"), _fn("controller_update"), _mode()
    rec = {"v1": {"calib_auc": 0.505}, "v2": {"calib_auc": 0.80}}
    b = tf(rec)
    assert b[0] == 0.5 and abs(b[1] - 0.79) < 1e-15
    assert cu(1.0, 0.505, b[0])[0] == pytest.approx(spec_update(mode, 1.0, 0.505, 0.5), abs=1e-12)  # floor: partial
    assert tf({"v1": {"calib_auc": 0.47}, "v2": {"calib_auc": 0.5}})[0] == 0.5
    assert cu(1.0, 0.47, 0.5)[0] < 1.0                                  # below-chance reference: protection relaxed
    for a in np.arange(0.51, 0.999, 0.0007):                            # receipt reuse -> exactly one unit step
        bb = tf({"v1": {"calib_auc": float(a)}, "v2": {"calib_auc": float(a)}})[0]
        assert cu(1.0, float(a), bb)[0] == 2.0


@needs_T
def test_binding_budget_responds_in_one_update_and_differs_materially_in_three():
    cu, alloc = _fn("controller_update"), _fn("allocation")
    b = (0.80, 0.82)
    w = [1.0, 1.0]
    hist = []
    for _ in range(3):                    # recipient 1 persistently violates by 0.015; recipient 2 meets its budget
        w = [cu(w[0], 0.815, b[0])[0], cu(w[1], 0.818, b[1])[0]]
        hist.append((list(w), alloc(*w)))
    assert hist[0][0][0] == 2.0 and hist[0][0][1] < 1.0                # responds within one update
    s = hist[2][1]
    assert s[0] / s[1] >= 2.0 and s[0] > 1.3                            # materially different within three
    w = [1.0, 1.0]                        # common-mode persistent violation: weights move, local allocation does not
    for _ in range(3):
        w = [cu(w[0], 0.815, b[0])[0], cu(w[1], 0.835, b[1])[0]]
    assert w[0] == w[1] > 1.0 and alloc(*w) == (1.0, 1.0)


@needs_T
def test_selectivity_untouched_recipient_keeps_its_weight():
    cu = _fn("controller_update")
    w1, w2 = 1.0, 1.0
    for _ in range(5):
        w1, w2 = cu(w1, 0.83, 0.80)[0], cu(w2, 0.79, 0.79)[0]          # only recipient 1 violates
    assert w2 == 1.0 and w1 > 5.0


def test_additive_rule_allocation_depends_on_common_mode_history():
    """ADVISORY A4 / repair A1 motivation (pure arithmetic, no fit). Under the registered additive rule the SAME
    asymmetric evidence (+1, -1) gives an 8:1 local allocation from w = (1, 1) but only 3:1 after the registered
    epoch-0 common-mode step (w = (2, 2)), and 5:3 after three common-mode steps; from saturation (8, 8) three
    asymmetric updates reach only 1.6. In a local arm common-mode evidence is meant to be inert."""
    def go(w, steps):
        for d in steps:
            w = [spec_update("additive", w[0], 0.80 + d[0], 0.80), spec_update("additive", w[1], 0.80 + d[1], 0.80)]
        return w
    up, asym = (0.02, 0.02), (0.02, -0.02)
    assert _ratio(go([1.0, 1.0], [asym])) == 8.0
    assert _ratio(go([1.0, 1.0], [up, asym])) == 3.0                   # after the designed epoch-0 violation
    assert abs(_ratio(go([1.0, 1.0], [up, up, up, asym])) - 5 / 3) < 1e-12
    assert abs(_ratio(go([8.0, 8.0], [asym] * 3)) - 1.6) < 1e-12
    assert abs(_ratio(go([1.0, 1.0], [asym] * 3)) - 16.0) < 1e-12


def test_log_rule_is_history_independent_inside_bounds_and_keeps_registered_behaviour():
    """Repair A1 spec: allocation depends only on accumulated ASYMMETRIC evidence while no bound binds; the registered
    epoch-0 step (w = 2), the bounds, selectivity and the joint local/pair shift under common-mode evidence are kept."""
    def go(w, steps):
        for d in steps:
            w = [spec_update("log", w[0], 0.80 + d[0], 0.80), spec_update("log", w[1], 0.80 + d[1], 0.80)]
        return w
    up, asym, half = (0.02, 0.02), (0.02, -0.02), (0.005, 0.005)
    r0 = _ratio(go([1.0, 1.0], [asym]))
    assert r0 == 4.0
    for hist in ([up], [up, up], [half, up], [(-0.02, -0.02)]):         # any unsaturated common-mode history
        assert abs(_ratio(go([1.0, 1.0], hist + [asym])) - r0) < 1e-12
    assert go([1.0, 1.0], [up]) == [2.0, 2.0]                           # identical registered epoch-0 step
    assert abs(_ratio(go([8.0, 8.0], [asym] * 3)) - 8.0) < 1e-12       # 5x the additive 1.6 from saturation
    assert go([8.0, 0.25], [(0.05, -0.05)]) == [8.0, 0.25]              # bounds
    w = go([1.0, 1.0], [(0.02, 0.0)] * 4)
    assert w[1] == 1.0                                                  # selectivity
    pair_share = [1 / (2 * x + 1) for x in (go([1.0, 1.0], [up] * k)[0] for k in range(4))]
    assert pair_share == sorted(pair_share, reverse=True) and pair_share[-1] < 0.06   # joint shift kept


@needs_T
def test_windup_documented_controller_authority_after_saturation():
    """Documented engine behaviour: from (8, 8), three asymmetric updates move the allocation ratio to 1.6 under the
    registered additive rule and to 8 under the log repair; from (1, 1) to 16 (additive) and 32 (log)."""
    cu, alloc, mode = _fn("controller_update"), _fn("allocation"), _mode()
    for start, expect in (((8.0, 8.0), {"additive": 1.6, "log": 8.0}), ((1.0, 1.0), {"additive": 16.0, "log": 32.0})):
        w = list(start)
        for _ in range(3):
            w = [cu(w[0], 0.83, 0.80)[0], cu(w[1], 0.77, 0.80)[0]]
        s = alloc(*w)
        assert abs(s[0] / s[1] - expect[mode]) < 1e-9


# ------------------------------------------------------------------ 4. controller probes
def _clue_views(seed, clue_amp=1e-6, shuffle=False, n=4000):
    """Frozen-view-shaped float32 arrays: r (16 dims; 15 nuisance directions of scale 0.5-3, one rotated direction
    carrying 1e-6 (2S - 1) + 1e-7 noise) followed by the centred logits of a fixed affine head (exact-null block)."""
    rng = np.random.default_rng(seed)
    S = (rng.random(n) < 0.67).astype(np.int64)
    base = rng.normal(size=(n, 15)) * np.linspace(3, 0.5, 15)
    clue = clue_amp * (2 * S - 1) + 1e-7 * rng.normal(size=n)
    Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    r = torch.tensor(np.hstack([base, clue[:, None]]) @ Q, dtype=torch.float32)
    W = torch.tensor(rng.normal(size=(2, 16)) * 2, dtype=torch.float32)
    b = torch.tensor(rng.normal(size=2), dtype=torch.float32)
    lg = F.linear(r, W, b)
    V = torch.cat([r, lg - lg.mean(1, keepdim=True)], 1).numpy()
    u = rng.random(n)
    d = types.SimpleNamespace(cf=np.flatnonzero(u < .7), cv=np.flatnonzero((u >= .7) & (u < .85)),
                              cal=np.flatnonzero(u >= .85))
    if shuffle:
        S = rng.permutation(S)
    return V, S, d


@needs_T
@pytest.mark.parametrize("seed", [0, 1])
def test_probe_detects_rotated_1e6_clue(seed):
    """REQUIRED R2. A suitably scaled linear reader (whitened LR on r with a float64 tolerance) reads this clue at
    AUC 1.0; the controller's 'scale-aware' reader must not be blind to it (prompt sections 10 and 12)."""
    pa = _fn("probe_auc")
    V, S, d = _clue_views(seed)
    out = pa(V, S, d, 0, "clue")
    assert out["calib_auc"] > 0.9, out


@needs_T
def test_probe_null_control_and_determinism():
    pa = _fn("probe_auc")
    V, S, d = _clue_views(3, shuffle=True)
    a = pa(V, S, d, 0, "null")
    assert abs(a["calib_auc"] - 0.5) < 0.08
    b = pa(V, S, d, 0, "null")
    assert a["calib_auc"] == b["calib_auc"] and a["selected"] == b["selected"]


@needs_T
def test_probe_orientation_is_chosen_on_critic_val_not_flipped_on_calibration():
    """Signal sign reverses on CONTROLLER_CALIB rows: the reported calibration AUC must be < 0.5 (orientation from
    CRITIC_VAL), never max(AUC, 1 - AUC) on the calibration rows."""
    pa = _fn("probe_auc")
    rng = np.random.default_rng(5)
    n = 3000
    S = (rng.random(n) < 0.6).astype(np.int64)
    u = rng.random(n)
    d = types.SimpleNamespace(cf=np.flatnonzero(u < .7), cv=np.flatnonzero((u >= .7) & (u < .85)),
                              cal=np.flatnonzero(u >= .85))
    sign = np.where(u >= .85, -1.0, 1.0)
    V = rng.normal(size=(n, 18))
    V[:, 0] += 1.5 * sign * (2 * S - 1)
    out = pa(V.astype(np.float32), S, d, 0, "flip")
    assert out["calib_auc"] < 0.3, out


# ------------------------------------------------------------------ 5. matched-count bookkeeping
@needs_T
def test_spread_is_exact_integer_bookkeeping():
    sp = _fn("spread")
    for E, S in ((0, 1220), (1, 1220), (6622, 1220), (1219, 1220), (1221, 1220), (53265, 1220), (7, 3)):
        xs = [sp(E, S, s) for s in range(1, S + 1)]
        assert sum(xs) == E and min(xs) >= 0 and max(xs) - min(xs) <= 1
        assert xs == [sp(E, S, s) for s in range(1, S + 1)]


@needs_T
def test_refit_counts_excludes_diagnostic_refit_and_counts_discarded_restarts():
    rc = _fn("refit_counts")

    def rec(c, r):
        return {"continued": {"updates": c}, "restart": {"updates": r}}
    views = ("v1", "v2", "pair")
    diag = {"refits": [{"epoch": e, "diagnostic_only": e == 20,
                        "receipts": {v: {"kinds": {"A": rec(10 + e, 3), "B": rec(5, 100 + e)}} for v in views}}
                       for e in (0, 4, 8, 12, 16, 20)]}
    out = rc(diag)
    assert out["v1|A"] == sum(10 + e + 3 for e in (0, 4, 8, 12, 16))
    assert out["pair|B"] == sum(5 + 100 + e for e in (0, 4, 8, 12, 16))


# ------------------------------------------------------------------ 6. integration through train_run (synthetic)
@needs_T
@pytest.mark.parametrize("base,sched", [("joint", "ONLINE"), ("local", "REFRESHED"), ("joint", "ONLINE_MATCHED"),
                                        ("local", "ONLINE_MATCHED")])
def test_rho_zero_reproduces_task_only_continuation_bitwise(world, base, sched):
    cnt = {f"{v}|{k}": 4 for v in ("v1", "v2", "pair") for k in ("A", "B")} if sched == "ONLINE_MATCHED" else None
    mt, *_ = run(world, task_spec(), 0.0, n_epochs=2)
    m, d, *_ = run(world, {"base": base, "schedule": sched}, 0.0, n_epochs=2, matched_counts=cnt)
    assert same_state(m.state_dict(), mt.state_dict())
    assert d["critic_online_updates"] > 0                                 # critics really ran


@needs_T
def test_rho_zero_feedback_arm_with_real_controller_probes_is_task_only(world):
    d = world["data"]
    m0 = T.Model(T.D_IN, T.KS, 0)
    m0.load_state_dict(world["warm"])
    rec0 = T.probe_receipt(m0, d, world["head"], 0, "ref")
    ctrl = {"receipt0": rec0, "b": T.targets_from(rec0)}
    mt, *_ = run(world, task_spec(), 0.0, n_epochs=2, stage="B")
    m, dg, *_ = run(world, {"base": "joint", "schedule": "ONLINE", "feedback": True}, 0.0, n_epochs=2, stage="B",
                    controller=ctrl)
    assert same_state(m.state_dict(), mt.state_dict())
    assert len(dg["controller"]) == 3 and dg["controller"][0]["reused_reference"]


@needs_T
def test_local_arm_has_no_coalition_gradient_despite_trained_shadow_pair_bank(world):
    out = {}
    for base in ("local", "joint"):
        for pert in (False, True):
            m, dg, ck, cap, fin = run(world, {"base": base, "schedule": "ONLINE"}, 1.5,
                                      init_critics=fresh_init_critics(perturb_pair=pert))
            out[(base, pert)] = (m.state_dict(), fin)
    assert same_state(out[("local", False)][0], out[("local", True)][0])        # pair bank never reaches the encoder
    assert not same_state(out[("joint", False)][0], out[("joint", True)][0])    # ... but it does in the joint arm
    init_pair = fresh_init_critics()["critics"]["pair"]["A"]
    trained_pair = out[("local", False)][1]["theta_T_minus_1"]["critics"]["pair"]["A"]
    assert any(not torch.equal(init_pair[k], trained_pair[k]) for k in init_pair)   # the shadow bank IS trained


@needs_T
def test_heads_receive_task_gradient_only(world):
    mt, _, _, capt, _ = run(world, task_spec(), 0.0, capture_steps=(2,))
    mj, _, _, capj, _ = run(world, {"base": "joint", "schedule": "ONLINE"}, 1.5, capture_steps=(2,))
    a, b = capt[2]["model"], capj[2]["model"]                  # theta after step 1
    assert all(torch.equal(a[k], b[k]) for k in a if k.startswith("head."))
    assert any(not torch.equal(a[k], b[k]) for k in a if k.startswith("enc."))


@needs_T
def test_global_clip_scales_encoders_and_heads_together(world, monkeypatch):
    monkeypatch.setitem(T.HP, "clip", 1e-3)
    m, dg, _, cap, _ = run(world, {"base": "joint", "schedule": "ONLINE"}, 1.5, capture_steps=(2,))
    a, b = world["warm"], cap[2]["model"]
    keys = [k for k in a if k.startswith(("enc.", "head."))]
    delta = math.sqrt(sum(float(((b[k] - a[k]).double() ** 2).sum()) for k in keys))
    assert dg["clip_hits"] == world["steps"]
    assert abs(delta - T.HP["sgd_lr"] * 1e-3) < 1e-3 * T.HP["sgd_lr"] * 1e-3
    assert abs(dg["epochs"][0]["ratio_mean_1"] - 1.5) < 1e-5      # uniform clip: the relative ratio is preserved


@needs_T
def test_zero_direction_events_are_logged_when_the_constant_wins_on_every_view(world, monkeypatch):
    """REQUIRED R1. Critics that are worse than the constant on every batch make every R_v an exact constant with no
    gradient, so q_i = 0 on every step. Those steps must be counted as zero-direction events (and the achieved ratio
    reported as 0 or the zero fraction disclosed), not silently skipped."""
    from rgj import train as RT
    monkeypatch.setitem(T.HP, "critic_lr", 0.0)
    ic = fresh_init_critics()
    for v in RT.VIEWS:
        for k in RT.KINDS:
            sd = ic["critics"][v][k]
            last_w = [kk for kk in sd if kk.endswith("weight")][-1]
            last_b = [kk for kk in sd if kk.endswith("bias")][-1]
            sd[last_w] = torch.zeros_like(sd[last_w])
            sd[last_b] = torch.log(torch.tensor([0.9, 0.1]))          # wrong-way constant: worse than the prior
    m, dg, *_ = run(world, {"base": "local", "schedule": "ONLINE"}, 0.75, init_critics=ic)
    S = world["steps"]
    assert dg["zero_events"] == [S, S], dg["zero_events"]


@needs_T
def test_gradient_archive_has_protection_proxy_and_post_clip_norms(world):
    """REQUIRED R3 (section 7: archive pre- and post-clipping task and protection norms, actual update norms, caps)."""
    _, dg, *_ = run(world, {"base": "joint", "schedule": "ONLINE"}, 0.75)
    e = dg["epochs"][0]
    keys = list(e)
    assert any("q_norm" in k for k in keys), keys
    assert any("p_norm" in k for k in keys), keys
    assert any(("post_clip" in k and k.endswith(("_1", "_2"))) or "clip_factor" in k for k in keys), keys


@needs_T
def test_checkpoints_save_critic_optimizer_state_and_hypothetical_weights(world, monkeypatch):
    """REQUIRED R4 (section 8: save theta, critic state, critic-view head, transform and optimizer state together)."""
    monkeypatch.setattr(T, "probe_receipt", lambda *a, **k: {"v1": {"calib_auc": 0.7, "selected": "x"},
                                                           "v2": {"calib_auc": 0.7, "selected": "x"}})
    ctrl = {"receipt0": {"v1": {"calib_auc": 0.72, "selected": "x"}, "v2": {"calib_auc": 0.7, "selected": "x"}},
            "b": [0.7, 0.7]}
    _, _, ck, *_ = run(world, {"base": "local", "schedule": "ONLINE", "feedback": False}, 0.75, stage="B",
                       ckpt_epochs=(1,), controller=ctrl)
    keys = list(ck[1])
    assert any("opt" in k for k in keys), keys
    assert any("hyp" in k for k in keys), keys


def _scripted(monkeypatch, script):
    """Replace controller probe fits by scripted receipts: script[ep] = (auc1, auc2)."""
    seen = []

    def fake(model, data, head, seed, tag):
        ep = int(str(tag).split("|")[-1])
        seen.append((ep, hashlib.sha256(b"".join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()))
        a = script[ep]
        return {"v1": {"calib_auc": a[0], "selected": "scripted"}, "v2": {"calib_auc": a[1], "selected": "scripted"}}
    monkeypatch.setattr(T, "probe_receipt", fake)
    return seen


def _ctrl(a0, b=(0.70, 0.70)):
    return {"receipt0": {"v1": {"calib_auc": a0[0], "selected": "ref"}, "v2": {"calib_auc": a0[1], "selected": "ref"}},
            "b": list(b)}


@needs_T
def test_asymmetric_violation_changes_allocation_and_actual_updates(world, monkeypatch):
    calls = []
    orig = T.normalized_direction

    def spy(t, p, rho_i, *a, **k):
        calls.append((p.detach().clone(), rho_i))
        return orig(t, p, rho_i, *a, **k)
    _scripted(monkeypatch, {1: (0.70, 0.70)})
    monkeypatch.setattr(T, "normalized_direction", spy)
    res = {}
    for arm, base, fb in (("L-F", "local", True), ("L-N", "local", False), ("J-F", "joint", True), ("J-N", "joint", False)):
        calls.clear()
        m, dg, *_ = run(world, {"base": base, "schedule": "ONLINE", "feedback": fb}, 0.75, stage="B",
                        controller=_ctrl((0.72, 0.70)))
        res[arm] = {"m": m.state_dict(), "d": dg, "p_step1": [calls[0][0], calls[1][0]], "rho_i": [calls[0][1], calls[1][1]]}
    lf, ln = res["L-F"]["d"], res["L-N"]["d"]
    assert lf["controller"][0]["w_after"] == [2.0, 1.0] and ln["controller"][0]["w_after"] == [1.0, 1.0]
    assert ln["controller"][0]["w_hypothetical"] == [2.0, 1.0]
    assert lf["controller"][0]["w_after"][1] == 1.0                                   # untouched recipient unchanged
    s1, s2 = 2 / math.sqrt(2.5), 1 / math.sqrt(2.5)
    e = lf["epochs"][0]
    assert abs(e["ratio_mean_1"] - 0.75 * s1) < 1e-5 and abs(e["ratio_mean_2"] - 0.75 * s2) < 1e-5
    assert abs(math.sqrt((e["ratio_mean_1"] ** 2 + e["ratio_mean_2"] ** 2) / 2) - 0.75) < 1e-5
    assert abs(ln["epochs"][0]["ratio_mean_1"] - 0.75) < 1e-5
    assert not same_state(res["L-F"]["m"], res["L-N"]["m"])

    def cos(a, b):
        return float(torch.dot(a.double(), b.double()) / (nrm64(a) * nrm64(b)))
    # step 1: local -> same p_1 direction, different strength; joint -> direction changes (local/pair mix 2:1 vs 1:1)
    assert cos(res["L-F"]["p_step1"][0], res["L-N"]["p_step1"][0]) > 1 - 1e-6
    assert res["L-F"]["rho_i"][0] > res["L-N"]["rho_i"][0] > res["L-F"]["rho_i"][1]
    assert cos(res["J-F"]["p_step1"][0], res["J-N"]["p_step1"][0]) < 1 - 1e-6


@needs_T
def test_common_mode_feedback_is_an_exact_alias_in_local_but_not_joint_arms(world, monkeypatch):
    _scripted(monkeypatch, {1: (0.75, 0.75)})
    st = {}
    for arm, base, fb in (("L-F", "local", True), ("L-N", "local", False), ("J-F", "joint", True), ("J-N", "joint", False)):
        m, dg, *_ = run(world, {"base": base, "schedule": "ONLINE", "feedback": fb}, 0.75, stage="B",
                        controller=_ctrl((0.72, 0.72)))
        st[arm] = (m.state_dict(), dg)
    assert st["L-F"][1]["controller"][0]["w_after"] == [2.0, 2.0]
    assert same_state(st["L-F"][0], st["L-N"][0])               # registered symmetry: logged weights change, update not
    assert not same_state(st["J-F"][0], st["J-N"][0])           # joint: the local/pair mix changes (1/3 -> 1/5 pair)
    for i in (1, 2):
        assert abs(st["J-F"][1]["epochs"][0][f"ratio_mean_{i}"] - 0.75) < 1e-5   # ... at unchanged strength


@needs_T
def test_controller_schedule_snapshot_pairing_and_diagnostic_final(world, monkeypatch):
    seen = _scripted(monkeypatch, {1: (0.75, 0.70), 2: (0.75, 0.70), 3: (0.60, 0.60)})
    m, dg, ck, *_ = run(world, {"base": "local", "schedule": "ONLINE", "feedback": True}, 0.75, n_epochs=3, stage="B",
                        controller=_ctrl((0.72, 0.70)), ckpt_epochs=(1, 2, 3))
    C = dg["controller"]
    assert [c["epoch"] for c in C] == [0, 1, 2, 3]
    assert C[0]["reused_reference"] and [s[0] for s in seen] == [1, 2, 3]     # epoch 0 never refits a probe
    applied = [c for c in C if not c["diagnostic_only"]]
    assert len(applied) == 3 and C[3]["diagnostic_only"]
    assert C[3]["w_after"] == C[3]["w_before"]                                  # the final measurement cannot act
    for ep, h in seen:                                                           # probe sees theta after ep epochs
        ref = hashlib.sha256(b"".join(v.numpy().tobytes() for v in ck[ep]["model"].values())).hexdigest()
        assert h == ref
    for e in dg["epochs"]:                                                       # weights used in epoch e = after update e
        assert e["w"] == C[e["epoch"]]["w_after"]


@needs_T
def test_online_matched_zero_template_is_bitwise_online(world):
    z = {f"{v}|{k}": 0 for v in ("v1", "v2", "pair") for k in ("A", "B")}
    m0, d0, *_ = run(world, {"base": "local", "schedule": "ONLINE"}, 0.75)
    m1, d1, *_ = run(world, {"base": "local", "schedule": "ONLINE_MATCHED"}, 0.75, matched_counts=z)
    assert same_state(m0.state_dict(), m1.state_dict()) and d1["critic_matched_extra_updates"] == 0


@needs_T
def test_online_matched_extra_updates_are_honoured_per_bank(world, monkeypatch):
    counts = {"v1|A": 5, "v1|B": 11, "v2|A": 0, "v2|B": 3, "pair|A": 7, "pair|B": 1}
    made = []
    Base = torch.optim.Adam

    class CountingAdam(Base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.n_steps = 0
            made.append(self)

        def step(self, *a, **k):
            self.n_steps += 1
            return super().step(*a, **k)
    monkeypatch.setattr(torch.optim, "Adam", CountingAdam)
    _, dg, *_ = run(world, {"base": "joint", "schedule": "ONLINE_MATCHED"}, 0.75, matched_counts=counts)
    S = world["steps"]
    assert len(made) == 6                                     # one Adam per critic (VIEWS x KINDS order)
    order = [f"{v}|{k}" for v in ("v1", "v2", "pair") for k in ("A", "B")]
    assert [o.n_steps for o in made] == [5 * S + counts[b] for b in order]
    assert dg["critic_matched_extra_updates"] == sum(counts.values())


@needs_T
def test_refreshed_template_receipt_matches_refit_updates(world):
    m, dg, *_ = run(world, {"base": "local", "schedule": "REFRESHED"}, 0.25, n_epochs=4)
    cnt = T.refit_counts(dg)
    assert sum(cnt.values()) == dg["critic_refit_updates"] > 0
    diag_only = [r for r in dg["refits"] if r["diagnostic_only"]]
    assert [r["epoch"] for r in diag_only] == [4]


# ------------------------------------------------------------------ 7. selection rules (PROTOCOL 4, 6)
@pytest.fixture
def real_pkg_guard():
    def snap():
        if not REAL_PKG.exists():
            return {}
        return {str(p.relative_to(REAL_PKG)): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in REAL_PKG.rglob("*") if p.is_file()}
    before = snap()
    yield
    assert snap() == before, "a selection fixture wrote into the real results folder"


@pytest.fixture
def selworld(tmp_path, monkeypatch, real_pkg_guard):
    try:
        from smf import run as R
        from smf import select as SEL
    except Exception as e:  # pragma: no cover
        pytest.skip(f"smf.run/select not importable: {e}")
    run_dir, pkg = tmp_path / "run", tmp_path / "pkg"
    run_dir.mkdir()
    pkg.mkdir()
    monkeypatch.setattr(R, "RUN", run_dir)
    monkeypatch.setattr(R, "PKG", pkg)
    monkeypatch.setattr(R, "UNITS", run_dir / "units")
    recs = {}
    monkeypatch.setattr(R, "rec", lambda name: recs[name])
    return R, SEL, recs, run_dir, pkg


def _inner(recs, unit, pair, v1, v2, acc=(0.84, 0.48), const=(0.73, 0.29)):
    recs[f"inner__{unit}"] = {"recovery": {"auc": {"pair": pair, "v1": v1, "v2": v2}},
                              "utility": {"0": {"acc": acc[0], "const_acc": const[0]},
                                          "1": {"acc": acc[1], "const_acc": const[1]}}}


def _phase_a_world(R, recs, feasible):
    """feasible[(sched, seed, rho)] -> (pair AUC, task-feasible?). Local points: worse-local set by rho."""
    for k in R.SEEDS:
        _inner(recs, f"tl__s{k}__e20", 0.88, 0.86, 0.87)
        for s in R.SCHEDS:
            for r in R.RHOS:
                pair, ok = feasible.get((s, k, r), (0.85, True))
                acc = (0.84, 0.48) if ok else (0.80, 0.40)
                _inner(recs, R.a_name(k, "NJ", s, r), pair, 0.80, 0.82, acc=acc)
                _inner(recs, R.a_name(k, "NL", s, r), 0.86, 0.80 - 0.01 * r, 0.83 - 0.02 * r, acc=acc)
                recs[R.a_run(k, "NJ", s, r)] = {"diag": {"critic_online_updates": 100, "critic_refit_updates":
                                                         (50 if s == "REFRESHED" else 0),
                                                         "critic_matched_extra_updates": (50 if s == "ONLINE_MATCHED" else 0)}}


def test_phase_a_schedule_rule_excludes_incomplete_schedules_and_ties_lower_rho(selworld):
    R, SEL, recs, run_dir, pkg = selworld
    f = {}
    for k in R.SEEDS:
        f[("REFRESHED", k, 1.5)] = (0.80, True)
        f[("REFRESHED", k, 0.75)] = (0.80, True)          # tie with 1.5 -> lower rho must be chosen
        f[("ONLINE", k, 1.5)] = (0.81, True)
    f[("ONLINE_MATCHED", 0, 1.5)] = (0.70, True)          # best AUC but infeasible everywhere on seed 2
    for r in R.RHOS:
        f[("ONLINE_MATCHED", 2, r)] = (0.70, False)
    _phase_a_world(R, recs, f)
    out = SEL.select_A(None)
    assert out["schedule"]["selected"] == "REFRESHED" and out["schedule"]["status"] == "SELECTED"
    assert "ONLINE_MATCHED" not in out["schedule"]["complete_schedules"]
    assert all(out["per_schedule"]["REFRESHED"][k]["selected"]["rho"] == 0.75 for k in R.SEEDS)
    for k in R.SEEDS:                                      # local reference: lowest worse-local AUC -> rho 1.5
        assert out["seeds"][str(k)]["reference"]["rho"] == 1.5
        assert out["seeds"][str(k)]["reference"]["status"] == "NOMINEE"
    assert (pkg / "PHASE_A_SELECTION.json").exists() and (run_dir / "selection_A.json").exists()


def test_phase_a_fallback_minimises_summed_smallest_shortfall(selworld):
    R, SEL, recs, *_ = selworld
    f = {(s, k, r): (0.85, False) for s in R.SCHEDS for k in R.SEEDS for r in R.RHOS}
    _phase_a_world(R, recs, f)
    # ONLINE gets a smaller shortfall on seed 0 (occupation 0.47 instead of 0.40)
    _inner(recs, R.a_name(0, "NJ", "ONLINE", 0.25), 0.85, 0.80, 0.82, acc=(0.84, 0.47))
    out = SEL.select_A(None)
    assert out["schedule"]["status"] == "FALLBACK" and out["schedule"]["selected"] == "ONLINE"


def test_phase_a_local_reference_falls_back_to_U_alias_and_no_valid_reference(selworld):
    R, SEL, recs, *_ = selworld
    _phase_a_world(R, recs, {})
    for k in R.SEEDS:
        for r in R.RHOS:                                    # no feasible local point on any seed
            _inner(recs, R.a_name(k, "NL", "ONLINE", r), 0.86, 0.80, 0.82, acc=(0.70, 0.30))
    _inner(recs, "tl__s2__e20", 0.88, 0.86, 0.87, acc=(0.75, 0.30))      # U lacks a 3-point gain on seed 2
    out = SEL.select_A(None)
    assert out["schedule"]["selected"] == "ONLINE"          # all schedules tie on AUC; lowest compute wins
    assert out["seeds"]["0"]["reference"]["status"] == "TASK_ONLY_ALIAS"
    assert out["seeds"]["0"]["reference"]["unit"] == "tl__s0__e20"
    assert out["seeds"]["2"]["reference"]["status"] == "NO_VALID_REFERENCE"


def _phase_b_world(R, SEL, recs, monkeypatch, ctrl, jf, fare=None):
    """ctrl[arm] = {rho_or_beta: (pair, v1, v2, feasible)}; jf = {rho: (pair, v1, v2, feasible)}."""
    (R.RUN / "selection_A.json").write_text(json.dumps(
        {"seeds": {str(k): {"reference": {"status": "NOMINEE", "unit": "x", "rho": 0.75}} for k in R.SEEDS}}))
    good, bad = (0.84, 0.48), (0.80, 0.40)
    for k in R.SEEDS:
        _inner(recs, f"tl__s{k}__e40", 0.88, 0.86, 0.87)
        _inner(recs, f"lc__s{k}__E", *ctrl.get("E", (0.86, 0.85, 0.86, False))[:3],
               acc=good if ctrl.get("E", (0, 0, 0, False))[3] else bad)
        for arm in ("L-F", "J-N", "L-N"):
            for r in R.RHOS:
                pair, v1, v2, ok = ctrl[arm].get(r, (0.86, 0.84, 0.85, False))
                _inner(recs, R.b_name(k, arm, r), pair, v1, v2, acc=good if ok else bad)
        for arm in ("RAW-J", "RAW-L"):
            for b in R.RAW_BETAS:
                pair, v1, v2, ok = ctrl[arm].get(b, (0.86, 0.84, 0.85, False))
                _inner(recs, f"raw__s{k}__{arm}__b{R.g(b)}__e40", pair, v1, v2, acc=good if ok else bad)
        for r in R.RHOS:
            pair, v1, v2, ok = jf.get(r, (0.86, 0.84, 0.85, False))
            _inner(recs, R.b_name(k, "J-F", r), pair, v1, v2, acc=good if ok else bad)
    fare = fare or {"F": (0.70, 0.68, 0.63, False), "F0": (0.87, 0.78, 0.86, True)}

    def select_fare(k, D, units, uref, *a, **kw):
        return {a_: {"status": "NOMINEE" if v[3] else "NO_FEASIBLE_NOMINEE", "units": [f"{a_}{k}"], "gates_ok": v[3],
                     "auc": {"pair": v[0], "v1": v[1], "v2": v[2]}} for a_, v in fare.items()} | {"per_purpose": None}
    fake = types.ModuleType("smf.baselines")
    fake.select_fare = select_fare
    import smf
    monkeypatch.setitem(sys.modules, "smf.baselines", fake)
    monkeypatch.setattr(smf, "baselines", fake, raising=False)


def test_phase_b_cstar_has_no_local_guard_and_includes_task_only_and_raw(selworld, monkeypatch):
    R, SEL, recs, run_dir, pkg = selworld
    ctrl = {"L-F": {0.75: (0.84, 0.80, 0.82, True)}, "J-N": {1.5: (0.83, 0.79, 0.84, True)},
            "L-N": {0.25: (0.85, 0.81, 0.82, True)},
            "RAW-J": {0.3: (0.815, 0.77, 0.80, True)},            # lowest pair, very low locals
            "RAW-L": {0.3: (0.82, 0.74, 0.81, True)}}
    jf = {0.25: (0.79, 0.774, 0.804, True),                       # within +0.005 of RAW-J's locals: eligible
          0.75: (0.78, 0.776, 0.806, True),                       # v1 0.776 > 0.770 + 0.005: not eligible
          1.5: (0.76, 0.80, 0.84, False)}                         # task-infeasible
    _phase_b_world(R, SEL, recs, monkeypatch, ctrl, jf)
    out = SEL.select_B(None)
    for k in R.SEEDS:
        o = out[str(k)]
        assert o["comparator"]["arm"] == "RAW-J"
        assert o["arms"]["J-F"]["status"] == "NOMINEE" and o["arms"]["J-F"]["rho"] == 0.25
        assert set(o["arms"]["J-F"]["nomination_guards"]) == {"L-F", "C*"}
    assert (pkg / "SEED_STATUS.json").exists() and (pkg / "SELECTION_TABLE.csv").exists()


def test_phase_b_task_only_U_can_be_cstar_and_descriptive_fallback(selworld, monkeypatch):
    R, SEL, recs, *_ = selworld
    ctrl = {"L-F": {0.75: (0.90, 0.80, 0.82, True)}, "J-N": {}, "L-N": {}, "RAW-J": {}, "RAW-L": {}}
    jf = {0.25: (0.85, 0.83, 0.86, True), 0.75: (0.84, 0.95, 0.95, True), 1.5: (0.80, 0.80, 0.82, False)}
    _phase_b_world(R, SEL, recs, monkeypatch, ctrl, jf, fare={"F": (0.70, 0.6, 0.6, False), "F0": (0.95, 0.9, 0.9, False)})
    out = SEL.select_B(None)
    o = out["0"]
    assert o["comparator"]["arm"] == "U" and o["comparator"]["is_task_only_release"]
    assert o["arms"]["J-F"]["status"] == "NO_FEASIBLE_NOMINEE"
    # descriptive fallback: smallest summed shortfall (rho 0.25: v1 0.83 vs 0.80+0.005, v2 0.86 vs 0.82+0.005)
    assert o["arms"]["J-F"]["rho"] == 0.25


# ------------------------------------------------------------------ 8. endpoint family
def test_z_value_for_18_slot_family_and_bootstrap_seed():
    z = norm.ppf(1 - 0.05 / (2 * 18))
    assert round(float(z), 6) == 2.991316 and abs(z - norm.isf(0.05 / 36)) < 1e-12
    assert norm.ppf(1 - 0.05 / 18) < z                    # a 9-slot or one-sided family would shrink z
    try:
        from smf import family as FAM
    except Exception:
        pytest.skip("smf.family not importable")
    assert FAM.PRIMARY_SIZE == 18 and len(FAM.PRIMARY) == 18
    assert abs(FAM.Z_PRIMARY - z) < 1e-12
    assert FAM.B == 1999 and FAM.BOOT_SEED == 20261005
    claims = {c: [e for e in FAM.PRIMARY if e["claim"] == c] for c in ("A", "B")}
    assert len(claims["A"]) == 9 and len(claims["B"]) == 9
    assert sum(1 for e in claims["B"] if e.get("alias_of")) == 6
    st = {k: {"valid_reference": True, "J-F": "NOMINEE", "L-F": "NOMINEE", "C*": "RAW-J"} for k in (0, 1, 2)}
    allpass = {e["id"]: "PASS" for e in FAM.PRIMARY}
    assert FAM.claim_decision("A", allpass, st)["decision"] == "PASS"
    st2 = dict(st)
    st2[1] = dict(st[1], **{"L-F": "NO_FEASIBLE_NOMINEE"})
    assert FAM.claim_decision("A", allpass, st2)["decision"] == "NOT_ESTABLISHED"
    assert FAM.claim_decision("B", allpass, st2)["decision"] == "PASS"
    one_fail = dict(allpass, P02="NOT_ESTABLISHED")
    assert FAM.claim_decision("A", one_fail, st)["decision"] == "NOT_ESTABLISHED"
