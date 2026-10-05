"""Design / math review fixtures for the online strength-frontier study (owned by the design reviewer).

Independent of osf/tests/test_pipeline.py: own synthetic data (SEX prior ~0.67, model seed 3), own functional forward
(encoders, fixed-head critic views, critics, transforms, recovery surrogate, task losses), own update algebra. The engine
is only used to produce trajectories / snapshots; every expected value below is recomputed here from first principles.

    cd <WORKTREE> && OMP_NUM_THREADS=1 PYTHONPATH=. <venv>/python -m pytest osf/tests/test_math_review.py -q

Mutation testing: OSF_REVIEW_TRAIN / OSF_REVIEW_DATA name an alternative module (a mutated copy) to test instead of
osf.train / osf.data; see MATH_REVIEW.md, "Injected defects".
"""
from __future__ import annotations

import importlib
import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

T = importlib.import_module(os.environ.get("OSF_REVIEW_TRAIN", "osf.train"))
OD = importlib.import_module(os.environ.get("OSF_REVIEW_DATA", "osf.data"))
SELM = os.environ.get("OSF_REVIEW_SELECT", "osf.select")        # later modules: imported lazily inside the fixtures
FAMM = os.environ.get("OSF_REVIEW_FAMILY", "osf.family")
INFM = os.environ.get("OSF_REVIEW_INFER", "osf.infer")
from rgj import train as RT  # noqa: E402  (pinned source: used only to build TData and as a fidelity reference)

torch.set_num_threads(1)
SEED = 3
ENC_LAYERS = ("0", "2", "4")
CRIT_LAYERS = {"A": ("0", "2"), "B": ("0", "2", "4")}
VIEWS = ("v1", "v2", "pair")


# ------------------------------------------------------------------ synthetic data (no study rows)
def synth_D(n_fit=820, n_other=80, seed=7, leak=1.5):
    n = n_fit + n_other
    rng = np.random.default_rng(seed)
    S = (rng.random(n) < 0.67).astype(np.int64)
    X = rng.normal(size=(n, 83)).astype(np.float32)
    X[:, :5] += (leak * (2 * S[:, None] - 1) * np.array([1.0, 0.7, 0.5, 0.3, 0.2])).astype(np.float32)
    inc = ((X[:, 10] + 0.5 * X[:, 0] + 0.3 * rng.normal(size=n)) > 0.2).astype(np.int64)
    occ = np.digitize(X[:, 11] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n), [-1.0, -0.4, 0.0, 0.4, 1.0]).astype(np.int64)
    role = np.array(["DEFENSE_FIT"] * n_fit + ["AUDIT_FIT"] * n_other, dtype="<U32")
    sub = np.full(n, "", dtype="<U32")
    u = rng.random(n_fit)
    sub[:n_fit] = np.where(u < 0.7, "CRITIC_FIT", np.where(u < 0.85, "CRITIC_VAL", "CALIB"))
    D = {"X": X, "sex": S, "y": {"income": inc, "occupation_group": occ}, "role": role, "subrole": sub}
    D["idx"] = {"DEFENSE_FIT": np.flatnonzero(role == "DEFENSE_FIT"), "AUDIT_FIT": np.flatnonzero(role == "AUDIT_FIT")}
    D["idx"].update({r: np.flatnonzero(sub == r) for r in ("CRITIC_FIT", "CRITIC_VAL", "CALIB")})
    return D


@pytest.fixture(scope="module")
def syn():
    D = synth_D()
    data = RT.TData(D)
    warm = T.Model(T.D_IN, T.KS, SEED).state_dict()
    warm = {k: v.clone() for k, v in warm.items()}
    head = {i: (warm[f"head.{i}.weight"].clone(), warm[f"head.{i}.bias"].clone()) for i in (0, 1)}
    prior = np.bincount(D["sex"][D["idx"]["DEFENSE_FIT"]], minlength=2) / len(D["idx"]["DEFENSE_FIT"])
    return {"D": D, "data": data, "warm": warm, "head": head, "prior": prior,
            "S_per_ep": math.ceil(data.n / 256)}


def run(syn, c, ep=1, **kw):
    head = syn["head"] if c["mode"] != "TASK" else None
    kw.setdefault("ckpt_epochs", (ep,))
    return T.train_run(c, syn["warm"], syn["data"], SEED, head, n_epochs=ep, **kw)


def same_state(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def RAW(t, b):
    return {"mode": "RAW", "treat": t, "beta": b}


def NORM(t, r, a=1.0):
    return {"mode": "NORM", "treat": t, "rho": r, "a": a}


TASK = {"mode": "TASK", "treat": None}


# ------------------------------------------------------------------ independent functional reference
def _mlp(sd, prefix, x, layers):
    for j, l in enumerate(layers):
        x = F.linear(x, sd[f"{prefix}{l}.weight"], sd[f"{prefix}{l}.bias"])   # same primitive as nn.Linear
        if j < len(layers) - 1:
            x = torch.relu(x)
    return x


def enc_keys(i):
    return [f"enc.{i}.{l}.{w}" for l in ENC_LAYERS for w in ("weight", "bias")]


def head_keys(i):
    return [f"head.{i}.weight", f"head.{i}.bias"]


def my_alloc(a):
    return a / math.sqrt((a * a + 1) / 2), 1 / math.sqrt((a * a + 1) / 2)


def unit_weights(treat):
    return {"v1": 1 / 3, "v2": 1 / 3, "pair": 1 / 3} if treat == "J" else {"v1": 0.5, "v2": 0.5, "pair": 0.0}


def my_gradients(syn, model_sd, critics, transforms, head, b, treat):
    """t_i (encoder i, task loss i only), task-head gradients, p_i = d P_unit / d enc_i, R per view, from scratch."""
    data = syn["data"]
    P = {k: v.detach().clone().requires_grad_(True) for k, v in model_sd.items()}
    bt = torch.from_numpy(np.asarray(b))
    X, S = data.X[bt], data.S[bt]
    r = [_mlp(P, f"enc.{i}.", X, ENC_LAYERS) for i in (0, 1)]
    t, th = [], []
    for i in (0, 1):
        lg = F.linear(r[i], P[f"head.{i}.weight"], P[f"head.{i}.bias"])
        Li = -torch.log_softmax(lg, 1)[torch.arange(len(bt)), data.Y[i][bt]].mean()
        gs = torch.autograd.grad(Li, [P[k] for k in enc_keys(i) + head_keys(i)], retain_graph=True)
        t.append(torch.cat([g.reshape(-1) for g in gs[:6]]))
        th.append(torch.cat([g.reshape(-1) for g in gs[6:]]))
    prior = syn["prior"]
    H = float(-(prior * np.log(prior)).sum())
    logprior = torch.tensor(np.log(prior), dtype=torch.float32)
    v = {}
    for i in (0, 1):
        lg = F.linear(r[i], head[i][0], head[i][1])
        v[i] = torch.cat([r[i], lg - lg.mean(1, keepdim=True)], 1)
    V = {"v1": v[0], "v2": v[1], "pair": torch.cat([v[0], v[1]], 1)}
    w = unit_weights(treat)
    Punit, Rs = 0.0, {}
    for vw in VIEWS:
        if w[vw] == 0:
            continue
        mu, W = transforms[vw]["mu"].float(), transforms[vw]["W"].float()
        Z = (V[vw] - mu) @ W
        ces = []
        for k in ("A", "B"):
            lg = _mlp(critics[vw][k], "", Z, CRIT_LAYERS[k])
            ces.append(-torch.log_softmax(lg, 1)[torch.arange(len(bt)), S].mean())
        ce_c = -(logprior[S]).mean()
        allc = ces + [ce_c]
        j = int(np.argmin([float(x) for x in allc]))
        R = (ce_c - allc[j]) / H
        Rs[vw] = float(R)
        Punit = Punit + w[vw] * R
    p = []
    for i in (0, 1):
        if isinstance(Punit, torch.Tensor) and Punit.requires_grad:
            gs = torch.autograd.grad(Punit, [P[k] for k in enc_keys(i)], retain_graph=True, allow_unused=True)
            p.append(torch.cat([(g if g is not None else torch.zeros_like(P[k])).reshape(-1)
                                for g, k in zip(gs, enc_keys(i))]))
        else:
            p.append(torch.zeros_like(t[i]))
    return t, th, p, Rs


def my_q(c, t, p, a_max=100.0, zero_tol=1e-12):
    """Registered q_i: RAW beta p_i; NORM stop_grad(rho s_i ||t_i|| / ||p_i||) p_i (float64, cap, zero rule)."""
    out, caps, zeros = [], [False, False], [False, False]
    if c["mode"] == "RAW":
        return [c["beta"] * p[i].double() for i in (0, 1)], caps, [bool(p[i].abs().max() == 0) for i in (0, 1)]
    s = my_alloc(c["a"])
    for i in (0, 1):
        nt, npn = float(t[i].double().norm()), float(p[i].double().norm())
        if npn <= zero_tol or nt == 0.0:
            out.append(torch.zeros_like(p[i], dtype=torch.float64))
            zeros[i] = True
            continue
        a = c["rho"] * s[i] * nt / npn
        if a > a_max:
            a, caps[i] = a_max, True
        out.append(a * p[i].double())
    return out, caps, zeros


def predicted_delta(t, th, q, lr=0.05, clip=5.0):
    u_enc = [-(t[i].double() + q[i]) for i in (0, 1)]
    u_head = [-th[i].double() for i in (0, 1)]
    nrm = math.sqrt(sum(float((x * x).sum()) for x in u_enc + u_head))
    kappa = min(1.0, clip / nrm)
    return [lr * kappa * x for x in u_enc], [lr * kappa * x for x in u_head], kappa, nrm


def flat(sd, keys):
    return torch.cat([sd[k].reshape(-1).double() for k in keys])


def batch_of(syn, step, salt=0):
    per = syn["S_per_ep"]
    ep, pos = (step - 1) // per, (step - 1) % per
    perm = np.random.default_rng([SEED, salt, ep]).permutation(syn["data"].n)
    return perm[pos * 256:(pos + 1) * 256]


def check_step(syn, c, before, after_model, rec, step, clip=5.0, a_max=100.0):
    """Reconstruct step `step` from the aligned snapshot `before` and compare with the engine's theta and receipts."""
    b = batch_of(syn, step)
    treat = c["treat"]
    t, th, p, Rs = my_gradients(syn, before["model"], before["critics"], before["transforms"], before["critic_head"], b,
                                treat)
    q, caps, zeros = my_q(c, t, p, a_max=a_max)
    de, dh, kappa, nrm = predicted_delta(t, th, q, clip=clip)
    j = step - 1
    for i in (0, 1):
        got = flat(after_model, enc_keys(i)) - flat(before["model"], enc_keys(i))
        assert float((got - de[i]).norm()) <= 2e-3 * float(de[i].norm()) + 1e-9, (c, step, i)
        goth = flat(after_model, head_keys(i)) - flat(before["model"], head_keys(i))
        assert float((goth - dh[i]).norm()) <= 2e-3 * float(dh[i].norm()) + 1e-9, (c, step, "head", i)
        tn = float(t[i].double().norm())
        assert math.isclose(rec["t_norm"][j, i], tn, rel_tol=1e-5)
        qn = float(q[i].norm())
        assert math.isclose(rec["q_norm"][j, i], qn, rel_tol=1e-4, abs_tol=1e-12)
        if not zeros[i]:
            assert math.isclose(rec["ratio"][j, i], qn / tn, rel_tol=1e-4)
            assert math.isclose(rec["realized_ratio"][j, i], qn / tn, rel_tol=1e-4)
            pn = float(p[i].double().norm())
            assert math.isclose(rec["p_norm"][j, i], pn, rel_tol=1e-4)
            cos = float(torch.dot(t[i].double(), p[i].double())) / (tn * pn)
            assert abs(rec["cos"][j, i] - cos) <= 1e-4
        else:
            assert rec["realized_ratio"][j, i] == 0.0 and rec["zero"][j, i] != T.Z_NONE
        assert bool(rec["cap"][j, i]) == caps[i]
    assert math.isclose(rec["kappa"][j], kappa, rel_tol=1e-5)
    assert math.isclose(rec["pre_total"][j], nrm, rel_tol=1e-4)   # engine: float32 norm of float32 u
    for vi, vw in enumerate(VIEWS):
        if vw in Rs:
            assert math.isclose(rec["R"][j, vi], Rs[vw], rel_tol=1e-4, abs_tol=1e-7)
        else:
            assert np.isnan(rec["R"][j, vi])
    return {"kappa": kappa, "caps": caps, "zeros": zeros}


# ================================================================== 1. update algebra (first principles)
@pytest.mark.parametrize("c", [RAW("J", 0.3), RAW("L", 0.6), NORM("J", 3.0, 2.0), NORM("L", 5.0, 0.5),
                               NORM("J", 1.5, 1.0), NORM("L", 3.0, 2.0)], ids=T.config_id)
def test_one_step_reconstruction_from_first_principles(syn, c):
    """theta_s - theta_{s-1} = -lr * kappa * [t_enc + q, t_head] with q from the registered formula, the declared salt-0
    minibatch, the fixed warm head, the aligned critics/transform; receipts equal the recomputed scalars."""
    S = syn["S_per_ep"] * 2
    caps = {1, 2, syn["S_per_ep"] + 1, syn["S_per_ep"] + 2}
    m, d, ck, cap, fin, rec = run(syn, c, ep=2, capture_steps=caps)
    for s in (1, syn["S_per_ep"] + 1):
        check_step(syn, c, cap[s], cap[s + 1]["model"], rec, s)
    check_step(syn, c, fin["theta_T_minus_1"], fin["theta_T"], rec, S)


def test_clip_is_global_and_applied_after_q(syn, monkeypatch):
    """Tiny clip: every step clipped by the GLOBAL norm over [t_enc + q, t_head]; ratios are pre-clip and invariant."""
    c = NORM("J", 3.0, 0.5)
    _, _, _, _, _, rec0 = run(syn, c, ep=1)
    monkeypatch.setitem(T.HP, "clip", 0.2)
    caps = {1, 2}
    m, d, ck, cap, fin, rec = run(syn, c, ep=1, capture_steps=caps)
    out = check_step(syn, c, cap[1], cap[2]["model"], rec, 1, clip=0.2)
    assert out["kappa"] < 1
    assert (rec["kappa"] < 1).all() and d["clip_hits"] == syn["S_per_ep"]
    np.testing.assert_allclose(rec["post_total"], 0.2, rtol=1e-5)
    tot2 = (rec["pre_enc"] ** 2).sum(1) + rec["pre_head"] ** 2
    np.testing.assert_allclose(np.sqrt(tot2), rec["pre_total"], rtol=1e-5)
    # step 1 starts from identical theta_0 / critics: the strength receipts do not depend on the clip
    for f in ("t_norm", "q_norm", "ratio", "p_norm"):
        np.testing.assert_allclose(rec[f][0], rec0[f][0], rtol=1e-6)


def test_cap_changes_magnitude_never_direction(syn, monkeypatch):
    c = NORM("J", 3.0, 1.0)
    monkeypatch.setitem(T.HP, "A_MAX", 1e-3)
    m, d, ck, cap, fin, rec = run(syn, c, ep=1, capture_steps={1, 2})
    out = check_step(syn, c, cap[1], cap[2]["model"], rec, 1, a_max=1e-3)
    assert all(out["caps"])
    ok = rec["zero"] == T.Z_NONE
    assert ok.all() and rec["cap"].all() and d["cap_hits"] == [syn["S_per_ep"]] * 2
    np.testing.assert_allclose(rec["ratio"], 1e-3 * rec["p_norm"] / rec["t_norm"], rtol=1e-4)
    assert (rec["ratio"] < 3.0).all()
    assert (rec["rms_ratio"] < 3.0).all()                   # the realized RMS falls short of rho; never inflated
    np.testing.assert_allclose(rec["scale"], 1e-3)


def test_float64_norms_in_the_scalar():
    """A task block whose float32 norm overflows: the scalar must still be finite and exact (float64 norms)."""
    t = torch.full((1000,), 3e37, dtype=torch.float32)        # ||t|| ~ 9.5e38 > float32 max
    p = torch.full((1000,), 1.0, dtype=torch.float32)
    q, info = T.normalized_direction(t, p, 1e-30, a_max=1e300, zero_tol=1e-12)
    exp = 1e-30 * math.sqrt(1000) * 3e37 / math.sqrt(1000)
    assert math.isfinite(info["a"]) and math.isclose(info["a"], exp, rel_tol=1e-6)
    tiny = torch.full((1000,), 1e-25, dtype=torch.float32)     # float32 sum of squares underflows to 0
    q2, info2 = T.normalized_direction(torch.ones(1000), tiny, 1.0, a_max=1e300, zero_tol=1e-30)
    assert info2["zero"] is None and math.isclose(info2["p_norm"], 1e-25 * math.sqrt(1000), rel_tol=1e-6)


def test_zero_rule_and_threshold():
    t = torch.ones(10)
    q, info = T.normalized_direction(t, torch.full((10,), 1e-14), 3.0, a_max=100.0, zero_tol=1e-12)
    assert info["zero"] == "p" and float(q.abs().max()) == 0.0          # ||p|| = 3.2e-14 <= 1e-12
    q, info = T.normalized_direction(t, torch.full((10,), 1e-12), 3.0, a_max=1e300, zero_tol=1e-12)
    assert info["zero"] is None                                          # ||p|| = 3.2e-12 > 1e-12
    q, info = T.normalized_direction(torch.zeros(10), torch.ones(10), 3.0, a_max=100.0, zero_tol=1e-12)
    assert info["zero"] == "t" and float(q.abs().max()) == 0.0
    q, info = T.normalized_direction(t, -torch.ones(10), 3.0, a_max=100.0, zero_tol=1e-12)
    assert info["a"] > 0 and torch.allclose(q, -3.0 * torch.ones(10))   # positive scalar: sign of p kept


def test_zero_threshold_is_wired_to_HP(syn, monkeypatch):
    """The registered threshold reaches the training call: with ZERO_TOL above every ||p_i|| all steps are zero events
    (reason 2), q = 0, realized ratio 0, and the encoder update equals the task-only update."""
    m0, *_ = run(syn, TASK, ep=1)
    monkeypatch.setitem(T.HP, "ZERO_TOL", 1e6)
    m, d, ck, cap, fin, rec = run(syn, NORM("J", 3.0, 1.0), ep=1)
    assert (rec["zero"] == T.Z_P).all() and (rec["q_norm"] == 0).all() and (rec["realized_ratio"] == 0).all()
    assert same_state(m.state_dict(), m0.state_dict())


def test_norm_field_has_no_rest_point_unless_task_stationary():
    """Design note (MATH_REVIEW section 8): uncapped, ||t + q|| >= |rho s - 1| ||t|| for every nonzero t, p, so a NORM
    run with rho s != 1 never has a full-batch rest point with t != 0; RAW rest points t = -beta p have ratio exactly 1."""
    g = torch.Generator().manual_seed(0)
    for rs in (0.5, 1.5, 3.0, 6.3):
        for _ in range(20):
            t = torch.randn(50, generator=g, dtype=torch.float64)
            p = torch.randn(50, generator=g, dtype=torch.float64)
            if _ == 0:
                p = -t * 0.37                                       # exactly anti-aligned: the worst case
            q, info = T.normalized_direction(t, p, rs, a_max=1e300, zero_tol=1e-12)
            assert float((t + q).norm()) >= abs(rs - 1) * float(t.norm()) * (1 - 1e-12)
    t = torch.randn(50, generator=g, dtype=torch.float64)
    beta = 0.3
    p = -t / beta                                                    # RAW stationary point: t + beta p = 0
    assert math.isclose(float((beta * p).norm()) / float(t.norm()), 1.0, rel_tol=1e-12)


# ================================================================== 2. allocation and strength receipts
def test_allocation_changes_split_not_budget(syn):
    """Same theta_0 / critics / minibatch at step 1: t, p identical across a; ||q_i|| = rho s_i(a) ||t_i||; RMS = rho;
    ratio_1/ratio_2 = a; proxy coefficients independent of a; combined ratio is the task-weighted mean."""
    rho = 3.0
    rows = {}
    for a in (0.5, 1.0, 2.0):
        m, d, ck, cap, fin, rec = run(syn, NORM("J", rho, a), ep=1)
        assert d["coefficients"] == run(syn, NORM("J", rho, 1.0), ep=1)[1]["coefficients"]
        rows[a] = rec
    for a, rec in rows.items():
        np.testing.assert_array_equal(rec["t_norm"][0], rows[1.0]["t_norm"][0])
        np.testing.assert_array_equal(rec["p_norm"][0], rows[1.0]["p_norm"][0])
        s = my_alloc(a)
        assert (rec["zero"][0] == T.Z_NONE).all() and not rec["cap"][0].any()
        for i in (0, 1):
            assert math.isclose(rec["q_norm"][0, i], rho * s[i] * rec["t_norm"][0, i], rel_tol=1e-5)
        assert math.isclose(rec["rms_ratio"][0], rho, rel_tol=1e-5)
        assert math.isclose(rec["ratio"][0, 0] / rec["ratio"][0, 1], a, rel_tol=1e-5)
        t1, t2 = rec["t_norm"][0]
        r1, r2 = rec["ratio"][0]
        comb = math.sqrt((r1 * r1 * t1 * t1 + r2 * r2 * t2 * t2) / (t1 * t1 + t2 * t2))
        assert math.isclose(rec["comb_ratio"][0], comb, rel_tol=1e-5)
    # the combined ratio differs from the declared RMS whenever a != 1 and t_1 != t_2
    rec = rows[2.0]
    assert abs(rec["comb_ratio"][0] - rho) > 1e-3


def test_raw_combined_ratio_matches_rgj_logged_norms(syn):
    """RAW receipts vs the pinned rgj log: combined ratio = penalty/task at every logged step; individual ratios are NOT
    constant and their RMS differs from the combined ratio (section 3 warning)."""
    ep = 6
    m, d, ck, cap, fin, rec = run(syn, RAW("J", 0.3), ep=ep)
    m2, d2, *_ = RT.train_run("J-O", 0.3, syn["warm"], syn["data"], SEED, "B", n_epochs=ep, critic_head=syn["head"],
                              ckpt_epochs=())
    assert same_state(m.state_dict(), m2.state_dict())
    assert d2["norms"]
    for e in d2["norms"]:
        j = e["step"] - 1
        assert math.isclose(rec["comb_ratio"][j], e["penalty"] / e["task"], rel_tol=1e-5)
    ok = (rec["zero"] == T.Z_NONE).all(1)
    assert ok.sum() > 3
    assert np.ptp(rec["ratio"][ok, 0]) > 0                      # not a constant individual ratio
    assert np.max(np.abs(rec["rms_ratio"][ok] - rec["comb_ratio"][ok])) > 0


def test_realized_vs_conditional_summary(syn, monkeypatch):
    """Zero steps count 0 in realized statistics; conditional statistics exclude them and are labelled conditional."""
    orig = T.new_critic

    def losing(seed, view, kind, tag):
        c = orig(seed, view, kind, tag)
        if view == "v1":
            with torch.no_grad():
                c[-1].weight.zero_()
                c[-1].bias.copy_(torch.tensor([20.0, -20.0]))
        return c
    monkeypatch.setattr(T, "new_critic", losing)
    m, d, ck, cap, fin, rec = run(syn, NORM("L", 3.0, 1.0), ep=3)
    z1 = rec["zero"][:, 0]
    assert set(np.unique(z1)) <= {T.Z_P, T.Z_NOGRAD}            # encoder 1 never receives protection
    assert (rec["q_norm"][:, 0] == 0).all() and (rec["realized_ratio"][:, 0] == 0).all()
    assert d["zero_events"][0] == len(z1)
    live = rec["zero"][:, 1] == T.Z_NONE
    assert live.any(), "fixture needs some live v2 steps"
    np.testing.assert_allclose(rec["ratio"][live, 1], 3.0, rtol=1e-5)   # not inflated to make up the budget
    np.testing.assert_allclose(rec["rms_ratio"][live], 3.0 / math.sqrt(2), rtol=1e-5)
    s = d["summary"]
    assert s["enc1"]["zero_fraction"] == 1.0 and s["enc1"]["conditional_ratio_nonzero_steps"]["n"] == 0
    assert math.isclose(s["enc1"]["realized_ratio"]["mean"], 0.0)
    frac = live.mean()
    assert math.isclose(s["enc2"]["realized_ratio"]["mean"], 3.0 * frac, rel_tol=1e-5)
    assert math.isclose(s["enc2"]["conditional_ratio_nonzero_steps"]["mean"], 3.0, rel_tol=1e-5)


def test_all_views_losing_gives_task_update_and_logged_zero(syn, monkeypatch):
    """Constant wins on every active view: P has no gradient, q = 0 with reason 1 on both encoders, the encoder update
    equals the task-only update bitwise, and the critics still train (diagnostic work running)."""
    orig = T.new_critic

    def losing(seed, view, kind, tag):
        c = orig(seed, view, kind, tag)
        with torch.no_grad():
            c[-1].weight.zero_()
            c[-1].bias.copy_(torch.tensor([20.0, -20.0]))
        return c
    m0, *_ = run(syn, TASK, ep=1)
    monkeypatch.setattr(T, "new_critic", losing)
    for c in (NORM("J", 3.0, 2.0), RAW("J", 0.6)):
        m, d, ck, cap, fin, rec = run(syn, c, ep=1)
        assert (rec["zero"] == T.Z_NOGRAD).all() and d["zero_events"] == [syn["S_per_ep"]] * 2
        assert (rec["realized_ratio"] == 0).all() and (rec["rms_ratio"] == 0).all()
        assert same_state(m.state_dict(), m0.state_dict())
        assert d["critic_online_updates"] == syn["S_per_ep"] * 5 * 6


# ================================================================== 3. local isolation, RNG, receipts
@pytest.mark.parametrize("c", [RAW("L", 0.6), NORM("L", 5.0, 2.0)], ids=T.config_id)
def test_local_encoder_update_invariant_to_pair_critics(syn, monkeypatch, c):
    m0, d0, ck0, *_ = run(syn, c, ep=2)
    mj0, *_ = run(syn, NORM("J", 5.0, 2.0), ep=2)
    orig = T.new_critic

    def other_pair(seed, view, kind, tag):
        cr = orig(seed, view, kind, tag)
        if view == "pair":
            with torch.no_grad():
                for prm in cr.parameters():
                    prm.mul_(3.0).add_(0.1)
        return cr
    monkeypatch.setattr(T, "new_critic", other_pair)
    m1, d1, ck1, *_ = run(syn, c, ep=2)
    mj1, *_ = run(syn, NORM("J", 5.0, 2.0), ep=2)
    assert same_state(m0.state_dict(), m1.state_dict())                 # pair critics never reach a local encoder
    assert not same_state(mj0.state_dict(), mj1.state_dict())           # ... but do reach a joint encoder
    assert not same_state(ck0[2]["critics"]["pair"]["A"], ck1[2]["critics"]["pair"]["A"])
    init = orig(SEED, "pair", "B", "init").state_dict()
    assert not same_state(init, ck0[2]["critics"]["pair"]["B"])          # the shadow bank trained
    assert d0["critic_online_updates"] == d1["critic_online_updates"] == 2 * syn["S_per_ep"] * 30


def test_joint_pair_term_reaches_encoder_with_dead_local_view(syn, monkeypatch):
    """Contrast with the local fixture: with v1 critics losing, a JOINT encoder 1 still gets protection via R_pair."""
    orig = T.new_critic

    def losing(seed, view, kind, tag):
        c = orig(seed, view, kind, tag)
        if view == "v1":
            with torch.no_grad():
                c[-1].weight.zero_()
                c[-1].bias.copy_(torch.tensor([20.0, -20.0]))
        return c
    monkeypatch.setattr(T, "new_critic", losing)
    m, d, ck, cap, fin, rec = run(syn, NORM("J", 3.0, 1.0), ep=3)
    live = (rec["zero"][:, 0] == T.Z_NONE)
    assert (rec["R"][:, 0] == 0).all()
    assert live.any() and (rec["R"][live, 2] > 0).all()


def test_receipts_and_snapshots_consume_no_randomness(syn):
    c = RAW("J", 0.3)
    S = 2 * syn["S_per_ep"]
    m0, d0, ck0, cap0, fin0, rec0 = run(syn, c, ep=2)
    logs = []
    np.random.seed(123)
    torch.manual_seed(99)
    m1, d1, ck1, cap1, fin1, rec1 = run(syn, c, ep=2, capture_steps=set(range(1, S + 1)), log=logs.append)
    assert same_state(m0.state_dict(), m1.state_dict()) and len(cap1) == S and logs
    for f in rec0:
        np.testing.assert_array_equal(rec0[f], rec1[f])
    for v in VIEWS:
        for k in ("A", "B"):
            assert same_state(ck0[2]["critics"][v][k], ck1[2]["critics"][v][k])


def test_common_minibatch_order_across_modes(syn):
    fps = {}
    for c in (TASK, RAW("L", 0.1), NORM("J", 3.0, 0.5)):
        fps[T.config_id(c)] = run(syn, c, ep=2)[5]["mb_fp"]
    vals = list(fps.values())
    assert all(np.array_equal(vals[0], x) for x in vals[1:])
    mine = np.array([T.fp64(batch_of(syn, s).astype(np.int64)) for s in range(1, 2 * syn["S_per_ep"] + 1)])
    np.testing.assert_array_equal(vals[0], mine)
    other = T.train_run(TASK, syn["warm"], syn["data"], SEED, None, n_epochs=2, salt=1, ckpt_epochs=())[5]["mb_fp"]
    assert not np.array_equal(other, vals[0])


def test_snapshot_alignment_theta_T_minus_1_vs_theta_T(syn):
    per = syn["S_per_ep"]
    S = 2 * per
    m, d, ck, cap, fin, rec = run(syn, NORM("J", 3.0, 1.0), ep=2, ckpt_epochs=(1, 2), capture_steps={per + 1})
    assert same_state(cap[per + 1]["model"], ck[1]["model"])            # capture at step s holds theta_{s-1}
    assert same_state(fin["theta_T"], ck[2]["model"]) and same_state(fin["theta_T"], m.state_dict())
    assert not same_state(fin["theta_T_minus_1"]["model"], fin["theta_T"])
    assert fin["theta_T_minus_1"]["step"] == S
    for v in VIEWS:                                                      # critics aligned with theta_{T-1}
        for k in ("A", "B"):
            assert same_state(fin["theta_T_minus_1"]["critics"][v][k], ck[2]["critics"][v][k])
    np.testing.assert_array_equal(rec["critic_updates"], 30 * np.arange(1, S + 1))
    assert "opts" in ck[2] and "opts" in fin["theta_T_minus_1"]         # resumable (predecessor R4)


def test_capture_steps_accepts_the_fraction_dict_or_its_values(syn):
    """Review A3 (applied): capture_steps_for returns {fraction: step}; train_run must capture the same steps whether it
    receives the dict or its values (before the repair the dict compared steps with FRACTIONS: only step 1 matched)."""
    per = syn["S_per_ep"]
    cs = T.capture_steps_for(2, syn["data"].n)
    assert sorted(cs.values()) == sorted({max(1, round(f * 2 * per)) for f in T.HP["progress_fractions"]})
    good = run(syn, TASK, ep=2, capture_steps=set(cs.values()))[3]
    assert sorted(good) == sorted(set(cs.values()))
    via_dict = run(syn, TASK, ep=2, capture_steps=cs)[3]
    assert sorted(via_dict) == sorted(good)
    for s in good:
        assert same_state(good[s]["model"], via_dict[s]["model"])


def test_raw_zero_task_gradient_with_protection_is_flagged_not_zero(syn):
    """Review A4 (applied): RAW with t_1 = 0 (zeroed TRAINING head 0; the critic-view head stays the warm head) still
    applies q_1 = beta p_1; the receipt must not report zero strength: ratio = realized = inf and the step is counted."""
    st = {k: v.clone() for k, v in syn["warm"].items()}
    st["head.0.weight"].zero_()
    m, d, ck, cap, fin, rec = T.train_run(RAW("J", 0.3), st, syn["data"], SEED, syn["head"], n_epochs=1,
                                          ckpt_epochs=(1,))
    hit = (rec["t_norm"][:, 0] == 0) & (rec["q_norm"][:, 0] > 0)
    assert hit.any()
    assert np.isinf(rec["realized_ratio"][hit, 0]).all() and np.isinf(rec["ratio"][hit, 0]).all()
    assert d["undefined_ratio_steps"][0] == int(hit.sum())
    assert not (rec["realized_ratio"][hit, 0] == 0).any()


def test_stale_transform_artifact_requires_aligned_snapshots(syn):
    """Review A8 / AMENDMENT_A2 context. An epoch checkpoint holds theta_e with the critics AND the floored-ZCA transform
    of theta_{e-1}. When the representation is rank-deficient (here 52 of 64 second-layer units dead, as can happen on
    real rows), one SGD step moves r out of the old span and the stale transform amplifies that by ~1e4: the critics
    then lose to the constant on every batch although training had a gradient on every step. The aligned
    theta_{T-1} snapshot, or theta_T with the transform recomputed from its own reference rows, is informative."""
    import torch.nn.functional as F
    data = syn["data"]
    warm = {k: v.clone() for k, v in syn["warm"].items()}
    for i in (0, 1):
        warm[f"enc.{i}.2.bias"][:52] = -50.0
    head = T.head_of(warm)
    m, d, ck, cap, fin, rec = T.train_run(RAW("J", 0.3), warm, data, SEED, head, n_epochs=4, ckpt_epochs=(4,))
    assert not (rec["zero"] == T.Z_NOGRAD).any()                      # training itself always had a gradient path
    perm = np.random.default_rng([SEED, 0, 4]).permutation(data.n)
    bs = [perm[s:s + 256] for s in range(0, data.n, 256)]
    stale = [T.frozen_equivalence(ck[4]["model"], data, SEED, head, "J", 0.3, b, snapshot=ck[4])["applicable"] for b in bs]
    sn = fin["theta_T_minus_1"]
    aligned = [T.frozen_equivalence(sn["model"], data, SEED, head, "J", 0.3, b, snapshot=sn)["applicable"] for b in bs]
    assert all(a == 0 for a in stale) and all(a == 2 for a in aligned)
    cal = torch.from_numpy(data.cal)
    V = T.critic_views(m, data.X[cal], list(VIEWS), head)
    Vr = T.critic_views(m, data.X[torch.from_numpy(data.ref)], list(VIEWS), head)
    for v in VIEWS:
        Ts, Tn = T.Transform(state=ck[4]["transforms"][v]), T.Transform(Vr[v], "floored")
        c = T.new_critic(SEED, v, "B", "init")
        c.load_state_dict(ck[4]["critics"][v]["B"])
        with torch.no_grad():
            ce_stale = float(F.cross_entropy(c(Ts(V[v])), data.S[cal]))
            ce_new = float(F.cross_entropy(c(Tn(V[v])), data.S[cal]))
            assert float(Ts(V[v]).abs().max()) > 1e3 and ce_stale > 50 * ce_new and ce_new < 1.0


# ================================================================== 4. raw fidelity, parity, admitted U
@pytest.mark.parametrize("treat,arm,beta", [("J", "J-O", 0.6), ("L", "L-O", 0.1)])
def test_raw_bitwise_rgj_other_betas_seed(syn, treat, arm, beta):
    m, d, ck, *_ = run(syn, RAW(treat, beta), ep=3)
    m2, d2, ck2, *_ = RT.train_run(arm, beta, syn["warm"], syn["data"], SEED, "B", n_epochs=3, critic_head=syn["head"],
                                   ckpt_epochs=(3,))
    assert same_state(m.state_dict(), m2.state_dict())
    assert d["clip_hits"] == d2["clip_hits"] and d["encoder_updates"] == d2["encoder_updates"]
    for v in VIEWS:
        for k in ("A", "B"):
            assert same_state(ck[3]["critics"][v][k], ck2[3]["critics"][v][k])
    assert d["coefficients"] == RT.coefficients({"J": "joint", "L": "local"}[treat], beta, (0.0, 0.0))


def test_smf_task_line_equals_osf_task(syn):
    """Admission premise for reused smf U: smf.train TASK (stage A = salt 0) == osf TASK (salt 0) bitwise."""
    from smf import train as ST
    m, *_ = run(syn, TASK, ep=2)
    m2, *_ = ST.train_run({"base": "none", "schedule": "ONLINE", "update": "task"}, 0.0, syn["warm"], syn["data"], SEED,
                          "A", None, n_epochs=2, ckpt_epochs=())
    assert same_state(m.state_dict(), m2.state_dict())


@pytest.mark.parametrize("c", [RAW("J", 0.0), NORM("J", 0.0, 2.0), NORM("L", 0.0, 0.5)], ids=T.config_id)
def test_zero_strength_parity_with_captures_running(syn, c):
    m0, *_ = run(syn, TASK, ep=2)
    m, d, ck, cap, fin, rec = run(syn, c, ep=2, capture_steps={1, 3, 5})
    assert same_state(m.state_dict(), m0.state_dict()) and d["critic_online_updates"] > 0 and len(cap) == 3
    assert (rec["zero"] == T.Z_OFF).all() and d["zero_events"] == [0, 0]


def test_no_controller_or_refit_paths():
    import inspect
    src = inspect.getsource(T)
    for bad in ("refit_banks", "fit_bounded", "controller", "probe_receipt", "dual_update", "transport_first_layer",
                "matched_counts", "w_hyp", "lam[0] +"):
        assert bad not in src.replace("No refits", "").replace("no controller", ""), bad
    sig = inspect.signature(T.train_run)
    assert set(sig.parameters) == {"c", "init_state", "data", "seed", "critic_head", "n_epochs", "salt", "ckpt_epochs",
                                   "capture_steps", "lr", "log"}


def test_identical_task_head_and_input_schema_for_every_mode(syn):
    """All modes train the same parameter set (two encoders + affine heads) from the same 83-column input; the task
    head always receives the plain task gradient (heads do not depend on the protection mode at step 1)."""
    heads = {}
    for c in (TASK, RAW("J", 0.6), NORM("L", 5.0, 2.0)):
        m, d, ck, cap, fin, rec = run(syn, c, ep=1, capture_steps={2})
        sd = cap[2]["model"]
        assert set(sd) == set(syn["warm"]) and sd["enc.0.0.weight"].shape[1] == 83 == sd["enc.1.0.weight"].shape[1]
        heads[T.config_id(c)] = rec["pre_head"][0]
    vals = list(heads.values())
    assert all(v == vals[0] for v in vals)


# ================================================================== 5. frozen-minibatch equivalence algebra
def _frozen(syn):
    m, d, ck, *_ = run(syn, RAW("J", 0.3), ep=2)
    b = np.random.default_rng([SEED, 0, 2]).permutation(syn["data"].n)[:256]
    return ck[2], b


def test_frozen_equivalence_independent_algebra(syn):
    snap, b = _frozen(syn)
    beta = 0.3
    t, th, p, _ = my_gradients(syn, snap["model"], snap["critics"], snap["transforms"], snap["critic_head"], b, "J")
    r = [float((beta * p[i].double()).norm()) / float(t[i].double().norm()) for i in (0, 1)]
    assert abs(r[0] - r[1]) > 1e-3 * max(r)
    for i in (0, 1):                           # rho_i = r_i reconstructs beta p_i (L2 relative error, float32 cast)
        q, info = T.normalized_direction(t[i], p[i], r[i], a_max=100.0, zero_tol=1e-12)
        raw = beta * p[i].double()
        assert float((q.double() - raw).norm()) <= 1e-6 * float(raw.norm()) and not info["cap"]
    common = math.sqrt((r[0] ** 2 + r[1] ** 2) / 2)
    errs = [float((T.normalized_direction(t[i], p[i], common)[0].double() - beta * p[i].double()).norm())
            / float((beta * p[i].double()).norm()) for i in (0, 1)]
    assert max(errs) > 1e-3                                       # a common rho fails when r_1 != r_2
    lead = T.frozen_equivalence(snap["model"], syn["data"], SEED, syn["head"], "J", beta, b, snapshot=snap)
    assert lead["equivalent"] and lead["applicable"] == 2
    for i in (0, 1):
        assert math.isclose(lead["encoders"][i]["r_i"], r[i], rel_tol=1e-4)
    bad = T.frozen_equivalence(snap["model"], syn["data"], SEED, syn["head"], "J", beta, b, snapshot=snap,
                               rho_override=common)
    assert not bad["equivalent"]
    capd = T.frozen_equivalence(snap["model"], syn["data"], SEED, syn["head"], "J", beta, b, snapshot=snap,
                                a_max=0.5 * beta)
    assert not capd["equivalent"]


def test_frozen_equivalence_fails_for_zero_task_gradient(syn):
    """t_1 = 0 (zeroed TRAINING head 0; the critic-view head stays the warm head, so p_1 != 0): r_1 is undefined and the
    norm expression cannot reconstruct beta p_1 -> not equivalent."""
    snap, b = _frozen(syn)
    st = {k: v.clone() for k, v in snap["model"].items()}
    st["head.0.weight"].zero_()
    t, th, p, _ = my_gradients(syn, st, snap["critics"], snap["transforms"], snap["critic_head"], b, "J")
    assert float(t[0].abs().max()) == 0.0 and float(p[0].abs().max()) > 0
    out = T.frozen_equivalence(st, syn["data"], SEED, syn["head"], "J", 0.3, b, snapshot=snap)
    assert not out["equivalent"]
    assert out["encoders"][0]["equivalent"] is False


@pytest.mark.parametrize("delta,expect", [(1e-2, False), (2e-4, False), (2e-5, True)])
def test_frozen_equivalence_tolerance_is_relative_L2(syn, monkeypatch, delta, expect):
    """Review A1 (applied): the declared tolerance is ||q_norm - q_raw||_2 <= 1e-4 ||q_raw||_2. A scalar misstatement
    of 2e-4 must be rejected (the earlier max-abs criterion accepted up to ~3.7e-4 here); 2e-5 is within tolerance.
    The honest two-pass float32 noise is 2e-5 / 5.2e-5 here (2e-6 to 6e-6 measured on real fitting rows)."""
    snap, b = _frozen(syn)
    e0 = [e["rel_err"] for e in T.frozen_equivalence(snap["model"], syn["data"], SEED, syn["head"], "J", 0.3, b,
                                                       snapshot=snap)["encoders"]]
    assert max(e0) < 1e-4                                           # honest two-pass float32 noise (2e-5, 5.2e-5 here)
    orig = T.normalized_direction

    def off(t, p, rho_i, **kw):
        q, info = orig(t, p, rho_i, **kw)
        return q * (1 + delta), info
    monkeypatch.setattr(T, "normalized_direction", off)
    out = T.frozen_equivalence(snap["model"], syn["data"], SEED, syn["head"], "J", 0.3, b, snapshot=snap)
    assert out["equivalent"] is expect
    for e, n0 in zip(out["encoders"], e0):                          # triangle inequality around the honest noise
        assert delta * (1 - n0) - n0 - 1e-9 <= e["rel_err"] <= delta * (1 + n0) + n0 + 1e-9


# ================================================================== 6. data roles (synthetic + light real check)
def _old_roles(seed=5):
    rng = np.random.default_rng(seed)
    names = ["defense_train"] * 600 + ["defense_val"] * 200 + ["attacker_fit"] * 120 + ["attacker_val"] * 60 + \
        ["assessment"] * 90 + ["cert"] * 40 + ["excluded_exposure"] * 3 + ["excluded_dup"] * 3
    old = np.array(names, dtype="<U32")
    unit = np.arange(len(old)) + 10_000
    a = np.flatnonzero(old == "assessment")
    unit[a[0]] = unit[np.flatnonzero(old == "excluded_dup")[0]]          # assessment row shares a group with an exclusion
    unit[a[1]] = unit[np.flatnonzero(old == "excluded_exposure")[0]]
    return old, unit


def test_assign_roles_union_and_exclusion_synthetic():
    old, unit = _old_roles()
    role, sub, pool, info = OD.assign_roles(old, unit)
    from smf import data as SD
    smf_role, smf_sub, rgj_role = SD.assign_roles(old, unit)
    # every fitting role is smf's, unchanged
    for r_osf, r_smf in (("OSF_DEFENSE_FIT", "NEW_DEFENSE_FIT"), ("HEAD_VALIDATION", "HEAD_VALIDATION"),
                         ("AUDIT_FIT", "AUDIT_FIT"), ("INNER_SELECTION", "INNER_SELECTION")):
        np.testing.assert_array_equal(role == r_osf, smf_role == r_smf)
    # assessment = union of the four pools minus groups touching a fitting role or an exclusion
    cand = (old == "assessment") | (old == "cert") | (rgj_role == "DEVELOPMENT_ASSESSMENT") | \
        (smf_role == "NEW_DEVELOPMENT_ASSESSMENT")
    blocked = set(unit[np.isin(smf_role, ("NEW_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")) |
                       np.isin(old, ("excluded_exposure", "excluded_dup"))].tolist())
    exp = cand & ~np.isin(unit, list(blocked))
    np.testing.assert_array_equal(role == "OSF_DEVELOPMENT_ASSESSMENT", exp)
    assert info["ORIG_ASSESSMENT"]["excluded_rows_group_overlap"] == 2
    # no group straddles the assessment and anything else
    ass_units = set(unit[role == "OSF_DEVELOPMENT_ASSESSMENT"].tolist())
    assert not ass_units & set(unit[(role != "OSF_DEVELOPMENT_ASSESSMENT")].tolist())
    # ineligible certification pool: the entire pool leaves
    role2, _, pool2, info2 = OD.assign_roles(old, unit, cert_eligible=False)
    assert not np.any(role2[old == "cert"] != "") and info2["CERT"]["excluded_rows_ineligible_pool"] == 40


def test_group_overlapping_a_fitting_role_fails_loudly():
    old, unit = _old_roles()
    a = np.flatnonzero(old == "assessment")[5]
    unit[a] = unit[np.flatnonzero(old == "attacker_fit")[0]]
    role, sub, pool, info = OD.assign_roles(old, unit)
    assert role[a] == ""                                                  # excluded, not pooled
    with pytest.raises(AssertionError):
        OD.check_partition(role, sub, pool, unit, old)                   # and the loader refuses (loud, not silent)


@pytest.fixture(scope="module")
def real():
    try:
        return OD.load()
    except SystemExit as e:                                               # pragma: no cover
        pytest.skip(str(e))


def test_real_roles_match_smf_fitting_roles_and_union(real):
    from smf import data as SD
    from rgj import data as RD
    D = real
    S = SD.load()
    for r_osf, r_smf in (("OSF_DEFENSE_FIT", "NEW_DEFENSE_FIT"), ("HEAD_VALIDATION", "HEAD_VALIDATION"),
                         ("AUDIT_FIT", "AUDIT_FIT"), ("INNER_SELECTION", "INNER_SELECTION"),
                         ("CRITIC_FIT", "CRITIC_FIT"), ("CRITIC_VAL", "CRITIC_VAL"), ("DIAGNOSTIC_CALIB", "CONTROLLER_CALIB")):
        np.testing.assert_array_equal(D["row_id"][D["idx"][r_osf]], S["row_id"][S["idx"][r_smf]])
        np.testing.assert_array_equal(D["X"][D["idx"][r_osf]], S["X"][S["idx"][r_smf]])
    z = np.load(RD.SRC, allow_pickle=False)
    old, unit, rid = z["role"], z["unit"], z["row_id"]
    smf_role, _, rgj_role = SD.assign_roles(old, unit)
    cand = (old == "assessment") | (old == "cert") | (rgj_role == "DEVELOPMENT_ASSESSMENT") | \
        (smf_role == "NEW_DEVELOPMENT_ASSESSMENT")
    blocked = np.isin(unit, unit[np.isin(smf_role, ("NEW_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT",
                                                     "INNER_SELECTION")) | np.isin(old, OD.EXCLUSIONS)])
    exp = np.sort(rid[cand & ~blocked]) if OD.CERT_ELIGIBLE else np.sort(rid[cand & ~blocked & (old != "cert")])
    np.testing.assert_array_equal(np.sort(D["row_id"][D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]]), exp)
    units = [set(D["unit"][D["idx"][r]].tolist()) for r in OD.ROLES]
    assert all(not (units[i] & units[j]) for i in range(5) for j in range(i + 1, 5))


def test_real_preprocessing_fitted_on_defense_fit_only_and_labels_sealed(real):
    D = real
    names = list(D["feature_names"])
    fit = D["idx"]["OSF_DEFENSE_FIT"]
    ass = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    for c in ("age", "education-num", "capital-gain", "capital-loss", "hours-per-week"):
        x = D["X"][:, names.index(c)].astype(np.float64)
        assert abs(x[fit].mean()) < 1e-6 and abs(x[fit].std() - 1) < 1e-6
        assert abs(x[np.concatenate([fit, ass])].mean()) > 1e-5             # not refitted on fit + assessment
    oh = [j for j, f in enumerate(names) if "=" in f]
    assert len(oh) == 78 and np.isin(D["X"][:, oh], (0.0, 1.0)).all() and D["X"].shape[1] == 83
    for k in ("sex", "race", "y_income", "y_occupation_group"):          # own list, not the module's
        assert (D[k][ass] == -1).all() and (D[k][fit] >= 0).all()
    assert (D["y"]["income"][ass] == -1).all() and (D["y"]["occupation_group"][ass] == -1).all()
    for proc in ("training", "heads", "inner_audit", "selection"):
        with pytest.raises(PermissionError):
            OD.labels_for(D, proc, "OSF_DEVELOPMENT_ASSESSMENT")
    with pytest.raises(PermissionError):
        OD.labels_for(D, "selection", "HEAD_VALIDATION")


# ================================================================== 7. selection (section 13), synthetic inner world
SEEDS3 = (0, 1, 2)
U_ACC, CONST = (0.85, 0.48), (0.75, 0.30)


def _real_pkg_digest():
    """Digest of the public files osf.select writes (other agents write elsewhere in the folder concurrently)."""
    import hashlib
    from pathlib import Path
    pkg = Path(__file__).resolve().parents[2] / "results" / "pcrl_online_strength_frontier_v1"
    return {n: (hashlib.sha256((pkg / n).read_bytes()).hexdigest() if (pkg / n).exists() else None)
            for n in ("SELECTION.json", "INNER_FRONTIERS.csv", "SELECTION_TABLE.csv")}


def sel_world(monkeypatch, tmp_path, aucs, accs, refs, compute=None):
    """Run osf.select.select_all on a synthetic inner world: aucs[cid] / accs[cid] are (v1, v2, pair) / (income, occ),
    either constant or a {seed: value} dict; refs[label] = (aucs, accs, status). Every path redirected to tmp_path."""
    import sys
    import types
    import osf
    from osf import run as R
    SEL = importlib.import_module(SELM)
    ids = [T.config_id(c) for c in T.bank("full")]
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path / "pkg")
    monkeypatch.setattr(R, "UNITS", tmp_path / "units")
    (tmp_path / "pkg").mkdir(exist_ok=True)
    monkeypatch.setattr(R, "event", lambda *a, **k: None)
    monkeypatch.setattr(SEL, "lock_bank", lambda: ids)
    name2 = {R.rel_name(k, cid): (k, cid) for k in SEEDS3 for cid in ids}

    def per(v, k):
        return v[k] if isinstance(v, dict) else v

    def util_of(k, cid, src=accs):
        a = per(src.get(cid, U_ACC), k)
        return {0: {"acc": a[0], "const_acc": CONST[0]}, 1: {"acc": a[1], "const_acc": CONST[1]}}

    def auc_of(k, cid):
        a = per(aucs.get(cid, (0.85, 0.86, 0.88)), k)
        return {"v1": a[0], "v2": a[1], "pair": a[2]}
    monkeypatch.setattr(SEL, "util", lambda name: util_of(*name2[name]))
    monkeypatch.setattr(SEL, "auc", lambda name: auc_of(*name2[name]))
    monkeypatch.setattr(SEL, "compute_of", lambda k, cid: (compute or {}).get(cid, 100))

    def reference_candidates(k):
        out = {}
        for lab, (a, u, st) in refs.items():
            a, u = per(a, k), per(u, k)
            out[lab] = {"unit": f"ref__s{k}__{lab}", "status": st, "inner": dict(zip(("v1", "v2", "pair"), a)),
                        "utility": {0: {"acc": u[0], "const_acc": CONST[0]}, 1: {"acc": u[1], "const_acc": CONST[1]}}}
        return out
    fake = types.ModuleType("osf.baselines")
    fake.reference_candidates = reference_candidates
    monkeypatch.setitem(sys.modules, "osf.baselines", fake)
    monkeypatch.setattr(osf, "baselines", fake, raising=False)
    return SEL.select_all(None)


REFS = {"E": ((0.70, 0.70, 0.72), (0.76, 0.31), None),          # LEACE: low recovery, fails G3 -> infeasible
        "F": ((0.60, 0.60, 0.62), (0.85, 0.48), "NO_FEASIBLE_NOMINEE"),   # reference path's own status binding
        "F0": ((0.77, 0.79, 0.79), (0.85, 0.478), None)}         # compression tree: feasible, strong


def test_selection_rules_section13(monkeypatch, tmp_path):
    before = _real_pkg_digest()
    acc_ok = (0.85, 0.478)
    aucs = {"RAW-L|b0.1": (0.81, 0.83, 0.86), "RAW-L|b0.3": (0.80, 0.82, 0.84), "RAW-L|b0.6": (0.70, 0.70, 0.70),
            "NORM-L|r3|a1": (0.80, 0.82, 0.84),
            "RAW-J|b0.1": (0.80, 0.82, 0.85), "RAW-J|b0.3": (0.78, 0.80, 0.795), "RAW-J|b0.6": (0.78, 0.80, 0.795),
            "NORM-J|r5|a1": {0: (0.77, 0.79, 0.75), 1: (0.77, 0.7951, 0.75), 2: (0.77, 0.79, 0.75)},
            "NORM-J|r3|a0.5": (0.77, 0.79, 0.77), "NORM-J|r3|a2": (0.77, 0.79, 0.77),
            "NORM-J|r1.5|a1": (0.77, 0.79, 0.771), "NORM-J|r5|a0.5": (0.70, 0.70, 0.70)}
    accs = {c: acc_ok for c in aucs}
    accs["RAW-L|b0.6"] = {0: acc_ok, 1: acc_ok, 2: (0.835, 0.478)}          # fails G1 on ONE seed only
    accs["NORM-J|r5|a0.5"] = {0: (0.85, 0.469), 1: acc_ok, 2: acc_ok}       # fails G1 (occ) on one seed
    out = sel_world(monkeypatch, tmp_path, aucs, accs, REFS, compute={"RAW-L|b0.3": 90})
    st = out["statuses"]
    assert out["U_valid"]
    assert not out["rows"]["RAW-L|b0.6"]["task_feasible"] and not out["rows"]["NORM-J|r5|a0.5"]["task_feasible"]
    assert st["L*"]["status"] == "NOMINEE" and st["L*"]["config"] == "RAW-L|b0.3"        # pair tie -> lower compute
    assert st["C*"]["config"] == "F0"            # F excluded by its own status, E by gates, RAW-J 0.795 > 0.79
    assert st["N*"]["config"] == "NORM-J|r3|a0.5"    # r5|a1 fails the C* guard on seed 1 by 1e-4; tie -> |a-1| smaller
    assert st["R*"]["config"] == "RAW-J|b0.3"        # pair tie with beta 0.6 -> lower beta
    assert sorted(st["N*"]["guards_used"]) == ["C*", "L*"] and st["R*"]["guards_used"] == ["L*"]
    assert out["deployable_best"]["config"] == "NORM-J|r3|a0.5"
    assert out["deployable_best"]["kind"] == "normalized joint training"
    assert _real_pkg_digest() == before


def test_selection_C_star_may_be_raw_joint_beta_06(monkeypatch, tmp_path):
    aucs = {"RAW-J|b0.6": (0.78, 0.80, 0.78), "RAW-L|b0.3": (0.80, 0.82, 0.84)}
    out = sel_world(monkeypatch, tmp_path, aucs, {}, {**REFS, "F0": ((0.8, 0.8, 0.85), (0.85, 0.478), None)})
    assert out["statuses"]["C*"]["config"] == "RAW-J|b0.6"
    assert out["statuses"]["R*"]["config"] == "RAW-J|b0.6"


def test_selection_descriptive_fallback_cannot_pass(monkeypatch, tmp_path):
    aucs = {"RAW-L|b0.3": (0.80, 0.82, 0.84),
            "NORM-J|r3|a0.5": (0.80, 0.827, 0.70),          # guard excess 0.002 on v2, every seed -> 0.006
            "NORM-J|r1.5|a1": (0.79, 0.80, 0.80)}           # passes guards but fails G1 by 0.001 on seed 0
    for c in ("NORM-J|r3|a1", "NORM-J|r5|a1", "NORM-J|r3|a2", "NORM-J|r5|a0.5", "NORM-J|r5|a2"):
        aucs[c] = (0.95, 0.95, 0.95)
    accs = {"NORM-J|r1.5|a1": {0: (0.839, 0.478), 1: (0.85, 0.478), 2: (0.85, 0.478)}}
    for c in ("NORM-J|r3|a1", "NORM-J|r5|a1", "NORM-J|r3|a2", "NORM-J|r5|a0.5", "NORM-J|r5|a2"):
        accs[c] = (0.70, 0.30)
    out = sel_world(monkeypatch, tmp_path, aucs, accs, {**REFS, "F0": ((0.9, 0.9, 0.9), (0.85, 0.478), None)})
    N = out["statuses"]["N*"]
    assert N["status"] == "NO_FEASIBLE_NOMINEE" and N["config"] is None
    assert N["descriptive_config"] == "NORM-J|r1.5|a1"              # 0.001 shortfall < 0.006 despite higher pair AUC
    assert N["row"]["nomination_shortfall"] == pytest.approx(0.001, abs=1e-9)
    ids = [e["id"] for e in importlib.import_module(FAMM).PRIMARY if e["claim"] == "A"]
    FAM = importlib.import_module(FAMM)
    stat = {x: {"status": s["status"], "config": s.get("config")} for x, s in out["statuses"].items()}
    assert FAM.claim_decision("A", {i: "PASS" for i in ids}, stat)["decision"] != "PASS"
    assert out["deployable_best"]["config"] != "NORM-J|r1.5|a1"


def test_selection_U_invalid_invalidates_everything(monkeypatch, tmp_path):
    accs = {"U": (0.77, 0.32)}                                       # U gain over the constant < 0.03
    out = sel_world(monkeypatch, tmp_path, {}, accs, REFS)
    assert not out["U_valid"]
    assert all(s["status"] == "INVALID" for s in out["statuses"].values())
    assert out["deployable_best"]["config"] == "U" and "truthful" in out["deployable_best"]["as"]


def test_REQUIRED_S1_missing_L_star_invalidates_dependent_nominees(monkeypatch, tmp_path):
    """Section 13: N* requires local AUC <= L* + 0.005 and R* requires <= L* + 0.005 on every seed; 'missing comparators
    make affected claims invalid'. With NO task-feasible local configuration (L* absent), N* is picked against C* only
    and R* against no guard at all, both reported as NOMINEE; Claim B (N* vs C*) can then PASS and an unguarded R*/N*
    can become the deployable best."""
    FAM = importlib.import_module(FAMM)
    aucs = {"NORM-J|r3|a0.5": (0.77, 0.79, 0.70), "RAW-J|b0.3": (0.78, 0.80, 0.75)}
    accs = {c: (0.80, 0.40) for c in [T.config_id(c) for c in T.bank("full")] if c.startswith(("RAW-L", "NORM-L"))}
    out = sel_world(monkeypatch, tmp_path, aucs, accs, REFS)
    st = out["statuses"]
    assert st["L*"]["status"] != "NOMINEE"
    stat = {x: {"status": s["status"], "config": s.get("config"), "missing_guards": s.get("missing_guards")}
            for x, s in st.items()}
    ids_b = [e["id"] for e in FAM.PRIMARY if e["claim"] == "B"]
    claim_b = FAM.claim_decision("B", {i: "PASS" for i in ids_b}, stat)["decision"]
    unguarded = {s.get("config") for x, s in st.items() if x in ("N*", "R*") and s.get("missing_guards")}
    assert claim_b != "PASS", "Claim B passes with an N* nominated without its required L* guard"
    assert out["deployable_best"]["config"] not in unguarded, "deployable best nominated without its required guard"


# ================================================================== 8. families, z, decisions, bootstrap (section 14)
def _inv_norm_upper(p):
    """x with P(Z > x) = p by bisection on erfc (independent of scipy)."""
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(mid / math.sqrt(2)) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_primary_family_27_slots_and_z():
    FAM = importlib.import_module(FAMM)
    z = _inv_norm_upper(0.05 / 54)
    assert abs(z - 3.113017) < 5e-7 and abs(FAM.Z_PRIMARY - z) < 1e-9
    assert FAM.PRIMARY_SIZE == 27 and len(FAM.PRIMARY) == 27
    want = {"A": ("N*", "L*"), "B": ("N*", "C*"), "C": ("R*", "L*")}
    for claim, (nom, ref) in want.items():
        es = [e for e in FAM.PRIMARY if e["claim"] == claim]
        assert len(es) == 9
        assert [(e["kind"], e.get("view"), e.get("task"), e["target"], e["side"]) for e in es] == [
            ("coalition", None, None, 0.02, "lower>"), ("local", "v1", None, 0.01, "upper<"),
            ("local", "v2", None, 0.01, "upper<"), ("acc", None, 0, -0.01, "lower>"), ("acc", None, 1, -0.01, "lower>"),
            ("retain", None, 0, 0.0, "lower>"), ("retain", None, 1, 0.0, "lower>"),
            ("useful", None, 0, 0.03, "lower>"), ("useful", None, 1, 0.03, "lower>")]
        assert all(e["nominee"] == nom for e in es) and all(e["ref"] == ref for e in es if "ref" in e)
        assert es[0]["stat"] == f"R_pair({ref}) - R_pair({nom})"            # comparator minus nominee
        assert es[1]["stat"] == f"R_v1({nom}) - R_v1({ref})"               # nominee minus comparator
    assert FAM.Z_SECONDARY == pytest.approx(_inv_norm_upper(0.05 / (2 * FAM.SECONDARY_SIZE)), abs=1e-9)


def test_strict_thresholds_tie_cannot_pass():
    INF = importlib.import_module(INFM)
    lo_e = {"side": "lower>", "target": 0.02}
    up_e = {"side": "upper<", "target": 0.01}
    assert INF.decide(lo_e, 0.02, 0.5) != "PASS" and INF.decide(lo_e, 0.0200001, 0.5) == "PASS"
    assert INF.decide(up_e, -1, 0.01) != "PASS" and INF.decide(up_e, -1, 0.0099999) == "PASS"


def test_gain_retention_statistic_algebra():
    """Clause 6-7 statistic Acc - 0.8 Acc(U) - 0.2 Acc(const) equals (Acc - const) - 0.8 (Acc(U) - const): > 0 iff the
    nominee keeps more than 80% of U's gain over the fitting-prior constant. The bootstrap graph's 'lin' op matches."""
    from jcv.infer import G
    rng = np.random.default_rng(4)
    a, u, c = rng.random(5), rng.random(5), rng.random(5)
    g = G()
    ids = [g.base_once(f"b{i}", f"b{i}", (lambda WT, x=x: np.atleast_1d(x)[:WT.shape[1]])) for i, x in
           enumerate((a, u, c))]
    s = g.add("lin", "lin", ids)
    bases, aggs = g.closure([s])
    v = g.evaluate(np.ones((1, 5)), bases, aggs)[s]
    np.testing.assert_allclose(v, a - 0.8 * u - 0.2 * c, rtol=0, atol=1e-15)
    np.testing.assert_allclose(v, (a - c) - 0.8 * (u - c), atol=1e-15)


def test_group_bootstrap_equals_explicit_group_resampling():
    """Replicates of the inherited weighted engine equal explicit resampling of exact-record GROUPS (rows of a group move
    together; a group with several rows is one draw), with the SAME draw for every statistic."""
    from sklearn.metrics import roc_auc_score
    from jcv.infer import G, acc_stat
    from stored_model_eval.bench_infer import class_auc, run
    from stored_model_eval.pilot_infer import UnitBootstrap
    rng = np.random.default_rng(11)
    sizes = rng.choice([1, 1, 1, 2, 3], size=120)
    units = np.repeat(np.arange(500, 620), sizes)
    n = len(units)
    sex = rng.integers(0, 2, n)
    s1, s2 = rng.normal(size=n) + 0.8 * sex, rng.normal(size=n) + 0.3 * sex
    corr = rng.random(n) < 0.7
    g = G()
    a1 = g.base_once("a1", "a1", class_auc(sex, np.stack([1 - s1, s1], 1), 1))
    a2 = g.base_once("a2", "a2", class_auc(sex, np.stack([1 - s2, s2], 1), 1))
    ac = g.base_once("ac", "ac", acc_stat(corr))
    d = g.add("d", "diff", [a1, a2])
    B, seed = 7, 20261006
    pts, reps = run(g, UnitBootstrap(units, B, seed, 3), [a1, a2, ac, d])
    assert pts[a1] == pytest.approx(roc_auc_score(sex, s1), abs=1e-12)
    uniq, inv = np.unique(units, return_inverse=True)
    r = np.random.default_rng(seed)
    p = np.full(len(uniq), 1 / len(uniq))
    for b in range(B):
        counts = r.multinomial(len(uniq), p)
        rows = np.concatenate([np.flatnonzero(inv == gi).repeat(cnt) for gi, cnt in enumerate(counts) if cnt])
        e1, e2 = roc_auc_score(sex[rows], s1[rows]), roc_auc_score(sex[rows], s2[rows])
        assert reps[a1][b] == pytest.approx(e1, abs=1e-12) and reps[a2][b] == pytest.approx(e2, abs=1e-12)
        assert reps[ac][b] == pytest.approx(corr[rows].mean(), abs=1e-12)
        assert reps[d][b] == pytest.approx(e1 - e2, abs=1e-12)


def _fake_outer(rng, n, units, sex, y_inc, y_occ, s_pair):
    out = {"assess_row_id": np.arange(n), "assess_unit": units, "sex": sex, "y_income": y_inc, "y_occ": y_occ,
           "hard1": np.where(rng.random(n) < 0.85, y_inc, 1 - y_inc),
           "hard2": np.where(rng.random(n) < 0.5, y_occ, (y_occ + 1) % 6)}
    for w in ("v1", "v2", "pair"):
        P = np.empty((3, n, 2))
        for a in range(3):
            q = 1 / (1 + np.exp(-((s_pair if w == "pair" else 0.4) * (2 * sex - 1) + rng.normal(size=n))))
            P[a] = np.stack([1 - q, q], 1)
        out[f"P_auc_{w}"] = P
        out[f"P_ce_{w}"] = P
    return out


def test_REQUIRED_S2_label_is_invalid_when_U_lacks_utility(monkeypatch, tmp_path):
    """Section 13 / 18: if U fails nontrivial utility the statuses are INVALID and the study label must be
    INCOMPLETE_OR_INVALID (separate from a complete negative). osf.infer.main calls FAM.overall_label(dec) with the
    default complete=True and never consults the lock's U_valid, so it reports EXPERIMENTAL_NO_ADVANTAGE."""
    from rgj import finalize as FN
    from osf import run as R
    FAM = importlib.import_module(FAMM)
    INF = importlib.import_module(INFM)
    EL = importlib.import_module("osf.eval_lock")
    monkeypatch.setattr(R, "UNITS", tmp_path / "units")
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path / "pkg")
    monkeypatch.setattr(INF.FAM, "B", 30)
    rng = np.random.default_rng(2)
    n = 300
    units = np.arange(n) // 3
    sex, y_inc, y_occ = rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 6, n)
    labels = ["U", "RAW-J|b0.3", "RAW-L|b0.3"]
    for k in SEEDS3:
        for j, lab in enumerate(labels):
            p = _fake_outer(np.random.default_rng([k, j]), n, units, sex, y_inc, y_occ, 0.5 + 0.2 * j)
            FN.save_unit(R.U(f"outer__s{k}__{INF.safe(lab)}"), {"preds.npz": lambda q, p=p: np.savez(q, **p)}, {"x": 1})
    D = OD.load()
    st = {x: {"status": "INVALID", "config": None, "reason": "U lacks nontrivial utility"} for x in ("L*", "C*", "N*", "R*")}
    lock = {"sex_prior_defense_fit_sha256": EL.prior_hash(D), "statuses": st, "U_valid": False,
            "resolved": {x: None for x in st}, "seeds": {str(k): {"score": {lab: {} for lab in labels}} for k in SEEDS3}}
    lp = tmp_path / "EL.json"
    lp.write_text(__import__("json").dumps(lock))
    out = INF.main(["--evaluation-lock", str(lp)])
    assert out["label"] == "INCOMPLETE_OR_INVALID", out["label"]


def test_infer_primary_endpoints_match_independent_group_bootstrap(monkeypatch, tmp_path):
    """osf.infer end to end on synthetic outer predictions (duplicated exact-record groups) versus an independent
    recomputation: per seed, recovery = mean over attacker refits of the SEX AUC; per-seed paired difference; mean over
    seeds; SE = sd (ddof 1) over B replicates of multinomial GROUP resampling with seed 20261006, the same draw for
    every arm and seed. Checks P19 (coalition), P20 (local), P22 (accuracy vs U), P24 (gain retention), P26 (gain)."""
    from sklearn.metrics import roc_auc_score
    from rgj import finalize as FN
    from osf import run as R
    INF = importlib.import_module(INFM)
    EL = importlib.import_module("osf.eval_lock")
    monkeypatch.setattr(R, "UNITS", tmp_path / "units")
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path / "pkg")
    B = 25
    monkeypatch.setattr(INF.FAM, "B", B)
    rng = np.random.default_rng(5)
    sizes = rng.choice([1, 1, 2, 3], size=110)
    units = np.repeat(np.arange(110) * 7 + 3, sizes)
    n = len(units)
    sex, y_inc, y_occ = rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 6, n)
    roles = {"N*": "NORM-J|r3|a1", "R*": "RAW-J|b0.3", "L*": "RAW-L|b0.3", "C*": "F0"}
    labels = ["U"] + list(roles.values())
    preds = {}
    for k in SEEDS3:
        for j, lab in enumerate(labels):
            p = _fake_outer(np.random.default_rng([k, j, 9]), n, units, sex, y_inc, y_occ, 0.3 + 0.25 * j)
            preds[(k, lab)] = p
            FN.save_unit(R.U(f"outer__s{k}__{INF.safe(lab)}"), {"preds.npz": lambda q, p=p: np.savez(q, **p)}, {"x": 1})
    D = OD.load()
    st = {x: {"status": "NOMINEE", "config": c} for x, c in roles.items()}
    st["N*"] = {"status": "NO_FEASIBLE_NOMINEE", "config": None, "descriptive_config": roles["N*"]}   # fallback
    lock = {"sex_prior_defense_fit_sha256": EL.prior_hash(D), "statuses": st, "U_valid": True, "resolved": dict(roles),
            "seeds": {str(k): {"score": {lab: {} for lab in labels}} for k in SEEDS3}}
    lp = tmp_path / "EL.json"
    lp.write_text(__import__("json").dumps(lock))
    out = INF.main(["--evaluation-lock", str(lp)])
    prim = {e["id"]: e for e in out["primary"]}
    tr = D["idx"]["OSF_DEFENSE_FIT"]
    const_inc = int(np.argmax(np.bincount(D["y_income"][tr])))

    def stats(rows):
        def R_(k, lab, w):
            P = preds[(k, lab)][f"P_auc_{w}"]
            return np.mean([roc_auc_score(sex[rows], P[s][rows, 1]) for s in range(3)])

        def A_(k, lab):
            return np.mean(preds[(k, lab)]["hard1"][rows] == y_inc[rows])
        c = np.mean(y_inc[rows] == const_inc)
        Rs, Ls = roles["R*"], roles["L*"]
        return {"P19": np.mean([R_(k, Ls, "pair") - R_(k, Rs, "pair") for k in SEEDS3]),
                "P20": np.mean([R_(k, Rs, "v1") - R_(k, Ls, "v1") for k in SEEDS3]),
                "P22": np.mean([A_(k, Rs) - A_(k, "U") for k in SEEDS3]),
                "P24": np.mean([A_(k, Rs) - 0.8 * A_(k, "U") - 0.2 * c for k in SEEDS3]),
                "P26": np.mean([A_(k, Rs) - c for k in SEEDS3])}
    pt = stats(np.arange(n))
    uniq, inv = np.unique(units, return_inverse=True)
    r = np.random.default_rng(20261006)
    reps = []
    for b in range(B):
        cnt = r.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))
        rows = np.concatenate([np.flatnonzero(inv == gi).repeat(c) for gi, c in enumerate(cnt) if c])
        reps.append(stats(rows))
    # review A7 (applied): clauses of the descriptive N* (Claims A, B) are DESCRIPTIVE_ONLY; Claim C stays numeric
    assert all(prim[f"P{i:02d}"]["decision"] == "DESCRIPTIVE_ONLY" for i in range(1, 19))
    assert all(prim[f"P{i:02d}"]["decision"] in ("PASS", "NOT_ESTABLISHED") for i in range(19, 28))
    assert out["claimA"]["decision"] != "PASS" and out["claimB"]["decision"] != "PASS"
    assert out["label"] in ("EXPERIMENTAL_NO_ADVANTAGE", "RAW_JOINT_DEVELOPMENT_CRITERION_MET")
    for pid in ("P19", "P20", "P22", "P24", "P26"):
        assert prim[pid]["point"] == pytest.approx(pt[pid], abs=1e-12), pid
        se = float(np.std([x[pid] for x in reps], ddof=1))
        assert prim[pid]["se"] == pytest.approx(se, rel=1e-9, abs=1e-12), pid
        z = prim[pid]["z"]
        assert z == pytest.approx(_inv_norm_upper(0.05 / 54), abs=1e-9)
        assert prim[pid]["lower"] == pytest.approx(pt[pid] - z * se, abs=1e-9)
