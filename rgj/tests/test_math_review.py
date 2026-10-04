"""Mathematics-and-design review fixtures for the refreshed guarded joint study (owned by the math reviewer).

Small synthetic tensors only (no study rows, no scientific fits). Every test is built so that it can FAIL on a real
defect: wrong transport order or sign, a surrogate that is not clamped or that uses the old 1 - CE/H offset, a dual
step with the wrong sign or clip, a coalition gradient that leaks into the local arm, a critic-input transform that
goes stale inside a refit block, a transform that discards a low-variance clue, bootstrap weights that are not
grouped, or a C* rule that drops a feasible control. See results/pcrl_refreshed_guarded_joint_v1/MATH_REVIEW.md.

    env OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m pytest rgj/tests/test_math_review.py -q
"""
from __future__ import annotations

import copy
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.stats import norm

try:
    from rgj import train as T
except Exception:  # pragma: no cover - engine not yet importable
    T = None

needs_T = pytest.mark.skipif(T is None, reason="rgj.train not importable")
torch.set_num_threads(1)


# ------------------------------------------------------------------ synthetic helpers (no study rows)
class SynthData:
    """DEFENSE_FIT-shaped synthetic data: 83 inputs, SEX prior about 0.67, income and 6-class occupation that depend on
    the inputs and on SEX, fixed CRITIC_FIT / CRITIC_VAL / CALIB index sets and the online reference subset."""

    def __init__(self, n=1500, d=83, seed=0, sex_strength=1.0):
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


def model_from(state, seed=0):
    m = T.Model(T.D_IN, T.KS, seed)
    m.load_state_dict(state)
    return m


def logit_views(R, W, b):
    """[r, centred logits of an affine head] in float64 numpy: the critic-view structure with K exact null directions."""
    lg = R @ W.T + b
    return np.hstack([R, lg - lg.mean(1, keepdims=True)])


def two_snapshots(rng, n=3000, dr=6, K=3, head_shift=0.02):
    """Old/new view snapshots: encoder drift plus a training-head change between them (amplified-direction case)."""
    A0 = rng.normal(size=(dr, dr))
    R0 = rng.normal(size=(n, dr)) @ A0 + 2.0
    W0, b0 = rng.normal(size=(K, dr)), rng.normal(size=K)
    R1 = R0 @ (np.eye(dr) + 0.05 * rng.normal(size=(dr, dr))) + 0.1
    W1, b1 = W0 + head_shift * rng.normal(size=(K, dr)), b0 + head_shift * rng.normal(size=K)
    V0 = torch.tensor(logit_views(R0, W0, b0), dtype=torch.float32)
    V1 = torch.tensor(logit_views(R1, W1, b1), dtype=torch.float32)
    return V0, V1


def first_layer_preacts64(c, Tr, V):
    """Float64 first-layer pre-activations of critic c on transform Tr applied to V (no float32 rounding)."""
    z = (V.double() - Tr.mu64) @ Tr.W64
    A, b = c[0].weight.detach().double(), c[0].bias.detach().double()
    return z @ A.T + b, z.abs() @ A.abs().T + b.abs()


# ------------------------------------------------------------------ 1. family critical value
def test_z_value_for_18_slot_two_sided_bonferroni():
    z = norm.ppf(1 - 0.05 / (2 * 18))
    assert round(float(z), 6) == 2.991316
    assert abs(z - norm.isf(0.05 / 36)) < 1e-12
    # a 9-slot (one claim only) or one-sided family gives a different, smaller z: the 18-slot value must not shrink
    assert norm.ppf(1 - 0.05 / 18) < z and norm.ppf(1 - 0.05 / 36 * 2) < z
    try:
        from rgj import family as FAM
    except Exception:
        return
    size = getattr(FAM, "PRIMARY_SIZE", None)
    if size is not None:
        assert size == 18
    zp = getattr(FAM, "Z_PRIMARY", None)
    if zp is not None:
        assert abs(zp - z) < 1e-12


# ------------------------------------------------------------------ 2. continued-critic transport
@needs_T
@pytest.mark.parametrize("kind", ["floored", "capped", "raw"])
def test_transport_row_column_convention_exact(kind):
    """c_new(T_new(V)) == c_old(T_old(V)) for every V (on and off the data manifold), with non-commuting W_old, W_new
    and a mean shift. Tolerance: float32 storage of A', b' (2^-24 relative to the pre-activation's absolute scale)."""
    rng = np.random.default_rng(1)
    V0, V1 = two_snapshots(rng)
    To, Tn = T.Transform(V0, kind), T.Transform(V1, kind)
    torch.manual_seed(0)
    c_old = T.critic("B", V0.shape[1])
    c_new = T.transport_first_layer(copy.deepcopy(c_old), To, Tn)
    Voff = torch.tensor(rng.normal(size=(400, V0.shape[1])) * 3, dtype=torch.float32)
    for Ve in (V1[:500], V0[:500], Voff):
        y_old, _ = first_layer_preacts64(c_old, To, Ve)
        y_new, scale = first_layer_preacts64(c_new, Tn, Ve)
        assert float(((y_new - y_old).abs() / scale).max()) < 1e-6
    # sensitivity: the swapped-order and inverse-direction formulas are measurably wrong on this case
    if kind != "raw":
        A = c_old[0].weight.detach().double()
        wrong = A @ torch.linalg.inv(Tn.W64) @ To.W64
        right = c_new[0].weight.detach().double()
        assert float((wrong - right).abs().max()) > 1e-3 * float(right.abs().max())


@needs_T
def test_transport_identity_when_transform_unchanged():
    rng = np.random.default_rng(2)
    V0, _ = two_snapshots(rng)
    To = T.Transform(V0, "floored")
    torch.manual_seed(0)
    c = T.critic("A", V0.shape[1])
    w0, b0 = c[0].weight.detach().clone(), c[0].bias.detach().clone()
    T.transport_first_layer(c, To, T.Transform(state=To.state()))
    assert torch.allclose(c[0].weight, w0, rtol=1e-5, atol=1e-6 * float(w0.abs().max()))
    assert torch.allclose(c[0].bias, b0, rtol=1e-5, atol=1e-6 * (1 + float(b0.abs().max())))


@needs_T
def test_transport_amplified_direction_float32_path():
    """Floored ZCA amplifies the exact null directions of [r, centred logits] by ~1e3-1e4. The float32 training path of
    the transported critic must still reproduce the old critic's logits to a small fraction of the logit scale."""
    rng = np.random.default_rng(0)
    V0, V1 = two_snapshots(rng)
    To, Tn = T.Transform(V0, "floored"), T.Transform(V1, "floored")
    amp = To.spectrum()["W_sv_max"] / To.spectrum()["W_sv_min"]
    assert amp > 1e3, "fixture must exercise the amplified-direction case"
    torch.manual_seed(0)
    c_old = T.critic("B", V0.shape[1])
    c_new = T.transport_first_layer(copy.deepcopy(c_old), To, Tn)
    with torch.no_grad():
        ref64 = copy.deepcopy(c_old).double()((V1.double() - To.mu64) @ To.W64)
        new32 = c_new(Tn(V1)).double()
    err = float((new32 - ref64).abs().max())
    assert err <= 5e-3 * (1.0 + float(ref64.abs().max())), err
    assert torch.isfinite(c_new[0].weight).all() and torch.isfinite(c_new[0].bias).all()


# ------------------------------------------------------------------ 3. the recovery surrogate
def _prior_logits(prior, n):
    return torch.log(torch.tensor(prior, dtype=torch.float32)).expand(n, 2).clone()


@needs_T
def test_constant_critic_gives_zero_recovery_and_no_gradient():
    rng = np.random.default_rng(3)
    prior = np.array([0.33, 0.67])
    H = T.entropy(prior)
    logprior = torch.tensor(np.log(prior), dtype=torch.float32)
    for share in (0.50, 0.67, 0.90):                     # batch SEX share different from the fitting prior
        S = torch.from_numpy((rng.random(256) < share).astype(np.int64))
        V = torch.randn(256, 5, requires_grad=True)
        crit = torch.nn.Linear(5, 2)
        with torch.no_grad():
            crit.weight.zero_()
            crit.bias.copy_(logprior)                    # constant critic = fitting prior
        R, j, _ = T.recovery([crit(V), crit(V) * 1.0], S, logprior, H)
        assert abs(float(R)) < 1e-6        # float32 log-softmax of the prior logits may tie or beat CE_const by ~1e-8
        if R.requires_grad:      # a tie may select the (constant) critic; its input gradient must still be exactly 0
            g = torch.autograd.grad(R, V, allow_unused=True)[0]
            assert g is None or float(g.abs().max()) == 0.0
        # the old minibatch convention 1 - CE/H would be nonzero here (offset of label composition)
        ce_const = float(F.nll_loss(logprior.expand(256, 2), S))
        if abs(share - 0.67) > 0.05:
            assert abs(1 - ce_const / H) > 1e-3


@needs_T
def test_worse_than_constant_critics_clamp_to_zero_without_gradient():
    """A critic worse than the constant must not produce negative recovery or any encoder gradient (catches a min that
    omits the constant, or an unclamped surrogate)."""
    prior = np.array([0.33, 0.67])
    H = T.entropy(prior)
    logprior = torch.tensor(np.log(prior), dtype=torch.float32)
    torch.manual_seed(4)
    S = torch.from_numpy((np.random.default_rng(4).random(256) < 0.67).astype(np.int64))
    V = torch.randn(256, 5, requires_grad=True)
    bad = torch.nn.Linear(5, 2)
    with torch.no_grad():
        bad.weight.mul_(20.0)                             # input-dependent, badly calibrated
    R, j, vals = T.recovery([bad(V), -bad(V)], S, logprior, H)
    assert min(vals[:2]) > vals[2], "fixture: both critics must be worse than the constant"
    assert float(R) == 0.0 and j == 2
    if R.requires_grad:
        g = torch.autograd.grad(R, V, allow_unused=True)[0]
        assert g is None or float(g.abs().max()) == 0.0


@needs_T
def test_recovery_value_and_gradient_come_from_selected_critic_only():
    prior = np.array([0.33, 0.67])
    H = T.entropy(prior)
    logprior = torch.tensor(np.log(prior), dtype=torch.float32)
    rng = np.random.default_rng(5)
    S = torch.from_numpy((rng.random(512) < 0.67).astype(np.int64))
    V = torch.randn(512, 4)
    V[:, 0] += 1.5 * (2 * S.float() - 1)
    V.requires_grad_(True)
    good, weak = torch.nn.Linear(4, 2), torch.nn.Linear(4, 2)
    with torch.no_grad():
        good.weight.zero_(); good.weight[1, 0] = 1.5; good.bias.copy_(logprior)
        weak.weight.zero_(); weak.weight[1, 0] = 0.2; weak.bias.copy_(logprior)
    R, j, vals = T.recovery([weak(V), good(V)], S, logprior, H)
    ce_good = F.cross_entropy(good(V), S)
    ce_const = F.nll_loss(logprior.expand(512, 2), S)
    assert j == 1
    assert abs(float(R) - float((ce_const - ce_good) / H)) < 1e-6
    g = torch.autograd.grad(R, V, retain_graph=True)[0]
    g_ref = torch.autograd.grad(-ce_good / H, V)[0]
    assert torch.allclose(g, g_ref, atol=1e-7)


@needs_T
def test_privacy_step_reduces_readable_signal_and_critic_learns_sex():
    """Critic learns SEX (bounded refit beats the constant); one encoder step along -grad R lowers R for that critic
    on the same rows (encoder sign)."""
    data = SynthData(n=1500, seed=6)
    st = warm_state(data, epochs=2)
    model = model_from(st)
    H = T.entropy(data.prior)
    logprior = torch.tensor(np.log(data.prior), dtype=torch.float32)
    cf, cv = torch.from_numpy(data.cf), torch.from_numpy(data.cv)
    head = T.head_of(st)
    V = T.frozen_views(model, data.X, head)
    Tr = T.Transform(V["v1"][cf], "floored")
    torch.manual_seed(0)
    c = T.critic("A", T.DV["v1"])
    c, rec = T.fit_bounded(c, Tr(V["v1"][cf]), data.S[cf], Tr(V["v1"][cv]), data.S[cv], [0, 1])
    ce_const = float(F.nll_loss(logprior.expand(len(cv), 2), data.S[cv]))
    assert rec["best_val_ce"] < ce_const - 0.02, "critic did not learn SEX"
    enc = [p for p in model.enc[0].parameters()]
    for p in enc:
        p.requires_grad_(True)
    b = torch.from_numpy(data.cf[:512])
    v1 = T.critic_views(model, data.X[b], ["v1"], head, grad=True)["v1"]
    R0, j, _ = T.recovery([c(Tr(v1))], data.S[b], logprior, H)
    assert j == 0 and float(R0) > 0
    g = torch.autograd.grad(R0, enc)
    with torch.no_grad():
        for p, gi in zip(enc, g):
            p.sub_(1e-3 * gi)
        v1b = T.critic_views(model, data.X[b], ["v1"], head)["v1"]
        R1, _, _ = T.recovery([c(Tr(v1b))], data.S[b], logprior, H)
    assert float(R1) < float(R0)


# ------------------------------------------------------------------ 4. multipliers and coefficients
@needs_T
@pytest.mark.parametrize("beta", [0.03, 0.1, 0.3])
def test_dual_update_sign_clip_and_cap(beta):
    c = 0.20
    assert T.dual_update(0.0, c + 0.05, c, beta) == pytest.approx(beta * 0.05)       # rises when R > c, eta = beta
    assert T.dual_update(0.0, c - 0.05, c, beta) == 0.0                              # stays at 0 when R < c
    assert T.dual_update(beta, c - 0.05, c, beta) == pytest.approx(beta - beta * 0.05)  # relaxes, not below 0
    assert T.dual_update(0.01 * beta, c - 0.5, c, beta) == 0.0
    assert T.dual_update(2.9 * beta, c + 1.0, c, beta) == pytest.approx(3 * beta)    # capped at 3 beta
    lam = 0.0
    for _ in range(100):
        lam = T.dual_update(lam, c + 0.5, c, beta)
    assert lam == pytest.approx(3 * beta)


@needs_T
def test_reachable_multiplier_range_with_registered_schedule():
    """Documented property behind MATH_REVIEW A1: with eta = beta and the updates that can still act on training
    (refits 0, 4, 8, 12, 16), lambda / beta <= sum of violations; the 3 beta cap needs a mean violation >= 0.6."""
    beta = 0.1
    for viol in (0.01, 0.03, 0.1):
        lam = 0.0
        for _ in range(5):
            lam = T.dual_update(lam, 0.4 + viol, 0.4, beta)
        assert lam == pytest.approx(5 * beta * viol)
        assert lam < 3 * beta


@needs_T
@pytest.mark.parametrize("beta", [0.0, 0.03, 0.1, 0.3])
def test_base_coefficient_sums_match_and_local_pair_is_zero(beta):
    j, l = T.base_coefficients("joint", beta), T.base_coefficients("local", beta)
    assert sum(j.values()) == pytest.approx(beta) and sum(l.values()) == pytest.approx(beta)
    assert j["v1"] == j["v2"] == j["pair"] == pytest.approx(beta / 3)
    assert l["v1"] == l["v2"] == pytest.approx(beta / 2) and l["pair"] == 0.0
    for lam in ((0.0, 0.0), (0.01, 0.2), (0.3, 0.0)):
        cl, cj = T.coefficients("local", beta, lam), T.coefficients("joint", beta, lam)
        assert cl["pair"] == 0.0 and cj["pair"] == pytest.approx(beta / 3)
        assert cl["v1"] - l["v1"] == pytest.approx(lam[0]) and cj["v2"] - j["v2"] == pytest.approx(lam[1])
    assert sum(T.base_coefficients("none", beta).values()) == 0.0


def _init_critics(model, data, seed, head, pair_tag="init"):
    V = T.frozen_views(model, data.X, head)
    cf = torch.from_numpy(data.cf)
    crit = {v: {k: T.new_critic(seed, v, k, pair_tag if v == "pair" else "init").state_dict() for k in T.KINDS}
            for v in T.VIEWS}
    return {"critics": crit, "transforms": {v: T.Transform(V[v][cf], "floored").state() for v in T.VIEWS}}


@needs_T
def test_local_arm_coalition_bank_never_reaches_the_encoder(monkeypatch):
    """Changing only the pair (shadow) bank must leave L-G's encoder bitwise unchanged (with and without refits), and
    must change J-G's encoder (sensitivity of the fixture)."""
    data = SynthData(n=1500, seed=7)
    st = warm_state(data, epochs=2)
    m0 = model_from(st)
    head = T.head_of(st)
    ic_a, ic_b = _init_critics(m0, data, 0, head, "init"), _init_critics(m0, data, 0, head, "other-pair")
    bud = (0.0, 0.0)

    def enc_state(arm, ic):
        m, diag, _, _, _ = T.train_run(arm, 0.3, st, data, 0, "C", init_critics=ic, budgets=bud, n_epochs=1,
                                       ckpt_epochs=(), critic_head=head)
        return {k: v for k, v in m.state_dict().items()}, diag

    sa, da = enc_state("L-G", ic_a)
    sb, db = enc_state("L-G", ic_b)
    assert all(torch.equal(sa[k], sb[k]) for k in sa), "pair bank changed the local arm's encoder"
    assert da["critic_online_updates"] == db["critic_online_updates"] > 0
    monkeypatch.setitem(T.HP, "refit_epochs", ())        # keep the inherited pair critics in use (no refit)
    sa, _ = enc_state("L-G", ic_a)
    sb, _ = enc_state("L-G", ic_b)
    assert all(torch.equal(sa[k], sb[k]) for k in sa), "pair bank changed the local arm's encoder (no refit)"
    ja, _ = enc_state("J-G", ic_a)
    jb, _ = enc_state("J-G", ic_b)
    assert not all(torch.equal(ja[k], jb[k]) for k in ja), "fixture insensitive: J-G should depend on its pair bank"


@needs_T
def test_guarded_multiplier_rises_only_for_the_violating_recipient():
    data = SynthData(n=1500, seed=8)
    st = warm_state(data, epochs=2)
    beta = 0.3
    _, diag, _, _, _ = T.train_run("L-G", beta, st, data, 0, "C", budgets=(0.0, 5.0), n_epochs=1, ckpt_epochs=(),
                                   critic_head=T.head_of(st))
    tr = diag["lambda_trace"][0]
    R1, R2 = tr["R_calib"]
    assert R1 > 0
    assert tr["lambda"][0] == pytest.approx(min(beta * R1, 3 * beta)) and tr["lambda"][1] == 0.0
    assert diag["epochs"][0]["lambda"] == tr["lambda"]
    assert tr["effective_weights"]["v1"] == pytest.approx(beta / 2 + tr["lambda"][0])
    assert tr["effective_weights"]["pair"] == 0.0


@needs_T
@pytest.mark.parametrize("arm", ["L-R", "L-G", "J-G", "J-O"])
def test_beta0_identity_against_task_line(arm):
    data = SynthData(n=1200, seed=9)
    st = warm_state(data, epochs=1)
    tm, _, _ = T.task_line(st, data, 0, 1, stage="C", save_every=1)
    m, diag, _, _, _ = T.train_run(arm, 0.0, st, data, 0, "C", budgets=(0.0, 0.0), n_epochs=1, ckpt_epochs=(),
                                   critic_head=T.head_of(st))
    assert all(torch.equal(m.state_dict()[k], tm.state_dict()[k]) for k in tm.state_dict())
    assert diag["critic_online_updates"] > 0


# ------------------------------------------------------------------ 5. refit block: transform validity and alignment
@pytest.fixture(scope="module")
def block_run():
    if T is None:
        pytest.skip("rgj.train not importable")
    data = SynthData(n=19230, seed=0)                  # admitted DEFENSE_FIT size: 76 encoder steps per epoch
    st = warm_state(data, epochs=5)
    steps = int(np.ceil(data.n / 256))
    m, diag, ck, cap, fin = T.train_run("L-R", 0.0, st, data, 0, "B", n_epochs=1, ckpt_epochs=(1,),
                                        capture_steps=(1, 20, steps), critic_head=T.head_of(st))
    return {"data": data, "diag": diag, "ck": ck, "cap": cap, "fin": fin, "steps": steps}


def _snap_eval(snap, data, rows):
    model = model_from(snap["model"])
    V = T.frozen_views(model, data.X[torch.from_numpy(rows)], snap["critic_head"])
    out = {}
    for v in T.VIEWS:
        Z = T.Transform(state=snap["transforms"][v])(V[v])
        ces = []
        for k in T.KINDS:
            c = T.critic(k, T.DV[v])
            c.load_state_dict(snap["critics"][v][k])
            ces.append(T.ce_of(c, Z, data.S[torch.from_numpy(rows)]))
        out[v] = {"maxabs": float(Z.abs().max()), "best_ce": min(ces)}
    return out


@needs_T
def test_block_transform_stays_valid_within_refit_block(block_run):
    """MATH_REVIEW R1. The refit transform is held for a whole block; the views the online critics and the penalty
    consume must stay in its coordinate system. With training-head drift the floored ZCA's exact null directions are
    excited and amplified ~1e3-1e4x: critic inputs explode and the online critics fall to or below the constant."""
    data, cap, steps = block_run["data"], block_run["cap"], block_run["steps"]
    e1, e20, eT = (_snap_eval(cap[s], data, data.cf) for s in (1, 20, steps))
    for v in T.VIEWS:
        assert e20[v]["maxabs"] < 3 * e1[v]["maxabs"], (v, e1[v]["maxabs"], e20[v]["maxabs"])
        assert eT[v]["maxabs"] < 3 * e1[v]["maxabs"], (v, e1[v]["maxabs"], eT[v]["maxabs"])
    prior_ce = T.entropy(data.prior)
    v_cv = _snap_eval(cap[steps], data, data.cv)
    for v in T.VIEWS:
        assert v_cv[v]["best_ce"] < prior_ce - 0.02, (v, v_cv[v]["best_ce"], prior_ce)


@needs_T
def test_final_snapshot_alignment_labels(block_run):
    """theta_T_minus_1 holds the model state BEFORE the last encoder step together with the online critics and the
    transform they were last trained with; theta_T is the released state; the epoch checkpoint holds theta_T but the
    online critics of theta_{T-1} (so a gap diagnosis must never pair checkpoint['model'] with checkpoint['critics'])."""
    fin, ck, cap, steps = block_run["fin"], block_run["ck"], block_run["cap"], block_run["steps"]
    a, b = fin["theta_T_minus_1"], fin["theta_T"]
    assert any(not torch.equal(a["model"][k], b[k]) for k in b)
    assert all(torch.equal(ck[1]["model"][k], b[k]) for k in b)
    for v in T.VIEWS:
        for k in T.KINDS:
            for p in a["critics"][v][k]:
                assert torch.equal(a["critics"][v][k][p], ck[1]["critics"][v][k][p])
        assert torch.equal(a["transforms"][v]["W"], cap[1]["transforms"][v]["W"])   # one block transform
    assert all(torch.equal(cap[steps]["model"][k], a["model"][k]) for k in b)


# ------------------------------------------------------------------ 6. planted 1e-6 clue (section 8)
def _zca64(A, rows):
    mu = A[rows].mean(0)
    ev, U = np.linalg.eigh(np.cov((A[rows] - mu).T))
    return lambda B: (B - mu) @ (U / np.sqrt(ev)) @ U.T


def _planted(n=8000, d=12, seed=0):
    rng = np.random.default_rng(seed)
    S = (rng.random(n) < 0.67).astype(np.int64)
    X = rng.normal(size=(n, d))
    X[:, d - 1] = X[:, d - 2] + 1e-6 * (2 * S - 1) + 1e-7 * rng.normal(size=n)   # rotated low-variance direction
    return torch.from_numpy(X.astype(np.float32)), S


def _oracle_auc(Z, S, fit, te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    Zw = _zca64(Z, fit)(Z)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(Zw[fit], S[fit])
    return roc_auc_score(S[te], lr.predict_proba(Zw[te])[:, 1])


@needs_T
@pytest.mark.parametrize("kind", ["floored", "capped", "raw"])
def test_planted_1e6_clue_survives_each_transform(kind):
    """Amplitude-1e-6 SEX clue along a rotated low-variance direction. After the critic-input transform (float32 path
    the critics actually receive) a suitably scaled classifier (exact float64 re-whitening, no floor) still detects it,
    and the transform is invertible on the clue direction (equivalently transported classifier)."""
    V, S = _planted()
    fit, te = np.arange(6000), np.arange(6000, len(S))
    Tr = T.Transform(V[torch.from_numpy(fit)], kind)
    Z = Tr(V).double().numpy()                       # float32 critic-input path
    assert _oracle_auc(Z, S, fit, te) > 0.95
    X64 = V.double().numpy()
    Z64 = (X64 - Tr.mu64.numpy()) @ Tr.W64.numpy()   # algebraic invertibility (float64 factors)
    back = Z64 @ np.linalg.inv(Tr.W64.numpy()) + Tr.mu64.numpy()
    d = np.zeros(V.shape[1]); d[-1], d[-2] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    clue_err = np.abs((back - X64) @ d).max()
    assert clue_err < 1e-9, clue_err


def test_planted_clue_fixture_rejects_a_truncating_transform():
    """Negative control: a whitening that drops directions with variance <= 1e-9 x max (as a rank-truncating
    canonicalisation would) makes the clue undetectable, so the planted test above can fail."""
    V, S = _planted()
    fit, te = np.arange(6000), np.arange(6000, len(S))
    X = V.double().numpy()
    mu = X[fit].mean(0)
    ev, U = np.linalg.eigh(np.cov((X[fit] - mu).T))
    keep = ev > 1e-9 * ev.max()
    Z = (X - mu) @ U[:, keep] / np.sqrt(ev[keep])
    assert _oracle_auc(Z, S, fit, te) < 0.6


# ------------------------------------------------------------------ 7. bootstrap grouping
def test_bootstrap_resamples_record_groups_and_seeds_add_no_records():
    from stored_model_eval.pilot_infer import UnitBootstrap
    rng = np.random.default_rng(11)
    G = 400
    sizes = rng.choice([1, 2, 3], size=G, p=[0.8, 0.15, 0.05])
    units = np.repeat(np.arange(G) * 7 + 3, sizes)
    x = np.repeat(rng.normal(size=G), sizes)                 # exact duplicates share their value
    boot = UnitBootstrap(units, 300, 20261004, 100)
    W = np.concatenate(list(boot.chunks()), axis=1)
    for u in np.unique(units)[:50]:
        rows = np.flatnonzero(units == u)
        assert np.all(W[rows] == W[rows[0]])
    assert np.allclose(W.sum(0)[:5] / len(units), W.sum(0)[:5] / len(units))
    stat = (x @ W) / W.sum(0)
    seeds = np.stack([x, x, x])                              # three identical seed outputs on the same rows
    paired = np.mean([(s @ W) / W.sum(0) for s in seeds], axis=0)
    assert np.allclose(paired, stat)                         # paired seed-averaging: no extra records, same SE
    stacked_units = np.concatenate([units + 10 ** 6 * k for k in range(3)])
    Ws = np.concatenate(list(UnitBootstrap(stacked_units, 300, 20261004, 100).chunks()), axis=1)
    xs = np.concatenate([x, x, x])
    se_wrong = np.std((xs @ Ws) / Ws.sum(0), ddof=1)
    assert se_wrong < 0.75 * np.std(stat, ddof=1)            # treating seeds as people would shrink the SE ~sqrt(3)


# ------------------------------------------------------------------ 8. stage-C selection rules
def _rec(acc, auc):
    return {"utility": {"0": {"acc": acc[0], "const_acc": 0.75}, "1": {"acc": acc[1], "const_acc": 0.30}},
            "recovery": {"auc": {"v1": auc[0], "v2": auc[1], "pair": auc[2]}}}


@pytest.fixture
def stage_c_world(monkeypatch, tmp_path):
    try:
        from rgj import run as R
        from rgj import select as SEL
    except Exception as e:  # pragma: no cover
        pytest.skip(f"rgj.select not importable: {e}")
    recs = {}
    good = (0.845, 0.598)
    recs["tl__s0__e40"] = _rec((0.850, 0.600), (0.720, 0.720, 0.780))        # U (fails the local guard)
    recs["tl__s0__e20"] = _rec(good, (0.700, 0.700, 0.750))                  # L-R = U-B alias
    recs["lc__s0__E"] = _rec(good, (0.690, 0.690, 0.800))
    lo = R.ck_name("B", 0, "L-O", 0.1, 10)
    recs[lo] = _rec(good, (0.705, 0.700, 0.760))
    pair = {"L-G": 0.790, "J-R": 0.785, "J-O": 0.790}
    for arm, p in pair.items():
        for b in R.BETAS:
            for e in SEL.CKPTS:
                recs[R.ck_name("C", 0, arm, b, e)] = _rec(good, (0.700, 0.690, p + 0.001 * e + 0.01 * b))
    for b in R.BETAS:
        for e in SEL.CKPTS:
            recs[R.ck_name("C", 0, "J-G", b, e)] = _rec(good, (0.700, 0.700, 0.700 + 0.001 * e))
    recs[R.ck_name("C", 0, "J-G", 0.3, 5)] = _rec(good, (0.699, 0.689, 0.690))  # only one meets the min guard
    selB = {"0": {"L-R": {"status": "TASK_ONLY_ALIAS", "unit": "tl__s0__e20", "beta": 0.0, "epoch": 20,
                          "auc": {"v1": 0.700, "v2": 0.700, "pair": 0.750}},
                  "L-O": {"status": "NOMINEE", "unit": lo, "beta": 0.1, "epoch": 10,
                          "auc": {"v1": 0.705, "v2": 0.700, "pair": 0.760}}}}
    monkeypatch.setattr(SEL, "inner", lambda name: recs[name])
    monkeypatch.setattr(R, "selB", lambda: selB)
    monkeypatch.setattr(R, "lr_init", lambda k: (None, None, "tl__s0__e20", 20))
    monkeypatch.setattr(R, "SEEDS", (0,))
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path)                 # select_C also writes tables to the package dir
    fake = types.ModuleType("rgj.baselines")
    fake.select_fare = lambda k, D, units, uref, guard: {
        "F": {"status": "NO_FEASIBLE_NOMINEE", "feasible": False, "auc": {"v1": 0.8, "v2": 0.8, "pair": 0.7}},
        "F0": {"status": "GATES_FAILED", "feasible": False, "auc": {"v1": 0.8, "v2": 0.8, "pair": 0.7}},
        "per_purpose": None}
    monkeypatch.setitem(sys.modules, "rgj.baselines", fake)
    import rgj
    monkeypatch.setattr(rgj, "baselines", fake, raising=False)
    return SEL.select_C(None)["0"]


def test_trained_controls_use_zero_buffer_coalition_rule(stage_c_world):
    res = stage_c_world
    for arm in ("L-G", "J-R", "J-O"):
        a = res["arms"][arm]
        assert a["status"] == "NOMINEE"
        assert a["beta"] == min(T.HP["betas"]) and a["epoch"] == min(T.HP["checkpoints"])
        assert a["auc"]["v1"] <= 0.700 and a["auc"]["v2"] <= 0.700


def test_cstar_considers_every_feasible_control_including_task_only_alias(stage_c_world):
    """MATH_REVIEW R2. PROMPT 7: C* = lowest inner coalition AUC among feasible L-G, L-R, L-O, J-R, J-O, U, LEACE,
    FARE, F0. A feasible TASK_ONLY_ALIAS L-R (or L-O) is a feasible control (U itself is task-only and eligible);
    dropping it makes claim B's comparator weaker. Here L-R (alias, pair 0.750) must be C*, with alias status kept."""
    res = stage_c_world
    assert res["arms"]["L-R"].get("feasible", False)
    assert res["comparator"]["arm"] == "L-R", res["comparator"]
    assert res["comparator"]["is_task_only_release"] is True


def test_jg_nomination_guard_is_min_over_available_comparators(stage_c_world):
    res = stage_c_world
    g = res["arms"]["J-G"]["nomination_guard"]
    cs = res["comparator"]["arm"]
    for w in ("v1", "v2"):
        want = min(0.700, res["arms"]["L-G"]["auc"][w], res["arms"][cs]["auc"][w] if cs else 1.0)
        assert g[w] == pytest.approx(want)
    jg = res["arms"]["J-G"]
    assert jg["status"] == "NOMINEE" and jg["beta"] == 0.3 and jg["epoch"] == 5
