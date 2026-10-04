"""Pre-fit checks owned by the lead (PROTOCOL.md section 9). Synthetic fixtures run in seconds; the real-data role check
loads only the admitted input file (no fit)."""
import copy
import math

import numpy as np
import pytest
import torch

from rgj import data as DA
from rgj import family as FAM
from rgj import finalize as FN
from rgj import select as SEL
from rgj import train as T


def synth_D(n=1600, seed=0, leak=2.0, task_is_sex=False):
    rng = np.random.default_rng(seed)
    S = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 83)).astype(np.float32)
    X[:, 0] += leak * (2 * S - 1)
    X[:, 1] += leak * (2 * S - 1) * 0.5
    y_inc = ((X[:, 2] + 0.5 * X[:, 0] + rng.normal(scale=0.5, size=n)) > 0).astype(np.int64)
    if task_is_sex:
        y_inc = S.copy()
    y_occ = np.clip(np.floor((X[:, 3] + 3) / 1.0), 0, 5).astype(np.int64)
    role = np.array(["DEFENSE_FIT"] * n, dtype="<U32")
    u = rng.random(n)
    role[u > 0.6] = "AUDIT_FIT"
    role[u > 0.75] = "INNER_SELECTION"
    role[u > 0.85] = "HEAD_VALIDATION"
    role[u > 0.92] = "DEVELOPMENT_ASSESSMENT"
    sub = np.full(n, "", dtype="<U32")
    df = np.flatnonzero(role == "DEFENSE_FIT")
    uu = rng.random(len(df))
    sub[df] = np.where(uu < 0.7, "CRITIC_FIT", np.where(uu < 0.85, "CRITIC_VAL", "CALIB"))
    D = {"X": X, "sex": S, "race": rng.integers(0, 3, n), "y": {"income": y_inc, "occupation_group": y_occ},
         "row_id": np.arange(n), "unit": np.arange(n), "role": role, "subrole": sub}
    D["idx"] = {r: np.flatnonzero(role == r) for r in DA.ROLES}
    D["idx"].update({r: np.flatnonzero(sub == r) for r in DA.SUBROLES})
    return D


@pytest.fixture(scope="module")
def syn():
    D = synth_D()
    data = T.TData(D)
    torch.manual_seed(0)
    warm = T.Model(T.D_IN, T.KS, 0).state_dict()
    return D, data, warm


def eq(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


# 1. roles, exclusion, groups, no old-assessment access ---------------------------------------------------------------
def test_real_roles_and_inputs():
    D = DA.load()
    assert set(np.unique(D["role"])) == set(DA.ROLES)
    for a in DA.ROLES:
        for b in DA.ROLES:
            if a < b:
                assert not set(D["unit"][D["idx"][a]]) & set(D["unit"][D["idx"][b]])
    assert len(D["row_id"]) == sum(len(D["idx"][r]) for r in DA.ROLES)
    assert D["X"].shape[1] == 83
    names = [f.lower() for f in D["feature_names"]]
    assert not any(f.startswith(("sex", "race", "income", "occupation=", "fnlwgt")) for f in names)
    assert set(D["idx"]["CRITIC_FIT"]) | set(D["idx"]["CRITIC_VAL"]) | set(D["idx"]["CALIB"]) == set(D["idx"]["DEFENSE_FIT"])
    old_role = np.load(DA.SRC)["role"]
    old_ids = np.load(DA.SRC)["row_id"][np.isin(old_role, DA.DROPPED)]
    assert not set(old_ids.tolist()) & set(D["row_id"].tolist())


# 2. beta = 0, lambda = 0 identity on every arm and both stage orders ----------------------------------------------
@pytest.mark.parametrize("arm", ["L-O", "L-R", "J-G", "L-G", "J-R", "J-O"])
def test_beta0_identity(syn, arm):
    D, data, warm = syn
    for stage in ("B", "C"):
        tm, _, _ = T.task_line(warm, data, 0, 6, stage=stage)
        m, diag, _, _, _ = T.train_run(arm, 0.0, warm, data, 0, stage, n_epochs=6, budgets=(0.0, 0.0), critic_head=T.head_of(warm))
        assert eq(m.state_dict(), tm.state_dict())
        assert diag["critic_online_updates"] > 0


def test_stage_orders_differ(syn):
    D, data, warm = syn
    a, _, _ = T.task_line(warm, data, 0, 2, stage="B")
    b, _, _ = T.task_line(warm, data, 0, 2, stage="C")
    assert not eq(a.state_dict(), b.state_dict())


# 3. gradient signs and the multiplier --------------------------------------------------------------------------------
def test_critics_learn_and_encoder_reduces_signal(syn):
    D, data, warm = syn
    _, d0, _, _, _ = T.train_run("J-R", 0.0, warm, data, 0, "B", n_epochs=8, critic_head=T.head_of(warm))
    _, d1, _, _, _ = T.train_run("J-R", 0.3, warm, data, 0, "B", n_epochs=8, critic_head=T.head_of(warm))
    r0 = d0["refits"][-1]["calib"]
    r1 = d1["refits"][-1]["calib"]
    assert r0["v1"]["R"] > 0.05 and r0["pair"]["winner"] != "const"      # critics read S from the leaky views
    assert r1["pair"]["R"] < r0["pair"]["R"] - 0.02                       # protection lowers the readable signal


def test_multiplier_rises_and_caps(syn):
    assert T.dual_update(0.0, 0.3, 0.1, 0.1) == pytest.approx(0.02)
    assert T.dual_update(0.0, 0.05, 0.1, 0.1) == 0.0
    assert T.dual_update(0.29, 5.0, 0.0, 0.1) == pytest.approx(0.3)
    D, data, warm = syn
    _, d, _, _, _ = T.train_run("J-G", 0.1, warm, data, 0, "B", n_epochs=8, budgets=(-1.0, 10.0), critic_head=T.head_of(warm))
    lam = [x["lambda"] for x in d["lambda_trace"]]
    assert lam[-1][0] > 0 and lam[-1][1] == 0.0
    assert all(0.0 <= l[0] <= 0.3 + 1e-12 for l in lam)


# 4. constant critic: zero recovery, no gradient ----------------------------------------------------------------------
def test_constant_critic_zero():
    prior = np.array([0.3, 0.7])
    lp = torch.tensor(np.log(prior), dtype=torch.float32)
    S = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0, 1, 1])
    z = torch.zeros(10, 18, requires_grad=True)
    logits = z[:, :2] * 0 + lp
    R, j, vals = T.recovery([logits, logits + 0.0], S, lp, 0.61)
    assert float(R) == 0.0
    g = torch.autograd.grad(R, z, allow_unused=True)[0]
    assert g is None or float(g.abs().max()) == 0.0


# 8. transport, including an amplified direction ----------------------------------------------------------------------
def test_transport_exact_amplified():
    g = torch.Generator().manual_seed(0)
    V0 = torch.randn(3000, 18, generator=g)
    V0[:, 5] = V0[:, 4] * 1.0 + 1e-6 * torch.randn(3000, generator=g)    # near-degenerate direction
    V1 = V0 @ (torch.eye(18) + 0.05 * torch.randn(18, 18, generator=g)) + 0.3
    To, Tn = T.Transform(V0, "floored"), T.Transform(V1, "floored")
    assert To.spectrum()["W_condition"] > 1e3
    c = T.critic("B", 18)
    Vt = torch.randn(500, 18, generator=g)
    with torch.no_grad():
        before = c(To(Vt)).double()
    c2 = T.transport_first_layer(copy.deepcopy(c), To, Tn)
    with torch.no_grad():
        after = c2(Tn(Vt)).double()
        exact = c2[0].weight.double()     # float64 check of the algebra
        A = c[0].weight.double()
        ref = A @ To.W64 @ torch.linalg.inv(Tn.W64)
    assert torch.allclose(exact, ref.float().double(), rtol=1e-5, atol=1e-6)
    rel = float((after - before).abs().max() / before.abs().max())
    assert rel < 5e-3, rel


# 9. local arms: coalition gradient exactly absent (poisoned shadow bank) ------------------------------------------
def poisoned(warm, data):
    crit = {"critics": {}, "transforms": {v: None for v in T.VIEWS}, "lam": [0.0, 0.0]}
    for v in T.VIEWS:
        crit["critics"][v] = {}
        for k in T.KINDS:
            c = T.new_critic(0, v, k, "init")
            if v == "pair":
                with torch.no_grad():
                    for p in c.parameters():
                        p.fill_(float("nan"))
            crit["critics"][v][k] = c.state_dict()
    return crit


def test_local_arm_has_no_coalition_gradient(syn):
    D, data, warm = syn
    crit = poisoned(warm, data)
    m, d, _, _, _ = T.train_run("L-O", 0.3, warm, data, 0, "B", n_epochs=2, init_critics=crit, critic_head=T.head_of(warm))
    assert d["nonfinite"] == 0 and all(torch.isfinite(p).all() for p in m.parameters())
    m2, d2, _, _, _ = T.train_run("J-O", 0.3, warm, data, 0, "B", n_epochs=2, init_critics=crit, critic_head=T.head_of(warm))
    assert d2["nonfinite"] > 0          # the same poison reaches a joint arm: the check can fail


# 10. coefficient sums and update counts --------------------------------------------------------------------------
def test_coefficients_and_counts(syn):
    for b in (0.03, 0.1, 0.3):
        cj, cl = T.base_coefficients("joint", b), T.base_coefficients("local", b)
        assert math.isclose(sum(cj.values()), b) and math.isclose(sum(cl.values()), b) and cl["pair"] == 0.0
    D, data, warm = syn
    steps = math.ceil(data.n / 256)
    for arm in ("L-R", "J-O"):
        _, d, ck, _, fin = T.train_run(arm, 0.1, warm, data, 0, "B", n_epochs=4, critic_head=T.head_of(warm))
        assert d["encoder_updates"] == 4 * steps
        assert d["critic_online_updates"] == 4 * steps * 5 * 6
        assert ("refit_critics" in fin) == T.ARMS[arm]["refresh"]
        assert not eq(fin["theta_T_minus_1"]["model"], fin["theta_T"])


# 6./12. selection guards and the full conjunction ------------------------------------------------------------------
def test_leakage_transfer_cannot_count(monkeypatch):
    table = {"a": {"v1": 0.80, "v2": 0.70, "pair": 0.70}, "b": {"v1": 0.74, "v2": 0.74, "pair": 0.78}}
    u = {0: {"acc": 0.84, "const_acc": 0.75}, 1: {"acc": 0.47, "const_acc": 0.27}}
    monkeypatch.setattr(SEL, "util", lambda n: u)
    monkeypatch.setattr(SEL, "auc", lambda n: table[n])
    guard = {"v1": 0.75, "v2": 0.75}
    a = SEL.candidate("a", u, 0.1, 5, guard)
    b = SEL.candidate("b", u, 0.1, 5, guard)
    assert not a["feasible"] and b["feasible"]       # lower pair AUC bought by a worse recipient is not eligible


def test_conjunction_logic():
    ids = [e["id"] for e in FAM.PRIMARY if e["claim"] == "A"]
    allp = {i: "PASS" for i in ids}
    ok = {k: {"valid_reference": True, "J-G": "NOMINEE", "L-G": "NOMINEE", "C*": "L-G"} for k in (0, 1, 2)}
    assert FAM.claim_decision("A", allp, ok)["decision"] == "PASS"
    one = dict(allp, P02="NOT_ESTABLISHED")
    assert FAM.claim_decision("A", one, ok)["decision"] == "NOT_ESTABLISHED"
    missing = {k: v for k, v in ok.items() if k != 2}
    assert FAM.claim_decision("A", allp, missing)["decision"] == "NOT_ESTABLISHED"
    alias = copy.deepcopy(ok)
    alias[1]["L-G"] = "TASK_ONLY_ALIAS"
    assert FAM.claim_decision("A", allp, alias)["decision"] == "NOT_ESTABLISHED"
    invalid = copy.deepcopy(ok)
    invalid[0]["valid_reference"] = False
    assert FAM.claim_decision("A", allp, invalid)["decision"] == "NOT_ESTABLISHED"
    idsB = [e["id"] for e in FAM.PRIMARY if e["claim"] == "B"]
    noc = copy.deepcopy(ok)
    noc[2]["C*"] = None
    assert FAM.claim_decision("B", {i: "PASS" for i in idsB}, noc)["decision"] == "NOT_ESTABLISHED"
    assert round(FAM.Z_PRIMARY, 6) == 2.991316 and FAM.PRIMARY_SIZE == 18


# 13. bootstrap grouping ----------------------------------------------------------------------------------------------
def test_bootstrap_groups_rows_not_seeds():
    from stored_model_eval.pilot_infer import UnitBootstrap
    units = np.array([7, 7, 3, 9, 9, 9, 4])
    bs = UnitBootstrap(units, 50, 20261004, 25)
    assert bs.n_units == 4
    for W in bs.chunks():
        assert np.array_equal(W[0], W[1]) and np.array_equal(W[3], W[5])
        assert np.allclose(W[[0, 2, 3, 6]].sum(0), 4)


# 11. output closure --------------------------------------------------------------------------------------------------
def test_output_closure(syn):
    D, data, warm = syn
    rng = np.random.default_rng(1)
    R1, R2 = rng.normal(size=(len(D["row_id"]), 16)), rng.normal(size=(len(D["row_id"]), 16))
    out, heads, _ = FN.heads_and_outputs([R1, R2], D)
    out2, _, _ = FN.heads_and_outputs([R1, R2 + 5 * rng.normal(size=R2.shape)], D)
    assert np.array_equal(out["c1"], out2["c1"]) and np.array_equal(out["hard1"], out2["hard1"])
    assert heads[0].n_features_in_ == 16 and heads[1].n_features_in_ == 16


# 7. task equals the sensitive attribute: the selection relies on independent attackers, not on training critics -----
def test_task_equals_sensitive_not_certified():
    from rgj import audit as AU
    D = synth_D(task_is_sex=True)
    S = D["sex"].astype(float)
    r = np.random.default_rng(2).normal(size=(len(S), 16))
    logits = np.stack([np.zeros_like(S), 4 * (2 * S - 1)], 1)
    V = {"v1": np.hstack([r, logits]), "v2": np.random.default_rng(3).normal(size=(len(S), 22))}
    V["pair"] = np.hstack([V["v1"], V["v2"]])
    res = AU.inner_audit(V, D)
    assert res["auc"]["v1"] > 0.95       # a broken training critic cannot hide what independent attackers read


# review R1: block transform stays bounded because critic views use the fixed per-seed head ------------------------
def test_fixed_head_keeps_block_inputs_bounded(syn):
    D, data, warm = syn
    head = T.head_of(warm)
    m = T.Model(T.D_IN, T.KS, 0)
    m.load_state_dict(warm)
    V0 = T.frozen_views(m, data.X, head)
    Tm = T.Transform(V0["v2"][torch.from_numpy(data.cf)], "floored")
    z0 = float(Tm(V0["v2"]).abs().max())
    m2, _, _ = T.task_line(warm, data, 0, 4)          # encoder AND training head move for a whole block
    z1 = float(Tm(T.frozen_views(m2, data.X, head)["v2"]).abs().max())
    assert z1 < 20 * max(z0, 1.0), (z0, z1)
