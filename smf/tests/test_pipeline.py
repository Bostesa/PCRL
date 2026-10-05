"""Lead pre-fit checks (PROTOCOL.md sections 2-6); synthetic fixtures run in seconds, the role check loads inputs only."""
import copy
import math

import numpy as np
import pytest
import torch

from rgj import train as RT
from smf import data as DA
from smf import train as T


def synth_D(n=1600, seed=0, leak=2.0):
    rng = np.random.default_rng(seed)
    S = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 83)).astype(np.float32)
    X[:, 0] += leak * (2 * S - 1)
    X[:, 1] += leak * (2 * S - 1) * 0.5
    y_inc = ((X[:, 2] + 0.5 * X[:, 0] + rng.normal(scale=0.5, size=n)) > 0).astype(np.int64)
    y_occ = np.clip(np.floor(X[:, 3] + 3), 0, 5).astype(np.int64)
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
    D["idx"] = {r: np.flatnonzero(role == r) for r in ("DEFENSE_FIT", "AUDIT_FIT", "INNER_SELECTION", "HEAD_VALIDATION",
                                                       "DEVELOPMENT_ASSESSMENT")}
    D["idx"].update({r: np.flatnonzero(sub == r) for r in ("CRITIC_FIT", "CRITIC_VAL", "CALIB")})
    return D


@pytest.fixture(scope="module")
def syn():
    D = synth_D()
    data = RT.TData(D)
    torch.manual_seed(0)
    warm = T.Model(T.D_IN, T.KS, 0).state_dict()
    return D, data, warm, T.head_of(warm)


def eq(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


TASK = {"base": "none", "schedule": "ONLINE", "update": "task"}


def test_real_roles_sealed_and_refit():
    D = DA.load()
    assert D["sealed"]
    a = D["idx"]["NEW_DEVELOPMENT_ASSESSMENT"]
    assert (D["sex"][a] == -1).all() and (D["y"]["income"][a] == -1).all()
    f = D["idx"]["NEW_DEFENSE_FIT"]
    assert np.allclose(D["X"][f][:, :5].mean(0), 0, atol=1e-4) and np.allclose(D["X"][f][:, :5].std(0), 1, atol=1e-3)
    for x in DA.ROLES:
        for y in DA.ROLES:
            if x < y:
                assert not set(D["unit"][D["idx"][x]]) & set(D["unit"][D["idx"][y]])
    from rgj import data as RD
    old = RD.load()
    assert not set(old["row_id"][old["idx"]["DEVELOPMENT_ASSESSMENT"]].tolist()) & set(D["row_id"].tolist())
    assert D["idx"]["DEFENSE_FIT"] is D["idx"]["NEW_DEFENSE_FIT"]


@pytest.mark.parametrize("spec", [{"base": b, "schedule": s} for b in ("joint", "local") for s in T.SCHEDULES])
def test_rho0_identity(syn, spec):
    D, data, warm, head = syn
    tm, *_ = T.train_run(TASK, 0.0, warm, data, 0, "A", None, n_epochs=3, ckpt_epochs=())
    cnt = {f"{v}|{k}": 40 for v in RT.VIEWS for k in RT.KINDS}
    m, d, *_ = T.train_run(spec, 0.0, warm, data, 0, "A", head, n_epochs=3, ckpt_epochs=(),
                           matched_counts=cnt if spec["schedule"] == "ONLINE_MATCHED" else None)
    assert eq(m.state_dict(), tm.state_dict()) and d["critic_online_updates"] > 0


def test_scale_invariance_and_zero_and_cap():
    g = torch.Generator().manual_seed(1)
    t, p = torch.randn(500, generator=g), torch.randn(500, generator=g)
    q1, i1 = T.normalized_direction(t, p, 0.75)
    q2, i2 = T.normalized_direction(t, 37.0 * p, 0.75)
    assert torch.allclose(q1, q2, rtol=1e-6, atol=1e-7) and abs(i1["ratio"] - 0.75) < 1e-6
    qz, iz = T.normalized_direction(t, torch.zeros(500), 0.75)
    assert float(qz.abs().max()) == 0 and iz["zero"] == "p"
    qc, ic = T.normalized_direction(t, 1e-6 * p, 0.75)
    assert ic["cap"] and ic["a"] == 100.0 and ic["ratio"] < 0.75


def test_rms_identity_and_local_common_symmetry():
    for w in ((1, 1), (3, 1), (0.25, 8)):
        s = T.allocation(*w)
        assert abs(math.sqrt((s[0] ** 2 + s[1] ** 2) / 2) - 1) < 1e-12
    assert T.allocation(2, 2) == T.allocation(1, 1)
    c1, c2 = T.coefficients("local", (2, 2)), T.coefficients("local", (1, 1))
    assert c1 == c2
    j1, j2 = T.coefficients("joint", (2, 2)), T.coefficients("joint", (1, 1))
    assert j1["pair"] < j2["pair"]


def test_controller_sign_selectivity_floor():
    w, v = T.controller_update(1.0, 0.80, 0.79)
    assert w == 2.0 and v > 0
    w, _ = T.controller_update(1.0, 0.70, 0.79)
    assert w == 0.25 + 0.0 or w == max(0.25, 1.0 - 1.0)
    assert T.targets_from({"v1": {"calib_auc": 0.505}, "v2": {"calib_auc": 0.80}}) == [0.5, 0.79]


def test_local_arm_has_no_coalition_gradient(syn):
    D, data, warm, head = syn
    crit = {"critics": {}, "transforms": {v: None for v in RT.VIEWS}}
    for v in RT.VIEWS:
        crit["critics"][v] = {}
        for k in RT.KINDS:
            c = RT.new_critic(0, v, k, "init")
            if v == "pair":
                with torch.no_grad():
                    for p_ in c.parameters():
                        p_.fill_(float("nan"))
            crit["critics"][v][k] = c.state_dict()
    m, d, *_ = T.train_run({"base": "local", "schedule": "ONLINE"}, 0.75, warm, data, 0, "A", head, n_epochs=2,
                           ckpt_epochs=(), init_critics=crit)
    assert d["nonfinite"] == 0 and all(torch.isfinite(p_).all() for p_ in m.parameters())
    m2, d2, *_ = T.train_run({"base": "joint", "schedule": "ONLINE"}, 0.75, warm, data, 0, "A", head, n_epochs=2,
                             ckpt_epochs=(), init_critics=crit)
    assert d2["nonfinite"] > 0


def test_matched_counts_exact(syn):
    D, data, warm, head = syn
    _, dr, *_ = T.train_run({"base": "joint", "schedule": "REFRESHED"}, 0.75, warm, data, 0, "A", head, n_epochs=4,
                            ckpt_epochs=())
    cnt = T.refit_counts(dr)
    assert sum(cnt.values()) == dr["critic_refit_updates"]
    _, dm, *_ = T.train_run({"base": "joint", "schedule": "ONLINE_MATCHED"}, 0.75, warm, data, 0, "A", head, n_epochs=4,
                            ckpt_epochs=(), matched_counts=cnt)
    assert dm["critic_matched_extra_updates"] == sum(cnt.values())
    S = 100
    for E in (0, 7, 99, 1234):
        assert sum(T.spread(E, S, s) for s in range(1, S + 1)) == E


def test_ratio_achieved_and_logged(syn):
    D, data, warm, head = syn
    _, d, *_ = T.train_run({"base": "joint", "schedule": "ONLINE"}, 0.75, warm, data, 0, "A", head, n_epochs=2,
                           ckpt_epochs=())
    e = d["epochs"][-1]
    assert abs(e["ratio_mean_1"] - 0.75) < 1e-5 and abs(e["ratio_mean_2"] - 0.75) < 1e-5


def test_feedback_vs_twin_record_same_measurements(syn):
    D, data, warm, head = syn
    m0 = T.Model(T.D_IN, T.KS, 0)
    m0.load_state_dict(warm)
    r0 = T.probe_receipt(m0, data, head, 0, "ref")
    ctrl = {"receipt0": r0, "b": [max(0.5, r0["v1"]["calib_auc"] - 0.01), max(0.5, r0["v2"]["calib_auc"] - 0.01)]}
    _, dF, *_ = T.train_run({"base": "joint", "schedule": "ONLINE", "feedback": True}, 0.75, warm, data, 0, "B", head,
                            n_epochs=2, ckpt_epochs=(), controller=ctrl)
    _, dN, *_ = T.train_run({"base": "joint", "schedule": "ONLINE", "feedback": False}, 0.75, warm, data, 0, "B", head,
                            n_epochs=2, ckpt_epochs=(), controller=ctrl)
    assert len(dF["controller"]) == len(dN["controller"]) == 3
    assert dF["controller"][0]["v"] == dN["controller"][0]["v"]                    # epoch 0 reuses the reference
    assert dN["controller"][-1]["w_after"] == [1.0, 1.0]                           # twin never applies
    assert dF["controller"][0]["w_after"] != [1.0, 1.0] or all(x < 0.0 for x in dF["controller"][0]["v"])
