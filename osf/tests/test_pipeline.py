"""Lead pre-fit checks on synthetic data (seconds): raw fidelity to pinned rgj.train, zero-strength parity, allocation
identity, receipts, local pair isolation, frozen-minibatch equivalence. The real-role check loads inputs only."""
import math

import numpy as np
import pytest
import torch

from osf import data as OD
from osf import train as T
from rgj import train as RT
from smf.tests.test_pipeline import synth_D


@pytest.fixture(scope="module")
def syn():
    D = synth_D()
    data = RT.TData(D)
    torch.manual_seed(0)
    warm = T.Model(T.D_IN, T.KS, 0).state_dict()
    return D, data, warm, T.head_of(warm)


def eq(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def run(c, syn, ep=2, **kw):
    D, data, warm, head = syn
    return T.train_run(c, warm, data, 0, head, n_epochs=ep, ckpt_epochs=(ep,), **kw)


@pytest.mark.parametrize("treat,arm", [("J", "J-O"), ("L", "L-O")])
def test_raw_bitwise_equals_pinned_rgj(syn, treat, arm):
    D, data, warm, head = syn
    m, d, ck, _, fin, rec = run({"mode": "RAW", "treat": treat, "beta": 0.3}, syn, ep=3)
    m2, d2, ck2, _, fin2 = RT.train_run(arm, 0.3, warm, data, 0, "B", n_epochs=3, critic_head=head, ckpt_epochs=(3,))
    assert eq(m.state_dict(), m2.state_dict())
    assert d["critic_online_updates"] == d2["critic_online_updates"] and d["clip_hits"] == d2["clip_hits"]
    for v in RT.VIEWS:
        for k in RT.KINDS:
            assert eq(ck[3]["critics"][v][k], ck2[3]["critics"][v][k])
    # rgj logs the combined penalty/task norms every 20 steps: the receipts reproduce them
    for e in d2["norms"]:
        j = e["step"] - 1
        assert math.isclose(math.sqrt((rec["q_norm"][j] ** 2).sum()), e["penalty"], rel_tol=1e-5)
        assert math.isclose(math.sqrt((rec["t_norm"][j] ** 2).sum()), e["task"], rel_tol=1e-5)


@pytest.mark.parametrize("c", [{"mode": "RAW", "treat": "J", "beta": 0.0}, {"mode": "RAW", "treat": "L", "beta": 0.0},
                               {"mode": "NORM", "treat": "J", "rho": 0.0, "a": 1.0},
                               {"mode": "NORM", "treat": "L", "rho": 0.0, "a": 2.0}])
def test_zero_strength_equals_task(syn, c):
    m0, d0, *_ = run({"mode": "TASK", "treat": None}, syn)
    m, d, *_ = run(c, syn)
    assert eq(m.state_dict(), m0.state_dict())
    assert d["critic_online_updates"] > 0 and d0["critic_online_updates"] == 0


def test_task_equals_rgj_task_line(syn):
    D, data, warm, head = syn
    m, *_ = run({"mode": "TASK", "treat": None}, syn)
    m2, *_ = RT.train_run("TASK", 0.0, warm, data, 0, "B", n_epochs=2, ckpt_epochs=())
    assert eq(m.state_dict(), m2.state_dict())


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
def test_allocation_rms_identity(a):
    s1, s2 = T.allocation(a)
    assert math.isclose(math.sqrt((s1 * s1 + s2 * s2) / 2), 1.0, rel_tol=1e-15)
    assert math.isclose(s1 / s2, a, rel_tol=1e-15)


@pytest.mark.parametrize("treat,a", [("J", 1.0), ("J", 2.0), ("L", 0.5)])
def test_norm_receipts_hit_declared_ratios(syn, treat, a):
    rho = 3.0
    m, d, _, _, _, rec = run({"mode": "NORM", "treat": treat, "rho": rho, "a": a}, syn)
    s = T.allocation(a)
    for i in (0, 1):
        ok = (rec["zero"][:, i] == T.Z_NONE) & ~rec["cap"][:, i]
        assert ok.any()
        np.testing.assert_allclose(rec["ratio"][ok, i], rho * s[i], rtol=1e-5)
    both = ((rec["zero"] == T.Z_NONE) & ~rec["cap"]).all(1)
    np.testing.assert_allclose(rec["rms_ratio"][both], rho, rtol=1e-5)
    # combined ratio differs in general (task norms differ by encoder)
    assert np.all(np.isfinite(rec["comb_ratio"]))


def test_local_has_no_pair_gradient(syn):
    D, data, warm, head = syn
    c = {"mode": "NORM", "treat": "L", "rho": 3.0, "a": 1.0}
    assert T.proxy_coefficients(c)["pair"] == 0.0
    assert T.proxy_coefficients({"mode": "RAW", "treat": "L", "beta": 0.3})["pair"] == 0.0
    m, d, ck, _, _, rec = run(c, syn)
    assert np.isnan(rec["R"][:, 2]).all()          # pair never enters the encoder objective
    # the shadow pair bank still trains (its weights moved)
    init = T.new_critic(0, "pair", "A", "init").state_dict()
    assert not eq(init, ck[2]["critics"]["pair"]["A"])


def test_frozen_equivalence_and_failures(syn):
    D, data, warm, head = syn
    bi = np.random.default_rng([0, 0, 2]).permutation(data.n)[:256]
    for treat in ("J", "L"):
        _, _, ck, _, _, _ = run({"mode": "RAW", "treat": treat, "beta": 0.3}, syn)
        r = T.frozen_equivalence(ck[2]["model"], data, 0, head, treat, 0.3, bi, snapshot=ck[2])
        assert r["equivalent"] and r["applicable"] == 2, r
        rr = [e["r_i"] for e in r["encoders"]]
        common = math.sqrt((rr[0] ** 2 + rr[1] ** 2) / 2)
        bad = T.frozen_equivalence(ck[2]["model"], data, 0, head, treat, 0.3, bi, snapshot=ck[2], rho_override=common)
        assert not bad["equivalent"]
        capped = T.frozen_equivalence(ck[2]["model"], data, 0, head, treat, 0.3, bi, snapshot=ck[2], a_max=1e-6)
        assert not capped["equivalent"] and all(e["cap"] for e in capped["encoders"])


def test_bank_ids_unique_and_parse():
    for kind, n in (("full", 21), ("reduced", 13)):
        b = T.bank(kind)
        ids = [T.config_id(c) for c in b]
        assert len(ids) == n == len(set(ids))
        assert all(T.config_id(T.parse_id(i)) == i for i in ids)


def test_real_roles_sealed_and_fit_tensors_unchanged():
    from smf import data as SD
    D = OD.load()
    assert D["sealed"]
    a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    assert (D["sex"][a] == -1).all() and (D["y"]["income"][a] == -1).all()
    assert OD.fit_tensor_sha(D) == OD.fit_tensor_sha(SD.load())
    with pytest.raises(PermissionError):
        OD.labels_for(D, "selection", "OSF_DEVELOPMENT_ASSESSMENT")
    with pytest.raises(PermissionError):
        OD.labels_for(D, "assessment", "OSF_DEVELOPMENT_ASSESSMENT")   # sealed
