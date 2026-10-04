"""Registered fixtures for the focused no-erasure study (synthetic; no real data)."""
import numpy as np
import torch

from jcv import fixtures as FX
from jcv import train as T
from pnx import train as PT


def _setup():
    X, y1, y2, s, a, b = FX.xor_data(6000, 5)
    idx = np.arange(2000)
    data = FX._engine_data(X, y1, y2, s, idx, 0)
    old = (T.HP["warm_epochs"], T.HP["prot_epochs"])
    T.HP["warm_epochs"], T.HP["prot_epochs"] = 3, 3
    m0 = T.warm_start(X.shape[1], [2, 2], data, 0)
    st = {k: x.clone() for k, x in m0.state_dict().items()}
    B = T.guard_losses(m0, data, [None, None])
    return X, data, st, {i: B[i] + 0.01 for i in B}, old


def test_beta_zero_reproduces_task_only_bitwise():
    X, data, st, budgets, old = _setup()
    try:
        mU, _ = T.train_arm("U", 0.0, st, X.shape[1], [2, 2], data, 0, budgets)
        for arm in ("PN", "LN"):
            m, diag, crit = PT.train_arm(arm, 0.0, st, X.shape[1], [2, 2], data, 0, budgets)
            assert diag["protection_steps_attempted"] == 0
            assert all(torch.equal(m.state_dict()[k], mU.state_dict()[k]) for k in mU.state_dict())
            assert crit["critics"]                       # critics trained but never fed the encoder
        m1, _, _ = PT.train_arm("PN", 1.0, st, X.shape[1], [2, 2], data, 0, budgets)
        assert not all(torch.equal(m1.state_dict()[k], mU.state_dict()[k]) for k in mU.state_dict())
    finally:
        T.HP["warm_epochs"], T.HP["prot_epochs"] = old


def test_copied_loop_matches_predecessor_for_existing_arms():
    X, data, st, budgets, old = _setup()
    try:
        for arm, beta in (("JP", 1.0), ("L", 1.0), ("U", 0.0)):
            a, _ = T.train_arm(arm, beta, st, X.shape[1], [2, 2], data, 0, budgets)
            b, _, _ = PT.train_arm(arm, beta, st, X.shape[1], [2, 2], data, 0, budgets)
            assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()), arm
    finally:
        T.HP["warm_epochs"], T.HP["prot_epochs"] = old


def test_no_erasure_specs():
    for arm in ("PN", "LN"):
        assert PT.arm_spec(arm)["erasure"] is False
        assert PT.arm_spec(arm)["stages"][0]["mode"] == "penalty"
    assert set(PT.arm_spec("PN")["stages"][0]["P"]) == {"v1", "v2", "pair"}
    assert set(PT.arm_spec("LN")["stages"][0]["P"]) == {"v1", "v2"}
    # equal critic-update budget: 6 critics each
    n = lambda a: sum(len(v) for v in PT.arm_spec(a)["stages"][0]["banks"].values())  # noqa: E731
    assert n("PN") == n("LN") == 6


def test_label_mapping_for_aliases():
    from pnx.infer import label_of
    assert label_of("nn__s0__U") == "U" and label_of("nn__s2__E") == "E"
    assert label_of("pn__s1__PN__b0.1") == "PN_b0.1" and label_of("nn__s0__JP__b10") == "JP_b10"


def test_coalition_bank_records_all_tables():
    from pnx.outer import audit_views_full
    from jcv import outer as JO
    rng = np.random.default_rng(0)
    n = 900
    s = rng.integers(0, 2, n)
    V = {"v1": rng.normal(size=(n, 3)) + s[:, None] * 0.8, "v2": rng.normal(size=(n, 3))}
    V["pair"] = np.hstack([V["v1"], V["v2"]])
    D = {"idx": {"attacker_fit": np.arange(0, 400), "attacker_val": np.arange(400, 600), "assessment": np.arange(600, n)}}
    out, P = audit_views_full(V, s, D, JO.secondary_slate, 2, False, "pair")
    c = out["pair"]
    assert "own_table" in c and set(c["ignore_other_tables"]) == {"v1", "v2"} and len(c["bank"]) == 3
    assert c["selected_view"] == min(c["bank"], key=lambda x: x["val_log_loss"])["view"]


def test_final_snapshot_is_last_critic_state():
    """Review R1: the saved snapshot is theta_{T-1} (state at the last critic update), one encoder step before the end."""
    X, data, st, budgets, old = _setup()
    try:
        m, diag, crit = PT.train_arm("PN", 1.0, st, X.shape[1], [2, 2], data, 0, budgets)
        s1 = crit["model_state_at_last_critic_update"]
        assert s1 is not None and not all(torch.equal(s1[k], m.state_dict()[k]) for k in s1)
        assert set(crit["whiteners"]) == {"v1", "v2", "pair"} and diag["norms"]["steps"] > 0
    finally:
        T.HP["warm_epochs"], T.HP["prot_epochs"] = old
