"""Synthetic inner records -> lcr.select (SELECTION_RULES.json roles; no headroom buffer). No real data is read."""
import json

import numpy as np
import pytest

from lcr import run as R
from lcr import select as SEL

U_UT = {"income": {"acc": 0.80, "logloss": 0.40, "brier": 0.28, "const_acc": 0.60},
        "occupation": {"acc": 0.50, "logloss": 1.20, "brier": 0.60, "const_acc": 0.20}}


def _util(dll=0.003, dbr=0.002, dacc=0.0):
    return {t: {"acc": m["acc"] + dacc, "logloss": m["logloss"] + dll, "brier": m["brier"] + dbr,
                "const_acc": m["const_acc"]} for t, m in U_UT.items()}


def _inner(pair, local=0.60, util=None, pres=True, states=16):
    return {"recovery": {"auc": {"v1": local, "v2": local, "pair": pair}}, "utility": util or _util(),
            "preserved": {"1": pres, "2": pres}, "token_states": states, "composed": None}


def _pair_of(cid):
    p = R.parse_id(cid)
    if p["kind"] != "policy":
        return 0.85
    base = {"d0": 0.0, "d1_fixed": -0.004, "weighted": -0.010, "constrained": -0.020, "ctask": 0.0}[p["arm"]]
    if not p["privacy_trained"]:
        return 0.846 + (0.002 if "DIRECT" in cid else 0.0) + (-0.001 if p["arm"] == "ctask" else 0.0)
    lam = p["lam"] if p["lam"] is not None else 0.05
    fam = {"LOCAL": 0.02, "SEQ-12": 0.01, "SEQ-21": 0.005, "JOINT": 0.006, "JOINT-SINGLE": 0.004,
           "JOINT-PAIR": 0.0}.get(p["base_family"], 0.0)
    return 0.845 - 0.2 * lam + base + fam


def _bank(tweak=None):
    inner, rel = {}, {}
    for k in R.SEEDS:
        for cid in R.scored_ids():
            p = R.parse_id(cid)
            if cid == "SRC|U":
                r = _inner(0.86, util=_util(0.0, 0.0), states=None)
            elif p["kind"] == "reference":
                r = _inner(0.80, util=_util(0.05, 0.03), states=None)
            elif p["kind"] == "source":
                r = _inner(0.79, util=_util(0.02, 0.01), states=None)
            elif cid == "U|CLASS|i1o1":
                r = _inner(0.74, util=_util(0.11, 0.04), states=7)
            else:
                lam = p["lam"] or 0.0
                r = _inner(_pair_of(cid), util=_util(0.003 + 0.06 * lam, 0.001 + 0.02 * lam), states=336)
            inner[R.inner_name(k, cid)] = r
            if p["kind"] == "policy":
                rel[(k, cid)] = f"{cid}|{k}"
                if p["arm"] == "constrained":
                    inner[R.unit_for(k, cid)] = {"status": "FEASIBLE", "deployed": {"feasible": True}}
    if tweak:
        tweak(inner, rel)
    return inner, rel


@pytest.fixture
def run_sel(tmp_path, monkeypatch):
    def go(inner, rel):
        monkeypatch.setattr(R, "RUN", tmp_path)
        monkeypatch.setattr(R, "PKG", tmp_path)
        monkeypatch.setattr(R, "event", lambda *a, **k: None)
        monkeypatch.setattr(R, "done", lambda n: n in inner)
        monkeypatch.setattr(R, "rec", lambda n: json.loads(json.dumps(inner[n])))
        monkeypatch.setattr(SEL, "release_hash", lambda k, cid: rel.get((k, cid)))
        return SEL.select_all()
    return go


def test_roles_follow_section_10(run_sel, tmp_path):
    out = run_sel(*_bank())
    st = out["statuses"]
    assert st["Q"]["config"] == "U|DIRECT-TASK|i8o64" and st["Q"]["status"] == "NOMINEE"
    assert st["T*"]["status"] == "NOMINEE" and st["T*"]["config"] in SEL.T_STAR_CLOSED
    assert st["T*"]["config"] not in ("SRC|RAW-J_b0.3",)                       # RAW-J is excluded from T*
    # lambda 0.1 constrained-free: weighted controls stronger than D0/D1; constrained strongest overall
    assert R.parse_id(st["P*"]["config"])["arm"] == "constrained" and st["P*"]["config"] == "U|K-JOINT-PAIR|i8o64|D1"
    assert R.parse_id(st["C*"]["config"])["arm"] in ("d0", "d1_fixed", "weighted")
    assert R.parse_id(st["C*"]["config"])["arm"] == "weighted"
    assert R.parse_id(st["N*"]["config"])["arm"] == "constrained"
    assert st["C_pair*"]["config"] != SEL.JOINT_PAIR and st["J*"]["config"] == SEL.JOINT_PAIR
    assert st["P*"]["winning"] == "JOINT-PAIR; constrained"
    d = out["diagnostics"]
    assert d["best_d1_fixed_privacy"]["paired_d0"] == d["best_d1_fixed_privacy"]["config"][:-3]
    json.loads((tmp_path / "SELECTION.json").read_text(), parse_constant=lambda c: pytest.fail(c))


def test_no_headroom_rule_ordinary_only(run_sel):
    # a code with LL excess 0.008 (inside the original 0.01, outside cbp's 0.006 buffer) stays ordinarily eligible
    def tw(inner, rel):
        for k in R.SEEDS:
            inner[R.inner_name(k, 'U|K-JOINT-PAIR|i8o64|D1')]["utility"] = _util(0.008, 0.004)
    out = run_sel(*_bank(tw))
    assert out["statuses"]["J*"]["status"] == "NOMINEE"


def test_failing_seed_is_not_averaged_away(run_sel):
    def tw(inner, rel):
        inner[R.inner_name(2, 'U|K-JOINT-PAIR|i8o64|D1')]["utility"] = _util(0.012, 0.004)
    out = run_sel(*_bank(tw))
    assert out["statuses"]["J*"]["status"] == "NO_ELIGIBLE_NOMINEE"
    assert out["statuses"]["J*"]["reason"] == "ORDINARY_UTILITY_FAILURE"
    assert out["statuses"]["P*"]["config"] != SEL.JOINT_PAIR


def test_guards_and_missing_comparator(run_sel):
    def tw(inner, rel):
        for k in R.SEEDS:
            for c in R.constrained_ids():
                inner[R.inner_name(k, c)]["recovery"]["auc"]["v2"] = 0.62   # > T* + 0.005
    out = run_sel(*_bank(tw))
    assert out["statuses"]["N*"]["status"] == "NO_ELIGIBLE_NOMINEE"
    assert out["statuses"]["N*"]["reason"] == "LOCAL_GUARD_FAILURE"
    assert out["claim_role_states"]["B"]["nominee"] == "NO_ELIGIBLE"


def test_decision_failure_is_technical(run_sel):
    def tw(inner, rel):
        inner[R.inner_name(1, 'U|W-LOCAL|i8o64|l0.04|D1')]["preserved"]["1"] = False
    out = run_sel(*_bank(tw))
    assert out["statuses"]["C*"]["status"] == "INVALID_COMPARATOR"
    assert out["statuses"]["C*"]["reason"] == "DECISION_PRESERVATION_FAILURE"
    assert out["claim_role_states"]["B"]["comparator"] == "TECHNICAL_FAILURE"


def test_exact_release_alias_names_simplest_construction(run_sel):
    def tw(inner, rel):
        for k in R.SEEDS:      # the constrained JOINT-PAIR deploys exactly the weighted W-SEQ-21 0.1 release
            j, w = R.inner_name(k, SEL.JOINT_PAIR), R.inner_name(k, "U|W-SEQ-21|i8o64|l0.1|D1")
            inner[j] = json.loads(json.dumps(inner[w]))
            rel[(k, SEL.JOINT_PAIR)] = rel[(k, "U|W-SEQ-21|i8o64|l0.1|D1")]
            inner[j]["recovery"]["auc"]["pair"] -= 0.05
            inner[w]["recovery"]["auc"]["pair"] -= 0.05
    out = run_sel(*_bank(tw))
    p = out["statuses"]["P*"]
    assert set(p["aliases"]["full"]) == {SEL.JOINT_PAIR, "U|W-SEQ-21|i8o64|l0.1|D1"}
    assert p["winning"] == "SEQ-21; weighted"


def test_infeasible_constrained_fit_is_never_nominated(run_sel):
    def tw(inner, rel):
        for c in R.constrained_ids():
            inner[R.unit_for(1, c)] = {"status": "INFEASIBLE", "deployed": {"feasible": False}}
    out = run_sel(*_bank(tw))
    st = out["statuses"]
    assert st["N*"]["status"] == "NO_ELIGIBLE_NOMINEE" and st["N*"]["reason"] == "CONSTRAINED_FIT_INFEASIBLE"
    assert st["J*"]["reason"] == "CONSTRAINED_FIT_INFEASIBLE"
    assert R.parse_id(st["P*"]["config"])["arm"] != "constrained"
    assert R.parse_id(st["C_pair*"]["config"])["arm"] != "constrained"
    assert out["claim_role_states"]["B"]["nominee"] == "NO_ELIGIBLE"
