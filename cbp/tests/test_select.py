"""Synthetic inner records -> cbp.select (registered rules, HEADROOM_SELECTION_RULES.json). No real data is read."""
import json

import pytest

from cbp import run as R
from cbp import select as SEL

U_UT = {"income": {"acc": 0.80, "logloss": 0.40, "brier": 0.28, "const_acc": 0.60},
        "occupation": {"acc": 0.50, "logloss": 1.20, "brier": 0.60, "const_acc": 0.20}}


def _util(dll=0.003, dbr=0.002, dacc=0.0):
    return {t: {"acc": m["acc"] + dacc, "logloss": m["logloss"] + dll, "brier": m["brier"] + dbr,
                "const_acc": m["const_acc"]} for t, m in U_UT.items()}


def _inner(pair, local=0.60, util=None, pres=True, states=16):
    return {"recovery": {"auc": {"v1": local, "v2": local, "pair": pair}}, "utility": util or _util(),
            "preserved": {"1": pres, "2": pres}, "token_states": states, "composed": None}


def _bank(tweak=None):
    """Privacy pair AUC 0.70 - 0.5 lam + family offset (SEQ-21 best); log-loss excess 0.002 + 0.07 lam (headroom fails at 0.06+)."""
    off = {"LOCAL": 0.03, "SEQ-12": 0.01, "SEQ-21": 0.0, "JOINT": 0.005}
    inner, pol = {}, {}
    for k in R.SEEDS:
        for cid in R.scored_ids():
            p = R.parse_id(cid)
            if cid == "SRC|U":
                r = _inner(0.75, util=_util(0.0, 0.0), states=None)
            elif p["kind"] != "policy":
                r = _inner(0.72, states=None)
            elif p["family"] in R.PRIVACY:
                lam = p["lam"]
                r = _inner(0.70 - 0.5 * lam + off[p["family"]], util=_util(0.002 + 0.07 * lam, 0.001 + 0.02 * lam),
                           states=72)
                pol[R.unit_for(k, cid)] = {"pair_fingerprint": f"{cid}|{k}"}
            else:
                r = _inner(0.74, states=72 if "i8o64" in cid else 2)
            inner[f"inner__{R.unit_for(k, cid)}"] = r
    if tweak:
        tweak(inner, pol)
    return inner, pol


@pytest.fixture
def run_sel(tmp_path, monkeypatch):
    def go(inner, pol):
        store = {**inner, **pol}
        monkeypatch.setattr(R, "RUN", tmp_path)
        monkeypatch.setattr(R, "PKG", tmp_path)
        monkeypatch.setattr(R, "event", lambda *a, **k: None)
        monkeypatch.setattr(R, "done", lambda n: n in store)
        monkeypatch.setattr(R, "rec", lambda n: json.loads(json.dumps(store[n])))
        return SEL.select_all()
    return go


def test_headroom_selects_intermediate_lambda_and_records_the_give_up(run_sel, tmp_path):
    out = run_sel(*_bank())
    st, d = out["statuses"], out["diagnostics"]
    assert st["Q"] == {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}
    assert st["T*"]["status"] == "NOMINEE" and st["T*"]["config"] in SEL.T_STAR_CLOSED
    assert st["T*"]["config"] != "SRC|RAW-J_b0.3"
    assert st["P*"]["status"] == "NOMINEE" and st["P*"]["config"] == "U|SEQ-21|i8o64|l0.04"
    assert st["P*"]["winning_family"] == "SEQ-21" and st["P*"]["aliases"]["full"] == ["U|SEQ-21|i8o64|l0.04"]
    assert st["J*"]["config"] == "U|JOINT|i8o64|l0.04"
    assert d["ordinary_privacy_winner_no_headroom"]["config"] == "U|SEQ-21|i8o64|l0.1"
    h = d["headroom_changes_winner"]
    assert h["changed"] is True and h["pair_auc_given_up_by_headroom"]["mean"] == pytest.approx(0.03)
    assert set(h["pair_auc_given_up_by_headroom"]["per_seed"]) == {"0", "1", "2"}
    assert out["rows"]["SRC|U"]["mean_states"] is None                         # continuous: null, never Infinity
    for f in ("SELECTION.json", "INNER_SELECTION_TABLE.csv", "HEADROOM_VS_STANDARD_SELECTION.csv"):
        assert (tmp_path / f).exists()
    json.loads((tmp_path / "SELECTION.json").read_text(), parse_constant=lambda c: pytest.fail(c))


def test_unchanged_joint_alias_names_the_simpler_family(run_sel):
    def tw(inner, pol):
        for k in R.SEEDS:
            j, s = R.unit_for(k, "U|JOINT|i8o64|l0.04"), R.unit_for(k, "U|SEQ-21|i8o64|l0.04")
            inner[f"inner__{j}"] = json.loads(json.dumps(inner[f"inner__{s}"]))
            pol[j] = dict(pol[s])
    out = run_sel(*_bank(tw))
    p = out["statuses"]["P*"]
    assert p["config"] == "U|JOINT|i8o64|l0.04"                                 # key 4: 'U|JOINT|' sorts first
    assert p["winning_family"] == "SEQ-21" and p["aliases"]["decided_by_config_id_tiebreak"] is True
    assert p["aliases"]["full"] == ["U|JOINT|i8o64|l0.04", "U|SEQ-21|i8o64|l0.04"]


def test_decision_failure_is_technical_not_a_shortfall(run_sel):
    def tw(inner, pol):
        inner[f"inner__{R.unit_for(1, 'U|LOCAL|i8o64|l0.08')}"]["preserved"]["2"] = False
    out = run_sel(*_bank(tw))
    assert out["statuses"]["P*"]["status"] == "INVALID_NOMINEE"
    assert out["statuses"]["P*"]["failed"] == ["U|LOCAL|i8o64|l0.08"]
    assert out["statuses"]["C_rate"]["status"] == "INVALID_COMPARATOR"
    assert out["claim_role_states"]["C"]["nominee"] == "TECHNICAL_FAILURE"


def test_no_headroom_candidate_gives_registered_fallback(run_sel):
    def tw(inner, pol):
        for n, r in inner.items():
            if any(f"_{f}_" in n for f in R.PRIVACY):
                for t in r["utility"]:
                    r["utility"][t]["logloss"] = U_UT[t]["logloss"] + 0.0065 + (0.001 if "LOCAL" in n else 0)
    out = run_sel(*_bank(tw))
    p = out["statuses"]["P*"]
    assert p["status"] == "NO_ELIGIBLE_NOMINEE" and p["reason"] == "HEADROOM_SELECTION_FAILURE"
    assert p["descriptive_only"] is True and "LOCAL" not in p["descriptive_config"]
    assert out["claim_role_states"]["C"]["nominee"] == "NO_ELIGIBLE"
    assert out["diagnostics"]["ordinary_privacy_winner_no_headroom"]["status"] == "NOMINEE"


def test_missing_guard_comparator_is_technical_only_when_needed(run_sel):
    def tw(inner, pol):                                         # every code of the closed T* list fails ordinary
        for k in R.SEEDS:
            for c in SEL.T_STAR_CLOSED:
                if c != "SRC|U":
                    inner[f"inner__{R.unit_for(k, c)}"]["utility"] = _util(0.02, 0.0)
    out = run_sel(*_bank(tw))
    assert out["statuses"]["T*"]["config"] == "SRC|U"            # the anchor is always ordinary against itself
    rows = out["rows"]
    for c in rows:                                              # pick() works on in-memory rows (int seed keys)
        rows[c]["seeds"] = {int(k): v for k, v in rows[c]["seeds"].items()}
    p = SEL.pick([rows["U|SEQ-21|i8o64|l0.04"]], guards={"T*": None}, need_headroom=True)
    assert p["status"] == "INVALID_NOMINEE" and p["reason"] == "MISSING_GUARD_COMPARATOR"
    p = SEL.pick([rows["U|SEQ-21|i8o64|l0.1"]], guards={"T*": None}, need_headroom=True)
    assert p["status"] == "NO_ELIGIBLE_NOMINEE" and p["reason"] == "HEADROOM_SELECTION_FAILURE"
    assert p["missing_guards_not_needed"] == ["T*"]


def test_local_guard_failure(run_sel):
    def tw(inner, pol):
        for n, r in inner.items():
            if any(f"_{f}_" in n for f in R.PRIVACY):
                r["recovery"]["auc"]["v1"] = 0.62                # > T* local 0.60 + 0.005
    out = run_sel(*_bank(tw))
    p = out["statuses"]["P*"]
    assert p["status"] == "NO_ELIGIBLE_NOMINEE" and p["reason"] == "LOCAL_GUARD_FAILURE"
    ev = {e["config"]: e for e in p["evaluated"]}
    assert ev[p["descriptive_config"]]["guard_shortfall"] == pytest.approx(3.0)
