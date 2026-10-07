"""[lra port of lcr/tests/test_select.py at 091afc2: lcr->lra renames; later edits: REVIEW_FINDINGS_DISPOSITION.json]
Synthetic inner records -> lra.select (SELECTION_RULES.json roles; no headroom buffer). No real data is read.
Regression tests of the inherited selection-stack findings are named test_fNN_*."""
import json

import numpy as np
import pytest

from lra import eval_lock as EL
from lra import run as R
from lra import select as SEL

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
    """inner: {unit name: record} (release units AND aud__ inner units are 'done'); rel: {(seed, cid): byte id};
    canon: {(seed, cid): canonical id} (defaults to the byte id)."""
    inner, rel, canon = {}, {}, {}
    for k in R.SEEDS:
        for cid in R.scored_ids():
            p = R.parse_id(cid)
            if cid == "SRC|U":
                r = _inner(0.86, util=_util(0.0, 0.0), states=None)
            elif p["kind"] == "reference":
                r = _inner(0.80, util=_util(0.05, 0.03), states=None)
            elif p["kind"] == "source":
                r = _inner(0.79, util=_util(0.02, 0.01), states=None)
            elif cid in ("U|CLASS|i1o1", "U|CLASS|i1o1|D1"):
                r = _inner(0.74, util=_util(0.11, 0.04), states=7)
            else:
                lam = p["lam"] or 0.0
                r = _inner(_pair_of(cid), util=_util(0.003 + 0.06 * lam, 0.001 + 0.02 * lam), states=336)
            inner[R.inner_name(k, cid)] = r
            inner[R.unit_for(k, cid)] = {"config": cid, "seed": k}
            if p["kind"] == "policy":
                rel[(k, cid)] = f"{cid}|{k}"
                if p["arm"] == "constrained":
                    inner[R.unit_for(k, cid)] = {"status": "FEASIBLE", "deployed": {"feasible": True}, "config": cid,
                                                 "seed": k}
    if tweak:
        tweak(inner, rel, canon)
    return inner, rel, canon


@pytest.fixture
def run_sel(tmp_path, monkeypatch):
    def go(inner, rel, canon=None):
        canon = canon or {}

        def rec(n):
            v = inner[n]
            if isinstance(v, Exception):
                raise v
            return json.loads(json.dumps(v))
        monkeypatch.setattr(R, "RUN", tmp_path)
        monkeypatch.setattr(R, "PKG", tmp_path)
        monkeypatch.setattr(R, "event", lambda *a, **k: None)
        monkeypatch.setattr(R, "done", lambda n: n in inner)
        monkeypatch.setattr(R, "rec", rec)
        monkeypatch.setattr(SEL, "release_identity",
                            lambda k, cid, rows: (rel.get((k, cid)), canon.get((k, cid), rel.get((k, cid)))))
        return SEL.select_all()
    return go


def test_roles_follow_section_10(run_sel, tmp_path):
    out = run_sel(*_bank())
    st = out["statuses"]
    assert st["Q"]["config"] == "U|DIRECT-TASK|i8o64" and st["Q"]["status"] == "NOMINEE"
    assert st["T*"]["status"] == "NOMINEE" and st["T*"]["config"] in SEL.T_STAR_CLOSED
    assert st["T*"]["config"] not in ("SRC|RAW-J_b0.3",)                       # RAW-J is excluded from T*
    assert R.parse_id(st["P*"]["config"])["arm"] == "constrained" and st["P*"]["config"] == "U|K-JOINT-PAIR|i8o64|D1"
    assert R.parse_id(st["C*"]["config"])["arm"] == "weighted"
    assert R.parse_id(st["N*"]["config"])["arm"] == "constrained"
    assert st["C_pair*"]["config"] != SEL.JOINT_PAIR and st["J*"]["config"] == SEL.JOINT_PAIR
    assert st["P*"]["winning"] == "JOINT-PAIR; constrained"
    d = out["diagnostics"]
    assert d["best_d1_fixed_privacy"]["paired_d0"] == d["best_d1_fixed_privacy"]["config"][:-3]
    json.loads((tmp_path / "SELECTION.json").read_text(), parse_constant=lambda c: pytest.fail(c))


def test_t_star_pool_is_the_closed_untrained_list_with_class_d1():
    assert "U|CLASS|i1o1|D1" in SEL.T_STAR_CLOSED and "SRC|RAW-J_b0.3" not in SEL.T_STAR_CLOSED
    assert all(not SEL.privacy_trained(c) for c in SEL.T_STAR_CLOSED)
    assert set(SEL.T_STAR_CLOSED) <= set(R.scored_ids())


def test_no_headroom_rule_ordinary_only(run_sel):
    # a code with LL excess 0.008 (inside the original 0.01, outside cbp's 0.006 buffer) stays ordinarily eligible
    def tw(inner, rel, canon):
        for k in R.SEEDS:
            inner[R.inner_name(k, 'U|K-JOINT-PAIR|i8o64|D1')]["utility"] = _util(0.008, 0.004)
    out = run_sel(*_bank(tw))
    assert out["statuses"]["J*"]["status"] == "NOMINEE"


def test_failing_seed_is_not_averaged_away(run_sel):
    def tw(inner, rel, canon):
        inner[R.inner_name(2, 'U|K-JOINT-PAIR|i8o64|D1')]["utility"] = _util(0.012, 0.004)
    out = run_sel(*_bank(tw))
    assert out["statuses"]["J*"]["status"] == "NO_ELIGIBLE_NOMINEE"
    assert out["statuses"]["J*"]["reason"] == "ORDINARY_UTILITY_FAILURE"
    assert out["statuses"]["J*"]["fallback_class"] == "UTILITY"
    assert out["statuses"]["P*"]["config"] != SEL.JOINT_PAIR


def test_guard_failure_fallback_is_local_guard_class(run_sel):
    def tw(inner, rel, canon):
        for k in R.SEEDS:
            for c in R.constrained_ids():
                inner[R.inner_name(k, c)]["recovery"]["auc"]["v2"] = 0.62   # > T* + 0.005
    out = run_sel(*_bank(tw))
    n = out["statuses"]["N*"]
    assert n["status"] == "NO_ELIGIBLE_NOMINEE" and n["reason"] == "LOCAL_GUARD_FAILURE"
    assert n["fallback_class"] == "LOCAL_GUARD" and n["descriptive_only"] and n["descriptive_config"]
    assert out["claim_role_states"]["B"]["nominee"] == "NO_ELIGIBLE"


def test_decision_failure_is_technical(run_sel):
    def tw(inner, rel, canon):
        inner[R.inner_name(1, 'U|W-LOCAL|i8o64|l0.04|D1')]["preserved"]["1"] = False
    out = run_sel(*_bank(tw))
    assert out["statuses"]["C*"]["status"] == "INVALID_COMPARATOR"
    assert out["statuses"]["C*"]["reason"] == "DECISION_PRESERVATION_FAILURE"
    assert out["statuses"]["C*"]["fallback_class"] == "TECHNICAL" and out["statuses"]["C*"]["descriptive_config"] is None
    assert out["claim_role_states"]["B"]["comparator"] == "TECHNICAL_FAILURE"
    assert "U|W-LOCAL|i8o64|l0.04|D1" in out["technical_failures"]


def test_missing_release_unit_is_a_fit_or_admission_failure(run_sel):
    def tw(inner, rel, canon):
        del inner[R.unit_for(0, "REF|F0")]
    out = run_sel(*_bank(tw))
    assert out["statuses"]["T*"]["status"] == "INVALID_COMPARATOR"
    assert out["statuses"]["T*"]["reason"] == "FIT_OR_ADMISSION_FAILURE"


def test_infeasible_constrained_fit_is_never_nominated(run_sel):
    def tw(inner, rel, canon):
        for c in R.constrained_ids():
            inner[R.unit_for(1, c)] = {"status": "INFEASIBLE", "deployed": {"feasible": False}, "config": c, "seed": 1}
    out = run_sel(*_bank(tw))
    st = out["statuses"]
    assert st["N*"]["status"] == "NO_ELIGIBLE_NOMINEE" and st["N*"]["reason"] == "CONSTRAINED_FIT_INFEASIBLE"
    assert st["J*"]["reason"] == "CONSTRAINED_FIT_INFEASIBLE" and st["J*"]["fallback_class"] == "UTILITY"
    assert R.parse_id(st["P*"]["config"])["arm"] != "constrained"
    assert R.parse_id(st["C_pair*"]["config"])["arm"] != "constrained"
    assert out["claim_role_states"]["B"]["nominee"] == "NO_ELIGIBLE"
    assert not out["technical_failures"]


def test_feasible_fallback_ranks_before_an_infeasible_one(run_sel):
    def tw(inner, rel, canon):
        for k in R.SEEDS:
            for c in R.constrained_ids():          # every constrained arm fails the guard -> no N* nominee
                inner[R.inner_name(k, c)]["recovery"]["auc"]["v2"] = 0.62
            inner[R.inner_name(k, "U|K-LOCAL|i8o64|D1")]["utility"] = _util(0.012, 0.004)   # feasible but shortfall > 0
        for c in R.constrained_ids():
            if c != "U|K-LOCAL|i8o64|D1":
                inner[R.unit_for(1, c)] = {"status": "INFEASIBLE", "deployed": {"feasible": False}, "config": c,
                                           "seed": 1}
    out = run_sel(*_bank(tw))
    n = out["statuses"]["N*"]
    assert n["status"] == "NO_ELIGIBLE_NOMINEE" and n["descriptive_config"] == "U|K-LOCAL|i8o64|D1"


# ------------------------------------------------------------------ F01: technical fit record != infeasible
@pytest.mark.parametrize("defect", ["missing_unit", "unreadable", "no_status", "status_drift", "feasible_not_bool",
                                    "no_deployed", "other_config"])
def test_f01_unreadable_or_incomplete_fit_record_is_technical_not_infeasible(run_sel, defect):
    victim = "U|K-SEQ-21|i8o64|D1"

    def tw(inner, rel, canon):
        u = R.unit_for(2, victim)
        good = {"status": "FEASIBLE", "deployed": {"feasible": True}, "config": victim, "seed": 2}
        inner[u] = {"missing_unit": None, "unreadable": ValueError("truncated JSON"),
                    "no_status": {"deployed": {"feasible": True}},
                    "status_drift": {**good, "status": "OK"},
                    "feasible_not_bool": {**good, "deployed": {"feasible": "yes"}},
                    "no_deployed": {"status": "FEASIBLE"},
                    "other_config": {**good, "config": "U|K-LOCAL|i8o64|D1"}}[defect]
        if defect == "missing_unit":
            del inner[u]
    out = run_sel(*_bank(tw))
    st = out["statuses"]
    for role in ("N*", "P*", "C_pair*"):
        assert st[role]["status"].startswith("INVALID_"), role
    assert "FIT_OR_ADMISSION_FAILURE" in st["N*"]["reason"] or "FIT_RECORD_TECHNICAL_FAILURE" in st["N*"]["reason"]
    if defect != "missing_unit":
        assert st["N*"]["reason"] == "FIT_RECORD_TECHNICAL_FAILURE"
    assert st["N*"]["reason"] != "CONSTRAINED_FIT_INFEASIBLE" and st["N*"]["fallback_class"] == "TECHNICAL"
    assert out["claim_role_states"]["B"]["nominee"] == "TECHNICAL_FAILURE"
    assert victim in out["technical_failures"]
    assert out["rows"][victim]["fit_feasible"] is None


def test_f01_fit_feasible_values(monkeypatch):
    recs = {}
    monkeypatch.setattr(R, "done", lambda n: n in recs)
    monkeypatch.setattr(R, "rec", lambda n: recs[n])
    c = "U|K-LOCAL|i8o64|D1"
    for k in R.SEEDS:
        recs[R.unit_for(k, c)] = {"status": "FEASIBLE", "deployed": {"feasible": True}, "config": c, "seed": k}
    assert SEL.fit_feasible(c) == (True, None)
    recs[R.unit_for(1, c)] = {"status": "INFEASIBLE", "deployed": {"feasible": False}, "config": c, "seed": 1}
    assert SEL.fit_feasible(c) == (False, None)
    recs[R.unit_for(2, c)] = {"status": "FEASIBLE"}
    v, why = SEL.fit_feasible(c)
    assert v is None and "schema-incomplete" in why
    assert SEL.fit_feasible("U|W-LOCAL|i8o64|l0.1|D1") == (True, None)


# ------------------------------------------------------------------ F02/F09: one real representative
def test_f02_f09_label_names_one_real_alias_construction_first(run_sel):
    """K-LOCAL deploys exactly the weighted W-JOINT l0.1 release. Independent minima would say 'LOCAL; weighted', a
    pairing no alias deploys; the representative rule names W-JOINT: 'JOINT; weighted'."""
    w = "U|W-JOINT|i8o64|l0.1|D1"

    def tw(inner, rel, canon):
        for k in R.SEEDS:
            kk, ww = R.inner_name(k, "U|K-LOCAL|i8o64|D1"), R.inner_name(k, w)
            inner[ww]["recovery"]["auc"]["pair"] = 0.70
            inner[kk] = json.loads(json.dumps(inner[ww]))
            rel[(k, "U|K-LOCAL|i8o64|D1")] = rel[(k, w)]
    out = run_sel(*_bank(tw))
    p = out["statuses"]["P*"]
    assert set(p["aliases"]["full"]) == {"U|K-LOCAL|i8o64|D1", w}
    assert p["config"] == "U|K-LOCAL|i8o64|D1"                               # ordering tie-break: 'U|K' < 'U|W'
    rep = p["aliases"]["representative"]
    assert rep == w and p["winning"] == "JOINT; weighted"
    fam, con = p["winning"].split("; ")[:2]
    assert out["rows"][rep]["family"] == fam and SEL.CONSTRUCTION_NAME[out["rows"][rep]["arm"]] == con
    assert p["winning"] != "LOCAL; weighted"


def test_f09_equal_rank_aliases_pick_one_by_config_id_never_a_joined_name(run_sel):
    a, b = "U|W-SEQ-12|i8o64|l0.1|D1", "U|W-SEQ-21|i8o64|l0.1|D1"

    def tw(inner, rel, canon):
        for k in R.SEEDS:
            inner[R.inner_name(k, b)]["recovery"]["auc"]["pair"] = 0.70
            inner[R.inner_name(k, a)] = json.loads(json.dumps(inner[R.inner_name(k, b)]))
            rel[(k, a)] = rel[(k, b)]
    out = run_sel(*_bank(tw))
    al = out["statuses"]["P*"]["aliases"]
    assert set(al["full"]) == {a, b} and al["representative"] == a and al["decided_by_config_id_tiebreak"] is True
    assert out["statuses"]["P*"]["winning"] == "SEQ-12; weighted"


# ------------------------------------------------------------------ F03: identical_to_untrained
def test_f03_untrained_exact_alias_is_disclosed_and_never_named(run_sel):
    """K-LOCAL accepts no move from C-TASK: its release IS the privacy-untrained C-TASK release."""
    k_local, ct = "U|K-LOCAL|i8o64|D1", "U|C-TASK|i8o64|D1"

    def tw(inner, rel, canon):
        for k in R.SEEDS:
            inner[R.inner_name(k, ct)]["recovery"]["auc"]["pair"] = 0.70
            inner[R.inner_name(k, k_local)] = json.loads(json.dumps(inner[R.inner_name(k, ct)]))
            rel[(k, k_local)] = rel[(k, ct)]
    out = run_sel(*_bank(tw))
    st = out["statuses"]
    assert st["T*"]["config"] == ct
    for role in ("P*", "N*"):
        assert st[role]["config"] == k_local, role
        assert st[role]["identical_to_untrained"] == [ct], role
        assert st[role]["aliases"]["representative"] == k_local              # untrained aliases never named
    assert st["P*"]["winning"] == f"LOCAL; constrained; identical to privacy-untrained {ct}"
    assert out["role_aliases"]["P*==T*"] == f"{k_local} == {ct} (exact release alias)"


# ------------------------------------------------------------------ F04: missing guard comparator keeps a fallback
def test_f04_missing_guard_comparator_keeps_the_descriptive_fallback_on_the_scored_list(run_sel):
    def tw(inner, rel, canon):                     # every T* candidate fails utility -> no T* nominee
        for k in R.SEEDS:
            for c in SEL.T_STAR_CLOSED:
                inner[R.inner_name(k, c)]["utility"] = _util(0.2, 0.2, -0.2)
    out = run_sel(*_bank(tw))
    st = out["statuses"]
    assert st["T*"]["status"] == "NO_ELIGIBLE_COMPARATOR"
    p = st["P*"]
    assert p["status"] == "INVALID_NOMINEE" and p["reason"] == "MISSING_GUARD_COMPARATOR"
    assert p["descriptive_only"] and p["fallback_rank_status"] == "INVALID_MISSING_GUARD_COMPARATOR"
    assert p["fallback_class"] == "MISSING_COMPARATOR"
    assert p["descriptive_config"] == SEL.JOINT_PAIR                       # the strongest eligible by ordering
    b = out["diagnostics"]["best_d1_fixed_privacy"]
    assert b["descriptive_config"] and b["paired_d0"] == b["descriptive_config"][:-3]
    assert out["diagnostics"]["best_weighted_privacy"]["descriptive_config"]
    labels = EL.scored_labels(out)
    assert p["descriptive_config"] in labels and b["descriptive_config"] in labels and b["paired_d0"] in labels
    assert SEL.JOINT_PAIR + R.D0SAME in labels                             # P*'s same-map D0 diagnostic


# ------------------------------------------------------------------ F05: truthful privacy-trained metadata
def test_f05_reference_metadata_truthful_and_pools_registered(run_sel, tmp_path):
    ids = R.scored_ids()
    for c in ("SRC|RAW-J_b0.3", "REF|F", "REF|E"):
        assert SEL.privacy_trained(c) and not SEL.private_code(c) and SEL.training(c) == "privacy-trained reference"
    for c in ("SRC|U", "REF|F0"):
        assert not SEL.privacy_trained(c) and SEL.training(c) == "privacy-untrained reference"
    assert len(SEL.private_all(ids)) == 77 and len(SEL.incumbent_private(ids)) == 72
    assert len(SEL.constrained(ids)) == 5
    assert not any(c.startswith(("SRC|", "REF|")) for c in SEL.private_all(ids))
    assert all(not SEL.privacy_trained(c) for c in (R.ctask_id(), "U|CLASS|i1o1|D1", "U|DIRECT-TASK|i8o64|D1"))
    run_sel(*_bank())
    pub = json.loads((tmp_path / "SELECTION.json").read_text())
    assert pub["rows"]["SRC|RAW-J_b0.3"]["privacy_trained"] is True
    assert pub["rows"]["SRC|RAW-J_b0.3"]["training"] == "privacy-trained reference"


# ------------------------------------------------------------------ F06: token states
@pytest.mark.parametrize("cid,ts", [("U|W-LOCAL|i8o64|l0.1|D1", None), ("U|W-LOCAL|i8o64|l0.1|D1", 0),
                                    ("U|W-LOCAL|i8o64|l0.1|D1", 3.5), ("U|W-LOCAL|i8o64|l0.1|D1", True),
                                    ("REF|F0", 12)])
def test_f06_malformed_token_states_is_technical_not_continuous(run_sel, cid, ts):
    def tw(inner, rel, canon):
        inner[R.inner_name(1, cid)]["token_states"] = ts
    out = run_sel(*_bank(tw))
    f = out["technical_failures"][cid]
    assert f and f[0]["code"] == "NON_ESTIMABLE_INNER_METRIC" and "token_states" in f[0]["detail"]
    assert out["rows"][cid]["ok"] is False


def test_f06_states_ordering_finite_for_codes_inf_only_for_continuous(run_sel):
    out = run_sel(*_bank())
    assert out["rows"]["SRC|U"]["mean_states"] is None
    assert out["rows"]["U|CLASS|i1o1"]["mean_states"] == 7
    r = {**out["rows"]["SRC|U"]}
    assert SEL.key(r)[2] == float("inf")


# ------------------------------------------------------------------ F07/F11: aliases by release identity, every role
def test_f07_f11_role_aliases_use_release_identity_for_every_role(run_sel):
    """K-JOINT-PAIR returns its feasible source-JOINT witness unchanged: its release equals U|JOINT|i8o64|l0.1|D1. The
    tie-break makes P* (and C*) the D1 control while N* = J* = K-JOINT-PAIR; every repeated comparison is flagged."""
    w = "U|JOINT|i8o64|l0.1|D1"

    def tw(inner, rel, canon):
        for k in R.SEEDS:
            inner[R.inner_name(k, w)]["recovery"]["auc"]["pair"] = 0.70
            inner[R.inner_name(k, SEL.JOINT_PAIR)] = json.loads(json.dumps(inner[R.inner_name(k, w)]))
            rel[(k, SEL.JOINT_PAIR)] = rel[(k, w)]
    out = run_sel(*_bank(tw))
    res, ra = out["resolved"], out["role_aliases"]
    assert res["P*"] == w and res["C*"] == w and res["N*"] == SEL.JOINT_PAIR and res["J*"] == SEL.JOINT_PAIR
    assert ra["P*==C*"] == w and ra["N*==J*"] == SEL.JOINT_PAIR
    assert ra["P*==N*"] == f"{w} == {SEL.JOINT_PAIR} (exact release alias)"
    assert ra["P*==J*"] == f"{w} == {SEL.JOINT_PAIR} (exact release alias)"
    assert ra["N*==C*"] == f"{SEL.JOINT_PAIR} == {w} (exact release alias)"
    for x in SEL.ROLES:                                                       # comparators and Q have alias sets
        assert out["statuses"][x]["aliases"]["representative"] == res[x] or \
            res[x] in out["statuses"][x]["aliases"]["full"]
    assert set(out["statuses"]["C*"]["aliases"]["full"]) == {w, SEL.JOINT_PAIR}
    assert out["statuses"]["Q"]["aliases"]["full"] == [SEL.Q_CONFIG]
    # the eval-lock re-derivation from the alias sets agrees, and the slots are flagged (prose never counts twice)
    al = EL.role_aliases(res, out["statuses"])
    assert al == ra
    by = EL.alias_of_by_role(al)
    assert by["P15"] == "P04" and by["P23"] == "P12" and "P12" not in by      # T* is not C*: P12 is no P01 alias


def test_f11_same_tokens_different_decoder_is_not_an_alias_and_renaming_is_informational(run_sel):
    d1, d0 = "U|JOINT|i8o64|l0.1|D1", "U|JOINT|i8o64|l0.1"
    ren = "U|W-JOINT|i8o64|l0.1|D1"

    def tw(inner, rel, canon):
        for k in R.SEEDS:
            inner[R.inner_name(k, d1)]["recovery"]["auc"]["pair"] = 0.70
            canon[(k, ren)] = f"canon-{k}"
            canon[(k, d1)] = f"canon-{k}"                     # same canonical identity, different byte identity
    out = run_sel(*_bank(tw))
    al = out["statuses"]["P*"]["aliases"]
    assert out["resolved"]["P*"] == d1
    assert d0 not in al["full"] and al["full"] == [d1]                       # D0/D1 share tokens: NOT an alias
    assert al["canonical_equivalent"] == [ren]
    assert out["resolved"]["C*"] == d1 and "P*==C*" in out["role_aliases"]


def test_release_identity_on_permitted_rows_only(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "UNITS", tmp_path)
    n = 12
    D = {"row_id": np.arange(100, 100 + n),
         "idx": {"DEFENSE_FIT": np.array([0, 1, 2, 3]), "AUDIT_FIT": np.array([4, 5]), "INNER_SELECTION": np.array([6, 7]),
                 "OSF_DEVELOPMENT_ASSESSMENT": np.array([8, 9, 10]), "HEAD_VALIDATION": np.array([11])}}
    rows = SEL.permitted_rows(D)
    assert list(rows["pos"]) == list(range(8))

    def write(cid, tok1, q1, k=0):
        d = tmp_path / R.unit_for(k, cid)
        d.mkdir(parents=True, exist_ok=True)
        hard1 = q1.argmax(1)
        np.savez(d / "release.npz", row_id=D["row_id"], tok1=tok1, q1=q1, hard1=hard1, alpha1=np.int64(4),
                 tok2=tok1, q2=q1, hard2=hard1, alpha2=np.int64(4))
    tok = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    q = np.where(np.arange(2)[None, :] == (tok % 2)[:, None], 0.8, 0.2)
    write("U|W-LOCAL|i8o64|l0.1|D1", tok, q)
    t2 = tok.copy()
    t2[9] = 3                                                                # differs on an ASSESSMENT row only
    write("U|W-SEQ-12|i8o64|l0.1|D1", t2, q)
    t3 = tok.copy()
    t3[6] = 0                                                                # differs on an INNER row
    write("U|W-SEQ-21|i8o64|l0.1|D1", t3, q)
    write("U|W-JOINT|i8o64|l0.1|D1", (tok + 1) % 4, q)                       # renamed tokens
    q2 = q.copy()
    q2[0] = [0.7, 0.3]
    write("U|JOINT|i8o64|l0.1|D1", tok, q2)                                  # same tokens, different decoder
    h = {c: SEL.release_identity(0, c, rows) for c in ("U|W-LOCAL|i8o64|l0.1|D1", "U|W-SEQ-12|i8o64|l0.1|D1",
                                                       "U|W-SEQ-21|i8o64|l0.1|D1", "U|W-JOINT|i8o64|l0.1|D1",
                                                       "U|JOINT|i8o64|l0.1|D1")}
    base = h["U|W-LOCAL|i8o64|l0.1|D1"]
    assert h["U|W-SEQ-12|i8o64|l0.1|D1"] == base                             # assessment bytes never enter
    assert h["U|W-SEQ-21|i8o64|l0.1|D1"][0] != base[0]
    assert h["U|W-JOINT|i8o64|l0.1|D1"][0] != base[0] and h["U|W-JOINT|i8o64|l0.1|D1"][1] == base[1]
    assert h["U|JOINT|i8o64|l0.1|D1"][0] != base[0] and h["U|JOINT|i8o64|l0.1|D1"][1] != base[1]
    assert SEL.release_identity(0, "SRC|U", rows) == (None, None)
    bad = {**D, "idx": {**D["idx"], "AUDIT_FIT": np.array([4, 8])}}
    with pytest.raises(ValueError, match="overlap"):
        SEL.permitted_rows(bad)
    with pytest.raises(ValueError, match="data handle"):
        SEL.permitted_rows(None)


def test_same_map_decoder_pairs_and_d0same_targets(run_sel):
    out = run_sel(*_bank())
    d = out["diagnostics"]
    names = {p["name"]: p for p in d["same_map_decoder_pairs"]}
    assert names["C-TASK"]["d0"] == R.ctask_id() + R.D0SAME
    assert names["P*"]["d1"] == SEL.JOINT_PAIR and names["P*"]["d0"] == SEL.JOINT_PAIR + R.D0SAME
    assert names["best_d1_fixed_privacy"]["d0"] == names["best_d1_fixed_privacy"]["d1"][:-3]
    assert d["d0same_targets"] == sorted([R.ctask_id(), SEL.JOINT_PAIR])
    assert SEL.paired_d0("U|JOINT|i8o64|l0.1") is None and SEL.paired_d0("U|CLASS|i1o1|D1") == "U|CLASS|i1o1"
