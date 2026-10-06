"""Synthetic tests of cbp.audit / cbp.baselines / cbp.assess (role D; no real data, no real labels).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m cbp.sema \\
        --label D:test-audit -- env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q \\
        cbp/tests/test_audit.py

Adapted from qpc/tests/test_audit.py (audit / baselines / assessment parts) plus the cbp-specific checks: the binding to
cbp.run, the registered 27 + 11 composition bank and its closure, finite records (token_states null for continuous
releases), the registered control limits, and the prompt sec. 14 deliberate defects relevant to the attacker role:
reversed best/worst reader, wrong AUC orientation, probabilities audited without token identities, shuffled pair
alignment, clean probabilities appended, and a missing composed-source winner. A small slate is registered for speed
under the name "tiny" (the pinned FINAL slate itself is untouched and is timed separately by `cbp.audit timing`).
"""
from __future__ import annotations

import json
import math
import subprocess
import types

import numpy as np
import pytest

from cbp import assess as AS
from cbp import audit as A
from cbp import baselines as BL
from cbp import run as R
from dpc import audit as DA
from jcv import audit as JA
from jcv.finalize import save_unit
from qpc import utility as UT
from smf import audit as SA

SMALL = dict(n_fit=3000, n_head=10, n_audit=1500, n_sel=800, n_assess=1200)


@pytest.fixture(autouse=True)
def tiny_slate(monkeypatch):
    """A small slate for synthetic tests (registered under a new name; dpc.audit's pinned slates are not edited)."""
    monkeypatch.setitem(DA.SLATES, "tiny", lambda: [("LR_C1", lambda s: JA._lr(1.0, s)),
                                                    ("MLP_16", lambda s: JA._mlp((16,), s)),
                                                    ("DA_LR", SA.da_lr)])


@pytest.fixture(scope="module")
def DS():
    return A.synthetic_D(seed=1, **SMALL)


@pytest.fixture(scope="module")
def TS(DS):
    return A.synthetic_teacher(DS, seed=1)


def _renumber_q(z, seed):
    rng = np.random.default_rng(seed)
    zp = dict(z)
    for i in (1, 2):
        perm = rng.permutation(int(z[f"alpha{i}"]))
        zp[f"tok{i}"] = perm[np.asarray(z[f"tok{i}"])]
    return zp


def _sexbit_release(D, t, i=2):
    """Collision split of recipient i by a noisy copy of the SEX label (decoded vectors unchanged)."""
    z = A.synthetic_release(D, t, 2, 4)
    S = np.asarray(D["sex"])
    noisy = SA.noisy_sex(np.where(S < 0, 0, S), 5)
    return DA.split_tokens(z, i, noisy, collide=True)


def _save(units, name, files, record):
    save_unit(units / name, {f: (lambda p, a=a: np.savez_compressed(p, **a)) for f, a in files.items()}, record)


def _save_inner(units, rec, name=None):
    rec = dict(rec)
    files = rec.pop("_files")
    _save(units, name or f"inner__{rec['unit_of']}", files, rec)


def _pub(rec):
    return json.loads(json.dumps({k: v for k, v in rec.items() if k != "_files"}, allow_nan=False))


def _arr(rec):
    return rec["_files"]["inner_preds.npz"]


# ====================================================================== binding to cbp.run and the registered bank
def test_bound_to_cbp_run_and_store():
    assert A._R() is R and BL.units_dir() == R.UNITS
    assert "cbp_v1" in str(R.UNITS) and "qpc_v1" not in str(BL.units_dir())
    assert A.parse_cid("U|JOINT|i8o64|l0.025") == R.parse_id("U|JOINT|i8o64|l0.025")
    assert A.unit_of(2, "U|SEQ-12|i8o64|l0.04") == R.unit_for(2, "U|SEQ-12|i8o64|l0.04") == \
        "pol__s2__U_SEQ-12_i8o64_l0.04"
    chk = A.registration_check()
    assert chk["ok"], chk
    U = A.expected_policies("U")
    assert len(U) == 38 and U[:27] == R.code_ids() and U[27:] == list(R.COMPOSED_EXTRA_IDS)
    assert A.expected_policies("RAW-J_b0.3") == []
    assert len([c for c in A.registered_code_bank() if A.parse_cid(c)["family"] in A.REG_PRIVACY]) == 24
    assert not set(A.COMPOSED_EXTRA) & set(R.scored_ids())                 # composition-only, never candidates


def test_registration_drift_is_refused(monkeypatch):
    fake = types.SimpleNamespace(**{k: getattr(R, k) for k in ("parse_id", "unit_for", "config_id")})
    fake.code_ids = lambda: R.code_ids()[:-1]                              # one privacy code dropped
    fake.COMPOSED_EXTRA_IDS = list(R.COMPOSED_EXTRA_IDS)
    fake.composition_ids = lambda: fake.code_ids() + fake.COMPOSED_EXTRA_IDS
    fake.scored_ids = lambda protocol=None: fake.code_ids() + ["SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"]
    assert not A.registration_check(fake)["ok"]
    monkeypatch.setattr(A, "_R", lambda: fake)
    with pytest.raises(SystemExit, match="disagrees with the registered composition bank"):
        A.expected_policies("U")
    fake.code_ids = R.code_ids
    fake.scored_ids = lambda protocol=None: R.scored_ids() + ["U|LOCAL|i8o64|l1"]   # an extra made a candidate
    assert "a composition-only extra is a selection candidate" in A.registration_check(fake)["mismatches"]


# ====================================================================== views and readers (source slate unchanged)
def test_release_key_contract_refuses_extra_and_missing(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4)
    A.check_release_keys(z)
    with pytest.raises(ValueError, match="extra \\['p1'\\]"):
        A.policy_views({**z, "p1": TS["p1"]}, DS)              # a continuous probability may never ride along
    with pytest.raises(ValueError, match="extra \\['fine2'\\]"):
        A.policy_views({**z, "fine2": z["tok2"]}, DS)
    with pytest.raises(ValueError, match="missing \\['q2'\\]"):
        A.policy_views({k: v for k, v in z.items() if k != "q2"}, DS)
    V = A.policy_views(z, DS)
    a2 = int(z["alpha2"])
    assert V["X"]["v2"].shape[1] == a2 + 6 + 6                   # one-hot over the FULL alphabet, q, one-hot decision
    assert set(np.unique(V["X"]["v2"][:, :a2])) == {0.0, 1.0}   # categorical, not an ordinal id column


def test_renumbering_leaves_every_prediction_unchanged(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.3)
    r0, a0 = A.inner_family(A.policy_views(z, DS), DS, slate="tiny")
    r1, a1 = A.inner_family(A.policy_views(_renumber_q(z, 3), DS), DS, slate="tiny")
    assert np.array_equal(r0["inner_predictions"]["P"], r1["inner_predictions"]["P"])
    assert r0["auc"] == r1["auc"] and r0["selected"] == r1["selected"]
    for k in a0:
        assert np.array_equal(a0[k], a1[k])


def test_two_ids_sharing_one_decoded_vector_stay_distinct(DS, TS):
    zc = _sexbit_release(DS, TS, i=2)
    assert DA.token_decoder_check(zc["tok2"], zc["q2"], zc["hard2"])["ok"]
    r, _ = A.inner_family(A.policy_views(zc, DS), DS, slate="tiny")
    assert r["auc"]["v2"] > 0.8                                    # token identity carries the bit
    qv = {"family": "q_only", "tokens": None, "X": {"v1": zc["q1"], "v2": zc["q2"],
                                                    "pair": np.hstack([zc["q1"], zc["q2"]])}}
    rq, _ = A.inner_family(qv, DS, slate="tiny")
    assert rq["auc"]["v2"] < 0.6                                   # decoded probabilities alone cannot see it


def test_unseen_local_token_uses_fit_prior_and_unseen_pair_rule_is_inner_selected():
    D = {"idx": {"AUDIT_FIT": np.arange(8), "INNER_SELECTION": np.arange(8, 16)}, "row_id": np.arange(16),
         "sex": np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1])}
    t1 = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 7, 1, 0, 0])   # token 7 unseen in AUDIT_FIT
    t2 = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 5, 5])   # tuples (0,5) etc unseen
    fit, sel = D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"]
    prior = D["sex"][fit].mean()
    out, _ = DA.cc_local(t1, D["sex"], fit, sel)
    assert out["CC_alpha1.0"][12 - 8] == pytest.approx(prior)          # unseen local -> AUDIT_FIT SEX prior
    n0 = (t1[fit] == 0).sum()
    n01 = ((t1[fit] == 0) & (D["sex"][fit] == 1)).sum()
    assert out["CC_alpha1.0"][0] == pytest.approx((n01 + prior) / (n0 + 1))
    _, ip = DA.cc_pair(t1, t2, D["sex"], fit, sel, np.arange(8))
    seen = ip["seen"]
    keys = DA._pair_keys(t1, t2)
    assert np.array_equal(seen, np.isin(keys[sel], keys[fit]))
    for a, rule in ip["fallback_rule"].items():
        ces = ip["fallback_ce_on_unseen_inner_rows"][a]
        assert rule == min(DA.FALLBACK_RULES, key=lambda k: (ces[k], DA.FALLBACK_RULES.index(k)))
    Dc = {**D, "idx": {**D["idx"], "OSF_DEFENSE_FIT": np.arange(0), "OSF_DEVELOPMENT_ASSESSMENT": np.arange(0)}}
    cov = A.coverage_receipt(t1, t2, Dc, 8, 6)
    assert cov["v1"]["fallback_use"]["INNER_SELECTION"]["fallback_rows"] == 1
    assert cov["v1"]["fallback_use"]["AUDIT_FIT"]["fallback_rows"] == 0
    assert cov["pair"]["fallback_use"]["INNER_SELECTION"]["fallback_rows"] == int((~seen).sum())
    assert cov["pair"]["declared_tuple_space"] == 48


def _reversed_relation_release(DS):
    S = np.asarray(DS["sex"])
    fit, sel = DS["idx"]["AUDIT_FIT"], DS["idx"]["INNER_SELECTION"]
    tok = np.zeros(len(S), dtype=np.int64)
    tok[fit] = S[fit]                     # fit: token = SEX
    tok[sel] = 1 - S[sel]                 # inner: token = 1 - SEX (reversed relation)
    q = np.where(tok[:, None] == 1, [0.3, 0.7], [0.7, 0.3])
    return {"row_id": DS["row_id"], "tok1": tok, "q1": q, "hard1": q.argmax(1), "alpha1": np.asarray(2),
            "tok2": np.zeros(len(S), dtype=np.int64), "q2": np.tile(np.eye(6)[0] * 0.4 + 0.1, (len(S), 1)),
            "hard2": np.zeros(len(S), dtype=np.int64), "alpha2": np.asarray(1)}


def test_auc_orientation_fixed_never_flipped(DS):
    r, _ = A.inner_family(A.policy_views(_reversed_relation_release(DS), DS), DS, slate="tiny")
    assert r["auc"]["v1"] < 0.05 and r["auc_seed0"]["v1"] < 0.05      # far below chance, reported as is


def test_auc_and_ce_selection_are_separate():
    y = np.array([0, 1] * 50)
    rank_perfect_uncal = np.where(y == 1, 0.501, 0.499)          # AUC 1, CE ~ log 2
    calibrated = np.where(y == 1, 0.9, 0.1)
    calibrated[:10] = 0.95                                       # a few ranking errors on y = 0 rows
    preds = {"v1:A": rank_perfect_uncal, "v1:B": calibrated, "v2:A": calibrated, "v2:B": calibrated,
             "pair:A": calibrated, "pair:B": calibrated}
    sel, _ = DA._select(preds, [""], y, np.arange(len(y)))
    assert sel["v1"]["auc"]["attacker"] == "A" and sel["v1"]["ce"]["attacker"] == "B"


def test_selected_attackers_refit_at_seeds_and_mean_is_reported(DS, TS):
    t = dict(TS)
    S = np.asarray(DS["sex"])
    rng = np.random.default_rng(0)
    t["c1"] = t["c1"] + 0.3 * np.where(S < 0, 0, S)[:, None] * np.array([1, -1]) + rng.normal(0, 0.01, t["c1"].shape)
    V = BL.source_view_sets(t, DS, families=("interface",))["interface"]
    r, arr = A.inner_family(V, DS, slate="tiny")
    for w in DA.PRIMARY_VIEWS:
        assert r["auc"][w] == pytest.approx(np.mean(r["auc_per_seed"][w]))
        assert r["auc_per_seed"][w][0] == pytest.approx(r["auc_seed0"][w])
        assert arr[f"auc_{w}"].shape == (3, len(DS["idx"]["INNER_SELECTION"]))
    assert r["attacker_seeds"] == [0, 1, 2]


def test_mi_diagnostic_exact_and_fixed_permutations(DS):
    fit = DS["idx"]["OSF_DEFENSE_FIT"]
    S = np.asarray(DS["sex"])
    p = S[fit].mean()
    H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    t_full = np.where(S < 0, 0, S)
    t_const = np.zeros(len(S), dtype=np.int64)
    m = A.mi_diagnostic(t_full, t_const, DS, perms=10)
    assert m["views"]["v1"]["fitted_mi"] == pytest.approx(H, rel=1e-12)
    assert m["views"]["v2"]["fitted_mi"] == pytest.approx(0.0, abs=1e-15)
    assert m["views"]["pair"]["fitted_mi"] == pytest.approx(H, rel=1e-12)
    assert m["views"]["v1"]["null_max"] < 0.01 and m["views"]["v1"]["z"] > 10
    assert A.mi_diagnostic(t_full, t_const, DS, perms=10)["perm_list_sha256"] == m["perm_list_sha256"]


def test_reference_views_cells_and_value_tokens(DS, TS):
    zr = A.synthetic_reference_cells(DS, TS, ncell=(5, 7))
    sets, out, _ = BL.reference_view_sets("F", 0, DS, arrays=zr, with_outputs=True)
    assert set(sets) == {"interface", "scores", "probs", "decisions", "cells"}
    assert sets["cells"]["tokens"]["v2"] is not None and sets["interface"]["tokens"] is not None
    assert np.array_equal(out["hard2"], zr["d2"])
    se = BL.reference_view_sets("E", 0, DS, arrays=zr)
    assert "complete" in se and se["interface"]["tokens"] is None      # E's views are continuous


# ====================================================================== unit contract + composed closure (explicit)
@pytest.fixture()
def store(tmp_path, DS, TS):
    units = tmp_path / "units"
    units.mkdir()
    _save(units, "tea__s0__U", {"teacher.npz": TS}, {"t": "U"})
    _save(units, "tea__s0__RAW-J_b0.3", {"teacher.npz": A.synthetic_teacher(DS, seed=9)}, {"t": "RAW-J"})
    _save(units, "pol__s0__U_DIRECT-TASK_i2o4", {"release.npz": A.synthetic_release(DS, TS, 2, 4)}, {})
    _save(units, "pol__s0__U_JOINT_i2o4_l0.1", {"release.npz": A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.6)}, {})
    _save(units, "pol__s0__U_CLASS_i1o1", {"release.npz": A.synthetic_release(DS, TS, 1, 1)}, {})
    _save(units, "ref__s0__F", {"reference.npz": A.synthetic_reference_cells(DS, TS, ncell=(5, 7))}, {})
    return units


POLS = ["U|DIRECT-TASK|i2o4", "U|JOINT|i2o4|l0.1", "U|CLASS|i1o1"]


def _release(units, cid, k=0):
    z = np.load(units / A.unit_of(k, cid) / "release.npz")
    return {x: z[x] for x in z.files}


def test_inner_unit_contract_finite_records_and_composed_source_bank(store, DS):
    recs = {}
    for c in POLS:
        r = A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny")
        recs[c] = _pub(r)
        assert A.validate_inner(recs[c], _arr(r), DS, release=_release(store, c))["ok"]
        _save_inner(store, r)
    p = recs["U|JOINT|i2o4|l0.1"]
    for key in ("recovery", "utility", "utility_gate", "preserved", "token_states", "coverage_receipt",
                "mi_diagnostic", "cbp_headroom", "decision_preservation"):
        assert key in p
    assert p["schema"] == "cbp-inner-v1"
    assert set(p["recovery"]["auc"]) == {"v1", "v2", "pair"} and "ce" in p["recovery"]
    assert set(p["utility"]) == {"income", "occupation"} and p["preserved"] == {"1": True, "2": True}
    assert p["token_states"] == 4 + 21                           # 2 x 2 income + 5 x 4 occupation + 1 reserved
    assert p["mi_diagnostic"]["views"]["pair"]["null_mean"] is not None
    assert p["coverage_receipt"]["pair"]["fallback_use"]["INNER_SELECTION"]["rows"] == SMALL["n_sel"]
    assert p["recovery"]["auc"]["v2"] > recs["U|DIRECT-TASK|i2o4"]["recovery"]["auc"]["v2"]
    # cbp headroom diagnostic uses 0.006 / 0.0035, not qpc.utility's 0.0075 flag
    h = p["cbp_headroom"]
    for t in ("income", "occupation"):
        assert h[t]["ll_ok"] == (p["utility_gate"][t]["ll_excess"] <= 0.006)
        assert h[t]["brier_ok"] == (p["utility_gate"][t]["brier_excess"] <= 0.0035)
    # the source bank composes with every code BEFORE selection; the leaky code's reader wins
    src = A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)
    sp = _pub(src)
    c = sp["composed"]
    assert c["winner"]["v2"] == "U|JOINT|i2o4|l0.1"
    assert sp["recovery"]["auc"]["v2"] == pytest.approx(p["recovery"]["auc"]["v2"])
    assert sp["recovery"]["own"]["auc"]["v2"] < sp["recovery"]["auc"]["v2"]
    assert "U|JOINT|i2o4|l0.1" in c["freeze"] and c["closure"]["ok"]
    assert c["policy_list_source"].startswith("EXPLICIT")
    assert sp["token_states"] is None and sp["preserved"] == {"1": True, "2": True}   # null, never Infinity
    assert sp["families"]["decisions"]["composed"]["policies"] == ["U|CLASS|i1o1"]
    assert A.validate_inner(sp, _arr(src), DS, registered=False)["ok"]
    _save_inner(store, src)
    assert A.composed_freeze_list(0, "U", store) == c["freeze"]
    rj = A.inner_unit("source", 0, "SRC|RAW-J_b0.3", DS, units_dir=store, slate="tiny", policy_cids=[])
    assert rj["composed"]["freeze"] == [] and rj["recovery"]["auc"] == rj["recovery"]["own"]["auc"]
    ref = A.inner_unit("reference", 0, "REF|F", DS, units_dir=store, slate="tiny")
    assert ref["preserved"] == {"1": True, "2": True} and "cells" in ref["families"]
    assert ref["token_states"] is None
    assert A.validate_inner(_pub(ref), _arr(ref), DS)["ok"]


def test_nonfinite_record_values_are_a_technical_failure():
    rec = {"cid": "x", "recovery": {"auc": {w: 0.6 for w in DA.PRIMARY_VIEWS}, "ce": {w: 0.6 for w in
                                                                                   DA.PRIMARY_VIEWS}},
           "utility": {"income": {"logloss": float("nan")}}, "token_states": None}
    with pytest.raises(RuntimeError, match="nonfinite values"):
        A._check_finite(rec)
    rec["utility"]["income"]["logloss"] = 0.4
    rec["token_states"] = float("inf")                            # Infinity is never a token-state count
    with pytest.raises(RuntimeError, match="nonfinite"):
        A._check_finite(rec)
    rec["token_states"] = None
    A._check_finite(rec)


def test_composed_bank_refuses_without_closure(store, DS):
    for c in POLS[:2]:
        _save_inner(store, A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny"))
    with pytest.raises(SystemExit, match="not closed"):           # the CLASS code's inner unit is missing
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)
    with pytest.raises(SystemExit, match=r"'unexpected_release_units': \['pol__s0__U_CLASS_i1o1'\]"):
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS[:2])


def test_composed_bank_refuses_records_from_other_rows(store, DS):
    for c in POLS:
        r = A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny")
        if c == "U|CLASS|i1o1":
            r["recovery"]["sel_row_id_sha256"] = "0" * 64              # scored on different INNER rows
        _save_inner(store, r)
    with pytest.raises(SystemExit, match="differs from the source on sel_row_id_sha256"):
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)


def test_policy_preservation_failure_is_recorded(store, DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4)
    z["hard2"] = z["hard2"].copy()
    j = int(np.flatnonzero(z["hard2"] == 0)[0])
    z["q2"] = z["q2"].copy()
    z["q2"][j] = np.eye(6)[1] * 0.88 + 0.02                         # decodes consistently, but changes one decision
    z["hard2"][j] = 1
    z["tok2"] = z["tok2"].copy()
    z["tok2"][j] = int(z["alpha2"])
    z["alpha2"] = np.asarray(int(z["alpha2"]) + 1)
    _save(store, "pol__s0__U_FINE-TASK_i2o4", {"release.npz": z}, {})
    r = A.inner_unit("policy", 0, "U|FINE-TASK|i2o4", DS, units_dir=store, slate="tiny")
    assert r["preserved"] == {"1": True, "2": False} and not r["utility_gate"]["eligible"]


# ====================================================================== the registered 27 + 11 composition bank
@pytest.fixture()
def reg_store(tmp_path, DS, TS):
    """cbp-named store with the REGISTERED 38-code composition bank of U at seed 0. Three distinct releases are audited
    (a plain i2o4 code, a SEX-tilted JOINT l0.1 code, the class-only code); every other registered name reuses the plain
    release and its inner record (composition reads records only)."""
    units = tmp_path / "units"
    units.mkdir()
    _save(units, "tea__s0__U", {"teacher.npz": TS}, {"t": "U"})
    _save(units, "tea__s0__RAW-J_b0.3", {"teacher.npz": A.synthetic_teacher(DS, seed=9)}, {"t": "RAW-J"})
    _save(units, "ref__s0__F", {"reference.npz": A.synthetic_reference_cells(DS, TS, ncell=(5, 7))}, {})
    rel = {"plain": A.synthetic_release(DS, TS, 2, 4), "leaky": A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.6),
           "cls": A.synthetic_release(DS, TS, 1, 1)}
    kind = {"U|JOINT|i8o64|l0.1": "leaky", "U|CLASS|i1o1": "cls"}
    recs = {}
    for cid in A.registered_composition_bank():
        kd = kind.get(cid, "plain")
        _save(units, A.unit_of(0, cid), {"release.npz": rel[kd]}, {})
        if kd not in recs:
            r = A.inner_unit("policy", 0, cid, DS, units_dir=units, slate="tiny")
            recs[kd] = (_pub(r), _arr(r))
        pub, arr = recs[kd]
        _save(units, f"inner__{A.unit_of(0, cid)}", {"inner_preds.npz": arr},
              {**pub, "cid": cid, "unit_of": A.unit_of(0, cid)})
    return units


def test_source_composes_over_the_registered_bank_in_order(reg_store, DS):
    src = A.inner_unit("source", 0, "SRC|U", DS, units_dir=reg_store, slate="tiny")
    c = src["composed"]
    assert c["policies"] == A.registered_composition_bank() and len(c["policies"]) == 38
    assert c["closure"]["ok"] and c["closure"]["expected"] == 38 and c["closure"]["release_units_on_disk"] == 38
    assert c["composition_only"] == list(A.COMPOSED_EXTRA)
    assert c["policy_list_source"].startswith("registered")
    assert c["winner"]["v2"] == "U|JOINT|i8o64|l0.1"              # the leaky code's reader wins
    # identical records under several names: the FIRST in registered order wins ties
    assert all(w in ("source", "U|DIRECT-TASK|i8o64", "U|JOINT|i8o64|l0.1", "U|CLASS|i1o1")
               for f in c["per_family"].values() for w in f["winner"].values())
    assert A.validate_inner(_pub(src), _arr(src), DS, registered=True)["ok"]


def test_registered_closure_refuses_missing_and_unexpected_units(reg_store, DS):
    import shutil
    victim = A.unit_of(0, "U|SEQ-21|i8o64|l0.06")
    shutil.rmtree(reg_store / victim)                                    # a registered release unit missing
    with pytest.raises(SystemExit, match=r"'missing_release_units': \['U\|SEQ-21\|i8o64\|l0\.06'\]"):
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=reg_store, slate="tiny")


def test_registered_closure_refuses_a_planted_or_stray_unit(reg_store, DS, TS):
    z = DA.split_tokens(A.synthetic_release(DS, TS, 2, 4), 2, np.asarray(DS["sex"]).clip(0), collide=True)
    _save(reg_store, "pol__s0__U_JOINT_i8o64_l0.2", {"release.npz": z}, {})   # unregistered lambda / a plant
    with pytest.raises(SystemExit, match=r"'unexpected_release_units': \['pol__s0__U_JOINT_i8o64_l0\.2'\]"):
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=reg_store, slate="tiny")


def test_validate_all_over_the_saved_registered_bank(reg_store, DS, TS):
    for cid in ("SRC|U", "SRC|RAW-J_b0.3", "REF|F"):
        kind = A.parse_cid(cid)["kind"]
        _save_inner(reg_store, A.inner_unit(kind, 0, cid, DS, units_dir=reg_store, slate="tiny"))
    v = A.validate_all(DS, reg_store, seeds=(0,))
    assert v["checked"] == 38 + 3 and not v["defects"], v["defects"][:2]
    assert sorted(v["missing"]) == ["inner__ref__s0__E", "inner__ref__s0__F0"] and not v["ok"]
    # a stored release that no longer matches its audited views is reported (not raised)
    _save(reg_store, A.unit_of(0, "U|LOCAL|i8o64|l0.08"), {"release.npz": A.synthetic_release(DS, TS, 2, 4,
                                                                                              sex_tilt=0.2)}, {})
    v2 = A.validate_all(DS, reg_store, seeds=(0,))
    assert len(v2["defects"]) == 1 and "U|LOCAL|i8o64|l0.08" in v2["defects"][0]


# ====================================================================== deliberate defects (prompt sec. 14)
def _policy_record(store, DS, cid="U|JOINT|i2o4|l0.1"):
    r = A.inner_unit("policy", 0, cid, DS, units_dir=store, slate="tiny")
    return _pub(r), _arr(r), _release(store, cid)


def test_defect_reversed_best_worst_reader_is_caught(store, DS, monkeypatch):
    rec, arr, z = _policy_record(store, DS)
    assert A.validate_inner(rec, arr, DS, release=z)["ok"]
    orig = DA._pick

    def worst(rows, key, maximize):                                      # the defect: best and worst swapped
        return orig(rows, key, not maximize)
    monkeypatch.setattr(DA, "_pick", worst)
    bad, barr, _ = _policy_record(store, DS)
    monkeypatch.setattr(DA, "_pick", orig)
    with pytest.raises(A.AuditDefect, match="best/worst reversal"):
        A.validate_inner(bad, barr, DS, release=z)


def test_defect_wrong_auc_orientation_is_caught(store, DS, monkeypatch):
    z = _reversed_relation_release(DS)
    _save(store, "pol__s0__U_LOCAL_i2o1_l0.1", {"release.npz": z}, {})
    r = A.inner_unit("policy", 0, "U|LOCAL|i2o1|l0.1", DS, units_dir=store, slate="tiny")
    assert r["recovery"]["auc"]["v1"] < 0.05
    assert A.validate_inner(_pub(r), _arr(r), DS, release=z)["ok"]
    orig = DA.auc1
    monkeypatch.setattr(DA, "auc1", lambda y, p: max(orig(y, p), 1 - orig(y, p)))   # the defect: flip to > 0.5
    bad = A.inner_unit("policy", 0, "U|LOCAL|i2o1|l0.1", DS, units_dir=store, slate="tiny")
    monkeypatch.setattr(DA, "auc1", orig)
    assert bad["recovery"]["selection"]["v1"]["auc"]["inner_auc"] > 0.95
    with pytest.raises(A.AuditDefect, match="orientation"):
        A.validate_inner(_pub(bad), _arr(bad), DS, release=z)


def test_defect_probabilities_without_token_identities_is_caught(store, DS, monkeypatch):
    def q_only(z, D, meta=None):                                         # the defect: tokens dropped
        A.check_release_keys(z)
        zz = DA.align(z, D)
        X = {f"v{i}": np.hstack([zz[f"q{i}"], DA.onehot(zz[f"hard{i}"], K)]) for i, K in ((1, 2), (2, 6))}
        X["pair"] = np.hstack([X["v1"], X["v2"]])
        return {"family": "code", "X": X, "tokens": None, "meta": {**(meta or {})}}
    z = _release(store, "U|JOINT|i2o4|l0.1")
    good = A.policy_views
    monkeypatch.setattr(A, "policy_views", q_only)
    r = A.inner_unit("policy", 0, "U|JOINT|i2o4|l0.1", DS, units_dir=store, slate="tiny")
    monkeypatch.setattr(A, "policy_views", good)
    with pytest.raises(A.AuditDefect, match="without token-identity readers"):
        A.validate_inner(_pub(r), _arr(r), DS, release=z)


def test_defect_shuffled_pair_alignment_is_caught(store, DS, TS, monkeypatch):
    good = A.policy_views
    perm = np.random.default_rng(0).permutation(len(DS["row_id"]))

    def misaligned(z, D, meta=None):                                     # the defect: v2 rows shuffled in the pair
        V = good(z, D, meta)
        X = dict(V["X"])
        X["pair"] = np.hstack([X["v1"], X["v2"][perm]])
        return {**V, "X": X}
    z = _release(store, "U|JOINT|i2o4|l0.1")
    monkeypatch.setattr(A, "policy_views", misaligned)
    r = A.inner_unit("policy", 0, "U|JOINT|i2o4|l0.1", DS, units_dir=store, slate="tiny")
    monkeypatch.setattr(A, "policy_views", good)
    with pytest.raises(A.AuditDefect, match="pair alignment"):
        A.validate_inner(_pub(r), _arr(r), DS, release=z)
    # and the XOR control is the real-data guard: misaligned pairs cannot see a coalition-only plant
    S = np.asarray(DS["sex"])
    halves = SA.null_split(DS)
    Sp, _ = SA.frozen_permutation(S, DS, halves)
    noisy = SA.noisy_sex(Sp, A.CONTROL_SEED)
    b1 = np.random.default_rng(A.CONTROL_SEED + 7).integers(0, 2, len(noisy))
    zx = DA.split_tokens(DA.split_tokens(A.synthetic_release(DS, TS, 2, 4), 1, b1), 2, b1 ^ noisy)
    ok = A._split_audit(good(zx, DS), Sp, DS, halves, "tiny")
    V = good(zx, DS)
    Vm = {**V, "X": {**V["X"], "pair": np.hstack([V["X"]["v1"], V["X"]["v2"][perm]])},
          "tokens": {"v1": V["tokens"]["v1"], "v2": V["tokens"]["v2"][perm]}}
    mis = A._split_audit(Vm, Sp, DS, halves, "tiny")
    assert ok["pair"]["heldout_auc_B"] > A.PLANT_MIN >= mis["pair"]["heldout_auc_B"]


def test_defect_clean_probabilities_appended_is_caught(store, DS, TS, monkeypatch):
    z = _release(store, "U|JOINT|i2o4|l0.1")
    with pytest.raises(ValueError, match="extra"):
        A.policy_views({**z, "p2": TS["p2"]}, DS)                       # the release contract refuses it
    with pytest.raises(ValueError, match="extra"):
        A.lazy_policy_views({**z, "p2": TS["p2"]}, DS)
    good = A.policy_views

    def appended(zz, D, meta=None):                                      # the defect: U's p2 ridden along in v2
        V = good(zz, D, meta)
        X = dict(V["X"])
        X["v2"] = np.hstack([X["v2"], TS["p2"]])
        X["pair"] = np.hstack([X["v1"], X["v2"]])
        return {**V, "X": X}
    monkeypatch.setattr(A, "policy_views", appended)
    r = A.inner_unit("policy", 0, "U|JOINT|i2o4|l0.1", DS, units_dir=store, slate="tiny")
    monkeypatch.setattr(A, "policy_views", good)
    with pytest.raises(A.AuditDefect, match="appended probabilities"):
        A.validate_inner(_pub(r), _arr(r), DS, release=z)
    rec, arr, _ = _policy_record(store, DS)
    with pytest.raises(A.AuditDefect, match="release refused"):
        A.validate_inner(rec, arr, DS, release={**z, "p2": TS["p2"]})


def test_defect_missing_composed_source_winner_is_caught(store, DS):
    for c in POLS:
        _save_inner(store, A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny"))
    src = A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)
    sp, arr = _pub(src), _arr(src)
    assert A.validate_inner(sp, arr, DS, registered=False)["ok"]
    bad = json.loads(json.dumps(sp))
    bad["composed"]["freeze"] = [c for c in bad["composed"]["freeze"] if c != "U|JOINT|i2o4|l0.1"]
    with pytest.raises(A.AuditDefect, match="missing from composed.freeze"):
        A.validate_inner(bad, arr, DS, registered=False)
    assert any("frozen policy" in x for x in A.validate_composed(
        {**sp, "composed": {**sp["composed"], "freeze": sp["composed"]["freeze"] + ["U|FINE-TASK|i9o9"]}}))
    # an explicit (non-registered) bank never passes as the registered one
    with pytest.raises(A.AuditDefect, match="registered composition bank"):
        A.validate_inner(sp, arr, DS, registered=True)


# ====================================================================== controls with explicit, registered pass rules
def test_control_limits_are_the_source_limits():
    src = json.loads((R.WT / "results" / "pcrl_confidence_capacity_v1" / "AUDIT_PRELOCK_CHECKS.json").read_text())
    assert A.CONTROL_LIMITS["null_z"] == src["thresholds"]["null_z"] == 3.5
    assert A.CONTROL_LIMITS["plant_min"] == src["thresholds"]["plant_min"] == 0.75
    assert A.CONTROL_LIMITS["rot_amplitude"] == src["thresholds"]["rot_amplitude"]
    thr = {r["null_threshold"] for r in src["releases"].values()}
    sd = {r["null_sd_B"] for r in src["releases"].values()}
    assert thr == {A.SOURCE_NULL_THRESHOLD} and sd == {A.SOURCE_NULL_SD0}
    assert A.SOURCE_NULL_THRESHOLD == pytest.approx(0.5 + 3.5 * A.SOURCE_NULL_SD0, abs=1e-15)
    json.dumps(A.CONTROL_LIMITS, allow_nan=False)


def test_controls_detect_every_plant_and_pass_the_null(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4)
    r = A.controls_for_release("code", A.policy_views(z, DS), DS, z_policy=z, slate="tiny")
    ch = r["checks"]
    assert ch["NULL"]["ok"], ch["NULL"]
    for p in ("CONF_r1", "CONF_r2", "COLL_r1", "COLL_r2", "XOR"):
        assert ch[p]["ok"], (p, ch[p])
    assert ch["CONF_r2"]["decisions_only_misses_it"]               # a decision-only audit cannot see the plant
    assert ch["COLL_r2"]["decoded_probability_only_misses_it"]      # nor can a decoded-probability-only audit
    assert ch["XOR"]["locals_null_ok"] and r["all_ok"]
    V = BL.source_view_sets(TS, DS, families=("interface",))["interface"]
    rs = A.controls_for_release("src", V, DS, rotate=True, slate="tiny")
    assert rs["checks"]["ROT_r1"]["ok"] and rs["checks"]["ROT_r2"]["ok"], rs["checks"]
    assert rs["checks"]["ROT_r2"]["serialisation_exact"]


def test_control_failure_is_reported_not_dropped(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4)
    S = np.asarray(DS["sex"])
    halves = SA.null_split(DS)
    Sp, _ = SA.frozen_permutation(S, DS, halves)
    z2 = DA.split_tokens(z, 1, SA.noisy_sex(Sp, A.CONTROL_SEED), collide=True)   # a release carrying S* itself
    r = A.controls_for_release("leaky", A.policy_views(z2, DS), DS, slate="tiny")
    assert not r["checks"]["NULL"]["ok"] and "NULL" in r["failures"] and not r["all_ok"]


def test_stage_controls_shards_merge_threshold_receipt_and_refuse_unsealed(store, DS, tmp_path):
    plan = {"policies": [(0, "U|DIRECT-TASK|i2o4")], "sources": [], "references": [], "rotate_references": [],
            "null_policy": (0, "U|DIRECT-TASK|i2o4"), "null_reps": 2}
    out = tmp_path / "AUDIT_PRELOCK_CHECKS.json"
    p0 = A.stage_controls(DS, "0/2", units_dir=store, out_path=out, slate="tiny", plan=plan, source_threshold=None)
    assert p0["status"] == "PARTIAL" and not out.exists()
    v = A.stage_controls(DS, "1/2", units_dir=store, out_path=out, slate="tiny", plan=plan, source_threshold=None)
    assert out.exists() and v["all_ok"] and v["null_calibration_present"]
    pub = json.loads(out.read_text())
    assert pub["study"] == A.STUDY and pub["registered_limits"]["plant_min"] == 0.75
    assert pub["verdict"]["positive_controls"]["U|DIRECT-TASK|i2o4|s0"]["XOR"]
    # the realised-threshold receipt: the synthetic split cannot reproduce the source threshold -> technical failure
    parts = [json.loads(p.read_text())["result"] for p in sorted((store.parent / "controls").glob("job*.json"))]
    s = A.summarise_controls(parts, plan, "tiny")
    assert not s["verdict"]["all_ok"] and s["verdict"]["realised_threshold_matches_source"] is False
    assert any(f.startswith("REALISED_NULL_THRESHOLD") for f in s["verdict"]["failures"])
    with pytest.raises(SystemExit, match="sealed"):
        A.stage_controls({**DS, "sealed": False}, None, units_dir=store, out_path=out, slate="tiny", plan=plan)


def test_control_plan_is_structural():
    p = A.control_plan()
    assert [c for _, c in p["policies"]] == ["U|DIRECT-TASK|i8o64", "U|JOINT|i8o64|l0.1", "U|CLASS|i1o1"]
    assert all(c in R.code_ids() for _, c in p["policies"])
    assert p["null_policy"] == (0, "U|DIRECT-TASK|i8o64") and p["rotate_references"] == ["E"]


# ====================================================================== synthetic timing / estimates
def test_estimates_cover_the_registered_bank_and_are_finite():
    pin = {r: {"unit_cpu_s": v} for r, v in (("i1o1", 4.0), ("i8o8", 9.0), ("i8o16", 11.0), ("i8o32", 17.0),
                                             ("i8o64", 30.0))}
    meas = {"policy_inner": pin, "source_unit": {"SRC|U_cpu_s": 25.0, "SRC|RAW-J_cpu_s": 23.0,
                                                 "composed_assembly_cpu_s": 0.5},
            "reference_unit": {"E_cpu_s": 25.0, "F_cpu_s": 34.0},
            "controls": {"code_null_and_5_plants_cpu_s": 280.0, "code_null_only_cpu_s": 40.0,
                         "source_interface_null_and_2_rot_cpu_s": 14.0},
            "final": {"code_i8o64_one_family_cpu_s": 34.0, "source_own_five_families_cpu_s": 50.0}}
    e = A.estimates(meas)
    assert e["inner_units"] == 96 + 33
    assert e["inner_cpu_s_per_seed_parts"]["registered_codes"] == 26 * 30.0 + 4.0
    assert e["inner_cpu_s_per_seed_parts"]["composition_only"] == 9 + 11 + 17 + 30 + 9 + 11 + 17 + 4 * 30
    assert e["total_cpu_h"] > 0 and e["budget_with_x2_margin_cpu_h"] == round(2 * e["total_cpu_h"], 2)
    json.dumps(e, allow_nan=False)


# ====================================================================== the locked assessment (cbp.assess)
def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True)


@pytest.fixture()
def repo(tmp_path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    r = tmp_path / "wt"
    r.mkdir()
    _git(r, "init", "-q", "-b", AS.STUDY_BRANCH)
    _git(r, "config", "user.email", "t@example.org")
    _git(r, "config", "user.name", "t")
    _git(r, "remote", "add", "origin", str(origin))
    (r / "README").write_text("x\n")
    _git(r, "add", "README")
    _git(r, "commit", "-q", "-m", "init")
    _git(r, "push", "-q", "origin", AS.STUDY_BRANCH)
    return r


def _write_lock(repo, body=None):
    p = repo / AS.LOCK_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body or {"seeds": {}, "locked_code_files": {}}))
    return p


def _push(repo, msg="lock"):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    _git(repo, "push", "-q", "origin", AS.STUDY_BRANCH)


def test_assess_is_bound_to_the_cbp_lock_branch_and_chain():
    assert AS.STUDY_BRANCH == "research/pcrl-confidence-budgeted-privacy-v1"
    assert AS.LOCK_REL == "results/pcrl_confidence_budgeted_privacy_v1/EVALUATION_LOCK.json"
    for f in ("cbp/assess.py", "cbp/audit.py", "cbp/baselines.py", "cbp/data.py", "cbp/run.py", "qpc/utility.py",
              "dpc/audit.py"):
        assert f in AS.CHAIN
    assert not any(f.startswith("qpc/") and f not in ("qpc/utility.py", "qpc/data.py") for f in AS.CHAIN)
    assert AS.AU is A and AS.BL is BL


def test_gate_refuses_uncommitted_unpushed_modified_and_misnamed_locks(repo):
    p = _write_lock(repo)
    assert AS.lock_is_pushed(p, repo, fetch=False)["reason"] == "lock file is not committed"
    _git(repo, "add", AS.LOCK_REL)
    _git(repo, "commit", "-q", "-m", "lock")
    r = AS.lock_is_pushed(p, repo, fetch=False)
    assert not r["ok"] and "not on origin" in r["reason"]
    _git(repo, "push", "-q", "origin", AS.STUDY_BRANCH)
    assert AS.lock_is_pushed(p, repo)["ok"]
    p.write_text(p.read_text() + " ")
    assert not AS.lock_is_pushed(p, repo)["ok"]
    other = repo / "results" / "EVALUATION_LOCK.json"
    other.write_text("{}")
    assert not AS.lock_is_pushed(other, repo)["ok"]
    wrong = repo / "LOCK.json"
    wrong.write_text("{}")
    assert "must be named" in AS.lock_is_pushed(wrong, repo)["reason"]


def test_open_requires_chain_hashes_loaded_modules_and_outer_requires_open(repo):
    AS.close_assessment()
    with pytest.raises(SystemExit, match="sealed"):
        AS.outer_unit({"sealed": False}, 0, "x", {"kind": "policy", "cid": "U|CLASS|i1o1"})
    with pytest.raises(SystemExit, match="sealed"):
        AS.load_unsealed()
    f = repo / "code.py"
    f.write_text("print(1)\n")
    p = _write_lock(repo, {"seeds": {}, "locked_code_files": {"code.py": AS._sha(f)}})
    _push(repo)
    with pytest.raises(SystemExit, match="does not list"):
        AS.open_assessment(p, repo, chain=("code.py", "other.py"))
    with pytest.raises(SystemExit, match="loaded worktree modules are not locked"):
        AS.open_assessment(p, repo, chain=("code.py",))          # cbp modules are loaded but not in this lock
    lock = json.loads(p.read_text())
    assert AS.verify_code(lock, repo, chain=("code.py",), check_modules=False)["code.py"] == "matches lock"
    f.write_text("print(2)\n")
    with pytest.raises(SystemExit, match="differs from the locked code hash"):
        AS.verify_code(lock, repo, chain=("code.py",), check_modules=False)
    AS.close_assessment()


def test_cbp_data_refuses_unsealing_outside_assess():
    from cbp import data as CD
    with pytest.raises((PermissionError, SystemExit)):
        CD.load(unseal=True)


def test_jobs_are_only_the_locked_list_and_shards_partition_it():
    lock = {"seeds": {str(k): {"score": {f"L{j}": {"kind": "policy", "cid": "U|CLASS|i1o1"} for j in range(5)}}
                      for k in (0, 1, 2)}}
    jobs = AS.jobs_from_lock(lock)
    assert len(jobs) == 15
    sh = [AS.jobs_from_lock(lock, shard=f"{i}/2") for i in (0, 1)]
    assert sorted(map(str, sh[0] + sh[1])) == sorted(map(str, jobs)) and not set(map(str, sh[0])) & set(map(str, sh[1]))
    assert AS.jobs_from_lock(lock, seeds=(1,), labels=["L2"]) == [(1, "L2", lock["seeds"]["1"]["score"]["L2"])]
    assert AS._policy_cid_of("pol__s0__U_JOINT_i8o64_l0.025") == "U|JOINT|i8o64|l0.025"
    assert AS._policy_cid_of("U|CLASS|i1o1") == "U|CLASS|i1o1"


def _unsealed(D, seed=3):
    rng = np.random.default_rng(seed)
    Du = {**D, "y": {k: v.copy() for k, v in D["y"].items()}, "sealed": False}
    a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    Du["y"]["income"][a] = rng.integers(0, 2, len(a))
    Du["y"]["occupation_group"][a] = rng.integers(0, 6, len(a))
    Du["race"] = rng.integers(0, 3, len(D["row_id"]))
    return Du


def test_outer_unit_end_to_end_composed_freeze_registered_bank_and_restore_hook(repo, reg_store, DS, monkeypatch):
    src = A.inner_unit("source", 0, "SRC|U", DS, units_dir=reg_store, slate="tiny")
    _save_inner(reg_store, src)
    freeze = src["composed"]["freeze"]
    assert "U|JOINT|i8o64|l0.1" in freeze
    score = {"SRC|U": {"kind": "source", "cid": "SRC|U", "unit": "tea__s0__U"},
             "J*": {"kind": "policy", "cid": "U|JOINT|i8o64|l0.1"},
             "REF|F": {"kind": "reference", "cid": "REF|F"}}
    lock = {"seeds": {"0": {"u_label": "SRC|U", "score": score,
                            "composed_policies": {"U": ["U|DIRECT-TASK|i8o64", "U|CLASS|i1o1"]}}},
            "locked_code_files": {}}
    p = _write_lock(repo, lock)
    _push(repo)
    Du = _unsealed(DS)
    monkeypatch.setitem(DA.SLATES, "final", DA.SLATES["tiny"])           # fast synthetic run (pinned slate untouched)
    L = AS.open_assessment(p, repo, check_code=False)
    try:
        with pytest.raises(SystemExit, match="not frozen in the lock"):     # the composed winner must be locked
            AS.outer_unit(Du, 0, "SRC|U", score["SRC|U"], units_dir=reg_store)
        L["seeds"]["0"]["composed_policies"] = ["pol__s1__U_JOINT_i8o64_l0.1"]
        with pytest.raises(SystemExit, match="not units of seed 0"):
            AS.outer_unit(Du, 0, "SRC|U", score["SRC|U"], units_dir=reg_store)
        L["seeds"]["0"]["composed_policies"] = {"U": freeze + ["U|JOINT|i8o64|l0.2"]}
        with pytest.raises(SystemExit, match="outside the registered composition bank"):
            AS.outer_unit(Du, 0, "SRC|U", score["SRC|U"], units_dir=reg_store)
        L["seeds"]["0"]["composed_policies"] = [A.unit_of(0, c) for c in
                                                dict.fromkeys(freeze + ["U|DIRECT-TASK|i8o64", "U|CLASS|i1o1"])]
        na = len(Du["idx"]["OSF_DEVELOPMENT_ASSESSMENT"])
        for k, label, spec in AS.jobs_from_lock(L, (0,)):
            r = AS.outer_unit(Du, k, label, spec, units_dir=reg_store)
            rr, z = AS.load_outer(k, label, reg_store)
            for key in ("assess_row_id", "assess_unit", "sex", "y_income", "y_occ", "hard1", "hard2", "ll1", "br2",
                        "u_hard1", "u_ll2", "const1"):
                assert z[key].shape == (na,), key
            assert z["prob1"].shape == (na, 2) and z["prob2"].shape == (na, 6) and z["const_class"].shape == (2,)
            for w in DA.PRIMARY_VIEWS:
                assert z[f"P_auc_{w}"].shape == (3, na, 2) and z[f"P_ce_{w}"].shape == (3, na, 2)
                assert np.allclose(z[f"P_auc_{w}"].sum(2), 1.0)
            assert np.array_equal(z["y_income"], Du["y"]["income"][Du["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]])
            assert set(r["utility"]) == {"0", "1"} and r["utility"]["1"]["u_acc"] is not None
            unit = {"policy": "pol__s0__U_JOINT_i8o64_l0.1", "source": "tea__s0__U", "reference": "ref__s0__F"}[
                spec["kind"]]
            if spec["kind"] == "source":
                assert r["composed"]["class_only_units"] == ["pol__s0__U_CLASS_i1o1"]
                assert "P_auc_decisions_pair" in z
                np.testing.assert_array_equal(z["prob2"], z["u_prob2"])
                assert r["families"]["interface"]["scored"]["v2"]["auc"]["candidate"].startswith("composed[")
            if spec["kind"] == "policy":
                assert r["decision_preservation"]["ok"]
            # restore hook: refit from the stored units on a SEALED D reproduces the saved predictions bitwise
            for w in ("pair", "v2"):
                P = AS.refit_selected_attacker(reg_store, DS, unit, r["unit"], view=w)
                assert np.array_equal(P, z[f"P_auc_{w}"]), (label, w)
            assert np.array_equal(AS.refit_selected_attacker(reg_store, DS, unit, r["unit"], attacker_seed=1),
                                  z["P_auc_pair"][1])
            assert not any(math.isnan(x) for x in rr["families"][rr["primary_family"]]["scored"]["pair"]["auc"][
                "auc_per_seed"])
    finally:
        AS.close_assessment()


def test_lazy_code_views_are_bit_identical_in_the_final_audit(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.3)
    V, L = A.policy_views(z, DS), A.lazy_policy_views(z, DS)
    for w in DA.PRIMARY_VIEWS:
        assert np.array_equal(V["X"][w], L["X"][w])
    Du = _unsealed(DS)
    a = Du["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    Vi = BL.source_view_sets(TS, DS, families=("interface",))["interface"]
    r1, P1 = DA.final_audit(Vi, Du, a, composed=[("c", V)], slate="tiny")
    r2, P2 = DA.final_audit(Vi, Du, a, composed=[("c", L)], slate="tiny")
    assert r1["selected"] == r2["selected"]
    for k in P1:
        assert np.array_equal(P1[k], P2[k])


def test_utility_contract_is_qpcs():
    """The inner utility / gate used by cbp.audit is qpc.utility unchanged (same rows, same functions)."""
    D = UT.synthetic_task_D(seed=0, n_fit=2000, n_head=10, n_audit=500, n_sel=500, n_assess=500)
    t = UT.synthetic_teacher(D, seed=0)
    u = UT.release_inner_utility({1: t["p1"], 2: t["p2"]}, {1: t["d1"], 2: t["d2"]}, D)
    g = UT.gate_record(u, u, {1: True, 2: True})
    assert g["eligible"] and A.cbp_headroom(g)["pass"]
    assert A.cbp_headroom(g)["income"]["ll_excess"] == 0.0
