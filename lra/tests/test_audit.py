"""[lra port of lcr/tests/test_audit.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
Synthetic tests of lra.audit / lra.baselines / lra.assess (role D; no real data, no real labels).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.sema --label D:test-audit -- env OMP_NUM_THREADS=1 \\
        OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q \\
        lra/tests/test_audit.py

Adapted from cbp/tests/test_audit.py (views, readers, unit contract, closure, controls, assessment) plus the lra checks:
the binding to lra.run (aud__ inner namespace, schema lra-inner-v1), the registered 83-code composition bank and its
closure (missing / stray release or inner units refused), D1 fixed-map releases audited with full token identities
(and refused if their tokens differ from the D0 map), the token-only / probability-only diagnostic families (a
probability-only family differs from the complete family when probabilities coincide across distinct tokens), the
mathematically identical token-family reuse, the records-only cbp parity receipt, controls on D0, D1 and new codes,
the assess gate, and the prompt sec. 14 defects in role D's scope, each shown to be CAUGHT: omitted token identities,
clean-output bypass, reversed AUC orientation (and reversed best/worst reader), pair misalignment and a missing source
composition winner. A small slate is registered for speed under the name "tiny" (the pinned FINAL slate itself is
untouched and is timed separately by `lra.audit timing`).
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import types

import numpy as np
import pytest

from lra import assess as AS
from lra import audit as A
from lra import baselines as BL
from lra import run as R
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


def _renumber(z, seed):
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
    A._save_unit(units, name, files, record)                 # lra.run.save writers: .npz arrays, .json finite JSON


def _save_inner(units, rec, name=None):
    rec = dict(rec)
    files = rec.pop("_files")
    _save(units, name or rec["inner_unit"], files, rec)


def _pub(rec):
    return json.loads(json.dumps({k: v for k, v in rec.items() if k != "_files"}, allow_nan=False))


def _arr(rec):
    return rec["_files"]["inner_preds.npz"]


# ====================================================================== binding to lra.run and the registered bank
def test_bound_to_lra_run_store_and_aud_namespace():
    assert A._R() is R and BL.units_dir() == R.UNITS
    assert "lra_v1" in str(R.UNITS) and "cbp_v1" not in str(BL.units_dir())
    for cid, unit in (("U|JOINT|i8o64|l0.025", "pol__s2__U_JOINT_i8o64_l0.025"),
                      ("U|JOINT|i8o64|l0.025|D1", "dec__s2__U_JOINT_i8o64_l0.025_D1"),
                      ("U|K-JOINT-PAIR|i8o64|D1", "new__s2__U_K-JOINT-PAIR_i8o64_D1"),
                      ("U|W-SEQ-12|i8o64|l0.1|D1", "new__s2__U_W-SEQ-12_i8o64_l0.1_D1")):
        assert A.parse_cid(cid) == R.parse_id(cid)
        assert A.unit_of(2, cid) == R.unit_for(2, cid) == unit
        assert A.inner_of(2, cid) == R.inner_name(2, cid) == f"aud__{unit}"
    chk = A.registration_check()
    assert chk["ok"], chk
    U = A.expected_policies("U")
    assert len(U) == 83 and U == R.code_ids() and A.COMPOSED_EXTRA == ()
    assert U[:27] == R.d0_ids() and U[27:53] == R.d1_fixed_ids() and U[53:] == R.new_fit_ids()
    assert A.expected_policies("RAW-J_b0.3") == []
    assert A.d0_of("U|JOINT|i8o64|l0.1|D1") == "U|JOINT|i8o64|l0.1" and A.d0_of("U|C-TASK|i8o64|D1") is None
    assert [c for c in R.scored_ids() if R.parse_id(c)["kind"] == "policy"] == U


def test_registration_drift_is_refused(monkeypatch):
    fake = types.SimpleNamespace(**{k: getattr(R, k) for k in ("parse_id", "unit_for", "inner_name")})
    fake.code_ids = lambda: R.code_ids()[:-1]                              # one constrained arm dropped
    fake.scored_ids = lambda: fake.code_ids() + ["SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"]
    assert not A.registration_check(fake)["ok"]
    monkeypatch.setattr(A, "_R", lambda: fake)
    with pytest.raises(SystemExit, match="disagrees with the registered composition bank"):
        A.expected_policies("U")
    fake.code_ids = R.code_ids
    fake.scored_ids = lambda: R.scored_ids() + ["U|LOCAL|i8o64|l1"]        # an unregistered code made a candidate
    assert "scored_ids (policy part) != the 83 registered codes" in A.registration_check(fake)["mismatches"]


# ====================================================================== views and readers (source slate unchanged)
def test_release_key_contract_and_three_code_families(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4)
    A.check_release_keys(z)
    with pytest.raises(ValueError, match="extra \\['p1'\\]"):
        A.policy_views({**z, "p1": TS["p1"]}, DS)              # a continuous probability may never ride along
    with pytest.raises(ValueError, match="extra \\['fine2'\\]"):
        A.token_views({**z, "fine2": z["tok2"]}, DS)
    with pytest.raises(ValueError, match="missing \\['q2'\\]"):
        A.prob_views({k: v for k, v in z.items() if k != "q2"}, DS)
    S = A.code_view_sets(z, DS)
    assert list(S) == ["code", "token", "prob"]
    a2 = int(z["alpha2"])
    assert S["code"]["X"]["v2"].shape[1] == a2 + 6 + 6           # one-hot over the FULL alphabet, q, one-hot decision
    assert set(np.unique(S["code"]["X"]["v2"][:, :a2])) == {0.0, 1.0}   # categorical, not an ordinal id column
    assert np.array_equal(S["token"]["X"]["v2"], S["code"]["X"]["v2"][:, :a2])
    assert np.array_equal(S["prob"]["X"]["v2"], z["q2"]) and S["prob"]["tokens"] is not None
    assert np.array_equal(S["token"]["tokens"]["v2"], z["tok2"])


def test_renumbering_leaves_every_prediction_unchanged(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.3)
    S0, S1 = A.code_view_sets(z, DS), A.code_view_sets(_renumber(z, 3), DS)
    for fam in A.CODE_FAMILIES:
        r0, a0 = A.family_auditor("policy", fam)(S0[fam], DS, slate="tiny")
        r1, a1 = A.family_auditor("policy", fam)(S1[fam], DS, slate="tiny")
        assert np.array_equal(r0["inner_predictions"]["P"], r1["inner_predictions"]["P"]), fam
        assert r0["auc"] == r1["auc"] and r0["selected"] == r1["selected"]
        for k in a0:
            assert np.array_equal(a0[k], a1[k])
        assert DA.view_fingerprint(S0[fam]) == DA.view_fingerprint(S1[fam])


def test_probability_only_family_differs_when_probabilities_coincide(DS, TS):
    zc = _sexbit_release(DS, TS, i=2)                              # two IDs share one decoded vector, different info
    assert DA.token_decoder_check(zc["tok2"], zc["q2"], zc["hard2"])["ok"]
    S = A.code_view_sets(zc, DS)
    out = {fam: A.family_auditor("policy", fam)(S[fam], DS, slate="tiny")[0] for fam in A.CODE_FAMILIES}
    assert out["code"]["auc"]["v2"] > 0.8 and out["token"]["auc"]["v2"] > 0.8     # token identity carries the bit
    assert out["prob"]["auc"]["v2"] < 0.6                                         # q-value cells cannot see it
    assert out["code"]["auc"]["pair"] > 0.8                       # pair identities are never collapsed by equal q
    assert len(np.unique(S["prob"]["tokens"]["v2"])) < len(np.unique(S["token"]["tokens"]["v2"]))


def test_token_only_family_uses_the_exact_cell_readers_only(DS, TS):
    """L4b: the token-only diagnostic = CC / CCpair on the exact token identities (+ both ignore-recipient banks), with
    the same dual selection; the complete and probability-only families keep the full slate."""
    z = A.synthetic_d1(A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.4))
    S = A.code_view_sets(z, DS)
    assert A.family_auditor("policy", "token") is A.inner_family_cells
    assert A.family_auditor("policy", "code") is A.inner_family is A.family_auditor("policy", "prob")
    assert A.family_auditor("source", "token") is A.inner_family
    rt, at = A.inner_family_cells(S["token"], DS, slate="tiny")
    rc, _ = A.inner_family(S["code"], DS, slate="tiny")
    assert rt["slate_members"] == [] and rt["readers"] == "cells_only" and rt["refits"] == 0
    assert rt["inner_predictions"]["keys"] == [f"{w}:CC_alpha{a}" for w in ("v1", "v2") for a in DA.CC_ALPHAS] + \
        [f"pair:CCpair_alpha{a}" for a in DA.CC_ALPHAS]
    assert {row["candidate"] for row in rt["tables"]["pair"]} == {"coalition", "ignore_recipient_2",
                                                                  "ignore_recipient_1"}
    Pc = dict(zip(rc["inner_predictions"]["keys"], rc["inner_predictions"]["P"]))
    for key, p in zip(rt["inner_predictions"]["keys"], rt["inner_predictions"]["P"]):
        assert np.array_equal(p, Pc[key]), key            # the same exact readers as inside the complete family
    for w in DA.PRIMARY_VIEWS:
        assert rt["auc"][w] == rt["auc_seed0"][w] and np.array_equal(at[f"auc_{w}"][0], at[f"auc_{w}"][2])
        assert rt["auc_seed0"][w] <= rc["auc_seed0"][w] + 1e-12       # a sub-bank of the complete family's bank
    Du = _unsealed(DS)
    a = Du["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    rf, Pf = A.final_audit_cells(S["token"], Du, a)
    assert set(Pf) == {f"{c}_{w}" for c in ("auc", "ce") for w in DA.PRIMARY_VIEWS}
    assert all(P.shape == (3, len(a), 2) and np.allclose(P.sum(2), 1.0) for P in Pf.values())
    assert rf["scored"]["pair"]["auc"]["label"] == rt["selected"]["pair"]
    with pytest.raises(ValueError, match="REFUSED: scored rows overlap"):
        A.final_audit_cells(S["token"], Du, DS["idx"]["INNER_SELECTION"][:5])


def test_unseen_local_token_uses_fit_prior_and_unseen_pair_rule_is_inner_selected():
    D = {"idx": {"AUDIT_FIT": np.arange(8), "INNER_SELECTION": np.arange(8, 16)}, "row_id": np.arange(16),
         "sex": np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1])}
    t1 = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 7, 1, 0, 0])   # token 7 unseen in AUDIT_FIT
    t2 = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 5, 5])   # tuples (0,5) etc unseen
    fit, sel = D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"]
    prior = D["sex"][fit].mean()
    out, _ = DA.cc_local(t1, D["sex"], fit, sel)
    assert out["CC_alpha1.0"][12 - 8] == pytest.approx(prior)          # unseen local -> AUDIT_FIT SEX prior
    _, ip = DA.cc_pair(t1, t2, D["sex"], fit, sel, np.arange(8))
    for a, rule in ip["fallback_rule"].items():
        ces = ip["fallback_ce_on_unseen_inner_rows"][a]
        assert rule == min(DA.FALLBACK_RULES, key=lambda k: (ces[k], DA.FALLBACK_RULES.index(k)))
    Dc = {**D, "idx": {**D["idx"], "OSF_DEFENSE_FIT": np.arange(0), "OSF_DEVELOPMENT_ASSESSMENT": np.arange(0)}}
    cov = A.coverage_receipt(t1, t2, Dc, 8, 6)
    assert cov["v1"]["fallback_use"]["INNER_SELECTION"]["fallback_rows"] == 1
    assert cov["v1"]["fallback_use"]["AUDIT_FIT"]["fallback_rows"] == 0
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
    S = A.code_view_sets(_reversed_relation_release(DS), DS)
    for fam in A.CODE_FAMILIES:
        r, _ = A.family_auditor("policy", fam)(S[fam], DS, slate="tiny")
        cc = {row["attacker"]: row["inner_auc"] for row in r["tables"]["v1"] if row["attacker"].startswith("CC")}
        assert len(cc) == 3 and max(cc.values()) < 0.05, fam      # deterministic readers: far below chance, as is
        assert r["auc"]["v1"] < 0.05 and r["auc_seed0"]["v1"] < 0.05, fam


def test_auc_and_ce_selection_are_separate():
    y = np.array([0, 1] * 50)
    rank_perfect_uncal = np.where(y == 1, 0.501, 0.499)          # AUC 1, CE ~ log 2
    calibrated = np.where(y == 1, 0.9, 0.1)
    calibrated[:10] = 0.95
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


def test_mi_diagnostic_exact_and_identical_for_d0_and_d1(DS, TS):
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
    z = A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.3)
    z1 = A.synthetic_d1(z)
    assert not np.array_equal(z["q2"], z1["q2"])
    assert A.mi_diagnostic(z["tok1"], z["tok2"], DS, perms=5) == A.mi_diagnostic(z1["tok1"], z1["tok2"], DS, perms=5)


def test_reference_views_cells_and_value_tokens(DS, TS):
    zr = A.synthetic_reference_cells(DS, TS, ncell=(5, 7))
    sets, out, _ = BL.reference_view_sets("F", 0, DS, arrays=zr, with_outputs=True)
    assert set(sets) == {"interface", "scores", "probs", "decisions", "cells"}
    assert sets["cells"]["tokens"]["v2"] is not None and sets["interface"]["tokens"] is not None
    se = BL.reference_view_sets("E", 0, DS, arrays=zr)
    assert "complete" in se and se["interface"]["tokens"] is None      # E's views are continuous


# ====================================================================== explicit store: D0 / D1 / new units
POLS = ["U|DIRECT-TASK|i2o4", "U|JOINT|i2o4|l0.1", "U|CLASS|i1o1", "U|JOINT|i2o4|l0.1|D1", "U|K-JOINT-PAIR|i2o4|D1"]


@pytest.fixture()
def store(tmp_path, DS, TS):
    units = tmp_path / "units"
    units.mkdir()
    _save(units, "tea__s0__U", {"teacher.npz": TS}, {"t": "U"})
    _save(units, "tea__s0__RAW-J_b0.3", {"teacher.npz": A.synthetic_teacher(DS, seed=9)}, {"t": "RAW-J"})
    leaky = A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.6)
    rel = {"U|DIRECT-TASK|i2o4": A.synthetic_release(DS, TS, 2, 4), "U|JOINT|i2o4|l0.1": leaky,
           "U|CLASS|i1o1": A.synthetic_release(DS, TS, 1, 1), "U|JOINT|i2o4|l0.1|D1": A.synthetic_d1(leaky),
           "U|K-JOINT-PAIR|i2o4|D1": A.synthetic_d1(A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.3, seed=4))}
    for cid, z in rel.items():
        _save(units, A.unit_of(0, cid), {"release.npz": z}, {})
    _save(units, "ref__s0__F", {"reference.npz": A.synthetic_reference_cells(DS, TS, ncell=(5, 7))}, {})
    return units


def _release(units, cid, k=0):
    z = np.load(units / A.unit_of(k, cid) / "release.npz")
    return {x: z[x] for x in z.files}


def test_d1_release_audited_with_full_token_identities_and_token_reuse(store, DS):
    d0, d1 = "U|JOINT|i2o4|l0.1", "U|JOINT|i2o4|l0.1|D1"
    r0 = A.inner_unit("policy", 0, d0, DS, units_dir=store, slate="tiny")
    _save_inner(store, r0)
    r1 = A.inner_unit("policy", 0, d1, DS, units_dir=store, slate="tiny")
    p1 = _pub(r1)
    assert p1["schema"] == "lra-inner-v1" and p1["arm"] == "d1_fixed" and p1["decoder"] == "D1"
    assert p1["inner_unit"] == "aud__dec__s0__U_JOINT_i2o4_l0.1_D1" and p1["privacy_trained"]
    assert p1["d0_token_parity"]["ok"] and p1["d0_token_parity"]["d0_unit"] == "pol__s0__U_JOINT_i2o4_l0.1"
    assert not any(p1["d0_token_parity"]["q_identical_to_d0"].values())          # D1 differs from D0 only in q
    # the PRIMARY family is the complete interface with full token identities (one-hot + CC readers on tokens)
    rec = p1["recovery"]
    assert rec["family"] == "code" and rec["finite"]
    assert {f"v2:CC_alpha{a}" for a in DA.CC_ALPHAS} <= set(rec["inner_prediction_keys"])
    z1 = _release(store, d1)
    assert p1["view_fingerprints"]["code"] == DA.view_fingerprint(A.policy_views(z1, DS))
    assert p1["view_fingerprints"]["token"] == _pub(r0)["view_fingerprints"]["token"]     # identical partition
    assert p1["view_fingerprints"]["code"] != _pub(r0)["view_fingerprints"]["code"]       # different q: distinct map
    assert set(p1["families"]) == {"token", "prob"}
    # token-family reuse is mathematically identical to recomputation
    assert p1["reused_families"]["token"]["from_unit"] == "aud__pol__s0__U_JOINT_i2o4_l0.1"
    assert p1["token_family_path"].startswith("reused from aud__pol__s0__U_JOINT_i2o4_l0.1")
    assert p1["family_paths"]["code"].startswith("computed") and p1["family_paths"]["prob"].startswith("computed")
    assert r1["_files"][A.SIDECAR]["view_fingerprints"] == p1["view_fingerprints"]
    fresh = A.inner_unit("policy", 0, d1, DS, units_dir=store, slate="tiny", reuse=False)
    assert _pub(fresh)["reused_families"] == {}
    for key, v in _arr(fresh).items():
        assert np.array_equal(v, _arr(r1)[key]), key
    assert _pub(fresh)["families"]["token"]["auc"] == p1["families"]["token"]["auc"]
    assert A.validate_inner(p1, _arr(r1), DS, release=z1)["ok"]
    # token counts and MI are exactly unchanged between D0 and D1 of the same map
    assert p1["token_states"] == _pub(r0)["token_states"] and p1["mi_diagnostic"] == _pub(r0)["mi_diagnostic"]


def test_d1_release_with_changed_tokens_is_refused(store, DS, TS):
    z = _release(store, "U|JOINT|i2o4|l0.1|D1")
    z["tok2"] = _renumber(z, 1)["tok2"]                                   # a mapping change, not calibration-only
    _save(store, A.unit_of(0, "U|JOINT|i2o4|l0.1|D1"), {"release.npz": z}, {})
    with pytest.raises(SystemExit, match="does not keep its D0 map's tokens exactly"):
        A.inner_unit("policy", 0, "U|JOINT|i2o4|l0.1|D1", DS, units_dir=store, slate="tiny")


def test_exact_alias_cache_reuses_every_family_bitwise_and_records_the_path(store, DS):
    z = _release(store, "U|K-JOINT-PAIR|i2o4|D1")
    _save(store, A.unit_of(0, "U|K-LOCAL|i2o4|D1"), {"release.npz": z}, {})        # e.g. a constrained arm that
    first = A.inner_unit("policy", 0, "U|K-JOINT-PAIR|i2o4|D1", DS, units_dir=store, slate="tiny")   # stopped at a
    assert all(v.startswith("computed (no completed") for v in first["family_paths"].values())       # witness
    _save_inner(store, first)
    hit = A.inner_unit("policy", 0, "U|K-LOCAL|i2o4|D1", DS, units_dir=store, slate="tiny")
    assert sorted(hit["reused_families"]) == ["code", "prob", "token"]
    assert all(v.startswith("reused from aud__new__s0__U_K-JOINT-PAIR_i2o4_D1")
               for v in hit["family_paths"].values())
    fresh = A.inner_unit("policy", 0, "U|K-LOCAL|i2o4|D1", DS, units_dir=store, slate="tiny", reuse=False)
    assert fresh["reused_families"] == {} and all(v == "computed (reuse disabled)" for v in
                                                  fresh["family_paths"].values())
    assert sorted(_arr(hit)) == sorted(_arr(fresh))
    for key in _arr(fresh):
        assert np.array_equal(_arr(hit)[key], _arr(fresh)[key]), key                 # bitwise identical
    ph, pf = _pub(hit), _pub(fresh)
    for key in ("auc", "ce", "auc_seed0", "ce_seed0", "auc_per_seed", "ce_per_seed", "selected", "ce_selected",
                "tables", "inner_predictions_sha256"):
        assert ph["recovery"][key] == pf["recovery"][key], key
        for fam in ("token", "prob"):
            assert ph["families"][fam][key] == pf["families"][fam][key], (fam, key)
    assert ph["utility"] == pf["utility"] and ph["view_fingerprints"] == pf["view_fingerprints"]
    assert ph["cid"] == "U|K-LOCAL|i2o4|D1" and ph["recovery"]["meta"]["config"] == "U|K-LOCAL|i2o4|D1"
    assert A.validate_inner(ph, _arr(hit), DS, release=z)["ok"]


def test_alias_cache_never_trusts_the_sidecar_alone(store, DS):
    z = _release(store, "U|K-JOINT-PAIR|i2o4|D1")
    _save(store, A.unit_of(0, "U|K-LOCAL|i2o4|D1"), {"release.npz": z}, {})
    first = A.inner_unit("policy", 0, "U|K-JOINT-PAIR|i2o4|D1", DS, units_dir=store, slate="tiny")
    rec = dict(first)
    files = rec.pop("_files")
    rec["view_fingerprints"] = {f: "0" * 64 for f in rec["view_fingerprints"]}       # record disagrees with sidecar
    _save(store, rec["inner_unit"], files, rec)
    r = A.inner_unit("policy", 0, "U|K-LOCAL|i2o4|D1", DS, units_dir=store, slate="tiny")
    assert r["reused_families"] == {}
    files[A.SIDECAR] = {**files[A.SIDECAR], "slate": "final"}                       # a different slate
    _save(store, first["inner_unit"], files, _pub(first))
    assert A.inner_unit("policy", 0, "U|K-LOCAL|i2o4|D1", DS, units_dir=store, slate="tiny")["reused_families"] == {}


def test_inner_unit_contract_finite_records_and_composed_source_bank(store, DS):
    recs = {}
    for c in POLS:
        r = A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny")
        recs[c] = _pub(r)
        assert A.validate_inner(recs[c], _arr(r), DS, release=_release(store, c))["ok"], c
        _save_inner(store, r)
    p = recs["U|JOINT|i2o4|l0.1"]
    for key in ("recovery", "utility", "utility_gate", "preserved", "token_states", "coverage_receipt",
                "mi_diagnostic", "decision_preservation", "families", "view_fingerprints", "arm", "decoder"):
        assert key in p
    assert "cbp_headroom" not in p                                # the cbp buffer is not lra selection
    assert set(p["recovery"]["auc"]) == {"v1", "v2", "pair"} and "ce" in p["recovery"]
    for t in ("income", "occupation"):
        assert {"acc", "logloss", "brier", "const_acc"} <= set(p["utility"][t])
    assert p["preserved"] == {"1": True, "2": True}
    assert p["token_states"] == 4 + 21                           # 2 x 2 income + 5 x 4 occupation + 1 reserved
    assert p["recovery"]["auc"]["v2"] > recs["U|DIRECT-TASK|i2o4"]["recovery"]["auc"]["v2"]
    assert recs["U|K-JOINT-PAIR|i2o4|D1"]["arm"] == "constrained" and recs["U|K-JOINT-PAIR|i2o4|D1"][
        "reused_families"] == {}
    # the source composes with every code (D0, D1 and new) BEFORE selection; a leaky code's reader wins
    src = A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)
    sp = _pub(src)
    c = sp["composed"]
    assert c["winner"]["v2"] in ("U|JOINT|i2o4|l0.1", "U|JOINT|i2o4|l0.1|D1")
    assert sp["recovery"]["own"]["auc"]["v2"] < sp["recovery"]["auc"]["v2"]
    assert c["winner"]["v2"] in c["freeze"] and c["closure"]["ok"] and c["policies"] == POLS
    assert c["policy_list_source"].startswith("EXPLICIT")
    assert sp["token_states"] is None and sp["preserved"] == {"1": True, "2": True}
    assert sp["families"]["decisions"]["composed"]["policies"] == ["U|CLASS|i1o1"]
    assert A.validate_inner(sp, _arr(src), DS, registered=False)["ok"]
    _save_inner(store, src)
    assert A.composed_freeze_list(0, "U", store) == c["freeze"]
    rj = A.inner_unit("source", 0, "SRC|RAW-J_b0.3", DS, units_dir=store, slate="tiny", policy_cids=[])
    assert rj["composed"]["freeze"] == [] and rj["recovery"]["auc"] == rj["recovery"]["own"]["auc"]
    ref = A.inner_unit("reference", 0, "REF|F", DS, units_dir=store, slate="tiny")
    assert ref["preserved"] == {"1": True, "2": True} and "cells" in ref["families"] and ref["token_states"] is None
    assert A.validate_inner(_pub(ref), _arr(ref), DS)["ok"]


def test_inner_job_names_and_stage_src_refuses_open_bank(store, DS, monkeypatch):
    name, files, rec = A.inner_job(0, "U|DIRECT-TASK|i2o4", DS, units_dir=store, slate="tiny")
    assert name == "aud__pol__s0__U_DIRECT-TASK_i2o4" and "_files" not in rec
    assert set(files) == {"inner_preds.npz", A.SIDECAR}
    assert rec["config"] == "U|DIRECT-TASK|i2o4" and rec["of"] == "pol__s0__U_DIRECT-TASK_i2o4"
    monkeypatch.setattr(R, "UNITS", store)                                  # lra.run.save writes the sidecar as JSON
    monkeypatch.setattr(R, "RUN", store.parent)
    R.save(name, files, rec)
    saved = json.loads((store / name / A.SIDECAR).read_text())
    assert saved["view_fingerprints"] == rec["view_fingerprints"] and A._lra_inner_ok(name, store)
    jobs = A.inner_jobs()
    assert len(jobs) == 3 * 83 + 9 and jobs[:81] == [(k, c) for c in R.d0_ids() for k in (0, 1, 2)]
    assert jobs[81:159] == [(k, c) for c in R.d1_fixed_ids() for k in (0, 1, 2)]
    assert A.source_jobs() == [(k, s) for k in (0, 1, 2) for s in ("SRC|U", "SRC|RAW-J_b0.3")]
    with pytest.raises(SystemExit, match="need the closed 83-code bank"):
        A.stage_inner_src(DS, None, units_dir=store, save=lambda *a: None, slate="tiny")


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
    for c in POLS[:4]:
        _save_inner(store, A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny"))
    with pytest.raises(SystemExit, match="not closed"):           # the K-JOINT-PAIR code's inner unit is missing
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)
    with pytest.raises(SystemExit, match=r"'unexpected_release_units': \['new__s0__U_K-JOINT-PAIR_i2o4_D1'\]"):
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS[:4])


def test_composed_bank_refuses_records_from_other_rows(store, DS):
    for c in POLS:
        r = A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny")
        if c == "U|CLASS|i1o1":
            r["recovery"]["sel_row_id_sha256"] = "0" * 64              # scored on different INNER rows
        _save_inner(store, r)
    with pytest.raises(SystemExit, match="differs from the source on sel_row_id_sha256"):
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)


def test_cbp_schema_records_never_pass_as_lra_records(store, DS):
    for c in POLS:
        r = A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny")
        if c == "U|DIRECT-TASK|i2o4":
            r["schema"] = "cbp-inner-v1"                               # an admitted cbp audit in the lra namespace
        _save_inner(store, r)
    with pytest.raises(SystemExit, match="schema 'cbp-inner-v1'"):
        A.load_inner(A.inner_of(0, "U|DIRECT-TASK|i2o4"), store)
    with pytest.raises(SystemExit, match=r"'missing_inner_units': \['U\|DIRECT-TASK\|i2o4'\]"):
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
    _save(store, "new__s0__U_C-TASK_i2o4_D1", {"release.npz": z}, {})
    r = A.inner_unit("policy", 0, "U|C-TASK|i2o4|D1", DS, units_dir=store, slate="tiny")
    assert r["preserved"] == {"1": True, "2": False} and not r["utility_gate"]["eligible"]


def test_cbp_parity_receipt_records_only(store, DS):
    r = A.inner_unit("policy", 0, "U|JOINT|i2o4|l0.1", DS, units_dir=store, slate="tiny")
    _save_inner(store, r)
    cb = _pub(r)
    cb["schema"], cb["families"] = "cbp-inner-v1", {}                  # a cbp audit: primary family only
    _save(store, "inner__pol__s0__U_JOINT_i2o4_l0.1", {"inner_preds.npz": _arr(r)}, cb)
    p = A.cbp_parity(0, "U|JOINT|i2o4|l0.1", store)
    assert p["ok"], p["mismatches"]
    assert p["families_compared"] == ["code"] and p["lra_only_families"] == ["prob", "token"]
    cb["recovery"]["auc_per_seed"]["pair"][1] += 1e-9
    _save(store, "inner__pol__s0__U_JOINT_i2o4_l0.1", {"inner_preds.npz": _arr(r)}, cb)
    assert "code.auc_per_seed differs" in A.cbp_parity(0, "U|JOINT|i2o4|l0.1", store)["mismatches"]
    with pytest.raises(ValueError, match="not an admitted D0 code"):
        A.cbp_parity(0, "U|JOINT|i2o4|l0.1|D1", store)


# ====================================================================== the registered 83-code composition bank
REG_KIND = {"U|JOINT|i8o64|l0.1": "leaky", "U|JOINT|i8o64|l0.1|D1": "leaky_d1", "U|CLASS|i1o1": "cls"}


def _reg_kind(cid):
    if cid in REG_KIND:
        return REG_KIND[cid]
    return {"d0": "plain", "d1_fixed": "plain_d1"}.get(A.parse_cid(cid)["arm"], "new")


@pytest.fixture(scope="module")
def reg_template(tmp_path_factory, DS, TS):
    """lra-named store with the REGISTERED 83-code bank of U at seed 0. Six distinct releases are audited (a plain D0
    code, a SEX-tilted D0 JOINT l0.1, their D1 versions, the class-only code, a new fit); every other registered name
    reuses the release and inner record of its kind (composition reads records only; D1 names keep their D0 tokens)."""
    DA.SLATES.setdefault("tiny", lambda: [("LR_C1", lambda s: JA._lr(1.0, s)), ("MLP_16", lambda s: JA._mlp((16,), s)),
                                          ("DA_LR", SA.da_lr)])
    units = tmp_path_factory.mktemp("reg") / "units"
    units.mkdir()
    _save(units, "tea__s0__U", {"teacher.npz": TS}, {"t": "U"})
    _save(units, "tea__s0__RAW-J_b0.3", {"teacher.npz": A.synthetic_teacher(DS, seed=9)}, {"t": "RAW-J"})
    _save(units, "ref__s0__F", {"reference.npz": A.synthetic_reference_cells(DS, TS, ncell=(5, 7))}, {})
    plain, leaky = A.synthetic_release(DS, TS, 2, 4), A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.6)
    rel = {"plain": plain, "leaky": leaky, "plain_d1": A.synthetic_d1(plain), "leaky_d1": A.synthetic_d1(leaky),
           "cls": A.synthetic_release(DS, TS, 1, 1), "new": A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.1, seed=5)}
    for cid in A.registered_code_bank():
        _save(units, A.unit_of(0, cid), {"release.npz": rel[_reg_kind(cid)]}, {})
    recs = {}
    for cid in A.registered_code_bank():               # D0 names first: D1 reuse finds the D0 token family
        kd = _reg_kind(cid)
        if kd not in recs:
            r = A.inner_unit("policy", 0, cid, DS, units_dir=units, slate="tiny")
            recs[kd] = (_pub(r), _arr(r), r["_files"][A.SIDECAR])
        pub, arr, side = recs[kd]
        _save(units, A.inner_of(0, cid), {"inner_preds.npz": arr, A.SIDECAR: {**side, "cid": cid}},
              {**pub, "cid": cid, "unit_of": A.unit_of(0, cid), "inner_unit": A.inner_of(0, cid)})
    return units


@pytest.fixture()
def reg_store(reg_template, tmp_path):
    dst = tmp_path / "units"
    shutil.copytree(reg_template, dst)
    return dst


def test_source_composes_over_the_registered_83_bank_in_order(reg_store, DS):
    src = A.inner_unit("source", 0, "SRC|U", DS, units_dir=reg_store, slate="tiny")
    c = src["composed"]
    assert c["policies"] == A.registered_code_bank() == R.code_ids() and len(c["policies"]) == 83
    assert c["closure"]["ok"] and c["closure"]["expected"] == 83 and c["closure"]["release_units_on_disk"] == 83
    assert c["closure"]["inner_units_on_disk"] == 83 and c["composition_only"] == []
    assert c["policy_list_source"].startswith("registered")
    assert c["winner"]["v2"] in ("U|JOINT|i8o64|l0.1", "U|JOINT|i8o64|l0.1|D1")     # the leaky code's reader wins
    assert all(w in ("source", "U|DIRECT-TASK|i8o64", "U|JOINT|i8o64|l0.1", "U|CLASS|i1o1",
                     "U|DIRECT-TASK|i8o64|D1", "U|JOINT|i8o64|l0.1|D1", "U|C-TASK|i8o64|D1")
               for f in c["per_family"].values() for w in f["winner"].values())     # first in registered order
    assert A.validate_inner(_pub(src), _arr(src), DS, registered=True)["ok"]


@pytest.mark.parametrize("victim,kind", [("U|SEQ-21|i8o64|l0.06", "release"), ("U|FINE-TASK|i8o64|D1", "release"),
                                         ("U|K-SEQ-12|i8o64|D1", "release"), ("U|W-JOINT|i8o64|l0.04|D1", "inner")])
def test_registered_closure_refuses_a_missing_unit(reg_store, DS, victim, kind):
    shutil.rmtree(reg_store / (A.unit_of(0, victim) if kind == "release" else A.inner_of(0, victim)))
    key = "missing_release_units" if kind == "release" else "missing_inner_units"
    with pytest.raises(SystemExit, match=rf"'{key}': \['{victim.replace('|', '[|]')}'\]"):
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=reg_store, slate="tiny")


@pytest.mark.parametrize("stray", ["pol__s0__U_JOINT_i8o64_l0.2", "new__s0__U_K-JOINT-TRIPLE_i8o64_D1",
                                   "dec__s0__U_CLASS_i1o1_D1"])
def test_registered_closure_refuses_a_planted_or_stray_release(reg_store, DS, TS, stray):
    z = DA.split_tokens(A.synthetic_release(DS, TS, 2, 4), 2, np.asarray(DS["sex"]).clip(0), collide=True)
    _save(reg_store, stray, {"release.npz": z}, {})
    with pytest.raises(SystemExit, match=rf"'unexpected_release_units': \['{stray}'\]"):
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=reg_store, slate="tiny")


def test_registered_closure_refuses_a_stray_inner_unit(reg_store, DS):
    shutil.copytree(reg_store / A.inner_of(0, "U|C-TASK|i8o64|D1"), reg_store / "aud__new__s0__U_C-TASK_i8o8_D1")
    with pytest.raises(SystemExit, match=r"'unexpected_inner_units': \['aud__new__s0__U_C-TASK_i8o8_D1'\]"):
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=reg_store, slate="tiny")


def test_validate_all_over_the_saved_registered_bank(reg_store, DS, TS):
    for cid in ("SRC|U", "SRC|RAW-J_b0.3", "REF|F"):
        _save_inner(reg_store, A.inner_unit(A.parse_cid(cid)["kind"], 0, cid, DS, units_dir=reg_store, slate="tiny"))
    v = A.validate_all(DS, reg_store, seeds=(0,))
    assert v["checked"] == 83 + 3 and not v["defects"], v["defects"][:2]
    assert sorted(v["missing"]) == ["aud__ref__s0__E", "aud__ref__s0__F0"] and not v["ok"]
    _save(reg_store, A.unit_of(0, "U|W-LOCAL|i8o64|l0.08|D1"),
          {"release.npz": A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.2)}, {})
    v2 = A.validate_all(DS, reg_store, seeds=(0,))
    assert len(v2["defects"]) == 1 and "U|W-LOCAL|i8o64|l0.08|D1" in v2["defects"][0]


# ====================================================================== deliberate defects (prompt sec. 14), CAUGHT
def _policy_record(store, DS, cid="U|JOINT|i2o4|l0.1"):
    r = A.inner_unit("policy", 0, cid, DS, units_dir=store, slate="tiny")
    return _pub(r), _arr(r), _release(store, cid)


def test_defect_reversed_best_worst_reader_is_caught(store, DS, monkeypatch):
    rec, arr, z = _policy_record(store, DS)
    assert A.validate_inner(rec, arr, DS, release=z)["ok"]
    orig = DA._pick
    monkeypatch.setattr(DA, "_pick", lambda rows, key, maximize: orig(rows, key, not maximize))
    bad, barr, _ = _policy_record(store, DS)
    monkeypatch.setattr(DA, "_pick", orig)
    with pytest.raises(A.AuditDefect, match="best/worst reversal"):
        A.validate_inner(bad, barr, DS, release=z)


def test_defect_reversed_auc_orientation_is_caught(store, DS, monkeypatch):
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


@pytest.mark.parametrize("cid", ["U|JOINT|i2o4|l0.1", "U|K-JOINT-PAIR|i2o4|D1"])
def test_defect_omitted_token_identities_is_caught(store, DS, monkeypatch, cid):
    def q_only(z, D, meta=None):                                         # the defect: tokens dropped
        A.check_release_keys(z)
        zz = DA.align(z, D)
        X = {f"v{i}": np.hstack([zz[f"q{i}"], DA.onehot(zz[f"hard{i}"], K)]) for i, K in ((1, 2), (2, 6))}
        X["pair"] = np.hstack([X["v1"], X["v2"]])
        return {"family": "code", "X": X, "tokens": None, "meta": {**(meta or {})}}
    z = _release(store, cid)
    good = A.policy_views
    monkeypatch.setattr(A, "policy_views", q_only)
    r = A.inner_unit("policy", 0, cid, DS, units_dir=store, slate="tiny")
    monkeypatch.setattr(A, "policy_views", good)
    with pytest.raises(A.AuditDefect, match="without token-identity readers"):
        A.validate_inner(_pub(r), _arr(r), DS, release=z)
    with pytest.raises(A.AuditDefect, match="registered code views"):
        A.validate_inner(_pub(r), _arr(r), DS, release=z)


def test_defect_pair_misalignment_is_caught(store, DS, TS, monkeypatch):
    good = A.policy_views
    perm = np.random.default_rng(0).permutation(len(DS["row_id"]))

    def misaligned(z, D, meta=None):                                     # the defect: v2 rows shuffled in the pair
        V = good(z, D, meta)
        X = dict(V["X"])
        X["pair"] = np.hstack([X["v1"], X["v2"][perm]])
        return {**V, "X": X}
    z = _release(store, "U|JOINT|i2o4|l0.1|D1")
    monkeypatch.setattr(A, "policy_views", misaligned)
    r = A.inner_unit("policy", 0, "U|JOINT|i2o4|l0.1|D1", DS, units_dir=store, slate="tiny", reuse=False)
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


def test_defect_clean_output_bypass_is_caught(store, DS, TS, monkeypatch):
    z = _release(store, "U|JOINT|i2o4|l0.1")
    for fn in (A.policy_views, A.lazy_policy_views, A.token_views, A.prob_views, A.code_view_sets):
        with pytest.raises(ValueError, match="extra"):
            fn({**z, "p2": TS["p2"]}, DS)                                # the release contract refuses it
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


def test_defect_missing_source_composition_winner_is_caught(store, DS):
    for c in POLS:
        _save_inner(store, A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny"))
    src = A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)
    sp, arr = _pub(src), _arr(src)
    assert A.validate_inner(sp, arr, DS, registered=False)["ok"]
    w = sp["composed"]["winner"]["v2"]
    assert w != "source"
    bad = json.loads(json.dumps(sp))
    bad["composed"]["freeze"] = [c for c in bad["composed"]["freeze"] if c != w]
    with pytest.raises(A.AuditDefect, match="missing from composed.freeze"):
        A.validate_inner(bad, arr, DS, registered=False)
    assert any("frozen code" in x for x in A.validate_composed(
        {**sp, "composed": {**sp["composed"], "freeze": sp["composed"]["freeze"] + ["U|FINE-TASK|i9o9"]}}))
    with pytest.raises(A.AuditDefect, match="registered composition bank"):    # an explicit bank never passes
        A.validate_inner(sp, arr, DS, registered=True)


def test_defect_missing_diagnostic_family_is_caught(store, DS):
    rec, arr, z = _policy_record(store, DS)
    bad = json.loads(json.dumps(rec))
    bad["families"].pop("prob")
    with pytest.raises(A.AuditDefect, match="diagnostic family prob missing"):
        A.validate_inner(bad, arr, DS, release=z)


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
    from cbp import audit as CA                                        # unchanged from cbp
    assert A.CONTROL_LIMITS == {**CA.CONTROL_LIMITS, "source": A.CONTROL_LIMITS["source"]}
    json.dumps(A.CONTROL_LIMITS, allow_nan=False)


@pytest.mark.parametrize("kind", ["d0", "d1", "new"])
def test_controls_detect_every_plant_and_pass_the_null(DS, TS, kind):
    z = A.synthetic_release(DS, TS, 2, 4, seed=3)
    z = {"d0": z, "d1": A.synthetic_d1(z), "new": A.synthetic_d1(A.synthetic_release(DS, TS, 3, 5))}[kind]
    r = A.controls_for_release("code", A.policy_views(z, DS), DS, z_policy=z, slate="tiny")
    ch = r["checks"]
    assert ch["NULL"]["ok"], ch["NULL"]
    for p in ("CONF_r1", "CONF_r2", "COLL_r1", "COLL_r2", "XOR"):
        assert ch[p]["ok"], (p, ch[p])
    assert ch["CONF_r2"]["decisions_only_misses_it"]               # a decision-only audit cannot see the plant
    assert ch["COLL_r2"]["decoded_probability_only_misses_it"]      # nor can a decoded-probability-only audit
    assert ch["XOR"]["locals_null_ok"] and r["all_ok"]


def test_rotated_source_control_is_detected(DS, TS):
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
    plan = {"policies": [(0, "U|DIRECT-TASK|i2o4"), (0, "U|JOINT|i2o4|l0.1|D1")], "sources": [], "references": [],
            "rotate_references": [], "null_policy": (0, "U|DIRECT-TASK|i2o4"), "null_reps": 2}
    out = tmp_path / "AUDIT_PRELOCK_CHECKS.json"
    p0 = A.stage_controls(DS, "0/2", units_dir=store, out_path=out, slate="tiny", plan=plan, source_threshold=None)
    assert p0["status"] == "PARTIAL" and not out.exists()
    v = A.stage_controls(DS, "1/2", units_dir=store, out_path=out, slate="tiny", plan=plan, source_threshold=None)
    assert out.exists() and v["all_ok"] and v["null_calibration_present"]
    assert v["code_arms_covered"] == ["d0", "d1_fixed"]
    pub = json.loads(out.read_text())
    assert pub["study"] == A.STUDY and pub["registered_limits"]["plant_min"] == 0.75
    assert pub["verdict"]["positive_controls"]["U|JOINT|i2o4|l0.1|D1|s0"]["COLL_r2"]
    parts = [json.loads(p.read_text())["result"] for p in sorted((store.parent / "controls").glob("job*.json"))]
    s = A.summarise_controls(parts, plan, "tiny")                  # synthetic split != the source threshold
    assert not s["verdict"]["all_ok"] and s["verdict"]["realised_threshold_matches_source"] is False
    with pytest.raises(SystemExit, match="sealed"):
        A.stage_controls({**DS, "sealed": False}, None, units_dir=store, out_path=out, slate="tiny", plan=plan)


def test_control_plan_is_structural_and_covers_d1_and_new_codes():
    p = A.control_plan()
    codes = [c for _, c in p["policies"]]
    assert codes == ["U|DIRECT-TASK|i8o64", "U|JOINT|i8o64|l0.1", "U|CLASS|i1o1", "U|JOINT|i8o64|l0.1|D1",
                     "U|K-JOINT-PAIR|i8o64|D1"]
    assert all(c in R.code_ids() for c in codes)
    assert {R.parse_id(c)["arm"] for c in codes} == {"d0", "d1_fixed", "constrained"}
    assert p["null_policy"] == (0, "U|DIRECT-TASK|i8o64") and p["rotate_references"] == ["E"]
    assert p["sources"] == [("U", 0), ("RAW-J_b0.3", 0)] and [r for r, _ in p["references"]] == ["E", "F", "F0"]


# ====================================================================== synthetic timing / estimates
def _meas():
    fam = {"code": 30.0, "token": 22.0, "prob": 9.0}
    return {"policy_inner": {"d0_i8o64": {"unit_cpu_s": 63.0, "family_cpu_s": fam},
                             "d1_i8o64_token_reused": {"unit_cpu_s": 41.0, "family_cpu_s": {**fam, "token": 0.1}},
                             "d0_i1o1": {"unit_cpu_s": 12.0, "family_cpu_s": {"code": 5.0, "token": 4.0,
                                                                              "prob": 3.0}}},
            "source_unit": {"SRC|U_cpu_s": 25.0, "SRC|RAW-J_cpu_s": 23.0, "composed_assembly_cpu_s": 0.5},
            "reference_unit": {"E_cpu_s": 25.0, "F_cpu_s": 34.0},
            "controls": {"code_null_and_5_plants_cpu_s": 280.0, "code_null_only_cpu_s": 40.0,
                         "source_interface_null_and_2_rot_cpu_s": 14.0},
            "final": {"code_i8o64_one_family_cpu_s": 34.0, "code_i8o64_token_family_cpu_s": 25.0,
                      "code_i8o64_prob_family_cpu_s": 10.0, "source_own_five_families_cpu_s": 50.0}}


def test_estimates_cover_the_lra_bank_and_are_finite():
    e = A.estimates(_meas())
    assert e["inner_units"] == 249 + 6 + 9
    assert e["inner_cpu_s_per_seed_parts"]["codes"] == 26 * 63 + 12 + 26 * 41 + 30 * 63
    assert e["inner_cpu_h_without_token_reuse"] > e["inner_cpu_h"]
    assert e["assessment_cpu_h_diagnostics_on_every_code"] > e["assessment_cpu_h_with_code_diagnostics"]
    assert e["d0_recompute_extra_cpu_h"] == round(3 * (26 * 30 + 5 + 25 + 2 * 34) / 3600, 3)
    assert e["assessment_cpu_h_upper_all_83_composed"] > e["assessment_cpu_h_with_code_diagnostics"] > \
        e["assessment_cpu_h"]
    assert e["total_cpu_h"] > 0 and e["budget_with_x2_margin_cpu_h"] == round(2 * e["total_cpu_h"], 2)
    json.dumps(e, allow_nan=False)


def test_reestimate_refreshes_the_registered_rules(tmp_path):
    ac = tmp_path / "AUDIT_COMPUTE.json"
    ac.write_text(json.dumps({"measured": _meas(), "code_families": {"primary": "code"}}))
    e = A.reestimate(ac)
    d = json.loads(ac.read_text())
    assert d["estimates"] == e and d["code_families"]["L4b"]["id"] == "L4b"
    assert d["code_families"]["token_family_readers"] == "cells_only"
    assert d["registered_control_limits"]["null_z"] == 3.5
    assert d["registered_composition_bank"]["n"] == 83 and len(d["control_plan"]["policies"]) == 5


def test_timing_key_merge_keeps_other_keys(tmp_path):
    ac = tmp_path / "AUDIT_COMPUTE.json"
    ac.write_text(json.dumps({"estimates": A.estimates(_meas()), "measured": _meas(), "written_at": "x",
                              "machine": {"python": "3"}}))
    tj = tmp_path / "TIMING.json"
    tj.write_text(json.dumps({"fitting": {"owner": "C", "v": 1}}))
    A.write_timing_key(ac, tj)
    T = json.loads(tj.read_text())
    assert T["fitting"] == {"owner": "C", "v": 1} and T["audit"]["owner"].startswith("role D")
    assert not list(tmp_path.glob("TIMING.json.tmp*"))


# ====================================================================== the locked assessment (lra.assess)
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


def test_assess_is_bound_to_the_lra_lock_branch_and_chain():
    assert AS.STUDY_BRANCH == "research/pcrl-adult-learned-decoder-release-v1"
    assert AS.LOCK_REL == "results/pcrl_adult_learned_decoder_release_v1/EVALUATION_LOCK.json"
    for f in ("lra/assess.py", "lra/audit.py", "lra/baselines.py", "lra/data.py", "lra/run.py", "lra/lock.py",
              "qpc/utility.py", "dpc/audit.py", "osf/data.py", "smf/audit.py", "jcv/finalize.py"):
        assert f in AS.CHAIN
    assert not any(f.startswith("cbp/") for f in AS.CHAIN)
    assert all((R.WT / f).exists() for f in AS.CHAIN)
    assert AS.AU is A and AS.BL is BL
    from lra import data as LD
    assert LD.UNSEAL_CALLER == "lra.assess" and LD.BRANCH == AS.STUDY_BRANCH and \
        str(LD.EVALUATION_LOCK).endswith(AS.LOCK_REL)


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
        AS.open_assessment(p, repo, chain=("code.py",))          # lra modules are loaded but not in this lock
    lock = json.loads(p.read_text())
    assert AS.verify_code(lock, repo, chain=("code.py",), check_modules=False)["code.py"] == "matches lock"
    f.write_text("print(2)\n")
    with pytest.raises(SystemExit, match="differs from the locked code hash"):
        AS.verify_code(lock, repo, chain=("code.py",), check_modules=False)
    AS.close_assessment()


def test_lra_data_refuses_unsealing_outside_assess():
    from lra import data as LD
    with pytest.raises((PermissionError, SystemExit), match="only lra.assess"):
        LD.load(unseal=True)


def test_jobs_are_only_the_locked_list_and_units_map_through_the_registered_bank():
    lock = {"seeds": {str(k): {"score": {f"L{j}": {"kind": "policy", "cid": "U|CLASS|i1o1"} for j in range(5)}}
                      for k in (0, 1, 2)}}
    jobs = AS.jobs_from_lock(lock)
    assert len(jobs) == 15
    sh = [AS.jobs_from_lock(lock, shard=f"{i}/2") for i in (0, 1)]
    assert sorted(map(str, sh[0] + sh[1])) == sorted(map(str, jobs)) and not set(map(str, sh[0])) & set(map(str, sh[1]))
    for u, c in (("pol__s0__U_JOINT_i8o64_l0.025", "U|JOINT|i8o64|l0.025"),
                 ("dec__s1__U_FINE-TASK_i8o64_D1", "U|FINE-TASK|i8o64|D1"),
                 ("new__s2__U_W-SEQ-21_i8o64_l0.06_D1", "U|W-SEQ-21|i8o64|l0.06|D1"),
                 ("new__s0__U_K-JOINT-SINGLE_i8o64_D1", "U|K-JOINT-SINGLE|i8o64|D1"), ("U|CLASS|i1o1", "U|CLASS|i1o1")):
        assert AS._policy_cid_of(u) == c
    for bad in ("pol__s0__U_JOINT_i8o64_l0.2", "new__s0__U_K-JOINT-TRIPLE_i8o64_D1", "U|JOINT|i8o64|l1"):
        with pytest.raises(ValueError, match="REFUSED"):
            AS._policy_cid_of(bad)


def _unsealed(D, seed=3):
    rng = np.random.default_rng(seed)
    Du = {**D, "y": {k: v.copy() for k, v in D["y"].items()}, "sealed": False}
    a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    Du["y"]["income"][a] = rng.integers(0, 2, len(a))
    Du["y"]["occupation_group"][a] = rng.integers(0, 6, len(a))
    Du["sex"] = D["sex"].copy()
    Du["sex"][a] = rng.integers(0, 2, len(a))
    Du["race"] = rng.integers(0, 3, len(D["row_id"]))
    return Du


def test_outer_unit_end_to_end_composed_freeze_registered_bank_and_restore_hook(repo, reg_store, DS, monkeypatch):
    src = A.inner_unit("source", 0, "SRC|U", DS, units_dir=reg_store, slate="tiny")
    _save_inner(reg_store, src)
    freeze = src["composed"]["freeze"]
    assert any(c.startswith("U|JOINT|i8o64|l0.1") for c in freeze)
    score = {"SRC|U": {"kind": "source", "cid": "SRC|U", "unit": "tea__s0__U"},
             "J*": {"kind": "policy", "cid": "U|K-JOINT-PAIR|i8o64|D1"},
             "D1ctl": {"kind": "policy", "cid": "U|JOINT|i8o64|l0.1|D1"},
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
        L["seeds"]["0"]["composed_policies"] = ["dec__s1__U_JOINT_i8o64_l0.1_D1"]
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
            unit = {"policy": A.unit_of(0, spec["cid"]), "source": "tea__s0__U", "reference": "ref__s0__F"}[
                spec["kind"]]
            if spec["kind"] == "source":
                assert r["composed"]["class_only_units"] == ["pol__s0__U_CLASS_i1o1"]
                assert "P_auc_decisions_pair" in z
                np.testing.assert_array_equal(z["prob2"], z["u_prob2"])
                assert r["families"]["interface"]["scored"]["v2"]["auc"]["candidate"].startswith("composed[")
            if spec["kind"] == "policy":
                assert r["decision_preservation"]["ok"] and r["primary_family"] == "code"
                assert {"P_auc_token_pair", "P_auc_prob_v2", "P_ce_token_v1"} <= set(z)
                np.testing.assert_array_equal(z["prob2"], _release(reg_store, spec["cid"])["q2"][
                    Du["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]])
            for w in ("pair", "v2"):        # restore hook: refit from stored units on a SEALED D, bitwise
                P = AS.refit_selected_attacker(reg_store, DS, unit, r["unit"], view=w)
                assert np.array_equal(P, z[f"P_auc_{w}"]), (label, w)
            if spec["kind"] == "policy":
                for fam in ("token", "prob"):
                    P = AS.refit_selected_attacker(reg_store, DS, unit, r["unit"], view="pair", family=fam)
                    assert np.array_equal(P, z[f"P_auc_{fam}_pair"]), (label, fam)
            assert np.array_equal(AS.refit_selected_attacker(reg_store, DS, unit, r["unit"], attacker_seed=1),
                                  z["P_auc_pair"][1])
            assert not any(math.isnan(x) for x in rr["families"][rr["primary_family"]]["scored"]["pair"]["auc"][
                "auc_per_seed"])
    finally:
        AS.close_assessment()


def test_lazy_code_views_are_bit_identical_in_the_final_audit(DS, TS):
    z = A.synthetic_d1(A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.3))
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
    """The inner utility / gate used by lra.audit is qpc.utility unchanged (same rows, same functions)."""
    D = UT.synthetic_task_D(seed=0, n_fit=2000, n_head=10, n_audit=500, n_sel=500, n_assess=500)
    t = UT.synthetic_teacher(D, seed=0)
    u = UT.release_inner_utility({1: t["p1"], 2: t["p2"]}, {1: t["d1"], 2: t["d2"]}, D)
    g = UT.gate_record(u, u, {1: True, 2: True})
    assert g["eligible"]
    for task in ("income", "occupation"):
        assert {"acc", "logloss", "brier", "const_acc"} <= set(u[task])
