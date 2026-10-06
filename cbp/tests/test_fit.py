"""Role B tests for cbp.fit and cbp.deploy. SYNTHETIC data only: no Adult row, task label or SEX is read.

    cd <WORKTREE> && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python \\
        -m cbp.sema --label B:test-fit -- ~/PCRL/.venv/bin/python -m pytest -q cbp/tests/test_fit.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from cbp import fit as FT
from qpc import compress as CP
from qpc import partition as PT
from qpc import release as RL
from qpc import stagea as SA

META = {"teacher": "U", "seed": 0, "teacher_model_sha256": "a" * 64, "feature_names_sha256": "b" * 64}
TIMING_KEYS = {"wall_seconds", "cpu_seconds", "augment_cpu_seconds", "augment_wall_seconds", "cpu_seconds_unit",
               "wall_seconds_unit"}


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in TIMING_KEYS}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _materialize(writer, tmp):
    p = Path(tmp) / "x.json"
    writer(p)
    return json.loads(p.read_text())


def _rehash(unit):
    """Rewrite COMPLETE.json after a deliberate tamper (so only the content checks can catch it)."""
    unit = Path(unit)
    hashes = {str(p.relative_to(unit)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(unit.rglob("*")) if p.is_file() and p.name != "COMPLETE.json"}
    (unit / "COMPLETE.json").write_text(json.dumps({"id": unit.name, "files": hashes}, indent=1))


def _tree_hash(d):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(d).rglob("*"))
            if p.is_file()}


# ----------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    T, tr, S = FT.synthetic_teacher(N=3000, n_fit=2100, seed=1, underflow=40)
    _, files = PT.fine_unit(T, tr, caps={1: 12, 2: 72})
    fine = _materialize(files["fine.json"], tmp_path_factory.mktemp("fine"))
    return T, tr, S, fine


@pytest.fixture(scope="module")
def units(syn, tmp_path_factory):
    """Synthetic analogues of the admitted qpc units (plain qpc, runner record update) plus NEW lambda-0.04 units fitted
    through cbp.fit.fit_unit."""
    T, tr, S, fine = syn
    root = tmp_path_factory.mktemp("units")
    recs, pols, dirs = {}, {}, {}

    def store(cid, rec, files):
        rec.update({"seed": 0, "config": cid})
        u = root / FT.unit_name(0, cid)
        FT.save_unit(u, files, rec)
        recs[cid], dirs[cid] = rec, u
        pols[cid] = json.loads((u / "policy.json").read_text())
    for fam in ("CLASS", "FINE-TASK"):
        cid = FT.config_id(fam)
        store(cid, *CP.fit_unit(fam, fine, T, tr, S, FT.M1, FT.M2, None, {**META, "config": cid}))
    p1, _ = SA.recipient_fit(T["p1"][tr], T["d1"][tr], 2, FT.M1, 1)
    p2, _ = SA.recipient_fit(T["p2"][tr], T["d2"][tr], 6, FT.M2, 2)
    cid = FT.config_id("DIRECT-TASK")
    store(cid, *SA.pair_unit(p1, p2, T, {**META, "config": cid}, tr=tr))
    for lam in (0.01, 0.1):
        for fam in FT.FAMILIES:
            cid = FT.config_id(fam, lam)
            wit = {f: pols[c] for f, c in FT.witness_configs(lam).items()} if fam == "JOINT" else None
            store(cid, *CP.fit_unit(fam, fine, T, tr, S, FT.M1, FT.M2, lam, {**META, "config": cid}, witnesses=wit))
    for fam in FT.FAMILIES:
        cid = FT.config_id(fam, 0.04)
        wit = {f: pols[c] for f, c in FT.witness_configs(0.04).items()} if fam == "JOINT" else None
        store(cid, *FT.fit_unit(fam, fine, T, tr, S, 0.04, {**META, "config": cid}, witnesses=wit))
    return dict(root=root, recs=recs, pols=pols, dirs=dirs)


# ----------------------------------------------------------------------------------------------- bank
def test_bank_shape_ids_and_order():
    b = FT.bank()
    assert len(b) == 72 and len({u["unit"] for u in b}) == 72
    assert FT.LAMS == (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
    assert FT.FAMILIES == ("LOCAL", "SEQ-12", "SEQ-21", "JOINT") and FT.SEEDS == (0, 1, 2)
    assert (FT.M1, FT.M2) == (8, 64)
    assert sum(u["status"] == "REUSED" for u in b) == 24 and sum(u["status"] == "NEW" for u in b) == 48
    assert {u["lam"] for u in b if u["status"] == "REUSED"} == {0.01, 0.1}
    assert {u["lam"] for u in b if u["status"] == "NEW"} == {0.025, 0.04, 0.06, 0.08}
    cids = {u["config"] for u in b}
    assert len(cids) == 24
    for lam, s in ((0.01, "0.01"), (0.025, "0.025"), (0.04, "0.04"), (0.06, "0.06"), (0.08, "0.08"), (0.1, "0.1")):
        for f in FT.FAMILIES:
            assert f"U|{f}|i8o64|l{s}" in cids
    assert FT.unit_name(2, "U|SEQ-21|i8o64|l0.025") == "pol__s2__U_SEQ-21_i8o64_l0.025"
    assert FT.config_id("FINE-TASK") == "U|FINE-TASK|i8o64" and FT.config_id("DIRECT-TASK") == "U|DIRECT-TASK|i8o64"
    assert FT.config_id("CLASS") == "U|CLASS|i1o1"
    refs = FT.reference_units()
    assert len(refs) == 9 and {r["config"] for r in refs} == {"U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1"}
    assert len(FT.reusable_ids()) == 11 and len(FT.registered_ids()) == 27
    for u in b:
        if u["family"] == "JOINT":
            assert u["witnesses"]["FINE-TASK"]["config"] == "U|FINE-TASK|i8o64"
            for f in ("LOCAL", "SEQ-12", "SEQ-21"):
                assert u["witnesses"][f]["config"] == FT.config_id(f, u["lam"])
    order = FT.new_fit_order()
    assert len(order) == 48 and all(j["lam"] in FT.NEW_LAMS for j in order)
    pos = {(j["seed"], j["config"]): i for i, j in enumerate(order)}
    for j in order:
        if j["family"] == "JOINT":
            for f in ("LOCAL", "SEQ-12", "SEQ-21"):
                assert pos[(j["seed"], FT.config_id(f, j["lam"]))] < pos[(j["seed"], j["config"])]
    c = FT.bank_counts()
    assert (c["logical_privacy_units"], c["reused_endpoint_units"], c["new_fits"], c["reused_reference_units"]) == \
        (72, 24, 48, 9)
    with pytest.raises(ValueError):
        FT.status_of(0.05)
    with pytest.raises(ValueError):
        FT.config_id("FINE-TASK", 0.1)


def test_qpc_constants_and_code_pins():
    assert (CP.SWEEPS, CP.TOL, CP.TIE_TOL) == (5, 1e-12, 1e-12)
    assert CP.JOINT_START_ORDER == ("JOINT-GREEDY", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
    assert CP.WITNESSES == FT.WITNESS_FAMILIES
    lock = json.loads((FT.WT / FT.QPC_PKG / "STAGE_B_LOCK.json").read_text())["code_files"]
    for f, h in FT.QPC_STAGE_B_PINS.items():
        assert lock[f] == h
    for fam in ("JOINT", "DIRECT-TASK"):
        cv = FT.code_version(fam)
        assert cv["ok"] and {"qpc/compress.py", "qpc/release.py"} <= set(cv["gated_files"])
    assert "stage_a" in FT.code_version("DIRECT-TASK")


# ----------------------------------------------------------------------------------------------- fit_unit
def test_fit_unit_is_unchanged_qpc_call(syn, units, tmp_path):
    T, tr, S, fine = syn
    for fam in ("LOCAL", "JOINT"):
        cid = FT.config_id(fam, 0.04)
        wit = {f: units["pols"][c] for f, c in FT.witness_configs(0.04).items()} if fam == "JOINT" else None
        rq, fq = CP.fit_unit(fam, fine, T, tr, S, 8, 64, 0.04, {**META, "config": cid}, witnesses=wit)
        rc = dict(units["recs"][cid])
        assert set(rc) - set(rq) == {"cbp", "seed"}
        rc.pop("cbp")
        rc.pop("seed")
        assert _strip(json.loads(json.dumps(rc))) == _strip(json.loads(json.dumps(rq)))
        assert _materialize(fq["policy.json"], tmp_path) == units["pols"][cid]
        with np.load(units["dirs"][cid] / "release.npz") as z:
            for k, v in fq["release.npz"].items():
                assert np.asarray(v).dtype == z[k].dtype and np.asarray(v).tobytes() == z[k].tobytes()
        assert units["recs"][cid]["cbp"]["qpc_call"] == {"function": "qpc.compress.fit_unit", "m1": 8, "m2": 64,
                                                         "lam": 0.04, "witness_slots": sorted(wit) if wit else []}
        assert units["recs"][cid]["cbp"]["bank_status"] == "NEW"


def test_fit_unit_deterministic(syn, units, tmp_path):
    T, tr, S, fine = syn
    cid = FT.config_id("SEQ-12", 0.04)
    a, fa = FT.fit_unit("SEQ-12", fine, T, tr, S, 0.04, {**META, "config": cid})
    b, fb = FT.fit_unit("SEQ-12", fine, T, tr, S, 0.04, {**META, "config": cid})
    assert _strip(a) == _strip(b)
    assert _materialize(fa["policy.json"], tmp_path) == _materialize(fb["policy.json"], tmp_path)
    for k in fa["release.npz"]:
        assert np.asarray(fa["release.npz"][k]).tobytes() == np.asarray(fb["release.npz"][k]).tobytes()
    assert _strip(a["cbp"]["section13"]) == _strip(units["recs"][cid]["cbp"]["section13"])


def test_fit_unit_refusals(syn, units):
    T, tr, S, fine = syn
    for fam in ("FINE-TASK", "CLASS", "DIRECT-TASK", "OCC-ONLY"):
        with pytest.raises(ValueError, match="REFUSED"):
            FT.fit_unit(fam, fine, T, tr, S, 0.04, dict(META))
    for lam in (0.05, 1.0, None):
        with pytest.raises(ValueError, match="REFUSED"):
            FT.fit_unit("LOCAL", fine, T, tr, S, lam, dict(META))
    for lam in (0.01, 0.1):
        with pytest.raises(ValueError, match="refit_reason"):
            FT.fit_unit("LOCAL", fine, T, tr, S, lam, dict(META))
    w = {f: units["pols"][c] for f, c in FT.witness_configs(0.04).items()}
    with pytest.raises(ValueError, match="four witness"):
        FT.fit_unit("JOINT", fine, T, tr, S, 0.04, dict(META))
    with pytest.raises(ValueError, match="four witness"):
        FT.fit_unit("JOINT", fine, T, tr, S, 0.04, dict(META), witnesses={k: v for k, v in w.items() if k != "LOCAL"})
    with pytest.raises(ValueError, match="JOINT only"):
        FT.fit_unit("LOCAL", fine, T, tr, S, 0.04, dict(META), witnesses=w)
    with pytest.raises(ValueError, match="differs"):
        FT.fit_unit("LOCAL", fine, T, tr, S, 0.04, {**META, "config": "U|LOCAL|i8o64|l0.06"})
    wrong = dict(w)
    wrong["LOCAL"] = units["pols"][FT.config_id("LOCAL", 0.1)]          # witness at another lambda: qpc refuses
    with pytest.raises(ValueError, match="different family/caps/lam"):
        FT.fit_unit("JOINT", fine, T, tr, S, 0.04, dict(META), witnesses=wrong)
    rec, _ = FT.fit_unit("LOCAL", fine, T, tr, S, 0.1, dict(META), refit_reason="synthetic test of the refit path")
    assert rec["cbp"]["bank_status"] == "ENDPOINT_REFIT" and rec["cbp"]["refit_reason"]


def test_section13_from_deployed_release(syn, units):
    T, tr, S, fine = syn
    for cid in (FT.config_id(f, 0.04) for f in FT.FAMILIES):
        rec = units["recs"][cid]
        s = rec["cbp"]["section13"]
        pair = RL.PolicyPair.from_dict(units["pols"][cid])
        a = s["alphabets"]
        assert a["full"] == {"r1": pair.p1.T, "r2": pair.p2.T, "pair": pair.p1.T * pair.p2.T}
        assert all(a["occupied_fit"][k] <= a["full"][k] for k in a["full"])
        assert s["unseen_fraction_fit"]["pair"] == pytest.approx(1 - a["occupied_fit"]["pair"] / a["full"]["pair"])
        st = s["states_per_predicted_class"]
        assert all(t <= 8 for t in st["r1"]["tokens"]) and all(t <= 64 for t in st["r2"]["tokens"])
        assert st["r2"]["tokens"][5] == 1 and st["r2"]["occupied_fit"][5] == 0          # absent class: one fallback
        assert s["fallback_tokens"]["r2"] and not s["fallback_tokens"]["r1"]
        assert s["total_occupied_states_fit"] == sum(st["r1"]["occupied_fit"]) + sum(st["r2"]["occupied_fit"])
        cmp_ = s["deployed_vs_record_final"]
        assert cmp_["within_rtol"] and cmp_["max_rel_diff"] <= 1e-12
        assert s["deployed_vs_qpc_row_level_check"]["all_bitwise_equal"]           # same arithmetic as qpc's receipt
        assert s["sparsity_equals_qpc_receipt"]
        r = s["reconstruction"]
        assert r["structure_ok"] and r["rows_fit"] == tr.size
        for i in (1, 2):
            assert r[f"r{i}"]["token_counts_equal_policy"] and r[f"r{i}"]["decoded_equals_prototype_bitwise"]
            assert r[f"r{i}"]["decisions_equal_teacher"] and r[f"r{i}"]["token_mean_within_atol"]
        assert set(s["mi_vs_permutation_null"]) >= {"I1", "I2", "I12"}
        se = s["search"]
        assert se["defined"] and se["sweep_cap"] == 5
        assert all(v["sweeps"] <= 5 for v in se["per"].values())
        tot = se["totals"]
        assert set(tot) >= {"merges_logged", "positive_increment_merges", "accepted_exchanges", "refine_merges"}
        if cid.split("|")[1] == "JOINT":
            assert se["kind"] == "joint_starts" and set(se["per"]) == set(CP.JOINT_START_ORDER)
            assert se["unresolved_local_optima"] == len(rec["unresolved_local_optima"])
        else:
            assert se["unresolved_local_optima"] is None


def test_reconstruction_detects_tampered_release(syn, units):
    T, tr, S, fine = syn
    cid = FT.config_id("JOINT", 0.04)
    pair = RL.PolicyPair.from_dict(units["pols"][cid])
    with np.load(units["dirs"][cid] / "release.npz") as z:
        out = {k: z[k].copy() for k in z.files}
    ok = FT.reconstruct_fit_statistics(pair, out, T, tr, S, 0.04)
    assert ok["structure_ok"]
    row = int(tr[0])
    bad = {k: v.copy() for k, v in out.items()}
    bad["q2"][row, 0] = np.nextafter(bad["q2"][row, 0], 1.0)
    r = FT.reconstruct_fit_statistics(pair, bad, T, tr, S, 0.04)
    assert not r["structure_ok"] and not r["r2"]["decoded_equals_prototype_bitwise"]
    bad = {k: v.copy() for k, v in out.items()}
    c = int(pair.p1.token_class[bad["tok1"][row]])
    other = [t for t in np.flatnonzero(pair.p1.token_class == c) if t != bad["tok1"][row]][0]
    bad["tok1"][row] = other
    r = FT.reconstruct_fit_statistics(pair, bad, T, tr, S, 0.04)
    assert not r["structure_ok"] and not r["r1"]["token_counts_equal_policy"]
    bad = {k: v.copy() for k, v in out.items()}
    bad["tok2"][row] = pair.p2.T + 3
    r = FT.reconstruct_fit_statistics(pair, bad, T, tr, S, 0.04)
    assert not r["structure_ok"] and not r["r2"]["tokens_in_alphabet"]


def test_asymmetry_receipts(units):
    recs = units["recs"]
    aj = recs[FT.config_id("JOINT", 0.04)]["cbp"]["computational_asymmetry"]
    assert aj["search_paths"] == 5 and aj["candidates_compared"] == 9 and aj["witnesses_consumed"] == 4
    assert set(aj["witness_sources"].values()) == {"passed_in"}
    a2 = recs[FT.config_id("SEQ-12", 0.04)]["cbp"]["computational_asymmetry"]
    assert a2["search_paths"] == 1 and a2["candidates_compared"] == 1 and a2["witnesses_consumed"] == 0
    assert a2["unselected_diagnostic_fits"] == 1
    assert "not an exact compute match" in aj["note"] and "Taylor" in aj["note"]
    table = FT.asymmetry_table({(0, c): r for c, r in recs.items()})
    rows = {(r["seed"], r["lam"]): r for r in table["rows"]}
    for lam in (0.01, 0.04, 0.1):
        r = rows[(0, lam)]
        assert r["complete"] and r["candidates"]["JOINT"] == 9 and r["candidates"]["SEQ-21"] == 1
        assert r["joint_cpu_s_incl_witness_fits"] >= r["joint_cpu_s_own"]
    assert not rows[(0, 0.06)]["complete"] and not rows[(1, 0.04)]["complete"]


# ----------------------------------------------------------------------------------------------- endpoint parity
def test_endpoint_parity_passes_without_refit(syn, units):
    T, tr, S, fine = syn
    before = _tree_hash(units["root"])
    for cid in FT.reusable_ids():
        wr = None
        if cid.split("|")[1] == "JOINT":
            lam = float(cid.split("|l")[1])
            wr = {f: units["recs"][c] for f, c in FT.witness_configs(lam).items()}
        pr = FT.endpoint_parity(units["dirs"][cid], T, tr, S, fine, expected_config=cid, meta=META,
                                witness_records=wr)
        assert pr["ok"], (cid, pr["failed"])
        assert pr["refit"] is False and pr["status"] == "PARITY_PASS"
        assert pr["release_bitwise_detail"]["_same_key_set"]
        assert all(pr["class_preservation"][f"r{i}"]["stored_failures"] == 0 for i in (1, 2))
        if cid.split("|")[1] == "JOINT":
            assert pr["checks"]["joint_witness_dominance"]
            assert all(v["within_rtol"] for v in pr["joint_witnesses"]["witness_unit_cross_check"].values())
        if cid == "U|DIRECT-TASK|i8o64":
            assert pr["fine_partition_detail"]["applicable"] is False
        else:
            assert pr["fine_partition_detail"]["applicable"] is True
        json.dumps(pr, allow_nan=False)
    assert _tree_hash(units["root"]) == before                     # nothing written, nothing refit


def _copy(units, cid, tmp_path):
    d = tmp_path / units["dirs"][cid].name
    shutil.copytree(units["dirs"][cid], d)
    return d


def _tamper_release(d, fn, rehash=True):
    with np.load(d / "release.npz") as z:
        a = {k: z[k].copy() for k in z.files}
    fn(a)
    np.savez_compressed(d / "release.npz", **a)
    if rehash:
        _rehash(d)


@pytest.mark.parametrize("case", ["token", "token_no_rehash", "q_ulp", "decision", "extra_key", "final_term",
                                  "fine_dict", "policy_proto", "code_pin", "binding", "new_lambda", "witness",
                                  "wrong_expected", "fingerprint"])
def test_endpoint_parity_detects_tampering(syn, units, tmp_path, monkeypatch, case):
    T, tr, S, fine = syn
    cid = FT.config_id("JOINT", 0.1)
    kw = {"expected_config": cid, "meta": META}
    d = _copy(units, cid, tmp_path)
    row = int(tr[1])
    expect = None
    if case in ("token", "token_no_rehash"):
        pair = RL.PolicyPair.from_dict(units["pols"][cid])

        def f(a):
            c = int(pair.p2.token_class[a["tok2"][row]])
            a["tok2"][row] = [t for t in np.flatnonzero(pair.p2.token_class == c) if t != a["tok2"][row]][0]
        _tamper_release(d, f, rehash=case == "token")
        expect = {"complete_json_hashes"} if case == "token_no_rehash" else set()
        expect |= {"release_bitwise", "fitting_statistics_exact", "fitting_terms_within_rtol"}
    elif case == "q_ulp":
        _tamper_release(d, lambda a: a["q1"].__setitem__((row, 1), np.nextafter(a["q1"][row, 1], 1.0)))
        expect = {"release_bitwise", "fitting_statistics_exact"}
    elif case == "decision":
        _tamper_release(d, lambda a: a["hard2"].__setitem__(row, (a["hard2"][row] + 1) % 6))
        expect = {"release_bitwise", "class_preservation_all_rows", "fitting_statistics_exact"}
    elif case == "extra_key":
        _tamper_release(d, lambda a: a.__setitem__("f1", np.zeros(3, dtype=np.int64)))
        expect = {"release_bitwise"}
    elif case == "final_term":
        r = json.loads((d / "record.json").read_text())
        r["final"]["D2"] *= 1 + 1e-10
        (d / "record.json").write_text(json.dumps(r))
        _rehash(d)
        expect = {"fitting_terms_within_rtol"}
    elif case == "fine_dict":
        T2, tr2, _ = FT.synthetic_teacher(N=3000, n_fit=2100, seed=2, underflow=40)
        _, files = PT.fine_unit(T2, tr2, caps={1: 12, 2: 72})
        kw["fine_override"] = _materialize(files["fine.json"], tmp_path)
        expect = {"fine_partition"}
    elif case == "policy_proto":
        z = json.loads((d / "policy.json").read_text())
        z["p1"]["token_proto"][0][0] += 1e-9
        (d / "policy.json").write_text(json.dumps(z))
        _rehash(d)
        expect = {"policy_integrity"}
    elif case == "code_pin":
        monkeypatch.setitem(FT.QPC_STAGE_B_PINS, "qpc/compress.py", "0" * 64)
        expect = {"code_version"}
    elif case == "binding":
        kw["meta"] = {**META, "teacher_model_sha256": "c" * 64}
        expect = {"binding"}
    elif case == "new_lambda":
        cid2 = FT.config_id("JOINT", 0.04)
        d = _copy(units, cid2, tmp_path / "n")
        kw["expected_config"] = cid2
        expect = {"config_registered_reusable"}
    elif case == "witness":
        wr = {f: json.loads(json.dumps(units["recs"][c])) for f, c in FT.witness_configs(0.1).items()}
        wr["SEQ-21"]["final"]["F_joint"] *= 1 + 1e-9
        kw["witness_records"] = wr
        expect = {"joint_witness_dominance"}
    elif case == "wrong_expected":
        kw["expected_config"] = FT.config_id("SEQ-12", 0.1)
        expect = {"config_consistent"}
    elif case == "fingerprint":
        r = json.loads((d / "record.json").read_text())
        r["pair_fingerprint"] = "f" * 64
        (d / "record.json").write_text(json.dumps(r))
        _rehash(d)
        expect = {"fingerprint_matches_record"}
    fine_used = kw.pop("fine_override", fine)
    pr = FT.endpoint_parity(d, T, tr, S, fine_used, **kw)
    assert not pr["ok"] and pr["status"] == "PARITY_FAIL"
    if case == "policy_proto":
        assert pr["failed"] == ["policy_integrity"]
    else:
        assert set(pr["failed"]) == expect, (case, pr["failed"])
    assert pr["refit"] is False


def test_parity_reference_tamper_and_unreadable(syn, units, tmp_path):
    T, tr, S, fine = syn
    for cid in ("U|DIRECT-TASK|i8o64", "U|CLASS|i1o1", "U|FINE-TASK|i8o64"):
        d = _copy(units, cid, tmp_path)
        _tamper_release(d, lambda a: a["q2"].__setitem__((int(tr[0]), 0), np.nextafter(a["q2"][int(tr[0]), 0], 1.0)))
        pr = FT.endpoint_parity(d, T, tr, S, fine, expected_config=cid)
        assert not pr["ok"] and "release_bitwise" in pr["failed"]
    d = _copy(units, "U|LOCAL|i8o64|l0.01", tmp_path)
    (d / "release.npz").write_bytes(b"not a zip")
    pr = FT.endpoint_parity(d, T, tr, S, fine)
    assert not pr["ok"] and pr["failed"] == ["unreadable"]


def test_aliases(units):
    pols = dict(units["pols"])
    pols["U|LOCAL|i8o64|l0.06"] = pols["U|LOCAL|i8o64|l0.04"]          # an exact copy is an alias, not a new fit
    al = FT.aliases(pols)
    assert al["pair_aliases"].get("U|LOCAL|i8o64|l0.06") == "U|LOCAL|i8o64|l0.04"
    assert al["pair_alias_count"] >= 1


# ----------------------------------------------------------------------------------------------- deploy
def _synthetic_teacher_unit(tmp_path, seed=0):
    """Copied from qpc/tests/test_method.py ``_synthetic_unit`` at d0c8a45: a random jcv Model + fitted heads."""
    import joblib
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from jcv.finalize import save_unit
    from jcv.train import Model
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(1200, 83)).astype(np.float64)
    names = [f"f{j:02d}" for j in range(83)]
    model = Model(83, [2, 6], seed)
    with torch.no_grad():
        R = [model.encode(i, torch.from_numpy(X.astype(np.float32))).double().numpy() for i in (0, 1)]
    y1 = (R[0][:, 0] > np.median(R[0][:, 0])).astype(int)
    y2 = np.digitize(R[1][:, 1], np.quantile(R[1][:, 1], [0.2, 0.4, 0.6, 0.8, 0.999]))
    heads = [make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(R[0], y1),
             make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(R[1], y2)]
    unit = tmp_path / "tea__s0__SYNTH"
    state = model.state_dict()
    save_unit(unit, {"model.pt": lambda p: torch.save(state, p),
                     "head_0.joblib": lambda p: joblib.dump(heads[0], p),
                     "head_1.joblib": lambda p: joblib.dump(heads[1], p)}, {"seed": seed})
    return unit, X, names


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    from qpc import deploy as QD
    tmp = tmp_path_factory.mktemp("deploy")
    unit, X, names = _synthetic_teacher_unit(tmp)
    model, heads, msha = QD.load_teacher(unit)
    P1, P2 = QD.teacher_probs(model, heads, X)
    T = {"row_id": np.arange(len(X)), "p1": P1, "p2": P2, "d1": P1.argmax(1), "d2": P2.argmax(1)}
    tr = np.arange(800)
    S = (np.random.default_rng(5).random(800) < 0.4 + 0.2 * (T["d1"][tr] == 1)).astype(np.int64)
    _, ff = PT.fine_unit(T, tr, caps={1: 12, 2: 72})
    fine = _materialize(ff["fine.json"], tmp)
    meta = {"teacher": "U", "seed": 0, "teacher_model_sha256": msha, "feature_names_sha256": QD.schema_sha256(names)}
    rec, files = FT.fit_unit("LOCAL", fine, T, tr, S, 0.04, dict(meta))
    pol = tmp / "policy.json"
    files["policy.json"](pol)
    unreg = {}
    for tag, (m1, m2, lam) in {"lam1": (8, 64, 1.0), "rate": (8, 32, 0.04)}.items():
        _, f2 = CP.fit_unit("LOCAL", fine, T, tr, S, m1, m2, lam, dict(meta))
        unreg[tag] = tmp / f"unreg_{tag}.json"
        f2["policy.json"](unreg[tag])
    np.savez(tmp / "schema.npz", feature_names=np.array(names))
    np.savez(tmp / "in.npz", X=X, feature_names=np.array(names))
    return dict(tmp=tmp, unit=unit, X=X, names=names, pol=pol, rel=files["release.npz"], unreg=unreg)


def _args(D, **kw):
    a = {"--unit": str(D["unit"]), "--policy": str(D["pol"]), "--X": str(D["tmp"] / "in.npz"),
         "--schema": str(D["tmp"] / "schema.npz"), "--out": str(D["tmp"] / "out.npz")}
    a.update(kw)
    out = []
    for k, v in a.items():
        out += [k] if v is True else [k, v]
    return out


def _refused(capsys, args, needle):
    from cbp import deploy as DP
    with pytest.raises(SystemExit) as e:
        DP.main(args)
    assert e.value.code == 2
    assert needle in capsys.readouterr().err


def test_deploy_outputs_only_release_and_matches_stored(deployed, tmp_path, capsys):
    from cbp import deploy as DP
    D = deployed
    DP.main(_args(D))
    info = json.loads(capsys.readouterr().out)
    assert info["registered"] and info["config"] == "U|LOCAL|i8o64|l0.04" and info["binding"] == "BOUND"
    with np.load(D["tmp"] / "out.npz", allow_pickle=False) as z:
        assert sorted(z.files) == sorted(DP.ALLOWED_OUTPUT) == sorted(
            ["tokens_1", "probs_1", "decision_1", "tokens_2", "probs_2", "decision_2"])
        for i in (1, 2):
            assert np.array_equal(z[f"tokens_{i}"], D["rel"][f"tok{i}"])
            assert np.array_equal(z[f"probs_{i}"], D["rel"][f"q{i}"])
            assert np.array_equal(z[f"decision_{i}"], D["rel"][f"hard{i}"])
    shutil.copy(D["pol"], tmp_path / "restored.json")                 # rebuild from a restored copy
    DP.main(_args(D, **{"--policy": str(tmp_path / "restored.json"), "--out": str(tmp_path / "o2.npz")}))
    with np.load(tmp_path / "o2.npz") as a, np.load(D["tmp"] / "out.npz") as b:
        assert all(np.array_equal(a[k], b[k]) for k in DP.ALLOWED_OUTPUT)


def test_deploy_refuses_schema_violations(deployed, capsys):
    D = deployed
    X, names = D["X"], D["names"]
    np.savez(D["tmp"] / "extra.npz", X=np.hstack([X, X[:, :1]]), feature_names=np.array(names + ["SEX"]))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "extra.npz")}), "refused")
    np.savez(D["tmp"] / "missing.npz", X=X[:, :82], feature_names=np.array(names[:82]))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "missing.npz")}), "refused")
    perm = list(range(83))
    perm[10], perm[11] = perm[11], perm[10]
    np.savez(D["tmp"] / "reorder.npz", X=X[:, perm], feature_names=np.array([names[j] for j in perm]))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "reorder.npz")}), "reordered")
    np.savez(D["tmp"] / "bundled.npz", X=X, feature_names=np.array(names), sex=np.zeros(len(X)))
    _refused(capsys, _args(D, **{"--X": str(D["tmp"] / "bundled.npz")}), "exactly X and feature_names")
    _refused(capsys, _args(D, **{"--schema-sha256": "0" * 64}), "schema hash mismatch")


def test_deploy_refuses_exports_unknown_flags_and_bad_policies(deployed, capsys, tmp_path):
    from cbp import deploy as DP
    D = deployed
    for flag in ("--export-fine-ids", "--fine-cells", "--raw-scores", "--teacher-probs", "--logits", "--export=all",
                 "--debug", "--allow-unbound-policy", "--continuous", "--include-sex"):
        _refused(capsys, _args(D) + [flag], "refused")
    _refused(capsys, _args(D) + ["--verbose"], "unknown flag")
    bad = {k: np.zeros(3) for k in DP.ALLOWED_OUTPUT}
    bad["fine_1"] = np.zeros(3, dtype=np.int64)
    with pytest.raises(DP.Refused):
        DP.write_release(tmp_path / "bad.npz", bad)
    z = json.loads(D["pol"].read_text())
    pp = RL.PolicyPair.from_dict(z)
    pp.config["teacher_model_sha256"] = "0" * 64
    for p in pp:
        p.meta["teacher_model_sha256"] = "0" * 64
    RL.save_policy(pp, tmp_path / "wrong.json")
    _refused(capsys, _args(D, **{"--policy": str(tmp_path / "wrong.json")}), "different teacher")
    pp = RL.PolicyPair.from_dict(z)
    pp.config.pop("teacher_model_sha256")
    RL.save_policy(pp, tmp_path / "unbound.json")
    _refused(capsys, _args(D, **{"--policy": str(tmp_path / "unbound.json")}), "not bound")
    zz = dict(z)
    zz["kind"] = "dpc.PolicyPair"
    (tmp_path / "old.json").write_text(json.dumps(zz))
    _refused(capsys, _args(D, **{"--policy": str(tmp_path / "old.json")}), "qpc.PolicyPair")
    for tag in ("lam1", "rate"):
        _refused(capsys, _args(D, **{"--policy": str(D["unreg"][tag])}), "not a registered cbp configuration")
    pp = RL.PolicyPair.from_dict(z)
    pp.config["config"] = "U|JOINT|i8o64|l0.04"                       # config inconsistent with family LOCAL
    RL.save_policy(pp, tmp_path / "relabelled.json")
    _refused(capsys, _args(D, **{"--policy": str(tmp_path / "relabelled.json")}), "not a registered cbp")
