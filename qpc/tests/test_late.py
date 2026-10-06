"""Runner integration on SYNTHETIC data only (temporary directories): Stage A units (A1 parity against a dpc-produced
DIRECT-TASK m8 release, cached recipient fits, 8 rate pairs) and the capacity gate outputs."""
import json

import numpy as np
import pytest

from qpc import run as R
from qpc import utility as UT


@pytest.fixture
def syn(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "UNITS", tmp_path / "units")
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path / "pkg")
    monkeypatch.setattr(R, "DPC_UNITS", tmp_path / "dpc_units")
    (tmp_path / "pkg").mkdir()
    D = UT.synthetic_task_D(n_fit=3000, n_head=200, n_audit=600, n_sel=800, n_assess=600, seed=3)
    D["feature_names"] = np.array([f"f{i}" for i in range(83)])
    T = UT.synthetic_teacher(D, seed=3)
    from dpc import compress as DCP
    from dpc import release as DRL
    tr = np.asarray(D["idx"]["OSF_DEFENSE_FIT"])
    D["idx"]["DEFENSE_FIT"] = tr
    for k in R.SEEDS:
        R.save(f"tea__s{k}__U", {"teacher.npz": {x: T[x] for x in ("row_id", "p1", "p2", "d1", "d2")}},
               {"model_sha256": "ab" * 32})
        pair, _ = DCP.fit_direct_task(T["p1"][tr], T["d1"][tr], T["p2"][tr], T["d2"][tr], D["sex"][tr], 8)
        out = {"row_id": T["row_id"]}
        for i in (1, 2):
            tok, q, dec = DRL.encode(pair[i - 1], T[f"p{i}"], T[f"d{i}"])
            out.update({f"tok{i}": tok, f"q{i}": q, f"hard{i}": dec,
                        f"alpha{i}": np.int64(DRL.alphabet_size(pair[i - 1]))})
        from jcv.finalize import save_unit
        save_unit(R.DPC_UNITS / f"pol__s{k}__U_DIRECT-TASK_m8",
                  {"release.npz": lambda p, o=out: np.savez_compressed(p, **o)}, {})
    return D, tmp_path


def test_stagea_and_gate_synthetic(syn):
    D, tmp = syn
    R.stage_stagea(D, "0/2")
    R.stage_stagea(D, "1/2")
    for k in R.SEEDS:
        a1 = R.rec(f"a1__s{k}")
        assert a1["parity_with_admitted_release"]["ok"]
        for cid in R.stagea_ids():
            z = R.npz(R.unit_for(k, cid), "release.npz")
            T = R.teacher(k)
            assert np.array_equal(z["hard1"], T["d1"]) and np.array_equal(z["hard2"], T["d2"])
    # resume: nothing is refitted
    n_before = len(list((tmp / "units").iterdir()))
    R.stage_stagea(D, None)
    assert len(list((tmp / "units").iterdir())) == n_before
    R.stage_gate(D)
    G = json.loads((tmp / "pkg" / "CAPACITY_GATE.json").read_text())
    assert G["decision"]["gate"] in ("CAPACITY_GATE_MET", "CAPACITY_GATE_NOT_MET")
    assert set(G["configs"]) == set(R.stagea_ids())
    assert len(G["decision"]["selected_rates"]) <= 2
    rows = (tmp / "pkg" / "CONVERGENCE_DIAGNOSTIC.csv").read_text().splitlines()
    assert len(rows) == 1 + 3 * 2 * 2                      # seeds x versions x recipients
    from qpc import report as RP
    cc = RP.capacity_curve(D)
    assert len(cc) == 8 and all(r["fit_kl_occupation_mean"] != "" for r in cc)
    # more occupation capacity never raises the fitted occupation KL in this fixture
    for m1 in (4, 8):
        kl = [r["fit_kl_occupation_mean"] for r in sorted(cc, key=lambda r: r["m2_cap"]) if r["m1_cap"] == m1]
        assert all(b <= a + 1e-12 for a, b in zip(kl, kl[1:]))


def test_stageb_partition_and_fit_synthetic(syn, monkeypatch):
    D, tmp = syn
    monkeypatch.setattr(R, "lock_protocol", lambda: {"stage_b": {"rates": [[8, 64]], "lams": [0.01, 1.0]}})
    D["sex"] = np.where(D["sex"] < 0, 0, D["sex"])
    R.stage_partition(D, "0/2")
    R.stage_partition(D, "1/2")
    assert all(R.done(f"fine__s{k}") for k in R.SEEDS)
    jobs = R.fit_jobs()
    assert len(jobs) == 3 and all(len(c) == 1 + 1 + 2 * 3 + 2 for _, c in jobs)   # CLASS, FINE, 3 fam x 2 lam, JOINT x 2
    R.stage_fit(D, "0/2")
    R.stage_fit(D, "1/2")
    for k in R.SEEDS:
        T = R.teacher(k)
        for cid in R.stage_b_ids():
            n = R.unit_for(k, cid)
            assert R.done(n), n
            z = R.npz(n, "release.npz")
            assert np.array_equal(z["hard1"], T["d1"]) and np.array_equal(z["hard2"], T["d2"])
            r = R.rec(n)
            assert r["config"] == cid
        j = R.rec(R.unit_for(k, "U|JOINT|i8o64|l1"))
        assert j["witness_dominance"]                                     # fitted F_joint <= unchanged witnesses
