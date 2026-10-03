"""Synthetic tests for the official-FARE wrapper (oar/fare_official.py). No study data is read.

Run from the worktree root with the project venv:   ~/PCRL/.venv/bin/python -m pytest oar/tests/test_fare_synthetic.py
The official code runs in the isolated FARE environment (OAR_FARE_PYTHON); tests that need the patched scikit-learn
directly run a short snippet there.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from oar import fare_official as F  # noqa: E402

pytestmark = pytest.mark.skipif(not F.fare_python().exists(), reason="FARE environment not installed")


# --------------------------------------------------------------------------------------------------------------
# synthetic data with planted structure
#   x0: task-only signal; x1: carries s (and, through s, the task); x2..x5: noise.  y = 1[x0 + 0.9 s + e > 0.4]
# --------------------------------------------------------------------------------------------------------------
def make(n, rng, n_groups=2):
    if n_groups == 2:
        s = (rng.rand(n) < 0.4).astype(int)
    else:
        p = np.array([0.45, 0.2, 0.15, 0.12, 0.08])[:n_groups]
        s = rng.choice(n_groups, n, p=p / p.sum())
    X = rng.randn(n, 6)
    X[:, 1] = 1.5 * (s > 0) + 0.3 * s + 0.7 * rng.randn(n)
    y = (X[:, 0] + 0.9 * (s > 0) + 0.5 * rng.randn(n) > 0.4).astype(int)
    return X, y, s


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(0)
    fit_, cert_, ev = make(6000, rng), make(6000, rng), make(6000, rng)
    return {"fit": fit_, "cert": cert_, "eval": ev}


CFG = {"name": "T", "max_leaf_nodes": 16, "min_samples_leaf": 100, "gamma": 0.9, "criterion": "fair_gini_dp"}


@pytest.fixture(scope="module")
def models(data):
    X, y, s = data["fit"]
    fair = F.fit(X, y, s, CFG, seed=0)
    zero = F.fit(X, y, s, F.zero_fairness(CFG), seed=0)
    return {"fair": fair, "zero": zero}


def importance(model):
    """Share of the total weighted impurity decrease attributable to each feature (official tree arrays)."""
    m = model.meta
    imp, n = np.asarray(m["impurity"]), np.asarray(m["n_node_samples"], dtype=float)
    L, R, f = m["children_left"], m["children_right"], m["feature"]
    out = np.zeros(m["n_features"])
    for nd, l in enumerate(L):
        if l != -1:
            out[f[nd]] += n[nd] * imp[nd] - n[l] * imp[l] - n[R[nd]] * imp[R[nd]]
    return out / out.sum()


def split_features(model):
    m = model.meta
    return [f for f, left in zip(m["feature"], m["children_left"]) if left != -1]


def tv_between_groups(cells, s, k):
    p0 = np.bincount(cells[s == 0], minlength=k) / (s == 0).sum()
    p1 = np.bincount(cells[s == 1], minlength=k) / (s == 1).sum()
    return 0.5 * np.abs(p0 - p1).sum()  # = 2 BA* - 1 of the optimal cell adversary = max DP distance of any g


def run_in_fare_env(code: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.json"
        env = dict(os.environ, PYTHONPATH=str(REPO))
        proc = subprocess.run([str(F.fare_python()), "-c", textwrap.dedent(code), str(out)], cwd=str(REPO), env=env,
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-3000:]
        return json.loads(out.read_text())


# --------------------------------------------------------------------------------------------------------------
def test_00_official_installation_verified():
    v = F.verify_official_fare()
    assert v["ok"], v
    assert F.official_tree_sha256() == F.FARE_PY_TREE_SHA256


def test_a_recovers_planted_structure(data, models):
    Xe, ye, se = data["eval"]
    zero, fair = models["zero"], models["fair"]
    # zero-fairness tree: the planted task-relevant features (x0 and the s-carrier x1) carry the impurity decrease
    iz = importance(zero)
    assert iz[0] + iz[1] > 0.98 and iz[1] > 0.1
    # fair tree: keeps the task-only feature at the root and never splits on the s-carrier
    ff = split_features(fair)
    assert ff[0] == 0 and 1 not in ff
    # task signal recovered (Bayes-ish accuracy on x0 alone is ~0.80; with x1 ~0.83)
    assert F.own_task_accuracy(zero, Xe, ye) > 0.82
    assert F.own_task_accuracy(fair, Xe, ye) > 0.78
    # sensitive structure removed by the fair tree, present in the zero-fairness tree
    assert tv_between_groups(F.encode(fair, Xe), se, fair.n_cells) < 0.1
    assert tv_between_groups(F.encode(zero, Xe), se, zero.n_cells) > 0.4


def test_b_zero_fairness_control_direction(data, models):
    Xe, ye, se = data["eval"]
    Xc, _, sc = data["cert"]
    z = F.zero_fairness(CFG)
    assert z["gamma"] == 0.0 and z["max_leaf_nodes"] == CFG["max_leaf_nodes"]
    assert z["min_samples_leaf"] == CFG["min_samples_leaf"] and z["criterion"] == "fair_gini_dp"
    fair, zero = models["fair"], models["zero"]
    assert zero.meta["config"]["gamma"] == 0.0 and fair.meta["config"]["gamma"] == 0.9
    tv_f = tv_between_groups(F.encode(fair, Xe), se, fair.n_cells)
    tv_z = tv_between_groups(F.encode(zero, Xe), se, zero.n_cells)
    assert tv_z > tv_f + 0.3
    assert F.own_task_accuracy(zero, Xe, ye) >= F.own_task_accuracy(fair, Xe, ye)
    cf, cz = F.certificate(fair, Xc, sc), F.certificate(zero, Xc, sc)
    assert cf["status"] == cz["status"] == "OK"
    assert cz["bound"] > cf["bound"] + 0.3


def test_b2_gamma0_is_the_official_gini_tree():
    """Official code path at gamma = 0: FairGiniDP impurity = Gini_y exactly -> identical to criterion='gini'."""
    r = run_in_fare_env("""
        import sys, json, numpy as np
        from sklearn.tree import DecisionTreeClassifier
        from oar.fare_official import _quiet_fd1
        rng = np.random.RandomState(1)
        X = rng.randn(4000, 5); s = rng.randint(0, 3, 4000)
        y = (X[:, 0] + 0.5 * s + rng.randn(4000) > 0.5).astype(int)
        out = {}
        for crit, a in (("fair_gini_dp", 0.0), ("gini", 0.0), ("fair_gini_dp", 0.5)):
            T = DecisionTreeClassifier(criterion=crit, max_leaf_nodes=12, min_samples_leaf=50, random_state=43)
            with _quiet_fd1():
                T.fit(X, y.reshape(-1, 1), s.reshape(-1, 1).astype(float), cat_pos=np.asarray([], np.int32), alpha=a)
            t = T.tree_
            out[f"{crit}_{a}"] = [t.feature.tolist(), t.threshold.tolist(), t.children_left.tolist()]
        json.dump(out, open(sys.argv[1], "w"))
    """)
    assert r["fair_gini_dp_0.0"] == r["gini_0.0"]
    assert r["fair_gini_dp_0.5"] != r["gini_0.0"]


def test_c_encode_is_label_free(data, models):
    fair = models["fair"]
    Xe, ye, se = data["eval"]
    assert list(inspect.signature(F.encode).parameters) == ["model", "X"]
    assert F._ALLOWED_INPUTS["encode"] == {"X"} and F._ALLOWED_INPUTS["embed"] == {"X"}
    cells = F.encode(fair, Xe)
    assert F.LAST_CALL["input_keys"] == ["X"]
    with pytest.raises(PermissionError):
        F._run("encode", {"X": Xe, "y": ye}, {}, fair)
    # features only: no labels or row keys can ride along
    with pytest.raises(ValueError):
        F.encode(fair, np.column_stack([Xe, ye]))
    with pytest.raises(TypeError):
        F.encode(fair, {"X": Xe, "y": ye})
    rec = np.zeros(5, dtype=[("x", float), ("row_id", int)])
    with pytest.raises(TypeError):
        F.encode(fair, rec)
    # a pure per-row function of the features: permutation-equivariant, label permutations irrelevant
    perm = np.random.RandomState(3).permutation(len(Xe))
    assert np.array_equal(F.encode(fair, Xe[perm]), cells[perm])
    assert cells.dtype == np.int32 and cells.min() >= 0 and cells.max() < fair.n_cells
    # official apply == numpy traversal of the stored official tree arrays
    assert np.array_equal(F.encode_portable(fair, Xe), cells)
    # the model holds no per-row label state: every list in meta is per node / per cell / per feature
    n_nodes = len(fair.meta["children_left"])
    for key, val in fair.meta.items():
        if isinstance(val, list):
            assert len(val) in (n_nodes, fair.n_cells, len(fair.meta["task_classes"]),
                                len(fair.meta["group_codes"])), key
    assert len(fair.fit_row_hashes) <= fair.meta["n_fit"]  # feature hashes, read only by certificate()
    # embed = official median representation, constant within a cell
    z = F.embed(fair, Xe[:500])
    med = np.asarray(fair.meta["medians"])
    assert np.array_equal(z, med[cells[:500]])


def test_d_save_load_roundtrip_and_determinism(data, models, tmp_path):
    X, y, s = data["fit"]
    Xe = data["eval"][0]
    fair = models["fair"]
    fair.save(tmp_path / "m")
    back = F.FareModel.load(tmp_path / "m")
    assert back.fingerprint == fair.fingerprint
    assert np.array_equal(F.encode(back, Xe), F.encode(fair, Xe))
    again = F.fit(X, y, s, CFG, seed=0)
    assert again.fingerprint == fair.fingerprint
    assert np.array_equal(F.encode(again, Xe), F.encode(fair, Xe))
    again.save(tmp_path / "m2")
    for f in ("model.json", "tree.pkl", "fit_row_hashes.npy", "manifest.json"):
        assert (tmp_path / "m" / f).read_bytes() == (tmp_path / "m2" / f).read_bytes(), f
    # tampering is detected on load
    (tmp_path / "m2" / "tree.pkl").write_bytes(b"x" + (tmp_path / "m2" / "tree.pkl").read_bytes())
    with pytest.raises(ValueError):
        F.FareModel.load(tmp_path / "m2")


def test_e_certificate_on_separate_rows_and_refused_on_fit_rows(data, models):
    X, y, s = data["fit"]
    Xc, _, sc = data["cert"]
    fair = models["fair"]
    c = F.certificate(fair, Xc, sc, {"delta": 0.05})
    assert c["status"] == "OK" and 0.0 < c["bound"] < 1.0
    assert c["n"] == len(Xc) and c["n_val"] + c["n_test"] == len(Xc)
    assert c["cert_cfg_used"]["delta"] == 0.05 and c["cert_cfg_used"]["groups_resolved"] == [0, 1]
    p = c["pairs"][0]
    assert (p["eps_b"], p["eps_c"], p["eps_s"]) == (0.005, 0.05 - 0.005 - 0.005, 0.005)  # paper/official split
    flags = {q["name"]: q["holds"] for q in c["premises"]}
    assert flags["cert_rows_disjoint_from_fit_rows"] is True and flags["budget_union_bound"] is True
    assert flags["rows_iid_from_target_distribution"] is None  # stated, not checkable
    # the bound dominates the empirical optimal-adversary DP on D_test (it is an upper bound)
    assert c["bound"] >= p["empirical_test"]
    with pytest.raises(F.CertificateRefused):
        F.certificate(fair, X[:3000], s[:3000])
    with pytest.raises(F.CertificateRefused):
        F.certificate(fair, np.vstack([Xc, X[:1]]), np.concatenate([sc, s[:1]]))
    with pytest.raises(ValueError):
        F.certificate(fair, Xc, sc, {"delta": 0.05, "bogus": 1})


def test_e2_multigroup_certificate_pairs_and_budget():
    rng = np.random.RandomState(5)
    (X, y, s), (Xc, _, sc) = make(20000, rng, 5), make(20000, rng, 5)
    m = F.fit(X, y, s, {"max_leaf_nodes": 6, "min_samples_leaf": 1000, "gamma": 0.5}, seed=0)
    assert m.meta["group_codes"] == [0, 1, 2, 3, 4]
    c = F.certificate(m, Xc, sc, {"delta": 0.05})
    assert c["n_pairs"] == 10 and len(c["pairs"]) == 10
    for p in c["pairs"]:
        assert abs(p["eps_pair"] - 0.005) < 1e-15 and p["eps_c"] > 0 and abs(
            p["eps_b"] + p["eps_c"] + p["eps_s"] - p["eps_pair"]) < 1e-15
    if c["status"] == "OK":
        assert c["bound"] == max(p["ub"] for p in c["pairs"])
    else:
        assert c["reason"] and c["bound"] is None
    c3 = F.certificate(m, Xc, sc, {"delta": 0.05, "groups": [0, 1, 2]})
    assert c3["n_pairs"] == 3 and c3["n_outside_groups_dropped"] == int(np.isin(sc, [3, 4]).sum())


def test_e3_unavailable_not_crash_when_a_cell_is_missing(data, models):
    """The official premise 'every cell present in every split of the pair' is reported, never patched."""
    fair = models["fair"]
    Xc, _, sc = data["cert"]
    cells = F.encode(fair, Xc)
    keep = cells != cells[0]  # drop every certificate row of one cell
    c = F.certificate(fair, Xc[keep], sc[keep])
    assert c["status"] == "UNAVAILABLE" and c["bound"] is None and "every cell" in c["reason"]


def test_f_multigroup_fit_matches_reference_objective_and_is_deterministic():
    """5 groups > 2 task classes (the HMDA shape): with the buffer fix the official tree's root split equals a
    brute-force search of the published FairGini objective, and repeated fits are identical."""
    r = run_in_fare_env("""
        import sys, json, numpy as np
        from sklearn.tree import DecisionTreeClassifier
        from oar.fare_official import _quiet_fd1
        rng = np.random.RandomState(2)
        n, G = 3000, 5
        s = rng.choice(G, n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
        X = rng.randn(n, 4); X[:, 1] += 0.6 * s
        y = (X[:, 0] + 0.3 * s + rng.randn(n) > 0.8).astype(int)
        gamma, msl = 0.6, 40
        def gini(counts):
            tot = counts.sum(-1, keepdims=True)
            return 1.0 - ((counts / tot) ** 2).sum(-1)
        best = (-np.inf, None, None)
        for f in range(X.shape[1]):
            xf = X[:, f].astype(np.float32)
            o = np.argsort(xf, kind="mergesort"); xs = xf[o]
            Y = np.eye(2)[y[o]].cumsum(0); S = np.eye(G)[s[o]].cumsum(0)
            for p in range(msl, n - msl + 1):
                if not xs[p - 1] < xs[p]:
                    continue
                yl, yr = Y[p - 1], Y[-1] - Y[p - 1]; sl, sr = S[p - 1], S[-1] - S[p - 1]
                il = (1 - gamma) * gini(yl) + gamma * (0.5 - gini(sl))
                ir = (1 - gamma) * gini(yr) + gamma * (0.5 - gini(sr))
                proxy = -(p * il + (n - p) * ir)
                if proxy > best[0]:
                    best = (proxy, f, float(xs[p - 1]) / 2.0 + float(xs[p]) / 2.0)
        roots, applies = [], set()
        for _ in range(3):
            T = DecisionTreeClassifier(criterion="fair_gini_dp", max_leaf_nodes=2, min_samples_leaf=msl,
                                       random_state=43)
            with _quiet_fd1():
                T.fit(X, y.reshape(-1, 1), s.reshape(-1, 1).astype(float), cat_pos=np.asarray([], np.int32),
                      alpha=gamma)
            roots.append([int(T.tree_.feature[0]), float(T.tree_.threshold[0])])
            applies.add(T.apply(X.astype(np.float32)).tobytes().hex()[:64])
        json.dump({"ref": [best[1], best[2]], "roots": roots, "n_distinct": len(applies),
                   "n_sens": int(T.n_sens_)}, open(sys.argv[1], "w"))
    """)
    assert r["n_sens"] == 5 and r["n_distinct"] == 1
    for f, thr in r["roots"]:
        assert f == r["ref"][0] and abs(thr - r["ref"][1]) < 1e-9


def test_g_fit_encode_cached(data, tmp_path, monkeypatch):
    from stored_model_eval.guards import FitAuthorization, ScientificFitRefused
    monkeypatch.setenv("OAR_RUN_UNITS", str(tmp_path / "units"))
    X, y, s = data["fit"]
    Xall = np.vstack([X, data["eval"][0]])
    led = []
    ledger = lambda uid, cpu, n: led.append((uid, cpu, n))  # noqa: E731
    with pytest.raises(ScientificFitRefused):
        F.fit_encode_cached("u0", X, y, s, Xall, CFG, seed=0, auth=FitAuthorization(), synthetic=True, ledger=ledger)
    assert not (tmp_path / "units" / "u0").exists() and led == []
    auth = FitAuthorization.synthetic_only()
    m, cells, rec = F.fit_encode_cached("u0", X, y, s, Xall, CFG, seed=0, auth=auth, synthetic=True, ledger=ledger)
    assert len(led) == 1 and led[0][0] == "u0" and led[0][2] == 1 and led[0][1] > 0
    for k in ("n_cells", "cfg", "seed", "cpu_s", "fare_commit", "fit_rows_sha256", "n_fit_per_cell"):
        assert k in rec, k
    assert rec["n_cells"] == m.n_cells and sum(rec["n_fit_per_cell"]) == len(X)
    assert len(cells) == len(Xall) and np.array_equal(cells, F.encode(m, Xall))
    comp = json.loads((tmp_path / "units" / "u0" / "COMPLETE.json").read_text())["files"]
    assert {"cells.npy", "rec.json", "model/model.json", "model/tree.pkl"} <= set(comp)
    m2, cells2, rec2 = F.fit_encode_cached("u0", X, y, s, Xall, CFG, seed=0, auth=auth, synthetic=True,
                                           ledger=ledger)
    assert rec2["cache_hit"] and len(led) == 1 and m2.fingerprint == m.fingerprint and np.array_equal(cells2, cells)
    with pytest.raises(RuntimeError):
        F.fit_encode_cached("u0", X, y, s, Xall, F.zero_fairness(CFG), seed=0, auth=auth, synthetic=True,
                            ledger=ledger)


def test_h_own_task_accuracy_is_official_predict(data, models, tmp_path):
    fair = models["fair"]
    Xe, ye, _ = data["eval"]
    fair.save(tmp_path / "m")
    np.save(tmp_path / "X.npy", Xe)
    r = run_in_fare_env(f"""
        import sys, json, numpy as np
        from oar import fare_official as F
        m = F.FareModel.load({str(tmp_path / 'm')!r}); X = np.load({str(tmp_path / 'X.npy')!r})
        T = F._tree_from_pickle(m)
        json.dump({{"pred": T.predict(X.astype(np.float32)).tolist()}}, open(sys.argv[1], "w"))
    """)
    official = float((np.asarray(r["pred"]) == ye).mean())
    assert F.own_task_accuracy(fair, Xe, ye) == official
