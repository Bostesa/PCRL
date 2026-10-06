"""Method tests for qpc (role B: k-means, Stage A, release, deploy; Stage B partition/compress). Synthetic data only;
no Adult row, task label or SEX column is read.

    cd <WORKTREE> && OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q qpc/tests/test_method.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from dpc import compress as DCP
from dpc import partition as DPT
from dpc import release as DRL
from qpc import kmeans as KM
from qpc import release as RL
from qpc import stagea as SA

SHA_T = "a" * 64
SHA_F = "b" * 64


# ----------------------------------------------------------------------------------------------- synthetic data
def synth(N=15434, seed=0, unseen_class=5, underflow=195):
    """Real-shaped synthetic teacher outputs (copied from dpc/tests/test_method.py at 0a7b05a5): income K=2 with exact
    0/1 underflow rows and class imbalance; occupation K=6 with one class never predicted; binary S correlated with
    both scores (used only by Stage B objectives)."""
    rng = np.random.default_rng(seed)
    S = rng.integers(0, 2, N)
    z = rng.normal(-1.4 + 0.9 * S, 1.6, N)
    p = 1 / (1 + np.exp(-z))
    P1 = np.stack([1 - p, p], 1)
    if underflow:
        u = rng.choice(N, underflow, replace=False)
        P1[u[: underflow // 2]] = [1.0, 0.0]
        P1[u[underflow // 2:]] = [0.0, 1.0]
    bias = np.array([0.0, 0.6, 0.1, -1.2, 1.0, 0.2])
    if unseen_class is not None:
        bias[unseen_class] = -14.0
    L = rng.normal(size=(N, 6)) * 1.2 + bias + np.outer(S - 0.5, [0.8, -0.6, 0.3, 0.0, -0.4, 0.0])
    E = np.exp(L - L.max(1, keepdims=True))
    P2 = E / E.sum(1, keepdims=True)
    return P1, P2, S


def teacher_dict(N=3000, seed=1, n_fit=None):
    P1, P2, S = synth(N=N, seed=seed, underflow=max(N // 80, 2))
    T = {"row_id": np.arange(N, dtype=np.int64) * 7 + 3, "p1": P1, "p2": P2, "d1": P1.argmax(1), "d2": P2.argmax(1)}
    tr = np.arange(n_fit if n_fit else int(N * 0.7))
    return T, tr, S


def adversarial_rows(K, rng, n_random=200):
    """Ties at every pair of positions, one-hot rows, 1e-300 components, uniform, random Dirichlet rows."""
    rows = []
    for a, b in itertools.combinations(range(K), 2):
        r = np.zeros(K)
        r[a] = r[b] = 0.5
        rows.append(r)
        r = np.full(K, 0.1 / max(K - 2, 1)) if K > 2 else np.zeros(K)
        r[a] = r[b] = (1 - r.sum() + r[a] + r[b]) / 2
        rows.append(r / r.sum())
    rows += list(np.eye(K))
    rows.append(np.full(K, 1.0 / K))
    for c in range(K):
        r = np.full(K, 1e-300)
        r[c] = 1.0 - (K - 1) * 1e-300
        rows.append(r)
    rows += list(rng.dirichlet(np.full(K, 0.3), n_random))
    P = np.array(rows, dtype=np.float64)
    return P / P.sum(1, keepdims=True)


META = {"teacher": "U", "seed": 0, "teacher_model_sha256": SHA_T, "feature_names_sha256": SHA_F}


@pytest.fixture(scope="module")
def tdata():
    return teacher_dict()


@pytest.fixture(scope="module")
def fits(tdata):
    T, tr, _ = tdata
    out = {}
    for r, K, m in ((1, 2, 4), (1, 2, 8), (2, 6, 8), (2, 6, 64)):
        out[(r, m)] = SA.recipient_fit(T[f"p{r}"][tr], T[f"d{r}"][tr], K, m, r)
    return out


# ----------------------------------------------------------------------------------------------- k-means starts
def test_source_init_equals_dpc_rule():
    rng = np.random.default_rng(0)
    for K in (2, 6):
        P = rng.dirichlet(np.ones(K), 400)
        c = 0
        for k in (1, 3, 8, 64):
            C, nU, pos = KM.source_init(P, c, min(k, 400))
            Cd, nUd, posd = DPT._init_centroids(P, c, min(k, 400))
            assert np.array_equal(C, Cd) and nU == nUd and pos == posd


def _kpp_reference(Pc, c, k, seed, K):
    """Independent re-statement of the registered KL k-means++ rule (for the test only)."""
    U, w = np.unique(Pc, axis=0, return_counts=True)
    rng = np.random.default_rng([seed, K, c])
    chosen = []
    for j in range(k):
        if j == 0:
            v = w.astype(float)
        else:
            Q = np.stack([DPT.smooth(U[i], c) for i in chosen])
            D = DPT.kl_matrix(U, Q)
            D = np.where((D < 0) & (D >= -1e-12), 0.0, D)
            v = w * D.min(1)
            v[chosen] = 0.0
            if v.sum() <= 0:
                v = w.astype(float)
                v[chosen] = 0.0
        cum = np.cumsum(v)
        t = rng.random() * cum[-1]
        i = int(np.searchsorted(cum, t, side="right"))
        if i >= len(v):
            i = int(np.flatnonzero(v > 0)[-1])
        chosen.append(i)
    return chosen


def test_kpp_registered_rule_with_multiplicities():
    rng = np.random.default_rng(3)
    base = rng.dirichlet(np.ones(6), 40)
    base = base[base.argmax(1) == base.argmax(1)[0]]
    c = int(base[0].argmax())
    reps = rng.integers(1, 30, base.shape[0])
    Pc = np.repeat(base, reps, axis=0)[rng.permutation(int(reps.sum()))]
    for seed in (20261006, 20261007):
        k = min(5, base.shape[0])
        C, nU, info = KM.kpp_init(Pc, c, k, seed, 6)
        ref = _kpp_reference(Pc, c, k, seed, 6)
        assert info["chosen_distinct_index"] == ref
        assert len(set(ref)) == k and nU == base.shape[0]
        U, w = np.unique(Pc, axis=0, return_counts=True)
        assert info["chosen_multiplicity"] == [int(w[i]) for i in ref]
        assert np.array_equal(C, U[ref])
    # the first centre is multiplicity weighted: a vector carrying 99% of the rows is picked almost always
    heavy = np.vstack([np.repeat(base[:1], 990, 0), base[1:]])
    firsts = [KM.kpp_init(heavy, c, 1, s, 6)[2]["chosen_distinct_index"][0] for s in range(200)]
    U = np.unique(heavy, axis=0)
    hi = int(np.flatnonzero((U == base[0]).all(1))[0])
    assert sum(f == hi for f in firsts) >= 180


def test_kl_guard_clips_roundoff_and_raises(monkeypatch):
    M = np.array([[0.1, -5e-13, 0.0], [-1e-12, 0.2, 0.3]])
    monkeypatch.setattr(KM, "kl_matrix", lambda P, Q: M.copy())
    out, n = KM.guarded_kl(None, None)
    assert n == 2 and out.min() == 0.0 and out[0, 0] == 0.1
    monkeypatch.setattr(KM, "kl_matrix", lambda P, Q: np.array([[-2e-12]]))
    with pytest.raises(ValueError, match="not roundoff"):
        KM.guarded_kl(None, None)


def test_assignment_never_clips():
    """The guard only touches initialisation probabilities: assignments use the raw KL argmin."""
    import inspect
    src = inspect.getsource(KM.kmeans_class)
    assert "guarded_kl" not in src and "kl_matrix(Pc, Qm).argmin(1)" in src


def test_multiple_starts_and_deterministic_selection(tdata):
    T, tr, _ = tdata
    P, d = T["p2"][tr], T["d2"][tr]
    f1 = KM.fit_recipient(P, d, 6, 16)
    f2 = KM.fit_recipient(P, d, 6, 16)
    assert f1.partition.fingerprint() == f2.partition.fingerprint()
    assert json.dumps(f1.receipt["per_class"], sort_keys=True) == json.dumps(f2.receipt["per_class"], sort_keys=True)
    winners = set()
    for pc in f1.receipt["per_class"]:
        if pc["fallback"]:
            continue
        objs = [r["objective"] for r in pc["starts"]]
        assert [r["start"] for r in pc["starts"]] == list(KM.STARTS)
        w = 0
        for i in range(1, 3):
            if objs[i] < objs[w] - 1e-12 * max(abs(objs[w]), 1.0):
                w = i
        assert pc["winner"] == KM.STARTS[w] and pc["winner_objective"] == objs[w]
        assert objs[w] <= min(objs) + 1e-12 * max(abs(min(objs)), 1)
        winners.add(pc["winner"])
        inits = [json.dumps(r["init"].get("chosen_distinct_index", r["init"].get("positions"))) for r in pc["starts"]]
        assert len(set(inits)) == 3                                    # three genuinely different starts
    # per-class winners assembled: winner partition cells of each class equal the winning start's cells
    for pc in f1.receipt["per_class"]:
        c = pc["class"]
        if pc["fallback"]:
            continue
        fw, fs = f1.partition, f1.by_start[pc["winner"]]
        assert np.array_equal(fw.centroid[fw.cells_of(c)], fs.centroid[fs.cells_of(c)])
    # tie rule: an exact tie keeps the earlier start
    recs = [{"objective": 1.0}, {"objective": 1.0}, {"objective": 1.0 - 1e-14}]
    assert KM.select_start(recs) == 0
    recs[2]["objective"] = 0.9
    assert KM.select_start(recs) == 2


def test_selection_ignores_labels(tdata):
    """No label argument exists; a true-label or SEX array in place of the decision is refused."""
    T, tr, S = tdata
    P = T["p1"][tr]
    with pytest.raises(ValueError, match="teacher decision"):
        KM.fit_recipient(P, S[tr], 2, 4)
    import inspect
    assert set(inspect.signature(KM.fit_recipient).parameters) == {"P", "d", "K", "m", "starts", "rounds", "rule",
                                                                    "tag"}


# ----------------------------------------------------------------------------------------------- convergence/coherence
def test_convergence_and_coherent_statistics(tdata):
    T, tr, _ = tdata
    for r, K, m in ((1, 2, 8), (2, 6, 32)):
        P, d = T[f"p{r}"][tr], T[f"d{r}"][tr]
        fit = KM.fit_recipient(P, d, K, m)
        pol = RL.make_policy(r, fit.partition)
        cell = DPT.assign_fine(P, d, fit.partition)
        n, S, A = DPT.cell_stats(P, cell, fit.partition.F)
        assert np.array_equal(n, fit.partition.n) and np.array_equal(S, fit.partition.S)
        assert np.array_equal(A, fit.partition.A)
        # the objective receipt equals the deployed policy's fitting distortion (row-level brute force)
        tok, q, _ = RL.encode(pol, P, d)
        D_rows = float(np.mean(DPT.kl_rows(P, q)))
        assert abs(D_rows - SA.fit_distortion(pol)) < 1e-12
        assert abs(D_rows * P.shape[0] - fit.receipt["objective_total"]) < 1e-9
        for pc in fit.receipt["per_class"]:
            for rec in pc["starts"]:
                tj = np.array(rec["objective_trajectory"])
                assert np.all(np.diff(tj) <= 1e-10 * np.abs(tj[:-1]))      # Lloyd monotone up to smoothing roundoff
                assert rec["objective"] == tj.min() and rec["returned_pass"] >= 1
                assert tj[rec["returned_pass"] - 1] == rec["objective"]
                assert rec["stop_reason"] in ("assignment_fixed_point", "relative_tolerance", "cap")
                assert rec["converged"] == (rec["stop_reason"] != "cap")
                assert rec["work"]["assign_passes"] == len(tj)
            if not pc["fallback"]:
                w = next(x for x in pc["starts"] if x["start"] == pc["winner"])
                if w["stop_reason"] == "assignment_fixed_point":
                    idx = fit.partition.cells_of(pc["class"])
                    cen = fit.partition.centroid[idx]
                    proto = DPT.smooth(fit.partition.mean[idx], np.full(idx.size, pc["class"]))
                    assert np.max(np.abs(cen - proto)) <= 1e-15             # fixed point: centroid = prototype


def test_tolerance_and_cap_labels(tdata):
    T, tr, _ = tdata
    P, d = T["p2"][tr], T["d2"][tr]
    c = 1
    Pc = P[d == c]
    Q, a, rec = KM.kmeans_class(Pc, c, 16, "source", rounds=2)
    assert rec["stop_reason"] == "cap" and not rec["converged"] and rec["work"]["assign_passes"] == 3
    Q, a, rec = KM.kmeans_class(Pc, c, 16, "source", rounds=200, rtol=1.0)
    assert rec["stop_reason"] == "relative_tolerance" and rec["converged"]
    assert rec["rounds_used"] == 4 and rec["work"]["assign_passes"] == 5 and rec["work"]["updates"] == 4
    # the returned iterate is coherent: a is the deployment of Q
    assert np.array_equal(DPT.kl_matrix(Pc, Q).argmin(1), a)
    Q, a, rec = KM.kmeans_class(Pc, c, 16, "source", rounds=200)
    assert rec["stop_reason"] == "assignment_fixed_point" and rec["rounds_used"] < 200


def test_empty_cells_kept_then_removed_and_reported():
    rng = np.random.default_rng(5)
    p0 = rng.uniform(0.75, 0.85, 300)
    r = rng.uniform(0.3, 0.7, 300)
    Pc = np.stack([p0, (1 - p0) * r, (1 - p0) * (1 - r)], 1)
    init = np.stack([Pc[0], np.array([0.34, 0.33, 0.33]), Pc[1]])   # cell 1 is never the nearest centroid
    Q, a, rec = KM.kmeans_class(Pc, 0, 3, rounds=50, init_override=init)
    assert rec["empty_cell_events"] == len(rec["empty_cells_per_pass"]) and rec["empty_cells_per_pass"][0] == 1
    assert np.array_equal(Q[1], DPT.smooth(init[1], 0))               # empty cell keeps its previous centroid
    assert 1 not in set(a.tolist()) and rec["empty_in_returned"] == 1
    cb, removed = KM._cells(Pc, 0, Q, a)
    assert removed == 1 and cb["n"].size == 2 and np.all(cb["n"] > 0)
    # removing the empty cell leaves the deployment of the training rows unchanged
    assert np.array_equal(DPT.kl_matrix(Pc, cb["cen"]).argmin(1), np.where(a > 1, 1, 0))


def test_absent_class_fallback_and_unseen_rows(tdata):
    T, tr, _ = tdata
    P, d = T["p2"][tr], T["d2"][tr]
    assert not np.any(d == 5)
    fit = KM.fit_recipient(P, d, 6, 8)
    idx = fit.partition.cells_of(5)
    assert idx.size == 1 and fit.partition.fallback[idx[0]] and fit.partition.n[idx[0]] == 0
    assert np.array_equal(fit.partition.mean[idx[0]], np.full(6, 1 / 6))
    assert np.array_equal(fit.partition.centroid[idx[0]], DPT.smooth(np.full(6, 1 / 6), 5))
    assert fit.receipt["fallback_classes"] == [5]
    # matches dpc's reserved fallback cell exactly
    ref = DPT.fit_fine(P, d, 6, max_cells=8, rounds=20)
    j = ref.cells_of(5)[0]
    assert np.array_equal(ref.centroid[j], fit.partition.centroid[idx[0]])


# ----------------------------------------------------------------------------------------------- capacity
def test_larger_capacity_creates_new_cells(fits, tdata):
    T, tr, _ = tdata
    pol8 = RL.Policy.from_dict(fits[(2, 8)][0])
    pol64 = RL.Policy.from_dict(fits[(2, 64)][0])
    d = T["d2"][tr]
    big = 0
    for c in range(5):
        nrows = int(np.sum(d == c))
        nU = np.unique(T["p2"][tr][d == c], axis=0).shape[0]
        pc = next(x for x in fits[(2, 64)][1]["per_class"] if x["class"] == c)
        w = next(s for s in pc["starts"] if s["start"] == pc["winner"])
        assert w["initial_cells"] == min(64, nU, nrows)
        assert pol64.tokens_per_class()[c] == w["initial_cells"] - w["removed_empty_cells"]
        assert pol64.effective_states_per_class()[c] >= min(60, nU)
        big = big + 1 if nU >= 64 else big
        assert pol8.tokens_per_class()[c] <= min(8, nU, nrows)
    assert big >= 3 and max(pol64.tokens_per_class()) == 64
    assert fits[(2, 64)][1]["fit_distortion"] < fits[(2, 8)][1]["fit_distortion"]
    # cap above the distinct-vector count: k = #distinct
    P = np.repeat(np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]), 5, 0)
    f = KM.fit_recipient(P, None, 2, 64)
    assert f.partition.cells_of(0).size == 3


def test_asymmetric_caps_and_config_ids(fits, tdata):
    T, tr, _ = tdata
    rec, files = SA.pair_unit(fits[(1, 4)][0], fits[(2, 64)][0], T, {**META, "config": "U|DIRECT-TASK|i4o64"}, tr=tr)
    pair = RL.PolicyPair.from_dict(json.loads(_write_read(files["policy.json"])))
    assert pair.config["m1"] == 4 and pair.config["m2"] == 64 and pair.config["config"] == "U|DIRECT-TASK|i4o64"
    assert max(pair.p1.tokens_per_class()) <= 4 and max(pair.p2.tokens_per_class()) <= 64
    assert max(pair.p2.tokens_per_class()) > 8
    assert rec["alpha1"] == pair.p1.T and rec["alpha2"] == pair.p2.T
    with pytest.raises(ValueError, match="caps"):
        RL.make_pair(RL.Policy.from_dict(fits[(1, 8)][0]), RL.Policy.from_dict(fits[(2, 8)][0]), "DIRECT-TASK", 4, 8)
    assert SA.config_id("U", "DIRECT-TASK", 8, 32) == "U|DIRECT-TASK|i8o32"
    assert SA.config_id("U", "JOINT", 8, 64, 0.01) == "U|JOINT|i8o64|l0.01"
    assert SA.config_id("U", "LOCAL", 8, 64, 1.0) == "U|LOCAL|i8o64|l1"
    assert SA.config_id("U", "CLASS", 1, 1) == "U|CLASS|i1o1"
    assert len(SA.rate_pairs()) == 8


def _write_read(writer):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.json"
        writer(p)
        return p.read_text()


# ----------------------------------------------------------------------------------------------- class preservation
def test_class_preservation_ties_tiny_unseen(fits):
    rng = np.random.default_rng(11)
    for (r, m), (pd, _) in fits.items():
        pol = RL.Policy.from_dict(pd)
        K = pol.K
        P = adversarial_rows(K, rng)
        tok, q, dec = RL.encode(pol, P)
        assert np.array_equal(dec, P.argmax(1))                      # first-index ties
        assert np.max(np.abs(q.sum(1) - 1)) <= 1e-12
        for i in range(P.shape[0]):
            assert all(q[i, dec[i]] > q[i, k] for k in range(K) if k != dec[i])
        if K == 6:
            u = dec == 5
            assert u.any() and np.all(pol.token_fallback[tok[u]])     # unseen class -> reserved fallback token
            assert np.array_equal(q[u], np.repeat(DPT.smooth(np.full(6, 1 / 6), 5)[None], u.sum(), 0))


def test_smoothing_formula_on_release(fits):
    pol = RL.Policy.from_dict(fits[(2, 8)][0])
    for t in range(pol.T):
        c = pol.token_class[t]
        mean = pol.token_S[t] / pol.token_n[t] if pol.token_n[t] else np.full(6, 1 / 6)
        e = np.zeros(6)
        e[c] = 1e-12
        assert np.array_equal(pol.token_proto[t], (mean + 1e-12 + e) / (1 + 7e-12))


def test_refuses_non_teacher_decisions(tdata):
    T, tr, S = tdata
    with pytest.raises(ValueError):
        SA.recipient_fit(T["p2"][tr], (T["d2"][tr] + 1) % 6, 6, 8, 2)
    with pytest.raises(ValueError):
        SA.recipient_fit(T["p2"][tr], T["d2"][tr], 2, 8, 2)            # wrong K for the recipient


# ----------------------------------------------------------------------------------------------- tokens
def test_token_collision_keeps_identity():
    q = DPT.smooth(np.array([0.8, 0.2]), 0)
    f = KM.FinePartition(K=2, cell_class=np.array([0, 0, 1]), centroid=np.stack([q, q, DPT.smooth([0.3, 0.7], 1)]),
                         mean=np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]]), n=np.array([2, 2, 4]),
                         S=np.array([[1.6, 0.4], [1.6, 0.4], [1.2, 2.8]]), A=np.zeros(3), fallback=np.zeros(3, bool))
    pol = RL.make_policy(1, f)
    assert pol.T == 3 and np.array_equal(pol.token_proto[0], pol.token_proto[1])
    v = DRL.recipient_view(pol, np.array([[0.8, 0.2]]))
    assert v["token_onehot"].shape[1] == 3 and v["token_onehot"][0, 0] == 1 and v["token_onehot"][0, 1] == 0


def test_relabelling_invariance_and_parity(fits, tdata):
    T, tr, _ = tdata
    pol = RL.Policy.from_dict(fits[(2, 8)][0])
    P = T["p2"]
    perm = np.random.default_rng(2).permutation(pol.T)
    oh = DRL.recipient_view(pol, P)["token_onehot"]
    oh2, _, _ = DRL.renumbered_view(pol, perm, P)
    assert np.array_equal(oh2[:, perm], oh)
    with pytest.raises(ValueError, match="canonical"):
        RL.Policy(recipient=2, fine=pol.fine, cell_token=perm[pol.cell_token])
    tok, q, _ = RL.encode(pol, P)
    par = RL.token_parity(tok, q, perm[tok], q)
    assert par["token_bijection"] and not par["tokens_bitwise_equal"] and par["q_bitwise_equal"] and par["ok"]
    merged = tok.copy()
    merged[merged == 1] = 0
    assert not RL.token_parity(merged, q, tok, q)["token_bijection"]
    # tokens never cross classes
    labels = np.zeros(pol.fine.F, dtype=np.int64)
    with pytest.raises(ValueError, match="mixes predicted classes"):
        RL.make_policy(2, pol.fine, labels)


# ----------------------------------------------------------------------------------------------- serialisation
def test_save_restore_identical_assignments(fits, tdata, tmp_path):
    T, tr, _ = tdata
    rec, files = SA.pair_unit(fits[(1, 8)][0], fits[(2, 64)][0], T, {**META, "config": "U|DIRECT-TASK|i8o64"})
    (tmp_path / "policy.json").write_text(_write_read(files["policy.json"]))
    pair = RL.load_policy(tmp_path / "policy.json")
    re = RL.release_arrays(pair, T["row_id"], T["p1"], T["d1"], T["p2"], T["d2"])
    for k, v in files["release.npz"].items():
        assert np.array_equal(np.asarray(v), np.asarray(re[k])), k
    assert pair.fingerprint() == rec["fingerprint"]
    np.savez(tmp_path / "rel.npz", **files["release.npz"])
    with np.load(tmp_path / "rel.npz", allow_pickle=False) as z:
        assert sorted(z.files) == sorted(["row_id", "tok1", "q1", "hard1", "alpha1", "tok2", "q2", "hard2", "alpha2"])
    z = json.loads((tmp_path / "policy.json").read_text())
    z["p2"]["token_proto"][0][0] += 1e-9
    with pytest.raises(ValueError):
        RL.PolicyPair.from_dict(z)
    z = json.loads((tmp_path / "policy.json").read_text())
    z["kind"] = "dpc.PolicyPair"
    with pytest.raises(ValueError, match="not a qpc"):
        RL.from_dict(z)
    z = json.loads((tmp_path / "policy.json").read_text())
    z["p1"]["fine"]["centroid"][0][0] += 1e-6
    with pytest.raises(ValueError):
        RL.PolicyPair.from_dict(z)
    # unbound metadata is refused at assembly
    with pytest.raises(ValueError, match="bind"):
        SA.pair_unit(fits[(1, 8)][0], fits[(2, 64)][0], T, {"teacher": "U"})


def test_pair_unit_record(fits, tdata):
    T, tr, _ = tdata
    rec, files = SA.pair_unit(fits[(1, 8)][0], fits[(2, 8)][0], T, {**META, "config": "U|DIRECT-TASK|i8o8"}, tr=tr)
    KM.json_safe(rec)
    out = files["release.npz"]
    assert np.array_equal(out["hard1"], T["d1"]) and np.array_equal(out["hard2"], T["d2"])
    assert rec["states_emitted_fit"][2] == sum(rec["r2"]["effective_states_per_class"])
    assert rec["r2"]["states_emitted_fit"] == rec["states_emitted_fit"][2]
    assert len(rec["r2"]["cell_population_per_class"]) == 6
    with pytest.raises(ValueError, match="differs"):
        SA.pair_unit(fits[(1, 8)][0], fits[(2, 8)][0], T, {**META, "config": "U|DIRECT-TASK|i4o8"})


# ----------------------------------------------------------------------------------------------- A1
@pytest.mark.parametrize("m,rounds", [(4, 5), (8, 20), (16, 20), (8, 3)])
def test_dpc_rule_equals_dpc_fit_fine(tdata, m, rounds):
    T, tr, _ = tdata
    for r, K in ((1, 2), (2, 6)):
        P, d = T[f"p{r}"][tr], T[f"d{r}"][tr]
        mine = KM.fit_recipient(P, d, K, m, starts=("source",), rounds=rounds, rule="dpc").partition
        ref = DPT.fit_fine(P, d, K, max_cells=m, rounds=rounds)
        assert mine.fingerprint() == ref.fingerprint()
        assert np.array_equal(mine.mean, ref.mean) and np.array_equal(mine.fallback, ref.fallback)


def _admitted_like_release(T, tr):
    """What dpc wrote for pol__s{k}__U_DIRECT-TASK_m8: dpc.compress.fit_direct_task + dpc.release.encode, all rows."""
    S = np.zeros(len(tr), dtype=int)
    pair, _ = DCP.fit_direct_task(T["p1"][tr], T["d1"][tr], T["p2"][tr], T["d2"][tr], S, 8)
    out = {"row_id": T["row_id"]}
    for i in (1, 2):
        tok, q, dec = DRL.encode(pair[i - 1], T[f"p{i}"], T[f"d{i}"])
        out.update({f"tok{i}": tok, f"q{i}": q, f"hard{i}": dec, f"alpha{i}": np.int64(pair[i - 1].T)})
    return out, pair


def test_a1_unit_parity_and_refit(tdata):
    T, tr, _ = tdata
    src, dpc_pair = _admitted_like_release(T, tr)
    rec, files = SA.a1_unit(T, tr, src, {**META, "config": "U|DIRECT-TASK|i8o8"})
    KM.json_safe(rec)
    par = rec["parity_with_admitted_release"]
    assert par["ok"] and par["ids_rule"] == "exact_ids" and par["q_rule"] == "bitwise"
    assert set(files) == {"release_src20.npz", "release_r200.npz", "policy_src20.json", "policy_r200.json"}
    for k in ("tok1", "q1", "tok2", "q2", "hard1", "hard2"):
        assert np.array_equal(files["release_src20.npz"][k], src[k])
    p20 = RL.PolicyPair.from_dict(json.loads(_write_read(files["policy_src20.json"])))
    assert p20.p1.fingerprint() == dpc_pair.p1.fingerprint() and p20.p2.fingerprint() == dpc_pair.p2.fingerprint()
    assert rec["rounds"] == {"src20": 20, "r200": 200}
    for r in (1, 2):
        pcs = rec["r200"]["per_recipient"][r]["per_class"]
        for pc in pcs:
            if not pc["fallback"]:
                assert [s["start"] for s in pc["starts"]] == ["source"]
    # A2 restart receipt: the source start of the A2 m=8 fit IS the A1 r200 refit (same computation, no extra fit)
    for r, K in ((1, 2), (2, 6)):
        f = KM.fit_recipient(T[f"p{r}"][tr], T[f"d{r}"][tr], K, 8)
        assert f.by_start["source"].fingerprint() == rec["r200"]["per_recipient"][r]["partition_fingerprint"]
    # a canonically relabelled admitted release still passes, labelled as a bijection
    src2 = dict(src)
    perm = np.random.default_rng(0).permutation(int(src["alpha2"]))
    src2["tok2"] = perm[src["tok2"]]
    rec2, _ = SA.a1_unit(T, tr, src2, {**META, "config": "U|DIRECT-TASK|i8o8"})
    assert rec2["parity_with_admitted_release"]["ok"]
    assert rec2["parity_with_admitted_release"]["ids_rule"] == "bijection"
    # a genuinely different release is flagged as an engineering blocker
    src3 = dict(src)
    src3["q1"] = src["q1"] + 1e-9
    rec3, _ = SA.a1_unit(T, tr, src3, {**META, "config": "U|DIRECT-TASK|i8o8"})
    assert not rec3["parity_with_admitted_release"]["ok"] and "ENGINEERING_BLOCKER" in rec3


# ----------------------------------------------------------------------------------------------- deploy
def _synthetic_unit(tmp_path, seed=0):
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
    unit = tmp_path / "rel__s0__SYNTH"
    state = model.state_dict()
    save_unit(unit, {"model.pt": lambda p: torch.save(state, p),
                     "head_0.joblib": lambda p: joblib.dump(heads[0], p),
                     "head_1.joblib": lambda p: joblib.dump(heads[1], p)}, {"seed": seed})
    return unit, X, names


@pytest.fixture(scope="module")
def deployed(tmp_path_factory):
    from qpc import deploy as DP
    tmp = tmp_path_factory.mktemp("deploy")
    unit, X, names = _synthetic_unit(tmp)
    model, heads, msha = DP.load_teacher(unit)
    P1, P2 = DP.teacher_probs(model, heads, X)
    T = {"row_id": np.arange(len(X)), "p1": P1, "p2": P2, "d1": P1.argmax(1), "d2": P2.argmax(1)}
    tr = np.arange(800)
    pd1, _ = SA.recipient_fit(P1[tr], T["d1"][tr], 2, 4, 1)
    pd2, _ = SA.recipient_fit(P2[tr], T["d2"][tr], 6, 16, 2)
    meta = {"teacher": "U", "seed": 0, "config": "U|DIRECT-TASK|i4o16", "teacher_model_sha256": msha,
            "feature_names_sha256": DP.schema_sha256(names)}
    rec, files = SA.pair_unit(pd1, pd2, T, meta)
    pol = tmp / "policy.json"
    files["policy.json"](pol)
    np.savez(tmp / "schema.npz", feature_names=np.array(names))
    np.savez(tmp / "in.npz", X=X, feature_names=np.array(names))
    return dict(tmp=tmp, unit=unit, X=X, names=names, T=T, pol=pol, msha=msha, rel=files["release.npz"])


def _args(D, **kw):
    a = {"--unit": str(D["unit"]), "--policy": str(D["pol"]), "--X": str(D["tmp"] / "in.npz"),
         "--schema": str(D["tmp"] / "schema.npz"), "--out": str(D["tmp"] / "out.npz")}
    a.update(kw)
    out = []
    for k, v in a.items():
        out += [k] if v is True else [k, v]
    return out


def _expect_refusal(D, capsys, args, needle):
    from qpc import deploy as DP
    with pytest.raises(SystemExit) as e:
        DP.main(args)
    assert e.value.code == 2
    assert needle in capsys.readouterr().err


def test_deploy_end_to_end_matches_stored(deployed, tmp_path):
    from qpc import deploy as DP
    D = deployed
    DP.main(_args(D))
    with np.load(D["tmp"] / "out.npz", allow_pickle=False) as z:
        assert sorted(z.files) == sorted(DP.ALLOWED_OUTPUT)
        for i in (1, 2):
            assert np.array_equal(z[f"tokens_{i}"], D["rel"][f"tok{i}"])
            assert np.array_equal(z[f"probs_{i}"], D["rel"][f"q{i}"])
            assert np.array_equal(z[f"decision_{i}"], D["rel"][f"hard{i}"])
    # restore: a copied policy file deploys identically
    import shutil
    shutil.copy(D["pol"], tmp_path / "restored.json")
    DP.main(_args(D, **{"--policy": str(tmp_path / "restored.json"), "--out": str(tmp_path / "o2.npz")}))
    with np.load(tmp_path / "o2.npz") as a, np.load(D["tmp"] / "out.npz") as b:
        assert all(np.array_equal(a[k], b[k]) for k in DP.ALLOWED_OUTPUT)


def test_deploy_refuses_schema_violations(deployed, capsys):
    D = deployed
    X, names = D["X"], D["names"]
    np.savez(D["tmp"] / "extra.npz", X=np.hstack([X, X[:, :1]]), feature_names=np.array(names + ["SEX"]))
    _expect_refusal(D, capsys, _args(D, **{"--X": str(D["tmp"] / "extra.npz")}), "refused")
    np.savez(D["tmp"] / "missing.npz", X=X[:, :82], feature_names=np.array(names[:82]))
    _expect_refusal(D, capsys, _args(D, **{"--X": str(D["tmp"] / "missing.npz")}), "refused")
    perm = list(range(83))
    perm[3], perm[4] = perm[4], perm[3]
    np.savez(D["tmp"] / "reorder.npz", X=X[:, perm], feature_names=np.array([names[j] for j in perm]))
    _expect_refusal(D, capsys, _args(D, **{"--X": str(D["tmp"] / "reorder.npz")}), "reordered")
    np.savez(D["tmp"] / "bundled.npz", X=X, feature_names=np.array(names), sex=np.zeros(len(X)))
    _expect_refusal(D, capsys, _args(D, **{"--X": str(D["tmp"] / "bundled.npz")}), "exactly X and feature_names")
    _expect_refusal(D, capsys, _args(D, **{"--schema-sha256": "0" * 64}), "schema hash mismatch")


def test_deploy_refuses_exports_and_unknown_flags(deployed, capsys):
    D = deployed
    for flag in ("--export-fine-ids", "--fine-cells", "--raw-scores", "--logits", "--teacher-probs", "--export=all",
                 "--debug", "--allow-unbound-policy", "--continuous"):
        _expect_refusal(D, capsys, _args(D) + [flag], "refused")
    _expect_refusal(D, capsys, _args(D) + ["--verbose"], "unknown flag")
    from qpc import deploy as DP
    bad = {k: np.zeros(3) for k in DP.ALLOWED_OUTPUT}
    bad["fine_1"] = np.zeros(3, dtype=np.int64)
    with pytest.raises(DP.Refused):
        DP.write_release(D["tmp"] / "bad.npz", bad)


def test_deploy_refuses_mismatched_or_old_policies(deployed, capsys, tmp_path):
    D = deployed
    z = json.loads(D["pol"].read_text())
    pp = RL.PolicyPair.from_dict(z)
    pp.config["teacher_model_sha256"] = "0" * 64
    for p in pp:
        p.meta["teacher_model_sha256"] = "0" * 64
    RL.save_policy(pp, tmp_path / "wrong.json")
    _expect_refusal(D, capsys, _args(D, **{"--policy": str(tmp_path / "wrong.json")}), "different teacher")
    pp = RL.PolicyPair.from_dict(z)
    pp.config["feature_names_sha256"] = "1" * 64
    for p in pp:
        p.meta["feature_names_sha256"] = "1" * 64
    RL.save_policy(pp, tmp_path / "schema.json")
    _expect_refusal(D, capsys, _args(D, **{"--policy": str(tmp_path / "schema.json")}), "different pinned feature")
    pp = RL.PolicyPair.from_dict(z)
    pp.config.pop("teacher_model_sha256")
    RL.save_policy(pp, tmp_path / "unbound.json")
    _expect_refusal(D, capsys, _args(D, **{"--policy": str(tmp_path / "unbound.json")}), "not bound")
    zz = dict(z)
    zz["kind"] = "dpc.PolicyPair"
    (tmp_path / "old.json").write_text(json.dumps(zz))
    _expect_refusal(D, capsys, _args(D, **{"--policy": str(tmp_path / "old.json")}), "qpc.PolicyPair")


# =============================================================================================== Stage B
from qpc import compress as CP  # noqa: E402
from qpc import partition as PA  # noqa: E402


@pytest.fixture(scope="module")
def bsmall():
    P1, P2, S = synth(N=2500, seed=3, underflow=40)
    d1, d2 = P1.argmax(1), P2.argmax(1)
    f1, f2, rec = PA.fit_fine_pair(P1, d1, P2, d2, caps={1: 8, 2: 12})
    return dict(P1=P1, P2=P2, S=S, d1=d1, d2=d2, f1=f1, f2=f2, rec=rec)


@pytest.fixture(scope="module")
def bbank(bsmall):
    s = bsmall
    out = {}
    m1, m2 = 3, 5
    out[("CLASS", None)] = CP.fit_policy_pair("CLASS", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"],
                                              1, 1, None)
    ft = CP.fit_policy_pair("FINE-TASK", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], m1, m2, None)
    out[("FINE-TASK", None)] = ft
    for lam in (0.1, 1.0, 10.0):
        w = {"FINE-TASK": ft[0]}
        for fam in ("LOCAL", "SEQ-12", "SEQ-21"):
            out[(fam, lam)] = CP.fit_policy_pair(fam, s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"],
                                                 m1, m2, lam)
            w[fam] = out[(fam, lam)][0]
        out[("JOINT", lam)] = CP.fit_policy_pair("JOINT", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"],
                                                 s["S"], m1, m2, lam, witnesses=w)
    return out


def test_fine_partitions_caps_starts_and_support(bsmall, tdata):
    s = bsmall
    for r, f, cap in ((1, s["f1"], 8), (2, s["f2"], 12)):
        for c in range(f.K):
            assert f.cells_of(c).size <= cap
        pcs = s["rec"][f"r{r}"]["per_class"]
        for pc in pcs:
            if not pc["fallback"]:
                assert [x["start"] for x in pc["starts"]] == list(KM.STARTS)
                assert pc["winner"] in KM.STARTS
    assert s["f2"].fallback[s["f2"].cells_of(5)].all()
    sup = s["rec"]["support"][2]
    assert sup[5]["fallback"] and sup[0]["cells"] == s["f2"].cells_of(0).size
    T, tr, _ = tdata
    rec, files = PA.fine_unit(T, tr, caps={1: 32, 2: 128})
    KM.json_safe(rec)
    f1, f2 = PA.load_fine(json.loads(_write_read(files["fine.json"])))
    assert max(f2.cells_of(c).size for c in range(6)) > 64                  # a real 128-cap fine partition
    assert np.array_equal(files["assign.npz"]["f2"], DPT.assign_fine(T["p2"], T["d2"], f2))
    assert set(files["assign.npz"]) == {"row_id", "f1", "f2"}


def _labels_merge(lab, a, b):
    lab = lab.copy()
    lab[lab == b] = a
    return lab


def _random_state(s, rng):
    """A random class-respecting coarse grouping of the fine cells of both recipients."""
    labs = []
    for f in (s["f1"], s["f2"]):
        lab = np.arange(f.F)
        for c in range(f.K):
            idx = f.cells_of(c)
            if idx.size > 1:
                g = rng.integers(0, max(2, idx.size // 2), idx.size)
                for gg in np.unique(g):
                    mem = idx[g == gg]
                    lab[mem] = mem.min()
        labs.append(lab)
    return labs


def _table(s):
    a1 = DPT.assign_fine(s["P1"], s["d1"], s["f1"])
    a2 = DPT.assign_fine(s["P2"], s["d2"], s["f2"])
    return CP.fine_table(a1, a2, s["S"], s["f1"].F, s["f2"].F)


def test_merge_and_move_deltas_vs_bruteforce(bsmall):
    s = bsmall
    T = _table(s)
    rng = np.random.default_rng(7)
    W = CP.Weights(1.0, 0.7, 0.3, 0.45, 0.9)
    checked = 0
    for _ in range(3):
        l1, l2 = _random_state(s, rng)
        st = CP.State(s["f1"], s["f2"], T, l1, l2)
        v0 = st.value(W)
        for r in (1, 2):
            for c in range(st.fine[r].K):
                if st.count(r, c) < 2:
                    continue
                labs, ia, ib, delta, dD, dI, dI12 = st.merge_deltas(r, c, W)
                for j in range(0, ia.size, max(1, ia.size // 4)):
                    a, b = int(labs[ia[j]]), int(labs[ib[j]])
                    nl = [l1, l2]
                    nl[r - 1] = _labels_merge(st.labels(r), a, b)
                    other = st.labels(2 if r == 1 else 1)
                    nl[2 - r] = other
                    v1 = CP.State(s["f1"], s["f2"], T, nl[0], nl[1]).value(W)
                    assert abs((v1 - v0) - delta[j]) < 1e-12
                    checked += 1
            G = st.G(r)
            for f in range(0, st.fine[r].F, 3):
                res = st.move_deltas(r, f, W, G)
                if res is None:
                    continue
                B, delta, _, _, _ = res
                for j in range(B.size):
                    lab = st.labels(r).copy()
                    lab[f] = B[j]
                    lab = np.array([min(np.flatnonzero(lab == lab[g])) for g in range(lab.size)])
                    nl = [st.labels(1), st.labels(2)]
                    nl[r - 1] = lab
                    v1 = CP.State(s["f1"], s["f2"], T, nl[0], nl[1]).value(W)
                    assert abs((v1 - v0) - delta[j]) < 1e-12
                    checked += 1
    assert checked > 200
    # applied merges/moves keep the incremental tables equal to a fresh rebuild
    l1, l2 = _random_state(s, rng)
    st = CP.State(s["f1"], s["f2"], T, l1, l2)
    labs, ia, ib, *_ = st.merge_deltas(2, 0, W)
    st.apply_merge(2, int(labs[ia[0]]), int(labs[ib[0]]))
    G = st.G(1)
    f = next(f for f in range(st.fine[1].F) if st.move_deltas(1, f, W, G) is not None)
    st.apply_move(1, f, int(st.move_deltas(1, f, W, G)[0][0]), G)
    fresh = CP.State(s["f1"], s["f2"], T, st.labels(1), st.labels(2))
    for k, v in st.terms().items():
        assert abs(v - fresh.terms()[k]) < 1e-13


def test_objective_matches_row_level_all_families(bbank):
    for key, (pair, rec) in bbank.items():
        assert rec["row_level_max_abs_diff"] <= 1e-12, key
        for k in ("D1", "D2", "I1", "I2", "I12"):
            assert abs(rec["final"][k] - rec["row_level_check"][k]) <= 1e-12


def test_at_most_cap_and_extra_merges(bbank, bsmall):
    s = bsmall
    caps = {1: 3, 2: 5}
    extra_seen = 0
    for (fam, lam), (pair, rec) in bbank.items():
        m1, m2 = (1, 1) if fam == "CLASS" else (3, 5)
        assert max(pair.p1.tokens_per_class()) <= m1 and max(pair.p2.tokens_per_class()) <= m2
        assert rec["r1"]["alphabet"] == pair.p1.T and rec["r2"]["alphabet"] == pair.p2.T
        merges = [x for st in rec.get("stages", []) for x in st["merges"] + st.get("moves", [])
                  if x.get("kind") == "extra"]
        if "starts" in rec:
            merges += [x for v in rec["starts"].values() for x in v.get("merges", []) + v["moves"]
                       if x.get("kind") == "extra"]
        for x in merges:
            assert x["increment"] < -CP.TOL
        extra_seen += len(merges)
        if fam == "FINE-TASK":                    # distortion merges never improve: exactly min(cap, fine cells)
            for r, pol, f in ((1, pair.p1, s["f1"]), (2, pair.p2, s["f2"])):
                for c in range(f.K):
                    assert pol.tokens_per_class()[c] == min(caps[r], f.cells_of(c).size)
    assert extra_seen > 0
    lo = bbank[("SEQ-21", 10.0)][0]
    assert sum(lo.p1.tokens_per_class()) + sum(lo.p2.tokens_per_class()) < 2 * 3 + 5 * 5 + 1


def test_sequential_first_stage_conditions_on_other_decision(bbank, bsmall):
    s = bsmall
    T = _table(s)
    for lam in (0.1, 1.0, 10.0):
        for fam, a, b in (("SEQ-12", 1, 2), ("SEQ-21", 2, 1)):
            pair, rec = bbank[(fam, lam)]
            st1, st2 = rec["stages"]
            assert st1["weights"] == CP.W_joint(lam).__dict__ and st2["weights"] == CP.W_joint(lam).__dict__
            assert st1["recipients"] == [a] and st2["recipients"] == [b]
            # stage-one objective = F_joint with the other recipient at its CLASS-ONLY release (brute force)
            fa = (s["f1"], s["f2"])[a - 1]
            pol_a = (pair.p1, pair.p2)[a - 1]
            labs = {a: CP.labels_from_policy(pol_a), b: CP.class_labels((s["f1"], s["f2"])[b - 1])}
            stc = CP.State(s["f1"], s["f2"], T, labs[1], labs[2])
            Pa, Pb = (s["P1"], s["P2"]) if a == 1 else (s["P2"], s["P1"])
            ta, qa, _ = RL.encode(pol_a, Pa)
            db = Pb.argmax(1)
            I_joint_cls = DCP.mi_plugin(s["S"], ta * 10 + db)
            bc = rec["baseline_correction"]
            assert bc["counterpart"] == "CLASS-ONLY"
            assert abs(bc["I12_with_class_counterpart"] - I_joint_cls) < 1e-12
            assert abs(bc["stage1_F_joint_with_class_counterpart"] - CP.F_values(stc.terms(), lam)["F_joint"]) < 1e-12
            assert bc["conditional_I_S_decision_given_code"] >= -1e-12
            assert "old_rule_stage1" in bc and "same_map_as_corrected" in bc["old_rule_stage1"]
            # the first map is frozen: the released first map equals the stage-one result
            assert abs(st1["after_refine"][f"D{a}"] - rec["final"][f"D{a}"]) < 1e-12


def _redundant_fixture(N=4000, seed=0):
    """Recipient 2's decision carries S; recipient 1's within-class clue u is a noisy copy of S. Under the old
    surrogate (other recipient constant) the clue looks expensive; given the disclosed decision d2 it adds little."""
    rng = np.random.default_rng(seed)
    S = rng.integers(0, 2, N)
    d2 = np.where(rng.random(N) < 0.97, S, 1 - S)
    u = np.where(rng.random(N) < 0.9, S, 1 - S)
    p1 = np.array([0.93, 0.70])[u] + rng.choice([0.0, 0.01], N)
    P1 = np.stack([p1, 1 - p1], 1)
    p2 = np.where(d2 == 0, 0.8, 0.2) + rng.choice([0.0, 0.01], N)
    P2 = np.stack([p2, 1 - p2], 1)
    return P1, P2, S


def test_sequential_correction_changes_stage_one():
    P1, P2, S = _redundant_fixture()
    f1, f2, _ = PA.fit_fine_pair(P1, None, P2, None, caps={1: 8, 2: 8})
    lam = 0.25
    pair, rec = CP.fit_policy_pair("SEQ-12", f1, f2, P1, None, P2, None, S, 2, 2, lam)
    bc = rec["baseline_correction"]
    old = bc["old_rule_stage1"]
    # the corrected stage one sees that d2 already discloses S: it keeps the clue the old surrogate would remove
    assert not old["same_map_as_corrected"]
    assert old["I1"] < bc["I1"]
    assert bc["stage1_F_joint_with_class_counterpart"] < old["F_joint_with_class_counterpart"]


def _xor_fixture(N=4000, seed=0):
    """S = u XOR v; u (resp. v) is a large within-class confidence difference of recipient 1 (resp. 2); w is a small
    uninformative difference. Each recipient alone carries ~0 information about S; the coalition carries ~1 bit."""
    rng = np.random.default_rng(seed)
    u, v, w1, w2 = (rng.integers(0, 2, N) for _ in range(4))
    S = u ^ v
    p1 = np.array([0.95, 0.65])[u] + np.array([0.0, 0.02])[w1]
    p2 = np.array([0.93, 0.62])[v] + np.array([0.0, 0.02])[w2]
    return np.stack([p1, 1 - p1], 1), np.stack([p2, 1 - p2], 1), S


def test_xor_coalition_fixture():
    P1, P2, S = _xor_fixture()
    f1, f2, _ = PA.fit_fine_pair(P1, None, P2, None, caps={1: 8, 2: 8})
    ft = CP.fit_policy_pair("FINE-TASK", f1, f2, P1, None, P2, None, S, 16, 16, None)[1]["final"]
    assert ft["I1"] < 0.003 and ft["I2"] < 0.003 and ft["I12"] > 0.6
    lam = 10.0
    loc = CP.fit_policy_pair("LOCAL", f1, f2, P1, None, P2, None, S, 2, 2, lam)[1]["final"]
    jnt = CP.fit_policy_pair("JOINT", f1, f2, P1, None, P2, None, S, 2, 2, lam)[1]["final"]
    seq = CP.fit_policy_pair("SEQ-12", f1, f2, P1, None, P2, None, S, 2, 2, lam)[1]["final"]
    assert loc["I12"] > 0.6                                           # local criteria cannot see the coalition
    assert jnt["I12"] < 0.05 and jnt["F_joint"] < loc["F_joint"]
    assert seq["I12"] < 0.05                                          # stage two sees the frozen first map


def _enumerate_partitions(items, maxblocks):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _enumerate_partitions(rest, maxblocks):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        if len(part) < maxblocks:
            yield [[first]] + part


def _all_maps(fine, m):
    per_class = []
    for c in range(fine.K):
        idx = [int(x) for x in fine.cells_of(c)]
        per_class.append(list(_enumerate_partitions(idx, m)))
    for combo in itertools.product(*per_class):
        lab = np.arange(fine.F)
        for part in combo:
            for block in part:
                lab[block] = min(block)
        yield lab


def test_exhaustive_optimiser_gaps_on_tiny_xor():
    """Tiny XOR fixture: every class-preserving map with <= m cells per class of both recipients is enumerated;
    every family's fitted objective is compared with the exhaustive optimum of its own objective (gaps recorded)."""
    rng = np.random.default_rng(12)
    N = 900
    u, v = rng.integers(0, 2, N), rng.integers(0, 2, N)
    w1, w2 = rng.integers(0, 2, N), rng.integers(0, 2, N)
    S = u ^ v
    S = np.where(rng.random(N) < 0.1, 1 - S, S)                     # 10% label noise
    p1 = np.array([0.95, 0.7])[u] + np.array([0.0, 0.05])[w1]
    p1 = np.where(rng.random(N) < 0.25, 0.3 + 0.1 * w1, p1)            # class 1 rows with two fine values
    p2 = np.array([0.92, 0.66])[v] + np.array([0.0, 0.05])[w2]
    P1, P2 = np.stack([p1, 1 - p1], 1), np.stack([p2, 1 - p2], 1)
    f1, f2, _ = PA.fit_fine_pair(P1, None, P2, None, caps={1: 4, 2: 4})
    T = CP.fine_table(DPT.assign_fine(P1, None, f1), DPT.assign_fine(P2, None, f2), S, f1.F, f2.F)
    m1, m2 = 2, 2
    gaps = {}
    for lam in (0.05, 0.5, 5.0):
        best = {"F_task": np.inf, "F_local": np.inf, "F_joint": np.inf}
        count = 0
        for l1 in _all_maps(f1, m1):
            for l2 in _all_maps(f2, m2):
                fv = CP.F_values(CP.State(f1, f2, T, l1, l2).terms(), lam)
                count += 1
                for k in best:
                    best[k] = min(best[k], fv[k])
        fits = {}
        for fam in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21", "JOINT"):
            lm = None if fam == "FINE-TASK" else lam
            pair, rec = CP.fit_policy_pair(fam, f1, f2, P1, None, P2, None, S, m1, m2, lm)
            fv = CP.F_values(rec["final"], lam)
            key = {"FINE-TASK": "F_task", "LOCAL": "F_local"}.get(fam, "F_joint")
            gap = fv[key] - best[key]
            assert gap >= -1e-12
            fits[fam] = {"objective": key, "fitted": fv[key], "exhaustive_best": best[key], "gap": gap}
            if fam == "JOINT":
                assert all(fv["F_joint"] <= rec["witness_dominance"][w]["witness_F_joint"] for w in CP.WITNESSES)
        gaps[lam] = {"maps": count, **fits}
    print(json.dumps({"exhaustive_xor_gaps": gaps}, indent=1))
    assert all(gaps[l]["maps"] > 50 for l in gaps)


def test_joint_witness_dominance_and_validation(bbank, bsmall):
    s = bsmall
    for lam in (0.1, 1.0, 10.0):
        jp, jr = bbank[("JOINT", lam)]
        fj = jr["final"]["F_joint"]
        for w in CP.WITNESSES:
            wp = bbank[(w, None if w == "FINE-TASK" else lam)][0]
            wv = CP._brute_terms(wp, s["P1"], s["d1"], s["P2"], s["d2"], s["S"])[0]
            assert fj <= CP.F_values(wv, lam)["F_joint"] + 1e-12
            assert jr["starts"][w]["source"] == "passed_in"
        assert len(jr["candidates"]) == 9 and jr["winner"]["start"] in CP.JOINT_START_ORDER
        for u in jr["unresolved_local_optima"]:
            assert u["gap_to_winner"] > 0
    # recomputed witnesses give the same JOINT pair as passed-in witnesses
    rp, rr = CP.fit_policy_pair("JOINT", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 3, 5, 1.0)
    assert rp.fingerprint() == bbank[("JOINT", 1.0)][0].fingerprint()
    assert all(rr["starts"][w]["source"] == "recomputed" for w in CP.WITNESSES)
    bad = {"LOCAL": bbank[("LOCAL", 0.1)][0]}
    with pytest.raises(ValueError, match="different family/caps/lam"):
        CP.fit_policy_pair("JOINT", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 3, 5, 1.0,
                           witnesses=bad)
    with pytest.raises(ValueError, match="unknown witness"):
        CP.fit_policy_pair("JOINT", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 3, 5, 1.0,
                           witnesses={"DIRECT-TASK": bbank[("FINE-TASK", None)][0]})


def test_stage_b_determinism_and_label_independence(bsmall, bbank):
    s = bsmall
    for fam, lam in (("LOCAL", 1.0), ("SEQ-21", 1.0), ("JOINT", 10.0)):
        w = None if fam != "JOINT" else {x: bbank[(x, None if x == "FINE-TASK" else lam)][0] for x in CP.WITNESSES}
        p, r = CP.fit_policy_pair(fam, s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 3, 5, lam,
                                  witnesses=w)
        p0, r0 = bbank[(fam, lam)]
        assert p.fingerprint() == p0.fingerprint()
        strip = lambda x: json.dumps({k: v for k, v in x.items() if k not in ("wall_seconds", "cpu_seconds")},  # noqa
                                     sort_keys=True, default=str)
        assert strip(r) == strip(r0)
    Sp = s["S"][np.random.default_rng(0).permutation(s["S"].size)]
    for fam in ("CLASS", "FINE-TASK"):
        m = (1, 1) if fam == "CLASS" else (3, 5)
        p = CP.fit_policy_pair(fam, s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], Sp, *m, None)[0]
        assert p.fingerprint() == bbank[(fam, None)][0].fingerprint()      # task families never read S
    # LOCAL with lam = 0 aliases FINE-TASK
    p = CP.fit_policy_pair("LOCAL", s["f1"], s["f2"], s["P1"], s["d1"], s["P2"], s["d2"], s["S"], 3, 5, 0.0)[0]
    al = CP.find_aliases({"FINE-TASK": bbank[("FINE-TASK", None)][0], "LOCAL0": p})
    assert al["pair"]["LOCAL0"] == "FINE-TASK"


def test_stage_b_class_preservation_and_restore(bbank, tmp_path):
    rng = np.random.default_rng(4)
    for key, (pair, rec) in bbank.items():
        for pol in pair:
            P = adversarial_rows(pol.K, rng, 50)
            tok, q, dec = RL.encode(pol, P)
            assert np.array_equal(dec, P.argmax(1))
        RL.save_policy(pair, tmp_path / "p.json")
        back = RL.load_policy(tmp_path / "p.json")
        assert back.fingerprint() == pair.fingerprint() and back.config == pair.config
    # receipts are strict JSON
    for key, (pair, rec) in bbank.items():
        KM.json_safe(rec)


def test_fit_unit_interface(tdata):
    T, tr, S = tdata
    rec, files = PA.fine_unit(T, tr, caps={1: 8, 2: 16})
    fine = json.loads(_write_read(files["fine.json"]))
    meta = {**META, "config": "U|FINE-TASK|i4o8"}
    r_ft, f_ft = CP.fit_unit("FINE-TASK", fine, T, tr, S[tr], 4, 8, None, meta)
    pol_ft = json.loads(_write_read(f_ft["policy.json"]))
    wit = {"FINE-TASK": pol_ft}
    for fam in ("LOCAL", "SEQ-12", "SEQ-21"):
        r, f = CP.fit_unit(fam, fine, T, tr, S[tr], 4, 8, 0.1, {**META, "config": f"U|{fam}|i4o8|l0.1"})
        wit[fam] = json.loads(_write_read(f["policy.json"]))
    r, f = CP.fit_unit("JOINT", fine, T, tr, S[tr], 4, 8, 0.1, {**META, "config": "U|JOINT|i4o8|l0.1"}, wit)
    assert r["config"] == "U|JOINT|i4o8|l0.1" and all(r["starts"][w]["source"] == "passed_in" for w in CP.WITNESSES)
    out = f["release.npz"]
    assert sorted(out) == sorted(["row_id", "tok1", "q1", "hard1", "alpha1", "tok2", "q2", "hard2", "alpha2"])
    assert np.array_equal(out["hard1"], T["d1"]) and np.array_equal(out["hard2"], T["d2"])
    pp = RL.PolicyPair.from_dict(json.loads(_write_read(f["policy.json"])))
    assert pp.config["m1"] == 4 and pp.config["m2"] == 8 and pp.config["lam"] == 0.1
    assert RL.check_bound(pp)["teacher_model_sha256"] == SHA_T
    rc, fc = CP.fit_unit("CLASS", fine, T, tr, S[tr], 1, 1, None, {**META, "config": "U|CLASS|i1o1"})
    assert rc["r2"]["alphabet"] == 6 and rc["config"] == "U|CLASS|i1o1"
    with pytest.raises(ValueError, match="differs"):
        CP.fit_unit("LOCAL", fine, T, tr, S[tr], 4, 8, 0.1, {**META, "config": "U|LOCAL|i4o8|l1"})
    with pytest.raises(ValueError, match="binary"):
        CP.fit_unit("LOCAL", fine, T, tr, S[tr] + 2, 4, 8, 0.1, META)
