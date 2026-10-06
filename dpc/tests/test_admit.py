"""Tests for dpc.data / dpc.admit (data/custody owner). Synthetic fixtures always run; real-data checks run when the
private store and the admitted copies exist (they read no label and write nothing).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q dpc/tests/test_admit.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dpc import admit as AD
from dpc import data as DD

REAL = (AD.ADMITTED / "SHA256SUMS").exists() and DD.SRC.exists()


def _state(seed=0):
    g = torch.Generator().manual_seed(seed)
    st = {}
    for i in (0, 1):
        for j, (o, n) in zip((0, 2, 4), ((64, 83), (64, 64), (16, 64))):
            st[f"enc.{i}.{j}.weight"] = torch.randn(o, n, generator=g) * 0.2
            st[f"enc.{i}.{j}.bias"] = torch.randn(o, generator=g) * 0.1
    return st


def test_forward_equals_sequential_module_bitwise():
    st = _state()
    X = np.random.default_rng(0).normal(size=(257, 83)).astype(np.float32)
    H = AD.forward(st, X)
    for i in (0, 1):
        m = torch.nn.Sequential(torch.nn.Linear(83, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 16))
        m.load_state_dict({k.split(".", 2)[2]: v for k, v in st.items() if k.startswith(f"enc.{i}.")})
        with torch.no_grad():
            ref = m(torch.from_numpy(X)).double().numpy()
        assert H[i].dtype == np.float64 and np.array_equal(H[i], ref)


@pytest.mark.parametrize("K", [2, 6])
def test_head_outputs_match_pinned_convention(K):
    from jcv.finalize import outputs as pinned_outputs
    rng = np.random.default_rng(K)
    R = rng.normal(size=(400, 16))
    y = rng.integers(0, K, 400)
    head = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)).fit(R, y)
    c, P, d = AD.head_outputs(head, R)
    c0, P0, d0 = pinned_outputs(head, R)
    assert np.array_equal(c, c0) and np.array_equal(P, P0) and np.array_equal(d, d0)
    assert np.allclose(c.sum(1), 0.0, atol=1e-12)
    if K == 2:
        m = head.decision_function(R)
        assert np.array_equal(c, np.stack([-m / 2, m / 2], 1))


def test_first_index_tie_rule_and_tie_count():
    P = np.array([[0.5, 0.5], [0.2, 0.8], [0.4, 0.4]])
    P = np.hstack([P, np.array([[0.0], [0.0], [0.2]])])
    assert np.argmax(P, 1).tolist() == [0, 1, 0]
    assert AD.ties(P) == 2


def _unit(tmp: Path, name="u1"):
    d = tmp / name
    d.mkdir(parents=True)
    files = {}
    for f, b in (("a.bin", b"alpha"), ("sub/b.json", b"{}")):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_bytes(b)
        files[f] = hashlib.sha256(b).hexdigest()
    (d / "COMPLETE.json").write_text(json.dumps({"id": name, "files": files}))
    return d, files


def test_copy_verified_copies_never_moves_and_refuses_differing(tmp_path):
    src, files = _unit(tmp_path / "src")
    before = {f: AD.sha(src / f) for f in files}
    dst = tmp_path / "dst" / "u1"
    r1 = AD.copy_verified(src, dst, files)
    assert r1["copied_now"] == 3 and r1["uncached_reread_matches_pinned"]
    r2 = AD.copy_verified(src, dst, files)
    assert r2["already_identical"] == 3 and r2["copied_now"] == 0
    assert {f: AD.sha(src / f) for f in files} == before and (src / "COMPLETE.json").exists()
    (dst / "a.bin").write_bytes(b"tampered")
    with pytest.raises(SystemExit):
        AD.copy_verified(src, dst, files)
    assert (dst / "a.bin").read_bytes() == b"tampered"          # nothing overwritten


def test_public_writer_refuses_identifying_paths(tmp_path):
    with pytest.raises(SystemExit):
        AD.write_public(tmp_path / "x.json", {"p": str(Path.home() / "secret")})
    with pytest.raises(SystemExit):
        AD.write_public(tmp_path / "x.json", {"p": "/Volumes/Some Drive/x"})
    AD.write_public(tmp_path / "ok.json", {"p": "<PRIVATE_CACHE>/dpc_v1"})
    assert json.loads((tmp_path / "ok.json").read_text())["p"] == "<PRIVATE_CACHE>/dpc_v1"


def test_tree_traversal_matches_sklearn_apply(tmp_path):
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 83)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 3] > 0).astype(int)
    t = DecisionTreeClassifier(max_leaf_nodes=12, random_state=0).fit(X, y).tree_
    leaves = np.flatnonzero(t.children_left == -1)
    (tmp_path / "model.json").write_text(json.dumps({
        "children_left": t.children_left.tolist(), "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(), "threshold": t.threshold.tolist(), "leaf_node_ids": leaves.tolist(),
        "n_fit": 600, "n_features": 83}))
    Xt = rng.normal(size=(300, 83))
    cells, n_fit, nf = AD.tree_cells(tmp_path / "model.json", Xt)
    ref = np.searchsorted(leaves, DecisionTreeClassifier(max_leaf_nodes=12, random_state=0).fit(X, y)
                          .apply(Xt.astype(np.float32)))
    assert np.array_equal(cells, ref) and n_fit == 600 and nf == 83


def test_state_sha_matches_osf_convention():
    from osf.train import state_sha
    st = _state(3)
    assert AD.state_sha(st) == state_sha(st)


def test_api_rejects_unknown_names():
    with pytest.raises(KeyError):
        AD.teacher("NORM-J", 0)
    with pytest.raises(KeyError):
        AD.teacher("U", 3)
    with pytest.raises(KeyError):
        AD.reference("LEACE", 0)


def test_unseal_gate_refuses_other_callers_and_unpushed_lock(tmp_path, monkeypatch):
    with pytest.raises(PermissionError):
        DD.unseal_gate("dpc.select")
    monkeypatch.setattr(DD, "EVALUATION_LOCK", tmp_path / "EVALUATION_LOCK.json")
    with pytest.raises(PermissionError):
        DD.unseal_gate("dpc.assess")
    with pytest.raises(PermissionError):           # called from a test module: refused before any lock check
        DD.load(unseal=True)


def test_labels_allowlist_on_synthetic_D():
    role = np.array(["OSF_DEFENSE_FIT", "AUDIT_FIT", "INNER_SELECTION", DD.ASSESS, "HEAD_VALIDATION"])
    D = {"role": role, "sealed": True, "idx": {r: np.flatnonzero(role == r) for r in DD.ROLES}}
    assert DD.labels_for(D, "fitting", "OSF_DEFENSE_FIT").tolist() == [0]
    assert DD.labels_for(D, "fitting", "DEFENSE_FIT").tolist() == [0]           # osf alias
    for proc, r in (("fitting", "AUDIT_FIT"), ("selection", "OSF_DEFENSE_FIT"), ("fitting", DD.ASSESS),
                    ("inner_audit", DD.ASSESS), ("selection", DD.ASSESS)):
        with pytest.raises(PermissionError):
            DD.labels_for(D, proc, r)
    with pytest.raises(PermissionError):                                         # sealed even when allowed
        DD.labels_for(D, "assessment", DD.ASSESS)


def test_resolve_gives_the_21_pinned_units():
    u = AD.resolve()
    names = sorted(e["unit"] for es in u.values() for e in es)
    assert len(names) == 21
    assert "rel__s2__RAW-J_b0.3" in names and "fare__s1__p1__Z1" in names and "lc__s0__E" in names


@pytest.mark.skipif(not REAL, reason="private store / admitted copies absent")
def test_real_roles_and_sealing():
    D = DD.load()
    assert len(D["row_id"]) == DD.N_ROWS
    for r, (n, g) in DD.EXPECTED.items():
        ix = D["idx"][r]
        assert len(ix) == n and len(np.unique(D["unit"][ix])) == g
    a = D["idx"][DD.ASSESS]
    assert all(np.all(D[k][a] == -1) for k in DD.LABEL_KEYS) and D["sealed"]
    _, bad = DD.check_against_source(D)
    assert bad == []


@pytest.mark.skipif(not REAL, reason="private store / admitted copies absent")
def test_real_teacher_api_reads_verified_copy():
    D = DD.load()
    for name in AD.TEACHERS:
        T = AD.teacher(name, 0)
        assert set(T) == {"row_id", "p1", "p2", "d1", "d2", "c1", "c2", "r1", "r2"}
        assert np.array_equal(T["row_id"], D["row_id"])
        for i, K in ((1, 2), (2, 6)):
            assert T[f"p{i}"].shape == (DD.N_ROWS, K) and np.array_equal(np.argmax(T[f"p{i}"], 1), T[f"d{i}"])
    F = AD.reference("F", 1)
    assert F["r1"].shape[1] == F["n_cells1"] and np.array_equal(np.argmax(F["r1"], 1), F["cells1"])
    assert AD.admission_record()["verdict"] == "ADMITTED"
