"""Tests for qpc.data / qpc.admit (data/custody owner E). Synthetic fixtures always run (a fake dpc/osf private store in
tmp_path, with pins monkeypatched); real checks are SEALED-ONLY (roles, hashes; no forward pass, no label read) and run
when the private input exists.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q qpc/tests/test_admit.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dpc import admit as DA
from dpc import data as DD
from qpc import admit as QA
from qpc import data as QD

REAL = QD.SRC.exists()
N = 300


# ------------------------------------------------------------------ synthetic store
def _sha(b: bytes):
    return hashlib.sha256(b).hexdigest()


def _state(seed):
    g = torch.Generator().manual_seed(seed)
    st = {}
    for i in (0, 1):
        for j, (o, n) in zip((0, 2, 4), ((64, 83), (64, 64), (16, 64))):
            st[f"enc.{i}.{j}.weight"] = torch.randn(o, n, generator=g) * 0.2
            st[f"enc.{i}.{j}.bias"] = torch.randn(o, generator=g) * 0.1
    for i, K in enumerate((2, 6)):
        st[f"head.{i}.weight"] = torch.randn(K, 16, generator=g)
        st[f"head.{i}.bias"] = torch.randn(K, generator=g)
    return st


def _D(seed=0):
    rng = np.random.default_rng(seed)
    role = np.array(["OSF_DEFENSE_FIT"] * 150 + ["AUDIT_FIT"] * 50 + ["INNER_SELECTION"] * 50 + [QD.ASSESS] * 50)
    D = {"X": rng.normal(size=(N, 83)), "row_id": np.arange(1000, 1000 + N, dtype=np.int64), "role": role,
         "unit": np.arange(N), "sealed": True, "feature_names": np.asarray([f"f{j}" for j in range(83)])}
    D["idx"] = {r: np.flatnonzero(role == r) for r in QD.ROLES}
    D["idx"]["DEFENSE_FIT"] = D["idx"]["OSF_DEFENSE_FIT"]
    return D


def _write_complete(d: Path, uid):
    files = {str(p.relative_to(d)): DA.sha(p) for p in sorted(d.rglob("*")) if p.is_file() and p.name != "COMPLETE.json"}
    (d / "COMPLETE.json").write_text(json.dumps({"id": uid, "files": files}))
    return files


def _teacher_artifacts(d: Path, st, D, seed):
    d.mkdir(parents=True)
    torch.save(st, d / "model.pt")
    H = DA.forward(st, D["X"])
    fit = D["idx"]["OSF_DEFENSE_FIT"]
    rng = np.random.default_rng(seed)
    rel = {"row_id": D["row_id"]}
    for i, K in enumerate((2, 6)):
        y = rng.integers(0, K, len(fit))
        y[:K] = np.arange(K)
        head = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300)).fit(H[i][fit], y)
        joblib.dump(head, d / f"head_{i}.joblib")
        c, P, hard = DA.head_outputs(head, H[i])
        rel.update({f"r{i + 1}": H[i], f"c{i + 1}": c, f"p{i + 1}": P, f"hard{i + 1}": hard})
    np.savez(d / "release.npz", **rel)
    (d / "record.json").write_text(json.dumps({"unit": d.name}))
    return rel


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A synthetic dpc admitted store + dpc run units + osf store, with matching synthetic pins."""
    D = _D()
    dpc_adm, dpc_units, osf_units = tmp_path / "dpc/admitted", tmp_path / "dpc/run/units", tmp_path / "osf/units"
    adm = {"teachers": {}, "references": {}}
    lock = {"seeds": {str(k): {"unit_file_sha256": {}} for k in QA.SEEDS}}
    for t in QA.TEACHERS:
        for k in QA.SEEDS:
            st = _state(10 * k + len(t))
            src = dpc_adm / QA.tea_unit(t, k)
            rel = _teacher_artifacts(src, st, D, k)
            files = _write_complete(src, src.name)
            adm["teachers"][f"{t}|{k}"] = {"unit": src.name, "pass": True, "epoch": 40,
                                           "model_sha256": DA.state_sha(st), "complete_files_sha256": files,
                                           "admitted_from_osf": "smf synthetic"}
            u = dpc_units / QA.dpc_tea_unit(t, k)
            u.mkdir(parents=True)
            np.savez(u / "teacher.npz", row_id=rel["row_id"], p1=rel["p1"], p2=rel["p2"], d1=rel["hard1"],
                     d2=rel["hard2"], c1=rel["c1"], c2=rel["c2"], r1=rel["r1"], r2=rel["r2"])
            (u / "record.json").write_text("{}")
            lock["seeds"][str(k)]["unit_file_sha256"][u.name] = _write_complete(u, u.name)
    refs = {}
    for lab in QA.REFERENCES:
        for k in QA.SEEDS:
            u = dpc_units / QA.ref_unit(lab, k)
            u.mkdir(parents=True)
            rng = np.random.default_rng(k)
            arrs = {"row_id": D["row_id"], "p1": rng.dirichlet(np.ones(2), N), "p2": rng.dirichlet(np.ones(6), N)}
            arrs.update({"d1": arrs["p1"].argmax(1), "d2": arrs["p2"].argmax(1)})
            meta = {"n_cells1": 3, "n_cells2": 4} if lab != "E" else {}
            np.savez(u / "reference.npz", **arrs)
            (u / "record.json").write_text(json.dumps({"seed": k, "label": lab, "arrays": sorted(arrs),
                                                       "admission": meta}))
            lock["seeds"][str(k)]["unit_file_sha256"][u.name] = _write_complete(u, u.name)
            refs[(lab, k)] = {**arrs, **meta}
            for x in QA.UNDERLYING[lab]:
                adm["references"][x.format(k=k)] = {"pass": True, "kind": lab, "complete_files_sha256": {},
                                                    "checks": {"alias": None, "fit_role": "OSF_DEFENSE_FIT"}}
    for name, val in (("DPC_ADMITTED", dpc_adm), ("DPC_UNITS", dpc_units), ("OSF_UNITS", osf_units),
                      ("ADMITTED", tmp_path / "qpc/admitted"), ("DERIVED", tmp_path / "qpc/inputs"),
                      ("RECEIPT", tmp_path / "qpc/admitted/ADMISSION_RECEIPT.json"),
                      ("RESULT", tmp_path / "pub/ADMISSION_RESULT.json")):
        monkeypatch.setattr(QA, name, val)
    monkeypatch.setattr(QA, "dpc_admission", lambda: adm)
    monkeypatch.setattr(QA, "dpc_eval_lock", lambda: lock)
    monkeypatch.setattr(QA.DA, "reference", lambda lab, k: {x: (v.copy() if isinstance(v, np.ndarray) else v)
                                                            for x, v in refs[(lab, k)].items()})
    return {"D": D, "adm": adm, "lock": lock, "tmp": tmp_path, "refs": refs}


# ------------------------------------------------------------------ teachers
def test_teacher_bitwise_parity_and_interface(store):
    D = store["D"]
    for t in QA.TEACHERS:
        for k in QA.SEEDS:
            T = QA.teacher(t, k, D)
            assert set(QA.TEACHER_KEYS) <= set(T)
            assert T["model_sha256"] == store["adm"]["teachers"][f"{t}|{k}"]["complete_files_sha256"]["model.pt"]
            assert T["state_sha256"] == store["adm"]["teachers"][f"{t}|{k}"]["model_sha256"]
            assert np.array_equal(T["row_id"], D["row_id"]) and T["p2"].shape == (N, 6)
            assert np.array_equal(T["d1"], T["p1"].argmax(1)) and T["p1"].dtype == np.float64
            par = QA.parity_with_source(t, k, T)
            assert par["ok"] and all(par["bitwise"].values())
            # the qpc copy exists, verified, and the source is untouched
            assert (QA.ADMITTED / QA.tea_unit(t, k) / "model.pt").exists()
            assert (QA.DPC_ADMITTED / QA.tea_unit(t, k) / "model.pt").exists()


def test_parity_rejects_a_one_ulp_difference(store):
    D = store["D"]
    T = QA.teacher("U", 0, D)
    T2 = dict(T)
    p = T["p2"].copy()
    p[7, 3] = np.nextafter(p[7, 3], 1.0)
    T2["p2"] = p
    par = QA.parity_with_source("U", 0, T2)
    assert not par["ok"] and par["bitwise"]["p2"] is False
    T3 = dict(T)
    T3["d1"] = T["d1"].astype(np.int32)                       # dtype change is a parity failure too
    assert not QA.parity_with_source("U", 0, T3)["ok"]


def test_teacher_refuses_when_dpc_parity_target_differs(store):
    """A re-pinned but different dpc teacher unit (one ULP) must refuse the teacher, not pass silently."""
    u = QA.DPC_UNITS / QA.dpc_tea_unit("U", 1)
    z = dict(np.load(u / "teacher.npz"))
    z["c2"] = z["c2"].copy()
    z["c2"][0, 0] = np.nextafter(z["c2"][0, 0], 10.0)
    np.savez(u / "teacher.npz", **z)
    store["lock"]["seeds"]["1"]["unit_file_sha256"][u.name] = _write_complete(u, u.name)
    with pytest.raises(SystemExit, match="parity"):
        QA.teacher("U", 1, store["D"])


def test_teacher_refuses_when_parity_target_fails_its_pin(store):
    u = QA.DPC_UNITS / QA.dpc_tea_unit("RAW-J_b0.3", 2)
    (u / "record.json").write_text('{"tampered": 1}')
    with pytest.raises(SystemExit, match="pin"):
        QA.teacher("RAW-J_b0.3", 2, store["D"])
    assert QA.parity_with_source("RAW-J_b0.3", 2, {})["ok"] is False


def test_fallback_to_osf_store_and_refusal_without_any_source(store):
    import shutil
    src = QA.DPC_ADMITTED / QA.tea_unit("U", 2)
    shutil.copytree(src, QA.OSF_UNITS / src.name)
    (src / "head_1.joblib").write_bytes(b"corrupt")             # dpc copy now fails its pin
    _, rec = QA.admit_teacher("U", 2, store["D"])
    assert rec["admission_copy"]["source"] == "osf_store" and rec["pass"]
    shutil.rmtree(QA.ADMITTED / src.name)
    shutil.rmtree(QA.OSF_UNITS / src.name)
    with pytest.raises(SystemExit, match="no verified source"):
        QA.admit_teacher("U", 2, store["D"])


def test_differing_admitted_copy_is_refused_and_not_overwritten(store):
    QA.teacher("U", 0, store["D"])
    q = QA.ADMITTED / QA.tea_unit("U", 0) / "release.npz"
    q.write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="differs"):
        QA.teacher("U", 0, store["D"])
    assert q.read_bytes() == b"tampered"


def test_wrong_model_hash_refuses(store):
    store["adm"]["teachers"]["U|1"]["model_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="parity"):
        QA.teacher("U", 1, store["D"])


def test_pinned_admission_must_have_passed(store):
    store["adm"]["teachers"]["U|0"]["pass"] = False
    with pytest.raises(SystemExit, match="did not pass"):
        QA.teacher("U", 0, store["D"])


def test_api_rejects_unknown_names(store):
    with pytest.raises(KeyError):
        QA.teacher("NORM-J", 0, store["D"])
    with pytest.raises(KeyError):
        QA.teacher("U", 3, store["D"])
    with pytest.raises(KeyError):
        QA.reference("LEACE", 0)


# ------------------------------------------------------------------ references
def test_reference_verified_copy_and_parity(store):
    for lab in QA.REFERENCES:
        for k in QA.SEEDS:
            R = QA.reference(lab, k)
            ref = store["refs"][(lab, k)]
            assert all(np.array_equal(R[x], v) for x, v in ref.items() if isinstance(v, np.ndarray))
            if lab != "E":
                assert R["n_cells1"] == 3 and R["n_cells2"] == 4
            assert (QA.ADMITTED / QA.ref_unit(lab, k) / "reference.npz").exists()
    _, rec = QA.admit_reference("F", 1)
    assert rec["provenance_valid"] and rec["parity_with_dpc_admitted_artifacts"]["pass"]


def test_reference_provenance_invalid_when_underlying_admission_failed(store):
    store["adm"]["references"]["fare__s0__p1__Z1"]["pass"] = False
    _, rec = QA.admit_reference("F0", 0)
    assert rec["provenance_valid"] is False
    with pytest.raises(SystemExit, match="provenance"):
        QA.reference("F0", 0)


def test_reference_parity_fails_when_admitted_artifacts_differ(store, monkeypatch):
    def bad(lab, k):
        r = {x: (v.copy() if isinstance(v, np.ndarray) else v) for x, v in store["refs"][(lab, k)].items()}
        r["p2"][0] = r["p2"][0][::-1]
        return r
    monkeypatch.setattr(QA.DA, "reference", bad)
    _, rec = QA.admit_reference("E", 2)
    assert rec["provenance_valid"] is False and rec["parity_with_dpc_admitted_artifacts"]["arrays_bitwise"]["p2"] is False


# ------------------------------------------------------------------ inputs and public files
def test_deploy_input_reconstruction_is_deterministic(store, tmp_path, monkeypatch):
    D = store["D"]
    monkeypatch.setattr(QD, "N_ROWS", N)
    monkeypatch.setattr(QD.DD, "FEATURE_NAMES_SHA", _sha("\n".join(str(x) for x in D["feature_names"]).encode()))
    r1 = QA.reconstruct_deploy_input(D, tmp_path / "inp")
    r2 = QA.reconstruct_deploy_input(D, tmp_path / "inp")
    assert r1["pass"] and r2["pass"] and r1["rewritten_now"] and not r2["rewritten_now"]
    assert r1["X_sha256"] == r2["X_sha256"] and r1["schema_sha256"] == r2["schema_sha256"]
    z = np.load(tmp_path / "inp" / "deploy_input.npz")
    assert sorted(z.files) == ["X", "feature_names"] and z["X"].shape == (N, 83)


def test_public_writer_refuses_identifying_paths(tmp_path):
    with pytest.raises(SystemExit):
        QA.write_json(tmp_path / "x.json", {"p": str(Path.home() / "secret")})
    with pytest.raises(SystemExit):
        QA.write_json(tmp_path / "x.json", {"p": "/Volumes/Some Drive/x"})
    QA.write_json(tmp_path / "ok.json", {"p": "<PRIVATE_CACHE>/qpc_v1"})
    assert json.loads((tmp_path / "ok.json").read_text())["p"] == "<PRIVATE_CACHE>/qpc_v1"


def test_reused_dpc_functions_are_the_pinned_ones():
    assert QA.reused_code_check() == {"dpc/admit.py": True, "dpc/data.py": True, "osf/data.py": True}
    assert QD.loader_blob_checks() == {"dpc/data.py": True, "osf/data.py": True}


# ------------------------------------------------------------------ data: sealing, gate, allowlist
def test_labels_allowlist_accepts_dpc_vocabulary_and_never_widens_it():
    role = np.array(["OSF_DEFENSE_FIT", "AUDIT_FIT", "INNER_SELECTION", QD.ASSESS, "HEAD_VALIDATION"])
    D = {"role": role, "sealed": True, "idx": {r: np.flatnonzero(role == r) for r in QD.ROLES}}
    assert set(DD.ALLOW) <= set(QD.ALLOW)
    for p in DD.ALLOW:
        assert set(QD.ALLOW[p]) <= set(DD.ALLOW[p])
    assert QD.labels_for(D, "fitting", "OSF_DEFENSE_FIT").tolist() == [0]
    assert QD.labels_for(D, "fitting", "DEFENSE_FIT").tolist() == [0]          # osf alias
    assert QD.labels_for(D, "selection", "INNER_SELECTION").tolist() == [2]
    assert QD.labels_for(D, "inner_audit", "AUDIT_FIT").tolist() == [1]
    assert QD.labels_for(D, "attack", "AUDIT_FIT").tolist() == [1]             # alias of inner_audit
    for proc, r in (("fitting", "AUDIT_FIT"), ("selection", "OSF_DEFENSE_FIT"), ("fitting", QD.ASSESS),
                    ("inner_audit", QD.ASSESS), ("selection", QD.ASSESS), ("stage_a_utility", QD.ASSESS),
                    ("controls", QD.ASSESS), ("head_validation_calibration", "HEAD_VALIDATION"),
                    ("fitting", "HEAD_VALIDATION"), ("nonsense", "OSF_DEFENSE_FIT")):
        with pytest.raises(PermissionError):
            QD.labels_for(D, proc, r)
    with pytest.raises(PermissionError, match="sealed"):                       # sealed even when allowed
        QD.labels_for(D, "assessment", QD.ASSESS)
    D["sealed"] = False
    assert QD.labels_for(D, "assessment", QD.ASSESS).tolist() == [3]


def test_unseal_gate_refuses_other_callers_dpc_assess_and_unpushed_lock(tmp_path, monkeypatch):
    for caller in ("qpc.select", "dpc.assess", "qpc.run", "__main__"):
        with pytest.raises(PermissionError, match="only qpc.assess"):
            QD.unseal_gate(caller)
    monkeypatch.setattr(QD, "EVALUATION_LOCK", tmp_path / "EVALUATION_LOCK.json")
    with pytest.raises(PermissionError, match="does not exist"):
        QD.unseal_gate("qpc.assess")
    (tmp_path / "EVALUATION_LOCK.json").write_text("{}")
    with pytest.raises(PermissionError, match="not on origin|differs"):          # not committed/pushed on origin
        QD.unseal_gate("qpc.assess")
    with pytest.raises(PermissionError, match="only qpc.assess"):               # from a test module: refused first
        QD.load(unseal=True)


def test_unseal_gate_requires_byte_identity_with_origin(tmp_path, monkeypatch):
    lock = tmp_path / "EVALUATION_LOCK.json"
    lock.write_bytes(b'{"a": 1}\n')
    monkeypatch.setattr(QD, "EVALUATION_LOCK", lock)
    calls = []

    def fake_git(*a, cwd=None):
        calls.append(a)
        return b'{"a": 1}\n' if a[0] == "show" else b""
    monkeypatch.setattr(QD, "git", fake_git)
    monkeypatch.setattr(QD.subprocess, "run", lambda *a, **k: None)
    assert QD.unseal_gate("qpc.assess") == "EVALUATION_LOCK.json byte-identical on origin"
    assert calls[-1] == ("show", f"origin/{QD.BRANCH}:{QD.REL}/EVALUATION_LOCK.json")
    lock.write_bytes(b'{"a": 2}\n')
    with pytest.raises(PermissionError, match="differs"):
        QD.unseal_gate("qpc.assess")


def test_unseal_gate_reads_the_calling_module_name():
    import types
    mod = types.ModuleType("qpc.assess")
    code = "def go(f):\n    return f()\n"
    exec(compile(code, "qpc/assess.py", "exec"), mod.__dict__)
    assert mod.go(lambda: QD._caller_module(2)) == "qpc.assess"


def test_isolation_check_detects_a_shared_defense_fit_group():
    D = _D()
    D["sex"] = np.zeros(N)
    out, bad = QD.check_counts_and_isolation(D)
    assert not [b for b in bad if "overlap" in b]
    D["unit"] = D["unit"].copy()
    D["unit"][D["idx"]["AUDIT_FIT"][0]] = D["unit"][D["idx"]["OSF_DEFENSE_FIT"][0]]
    out, bad = QD.check_counts_and_isolation(D)
    assert "defense_fit_group_overlap:AUDIT_FIT" in bad and out["defense_fit_groups_shared_with"]["AUDIT_FIT"] == 1


# ------------------------------------------------------------------ real, sealed-only (no forward pass, no labels)
@pytest.mark.skipif(not REAL, reason="private input absent")
def test_real_roles_sealing_and_isolation():
    D = QD.load()
    assert D["sealed"] and len(D["row_id"]) == QD.N_ROWS
    for r, (n, g) in QD.EXPECTED.items():
        ix = D["idx"][r]
        assert len(ix) == n and len(np.unique(D["unit"][ix])) == g
    a = D["idx"][QD.ASSESS]
    assert all(np.all(D[k][a] == -1) for k in QD.LABEL_KEYS)
    rep, bad = QD.verify_D(D)
    assert bad == [] and rep["counts_and_isolation"]["defense_fit_groups_shared_with"] == {
        "AUDIT_FIT": 0, "INNER_SELECTION": 0, QD.ASSESS: 0}
    assert np.array_equal(D["idx"]["DEFENSE_FIT"], D["idx"]["OSF_DEFENSE_FIT"])


@pytest.mark.skipif(not REAL, reason="private input absent")
def test_real_input_hash_and_pinned_parity_targets_present():
    assert QD.sha_file(QD.SRC) == QD.INPUT_SHA
    for t in QA.TEACHERS:
        for k in QA.SEEDS:
            ok, why = QA.files_match(QA.DPC_UNITS / QA.dpc_tea_unit(t, k), QA.pinned_unit_files(QA.dpc_tea_unit(t, k), k))
            assert ok, why
            ok, why = QA.files_match(QA.DPC_ADMITTED / QA.tea_unit(t, k),
                                     QA.pinned_teacher(t, k)["complete_files_sha256"])
            assert ok, why
    for lab in QA.REFERENCES:
        for k in QA.SEEDS:
            u = QA.ref_unit(lab, k)
            assert QA.files_match(QA.DPC_UNITS / u, QA.pinned_unit_files(u, k))[0]
