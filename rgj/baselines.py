"""Matched official references for the refreshed guarded joint study: LEACE on U's features, FARE / zero-fairness twin.

Nothing here re-implements a reference method. LEACE is EleutherAI concept-erasure 0.2.4 via the pinned wrapper
stored_model_eval.defenses.fit_leace (official LeaceFitter defaults, float64). FARE is the official eth-sri/fare tree
via oar.fare_official (pinned commit and patched scikit-learn in its own environment).

LEACE reference E (`leace_unit`)
  Encoder features of the study's task-only model U (U's model.pt, jcv.train.Model(83, [2, 6], seed), h_i =
  model.encode(i, X) in float32, cast to float64). One official LEACE map PER RECIPIENT, concept = one-hot SEX,
  fitted on DEFENSE_FIT rows only. Release r_i = map_i(h_i) for every D row; deployed heads refitted with
  rgj.finalize.heads_and_outputs (DEFENSE_FIT fit, HEAD_VALIDATION C). LEACE's native condition is checked on the
  fitting rows only (finite-sample linear guardedness of those rows; not a held-out or nonlinear claim). U's release
  must equal the recomputed features (identity release) or the unit is refused. Unit: lc__s{k}__E.

FARE reference F and zero-fairness twin F0
  Trees: the predecessor's official FARE trees are reused, never refitted: <jcv units>/fare__s{k}__p{i}__c{id} and
  zero-fairness twins __Z{id}. Before reuse each one must (1) be hash-complete (its COMPLETE.json and the FARE cache
  unit's COMPLETE.json), (2) carry the expected seed / purpose / config, (3) have been fitted on exactly the rows and
  labels of DEFENSE_FIT (old defense_train): its recorded fit-row and fit-label sha256 must equal those recomputed
  from D's DEFENSE_FIT rows, and n_fit must match, (4) map to every D row by row_id, and (5) when `reencode` is on,
  the official encode of D's features must reproduce the mapped cells exactly.
  Release: r = one-hot cells, deployed head refitted on the NEW roles (jcv.finalize.fit_head with DEFENSE_FIT /
  HEAD_VALIDATION), centred logits / probabilities / hard decisions from that head. Unit names are kept:
  fare__s{k}__p{i}__c{id} (F grid) and fare__s{k}__p{i}__Z{id} (F0), in the rgj units directory.
  Selection (identical rule to jcv/select.py): per purpose, among grid configurations passing that purpose's
  utility gates versus U on INNER_SELECTION (deployed heads), the lowest inner local SEX AUC; ties -> lower id. The
  inner local AUC is this study's inner measure (rgj.audit.inner_local, finite view [one-hot cells, centred logits],
  inner slate + cell-conditional attackers), so the rule is the predecessor's but the recovery numbers are new.
  No admissible configuration -> NO_FEASIBLE_NOMINEE, the closest configuration (largest worst-gate margin, ties ->
  lower id) is kept descriptively.
  Pair (select_fare): F's pair = the two purposes' selected releases. F is NOMINEE iff both purposes have a
  gate-passing configuration, the pair passes the gates versus U (rgj.select.gate_margins) and the C* local guard
  (inner local AUC_i <= L-R's AUC_i, zero buffer); otherwise NO_FEASIBLE_NOMINEE (descriptive).
  F0 = zero-fairness twin at F's selected configuration per purpose: the predecessor's Z unit when it exists for that
  seed and configuration, otherwise a NEW official fit (FO.fit_encode_cached with FO.zero_fairness(cfg), DEFENSE_FIT
  rows only, seed = k, cached under rgj_v1/fare_cache; allow_f0_fit). F0's status is its own gates + guard result
  (NOMINEE / INFEASIBLE_CONTROL); its pairing to F (configs, F's statuses, tree origin) is a separate field.
  Inner pair audits are archived as units inner__pair__s{k}__F / inner__pair__s{k}__F0 (recovery, utility).
  Certificates are NOT recomputed: the old certification pool is excluded from this study (CERT_STATUS).

Utility gates (shared with the lead's selection; INNER_SELECTION, deployed heads; const = DEFENSE_FIT majority class)
  G1  Acc >= Acc(U) - 0.01
  G2  Acc - const >= 0.8 * (Acc(U) - const)
  G3  Acc - const >= 0.03

API used by rgj.select / rgj.run: leace_unit(k, u_unit_name, D, units_dir); fare_units(k, D, units_dir);
select_fare(k, D, units_dir, uref, guard). CLI (writes units; the lead runs it on study units):
  OMP_NUM_THREADS=1 python -m rgj.baselines leace --seeds 0 1 2 --u-unit tl__s{k}__e30
  OMP_NUM_THREADS=1 python -m rgj.baselines fare-units --seeds 0 1 2
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from jcv.finalize import fit_head, outputs
from rgj import audit as AU
from rgj import finalize as FN

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private"
UNITS = PRIV / "rgj_v1" / "run" / "units"
RGJ_FARE_CACHE = PRIV / "rgj_v1" / "fare_cache"
SRC_UNITS = PRIV / "jcv_v1" / "run" / "units"          # predecessor official FARE units (read-only)
SRC_FARE_CACHE = PRIV / "jcv_v1" / "fare_cache"         # predecessor official FARE trees (read-only)
KS = [2, 6]
TASKS = ("income", "occupation_group")
N_SOURCE_ROWS = 39205
FARE_GRID = [  # the predecessor's bounded configuration bank (jcv.run.FARE_GRID), unchanged
    {"id": 1, "name": "F1_task_first_k100_g0.3", "max_leaf_nodes": 100, "min_samples_leaf": 100, "gamma": 0.3, "criterion": "fair_gini_dp"},
    {"id": 2, "name": "F2_task_first_k50_g0.7", "max_leaf_nodes": 50, "min_samples_leaf": 100, "gamma": 0.7, "criterion": "fair_gini_dp"},
    {"id": 3, "name": "F3_balanced_k20_g0.85", "max_leaf_nodes": 20, "min_samples_leaf": 100, "gamma": 0.85, "criterion": "fair_gini_dp"},
    {"id": 4, "name": "F4_protect_k10_g0.9_ni1000", "max_leaf_nodes": 10, "min_samples_leaf": 1000, "gamma": 0.9, "criterion": "fair_gini_dp"},
    {"id": 5, "name": "F5_protect_k5_g0.95_ni1000", "max_leaf_nodes": 5, "min_samples_leaf": 1000, "gamma": 0.95, "criterion": "fair_gini_dp"},
    {"id": 6, "name": "F6_protect_k3_g0.95_ni1000", "max_leaf_nodes": 3, "min_samples_leaf": 1000, "gamma": 0.95, "criterion": "fair_gini_dp"},
]
CFG = {c["id"]: c for c in FARE_GRID}
CERT_STATUS = {
    "status": "not recomputed (cert pool excluded); predecessor reported vacuous/unavailable",
    "pointer": ["results/pcrl_joint_complete_view_method_v1/AMENDMENT_A3_2026-10-03.md",
                "results/pcrl_joint_complete_view_method_v1/NATIVE_VS_AUDIT.csv",
                "results/pcrl_penalty_no_erasure_v1/PROTOCOL.md (FARE certificates)"],
    "note": "FARE's certificate is a demographic-parity certificate with its own premises, not an attribute-recovery "
            "bound; the old certification pool is excluded from every call of this study.",
}


# ------------------------------------------------------------------ shared utility gates
def constants(D):
    tr = D["idx"]["DEFENSE_FIT"]
    return {i: int(np.argmax(np.bincount(D["y"][t][tr]))) for i, t in enumerate(TASKS)}


def inner_utility(out, D, purposes=(0, 1)):
    """Deployed-head accuracy and constant accuracy on INNER_SELECTION. out has hard1/hard2 (D-row aligned)."""
    v = D["idx"]["INNER_SELECTION"]
    cst = constants(D)
    res = {}
    for i in purposes:
        y = D["y"][TASKS[i]][v]
        res[i] = {"acc": float((out[f"hard{i + 1}"][v] == y).mean()), "const_acc": float((y == cst[i]).mean()),
                  "const_class": cst[i]}
    return res


def gate_one(u, uU):
    a, aU, c = u["acc"], uU["acc"], u["const_acc"]
    g = {"G1": a - (aU - 0.01), "G2": (a - c) - 0.8 * (aU - c), "G3": (a - c) - 0.03}
    return all(x >= 0 for x in g.values()), min(g.values()), g


def gates(util, utilU):
    """Both tasks. Returns (ok, worst margin, per-task margins)."""
    per = {i: gate_one(util[i], utilU[i]) for i in (0, 1)}
    return all(p[0] for p in per.values()), min(p[1] for p in per.values()), {i: p[2] for i, p in per.items()}


# ------------------------------------------------------------------ io helpers
def udir(name, units_dir=None):
    return Path(units_dir or UNITS) / name


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sha_arrays(*arrays):   # identical construction to oar.fare_official._sha256_arrays
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _complete_dir(d):
    c = Path(d) / "COMPLETE.json"
    if not c.exists():
        return False
    return all((Path(d) / f).exists() and _sha_file(Path(d) / f) == h for f, h in json.loads(c.read_text())["files"].items())


def _rows_to_D(row_id_src, D):
    pos = {int(r): j for j, r in enumerate(row_id_src)}
    missing = [int(r) for r in D["row_id"] if int(r) not in pos]
    if missing:
        raise ValueError(f"{len(missing)} D rows absent from the source release")
    return np.array([pos[int(r)] for r in D["row_id"]])


@contextlib.contextmanager
def _fare_units_root(path):
    old = os.environ.get("OAR_RUN_UNITS")
    os.environ["OAR_RUN_UNITS"] = str(path)
    try:
        from oar import fare_official as FO
        yield FO
    finally:
        if old is None:
            os.environ.pop("OAR_RUN_UNITS", None)
        else:
            os.environ["OAR_RUN_UNITS"] = old


# ------------------------------------------------------------------ LEACE (official, per recipient, on U's features)
def leace_unit(k, u_unit_name, D, units_dir=None, name=None, model_factory=None):
    """Official LEACE per recipient on U's encoder features (DEFENSE_FIT fit, float64). Saves lc__s{k}__E."""
    from stored_model_eval.defenses import fit_leace, verify_official_leace
    name = name or f"lc__s{k}__E"
    if FN.unit_complete(udir(name, units_dir)):
        return json.loads((udir(name, units_dir) / "record.json").read_text())
    ud = udir(u_unit_name, units_dir)
    if not FN.unit_complete(ud):
        raise SystemExit(f"REFUSED: U unit {u_unit_name} is not hash-complete")
    t0 = time.time()
    prov = verify_official_leace()
    if model_factory is None:
        from jcv.train import Model
        model_factory = lambda: Model(D["X"].shape[1], KS, k)   # noqa: E731
    model = model_factory()
    model.load_state_dict(torch.load(ud / "model.pt"))
    model.eval()
    Hs = FN.encode_all(model, D["X"])
    zU = np.load(ud / "release.npz")
    take = _rows_to_D(zU["row_id"], D)
    ident = {i: float(np.max(np.abs(zU[f"r{i + 1}"][take] - Hs[i]))) for i in (0, 1)}
    if max(ident.values()) > 1e-6:
        raise SystemExit(f"REFUSED: U's released features differ from its encoder outputs ({ident})")
    tr = D["idx"]["DEFENSE_FIT"]
    Z = np.eye(2)[D["sex"][tr]]
    maps, Rs, native, meta = {}, [], {}, {}
    for i in (0, 1):
        m = fit_leace(Hs[i][tr], Z, fit_row_ids=D["row_id"][tr])
        maps[i] = m
        Rs.append(m.transform(Hs[i]))
        native[i] = m.native_check(Hs[i][tr], Z)
        meta[i] = {k_: m.metadata[k_] for k_ in ("estimator", "settings_used", "dtype", "n_fit", "dim", "rank",
                                                 "fit_row_ids_sha256", "fit_row_ids_count", "diagnostics")}
    out, heads, hmeta = FN.heads_and_outputs(Rs, D)
    files = {"release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out),
             "model.pt": lambda p: shutil.copy2(ud / "model.pt", p)}
    for i in (0, 1):
        files[f"head_{i}.joblib"] = (lambda p, h=heads[i]: joblib.dump(h, p))
        files[f"leace_{i}/.keep"] = (lambda p, m=maps[i]: (m.save(p.parent), p.write_text("")))
    rec = {"unit": name, "arm": "E", "seed": k, "reference": "official LEACE (concept-erasure 0.2.4) per recipient on "
           "U's encoder features; concept one-hot SEX; fitted on DEFENSE_FIT rows only, float64",
           "source_unit": u_unit_name, "source_model_pt_sha256": _sha_file(ud / "model.pt"),
           "source_complete_sha256": _sha_file(ud / "COMPLETE.json"), "identity_release_max_abs_diff": ident,
           "leace_provenance": prov, "leace": meta, "finalize": {"heads": hmeta, "leace": {
               i: {"native_check_fit_rows": native[i]} for i in (0, 1)}},
           "native_check_status": {i: native[i]["status"] for i in (0, 1)},
           "fit_role": "DEFENSE_FIT", "n_fit": int(len(tr)), "wall_s": time.time() - t0}
    FN.save_unit(udir(name, units_dir), files, rec)
    return json.loads((udir(name, units_dir) / "record.json").read_text())


# ------------------------------------------------------------------ FARE (reused official trees; new heads)
def verify_source_fare(k, i, cfg_id, D, zero=False, src_units=None, src_cache=None, reencode=True):
    """All reuse conditions for one predecessor FARE unit. Returns (cells in D row order, n_cells, provenance)."""
    tag = "Z" if zero else "c"
    nm = f"fare__s{k}__p{i}__{tag}{cfg_id}"
    src = Path(src_units or SRC_UNITS) / nm
    if not FN.unit_complete(src):
        raise SystemExit(f"REFUSED: predecessor unit {nm} is missing or not hash-complete")
    rec = json.loads((src / "record.json").read_text())
    exp_arm = "F0" if zero else "F"
    cfg = rec["config"]
    problems = []
    if rec.get("purpose") != i or rec.get("seed") != k or rec.get("arm") != exp_arm or cfg.get("id") != cfg_id:
        problems.append("seed/purpose/arm/config id mismatch")
    base = CFG[cfg_id]
    keys = ("max_leaf_nodes", "min_samples_leaf", "criterion")
    if any(cfg.get(x) != base[x] for x in keys):
        problems.append("tree budget differs from the registered grid")
    if zero and not (float(cfg.get("gamma", -1)) == 0.0 and cfg.get("zero_fairness_of") == base["name"]):
        problems.append("Z unit is not the zero-fairness twin of the registered config")
    if not zero and float(cfg.get("gamma", -1)) != base["gamma"]:
        problems.append("gamma differs from the registered grid")
    cache = Path(src_cache or SRC_FARE_CACHE) / rec["fare_uid"]
    if not _complete_dir(cache):
        problems.append("FARE cache unit not hash-complete")
    frec = json.loads((cache / "rec.json").read_text())
    tr = D["idx"]["DEFENSE_FIT"]
    y = D["y"][TASKS[i]]
    fit_rows = _sha_arrays(np.ascontiguousarray(D["X"][tr], dtype=np.float64))
    fit_labels = _sha_arrays(y[tr].astype(np.int64), D["sex"][tr].astype(np.int64))
    role_ok = (frec.get("n_fit") == len(tr) and frec.get("fit_rows_sha256") == fit_rows
               and frec.get("fit_labels_sha256") == fit_labels)
    if not role_ok:
        problems.append("tree was not fitted on exactly DEFENSE_FIT rows/labels")
    z = np.load(src / "release.npz")
    cells_cache = np.load(cache / "cells.npy", allow_pickle=False)
    if frec.get("n_all") != N_SOURCE_ROWS or not np.array_equal(cells_cache, z["cells"]):
        problems.append("release cells differ from the FARE cache cells")
    take = _rows_to_D(z["row_id"], D)
    cells = np.asarray(z["cells"])[take].astype(np.int64)
    n_cells = int(frec["n_cells"])
    if cells.max() >= n_cells or cells.min() < 0:
        problems.append("cell id outside the tree's cells")
    reenc = None
    if reencode:
        with _fare_units_root(RGJ_FARE_CACHE) as FO:
            model = FO.FareModel.load(cache / "model")
            if model.fingerprint != frec["model_fingerprint"]:
                problems.append("model fingerprint differs from the record")
            enc = np.asarray(FO.encode(model, np.ascontiguousarray(D["X"], dtype=np.float64))).astype(np.int64)
        reenc = bool(np.array_equal(enc, cells))
        if not reenc:
            problems.append("official re-encode of D features does not reproduce the mapped cells")
    if problems:
        raise SystemExit(f"REFUSED: {nm}: " + "; ".join(problems))
    prov = {"source_unit": nm, "source_complete_sha256": _sha_file(src / "COMPLETE.json"),
            "fare_uid": rec["fare_uid"], "fare_cache_complete_sha256": _sha_file(cache / "COMPLETE.json"),
            "model_fingerprint": frec["model_fingerprint"], "fare_commit": frec.get("fare_commit"),
            "config": cfg, "fit_role_check": {"old_role": "defense_train", "new_role": "DEFENSE_FIT", "n_fit": len(tr),
                                               "fit_rows_sha256": fit_rows, "fit_labels_sha256": fit_labels,
                                               "matches_record": True},
            "rows_mapped_by_row_id": int(len(take)), "official_reencode_matches": reenc}
    return cells, n_cells, prov


def write_fare_unit(name, k, i, cells, n_cells, cfg, D, prov, units_dir=None, zero=False):
    tr, va = D["idx"]["DEFENSE_FIT"], D["idx"]["HEAD_VALIDATION"]
    y = D["y"][TASKS[i]]
    R = np.eye(n_cells)[cells]
    head, hm = fit_head(R, y, tr, va, KS[i])
    cen, P, hard = outputs(head, R)
    rec = {"unit": name, "arm": "F0" if zero else "F", "purpose": i, "task": TASKS[i], "seed": k, "config": cfg,
           "n_cells": n_cells, "provenance": prov, "head": hm, "head_roles": {"fit": "DEFENSE_FIT", "C": "HEAD_VALIDATION"},
           "certificate": CERT_STATUS}
    FN.save_unit(udir(name, units_dir), {
        "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], cells=cells, r=R, c=cen, p=P, hard=hard),
        "head.joblib": lambda p: joblib.dump(head, p)}, rec)
    return json.loads((udir(name, units_dir) / "record.json").read_text())


def fare_import(k, i, cfg_id, D, zero=False, units_dir=None, src_units=None, src_cache=None, reencode=True):
    name = f"fare__s{k}__p{i}__{'Z' if zero else 'c'}{cfg_id}"
    if FN.unit_complete(udir(name, units_dir)):
        return json.loads((udir(name, units_dir) / "record.json").read_text())
    cells, n_cells, prov = verify_source_fare(k, i, cfg_id, D, zero, src_units, src_cache, reencode)
    cfg = json.loads((Path(src_units or SRC_UNITS) / prov["source_unit"] / "record.json").read_text())["config"]
    prov["origin"] = "predecessor official FARE tree reused (not refitted)"
    return write_fare_unit(name, k, i, cells, n_cells, cfg, D, prov, units_dir, zero)


def fare_fit_zero_twin(k, i, cfg_id, D, units_dir=None, allow_fit=False):
    """New official zero-fairness fit at a registered config (only when no predecessor Z unit exists)."""
    name = f"fare__s{k}__p{i}__Z{cfg_id}"
    if FN.unit_complete(udir(name, units_dir)):
        return json.loads((udir(name, units_dir) / "record.json").read_text())
    if not allow_fit:
        raise SystemExit(f"REFUSED: {name} needs a new official FARE fit; rerun with allow_fit=True")
    from stored_model_eval.guards import FitAuthorization
    tr = D["idx"]["DEFENSE_FIT"]
    y = D["y"][TASKS[i]]
    t0 = time.time()
    with _fare_units_root(RGJ_FARE_CACHE) as FO:
        zc = FO.zero_fairness(CFG[cfg_id])
        uid = f"rgj__s{k}__p{i}__Z{cfg_id}"
        model, cells, frec = FO.fit_encode_cached(uid, np.ascontiguousarray(D["X"][tr], dtype=np.float64), y[tr],
                                                  D["sex"][tr], np.ascontiguousarray(D["X"], dtype=np.float64), zc,
                                                  seed=k, auth=FitAuthorization(execute_scientific_fits=True),
                                                  synthetic=False, ledger=lambda *a: None)
    prov = {"origin": "new official FARE zero-fairness fit (no predecessor Z unit at this config)", "fare_uid": uid,
            "fare_cache": "rgj_v1/fare_cache", "fit_role": "DEFENSE_FIT", "seed": k,
            "fare_rec": {kk: v for kk, v in frec.items() if kk != "cells"}, "wall_s": time.time() - t0}
    return write_fare_unit(name, k, i, np.asarray(cells).astype(np.int64), int(frec["n_cells"]), zc, D, prov,
                           units_dir, zero=True)


def fare_pair_views(n1, n2, units_dir=None):
    a, b = np.load(udir(n1, units_dir) / "release.npz"), np.load(udir(n2, units_dir) / "release.npz")
    v1 = np.hstack([a["r"], a["c"]])
    v2 = np.hstack([b["r"], b["c"]])
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2]),
            "out": {"p1": a["p"], "p2": b["p"], "hard1": a["hard"], "hard2": b["hard"]},
            "r": [a["r"], b["r"]]}


def fare_units(k, D, units_dir=None, reencode=True, src_units=None, src_cache=None):
    """Re-headed FARE units for every registered config and purpose (reused predecessor trees). Skips complete units."""
    out = {}
    for i in (0, 1):
        for cfg in FARE_GRID:
            r = fare_import(k, i, cfg["id"], D, units_dir=units_dir, src_units=src_units, src_cache=src_cache,
                            reencode=reencode)
            out[r["unit"]] = {"n_cells": r["n_cells"], "selected_C": r["head"]["selected_C"]}
    return out


def fare_inner(name, D, units_dir=None):
    """Cached inner record of one FARE purpose unit: local SEX AUC (finite single view) + deployed-head utility."""
    iname = f"inner__{name}"
    if FN.unit_complete(udir(iname, units_dir)):
        return json.loads((udir(iname, units_dir) / "record.json").read_text())
    rec0 = json.loads((udir(name, units_dir) / "record.json").read_text())
    i = rec0["purpose"]
    z = np.load(udir(name, units_dir) / "release.npz")
    assert np.array_equal(z["row_id"], D["row_id"])
    auc, ar = AU.inner_local(np.hstack([z["r"], z["c"]]), D, finite=True)
    u = inner_utility({f"hard{i + 1}": z["hard"]}, D, purposes=(i,))[i]
    rec = {"unit": iname, "of": name, "purpose": i, "recovery_local": auc, "recovery": ar, "utility": u}
    FN.save_unit(udir(iname, units_dir), {}, rec)
    return rec


def pick_fare_config(rows):
    """jcv/select.py rule: lowest inner local AUC among gate-passing configs, ties -> lower id; none -> closest
    (largest worst-gate margin, ties -> lower id), descriptive only."""
    adm = [x for x in rows if x["gate_ok"]]
    if adm:
        return {"status": "NOMINEE", **min(adm, key=lambda x: (round(x["R_local"], 12), x["config"]))}
    return {"status": "NO_FEASIBLE_NOMINEE", "descriptive": "INFEASIBLE",
            **max(rows, key=lambda x: (x["margin"], -x["config"]))}


def _pair_inner(k, arm, n1, n2, D, units_dir):
    iname = f"inner__pair__s{k}__{arm}"
    d = udir(iname, units_dir)
    if FN.unit_complete(d):
        rec = json.loads((d / "record.json").read_text())
        if rec["of"] != [n1, n2]:
            raise RuntimeError(f"{iname} exists for {rec['of']}, not {[n1, n2]}; move it aside before reselecting")
        return rec
    V = fare_pair_views(n1, n2, units_dir)
    assert len(V["v1"]) == len(D["row_id"])
    t0 = time.time()
    rec = {"unit": iname, "of": [n1, n2], "recovery": AU.inner_audit({w: V[w] for w in ("v1", "v2", "pair")}, D,
                                                                    finite=True),
           "utility": inner_utility(V["out"], D)}
    rec["wall_s"] = time.time() - t0
    FN.save_unit(d, {}, rec)
    return rec


def select_fare(k, D, units_dir, uref, guard, allow_f0_fit=True, src_units=None, src_cache=None, reencode=True):
    """Registered FARE / F0 selection on the inner roles (see module docstring).

    uref  = {0: {"acc", "const_acc"}, 1: {...}} of U on INNER_SELECTION (deployed heads)
    guard = {"v1": x, "v2": y}: L-R's inner local AUCs (zero buffer: feasible needs auc_vi <= guard[vi])
    Returns {"F": arm, "F0": arm, "per_purpose": {0: ..., 1: ...}, "certificate": CERT_STATUS, "rule": ...}; each arm
    has status, units, configs, auc {v1, v2, pair}, utility, gate_margins, gates_ok, worst_gate_margin, guard_ref,
    guard_excess, guard_ok, feasible, worse_local, mean_local, worst_violation, selected_attackers.
    F: NOMINEE iff both purposes have a gate-passing config AND the pair passes the gates AND the guard.
    F0: status from its own gates + guard (NOMINEE / INFEASIBLE_CONTROL); its pairing to F is a separate field.
    """
    from rgj.select import gate_margins, gates_ok, worst_gate   # the lead's gate definitions (identical formulas)
    uref = {int(i): v for i, v in uref.items()}
    per = {}
    for i in (0, 1):
        rows = []
        for cfg in FARE_GRID:
            nm = f"fare__s{k}__p{i}__c{cfg['id']}"
            if not FN.unit_complete(udir(nm, units_dir)):
                raise SystemExit(f"REFUSED: {nm} missing; run fare_units first")
            r = fare_inner(nm, D, units_dir)
            ok, margin, g = gate_one(r["utility"], uref[i])
            rows.append({"config": cfg["id"], "unit": nm, "gate_ok": ok, "margin": margin, "gates": g,
                         "R_local": r["recovery_local"], "acc": r["utility"]["acc"],
                         "const_acc": r["utility"]["const_acc"], "selected_attacker": r["recovery"]["selected"]["v"]})
        per[i] = {**pick_fare_config(rows), "table": rows}
    cfgs = [per[0]["config"], per[1]["config"]]
    f0_origin = {}
    for i in (0, 1):
        src = Path(src_units or SRC_UNITS) / f"fare__s{k}__p{i}__Z{cfgs[i]}"
        if src.exists():
            r = fare_import(k, i, cfgs[i], D, zero=True, units_dir=units_dir, src_units=src_units, src_cache=src_cache,
                            reencode=reencode)
        else:
            r = fare_fit_zero_twin(k, i, cfgs[i], D, units_dir, allow_fit=allow_f0_fit)
        f0_origin[i] = r["provenance"]["origin"]
    out = {"seed": k, "per_purpose": per, "certificate": CERT_STATUS,
           "rule": "per purpose: lowest inner local SEX AUC among configs passing that purpose's gates vs U "
                   "(INNER_SELECTION, deployed heads); ties -> lower id (jcv/select.py); F0 = zero-fairness twin at "
                   "F's selected configs; pair feasibility = gates vs U and local AUC_i <= guard_i (zero buffer)"}
    purposes_ok = all(per[i]["status"] == "NOMINEE" for i in (0, 1))
    for arm, tag in (("F", "c"), ("F0", "Z")):
        n1, n2 = f"fare__s{k}__p0__{tag}{cfgs[0]}", f"fare__s{k}__p1__{tag}{cfgs[1]}"
        r = _pair_inner(k, arm, n1, n2, D, units_dir)
        util = {int(i): v for i, v in r["utility"].items()}
        a = {w: r["recovery"]["auc"][w] for w in ("v1", "v2", "pair")}
        gm = gate_margins(util, uref)
        exc = {w: a[w] - guard[w] for w in ("v1", "v2")}
        rec = {"units": [n1, n2], "configs": cfgs, "auc": a, "utility": util, "gate_margins": gm,
               "gates_ok": gates_ok(gm), "worst_gate_margin": worst_gate(gm), "guard_ref": dict(guard),
               "guard_excess": exc, "guard_ok": all(x <= 0 for x in exc.values()),
               "worse_local": max(a["v1"], a["v2"]), "mean_local": (a["v1"] + a["v2"]) / 2,
               "selected_attackers": r["recovery"]["selected"], "inner_unit": r["unit"], "certificate": CERT_STATUS}
        rec["feasible"] = rec["gates_ok"] and rec["guard_ok"]
        rec["worst_violation"] = max([0.0, -rec["worst_gate_margin"]] + [max(0.0, x) for x in exc.values()])
        if arm == "F":
            rec["per_purpose_status"] = {i: per[i]["status"] for i in (0, 1)}
            rec["status"] = "NOMINEE" if (purposes_ok and rec["feasible"]) else "NO_FEASIBLE_NOMINEE"
            if rec["status"] != "NOMINEE":
                rec["descriptive"] = "INFEASIBLE (closest configuration kept descriptively)"
        else:
            rec["own_gates_per_purpose"] = {i: gate_one(util[i], uref[i])[0] for i in (0, 1)}
            rec["status"] = "NOMINEE" if rec["feasible"] else "INFEASIBLE_CONTROL"
            rec["pairing_to_F"] = {"twin_of_F_configs": cfgs, "F_per_purpose_status": {i: per[i]["status"] for i in (0, 1)},
                                   "F_status": out["F"]["status"], "origin_per_purpose": f0_origin}
        out[arm] = rec
    return out


def main(argv=None):
    import argparse

    from rgj import data as DA
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("leace", "fare-units"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--u-unit", default=None, help="U unit name template, e.g. tl__s{k}__e30 ({k} = seed)")
    ap.add_argument("--units-dir", default=None)
    ap.add_argument("--no-reencode", action="store_true")
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    D = DA.load()
    for k in a.seeds:
        if a.cmd == "leace":
            if not a.u_unit:
                raise SystemExit("--u-unit is required")
            r = leace_unit(k, a.u_unit.format(k=k), D, a.units_dir)
            print(r["unit"], r["native_check_status"])
        else:
            print(json.dumps(fare_units(k, D, a.units_dir, reencode=not a.no_reencode), indent=1))


if __name__ == "__main__":
    main()
