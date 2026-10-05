"""Official references for the strength-matched feedback study, all REFIT on the new fitting rows (audit/baseline owner).

Nothing from an earlier study is reused: the predecessors' LEACE maps and FARE trees were fitted on old defense_train,
which contains this study's NEW_DEVELOPMENT_ASSESSMENT rows. Nothing here re-implements a reference method: LEACE is
EleutherAI concept-erasure via the pinned wrapper stored_model_eval.defenses.fit_leace (official LeaceFitter defaults,
float64); FARE is the official eth-sri/fare tree via oar.fare_official.fit_encode_cached (pinned commit, patched
scikit-learn in its own environment). Roles (smf.data aliases): DEFENSE_FIT = NEW_DEFENSE_FIT, HEAD_VALIDATION
preserved, inner audits AUDIT_FIT -> INNER_SELECTION. No function here reads assessment labels; every fit refuses
negative (sealed) labels on its rows.

LEACE reference E (`leace_unit(k, u_unit_name, D, units_dir)`)
  Encoder features of the study's task-only model U (U's model.pt for jcv.train.Model(83, [2, 6], k); h_i =
  model.encode(i, X) in float32, cast to float64). One official LEACE map PER RECIPIENT, concept = one-hot SEX, fitted
  on DEFENSE_FIT (= NEW_DEFENSE_FIT) rows only. Release r_i = map_i(h_i) for every D row; deployed heads refitted with
  rgj.finalize.heads_and_outputs (DEFENSE_FIT fit, HEAD_VALIDATION C). LEACE's native condition is checked on the
  fitting rows only (finite-sample linear guardedness of those rows; not a held-out or nonlinear claim). U's released
  features must equal the recomputed encoder outputs (identity release, max |diff| <= 1e-6) or the unit is refused.
  Unit: lc__s{k}__E (model.pt copied, head_{0,1}.joblib, leace_{0,1}/ map files, release.npz in the rgj format).

FARE reference F and zero-fairness twin F0 (`fare_units`, `select_fare`)
  New official fits for all 6 registered configurations (jcv.run.FARE_GRID, unchanged) x 2 purposes per seed:
  fit inputs = D["X"][DEFENSE_FIT] (float64), task labels and SEX of those rows, seed = k; the label-free official
  encode is applied to every D row. FARE cache: <PRIVATE_CACHE>/smf_v1/fare_cache/smf__s{k}__p{i}__{c|Z}{id} (new;
  fit_encode_cached refuses a cached unit whose recorded inputs differ). Each tree's recorded n_fit, fit-row and
  fit-label sha256 are checked against DEFENSE_FIT before a release is written. Release (rgj format for FARE units):
  r = one-hot cells (n_cells = the tree's cell count), deployed head jcv.finalize.fit_head on DEFENSE_FIT with C by
  HEAD_VALIDATION log loss, centred logits / probabilities / hard decisions from that head. Units
  fare__s{k}__p{i}__c{id} (F grid) and fare__s{k}__p{i}__Z{id} (F0).
  Selection (`select_fare`, the predecessor's published rule, jcv/select.py): per purpose, among grid configurations
  passing that purpose's task gates versus U on INNER_SELECTION (deployed heads), the lowest inner local SEX AUC
  (smf.audit.inner_local on the finite view [one-hot cells, centred logits]; inner slate + cell-conditional attackers);
  ties -> lower id. No gate-passing configuration -> NO_FEASIBLE_NOMINEE; the closest configuration (largest worst-gate
  margin, ties -> lower id) is kept descriptively. F = the two purposes' selected releases; F is NOMINEE iff both
  purposes have a gate-passing configuration AND the pair passes the gates versus U. There is NO local guard (Phase B
  C* has none); the inner AUCs {v1, v2, pair} are returned for the lead's C* rule. F0 = the zero-fairness twin
  (oar.fare_official.zero_fairness: same tree budget, gamma = 0) at F's selected configuration per purpose, fitted if
  absent; F0 status = NOMINEE iff its own pair passes the gates, else INFEASIBLE_CONTROL; its pairing to F is a
  separate field. Inner pair audits are cached as units inner__pair__s{k}__F / inner__pair__s{k}__F0.

Certificates: NOT AVAILABLE under the new roles (CERT_STATUS). The old certification pool is excluded from every
operation of this study and the role manifest defines no new certification role; no FARE certificate is computed,
and none is ever renamed an attack bound.

Utility gates (prompt section 9; INNER_SELECTION, deployed heads; const = DEFENSE_FIT majority class)
  G1  Acc >= Acc(U) - 0.01        G2  Acc - const >= 0.8 * (Acc(U) - const)        G3  Acc - const >= 0.03

API: leace_unit(k, u_unit_name, D, units_dir); fare_units(k, D, units_dir); select_fare(k, D, units_dir, uref).
CLI (writes units; run by the lead on study units):
  OMP_NUM_THREADS=1 python -m smf.baselines leace --seeds 0 1 2 --u-unit <U unit template with {k}>
  OMP_NUM_THREADS=1 python -m smf.baselines fare-units --seeds 0 1 2
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
from rgj import finalize as FN
from smf import audit as AU

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private"
UNITS = PRIV / "smf_v1" / "run" / "units"
FARE_CACHE = PRIV / "smf_v1" / "fare_cache"
KS = [2, 6]
TASKS = ("income", "occupation_group")
FIT_ROLE, HEAD_ROLE = "DEFENSE_FIT", "HEAD_VALIDATION"


def _registered_grid():
    from jcv.run import FARE_GRID as G      # the predecessor's bounded configuration bank, pinned (not copied)
    return [dict(c) for c in G]


FARE_GRID = _registered_grid()
CFG = {c["id"]: c for c in FARE_GRID}
CERT_STATUS = {
    "status": "NOT_AVAILABLE_UNDER_NEW_ROLES",
    "reason": "the old certification pool is excluded from this study and the new role manifest defines no "
              "certification role; no FARE certificate is computed",
    "note": "FARE's certificate is a demographic-parity certificate with its own premises, not an attribute-recovery "
            "bound; it is never reported as an attack bound.",
}


# ------------------------------------------------------------------ shared utility gates
def constants(D):
    tr = D["idx"][FIT_ROLE]
    return {i: int(np.argmax(np.bincount(D["y"][t][tr]))) for i, t in enumerate(TASKS)}


def inner_utility(out, D, purposes=(0, 1)):
    """Deployed-head accuracy and constant accuracy on INNER_SELECTION. out has hard1/hard2 (D-row aligned)."""
    v = D["idx"]["INNER_SELECTION"]
    cst = constants(D)
    res = {}
    for i in purposes:
        y = D["y"][TASKS[i]][v]
        AU.check_labels(y, np.arange(len(y)))
        res[i] = {"acc": float((np.asarray(out[f"hard{i + 1}"])[v] == y).mean()),
                  "const_acc": float((y == cst[i]).mean()), "const_class": cst[i]}
    return res


def gate_one(u, uU):
    a, aU, c = u["acc"], uU["acc"], u["const_acc"]
    g = {"G1": a - (aU - 0.01), "G2": (a - c) - 0.8 * (aU - c), "G3": (a - c) - 0.03}
    return all(x >= 0 for x in g.values()), min(g.values()), g


def gate_margins(util, uref):
    return {i: gate_one(util[i], uref[i])[2] for i in (0, 1)}


def gates(util, uref):
    """Both tasks. Returns (ok, worst margin, per-task margins)."""
    per = {i: gate_one(util[i], uref[i]) for i in (0, 1)}
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


def _rows_to_D(row_id_src, D):
    pos = {int(r): j for j, r in enumerate(row_id_src)}
    missing = [int(r) for r in D["row_id"] if int(r) not in pos]
    if missing:
        raise ValueError(f"{len(missing)} D rows absent from the source release")
    return np.array([pos[int(r)] for r in D["row_id"]])


def _fit_rows(D):
    tr = np.asarray(D["idx"][FIT_ROLE])
    if "NEW_DEFENSE_FIT" in D["idx"] and not np.array_equal(D["idx"]["NEW_DEFENSE_FIT"], tr):
        raise SystemExit("REFUSED: DEFENSE_FIT is not the NEW_DEFENSE_FIT alias")
    return tr


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
    tr = _fit_rows(D)
    AU.check_labels(D["sex"], tr)
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
    rec = {"unit": name, "arm": "E", "seed": k, "reference": "official LEACE (concept-erasure) per recipient on U's "
           "encoder features; concept one-hot SEX; fitted on NEW_DEFENSE_FIT rows only, float64",
           "source_unit": u_unit_name, "source_model_pt_sha256": _sha_file(ud / "model.pt"),
           "source_complete_sha256": _sha_file(ud / "COMPLETE.json"), "identity_release_max_abs_diff": ident,
           "leace_provenance": prov, "leace": meta, "finalize": {"heads": hmeta, "leace": {
               i: {"native_check_fit_rows": native[i]} for i in (0, 1)}},
           "native_check_status": {i: native[i]["status"] for i in (0, 1)},
           "fit_role": "NEW_DEFENSE_FIT (alias DEFENSE_FIT)", "n_fit": int(len(tr)),
           "fit_row_id_sha256": _sha_arrays(np.sort(D["row_id"][tr]).astype(np.int64)), "wall_s": time.time() - t0}
    FN.save_unit(udir(name, units_dir), files, rec)
    return json.loads((udir(name, units_dir) / "record.json").read_text())


# ------------------------------------------------------------------ FARE (new official trees; new heads)
def write_fare_unit(name, k, i, cells, n_cells, cfg, D, prov, units_dir=None, zero=False):
    tr, va = _fit_rows(D), D["idx"][HEAD_ROLE]
    y = D["y"][TASKS[i]]
    AU.check_labels(y, tr, va)
    R = np.eye(n_cells)[cells]
    head, hm = fit_head(R, y, tr, va, KS[i])
    cen, P, hard = outputs(head, R)
    rec = {"unit": name, "arm": "F0" if zero else "F", "purpose": i, "task": TASKS[i], "seed": k, "config": cfg,
           "n_cells": n_cells, "provenance": prov, "head": hm,
           "head_roles": {"fit": "NEW_DEFENSE_FIT (alias DEFENSE_FIT)", "C": HEAD_ROLE}, "certificate": CERT_STATUS}
    FN.save_unit(udir(name, units_dir), {
        "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], cells=cells, r=R, c=cen, p=P, hard=hard),
        "head.joblib": lambda p: joblib.dump(head, p)}, rec)
    return json.loads((udir(name, units_dir) / "record.json").read_text())


def fare_fit(k, i, cfg_id, D, zero=False, units_dir=None, fare_cache=None, synthetic=False, auth=None):
    """One NEW official FARE fit (DEFENSE_FIT rows, seed k) + re-headed release unit. Skips a complete unit."""
    name = f"fare__s{k}__p{i}__{'Z' if zero else 'c'}{cfg_id}"
    if FN.unit_complete(udir(name, units_dir)):
        return json.loads((udir(name, units_dir) / "record.json").read_text())
    from stored_model_eval.guards import FitAuthorization
    auth = auth or (FitAuthorization.synthetic_only() if synthetic else FitAuthorization(execute_scientific_fits=True))
    tr = _fit_rows(D)
    y, S = D["y"][TASKS[i]], D["sex"]
    AU.check_labels(y, tr)
    AU.check_labels(S, tr)
    Xf = np.ascontiguousarray(D["X"][tr], dtype=np.float64)
    Xall = np.ascontiguousarray(D["X"], dtype=np.float64)
    uid = f"smf__s{k}__p{i}__{'Z' if zero else 'c'}{cfg_id}"
    ledger = []
    t0 = time.time()
    with _fare_units_root(fare_cache or FARE_CACHE) as FO:
        cfg = FO.zero_fairness(CFG[cfg_id]) if zero else dict(CFG[cfg_id])
        model, cells, frec = FO.fit_encode_cached(uid, Xf, y[tr], S[tr], Xall, cfg, seed=k, auth=auth,
                                                  synthetic=synthetic, ledger=lambda *a: ledger.append(list(a)))
        fare_commit = FO.FARE_COMMIT
    problems = []
    if frec.get("n_fit") != len(tr) or frec.get("fit_rows_sha256") != _sha_arrays(Xf):
        problems.append("tree fit rows are not DEFENSE_FIT")
    if frec.get("fit_labels_sha256") != _sha_arrays(np.asarray(y[tr]).astype(np.int64), np.asarray(S[tr]).astype(np.int64)):
        problems.append("tree fit labels are not DEFENSE_FIT's")
    if frec.get("n_all") != len(D["row_id"]):
        problems.append("encode did not cover every D row")
    cells = np.asarray(cells).astype(np.int64)
    n_cells = int(frec["n_cells"])
    if cells.min() < 0 or cells.max() >= n_cells:
        problems.append("cell id outside the tree's cells")
    if problems:
        raise SystemExit(f"REFUSED: {name}: " + "; ".join(problems))
    prov = {"origin": "NEW official FARE fit on NEW_DEFENSE_FIT rows (nothing reused)", "fare_uid": uid,
            "fare_cache": "<PRIVATE_CACHE>/smf_v1/fare_cache" if fare_cache is None else "custom (test)",
            "fare_cache_complete_sha256": _sha_file(Path(fare_cache or FARE_CACHE) / uid / "COMPLETE.json"),
            "fare_commit": fare_commit, "seed": k, "fit_role": "NEW_DEFENSE_FIT (alias DEFENSE_FIT)",
            "n_fit": int(len(tr)), "fit_rows_sha256": frec["fit_rows_sha256"],
            "fit_labels_sha256": frec["fit_labels_sha256"], "model_fingerprint": frec.get("model_fingerprint"),
            "cache_hit": bool(frec.get("cache_hit", False)), "ledger": ledger, "synthetic": bool(synthetic),
            "fare_rec": {kk: v for kk, v in frec.items() if kk not in ("cells", "n_fit_by_group_per_cell")},
            "wall_s": time.time() - t0}
    return write_fare_unit(name, k, i, cells, n_cells, cfg, D, prov, units_dir, zero)


def fare_units(k, D, units_dir=None, fare_cache=None, synthetic=False, auth=None, grid=None):
    """New official FARE fits for every registered config x purpose (seed k) + re-headed units. Skips complete units."""
    out = {}
    for i in (0, 1):
        for cfg in (grid or FARE_GRID):
            r = fare_fit(k, i, cfg["id"], D, False, units_dir, fare_cache, synthetic, auth)
            out[r["unit"]] = {"n_cells": r["n_cells"], "selected_C": r["head"]["selected_C"]}
    return out


def fare_pair_views(n1, n2, units_dir=None):
    a, b = np.load(udir(n1, units_dir) / "release.npz"), np.load(udir(n2, units_dir) / "release.npz")
    v1 = np.hstack([a["r"], a["c"]])
    v2 = np.hstack([b["r"], b["c"]])
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2]),
            "out": {"p1": a["p"], "p2": b["p"], "hard1": a["hard"], "hard2": b["hard"]},
            "r": [a["r"], b["r"]]}


def fare_release_dict(n1, n2, units_dir=None):
    """The two purpose units as one release in the neural format (r1, c1, p1, hard1, r2, ..., row_id)."""
    a, b = np.load(udir(n1, units_dir) / "release.npz"), np.load(udir(n2, units_dir) / "release.npz")
    assert np.array_equal(a["row_id"], b["row_id"])
    return {"row_id": a["row_id"], "r1": a["r"], "c1": a["c"], "p1": a["p"], "hard1": a["hard"],
            "r2": b["r"], "c2": b["c"], "p2": b["p"], "hard2": b["hard"]}


def fare_inner(name, D, units_dir=None):
    """Cached inner record of one FARE purpose unit: local SEX AUC (finite single view) + deployed-head utility."""
    iname = f"inner__{name}"
    src = _sha_file(udir(name, units_dir) / "COMPLETE.json")
    if FN.unit_complete(udir(iname, units_dir)):
        rec = json.loads((udir(iname, units_dir) / "record.json").read_text())
        if rec.get("of_complete_sha256") != src:
            raise RuntimeError(f"{iname} was computed for another version of {name}; move it aside")
        return rec
    rec0 = json.loads((udir(name, units_dir) / "record.json").read_text())
    i = rec0["purpose"]
    z = np.load(udir(name, units_dir) / "release.npz")
    assert np.array_equal(z["row_id"], D["row_id"])
    auc, ar = AU.inner_local(np.hstack([z["r"], z["c"]]), D, finite=True)
    u = inner_utility({f"hard{i + 1}": z["hard"]}, D, purposes=(i,))[i]
    rec = {"unit": iname, "of": name, "of_complete_sha256": src, "purpose": i, "recovery_local": auc, "recovery": ar,
           "utility": u}
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
    src = [_sha_file(udir(n, units_dir) / "COMPLETE.json") for n in (n1, n2)]
    if FN.unit_complete(d):
        rec = json.loads((d / "record.json").read_text())
        if rec["of"] != [n1, n2] or rec.get("of_complete_sha256") != src:
            raise RuntimeError(f"{iname} exists for {rec['of']} (or other unit versions), not {[n1, n2]}; move it "
                               "aside before reselecting")
        return rec
    V = fare_pair_views(n1, n2, units_dir)
    assert len(V["v1"]) == len(D["row_id"])
    t0 = time.time()
    rec = {"unit": iname, "of": [n1, n2], "of_complete_sha256": src,
           "recovery": AU.inner_audit({w: V[w] for w in ("v1", "v2", "pair")}, D, finite=True),
           "utility": inner_utility(V["out"], D)}
    rec["wall_s"] = time.time() - t0
    FN.save_unit(d, {}, rec)
    return rec


def select_fare(k, D, units_dir, uref, allow_f0_fit=True, fare_cache=None, synthetic=False, auth=None, grid=None):
    """Registered FARE / F0 selection on the inner roles (module docstring). No local guard (Phase B C* has none).

    uref = {0: {"acc", "const_acc"}, 1: {...}}: U's deployed-head utility on INNER_SELECTION.
    Returns {"F": arm, "F0": arm, "per_purpose": {0: ..., 1: ...}, "certificate": CERT_STATUS, "rule": ...}; each arm
    has status, units, configs, auc {v1, v2, pair} (inner, finite=True), utility, gate_margins, gates_ok,
    worst_gate_margin, worse_local, mean_local, selected_attackers, inner_unit, certificate.
    """
    uref = {int(i): v for i, v in uref.items()}
    grid = grid or FARE_GRID
    per = {}
    for i in (0, 1):
        rows = []
        for cfg in grid:
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
        nmZ = f"fare__s{k}__p{i}__Z{cfgs[i]}"
        if not FN.unit_complete(udir(nmZ, units_dir)) and not allow_f0_fit:
            raise SystemExit(f"REFUSED: {nmZ} needs a new official FARE fit; rerun with allow_f0_fit=True")
        r = fare_fit(k, i, cfgs[i], D, True, units_dir, fare_cache, synthetic, auth)
        f0_origin[i] = r["provenance"]["origin"]
    out = {"seed": k, "per_purpose": per, "certificate": CERT_STATUS, "local_guard": None,
           "rule": "per purpose: lowest inner local SEX AUC among configs passing that purpose's gates vs U "
                   "(INNER_SELECTION, deployed heads); ties -> lower id (jcv/select.py); none -> closest by worst-gate "
                   "margin (descriptive); F0 = zero-fairness twin at F's selected configs; pair status = both purposes "
                   "admissible and pair gates vs U; no local guard"}
    purposes_ok = all(per[i]["status"] == "NOMINEE" for i in (0, 1))
    for arm, tag in (("F", "c"), ("F0", "Z")):
        n1, n2 = f"fare__s{k}__p0__{tag}{cfgs[0]}", f"fare__s{k}__p1__{tag}{cfgs[1]}"
        r = _pair_inner(k, arm, n1, n2, D, units_dir)
        util = {int(i): v for i, v in r["utility"].items()}
        a = {w: r["recovery"]["auc"][w] for w in ("v1", "v2", "pair")}
        ok, worst, gm = gates(util, uref)
        rec = {"units": [n1, n2], "configs": cfgs, "auc": a, "auc_kind": "inner (smf.audit.inner_audit, finite=True)",
               "utility": util, "gate_margins": gm, "gates_ok": ok, "worst_gate_margin": worst,
               "worse_local": max(a["v1"], a["v2"]), "mean_local": (a["v1"] + a["v2"]) / 2,
               "selected_attackers": r["recovery"]["selected"], "inner_unit": r["unit"], "certificate": CERT_STATUS}
        if arm == "F":
            rec["per_purpose_status"] = {i: per[i]["status"] for i in (0, 1)}
            rec["status"] = "NOMINEE" if (purposes_ok and ok) else "NO_FEASIBLE_NOMINEE"
            if rec["status"] != "NOMINEE":
                rec["descriptive"] = "INFEASIBLE (closest configuration kept descriptively)"
        else:
            rec["own_gates_per_purpose"] = {i: gate_one(util[i], uref[i])[0] for i in (0, 1)}
            rec["status"] = "NOMINEE" if ok else "INFEASIBLE_CONTROL"
            rec["pairing_to_F"] = {"twin_of_F_configs": cfgs, "F_per_purpose_status": {i: per[i]["status"] for i in (0, 1)},
                                   "F_status": out["F"]["status"], "origin_per_purpose": f0_origin}
        out[arm] = rec
    return out


def main(argv=None):
    import argparse

    from smf import data as DA
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("leace", "fare-units"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--u-unit", default=None, help="U unit name template with {k} = seed")
    ap.add_argument("--units-dir", default=None)
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
            print(json.dumps(fare_units(k, D, a.units_dir), indent=1))


if __name__ == "__main__":
    main()
