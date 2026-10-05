"""Admission of reusable strength-matched feedback (smf) checkpoints for the online-strength frontier study.

Owner: data/custody role. Prompt section 6: source smf checkpoints may be reused only when receipts prove the
unchanged OSF_DEFENSE_FIT (= smf NEW_DEFENSE_FIT), preprocessing, task/critic configuration, head-selection roles and
full exclusion of the consolidated assessment from their fitting. Missing fit provenance requires a fresh fit.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m osf.admit            # check, copy, write ADMISSION.json
    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m osf.admit --check-only

Candidates (smf private store <PRIVATE_CACHE>/smf_v1, READ-ONLY here):
  warm__s{k}                                    jcv.train.warm_start on NEW_DEFENSE_FIT (fixed critic-view head)
  tl__s{k}__e40                                 U: task-only, 40 epochs, smf stage "A" = salt 0
  raw__s{k}__RAW-{J,L}__b{0.1,0.3}__e{20,40}    rgj.train J-O / L-O unchanged, stage "B" = salt 0, 40 epochs,
                                                critic head = warm head (e20 = diagnostic snapshot of the same run)
  run__raw__s{k}__RAW-{J,L}__b{0.1,0.3}         training receipts of those runs (final.pt, diag)
  lc__s{k}__E                                   official LEACE maps per recipient on U's features (NEW_DEFENSE_FIT)
  fare__s{k}__p{i}__{c1..c6,Z1}                 official FARE releases (+ the fare_cache trees smf__s{k}__p{i}__*)
Checks per unit (every one must pass; nothing is admitted on an assumed match):
  integrity    COMPLETE.json verifies (jcv/rgj unit_complete) and lists every file; EVALUATION_LOCK pins where present
  recipe       record / receipt fields and the pinned source text that fixes the recipe (salts, epochs, critic head)
  fit rows     deployed heads' scaler moments = OSF_DEFENSE_FIT rows; LEACE fit-row id / feature / concept hashes;
               FARE tree n_fit, ordered fit-row and fit-label fingerprints, full input fingerprint, fit-row hash set
  preprocess   model.pt forward pass on osf.data X (refit numerics) reproduces the released features bitwise on every
               smf row; LEACE maps reproduce the LEACE release; FARE fit-row fingerprint on osf X
  head roles   saved head's HEAD_VALIDATION log loss equals its recorded table entry; selected C is the table argmin;
               head outputs reproduce the released centred logits / probabilities / decisions
  disjointness release rows contain no ORIG_ASSESSMENT / RGJ_DEV / CERT row; fitting rows are OSF_DEFENSE_FIT, which
               is group-disjoint from OSF_DEVELOPMENT_ASSESSMENT
Labels read: OSF_DEFENSE_FIT (procedure "training") and HEAD_VALIDATION (procedure "heads") through osf.data's
allowlist; assessment labels stay sealed. Then COPY (never move or delete) every admitted file into
<PRIVATE_CACHE>/osf_v1/admitted/<smf unit name>/ (trees: admitted/fare_cache/<uid>/) with sha256 re-verification,
and write results/pcrl_online_strength_frontier_v1/ADMISSION.json (placeholders only).

API for the runner:  admitted_path(smf_unit_name) -> Path ; admitted_fare_tree(uid) -> Path ; admission_record() -> dict
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_online_strength_frontier_v1"
SMF_PKG = WT / "results" / "pcrl_strength_matched_feedback_v1"
SMF = HOME / "PCRL_eval_cache_private" / "smf_v1"
SMF_UNITS = SMF / "run" / "units"
SMF_FARE = SMF / "fare_cache"
PRIV = HOME / "PCRL_eval_cache_private" / "osf_v1"
ADMITTED = PRIV / "admitted"
OUT = PKG / "ADMISSION.json"
SOURCE_PIN = "a9951ed2fed9943d445a208a8a7e456a56f39114"
SEEDS = (0, 1, 2)
BETAS = ("0.1", "0.3")
FARE_IDS = (1, 2, 3, 4, 5, 6)
STEPS_PER_EPOCH = 61
TASKS = ("income", "occupation_group")
KS = (2, 6)


# ------------------------------------------------------------------ API (used by osf.run)
def admission_record() -> dict:
    if not OUT.exists():
        raise FileNotFoundError("ADMISSION.json not written yet: run python -m osf.admit")
    return json.loads(OUT.read_text())


def _verify_copy(d: Path, complete_sha: str):
    from rgj.finalize import unit_complete
    if not (d / "COMPLETE.json").exists() or sha_file(d / "COMPLETE.json") != complete_sha or not unit_complete(d):
        raise RuntimeError(f"admitted copy {d.name} does not verify against ADMISSION.json")


def admitted_path(name: str, verify: bool = True) -> Path:
    """Directory of an admitted smf unit's private copy (hash-verified against ADMISSION.json by default)."""
    rec = admission_record()
    if rec.get("verdict") != "ADMITTED" or name not in rec["admitted"]:
        raise KeyError(f"{name} is not an admitted unit")
    d = ADMITTED / name
    if verify:
        _verify_copy(d, rec["admitted"][name]["complete_json_sha256"])
    return d


def admitted_fare_tree(uid: str, verify: bool = True) -> Path:
    """Directory of an admitted official FARE tree (fare_cache layout: model/, cells.npy, rec.json, COMPLETE.json)."""
    rec = admission_record()
    if rec.get("verdict") != "ADMITTED" or uid not in rec["fare_trees"]:
        raise KeyError(f"{uid} is not an admitted FARE tree")
    d = ADMITTED / "fare_cache" / uid
    if verify:
        c = json.loads((d / "COMPLETE.json").read_text())
        if sha_file(d / "COMPLETE.json") != rec["fare_trees"][uid]["complete_json_sha256"] or not all(
                sha_file(d / f) == h for f, h in c["files"].items()):
            raise RuntimeError(f"admitted FARE tree {uid} does not verify")
    return d


# ------------------------------------------------------------------ helpers
def sha_file(p, nocache=False) -> str:
    fd = os.open(p, os.O_RDONLY)
    try:
        if nocache:
            fcntl.fcntl(fd, 48, 1)    # F_NOCACHE (macOS): uncached read, not a physical cold-disk read
        h = hashlib.sha256()
        while True:
            b = os.read(fd, 1 << 22)
            if not b:
                break
            h.update(b)
        return h.hexdigest()
    finally:
        os.close(fd)


def sha_bytes(b) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_arrays(*arrays) -> str:        # oar.fare_official._sha256_arrays convention
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def jload(p):
    return json.loads(Path(p).read_text())


def pinned_text(path):
    return subprocess.run(["git", "-C", str(WT), "show", f"{SOURCE_PIN}:{path}"], capture_output=True,
                          text=True).stdout


def candidates():
    c = {}
    for k in SEEDS:
        c[f"warm__s{k}"] = {"kind": "warm", "seed": k}
        c[f"tl__s{k}__e40"] = {"kind": "U", "seed": k, "epoch": 40}
        for t in "JL":
            for b in BETAS:
                for e in (20, 40):
                    c[f"raw__s{k}__RAW-{t}__b{b}__e{e}"] = {"kind": "RAW", "seed": k, "treat": t, "beta": float(b),
                                                           "epoch": e}
                c[f"run__raw__s{k}__RAW-{t}__b{b}"] = {"kind": "RAW_receipt", "seed": k, "treat": t, "beta": float(b)}
        c[f"lc__s{k}__E"] = {"kind": "LEACE", "seed": k}
        for i in (0, 1):
            for cid in [f"c{j}" for j in FARE_IDS] + ["Z1"]:
                c[f"fare__s{k}__p{i}__{cid}"] = {"kind": "FARE", "seed": k, "purpose": i, "config": cid}
    return c


def fare_trees():
    return [f"smf__s{k}__p{i}__{cid}" for k in SEEDS for i in (0, 1) for cid in [f"c{j}" for j in FARE_IDS] + ["Z1"]]


def integrity(d: Path):
    from rgj.finalize import unit_complete
    c = jload(d / "COMPLETE.json")
    present = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()} - {"COMPLETE.json"}
    return {"unit_complete": bool(unit_complete(d)), "all_files_listed": present == set(c["files"]),
            "complete_json_sha256": sha_file(d / "COMPLETE.json"), "files_sha256": c["files"]}


# ------------------------------------------------------------------ checks
class Ctx:
    def __init__(self):
        from osf import data as OD
        self.OD = OD
        D = OD.load()
        assert D["sealed"]
        self.D = D
        self.fit = OD.labels_for(D, "training", "OSF_DEFENSE_FIT")
        self.hv = OD.labels_for(D, "heads", "HEAD_VALIDATION")
        self.pos = {int(r): j for j, r in enumerate(D["row_id"])}
        self.cat = np.where(D["role"] == "OSF_DEVELOPMENT_ASSESSMENT", D["pool"], D["role"])
        el = jload(SMF_PKG / "EVALUATION_LOCK.json")
        self.pins = {}
        for s in el["seeds"].values():
            self.pins.update(s["unit_file_sha256"])
        a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
        self.disjoint = {r: int(len(np.intersect1d(D["unit"][D["idx"][r]], D["unit"][a]))) for r in
                         ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")}

    def rows_of(self, rid):
        """Positions in D of release row ids (all must exist in D)."""
        return np.array([self.pos[int(r)] for r in rid])

    def composition(self, ix):
        c = self.cat[ix]
        return {k: int((c == k).sum()) for k in ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION",
                                                 "ORIG_ASSESSMENT", "RGJ_DEV", "SMF_DEV", "CERT")}


def head_checks(ctx, head, R_all, ix_D, y, K, rec_head, z_out=None):
    """R_all: release features (rows = release order); ix_D: D positions of those rows."""
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import StandardScaler

    from jcv.finalize import outputs
    D = ctx.D
    inv = {int(p): j for j, p in enumerate(ix_D)}
    fit_j = np.array([inv[int(p)] for p in ctx.fit])
    hv_j = np.array([inv[int(p)] for p in ctx.hv])
    sc = head.steps[0][1]
    ref = StandardScaler().fit(R_all[fit_j])
    ll = float(log_loss(y[ctx.hv], head.predict_proba(R_all[hv_j]), labels=list(range(K))))
    C = float(head.steps[1][1].C)
    table = {float(t["C"]): t["defense_val_log_loss"] for t in rec_head["table"]}
    best = None
    for t in rec_head["table"]:             # jcv.finalize.fit_head rule: strict improvement > 1e-12, ties -> smaller C
        if best is None or t["defense_val_log_loss"] < best[0] - 1e-12:
            best = (t["defense_val_log_loss"], float(t["C"]))
    out = {"scaler_n_equals_OSF_DEFENSE_FIT": bool(np.all(np.asarray(sc.n_samples_seen_) == len(ctx.fit))),
           "scaler_mean_var_bitwise_OSF_DEFENSE_FIT": bool(np.array_equal(sc.mean_, ref.mean_) and
                                                           np.array_equal(sc.var_, ref.var_)),
           "head_C_equals_recorded_selected_C": C == float(rec_head["selected_C"]),
           "HEAD_VALIDATION_log_loss_equals_recorded": ll == table.get(C),
           "HEAD_VALIDATION_log_loss_abs_diff": abs(ll - table[C]) if C in table else None,
           "selected_C_is_table_argmin": best is not None and best[1] == C}
    if z_out is not None:
        cen, P, hard = outputs(head, R_all)
        out["outputs_reproduce_release"] = bool(np.array_equal(cen, z_out[0]) and np.array_equal(P, z_out[1]) and
                                                np.array_equal(hard, z_out[2]))
    return out


def neural_release_checks(ctx, d: Path, model_state, k):
    import joblib
    import torch  # noqa: F401

    from jcv.train import Model
    from rgj.finalize import encode_all
    z = np.load(d / "release.npz", allow_pickle=False)
    rid = z["row_id"].astype(np.int64)
    ix = ctx.rows_of(rid)
    out = {"release_keys": sorted(z.files), "release_rows": int(len(rid)),
           "release_rows_by_category": ctx.composition(ix)}
    m = Model(ctx.D["X"].shape[1], list(KS), k)
    m.load_state_dict(model_state)
    m.eval()
    H = encode_all(m, ctx.D["X"][ix])
    out["forward_on_osf_X_reproduces_release_features_bitwise"] = bool(
        np.array_equal(H[0], z["r1"]) and np.array_equal(H[1], z["r2"]))
    rec = jload(d / "record.json")
    for i in (0, 1):
        h = joblib.load(d / f"head_{i}.joblib")
        out[f"head_{i}"] = head_checks(ctx, h, z[f"r{i + 1}"], ix, ctx.D["y"][TASKS[i]], KS[i], rec["heads"][str(i)],
                                       (z[f"c{i + 1}"], z[f"p{i + 1}"], z[f"hard{i + 1}"]))
    return out, z, ix


def check_unit(ctx, name, spec):
    import torch
    d = SMF_UNITS / name
    ck = {"integrity": integrity(d)}
    ck["evaluation_lock_pin"] = (ck["integrity"]["files_sha256"] == ctx.pins[name]) if name in ctx.pins else "not pinned"
    rec = jload(d / "record.json")
    k = spec["seed"]
    kind = spec["kind"]
    fails = []
    if kind == "warm":
        from jcv.train import Model
        st = torch.load(d / "warm.pt")
        ref = Model(ctx.D["X"].shape[1], list(KS), k).state_dict()
        ck["recipe"] = {"record_rows": rec.get("rows"), "record_schedule": rec.get("schedule"),
                        "rows_NEW_DEFENSE_FIT_15434": rec.get("rows") == "NEW_DEFENSE_FIT (15,434)",
                        "schedule_jcv_warm_start": str(rec.get("schedule", "")).startswith("jcv.train.warm_start"),
                        "state_keys_and_shapes_match_Model": set(st) == set(ref) and all(
                            st[q].shape == ref[q].shape for q in ref),
                        "seed": rec.get("seed") == k,
                        "warm_epochs_logged": len(rec.get("log", []))}
        ok = all(v for kk, v in ck["recipe"].items() if kk not in ("record_rows", "record_schedule",
                                                                    "warm_epochs_logged"))
        ok &= ck["recipe"]["warm_epochs_logged"] == 20
        if not ok:
            fails.append("warm recipe")
    elif kind in ("U", "RAW"):
        st = torch.load(d / "model.pt")
        r, z, ix = neural_release_checks(ctx, d, st, k)
        ck["release"] = r
        if kind == "U":
            ck["recipe"] = {"arm": rec.get("arm") == "U", "epoch": rec.get("epoch") == 40,
                            "map_identity": rec.get("map") == "identity (no erasure)", "seed": rec.get("seed") == k}
        else:
            run = SMF_UNITS / f"run__raw__s{k}__RAW-{spec['treat']}__b{spec['beta']:g}"
            dg = jload(run / "record.json")["diag"]
            fin = torch.load(run / "final.pt", weights_only=False)
            warm = torch.load(SMF_UNITS / f"warm__s{k}" / "warm.pt")
            ch = fin["theta_T_minus_1"]["critic_head"]
            ck["recipe"] = {
                "arm": rec.get("arm") == f"RAW-{spec['treat']}", "beta": rec.get("beta") == spec["beta"],
                "epoch": rec.get("epoch") == spec["epoch"],
                "record_recipe_rgj_unchanged": rec.get("recipe") == "rgj.train J-O/L-O unchanged (raw penalty)",
                "run_arm": dg["arm"] == {"J": "J-O", "L": "L-O"}[spec["treat"]], "run_beta": dg["beta"] == spec["beta"],
                "run_stage_B": dg["stage"] == "B", "run_lr_0.05": dg["lr"] == 0.05,
                "run_40_epochs": dg["encoder_updates"] == 40 * STEPS_PER_EPOCH and len(dg["epochs"]) == 40,
                "run_no_nonfinite_no_rescue": dg["nonfinite"] == 0 and "rescue" not in dg,
                "critic_online_updates_5_per_step_6_critics": dg["critic_online_updates"] == 40 * STEPS_PER_EPOCH * 5 * 6,
                "critic_head_equals_warm_head": all(
                    torch.equal(ch[i][0], warm[f"head.{i}.weight"].float()) and
                    torch.equal(ch[i][1], warm[f"head.{i}.bias"].float()) for i in (0, 1)),
                "pair_coefficient_matches_treatment": (dg["coefficients_base"].get("pair") == 0.0) if spec["treat"] == "L"
                else dg["coefficients_base"].get("pair") == spec["beta"] / 3}
            if spec["epoch"] == 40:
                ck["recipe"]["final_state_equals_model"] = all(torch.equal(fin["theta_T"][q], st[q]) for q in st)
            critics = torch.load(d / "critics.pt", weights_only=False)
            ck["recipe"]["critics_saved"] = sorted(critics) == ["critics", "lam", "transforms"]
        r = ck["release"]
        ok = (all(ck["recipe"].values()) and r["forward_on_osf_X_reproduces_release_features_bitwise"]
              and all(all(v for kk, v in r[f"head_{i}"].items() if kk != "HEAD_VALIDATION_log_loss_abs_diff")
                      for i in (0, 1)))
        if not ok:
            fails.append(f"{kind} recipe/release/heads")
    elif kind == "RAW_receipt":
        dg = rec["diag"]
        ck["recipe"] = {"stage": dg["stage"], "arm": dg["arm"], "beta": dg["beta"], "nonfinite": dg["nonfinite"],
                        "encoder_updates": dg["encoder_updates"], "rescued": "rescue" in dg}
        if not (dg["stage"] == "B" and dg["nonfinite"] == 0 and "rescue" not in dg):
            fails.append("RAW receipt")
    elif kind == "LEACE":
        import joblib

        from stored_model_eval.defenses import LeaceMap
        src = rec["source_unit"]
        st = torch.load(d / "model.pt")
        zU = np.load(SMF_UNITS / src / "release.npz")
        z = np.load(d / "release.npz")
        rid = z["row_id"].astype(np.int64)
        ix = ctx.rows_of(rid)
        assert np.array_equal(zU["row_id"], z["row_id"])
        inv = {int(p): j for j, p in enumerate(ix)}
        fj = np.array([inv[int(p)] for p in ctx.fit])
        Zfit = np.eye(2)[ctx.D["sex"][ctx.fit]]
        maps = {}
        mk = {}
        for i in (0, 1):
            mj = jload(d / f"leace_{i}" / "leace_map.json")
            H = np.ascontiguousarray(zU[f"r{i + 1}"][fj], dtype=np.float64)
            m = LeaceMap.load(d / f"leace_{i}")
            maps[i] = m
            mk[f"map_{i}"] = {
                "n_fit_equals_OSF_DEFENSE_FIT": mj["n_fit"] == len(ctx.fit),
                "fit_row_ids_sha256_equals_OSF_DEFENSE_FIT": mj["fit_row_ids_sha256"] == sha_bytes(
                    np.ascontiguousarray(ctx.D["row_id"][ctx.fit].astype("<i8")).tobytes()),
                "H_fit_sha256_equals_U_features_on_OSF_DEFENSE_FIT": mj["H_fit_sha256"] == sha_bytes(H.tobytes()),
                "Z_fit_sha256_equals_OSF_DEFENSE_FIT_SEX_onehot": mj["Z_fit_sha256"] == sha_bytes(
                    np.ascontiguousarray(Zfit).tobytes()),
                "official_package": mj["provenance"]["package"] == "concept-erasure" and mj["provenance"]["version"]
                == "0.2.4",
                "map_reproduces_release_features": bool(np.array_equal(m.transform(zU[f"r{i + 1}"]), z[f"r{i + 1}"]))}
        ck["recipe"] = {"source_unit_is_U_e40": src == f"tl__s{k}__e40",
                        "source_model_sha256_matches": rec["source_model_pt_sha256"] == sha_file(
                            SMF_UNITS / src / "model.pt") == sha_file(d / "model.pt"),
                        "source_complete_sha256_matches": rec["source_complete_sha256"] == sha_file(
                            SMF_UNITS / src / "COMPLETE.json"),
                        "fit_role_record": rec.get("fit_role") == "NEW_DEFENSE_FIT (alias DEFENSE_FIT)",
                        "n_fit_record": rec.get("n_fit") == len(ctx.fit), **mk}
        from jcv.train import Model
        from rgj.finalize import encode_all
        mm = Model(ctx.D["X"].shape[1], list(KS), k)
        mm.load_state_dict(st)
        mm.eval()
        Hx = encode_all(mm, ctx.D["X"][ix])
        rel = {"release_rows_by_category": ctx.composition(ix),
               "forward_on_osf_X_reproduces_U_features_bitwise": bool(np.array_equal(Hx[0], zU["r1"]) and
                                                                       np.array_equal(Hx[1], zU["r2"]))}
        for i in (0, 1):
            h = joblib.load(d / f"head_{i}.joblib")
            rel[f"head_{i}"] = head_checks(ctx, h, z[f"r{i + 1}"], ix, ctx.D["y"][TASKS[i]], KS[i],
                                           rec["finalize"]["heads"][str(i)],
                                           (z[f"c{i + 1}"], z[f"p{i + 1}"], z[f"hard{i + 1}"]))
        ck["release"] = rel
        ok = (all(v for kk, v in ck["recipe"].items() if not isinstance(v, dict)) and
              all(all(v.values()) for kk, v in ck["recipe"].items() if isinstance(v, dict)) and
              rel["forward_on_osf_X_reproduces_U_features_bitwise"] and
              all(all(v for kk, v in rel[f"head_{i}"].items() if kk != "HEAD_VALIDATION_log_loss_abs_diff")
                  for i in (0, 1)))
        if not ok:
            fails.append("LEACE recipe/maps/heads")
    elif kind == "FARE":
        import joblib
        i = spec["purpose"]
        prov = rec["provenance"]
        uid = prov["fare_uid"]
        t = ctx.tree_checks[uid]
        z = np.load(d / "release.npz")
        rid = z["row_id"].astype(np.int64)
        ix = ctx.rows_of(rid)
        h = joblib.load(d / "head.joblib")
        cells_tree = np.load(SMF_FARE / uid / "cells.npy").astype(np.int64)
        ck["recipe"] = {"tree_uid_matches_unit": uid == "smf__" + name[len("fare__"):],
                        "tree_checks_pass": t["pass"],
                        "fare_cache_complete_sha256_matches": prov["fare_cache_complete_sha256"] ==
                        t["complete_json_sha256"],
                        "unit_fit_hashes_equal_tree": prov["fit_rows_sha256"] == t["fit_rows_sha256"] and
                        prov["fit_labels_sha256"] == t["fit_labels_sha256"] and prov["n_fit"] == len(ctx.fit),
                        "zero_fairness_twin": (rec["arm"] == "F0") == spec["config"].startswith("Z"),
                        "purpose": rec["purpose"] == i, "seed": rec["seed"] == k == prov["seed"],
                        "release_cells_equal_tree_cells": bool(np.array_equal(z["cells"], cells_tree)),
                        "release_onehot_equals_cells": bool(np.array_equal(z["r"], np.eye(rec["n_cells"])[z["cells"]])),
                        "head_roles_record": rec["head_roles"] == {"fit": "NEW_DEFENSE_FIT (alias DEFENSE_FIT)",
                                                                   "C": "HEAD_VALIDATION"}}
        ck["release"] = {"release_rows_by_category": ctx.composition(ix),
                         "head": head_checks(ctx, h, z["r"], ix, ctx.D["y"][TASKS[i]], KS[i], rec["head"],
                                             (z["c"], z["p"], z["hard"]))}
        ok = all(ck["recipe"].values()) and all(v for kk, v in ck["release"]["head"].items()
                                                if kk != "HEAD_VALIDATION_log_loss_abs_diff")
        if not ok:
            fails.append("FARE recipe/head")
    if "release" in ck:
        c = ck["release"]["release_rows_by_category"]
        if c["ORIG_ASSESSMENT"] or c["RGJ_DEV"] or c["CERT"]:
            fails.append("release covers rows outside the smf roles")
    if not (ck["integrity"]["unit_complete"] and ck["integrity"]["all_files_listed"]):
        fails.append("integrity")
    if ck["evaluation_lock_pin"] is False:
        fails.append("EVALUATION_LOCK pin")
    ck["pass"] = not fails
    ck["failures"] = fails
    return ck


def check_tree(ctx, uid):
    d = SMF_FARE / uid
    c = jload(d / "COMPLETE.json")
    fr = jload(d / "rec.json")
    meta = jload(d / "model" / "model.json")
    mf = jload(d / "model" / "manifest.json")
    rh = np.load(d / "model" / "fit_row_hashes.npy", allow_pickle=False)
    D, fit = ctx.D, ctx.fit
    i = int(uid.split("__")[2][1:])
    k = int(uid.split("__")[1][1:])
    X = np.ascontiguousarray(D["X"][fit], dtype=np.float64)
    y = np.asarray(D["y"][TASKS[i]][fit]).astype(np.int64)
    S = np.asarray(D["sex"][fit]).astype(np.int64)
    smf_rows = np.flatnonzero(np.isin(ctx.cat, ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION",
                                                "SMF_DEV")))      # the 29,030 smf rows, ascending (smf loader order)
    XA = np.ascontiguousarray(D["X"][smf_rows], dtype=np.float64)
    cfg = fr["cfg"]
    cfg_canon = json.dumps({kk: cfg[kk] for kk in sorted(cfg)}, sort_keys=True, default=str)
    inputs = sha_bytes("|".join([sha_arrays(X), sha_arrays(y), sha_arrays(S), sha_arrays(XA), cfg_canon,
                                 str(int(k))]).encode())
    rows = np.empty(X.shape[0], dtype=np.uint64)
    for j in range(X.shape[0]):
        rows[j] = int.from_bytes(hashlib.blake2b(X[j].tobytes(), digest_size=8).digest(), "little")
    h = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
    h.update(np.asarray(rh, dtype=np.uint64).tobytes())
    present = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()} - {"COMPLETE.json"}
    t = {"complete_ok": all(sha_file(d / f) == hh for f, hh in c["files"].items()) and present == set(c["files"]),
         "complete_json_sha256": sha_file(d / "COMPLETE.json"), "files_sha256": c["files"],
         "n_fit_equals_OSF_DEFENSE_FIT": fr["n_fit"] == len(fit),
         "fit_rows_sha256": fr["fit_rows_sha256"], "fit_labels_sha256": fr["fit_labels_sha256"],
         "fit_rows_sha256_equals_OSF_DEFENSE_FIT": fr["fit_rows_sha256"] == sha_arrays(X),
         "fit_labels_sha256_equals_OSF_DEFENSE_FIT": fr["fit_labels_sha256"] == sha_arrays(y, S),
         "inputs_sha256_recomputed_from_osf_rows": fr["inputs_sha256"] == inputs,
         "fit_row_hash_set_equals_OSF_DEFENSE_FIT": bool(np.array_equal(np.sort(rh), np.unique(rows))),
         "model_fingerprint_consistent": h.hexdigest() == mf["fingerprint"] == fr["model_fingerprint"],
         "tree_pkl_sha256_matches_manifest": sha_file(d / "model" / "tree.pkl") == mf["tree_pkl_sha256"],
         "encode_covered_smf_rows_only": fr["n_all"] == len(smf_rows) == len(np.load(d / "cells.npy")),
         "seed": fr["seed"] == k, "fare_commit": fr["fare_commit"], "n_cells": fr["n_cells"], "cfg_id": cfg.get("id"),
         "zero_fairness": "zero_fairness_of" in cfg}
    t["pass"] = all(t[q] for q in ("complete_ok", "n_fit_equals_OSF_DEFENSE_FIT", "fit_rows_sha256_equals_OSF_DEFENSE_FIT",
                                   "fit_labels_sha256_equals_OSF_DEFENSE_FIT", "inputs_sha256_recomputed_from_osf_rows",
                                   "fit_row_hash_set_equals_OSF_DEFENSE_FIT", "model_fingerprint_consistent",
                                   "tree_pkl_sha256_matches_manifest", "encode_covered_smf_rows_only", "seed"))
    return t


def source_recipe_text():
    """The recipe facts fixed by the pinned source (text checks at a9951ed)."""
    smf_run, smf_train, rgj_train = pinned_text("smf/run.py"), pinned_text("smf/train.py"), pinned_text("rgj/train.py")
    return {
        "smf_train_SALT_A_is_0": 'SALT = {"A": 0, "B": 1}' in smf_train,
        "smf_run_U_task_line_stage_A_40_epochs": ('T.train_run(task_spec(), 0.0, load_warm(k), data, k, "A", None, '
                                                  'n_epochs=40, ckpt_epochs=(20, 40))') in smf_run,
        "smf_run_RAW_rgj_train_stage_B_40_epochs_warm_head": all(s in smf_run for s in (
            'RT.train_run({"RAW-J": "J-O", "RAW-L": "L-O"}[a], b, warm, RT.TData(D), k, "B",',
            "n_epochs=40, critic_head=T.head_of(warm), ckpt_epochs=(20, 40))")),
        "rgj_train_stage_B_salt_0": 'STAGE_ORDER_SALT = {"B": 0, "C": 1}' in rgj_train,
        "smf_run_warm_jcv_warm_start_on_TData": "m = JT.warm_start(T.D_IN, T.KS, data, k, log=log.append)" in smf_run,
        "rgj_TData_uses_DEFENSE_FIT_only": 'tr = D["idx"]["DEFENSE_FIT"]' in rgj_train,
        "smf_run_loader_is_smf_data_load": "D = DA.load()" in smf_run,
    }


def ineligible_inventory():
    """Read-only directory counts of predecessor stores (nothing opened beyond names)."""
    out = {}
    root = HOME / "PCRL_eval_cache_private"
    for study in ("jcv_v1", "pnx_v1", "rgj_v1"):
        d = root / study / "run" / "units"
        kinds = {}
        if d.exists():
            for q in d.iterdir():
                kk = q.name.split("__")[0]
                kinds[kk] = kinds.get(kk, 0) + 1
        fc = root / study / "fare_cache"
        out[study] = {"unit_directories_by_kind": dict(sorted(kinds.items())),
                      "fare_cache_trees": len([p for p in fc.iterdir()]) if fc.exists() else 0,
                      "status": "INELIGIBLE",
                      "reason": "trained/fitted on all 19,230 old defense_train rows (contains SMF_DEV and the "
                                "OSF_DEFENSE_FIT rows under another standardisation); jcv FARE trees n_fit = 19,230; "
                                "heads C-selected on old defense_val (contains RGJ_DEV); prompt section 6"}
    smf = {}
    for q in SMF_UNITS.iterdir():
        kk = q.name.split("__")[0]
        smf[kk] = smf.get(kk, 0) + 1
    out["smf_v1_not_admitted"] = {
        "unit_directories_by_kind": dict(sorted(smf.items())),
        "not_admitted": {
            "A__*, run__A__*": "Phase A normalized arms (20-epoch, REFRESHED/ONLINE/ONLINE_MATCHED schedules): not part "
                               "of the osf bank; all normalized trajectories are fitted anew under the new lock",
            "B__*, run__B__*": "Phase B arms (selected local initialization + 20-epoch continuation, AUC controller): "
                               "trajectory differs from the osf common 40-epoch path; not needed",
            "tl__s{k}__e20": "U epoch-20 snapshot: diagnostic only; nomination uses final epoch 40",
            "parity__*, calib__*, preflight__*, track__*, controls__*, whiten__*": "engineering / diagnostic records",
            "inner__*": "smf inner audit records (osf re-audits every release with its own attacker slate)",
            "outer__*": "smf assessment records and per-row SMF_DEV predictions (closed assessment; never reused)",
            "*.quarantined_*": "superseded smf receipts (kept by smf, never reused)"}}
    return out


def copy_unit(src: Path, dst: Path, files: dict):
    """Copy every listed file + COMPLETE.json; never overwrite a differing file; re-hash the copy (uncached)."""
    copied = same = 0
    for f in list(files) + ["COMPLETE.json"]:
        s, q = src / f, dst / f
        h = files.get(f) or sha_file(s)
        if q.exists():
            if sha_file(q) != h:
                raise SystemExit(f"REFUSED: differing file already admitted at {dst.name}/{f}")
            same += 1
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        tmp = q.with_name(q.name + ".copying")
        shutil.copy2(s, tmp)
        if sha_file(tmp) != h:
            raise SystemExit(f"REFUSED: copy hash mismatch {dst.name}/{f}")
        tmp.rename(q)
        copied += 1
    reread = all(sha_file(dst / f, nocache=True) == h for f, h in files.items()) and \
        sha_file(dst / "COMPLETE.json", nocache=True) == sha_file(src / "COMPLETE.json")
    return {"files": len(files) + 1, "copied_now": copied, "already_identical": same,
            "uncached_reread_matches_source": bool(reread)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    t0, c0 = time.time(), time.process_time()
    ctx = Ctx()
    trees = {uid: check_tree(ctx, uid) for uid in fare_trees()}
    ctx.tree_checks = trees
    cands = candidates()
    res = {n: check_unit(ctx, n, s) for n, s in cands.items()}
    text = source_recipe_text()
    failed = [n for n, r in res.items() if not r["pass"]] + [u for u, t in trees.items() if not t["pass"]]
    if not all(text.values()):
        failed.append("pinned recipe text")
    disjoint_ok = all(v == 0 for v in ctx.disjoint.values())
    if not disjoint_ok:
        failed.append("fitting roles overlap the consolidated assessment")
    verdict = "ADMITTED" if not failed else "REFUSED"
    copies = {}
    if verdict == "ADMITTED" and not a.check_only:
        for n in cands:
            copies[n] = copy_unit(SMF_UNITS / n, ADMITTED / n, res[n]["integrity"]["files_sha256"])
        for uid in trees:
            copies[f"fare_cache/{uid}"] = copy_unit(SMF_FARE / uid, ADMITTED / "fare_cache" / uid,
                                                    trees[uid]["files_sha256"])
        sums = sorted((str(p.relative_to(ADMITTED)), sha_file(p)) for p in ADMITTED.rglob("*")
                      if p.is_file() and p.name not in ("SHA256SUMS", "ADMISSION.json"))
        (ADMITTED / "SHA256SUMS").write_text("".join(f"{h}  {r}\n" for r, h in sums))
    lm = ctx.OD.manifest(ctx.D)
    rm = jload(PKG / "ROLE_MANIFEST.json") if (PKG / "ROLE_MANIFEST.json").exists() else {}
    adm = {
        "schema": "osf-admission-v1",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": verdict,
        "checks_failed": failed,
        "mode": "check-only" if a.check_only else "check+copy",
        "source_store": "<PRIVATE_CACHE>/smf_v1 (read-only; nothing moved, modified or deleted)",
        "destination": "<PRIVATE_CACHE>/osf_v1/admitted/<smf unit name>/ ; trees: <PRIVATE_CACHE>/osf_v1/admitted/"
                       "fare_cache/<uid>/ ; SHA256SUMS at <PRIVATE_CACHE>/osf_v1/admitted/SHA256SUMS",
        "source_pin": SOURCE_PIN,
        "code": {"osf/admit.py": sha_file(Path(__file__)), "osf/data.py": sha_file(WT / "osf" / "data.py")},
        "roles": {"OSF_DEFENSE_FIT": lm["OSF_DEFENSE_FIT"], "HEAD_VALIDATION": lm["HEAD_VALIDATION"],
                  "fit_tensor_sha_osf": ctx.OD.fit_tensor_sha(ctx.D),
                  "fit_tensor_identical_to_smf (ROLE_MANIFEST loader_comparison)":
                      (rm.get("loader_comparison") or {}).get("fit_tensor_identical_to_smf"),
                  "fitting_role_groups_shared_with_OSF_DEVELOPMENT_ASSESSMENT": ctx.disjoint,
                  "labels_read": "OSF_DEFENSE_FIT (training) and HEAD_VALIDATION (heads) via osf.data.labels_for; "
                                 "assessment labels sealed (-1) throughout"},
        "pinned_recipe_text": text,
        "recipe_notes": {
            "U": ("smf stage 'A' = salt 0 (smf.train.SALT) = the raw stage-B order of rgj (rgj.train.STAGE_ORDER_SALT): "
                  "the osf common convention. U was produced under the smf DATA_AND_ENGINEERING_LOCK hash of "
                  "smf/train.py; the later Phase A re-lock (review fixes R1-R4) did not change U (smf lead recomputed "
                  "U at e20/e40 bitwise, smf VALIDATION.md; smf verifier WARN 1). osf.run 'replay' re-derives it."),
            "RAW": ("rgj.train J-O / L-O called unchanged by smf.run stage_raw, stage 'B' (salt 0), 40 epochs, "
                    "critic_head = warm head, checkpoints 20/40; e20 units are diagnostic snapshots of the same runs."),
            "LEACE": "official concept-erasure 0.2.4 maps fitted on U (tl e40) features of OSF_DEFENSE_FIT, float64.",
            "FARE": ("official eth-sri/fare trees (oar.fare_official, pinned commit) fitted on OSF_DEFENSE_FIT inputs, "
                     "task labels and SEX; label-free encode of the 29,030 smf rows only. The new OSF_DEVELOPMENT_"
                     "ASSESSMENT pools (ORIG_ASSESSMENT, RGJ_DEV, CERT) have no cells yet: the audit/baseline owner "
                     "must apply the official label-free encode of each admitted tree to the osf rows (a cached "
                     "fit_encode_cached call refuses because its recorded X_all differs; the tree itself is reused, "
                     "not refitted), then refit/verify the deployed heads on OSF_DEFENSE_FIT / HEAD_VALIDATION."),
            "warm": ("jcv.train.warm_start (Adam 1e-3, 20 epochs) on smf NEW_DEFENSE_FIT tensors; fit rows rest on the "
                     "pinned code and record (no release exists); a warm-start bitwise replay on OSF_DEFENSE_FIT is "
                     "recommended for receipt-level confirmation.")},
        "admitted": {n: {**cands[n], "complete_json_sha256": res[n]["integrity"]["complete_json_sha256"],
                         "files_sha256": res[n]["integrity"]["files_sha256"],
                         "checks": {kk: v for kk, v in res[n].items() if kk not in ("integrity",)},
                         "integrity": {kk: v for kk, v in res[n]["integrity"].items() if kk != "files_sha256"},
                         "copy": copies.get(n)}
                     for n in cands} if verdict == "ADMITTED" else {},
        "fare_trees": {uid: {**{kk: v for kk, v in t.items() if kk != "files_sha256"},
                             "files_sha256": t["files_sha256"], "copy": copies.get(f"fare_cache/{uid}")}
                       for uid, t in trees.items()} if verdict == "ADMITTED" else {},
        "candidates_checked": {n: {"pass": r["pass"], "failures": r["failures"]} for n, r in res.items()},
        "ineligible_or_not_admitted": ineligible_inventory(),
        "counts": {"units_admitted": len(cands) if verdict == "ADMITTED" else 0,
                   "by_kind": {kk: sum(1 for s in cands.values() if s["kind"] == kk) for kk in
                               ("warm", "U", "RAW", "RAW_receipt", "LEACE", "FARE")},
                   "fare_trees_admitted": len(trees) if verdict == "ADMITTED" else 0,
                   "continuation_slots_supplied": "15 of 63 (U x3, RAW-J/RAW-L beta 0.1/0.3 x3 seeds); reused "
                                                  "evidence, not new fits"},
        "api": {"osf.admit.admitted_path(smf_unit_name)": "verified private copy directory",
                "osf.admit.admitted_fare_tree(uid)": "verified fare_cache tree directory",
                "osf.admit.admission_record()": "this file"},
        "wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1),
        "privacy": "aggregates, hashes and placeholders only; no row ids, labels, per-person values or local paths",
    }
    txt = json.dumps(adm, indent=1, default=str) + "\n"
    if re.search(r"/Users/|/Volumes/|" + re.escape(HOME.name), txt):
        raise SystemExit("REFUSED: identifying path in ADMISSION.json")
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(txt)
    tmp.replace(OUT)
    if verdict == "ADMITTED" and not a.check_only:
        shutil.copy2(OUT, ADMITTED / "ADMISSION.json")
    print(json.dumps({"verdict": verdict, "failed": failed[:20], "counts": adm["counts"],
                      "wall_s": adm["wall_s"]}, indent=1))


if __name__ == "__main__":
    main()
