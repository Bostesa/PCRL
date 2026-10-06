"""Admission of the frozen teachers and official references for the decision-preserving compression study (dpc).

Owner: data/custody role. Light compute only (hashing, copying, one forward pass per teacher). No ground-truth task
label and no SEX value is read anywhere in admission (D's label arrays are never indexed here; the FARE trees'
label-derived cell counts are never read).

Admitted (names and file hashes resolved from results/pcrl_online_strength_frontier_v1/MODEL_MANIFEST.json as
committed at the pinned evidence commit 925e0fd; the working-tree copy must equal that blob):
  teachers    U           rel__s{k}__U            (epoch 40; admitted by osf from smf tl__s{k}__e40)
              RAW-J_b0.3  rel__s{k}__RAW-J_b0.3   (epoch 40; admitted by osf from smf raw__s{k}__RAW-J__b0.3__e40)
  references  E   official LEACE           lc__s{k}__E
              F   official FARE            fare__s{k}__p0__c1, fare__s{k}__p1__c1  (+ their official trees)
              F0  no-fairness FARE twin    fare__s{k}__p0__Z1, fare__s{k}__p1__Z1  (+ their official trees)
Source store <PRIVATE_CACHE>/osf_v1 is READ-ONLY: files are copied (never moved) into
<PRIVATE_CACHE>/dpc_v1/admitted/<unit>/ (trees: admitted/fare_cache/<uid>/), every copy re-read uncached (F_NOCACHE)
and re-verified against the pinned hashes; an existing identical copy is verified, a differing one is refused.

Checks per unit (all label-free):
  integrity   COMPLETE.json files == pinned MODEL_MANIFEST complete_files_sha256; every source file re-hashed; copy
              re-hashed uncached.
  teachers    record (config, epoch 40, seed, identity map, admitted_from); state-dict keys/shapes; model_sha256
              recomputed; byte identity of model.pt/heads with the osf-admitted smf checkpoint (osf ADMISSION.json at
              the pin, whose receipts state fit on OSF_DEFENSE_FIT and C selected on HEAD_VALIDATION); an
              INDEPENDENT forward pass (torch functional ops on model.pt: per recipient Linear(83,64)-ReLU-
              Linear(64,64)-ReLU-Linear(64,16), float32, cast to float64) through the deployed sklearn heads
              (StandardScaler + LogisticRegression; predict_proba on float64 features; decision = numpy argmax, first
              index on ties; centred logits = decision_function margins, binary (0, d), minus the row mean, as in
              rgj/finalize.py `outputs`) compared bitwise with the saved release.npz on all 39,170 osf rows; the
              heads' scaler moments equal those of the OSF_DEFENSE_FIT features (label-free fit-role receipt); the
              recorded C is the recorded HEAD_VALIDATION-table argmin (smaller C on ties).
  LEACE       model.pt == U's; map fit-row ids == OSF_DEFENSE_FIT; map fit-feature hash == own U features on
              OSF_DEFENSE_FIT; own map application (official formula x - ((x - mu) Pr^T) Pl^T, float64) + heads vs
              release.
  FARE        own traversal of the official tree arrays (float32 cast, x <= threshold goes left) vs the released cells;
              tree fit-row fingerprint set == OSF_DEFENSE_FIT inputs; one-hot cells == released r; heads vs release.

API:
  teacher(name, k) -> dict(row_id, p1, p2, d1, d2, c1, c2, r1, r2)      name in {"U", "RAW-J_b0.3"}, k in {0, 1, 2}
  reference(label, k) -> dict                                            label in {"E", "F", "F0"}
  admission_record() -> dict (ADMISSION.json)
  admitted_dir(unit) -> Path of the verified private copy

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.admit [--check-only]
"""
from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from dpc import data as DD

HOME = Path.home()
CACHE = HOME / "PCRL_eval_cache_private"
OSF_PRIV = CACHE / "osf_v1"
OSF_UNITS = OSF_PRIV / "run" / "units"
OSF_FARE = OSF_PRIV / "admitted" / "fare_cache"
DPC = CACHE / "dpc_v1"
ADMITTED = DPC / "admitted"
WT = DD.WT
PKG = DD.PKG
OUT = PKG / "ADMISSION.json"
SOURCE_PIN = DD.SOURCE_PIN
OSF_REL = DD.OSF_REL
SEEDS = (0, 1, 2)
TEACHERS = {"U": "U", "RAW-J_b0.3": "RAW-J|b0.3"}
REFERENCES = {"E": "official LEACE (lc__s{k}__E)", "F": "official FARE (fare__s{k}__p{0,1}__c1)",
              "F0": "matched no-fairness FARE (fare__s{k}__p{0,1}__Z1)"}
KS = (2, 6)
TASKS = ("income", "occupation_group")
FIT = "OSF_DEFENSE_FIT"
EXPECT_KEYS = {f"enc.{i}.{j}.{w}": s for i in (0, 1)
               for j, (wshape, bshape) in zip((0, 2, 4), (((64, 83), (64,)), ((64, 64), (64,)), ((16, 64), (16,))))
               for w, s in (("weight", wshape), ("bias", bshape))}
EXPECT_KEYS.update({f"head.{i}.weight": (K, 16) for i, K in enumerate(KS)})
EXPECT_KEYS.update({f"head.{i}.bias": (K,) for i, K in enumerate(KS)})


# ------------------------------------------------------------------ helpers
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(p, nocache=False):
    fd = os.open(p, os.O_RDONLY)
    try:
        if nocache:
            fcntl.fcntl(fd, 48, 1)    # F_NOCACHE: uncached read (not a physical cold-disk read)
        h = hashlib.sha256()
        while True:
            b = os.read(fd, 1 << 22)
            if not b:
                break
            h.update(b)
        return h.hexdigest()
    finally:
        os.close(fd)


def jload(p):
    return json.loads(Path(p).read_text())


def scrub(text):
    if re.search(r"/Users/|/Volumes/|" + re.escape(HOME.name), text):
        raise SystemExit("REFUSED: identifying path or user name in a public file")
    return text


def write_public(path, obj):
    txt = scrub(json.dumps(obj, indent=1, default=str) + "\n")
    tmp = Path(path).with_suffix(".json.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(txt)
    tmp.replace(path)


def git_show(rel):
    r = subprocess.run(["git", "-C", str(WT), "show", f"{SOURCE_PIN}:{rel}"], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: {rel} absent at the pin")
    return r.stdout


@functools.lru_cache(maxsize=None)
def pinned(rel):
    return json.loads(git_show(rel))


def model_manifest():
    return pinned(f"{OSF_REL}/MODEL_MANIFEST.json")


def osf_admission():
    return pinned(f"{OSF_REL}/ADMISSION.json")


def maxdiff(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).max()) if a.size else 0.0


# ------------------------------------------------------------------ unit resolution (pinned manifest)
def resolve():
    """{(label, k): [manifest entries]} for the admitted teachers and references, from the pinned MODEL_MANIFEST."""
    want = {"U": "U", "RAW-J|b0.3": "RAW-J_b0.3", "E": "E", "F": "F", "F0": "F0"}
    out = {}
    for u in model_manifest()["units"]:
        if u["label"] in want and u["seed"] in SEEDS:
            out.setdefault((want[u["label"]], u["seed"]), []).append(u)
    for (lab, k), us in out.items():
        us.sort(key=lambda u: u["unit"])
    expect = {"U": ["rel__s{k}__U"], "RAW-J_b0.3": ["rel__s{k}__RAW-J_b0.3"], "E": ["lc__s{k}__E"],
              "F": ["fare__s{k}__p0__c1", "fare__s{k}__p1__c1"], "F0": ["fare__s{k}__p0__Z1", "fare__s{k}__p1__Z1"]}
    for lab, names in expect.items():
        for k in SEEDS:
            got = [u["unit"] for u in out.get((lab, k), [])]
            if got != [n.format(k=k) for n in names]:
                raise SystemExit(f"REFUSED: manifest units for {lab} s{k} are {got}")
    return out


def admitted_dir(unit):
    return ADMITTED / unit


# ------------------------------------------------------------------ copy with verification
def copy_verified(src: Path, dst: Path, files: dict):
    """Copy COMPLETE.json + listed files src -> dst (never moves; never overwrites a differing file); re-read the
    copies uncached and compare with the pinned hashes."""
    rels = sorted(files) + ["COMPLETE.json"]
    copied = identical = 0
    for r in rels:
        s, q = src / r, dst / r
        if q.exists():
            if r != "COMPLETE.json" and sha(q) != files[r]:
                raise SystemExit(f"REFUSED: an admitted copy differs ({dst.name}/{r}); nothing overwritten")
            if r == "COMPLETE.json" and q.read_bytes() != s.read_bytes():
                raise SystemExit(f"REFUSED: an admitted COMPLETE.json differs ({dst.name}); nothing overwritten")
            identical += 1
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        tmp = q.with_name(q.name + ".tmp")
        shutil.copy2(s, tmp)
        tmp.replace(q)
        copied += 1
    reread = all(sha(dst / r, nocache=True) == h for r, h in files.items()) and \
        (dst / "COMPLETE.json").read_bytes() == (src / "COMPLETE.json").read_bytes()
    return {"files": len(rels), "copied_now": copied, "already_identical": identical,
            "uncached_reread_matches_pinned": bool(reread)}


def integrity(entry, src_root=OSF_UNITS):
    d = src_root / entry["unit"]
    c = jload(d / "COMPLETE.json")
    files = c["files"]
    present = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()} - {"COMPLETE.json"}
    return files, {
        "complete_id_equals_unit": c.get("id") == entry["unit"],
        "complete_files_equal_pinned_manifest": files == entry["complete_files_sha256"],
        "source_files_rehashed_equal": all(sha(d / f) == h for f, h in files.items()),
        "no_unlisted_files": present == set(files),
        "complete_json_sha256": sha(d / "COMPLETE.json")}


# ------------------------------------------------------------------ independent forward pass and heads
def state_dict(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def state_sha(state):
    """Own re-implementation of the osf model hash: sorted keys, key bytes then raw tensor bytes."""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def forward(state, X):
    """Own forward pass: per recipient Linear(83,64)-ReLU-Linear(64,64)-ReLU-Linear(64,16), float32 -> float64."""
    import torch
    import torch.nn.functional as F
    Xt = torch.from_numpy(np.ascontiguousarray(np.asarray(X, dtype=np.float32)))
    out = []
    with torch.no_grad():
        for i in (0, 1):
            h = F.relu(F.linear(Xt, state[f"enc.{i}.0.weight"], state[f"enc.{i}.0.bias"]))
            h = F.relu(F.linear(h, state[f"enc.{i}.2.weight"], state[f"enc.{i}.2.bias"]))
            h = F.linear(h, state[f"enc.{i}.4.weight"], state[f"enc.{i}.4.bias"])
            out.append(h.double().numpy())
    return out


def leace_apply(npz_path, H):
    """Official LeaceEraser formula, own code: x - ((x - mean_x) @ proj_right^T) @ proj_left^T in float64."""
    import torch
    z = np.load(npz_path, allow_pickle=False)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))  # noqa: E731
    pl, pr, mu = t(z["proj_left"]), t(z["proj_right"]), t(z["mean_x"])
    x = t(H)
    with torch.no_grad():
        return (x - ((x - mu) @ pr.mH) @ pl.mH).numpy().astype(np.float64, copy=True)


def head_outputs(head, R):
    """Deployed-head outputs (rgj/finalize.py `outputs` convention, own code): centred logits, probabilities, decision."""
    R = np.asarray(R, dtype=np.float64)
    z = np.asarray(head.decision_function(R), dtype=np.float64)
    if z.ndim == 1:                       # binary sklearn head: margin d -> logits (0, d)
        z = np.stack([np.zeros_like(z), z], 1)
    P = head.predict_proba(R)
    return z - z.mean(1, keepdims=True), P, np.argmax(P, axis=1)   # numpy argmax: first index on ties


def head_checks(head, R_fit, rec_head, K):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    ok_type = isinstance(head, Pipeline) and len(head.steps) == 2 and isinstance(head.steps[0][1], StandardScaler) \
        and isinstance(head.steps[1][1], LogisticRegression)
    sc, lr = head.steps[0][1], head.steps[1][1]
    ref = StandardScaler().fit(R_fit)        # moments only (label-free)
    table = rec_head["table"]
    best = min(table, key=lambda t: (t["defense_val_log_loss"], t["C"]))
    return {"pipeline_scaler_logreg": bool(ok_type), "n_classes": int(len(lr.classes_)),
            "classes_are_0_to_K_minus_1": bool(np.array_equal(lr.classes_, np.arange(K))),
            "scaler_n_equals_OSF_DEFENSE_FIT": bool(np.all(np.asarray(sc.n_samples_seen_) == len(R_fit))),
            "scaler_mean_bitwise_OSF_DEFENSE_FIT_features": bool(np.array_equal(sc.mean_, ref.mean_)),
            "scaler_var_bitwise_OSF_DEFENSE_FIT_features": bool(np.array_equal(sc.var_, ref.var_)),
            "C_equals_recorded_selected_C": float(lr.C) == float(rec_head["selected_C"]),
            "recorded_C_is_HEAD_VALIDATION_table_argmin": float(best["C"]) == float(rec_head["selected_C"]),
            "selected_C": float(rec_head["selected_C"])}


def compare_outputs(mine, rel, keys):
    out = {}
    for k in keys:
        a, b = mine[k], rel[k]
        if k.startswith(("hard", "cells", "d")):
            out[k] = {"mismatches": int(np.sum(np.asarray(a) != np.asarray(b))), "bitwise": bool(np.array_equal(a, b))}
        else:
            out[k] = {"max_abs_diff": maxdiff(a, b), "bitwise": bool(np.asarray(a).dtype == np.asarray(b).dtype and
                                                                      np.array_equal(a, b))}
    return out


def ties(P):
    s = np.sort(P, axis=1)
    return int(np.sum(s[:, -1] == s[:, -2]))


# ------------------------------------------------------------------ per-kind admission
def admit_teacher(entry, D, fit, root=OSF_UNITS):
    d = Path(root) / entry["unit"]
    rec = jload(d / "record.json")
    k = entry["seed"]
    st = state_dict(d / "model.pt")
    keys = {kk: tuple(v.shape) for kk, v in st.items()}
    osf_adm = osf_admission()["admitted"]
    smf_unit = entry["admitted_from"].split(" ", 1)[1]
    src = osf_adm.get(smf_unit, {})
    cfiles = entry["complete_files_sha256"]
    chk = {"record": {"config_equals_manifest_label": rec["config"] == model_manifest_label(entry),
                      "epoch_40": rec["epoch"] == 40, "seed": rec["seed"] == k, "unit": rec["unit"] == entry["unit"],
                      "map_identity": rec["map"] == "identity (no erasure)",
                      "admitted_from_equals_manifest": rec["admitted_from"] == entry["admitted_from"],
                      "admitted_release_equal_on_smf_rows": all(rec["admitted_release_equal_on_smf_rows"]["equal"]
                                                                .values())},
           "state_dict": {"keys_and_shapes_as_expected": keys == EXPECT_KEYS, "keys": sorted(keys),
                          "dtype_float32": all(str(v.dtype) == "torch.float32" for v in st.values()),
                          "model_sha256_recomputed_equals_manifest": state_sha(st) == entry["model_sha256"],
                          "training_heads_in_state_dict_unused": True},
           "osf_fit_role_receipts": {
               "osf_admitted_unit": smf_unit,
               "osf_admission_pass": bool(src.get("checks", {}).get("pass")),
               "model_pt_byte_identical_to_osf_admitted": src.get("files_sha256", {}).get("model.pt") == cfiles["model.pt"],
               "heads_byte_identical_to_osf_admitted": all(src.get("files_sha256", {}).get(f) == cfiles[f]
                                                           for f in ("head_0.joblib", "head_1.joblib")),
               "osf_head_receipts": {i: {x: src["checks"]["release"][f"head_{i}"][x] for x in (
                   "scaler_n_equals_OSF_DEFENSE_FIT", "scaler_mean_var_bitwise_OSF_DEFENSE_FIT",
                   "head_C_equals_recorded_selected_C", "HEAD_VALIDATION_log_loss_equals_recorded",
                   "selected_C_is_table_argmin")} for i in (0, 1)} if src else None,
               "fit_role": "encoder trained on OSF_DEFENSE_FIT (= smf NEW_DEFENSE_FIT); deployed heads fitted on "
                           "OSF_DEFENSE_FIT, C selected on HEAD_VALIDATION; assessment groups excluded "
                           "(osf ADMISSION.json / ROLE_MANIFEST.json assessment_exclusion_from_eligible_models at the pin)"}}
    H = forward(st, D["X"])
    z = np.load(d / "release.npz", allow_pickle=False)
    order_ok = bool(np.array_equal(z["row_id"], D["row_id"]))
    mine = {"row_id": D["row_id"]}
    heads = {}
    for i in (0, 1):
        head = __import__("joblib").load(d / f"head_{i}.joblib")
        heads[i] = head_checks(head, H[i][fit], rec["heads"][str(i)], KS[i])
        c, P, hard = head_outputs(head, H[i])
        mine.update({f"r{i + 1}": H[i], f"c{i + 1}": c, f"p{i + 1}": P, f"hard{i + 1}": hard})
    keys_cmp = ["r1", "c1", "p1", "hard1", "r2", "c2", "p2", "hard2"]
    cmp = compare_outputs(mine, z, keys_cmp)
    chk["heads"] = heads
    chk["forward_pass"] = {"release_row_order_equals_D": order_ok, "rows": int(len(z["row_id"])),
                           "release_keys": sorted(z.files), "vs_release": cmp,
                           "all_bitwise": all(v["bitwise"] for v in cmp.values()),
                           "decision_equals_argmax_p_first_index": {
                               f"d{i}": bool(np.array_equal(np.argmax(z[f"p{i}"], 1), z[f"hard{i}"])) for i in (1, 2)},
                           "probability_rows_with_exact_max_ties": {f"p{i}": ties(z[f"p{i}"]) for i in (1, 2)},
                           "probability_min": {f"p{i}": float(z[f"p{i}"].min()) for i in (1, 2)},
                           "probability_exact_zeros": {f"p{i}": int((z[f"p{i}"] == 0).sum()) for i in (1, 2)},
                           "row_sum_max_abs_dev_from_1": {f"p{i}": float(np.abs(z[f"p{i}"].sum(1) - 1).max())
                                                          for i in (1, 2)},
                           "finite": all(bool(np.isfinite(z[x]).all()) for x in keys_cmp)}
    chk["pass"] = bool(all(chk["record"].values()) and chk["state_dict"]["keys_and_shapes_as_expected"]
                       and chk["state_dict"]["dtype_float32"] and chk["state_dict"]["model_sha256_recomputed_equals_manifest"]
                       and chk["osf_fit_role_receipts"]["osf_admission_pass"]
                       and chk["osf_fit_role_receipts"]["model_pt_byte_identical_to_osf_admitted"]
                       and chk["osf_fit_role_receipts"]["heads_byte_identical_to_osf_admitted"]
                       and all(all(v for kk, v in h.items() if kk not in ("n_classes", "selected_C")) for h in heads.values())
                       and order_ok and chk["forward_pass"]["all_bitwise"]
                       and all(chk["forward_pass"]["decision_equals_argmax_p_first_index"].values())
                       and chk["forward_pass"]["finite"])
    return chk, H


def model_manifest_label(entry):
    return entry["label"]


def admit_leace(entry, D, fit, U_entry, root=OSF_UNITS):
    d = Path(root) / entry["unit"]
    k = entry["seed"]
    rec = jload(d / "record.json")
    st = state_dict(d / "model.pt")
    H = forward(st, D["X"])
    chk = {"model_pt_equals_U_teacher_same_seed": entry["complete_files_sha256"]["model.pt"] ==
           U_entry["complete_files_sha256"]["model.pt"],
           "osf_admission_pass": bool(osf_admission()["admitted"].get(entry["unit"], {}).get("checks", {}).get("pass")),
           "maps_and_heads_byte_identical_to_osf_admitted": all(
               osf_admission()["admitted"].get(entry["unit"], {}).get("files_sha256", {}).get(f) == h
               for f, h in entry["complete_files_sha256"].items() if f not in ("record.json", "release.npz")),
           "record_fit_role": rec.get("fit_role"), "record_n_fit": rec.get("n_fit"),
           "record_admission_failed": rec.get("admission", {}).get("failed")}
    fit_ids = np.ascontiguousarray(D["row_id"][fit].astype("<i8"))
    maps = {}
    R = []
    for i in (0, 1):
        m = jload(d / f"leace_{i}" / "leace_map.json")
        maps[i] = {"n_fit_equals_OSF_DEFENSE_FIT": m["n_fit"] == len(fit),
                   "fit_row_ids_equal_OSF_DEFENSE_FIT_in_order": m["fit_row_ids_sha256"] == hashlib.sha256(
                       fit_ids.tobytes()).hexdigest(),
                   "fit_features_equal_own_U_features_on_OSF_DEFENSE_FIT": m["H_fit_sha256"] == hashlib.sha256(
                       np.ascontiguousarray(H[i][fit]).tobytes()).hexdigest(),
                   "npz_sha256_matches": m["npz_sha256"] == sha(d / f"leace_{i}" / "leace_map.npz"),
                   "inference_uses_labels": m["inference_uses_labels"], "dtype": m["dtype"],
                   "erasure_rank": m.get("diagnostics", {}).get("erasure_rank")}
        R.append(leace_apply(d / f"leace_{i}" / "leace_map.npz", H[i]))
    z = np.load(d / "release.npz", allow_pickle=False)
    mine, heads = {}, {}
    for i in (0, 1):
        head = __import__("joblib").load(d / f"head_{i}.joblib")
        heads[i] = head_checks(head, R[i][fit], rec["finalize"]["heads"][str(i)], KS[i])
        c, P, hard = head_outputs(head, R[i])
        mine.update({f"r{i + 1}": R[i], f"c{i + 1}": c, f"p{i + 1}": P, f"hard{i + 1}": hard})
    keys_cmp = ["r1", "c1", "p1", "hard1", "r2", "c2", "p2", "hard2"]
    cmp = compare_outputs(mine, z, keys_cmp)
    chk.update({"maps": maps, "heads": heads, "release_row_order_equals_D": bool(np.array_equal(z["row_id"], D["row_id"])),
                "vs_release": cmp, "all_bitwise": all(v["bitwise"] for v in cmp.values()),
                "max_abs_diff_any": max(v.get("max_abs_diff", 0.0) for v in cmp.values()),
                "decision_mismatches_any": int(sum(v.get("mismatches", 0) for v in cmp.values())),
                "fit_role": "LEACE maps fitted on U features of OSF_DEFENSE_FIT (fit-row ids and feature hash verified "
                            "here); heads fitted on OSF_DEFENSE_FIT LEACE features (scaler moments verified), C on "
                            "HEAD_VALIDATION (recorded table)"})
    tol_ok = chk["max_abs_diff_any"] <= 1e-12 and chk["decision_mismatches_any"] == 0
    chk["pass"] = bool(chk["model_pt_equals_U_teacher_same_seed"] and chk["osf_admission_pass"]
                       and chk["maps_and_heads_byte_identical_to_osf_admitted"] and chk["record_n_fit"] == len(fit)
                       and chk["record_admission_failed"] == []
                       and all(all(v for kk, v in m.items() if kk not in ("inference_uses_labels", "dtype", "erasure_rank"))
                               and m["inference_uses_labels"] is False for m in maps.values())
                       and all(all(v for kk, v in h.items() if kk not in ("n_classes", "selected_C")) for h in heads.values())
                       and chk["release_row_order_equals_D"] and (chk["all_bitwise"] or tol_ok))
    return chk


def tree_cells(model_json, X):
    """Own traversal of the official FARE tree arrays (sklearn float32 split semantics); returns cell index."""
    m = jload(model_json)
    Xf = np.asarray(X).astype(np.float32).astype(np.float64)
    left, right = np.asarray(m["children_left"]), np.asarray(m["children_right"])
    feat, thr = np.asarray(m["feature"]), np.asarray(m["threshold"], dtype=np.float64)
    node = np.zeros(Xf.shape[0], dtype=np.int64)
    active = left[node] != -1
    while np.any(active):
        idx = np.nonzero(active)[0]
        nd = node[idx]
        node[idx] = np.where(Xf[idx, feat[nd]] <= thr[nd], left[nd], right[nd])
        active = left[node] != -1
    leaves = np.asarray(m["leaf_node_ids"])
    pos = np.searchsorted(leaves, node)
    assert np.all(leaves[np.minimum(pos, len(leaves) - 1)] == node), "row reached a non-cell leaf"
    return pos.astype(np.int64), int(m["n_fit"]), int(m["n_features"])


def fit_row_fingerprint(X64):
    out = np.empty(X64.shape[0], dtype=np.uint64)
    for i in range(X64.shape[0]):
        out[i] = int.from_bytes(hashlib.blake2b(X64[i].tobytes(), digest_size=8).digest(), "little")
    return np.unique(out)


def admit_fare(entry, D, fit, fp_fit, root=OSF_UNITS, fare_root=OSF_FARE):
    d = Path(root) / entry["unit"]
    rec = jload(d / "record.json")
    i = rec["purpose"]
    uid = rec["provenance"]["admission"]["source"]
    tdir = Path(fare_root) / uid
    tc = jload(tdir / "COMPLETE.json")
    pinned_tree = osf_admission()["fare_trees"].get(uid, {})
    tree = {"uid": uid, "complete_files_equal_osf_admission": tc["files"] == pinned_tree.get("files_sha256"),
            "files_rehashed_equal": all(sha(tdir / f) == h for f, h in tc["files"].items()),
            "osf_tree_admission_pass": bool(pinned_tree.get("pass"))}
    cells, n_fit, nf = tree_cells(tdir / "model" / "model.json", D["X"])
    fr = np.load(tdir / "model" / "fit_row_hashes.npy", allow_pickle=False)
    tree.update({"n_fit_equals_OSF_DEFENSE_FIT": n_fit == len(fit), "n_features_83": nf == 83,
                 "fit_row_fingerprint_set_equals_OSF_DEFENSE_FIT_inputs": bool(np.array_equal(np.sort(fr), fp_fit))})
    z = np.load(d / "release.npz", allow_pickle=False)
    ncell = int(rec["n_cells"])
    onehot = np.eye(ncell)[z["cells"]]
    head = __import__("joblib").load(d / "head.joblib")
    hc = head_checks(head, z["r"][fit], rec["head"], KS[i])
    c, P, hard = head_outputs(head, z["r"])
    cmp = compare_outputs({"cells": cells, "c": c, "p": P, "hard": hard}, z, ["cells", "c", "p", "hard"])
    prov = rec["provenance"]
    chk = {"purpose": i, "task": TASKS[i], "arm": rec["arm"], "config": rec["config"]["name"], "n_cells": ncell,
           "tree": tree, "release_row_order_equals_D": bool(np.array_equal(z["row_id"], D["row_id"])),
           "r_equals_one_hot_cells": bool(np.array_equal(onehot, z["r"])), "head": hc, "vs_release": cmp,
           "all_bitwise": all(v["bitwise"] for v in cmp.values()),
           "record_fit_role": prov.get("fit_role"), "record_head_roles": rec.get("head_roles"),
           "record_admission_checks_all_true": bool(all(prov["admission"]["checks"].values())
                                                   and not prov["admission"]["failed"]),
           "alias": prov.get("alias"), "zero_fairness": rec["arm"] == "F0",
           "fit_role": "official FARE tree fitted on OSF_DEFENSE_FIT inputs, task labels and SEX (receipts: osf "
                       "ADMISSION.json fare_trees; fit-row fingerprint verified here label-free); head fitted on "
                       "OSF_DEFENSE_FIT cells, C on HEAD_VALIDATION"}
    chk["pass"] = bool(tree["complete_files_equal_osf_admission"] and tree["files_rehashed_equal"]
                       and tree["osf_tree_admission_pass"] and tree["n_fit_equals_OSF_DEFENSE_FIT"]
                       and tree["fit_row_fingerprint_set_equals_OSF_DEFENSE_FIT_inputs"]
                       and chk["release_row_order_equals_D"] and chk["r_equals_one_hot_cells"]
                       and all(v for kk, v in hc.items() if kk not in ("n_classes", "selected_C"))
                       and chk["all_bitwise"] and chk["record_admission_checks_all_true"])
    return chk, uid, tc["files"]


# ------------------------------------------------------------------ main admission
def drive_probe():
    """Names-free probe of external volumes (the boot volume and the unrelated installer image are skipped by name)."""
    try:
        from dpc import closeout as CO
        vol, ev = CO.locate_drive()
        return {"external_drive_with_verified_prior_copies_mounted": vol is not None, **ev}
    except Exception as e:  # noqa: BLE001  (closeout not yet available)
        return {"external_drive_with_verified_prior_copies_mounted": None, "note": f"probe unavailable: {type(e).__name__}"}


def run(check_only=False):
    t0, c0 = time.time(), time.process_time()
    D = DD.load()                            # sealed; labels are never indexed below
    fit = D["idx"][FIT]
    fp_fit = fit_row_fingerprint(np.ascontiguousarray(D["X"][fit].astype(np.float64)))
    units = resolve()
    wt_manifest_equal = (WT / OSF_REL / "MODEL_MANIFEST.json").read_bytes() == git_show(f"{OSF_REL}/MODEL_MANIFEST.json")
    rec = {"schema": "dpc-admission-v1", "written_at": now(), "mode": "check-only" if check_only else "check+copy",
           "source_pin": SOURCE_PIN, "source_store": "<PRIVATE_CACHE>/osf_v1 (read-only; nothing moved, modified or "
                                                     "deleted)",
           "destination": "<PRIVATE_CACHE>/dpc_v1/admitted/<unit>/ ; trees: <PRIVATE_CACHE>/dpc_v1/admitted/fare_cache/"
                          "<uid>/ ; SHA256SUMS at <PRIVATE_CACHE>/dpc_v1/admitted/SHA256SUMS",
           "manifest": {"file": f"{OSF_REL}/MODEL_MANIFEST.json",
                        "sha256_at_pin": hashlib.sha256(git_show(f"{OSF_REL}/MODEL_MANIFEST.json")).hexdigest(),
                        "working_tree_equals_pin_blob": wt_manifest_equal},
           "receipts_used": {f: hashlib.sha256(git_show(f"{OSF_REL}/{f}")).hexdigest()
                             for f in ("ADMISSION.json", "ROLE_MANIFEST.json", "MODEL_MANIFEST.json")},
           "code": {"dpc/admit.py": sha(Path(__file__)), "dpc/data.py": sha(Path(DD.__file__))},
           "roles": {r: {"rows": int(len(D["idx"][r])), "groups": int(len(np.unique(D["unit"][D["idx"][r]]))),
                         "row_id_sha256": DD.rowid_hash(D["row_id"][D["idx"][r]])} for r in DD.ROLES},
           "labels_read": "none (no task label and no SEX value is read in admission; assessment labels sealed (-1))",
           "stage_note": "admission = restore/parity forward passes and hash checks (allowed before the engineering "
                         "lock, prompt section 10); no label-based scientific check, no fitting",
           "teachers": {}, "references": {}, "fare_trees": {}}
    rec["roles"]["assessment_groups_shared_with_fit_roles"] = {
        r: int(len(set(np.unique(D["unit"][D["idx"][r]]).tolist()) & set(np.unique(D["unit"][D["idx"][DD.ASSESS]])
                                                                          .tolist()))) for r in DD.FIT_ROLES}
    sums = {}
    for name in TEACHERS:
        for k in SEEDS:
            e = units[(name, k)][0]
            files, integ = integrity(e)
            chk, _ = admit_teacher(e, D, fit)
            ent = {"unit": e["unit"], "seed": k, "epoch": 40, "manifest_label": e["label"],
                   "admitted_from_osf": e["admitted_from"], "model_sha256": e["model_sha256"],
                   "complete_files_sha256": files, "integrity": integ, "checks": chk}
            if not check_only:
                ent["copy"] = copy_verified(OSF_UNITS / e["unit"], admitted_dir(e["unit"]), files)
                sums.update({f"{e['unit']}/{f}": h for f, h in files.items()})
            ent["pass"] = bool(chk["pass"] and all(v for kk, v in integ.items() if kk != "complete_json_sha256")
                               and (check_only or ent["copy"]["uncached_reread_matches_pinned"]))
            rec["teachers"][f"{name}|{k}"] = ent
    for lab in ("E", "F", "F0"):
        for k in SEEDS:
            for e in units[(lab, k)]:
                files, integ = integrity(e)
                if lab == "E":
                    chk = admit_leace(e, D, fit, units[("U", k)][0])
                else:
                    chk, uid, tfiles = admit_fare(e, D, fit, fp_fit)
                    if not check_only:
                        tc = copy_verified(OSF_FARE / uid, ADMITTED / "fare_cache" / uid, tfiles)
                        sums.update({f"fare_cache/{uid}/{f}": h for f, h in tfiles.items()})
                        rec["fare_trees"][uid] = {"copy": tc, "files_sha256": tfiles}
                ent = {"unit": e["unit"], "seed": k, "label": lab, "kind": e["kind"],
                       "complete_files_sha256": files, "integrity": integ, "checks": chk}
                if not check_only:
                    ent["copy"] = copy_verified(OSF_UNITS / e["unit"], admitted_dir(e["unit"]), files)
                    sums.update({f"{e['unit']}/{f}": h for f, h in files.items()})
                ent["pass"] = bool(chk["pass"] and all(v for kk, v in integ.items() if kk != "complete_json_sha256")
                                   and (check_only or ent["copy"]["uncached_reread_matches_pinned"]))
                rec["references"][e["unit"]] = ent
    inp = {"file": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz", "sha256": DD.INPUT_SHA,
           "source_matches": sha(DD.SRC) == DD.INPUT_SHA}
    if not check_only:
        q = ADMITTED / "inputs" / "adult_jcv.npz"
        if q.exists():
            inp["copy"] = "already present" if sha(q) == DD.INPUT_SHA else "DIFFERS (refused)"
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(DD.SRC, q.with_name(q.name + ".tmp"))
            q.with_name(q.name + ".tmp").replace(q)
            inp["copy"] = "copied"
        inp["copy_uncached_reread_matches"] = sha(q, nocache=True) == DD.INPUT_SHA
        inp["destination"] = "<PRIVATE_CACHE>/dpc_v1/admitted/inputs/adult_jcv.npz (so a backup of dpc_v1 restores " \
                             "from the copy alone)"
        sums["inputs/adult_jcv.npz"] = DD.INPUT_SHA
    rec["input"] = inp
    failed = [k for k, v in {**rec["teachers"], **rec["references"]}.items() if not v["pass"]]
    if not (inp["source_matches"] and (check_only or inp.get("copy_uncached_reread_matches"))):
        failed.append("input")
    if not check_only:
        (ADMITTED / "SHA256SUMS").write_text("".join(f"{h}  {r}\n" for r, h in sorted(sums.items())))
        rec["SHA256SUMS_sha256"] = sha(ADMITTED / "SHA256SUMS")
        rec["files_admitted"] = len(sums)
    rec["verdict"] = "ADMITTED" if not failed else "REFUSED"
    rec["checks_failed"] = failed
    tmax = {}
    for key, v in rec["teachers"].items():
        cmpd = v["checks"]["forward_pass"]["vs_release"]
        fp = v["checks"]["forward_pass"]
        tmax[key] = {"max_abs_diff": max(x.get("max_abs_diff", 0.0) for x in cmpd.values()),
                     "decision_mismatches": int(sum(x.get("mismatches", 0) for x in cmpd.values())),
                     "bitwise": fp["all_bitwise"], "probability_exact_zeros": fp["probability_exact_zeros"],
                     "probability_min": fp["probability_min"],
                     "probability_rows_with_exact_max_ties": fp["probability_rows_with_exact_max_ties"]}
    rec["summary"] = {
        "teachers_admitted": int(sum(v["pass"] for v in rec["teachers"].values())),
        "references_admitted": int(sum(v["pass"] for v in rec["references"].values())),
        "teacher_reconstruction": tmax,
        "teacher_reconstruction_rule": "own forward pass + deployed heads vs saved release.npz, all 39,170 osf rows "
                                       "(r, centred logits, probabilities, decisions; bitwise unless a max abs diff is "
                                       "reported)",
        "underflow_note": "exact-zero probabilities are float64 underflow of the deployed heads' predict_proba (binary "
                          "income head); centred logits stay finite (osf amendment A1). Any log/KL arithmetic on p "
                          "must treat 0*log 0 = 0 and use the pinned 1e-12 clipping for true-label log loss.",
        "reference_availability": {lab: {f"s{k}": [e["unit"] for e in units[(lab, k)]] for k in SEEDS}
                                   for lab in ("E", "F", "F0")},
        "reference_arrays": {"E": "per purpose: r (16 LEACE features, descriptive full-feature view), centred logits, "
                                  "probabilities, hard decisions",
                             "F/F0": "per purpose: FARE cell code (int), r = one-hot cells, centred logits, "
                                     "probabilities, hard decisions"},
        "fare_alias_note": "FARE/F0 trees are deterministic: several seed units are recorded aliases of the seed-0 tree "
                           "(same reference, not an independent replication); see each unit's checks.alias."}
    rec["custody"] = {"drive_probe": drive_probe(),
                      "predecessor_custody_pending": pending_predecessor_commands()}
    if OUT.exists():                          # keep the latest record written by dpc.closeout predecessor
        prev = jload(OUT).get("custody", {})
        for key in ("predecessor_custody", "updated_at"):
            if key in prev:
                rec["custody"][key] = prev[key]
    rec["api"] = {"dpc.admit.teacher(name, k)": "dict(row_id, p1, p2, d1, d2, c1, c2, r1, r2) from the admitted, "
                                                "hash-verified copy; name in {'U', 'RAW-J_b0.3'}",
                  "dpc.admit.reference(label, k)": "label in {'E', 'F', 'F0'}: dict(row_id, c1, c2, p1, p2, d1, d2, r1, "
                                                   "r2[, cells1, cells2, n_cells1, n_cells2])",
                  "dpc.admit.admission_record()": "this file", "dpc.admit.admitted_dir(unit)": "verified private copy"}
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["cpu_s"] = round(time.process_time() - c0, 1)
    rec["privacy"] = "aggregates, hashes, booleans and placeholders only; no row ids, labels, per-person values or paths"
    write_public(OUT, rec)
    print(json.dumps({"verdict": rec["verdict"], "failed": failed, "teachers": rec["summary"]["teachers_admitted"],
                      "references": rec["summary"]["references_admitted"], "wall_s": rec["wall_s"]}, indent=1))
    return rec


def pending_predecessor_commands():
    from dpc import closeout as CO
    return {"status": "PENDING unless the drive probe above shows the drive mounted", **CO.PENDING,
            "osf_backup_only": "OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.closeout predecessor  (osf.closeout "
                               "backup --dest <DRIVE_ROOT> only, receipts redirected)",
            "never": "the closed osf/smf results directories and private stores are not written"}


# ------------------------------------------------------------------ API
@functools.lru_cache(maxsize=None)
def _verified(unit):
    d = admitted_dir(unit)
    entries = {e["unit"]: e for es in resolve().values() for e in es}
    if unit not in entries:
        raise KeyError(f"{unit} is not an admitted unit")
    want = entries[unit]["complete_files_sha256"]
    if not (d / "COMPLETE.json").exists() or jload(d / "COMPLETE.json")["files"] != want:
        raise RuntimeError(f"{unit}: admitted copy absent or COMPLETE differs; run python -m dpc.admit")
    bad = [f for f, h in want.items() if sha(d / f) != h]
    if bad:
        raise RuntimeError(f"{unit}: admitted copy fails hash verification: {bad}")
    return d


def teacher(name, k):
    """Teacher outputs on all 39,170 osf rows (osf D order) from the admitted, hash-verified copy.
    p_i probabilities (float64), d_i decisions (argmax, first index on ties), c_i centred logits, r_i 16-d features."""
    if name not in TEACHERS or k not in SEEDS:
        raise KeyError(f"teacher {name!r} seed {k!r}")
    d = _verified(f"rel__s{k}__{name}")
    z = np.load(d / "release.npz", allow_pickle=False)
    out = {"row_id": z["row_id"].copy()}
    for i in (1, 2):
        p = z[f"p{i}"].copy()
        dd = z[f"hard{i}"].copy()
        assert np.array_equal(np.argmax(p, 1), dd), "decision is not the first-index argmax"
        out.update({f"p{i}": p, f"d{i}": dd, f"c{i}": z[f"c{i}"].copy(), f"r{i}": z[f"r{i}"].copy()})
    return out


def reference(label, k):
    """Official reference releases: E (LEACE) or F / F0 (FARE / no-fairness FARE), all osf rows, osf D order."""
    if label not in REFERENCES or k not in SEEDS:
        raise KeyError(f"reference {label!r} seed {k!r}")
    if label == "E":
        z = np.load(_verified(f"lc__s{k}__E") / "release.npz", allow_pickle=False)
        out = {"row_id": z["row_id"].copy()}
        for i in (1, 2):
            out.update({f"c{i}": z[f"c{i}"].copy(), f"p{i}": z[f"p{i}"].copy(), f"d{i}": z[f"hard{i}"].copy(),
                        f"r{i}": z[f"r{i}"].copy()})
        return out
    tag = "c1" if label == "F" else "Z1"
    out = {}
    for i in (0, 1):
        z = np.load(_verified(f"fare__s{k}__p{i}__{tag}") / "release.npz", allow_pickle=False)
        if "row_id" in out:
            assert np.array_equal(out["row_id"], z["row_id"])
        out["row_id"] = z["row_id"].copy()
        j = i + 1
        out.update({f"c{j}": z["c"].copy(), f"p{j}": z["p"].copy(), f"d{j}": z["hard"].copy(),
                    f"r{j}": z["r"].copy(), f"cells{j}": z["cells"].copy(), f"n_cells{j}": int(z["r"].shape[1])})
    return out


def admission_record():
    return jload(OUT)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    import torch
    torch.set_num_threads(1)
    run(a.check_only)


if __name__ == "__main__":
    main()
