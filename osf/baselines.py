"""Mandatory official references for the online-strength frontier study (audit/baseline owner).

Nothing here re-implements a reference method. LEACE = EleutherAI concept-erasure through the pinned wrapper
stored_model_eval.defenses (official LeaceFitter defaults, float64); FARE = the official eth-sri/fare tree through
oar.fare_official (pinned commit, patched scikit-learn in its own environment). Roles (osf.data aliases):
DEFENSE_FIT = OSF_DEFENSE_FIT (= smf NEW_DEFENSE_FIT exactly), HEAD_VALIDATION, inner audits AUDIT_FIT ->
INNER_SELECTION. No function here reads labels of OSF_DEVELOPMENT_ASSESSMENT; every fit refuses sealed labels.

Admission of predecessor (smf) artefacts. A source artefact is read from the custody copy
osf.admit.admitted_path(<smf name>) when that module and copy exist, otherwise from the smf original READ-ONLY (the
record says which). It is admitted only when every receipt below verifies against this study's D; otherwise the
reference is refit on OSF_DEFENSE_FIT with the same pinned official wrapper (never silently).

U (task-only reference; `u_release`). smf tl__s{k}__e40 (model.pt for jcv.train.Model(83, [2, 6], k)); its release on
ALL osf D rows (including the sealed assessment rows; labels are not read) is rebuilt with rgj.finalize.finalize_model
(identity map; heads fit on OSF_DEFENSE_FIT, C by HEAD_VALIDATION log loss). Receipt: the rebuilt release equals the
smf release bit-for-bit on every row the smf release covers. If the lead's rel__s{k}__U exists it must equal it.

LEACE reference E (`leace_unit` -> unit lc__s{k}__E). One official LEACE map per recipient on U's encoder features,
concept = one-hot SEX. The smf maps lc__s{k}__E/leace_{0,1} are ADMITTED iff, for both recipients: the smf record's
source_unit is tl__s{k}__e40 and its source_model_pt_sha256 equals the admitted U model.pt; the map metadata has the
pinned official settings (LEACE_DEFAULTS, float64), n_fit = |OSF_DEFENSE_FIT|, fit_row_ids_sha256 = hash of
OSF_DEFENSE_FIT row ids (source order), H_fit_sha256 = hash of U's float64 features of those rows recomputed from osf
D's preprocessing (proves the fitting inputs and preprocessing), Z_fit_sha256 = hash of one-hot SEX of those rows;
and OSF_DEFENSE_FIT shares no exact-record group with OSF_DEVELOPMENT_ASSESSMENT (assessment exclusion). Otherwise
the map is refit (fit_leace on OSF_DEFENSE_FIT). Release r_i = map_i(U features) for every osf D row; deployed heads
refitted with rgj.finalize.heads_and_outputs; replay receipt: equality with the smf lc release on shared rows.
LEACE's native condition is checked on the fitting rows only (finite-sample linear guardedness of those rows).

FARE reference F and zero-fairness twin F0. Grid = jcv.run.FARE_GRID (6 configurations, pinned) x 2 purposes x
seed k (official random_state 43 + k). The smf tree smf__s{k}__p{i}__{c|Z}{id} is ADMITTED iff its cache entry is
hash-complete and its recorded n_fit = |OSF_DEFENSE_FIT|, fit_rows_sha256 = sha(OSF_DEFENSE_FIT X, float64),
fit_labels_sha256 = sha(task labels, SEX of those rows), configuration = the registered one (or its official
zero-fairness twin), seed = k, FARE commit / official tree / criterion hashes = the pins, and the saved model's
fingerprint = the recorded one. The label-free official encode (FO.encode, features only) is then applied to EVERY
osf D row (including the sealed assessment rows), cross-checked against the pure-numpy traversal of the stored tree
(FO.encode_portable) and, on rows the smf release covers, against the smf cells. A tree failing a receipt is refit
with FO.fit_encode_cached on OSF_DEFENSE_FIT under <PRIVATE_CACHE>/osf_v1/fare_cache/osf__s{k}__p{i}__{c|Z}{id}.
Purpose unit fare__s{k}__p{i}__{c|Z}{id}: release.npz {row_id, cells, r = one-hot cells, c, p, hard}; deployed head
jcv.finalize.fit_head (OSF_DEFENSE_FIT fit, C by HEAD_VALIDATION log loss). Seed aliases: when the seed-k tree induces
exactly the seed-0 partition of the osf D rows (bijective cell map) its unit records alias_of the seed-0 unit; aliased
seeds are the same reference, NOT independent replications.
Pair units (neural release format, finite): fare__s{k}__F__c{a}_c{b} and fare__s{k}__F0__Z{a}_Z{b} with r1 = one-hot
cells of purpose 0, c1, p1, hard1 and r2, c2, p2, hard2 of purpose 1; record "finite": true.

Configuration selection (inherited rule, jcv/select.py, applied globally across seeds as section 13 requires one
configuration per family): per purpose, among configurations passing that purpose's three gates versus U on
INNER_SELECTION on EVERY seed, the lowest seed-mean inner local SEX AUC (osf.audit inner audit of the finite single
view [one-hot cells, centred logits]; inner slate + cell-conditional attackers); ties -> lower id. No such
configuration -> NO_FEASIBLE_CONFIGURATION and the closest configuration (largest min-over-seeds worst-gate margin,
ties -> lower id) is kept descriptively (utility-infeasible reference points stay visible). The per-seed inherited
choice is archived next to it. F = the two purposes' selected configurations; F0 = the zero-fairness twin at F's
configurations (same tree budget, gamma = 0). No local guard. Certificates: none (FARE's demographic-parity
certificate is not an attribute-inference bound and no certification role exists here).

Utility gates (INNER_SELECTION, deployed heads; const = OSF_DEFENSE_FIT majority class)
  G1  Acc >= Acc(U) - 0.01        G2  Acc - const >= 0.8 (Acc(U) - const)        G3  Acc - const >= 0.03

API for the lead
  run_references(D, seeds=(0, 1, 2), shard=None, units_dir=None) -> dict   (units + inner audits; selection when
      every seed's grid is complete; shard = None | "i/n" | (i, n) over seeds; deterministic and resumable: complete
      units are skipped, interrupted .tmp writes replaced; compare_unit_dirs() checks a re-run against an earlier copy)
  Status vocabulary (osf.select reads "NOMINEE" as usable): E / F0 NOMINEE | INFEASIBLE_CONTROL; F NOMINEE |
      NO_FEASIBLE_NOMINEE (closest configuration kept visible). Seed k's own gates are in task_feasible.
  reference_candidates(k, units_dir=None) -> {"E": {...}, "F": {...}, "F0": {...}}, each with unit, units (F / F0:
      the two purpose units for a {"kind": "fare"} lock spec; E: None -> {"kind": "release", "unit": unit}), status,
      task_feasible (this seed), task_feasible_all_seeds, inner {v1, v2, pair}, utility {0, 1}, gate_margins,
      inner_unit, alias_of
  reference_table(units_dir=None) -> list of rows (every grid point, both purposes, every seed; for frontiers)
CLI: OMP_NUM_THREADS=1 python -m osf.baselines run [--seeds 0 1 2]
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from jcv.finalize import fit_head, outputs
from osf import audit as OA
from rgj import finalize as FN

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private"
UNITS = PRIV / "osf_v1" / "run" / "units"
RUN = PRIV / "osf_v1" / "run"
FARE_CACHE = PRIV / "osf_v1" / "fare_cache"
SMF_UNITS = PRIV / "smf_v1" / "run" / "units"
SMF_FARE_CACHE = PRIV / "smf_v1" / "fare_cache"
KS = [2, 6]
TASKS = ("income", "occupation_group")
SEEDS = (0, 1, 2)
FIT_ROLE, HEAD_ROLE = "DEFENSE_FIT", "HEAD_VALIDATION"
SELECTION_FILE = "references_selection.json"
CERT_STATUS = {"status": "NOT_COMPUTED", "note": "FARE's certificate is a demographic-parity certificate with its own "
               "premises, not an attribute-inference bound; no certification role exists in this study."}


def _registered_grid():
    from jcv.run import FARE_GRID as G       # the predecessor's bounded configuration bank, pinned (not copied)
    return [dict(c) for c in G]


FARE_GRID = _registered_grid()
CFG = {c["id"]: c for c in FARE_GRID}


# ------------------------------------------------------------------ helpers
def udir(name, units_dir=None):
    return Path(units_dir or UNITS) / name


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_arrays(*arrays):        # identical construction to oar.fare_official._sha256_arrays
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _rec(name, units_dir=None):
    return json.loads((udir(name, units_dir) / "record.json").read_text())


def fit_rows(D):
    tr = np.asarray(D["idx"][FIT_ROLE])
    if "OSF_DEFENSE_FIT" in D["idx"] and not np.array_equal(D["idx"]["OSF_DEFENSE_FIT"], tr):
        raise SystemExit("REFUSED: DEFENSE_FIT is not the OSF_DEFENSE_FIT alias")
    return tr


def assessment_excluded(D):
    """OSF_DEFENSE_FIT shares no row id and no exact-record group with the consolidated assessment."""
    tr = fit_rows(D)
    a = D["idx"].get("OSF_DEVELOPMENT_ASSESSMENT", D["idx"].get("DEVELOPMENT_ASSESSMENT", np.array([], int)))
    return {"rows_overlap": int(np.intersect1d(D["row_id"][tr], D["row_id"][a]).size),
            "groups_overlap": int(np.intersect1d(D["unit"][tr], D["unit"][a]).size)}


def source_path(name, fallback_root):
    """(path, origin) of a predecessor artefact: the hash-verified custody copy from osf.admit (admitted_path for
    units, admitted_fare_tree for FARE cache entries) when the custody owner admitted it, else the smf original
    (read-only; nothing is ever written there). Either way this module's own receipts decide admission. A custody copy
    that fails its own verification is reported in `origin`, never used. A non-default fallback_root (tests) never
    consults osf.admit."""
    if Path(fallback_root) not in (SMF_UNITS, SMF_FARE_CACHE):
        return Path(fallback_root) / name, "custom_root"
    try:
        from osf import admit as AD          # written by the data/custody owner
        fn = AD.admitted_fare_tree if Path(fallback_root) == SMF_FARE_CACHE else AD.admitted_path
        p = fn(name)
        if p is not None and Path(p).exists():
            return Path(p), "osf_admitted_copy"
        note = "custody copy missing"
    except (ImportError, AttributeError) as e:
        note = f"osf.admit unavailable ({type(e).__name__})"
    except KeyError:
        note = "not in the custody admission list"
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        note = f"custody copy failed its verification: {e}"
    return Path(fallback_root) / name, f"smf_original_read_only ({note})"


def software_pins():
    import sklearn
    pins = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "scikit_learn": sklearn.__version__}
    try:
        from stored_model_eval.defenses import verify_official_leace
        pins["leace"] = {k: v for k, v in verify_official_leace().items() if k != "installed_path"}
    except Exception as e:  # noqa: BLE001
        pins["leace"] = {"error": repr(e)}
    from oar import fare_official as FO
    pins["fare"] = {"commit": FO.FARE_COMMIT, "py_tree_sha256": FO.FARE_PY_TREE_SHA256,
                    "criterion_pyx_sha256": FO.CRITERION_PYX_FIXED_SHA256,
                    "sklearn_base_commit": FO.SKLEARN_BASE_COMMIT, "sklearn_expected_version": FO.SKLEARN_EXPECTED_VERSION,
                    "env_python": "<PRIVATE_CACHE>/oar_v1/env_fare/bin/python"}
    return pins


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


def _shared_positions(src_row_id, D):
    """(positions in D, positions in the source) of rows present in both."""
    pos = {int(r): j for j, r in enumerate(D["row_id"])}
    src = np.array([j for j, r in enumerate(src_row_id) if int(r) in pos], dtype=np.int64)
    dst = np.array([pos[int(src_row_id[j])] for j in src], dtype=np.int64)
    return dst, src


def _maxdiff(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# ------------------------------------------------------------------ U (task-only reference)
def u_model(k, model_factory=None, smf_units=None):
    src, origin = source_path(f"tl__s{k}__e40", smf_units or SMF_UNITS)
    if not FN.unit_complete(src):
        raise SystemExit(f"REFUSED: U source tl__s{k}__e40 is not hash-complete ({origin})")
    if model_factory is None:
        from jcv.train import Model
        model_factory = lambda: Model(83, KS, k)   # noqa: E731
    m = model_factory()
    m.load_state_dict(torch.load(src / "model.pt"))
    m.eval()
    return m, {"source": f"tl__s{k}__e40", "origin": origin, "model_pt_sha256": _sha_file(src / "model.pt"),
               "source_complete_sha256": _sha_file(src / "COMPLETE.json")}, src


def u_release(k, D, units_dir=None, smf_units=None, model_factory=None):
    """U's release on every osf D row (in memory) + receipts. Returns (out, heads, receipt, model, features)."""
    m, prov, src = u_model(k, model_factory, smf_units)
    Hs = FN.encode_all(m, D["X"])
    out, heads, hm = FN.heads_and_outputs(Hs, D)
    z = np.load(src / "release.npz")
    dst, sp = _shared_positions(z["row_id"], D)
    replay = {key: _maxdiff(out[key][dst], z[key][sp]) for key in z.files if key != "row_id"}
    rec = {**prov, "rows": int(len(D["row_id"])), "smf_rows_shared": int(len(dst)),
           "replay_max_abs_diff_vs_smf_release": replay, "replay_exact": all(v == 0.0 for v in replay.values()),
           "heads_selected_C": {i: hm[i]["selected_C"] for i in (0, 1)}}
    lead = udir(f"rel__s{k}__U", units_dir)
    if FN.unit_complete(lead):
        zl = np.load(lead / "release.npz")
        if not np.array_equal(zl["row_id"], D["row_id"]):
            raise SystemExit(f"REFUSED: rel__s{k}__U rows are not aligned with osf D")
        rec["lead_unit_max_abs_diff"] = {key: _maxdiff(out[key], zl[key]) for key in out}
    return out, heads, rec, m, Hs


def u_utility(k, D, units_dir=None, smf_units=None, model_factory=None):
    out, _, rec, _, _ = u_release(k, D, units_dir, smf_units, model_factory)
    return OA.inner_utility(out, D), rec


# ------------------------------------------------------------------ gates
def gate_one(u, uU):
    a, aU, c = u["acc"], uU["acc"], u["const_acc"]
    g = {"G1": a - (aU - 0.01), "G2": (a - c) - 0.8 * (aU - c), "G3": (a - c) - 0.03}
    return all(x >= 0 for x in g.values()), min(g.values()), g


def gates(util, uref):
    per = {i: gate_one(util[i], uref[i]) for i in (0, 1)}
    return all(p[0] for p in per.values()), min(p[1] for p in per.values()), {i: p[2] for i, p in per.items()}


# ------------------------------------------------------------------ LEACE (E)
def _leace_admission(k, D, Hs, u_prov, smf_units=None):
    from stored_model_eval.defenses import LEACE_DEFAULTS, LeaceMap
    tr = fit_rows(D)
    src, origin = source_path(f"lc__s{k}__E", smf_units or SMF_UNITS)
    checks, maps = {"source": f"lc__s{k}__E", "origin": origin}, {}
    if not FN.unit_complete(src):
        checks["failed"] = ["source unit missing or not hash-complete"]
        return None, checks, src
    r = json.loads((src / "record.json").read_text())
    fails = []
    if r.get("source_unit") != f"tl__s{k}__e40":
        fails.append(f"source_unit {r.get('source_unit')} != tl__s{k}__e40")
    if r.get("source_model_pt_sha256") != u_prov["model_pt_sha256"]:
        fails.append("source U model.pt hash differs from the admitted U")
    Z = np.eye(2)[np.asarray(D["sex"])[tr]]
    want = {"n_fit": int(len(tr)), "fit_row_ids_sha256": _sha_bytes(np.ascontiguousarray(
        np.asarray(D["row_id"][tr]).astype("<i8")).tobytes()), "Z_fit_sha256": _sha_bytes(np.ascontiguousarray(Z).tobytes())}
    per = {}
    for i in (0, 1):
        try:
            m = LeaceMap.load(src / f"leace_{i}")
        except Exception as e:  # noqa: BLE001
            fails.append(f"leace_{i} unloadable: {e!r}")
            continue
        md = m.metadata
        H = np.ascontiguousarray(np.asarray(Hs[i][tr], dtype=np.float64))
        c = {"settings_official": md.get("settings_used") == LEACE_DEFAULTS, "dtype_float64": md.get("dtype") == "float64",
             "n_fit": md.get("n_fit") == want["n_fit"], "fit_row_ids_sha256": md.get("fit_row_ids_sha256") == want[
                 "fit_row_ids_sha256"], "H_fit_sha256": md.get("H_fit_sha256") == _sha_bytes(H.tobytes()),
             "Z_fit_sha256": md.get("Z_fit_sha256") == want["Z_fit_sha256"]}
        per[i] = c
        fails += [f"leace_{i}: {kk}" for kk, ok in c.items() if not ok]
        maps[i] = m
    ex = assessment_excluded(D)
    if ex["rows_overlap"] or ex["groups_overlap"]:
        fails.append(f"OSF_DEFENSE_FIT overlaps the assessment {ex}")
    checks.update({"per_recipient": per, "assessment_exclusion": ex, "smf_record_n_fit": r.get("n_fit"),
                   "failed": fails})
    return (maps if not fails and len(maps) == 2 else None), checks, src


def leace_unit(k, D, units_dir=None, smf_units=None, model_factory=None, force_refit=False):
    """Official LEACE per recipient on U's features (admitted smf maps or a refit on OSF_DEFENSE_FIT) -> lc__s{k}__E."""
    from stored_model_eval.defenses import fit_leace, verify_official_leace
    name = f"lc__s{k}__E"
    if FN.unit_complete(udir(name, units_dir)):
        return _rec(name, units_dir)
    tr = fit_rows(D)
    OA.check_labels(D["sex"], tr)
    t0 = time.time()
    prov = verify_official_leace()
    out_u, _, u_rec, model, Hs = u_release(k, D, units_dir, smf_units, model_factory)
    maps, adm, src = (None, {"failed": ["refit forced"]}, None) if force_refit else \
        _leace_admission(k, D, Hs, u_rec, smf_units)
    decision = "ADMITTED_SMF_MAPS" if maps is not None else "REFIT_ON_OSF_DEFENSE_FIT"
    if maps is None:
        Z = np.eye(2)[D["sex"][tr]]
        maps = {i: fit_leace(Hs[i][tr], Z, fit_row_ids=D["row_id"][tr]) for i in (0, 1)}
    Z = np.eye(2)[D["sex"][tr]]
    Rs = [maps[i].transform(Hs[i]) for i in (0, 1)]
    native = {i: maps[i].native_check(Hs[i][tr], Z) for i in (0, 1)}
    out, heads, hmeta = FN.heads_and_outputs(Rs, D)
    replay = None
    if src is not None and FN.unit_complete(src):
        z = np.load(src / "release.npz")
        dst, sp = _shared_positions(z["row_id"], D)
        replay = {key: _maxdiff(out[key][dst], z[key][sp]) for key in z.files if key != "row_id"}
    usrc = source_path(f"tl__s{k}__e40", smf_units or SMF_UNITS)[0]
    files = {"release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out),
             "model.pt": lambda p: shutil.copy2(usrc / "model.pt", p)}
    for i in (0, 1):
        files[f"head_{i}.joblib"] = (lambda p, h=heads[i]: joblib.dump(h, p))
        files[f"leace_{i}/.keep"] = (lambda p, m=maps[i]: (m.save(p.parent), p.write_text("")))
    meta = {i: {k_: maps[i].metadata.get(k_) for k_ in ("estimator", "settings_used", "dtype", "n_fit", "dim", "rank",
                                                         "fit_row_ids_sha256", "H_fit_sha256", "Z_fit_sha256",
                                                         "diagnostics")} for i in (0, 1)}
    rec = {"unit": name, "arm": "E", "seed": k, "decision": decision, "admission": adm,
           "reference": "official LEACE (concept-erasure) per recipient on U's encoder features; concept one-hot SEX; "
                        "fitted on OSF_DEFENSE_FIT rows only, float64",
           "u_source": u_rec, "leace_provenance": {kk: v for kk, v in prov.items() if kk != "installed_path"},
           "leace": meta, "finalize": {"heads": hmeta, "leace": {i: {"native_check_fit_rows": native[i]} for i in (0, 1)}},
           "native_check_status": {i: native[i]["status"] for i in (0, 1)},
           "replay_vs_smf_release_max_abs_diff": replay, "fit_role": "OSF_DEFENSE_FIT (alias DEFENSE_FIT)",
           "n_fit": int(len(tr)), "rows_released": int(len(D["row_id"])),
           "release_covers": "every osf D row incl. sealed assessment rows (label-free map + heads)",
           "wall_s": time.time() - t0}
    FN.save_unit(udir(name, units_dir), files, rec)
    return _rec(name, units_dir)


# ------------------------------------------------------------------ FARE (F, F0)
def _expected_cfg(cfg_id, zero):
    from oar import fare_official as FO
    return FO.zero_fairness(CFG[cfg_id]) if zero else dict(CFG[cfg_id])


def _fare_admission(k, i, cfg_id, zero, D, smf_cache=None):
    from oar import fare_official as FO
    uid = f"smf__s{k}__p{i}__{'Z' if zero else 'c'}{cfg_id}"
    src, origin = source_path(uid, smf_cache or SMF_FARE_CACHE)
    chk = {"source": uid, "origin": origin}
    if not FO._complete_ok(src):
        chk["failed"] = ["source tree missing or not hash-complete"]
        return None, chk
    rec = json.loads((src / "rec.json").read_text())
    tr = fit_rows(D)
    y, S = np.asarray(D["y"][TASKS[i]]), np.asarray(D["sex"])
    OA.check_labels(y, tr)
    OA.check_labels(S, tr)
    cfg = _expected_cfg(cfg_id, zero)
    c = {"n_fit": rec.get("n_fit") == len(tr),
         "fit_rows_sha256": rec.get("fit_rows_sha256") == _sha_arrays(np.ascontiguousarray(D["X"][tr], dtype=np.float64)),
         "fit_labels_sha256": rec.get("fit_labels_sha256") == _sha_arrays(y[tr].astype(np.int64), S[tr].astype(np.int64)),
         "config": {kk: rec.get("cfg", {}).get(kk) for kk in ("max_leaf_nodes", "min_samples_leaf", "gamma", "criterion")}
         == {kk: cfg.get(kk) for kk in ("max_leaf_nodes", "min_samples_leaf", "gamma", "criterion")},
         "seed": rec.get("seed") == k, "fare_commit": rec.get("fare_commit") == FO.FARE_COMMIT,
         "fare_py_tree_sha256": rec.get("fare_py_tree_sha256") == FO.FARE_PY_TREE_SHA256,
         "criterion_pyx_sha256": rec.get("criterion_pyx_sha256") == FO.CRITERION_PYX_FIXED_SHA256}
    model = None
    try:
        model = FO.FareModel.load(src / "model")
        c["model_fingerprint"] = model.fingerprint == rec.get("model_fingerprint")
    except Exception as e:  # noqa: BLE001
        c["model_fingerprint"] = False
        chk["load_error"] = repr(e)
    ex = assessment_excluded(D)
    c["assessment_excluded"] = not (ex["rows_overlap"] or ex["groups_overlap"])
    chk.update({"checks": c, "recorded": {kk: rec.get(kk) for kk in ("n_fit", "n_all", "seed", "random_state", "n_cells",
                                                                       "model_fingerprint", "fit_rows_sha256",
                                                                       "fit_labels_sha256", "fare_commit")},
                "failed": [kk for kk, ok in c.items() if not ok], "src": src})
    return (model if not chk["failed"] else None), chk


def write_fare_unit(name, k, i, cells, n_cells, cfg, D, prov, units_dir=None, zero=False):
    tr, va = fit_rows(D), D["idx"][HEAD_ROLE]
    y = np.asarray(D["y"][TASKS[i]])
    OA.check_labels(y, tr, va)
    cells = np.asarray(cells).astype(np.int64)
    R = np.eye(n_cells)[cells]
    head, hm = fit_head(R, y, tr, va, KS[i])
    cen, P, hard = outputs(head, R)
    rec = {"unit": name, "arm": "F0" if zero else "F", "purpose": i, "task": TASKS[i], "seed": k, "config": cfg,
           "n_cells": int(n_cells), "provenance": prov, "head": hm,
           "head_roles": {"fit": "OSF_DEFENSE_FIT (alias DEFENSE_FIT)", "C": HEAD_ROLE}, "certificate": CERT_STATUS,
           "release_covers": "every osf D row incl. sealed assessment rows (label-free official encode + head)",
           "finite": True}
    FN.save_unit(udir(name, units_dir), {
        "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], cells=cells, r=R, c=cen, p=P, hard=hard),
        "head.joblib": lambda p: joblib.dump(head, p)}, rec)
    return _rec(name, units_dir)


def fare_unit(k, i, cfg_id, D, zero=False, units_dir=None, smf_cache=None, fare_cache=None, synthetic=False,
              auth=None, allow_fit=True, smf_units=None):
    """Admitted smf tree (receipts above) re-encoded on every osf D row, or a new official fit; -> purpose unit."""
    name = f"fare__s{k}__p{i}__{'Z' if zero else 'c'}{cfg_id}"
    if FN.unit_complete(udir(name, units_dir)):
        return _rec(name, units_dir)
    from oar import fare_official as FO
    t0 = time.time()
    Xall = np.ascontiguousarray(D["X"], dtype=np.float64)
    model, chk = _fare_admission(k, i, cfg_id, zero, D, smf_cache)
    src = chk.pop("src", None)
    cfg = _expected_cfg(cfg_id, zero)
    if model is not None:
        cells = np.asarray(FO.encode(model, Xall)).astype(np.int64)
        port = np.asarray(FO.encode_portable(model, Xall)).astype(np.int64)
        cross = {"official_vs_portable_mismatches": int((cells != port).sum())}
        smf_unit = source_path(name, smf_units or SMF_UNITS)[0]
        if FN.unit_complete(smf_unit):
            zs = np.load(smf_unit / "release.npz")
            dst, sp = _shared_positions(zs["row_id"], D)
            cross["smf_rows_shared"] = int(len(dst))
            cross["vs_smf_cells_mismatches"] = int((cells[dst] != zs["cells"][sp]).sum())
            cached = np.load(src / "cells.npy").astype(np.int64)
            cross["smf_cache_cells_equal_smf_release"] = bool(len(cached) == len(zs["cells"]) and
                                                             np.array_equal(cached, zs["cells"]))
        if cross["official_vs_portable_mismatches"] or cross.get("vs_smf_cells_mismatches", 0):
            raise SystemExit(f"REFUSED: {name}: admitted tree encode cross-check failed {cross}")
        n_cells = int(model.n_cells)
        prov = {"origin": "ADMITTED smf official FARE tree (receipts verified) re-encoded on every osf D row",
                "decision": "ADMITTED_SMF_TREE", "admission": chk, "encode_cross_check": cross,
                "fare_commit": FO.FARE_COMMIT, "model_fingerprint": model.fingerprint, "seed": k,
                "random_state": model.meta.get("random_state"), "fit_role": "OSF_DEFENSE_FIT (= smf NEW_DEFENSE_FIT)",
                "n_fit": int(len(fit_rows(D))), "encode": "official FO.encode (DecisionTreeClassifier.apply, X only)",
                "wall_s": time.time() - t0}
    else:
        if not allow_fit:
            raise SystemExit(f"REFUSED: {name} needs a new official FARE fit ({chk.get('failed')})")
        from stored_model_eval.guards import FitAuthorization
        auth = auth or (FitAuthorization.synthetic_only() if synthetic else FitAuthorization(execute_scientific_fits=True))
        tr = fit_rows(D)
        y, S = np.asarray(D["y"][TASKS[i]]), np.asarray(D["sex"])
        OA.check_labels(y, tr)
        OA.check_labels(S, tr)
        Xf = np.ascontiguousarray(D["X"][tr], dtype=np.float64)
        uid = f"osf__s{k}__p{i}__{'Z' if zero else 'c'}{cfg_id}"
        ledger = []
        with _fare_units_root(fare_cache or FARE_CACHE) as FO2:
            model, cells, frec = FO2.fit_encode_cached(uid, Xf, y[tr], S[tr], Xall, cfg, seed=k, auth=auth,
                                                       synthetic=synthetic, ledger=lambda *a: ledger.append(list(a)))
        problems = []
        if frec.get("n_fit") != len(tr) or frec.get("fit_rows_sha256") != _sha_arrays(Xf):
            problems.append("tree fit rows are not OSF_DEFENSE_FIT")
        if frec.get("fit_labels_sha256") != _sha_arrays(y[tr].astype(np.int64), S[tr].astype(np.int64)):
            problems.append("tree fit labels are not OSF_DEFENSE_FIT's")
        if frec.get("n_all") != len(D["row_id"]):
            problems.append("encode did not cover every osf D row")
        if problems:
            raise SystemExit(f"REFUSED: {name}: " + "; ".join(problems))
        cells = np.asarray(cells).astype(np.int64)
        n_cells = int(frec["n_cells"])
        prov = {"origin": "NEW official FARE fit on OSF_DEFENSE_FIT rows", "decision": "REFIT_ON_OSF_DEFENSE_FIT",
                "admission_failed": chk, "fare_uid": uid,
                "fare_cache": "<PRIVATE_CACHE>/osf_v1/fare_cache" if fare_cache is None else "custom (test)",
                "fare_commit": FO.FARE_COMMIT, "seed": k, "n_fit": int(len(tr)), "fit_rows_sha256": frec["fit_rows_sha256"],
                "fit_labels_sha256": frec["fit_labels_sha256"], "model_fingerprint": frec.get("model_fingerprint"),
                "cache_hit": bool(frec.get("cache_hit", False)), "ledger": ledger, "synthetic": bool(synthetic),
                "wall_s": time.time() - t0}
    if cells.min() < 0 or cells.max() >= n_cells:
        raise SystemExit(f"REFUSED: {name}: cell id outside the tree's cells")
    prov["alias"] = fare_alias(name, cells, D, units_dir)
    return write_fare_unit(name, k, i, cells, n_cells, cfg, D, prov, units_dir, zero)


def partition_equal(a, b):
    """True iff two cell labelings of the same rows induce the same partition (a bijective relabeling)."""
    a, b = np.asarray(a), np.asarray(b)
    pairs = np.unique(np.stack([a, b], 1), axis=0)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))


def fare_alias(name, cells, D, units_dir=None):
    """Seed-alias receipt: compare with the seed-0 unit of the same purpose/configuration (if it exists)."""
    parts = name.split("__")
    if parts[1] == "s0":
        return {"alias_of": None, "note": "seed-0 tree (alias root)"}
    ref = "__".join([parts[0], "s0", *parts[2:]])
    if not FN.unit_complete(udir(ref, units_dir)):
        return {"alias_of": None, "note": f"{ref} not available when written"}
    c0 = np.load(udir(ref, units_dir) / "release.npz")["cells"]
    same = bool(len(c0) == len(cells) and partition_equal(c0, cells))
    return {"alias_of": ref if same else None, "same_partition_as_seed0": same,
            "note": "seed-invariant tree: the same reference, not an independent replication" if same else
                    "distinct partition from seed 0"}


def _alias_of(name, D, units_dir=None):
    return fare_alias(name, np.load(udir(name, units_dir) / "release.npz")["cells"], D, units_dir)["alias_of"]


def fare_inner(name, D, units_dir=None):
    """Cached inner record of one FARE purpose unit: finite single-view local SEX AUC + deployed-head utility."""
    iname = f"inner__{name}"
    src = _sha_file(udir(name, units_dir) / "COMPLETE.json")
    if FN.unit_complete(udir(iname, units_dir)):
        rec = _rec(iname, units_dir)
        if rec.get("of_complete_sha256") != src:
            raise RuntimeError(f"{iname} was computed for another version of {name}; move it aside")
        return rec
    rec0 = _rec(name, units_dir)
    i = rec0["purpose"]
    z = np.load(udir(name, units_dir) / "release.npz")
    if not np.array_equal(z["row_id"], D["row_id"]):
        raise SystemExit(f"REFUSED: {name} rows are not aligned with osf D")
    r = OA.inner_audit({"v": np.hstack([z["r"], z["c"]])}, D, finite=True)
    u = OA.inner_utility({f"hard{i + 1}": z["hard"]}, D, purposes=(i,))[i]
    rec = {"unit": iname, "of": name, "of_complete_sha256": src, "purpose": i, "recovery_local": r["auc"]["v"],
           "recovery": r, "utility": u}
    FN.save_unit(udir(iname, units_dir), {}, rec)
    return rec


def fare_pair_unit(k, arm, cfgs, D, units_dir=None):
    """Neural-format pair release of two purpose units (finite) -> fare__s{k}__F__c{a}_c{b} / F0__Z{a}_Z{b}."""
    tag = "Z" if arm == "F0" else "c"
    n1, n2 = f"fare__s{k}__p0__{tag}{cfgs[0]}", f"fare__s{k}__p1__{tag}{cfgs[1]}"
    name = f"fare__s{k}__{arm}__{tag}{cfgs[0]}_{tag}{cfgs[1]}"
    if FN.unit_complete(udir(name, units_dir)):
        return name
    a, b = np.load(udir(n1, units_dir) / "release.npz"), np.load(udir(n2, units_dir) / "release.npz")
    if not (np.array_equal(a["row_id"], D["row_id"]) and np.array_equal(b["row_id"], D["row_id"])):
        raise SystemExit("REFUSED: FARE purpose units are not aligned with osf D")
    rel = {"row_id": a["row_id"], "r1": a["r"], "c1": a["c"], "p1": a["p"], "hard1": a["hard"],
           "r2": b["r"], "c2": b["c"], "p2": b["p"], "hard2": b["hard"]}
    rec = {"unit": name, "arm": arm, "seed": k, "configs": list(cfgs), "purpose_units": [n1, n2],
           "purpose_units_complete_sha256": [_sha_file(udir(n, units_dir) / "COMPLETE.json") for n in (n1, n2)],
           "finite": True, "certificate": CERT_STATUS,
           "alias_of": [_alias_of(n1, D, units_dir), _alias_of(n2, D, units_dir)],
           "format": "rgj release format; r_i = one-hot FARE cells of purpose i, c_i/p_i/hard_i = its deployed head"}
    FN.save_unit(udir(name, units_dir), {"release.npz": lambda p: np.savez_compressed(p, **rel)}, rec)
    return name


# ------------------------------------------------------------------ selection (global, inner roles)
def pick_global(rows_by_cfg):
    """rows_by_cfg: {cfg_id: {"gate_ok_all", "mean_R_local", "min_margin"}} -> selection dict."""
    adm = [(r["mean_R_local"], c) for c, r in rows_by_cfg.items() if r["gate_ok_all"]]
    if adm:
        c = min(adm, key=lambda x: (round(x[0], 12), x[1]))[1]
        return {"status": "SELECTED", "config": c}
    c = max(rows_by_cfg, key=lambda c: (rows_by_cfg[c]["min_margin"], -c))
    return {"status": "NO_FEASIBLE_CONFIGURATION", "config": c, "descriptive": "closest configuration (largest "
            "min-over-seeds worst-gate margin), kept visible; not a feasible control"}


def pick_per_seed(rows):
    """Inherited per-seed rule (jcv/select.py), archived for comparison."""
    adm = [x for x in rows if x["gate_ok"]]
    if adm:
        return {"status": "NOMINEE", "config": min(adm, key=lambda x: (round(x["R_local"], 12), x["config"]))["config"]}
    return {"status": "NO_FEASIBLE_NOMINEE", "config": max(rows, key=lambda x: (x["margin"], -x["config"]))["config"]}


def select_references(D, seeds=SEEDS, units_dir=None, urefs=None, grid=None, smf_cache=None, fare_cache=None,
                      synthetic=False, auth=None, smf_units=None, model_factory=None):
    grid = grid or FARE_GRID
    t0 = time.time()
    if urefs is None:
        urefs, u_receipts = {}, {}
        for k in seeds:
            urefs[k], u_receipts[k] = u_utility(k, D, units_dir, smf_units, model_factory)
    else:
        u_receipts = {k: "supplied" for k in seeds}
    urefs = {int(k): {int(i): v for i, v in u.items()} for k, u in urefs.items()}
    per, table = {}, []
    for i in (0, 1):
        by_cfg, per_seed = {}, {}
        for cfg in grid:
            cid = cfg["id"]
            rs = []
            for k in seeds:
                nm = f"fare__s{k}__p{i}__c{cid}"
                r = fare_inner(nm, D, units_dir)
                ok, margin, g = gate_one(r["utility"], urefs[k][i])
                row = {"purpose": i, "task": TASKS[i], "seed": k, "config": cid, "unit": nm, "gate_ok": ok,
                       "margin": margin, "gates": g, "R_local": r["recovery_local"], "acc": r["utility"]["acc"],
                       "const_acc": r["utility"]["const_acc"], "u_acc": urefs[k][i]["acc"],
                       "selected_attacker": r["recovery"]["selected"]["v"], "alias_of": _alias_of(nm, D, units_dir)}
                rs.append(row)
                per_seed.setdefault(k, []).append(row)
            table += rs
            by_cfg[cid] = {"gate_ok_all": all(x["gate_ok"] for x in rs), "mean_R_local": float(np.mean([x["R_local"] for x in rs])),
                           "min_margin": min(x["margin"] for x in rs)}
        per[i] = {**pick_global(by_cfg), "by_config": by_cfg,
                  "per_seed_inherited_rule": {k: pick_per_seed(v) for k, v in per_seed.items()}}
    cfgs = [per[0]["config"], per[1]["config"]]
    out = {"rule": "per purpose: configs passing that purpose's gates vs U on INNER_SELECTION on every seed; lowest "
                   "seed-mean inner local SEX AUC; ties -> lower id; none -> closest by min-over-seeds worst-gate margin "
                   "(descriptive). F0 = zero-fairness twin at F's configs. No local guard.",
           "seeds": list(seeds), "configs": cfgs, "per_purpose": per, "u_utility": urefs, "u_receipts": u_receipts,
           "table": table, "certificate": CERT_STATUS, "arms": {}}
    purposes_ok = all(per[i]["status"] == "SELECTED" for i in (0, 1))
    for arm in ("F", "F0"):
        if arm == "F0":
            for k in seeds:
                for i in (0, 1):
                    fare_unit(k, i, cfgs[i], D, True, units_dir, smf_cache, fare_cache, synthetic, auth,
                              smf_units=smf_units)
        seeds_rec = {}
        for k in seeds:
            pn = fare_pair_unit(k, arm, cfgs, D, units_dir)
            ir = OA.inner_unit(pn, D, units_dir, finite=True)
            util = {int(i): v for i, v in ir["utility"].items()}
            ok, worst, gm = gates(util, urefs[k])
            a = {w: ir["recovery"]["auc"][w] for w in ("v1", "v2", "pair")}
            seeds_rec[k] = {"unit": pn, "units": _rec(pn, units_dir)["purpose_units"], "inner": a,
                            "inner_unit": ir["unit"], "utility": util, "gate_margins": gm, "task_feasible": ok,
                            "worst_gate_margin": worst, "selected_attackers": ir["recovery"]["selected"],
                            "alias_of": _rec(pn, units_dir)["alias_of"]}
        all_ok = all(s["task_feasible"] for s in seeds_rec.values())
        if arm == "F":
            status = "NOMINEE" if (purposes_ok and all_ok) else "NO_FEASIBLE_NOMINEE"
        else:
            status = "NOMINEE" if all_ok else "INFEASIBLE_CONTROL"
        out["arms"][arm] = {"status": status, "configs": cfgs, "task_feasible_all_seeds": all_ok,
                            "purposes_selected": purposes_ok, "seeds": seeds_rec}
    out["wall_s"] = time.time() - t0
    return out


# ------------------------------------------------------------------ entry points for the lead
def run_references(D, seeds=SEEDS, shard=None, units_dir=None, smf_units=None, smf_cache=None, fare_cache=None,
                   synthetic=False, auth=None, model_factory=None, grid=None, run_dir=None, select=True):
    """Per-seed reference units (E + the FARE grid) and their inner audits for seeds in this shard; when every seed's
    grid is complete, the global FARE/F0 selection, pair units, pair inner audits and E status are written to
    <run_dir>/references_selection.json. shard = None | (index, count) over `seeds`."""
    grid = grid or FARE_GRID
    if isinstance(shard, str):                       # osf.run passes the "i/n" spec string
        shard = tuple(int(x) for x in shard.split("/"))
    mine = list(seeds) if not shard else [k for j, k in enumerate(seeds) if j % shard[1] == shard[0]]
    t0, cpu0 = time.time(), time.process_time()
    done = {}
    for k in mine:
        e = leace_unit(k, D, units_dir, smf_units, model_factory)
        OA.inner_unit(e["unit"], D, units_dir)          # reuses an inner__lc unit written by osf.inner (same slate)
        for i in (0, 1):
            for cfg in grid:
                r = fare_unit(k, i, cfg["id"], D, False, units_dir, smf_cache, fare_cache, synthetic, auth,
                              smf_units=smf_units)
                fare_inner(r["unit"], D, units_dir)
        done[k] = e["decision"]
    res = {"seeds_done": done, "wall_s": time.time() - t0, "cpu_s": time.process_time() - cpu0}
    complete = all(FN.unit_complete(udir(f"inner__fare__s{k}__p{i}__c{c['id']}", units_dir))
                   for k in seeds for i in (0, 1) for c in grid) and \
        all(FN.unit_complete(udir(f"inner__lc__s{k}__E", units_dir)) for k in seeds)
    if select and complete:
        sel = select_references(D, seeds, units_dir, None, grid, smf_cache, fare_cache, synthetic, auth, smf_units,
                                model_factory)
        urefs = sel["u_utility"]
        E = {}
        for k in seeds:
            s = OA.inner_summary(f"lc__s{k}__E", units_dir)
            ok, worst, gm = gates(s["utility"], urefs[k])
            E[k] = {"unit": f"lc__s{k}__E", "inner": s["auc"], "inner_unit": f"inner__lc__s{k}__E",
                    "utility": s["utility"], "gate_margins": gm, "task_feasible": ok, "worst_gate_margin": worst,
                    "selected_attackers": s["selected"], "decision": _rec(f"lc__s{k}__E", units_dir)["decision"]}
        allE = all(v["task_feasible"] for v in E.values())
        sel["arms"]["E"] = {"status": "NOMINEE" if allE else "INFEASIBLE_CONTROL",
                            "task_feasible_all_seeds": allE, "seeds": E}
        sel["software"] = software_pins()
        rd = Path(run_dir or RUN)
        rd.mkdir(parents=True, exist_ok=True)
        tmp = rd / (SELECTION_FILE + ".tmp")
        tmp.write_text(json.dumps(sel, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        tmp.rename(rd / SELECTION_FILE)
        res["selection"] = {"configs": sel["configs"], "status": {a: v["status"] for a, v in sel["arms"].items()}}
    res["complete"] = complete
    res["wall_s_total"] = time.time() - t0
    res["cpu_s_total"] = time.process_time() - cpu0
    return res


VOLATILE_KEYS = ("wall_s", "fit_s", "cpu_s", "cpu_s_detail", "ledger", "wall_s_total", "cpu_s_total", "origin",
                 "of_complete_sha256", "purpose_units_complete_sha256")


def strip_volatile(o, keys=VOLATILE_KEYS):
    """Record without run-dependent fields (timings, source-origin labels, hashes of record-bearing COMPLETE.json)."""
    if isinstance(o, dict):
        return {k: strip_volatile(v, keys) for k, v in o.items() if k not in keys}
    if isinstance(o, list):
        return [strip_volatile(v, keys) for v in o]
    return o


def compare_unit_dirs(a, b):
    """Compare two copies of a unit (e.g. quarantined early run vs post-lock run): every file other than record.json /
    COMPLETE.json must be byte-identical and record.json equal after strip_volatile. Returns a receipt."""
    a, b = Path(a), Path(b)
    fa = json.loads((a / "COMPLETE.json").read_text())["files"]
    fb = json.loads((b / "COMPLETE.json").read_text())["files"]
    diff = sorted(f for f in set(fa) | set(fb) if f != "record.json" and fa.get(f) != fb.get(f))
    ra, rb = (json.loads((x / "record.json").read_text()) for x in (a, b))
    same_rec = strip_volatile(ra) == strip_volatile(rb)
    return {"unit": a.name, "files_compared": len(set(fa) | set(fb)), "differing_files": diff,
            "record_equal_modulo_volatile": same_rec, "identical": not diff and same_rec}


def _load_selection(run_dir=None):
    p = Path(run_dir or RUN) / SELECTION_FILE
    if not p.exists():
        raise SystemExit("REFUSED: reference selection not written yet (run_references on every seed first)")
    return json.loads(p.read_text())


def reference_candidates(k, units_dir=None, run_dir=None):
    """{label: {...}} for E, F, F0 at seed k (inner roles only). See module docstring."""
    sel = _load_selection(run_dir)
    out = {}
    for arm in ("E", "F", "F0"):
        a = sel["arms"][arm]
        s = a["seeds"][str(k)]
        out[arm] = {"unit": s["unit"], "units": s.get("units"), "status": a["status"],
                    "task_feasible": s["task_feasible"], "task_feasible_all_seeds": a["task_feasible_all_seeds"],
                    "inner": s["inner"], "utility": {int(i): v for i, v in s["utility"].items()},
                    "gate_margins": s["gate_margins"], "worst_gate_margin": s["worst_gate_margin"],
                    "inner_unit": s["inner_unit"], "alias_of": s.get("alias_of"), "finite": arm != "E",
                    "configs": a.get("configs"), "decision": s.get("decision")}
    return out


def reference_table(run_dir=None):
    """Every FARE grid point (both purposes, every seed) with inner local AUC, utility, gates and alias status."""
    return _load_selection(run_dir)["table"]


def main(argv=None):
    import argparse

    from osf import data as OD
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--units-dir", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    D = OD.load()
    sh = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
    print(json.dumps(run_references(D, tuple(a.seeds), sh, a.units_dir), indent=1, default=str))


if __name__ == "__main__":
    main()
