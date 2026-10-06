"""Independent recomputation of the decision-preserving compression study's roles (data/custody owner).

Adapted from results/pcrl_online_strength_frontier_v1/provenance/role_check.py (pinned at 925e0fd). The roles of this
study ARE the osf roles (dpc.data imports osf.data unchanged); this script re-derives them from the rule text alone.

Independent step (import guard: no osf, smf, rgj, jcv or dpc module may be loaded; enforced below):
  - reads ONLY the non-label arrays of the admitted input (role, unit, row_id, feature_names, X); never sex, race,
    y_income or y_occupation_group;
  - rgj: old defense_val groups with u = int(sha256('20261004|dev|<unit>')[:16], 16) / 2**64 < 0.30 -> HEAD_VALIDATION,
    else rgj DEVELOPMENT_ASSESSMENT; DEFENSE_FIT = old defense_train, AUDIT_FIT = old attacker_fit, INNER_SELECTION =
    old attacker_val;
  - smf: DEFENSE_FIT groups with u('20261005|assess') < 0.20 -> NEW_DEVELOPMENT_ASSESSMENT (SMF_DEV), else
    NEW_DEFENSE_FIT = OSF_DEFENSE_FIT;
  - OSF_DEVELOPMENT_ASSESSMENT = ORIG_ASSESSMENT (old assessment) + RGJ_DEV + SMF_DEV + CERT (old cert; eligible by the
    source custody verdict) minus every group that also has a row in a fitting role or in excluded_exposure /
    excluded_dup; groups are never split;
  - every hash split is evaluated with the float rule as written AND with exact rationals (disagreements counted);
  - numeric refit: raw integers recovered from the admitted normalisation (predecessor DATA_ADMISSION.json
    numeric_norm), re-standardised with population sd on OSF_DEFENSE_FIT rows only, applied to every kept row.
Comparison steps (after the independent step; recorded separately):
  - against the pinned osf ROLE_MANIFEST.json (git show at the pin): counts, groups, row-id and group-set hashes;
  - against dpc.data.load() (sealed): kept rows and order, role/pool sets, X bitwise for every kept row, sealing.
Writes ROLE_MANIFEST.json["independent_recomputation"] (other sections are kept). Counts, hashes, booleans only.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python results/pcrl_decision_preserving_compression_v1/provenance/role_check.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
WT = PKG.parents[1]
OUT = PKG / "ROLE_MANIFEST.json"
SRC = Path.home() / "PCRL_eval_cache_private" / "jcv_v1" / "inputs" / "adult_jcv.npz"
SRC_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
PIN = "925e0fddfcb666116c6179575339728a324ed78e"
OSF_MANIFEST = "results/pcrl_online_strength_frontier_v1/ROLE_MANIFEST.json"
PRED_ADMISSION = "results/pcrl_joint_complete_view_method_v1/DATA_ADMISSION.json"
FORBIDDEN = ("osf", "smf", "rgj", "jcv", "dpc")
NON_LABEL = ("role", "unit", "row_id", "feature_names", "X")
LABEL_KEYS = ("sex", "race", "y_income", "y_occupation_group")
RGJ_SEED, SEED = 20261004, 20261005
HEAD_SHARE, ASSESS_SHARE = Fraction(3, 10), Fraction(1, 5)
RGJ_KEPT = {"defense_train": "DEFENSE_FIT", "attacker_fit": "AUDIT_FIT", "attacker_val": "INNER_SELECTION"}
EXCLUSIONS = ("excluded_exposure", "excluded_dup")
FIT_ROLES = ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
ASSESS = "OSF_DEVELOPMENT_ASSESSMENT"
ROLES = ("OSF_DEFENSE_FIT", ASSESS, "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
POOLS = ("ORIG_ASSESSMENT", "RGJ_DEV", "SMF_DEV", "CERT")
NUMERIC = ("age", "education-num", "capital-gain", "capital-loss", "hours-per-week")
EXPECTED = {"OSF_DEFENSE_FIT": (15434, 15428), "HEAD_VALIDATION": (1500, 1499), "AUDIT_FIT": (6065, 6061),
            "INNER_SELECTION": (2235, 2234), ASSESS: (13936, 13929)}


def guard():
    bad = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN)
    if bad:
        raise SystemExit(f"REFUSED: independent step sees forbidden modules {bad}")
    return bad


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def rowid_hash(r):
    return sha_bytes(np.ascontiguousarray(np.sort(np.asarray(r)).astype(np.int64)).tobytes())


def unit_set_hash(u):
    return sha_bytes(np.ascontiguousarray(np.unique(np.asarray(u)).astype(np.int64)).tobytes())


def record(ix, row_id, unit):
    return {"rows": int(len(ix)), "groups": int(len(np.unique(unit[ix]))), "row_id_sha256": rowid_hash(row_id[ix]),
            "group_id_set_sha256": unit_set_hash(unit[ix])}


def pinned(rel):
    r = subprocess.run(["git", "-C", str(WT), "show", f"{PIN}:{rel}"], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: {rel} absent at the pin")
    return r.stdout


def split(units, seed, salt, cut_exact, cut_float):
    """Group -> True iff u < cut; float rule as written and exact rational rule; returns (map, disagreements)."""
    out, dis = {}, 0
    for g in np.unique(units).tolist():
        n = int(hashlib.sha256(f"{seed}|{salt}|{int(g)}".encode()).hexdigest()[:16], 16)
        a, b = (n / 2.0 ** 64) < cut_float, Fraction(n, 2 ** 64) < cut_exact
        dis += int(a != b)
        out[g] = a
    return out, dis


def independent():
    guard()
    assert sha_file(SRC) == SRC_SHA, "REFUSED: input hash mismatch"
    z = np.load(SRC, allow_pickle=False)
    arrays_present = sorted(z.files)
    old_role, unit, row_id = z["role"].astype(str), z["unit"].astype(np.int64), z["row_id"].astype(np.int64)
    fn = [str(f) for f in z["feature_names"]]
    X = z["X"]
    del z
    n = len(old_role)
    rgj = np.array([RGJ_KEPT.get(r, "") for r in old_role.tolist()], dtype=object)
    dv = old_role == "defense_val"
    hv, d1 = split(unit[dv], RGJ_SEED, "dev", HEAD_SHARE, 0.30)
    rgj[dv] = ["HEAD_VALIDATION" if hv[g] else "DEVELOPMENT_ASSESSMENT" for g in unit[dv].tolist()]
    df = rgj == "DEFENSE_FIT"
    sa, d2 = split(unit[df], SEED, "assess", ASSESS_SHARE, 0.20)
    fit = np.array([""] * n, dtype=object)
    smf_dev = np.zeros(n, bool)
    smf_dev[df] = [sa[g] for g in unit[df].tolist()]
    fit[df & ~smf_dev] = "OSF_DEFENSE_FIT"
    for r in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
        fit[rgj == r] = r
    pool = np.array([""] * n, dtype=object)
    pool[old_role == "assessment"] = "ORIG_ASSESSMENT"
    pool[rgj == "DEVELOPMENT_ASSESSMENT"] = "RGJ_DEV"
    pool[smf_dev] = "SMF_DEV"
    pool[old_role == "cert"] = "CERT"
    nominal_pool = pool.astype(str).copy()
    blocked = np.unique(unit[(fit != "") | np.isin(old_role, EXCLUSIONS)])
    cand = pool != ""
    assert not np.any((fit != "") & cand), "a pool row holds a fitting role"
    overlap = cand & np.isin(unit, blocked)
    keep_pool = cand & ~overlap
    role = fit.copy()
    role[keep_pool] = ASSESS
    role, pool = role.astype(str), np.where(keep_pool, pool, "").astype(str)
    keep = role != ""
    # group integrity: a group never spans a kept role and any other role or a dropped row
    seen = {}
    for r, u in zip(np.where(keep, role, "DROPPED"), unit.tolist()):
        seen.setdefault(u, set()).add(r)
    span = sum(1 for s in seen.values() if len(s) > 1 and any(x != "DROPPED" for x in s))
    roles = {r: record(np.flatnonzero(role == r), row_id, unit) for r in ROLES}
    a = np.flatnonzero(role == ASSESS)
    by_pool = {p: record(a[pool[a] == p], row_id, unit) for p in POOLS}
    # numeric refit from the admitted normalisation (no label read)
    norm = json.loads(pinned(PRED_ADMISSION))["numeric_norm"]
    Xn = X[keep].astype(np.float64).copy()
    fitm = role[keep] == "OSF_DEFENSE_FIT"
    numeric = {}
    for c in NUMERIC:
        j = fn.index(c)
        mu, sd = norm[c]
        raw = Xn[:, j] * sd + mu
        r = np.round(raw)
        err = float(np.abs(raw - r).max())
        m, s = float(r[fitm].mean()), float(r[fitm].std())
        Xn[:, j] = (r - m) / s
        numeric[c] = {"inversion_max_abs_err": err, "integer_after_inversion": err < 0.05, "mean_osf_fit": m,
                      "sd_osf_fit": s}
    Xk = Xn.astype(np.float32)
    oh = [j for j, f in enumerate(fn) if f not in NUMERIC]
    onehot_ok = bool(np.all(np.isin(Xk[:, oh], (0.0, 1.0))))
    low = [f.lower().split("=")[0] for f in fn]
    return {"arrays_present": arrays_present, "arrays_read": list(NON_LABEL), "label_arrays_read": [],
            "rows_total": int(n), "rows_kept": int(keep.sum()), "rows_dropped": int((~keep).sum()),
            "dropped_by_old_role": {r: int(((~keep) & (old_role == r)).sum()) for r in sorted(set(old_role[~keep]))},
            "roles": roles, "expected_counts_hold": {r: (roles[r]["rows"], roles[r]["groups"]) == EXPECTED[r]
                                                     for r in ROLES},
            "assessment_by_pool": by_pool,
            "pool_rows_excluded_for_group_overlap": {p: int((overlap & (nominal_pool == p)).sum()) for p in POOLS},
            "groups_spanning_kept_role_and_other": int(span),
            "assessment_groups_shared_with_role": {r: int(len(set(unit[a].tolist()) &
                                                              set(unit[role == r].tolist()))) for r in FIT_ROLES},
            "float_vs_exact_rational_disagreements": {"rgj_dev_split": d1, "smf_assess_split": d2},
            "permitted_columns": {"n_columns": len(fn), "feature_names_sha256": sha_bytes("\n".join(fn).encode()),
                                  "forbidden_names_present": [f for f, l in zip(fn, low)
                                                              if l in ("sex", "race", "income", "occupation",
                                                                       "fnlwgt")],
                                  "one_hot_columns_binary": onehot_ok},
            "numeric_refit": numeric,
            "_private": {"keep": np.flatnonzero(keep), "role": role[keep], "pool": pool[keep], "X": Xk,
                         "row_id": row_id[keep]}}


def compare_source(ind):
    src = json.loads(pinned(OSF_MANIFEST))
    keys = ("rows", "groups", "row_id_sha256", "group_id_set_sha256")
    roles = {r: all(ind["roles"][r][k] == src["roles"][r][k] for k in keys) for r in ROLES}
    pools = {p: all(ind["assessment_by_pool"][p][k] == src["roles"][ASSESS]["by_pool"][p][k]
                    for k in ("rows", "groups", "row_id_sha256")) for p in POOLS}
    num = {c: (v["mean_osf_fit"] == src["numeric_refit"]["per_column"][c]["mean_osf_fit"]
               and v["sd_osf_fit"] == src["numeric_refit"]["per_column"][c]["sd_osf_fit"])
           for c, v in ind["numeric_refit"].items()}
    return {"source": f"{OSF_MANIFEST} at {PIN}", "source_sha256": sha_bytes(pinned(OSF_MANIFEST)),
            "roles_equal": roles, "pools_equal": pools, "numeric_refit_equal": num,
            "feature_names_sha256_equal": ind["permitted_columns"]["feature_names_sha256"] ==
            src["permitted_columns"]["feature_names_sha256"],
            "all_equal": all(roles.values()) and all(pools.values()) and all(num.values())}


def compare_loader(ind):
    """Separate step: dpc.data (which imports osf.data) is loaded only now, after the independent step."""
    sys.path.insert(0, str(WT))
    from dpc import data as DD
    D = DD.load()
    p = ind["_private"]
    same_rows = bool(np.array_equal(D["row_id"], p["row_id"]))
    out = {"loader": "dpc.data.load(verify=True, unseal=False)", "dpc_data_py_sha256": sha_file(WT / "dpc" / "data.py"),
           "rows_equal_in_order": same_rows,
           "role_per_row_equal": bool(same_rows and np.array_equal(D["role"].astype(str), p["role"])),
           "pool_per_row_equal": bool(same_rows and np.array_equal(D["pool"].astype(str), p["pool"])),
           "X_dtype": str(D["X"].dtype), "X_bitwise_equal_all_kept_rows": bool(same_rows and D["X"].dtype == p["X"].dtype
                                                                              and np.array_equal(D["X"], p["X"])),
           "feature_names_sha256_equal": sha_bytes("\n".join(str(f) for f in D["feature_names"]).encode()) ==
           ind["permitted_columns"]["feature_names_sha256"],
           "assessment_labels_sealed": {k: bool(np.all(D[k][D["idx"][ASSESS]] == -1)) for k in LABEL_KEYS},
           "loader_flag_sealed": bool(D["sealed"])}
    out["verdict"] = "MATCH" if (out["rows_equal_in_order"] and out["role_per_row_equal"] and out["pool_per_row_equal"]
                                 and out["X_bitwise_equal_all_kept_rows"] and all(out["assessment_labels_sealed"].values())
                                 and out["loader_flag_sealed"]) else "MISMATCH"
    return out


def main():
    t0 = time.time()
    ind = independent()
    loaded_during = guard()
    src = compare_source(ind)
    ldr = compare_loader(ind)
    pub = {k: v for k, v in ind.items() if k != "_private"}
    sec = {"produced_by": "results/pcrl_decision_preserving_compression_v1/provenance/role_check.py",
           "role_check_py_sha256": sha_file(Path(__file__)), "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                          time.gmtime()),
           "adapted_from": "results/pcrl_online_strength_frontier_v1/provenance/role_check.py at " + PIN,
           "independence": {"forbidden_packages": list(FORBIDDEN), "loaded_during_independent_step": loaded_during,
                            "imports_used": ["numpy", "hashlib", "fractions", "git show (pinned blobs)"],
                            "labels_read": "none"},
           "input": {"file": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz", "sha256": SRC_SHA, "matches": True},
           "recomputed": pub, "comparison_with_source_role_manifest": src, "comparison_with_dpc_loader": ldr,
           "verdict": "MATCH" if (src["all_equal"] and ldr["verdict"] == "MATCH"
                                  and all(pub["expected_counts_hold"].values())
                                  and pub["groups_spanning_kept_role_and_other"] == 0
                                  and all(v == 0 for v in pub["assessment_groups_shared_with_role"].values())
                                  and all(v == 0 for v in pub["float_vs_exact_rational_disagreements"].values()))
           else "MISMATCH", "wall_s": round(time.time() - t0, 1)}
    man = json.loads(OUT.read_text()) if OUT.exists() else {}
    man["independent_recomputation"] = sec
    txt = json.dumps(man, indent=1, default=str) + "\n"
    if re.search(r"/Users/|/Volumes/|" + re.escape(Path.home().name), txt):
        raise SystemExit("REFUSED: identifying path in a public file")
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(txt)
    tmp.replace(OUT)
    print(json.dumps({"verdict": sec["verdict"], "source_equal": src["all_equal"], "loader": ldr["verdict"],
                      "wall_s": sec["wall_s"]}, indent=1))


if __name__ == "__main__":
    main()
