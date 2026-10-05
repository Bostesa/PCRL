"""Independent recomputation of the online-strength frontier study's roles, consolidated assessment and numeric refit,
plus the custody determination of the certification pool's eligibility.

Owner: data/custody role. Written from the rule text (prompt section 6; osf/data.py docstring; the smf and rgj rule
texts already recomputed by results/pcrl_strength_matched_feedback_v1/provenance/role_check.py). The independent step
does NOT import osf, smf or rgj. It reads the admitted input file's non-label arrays (role, unit, row_id,
feature_names, X) and never reads sex, race, y_income or y_occupation_group.

Rule (no labels read; analysis unit = exact-record group `unit`):
  Inherited roles, recomputed from their own rules:
    rgj (seed 20261004): DEFENSE_FIT = old defense_train; AUDIT_FIT = old attacker_fit; INNER_SELECTION = old
        attacker_val; old defense_val by group, u = int(sha256("20261004|dev|<unit>")[:16], 16) / 2**64:
        HEAD_VALIDATION iff u < 0.30, else rgj DEVELOPMENT_ASSESSMENT.
    smf (seed 20261005): rgj DEFENSE_FIT groups with u_assess < 0.20 ("20261005|assess|<unit>") are
        NEW_DEVELOPMENT_ASSESSMENT, the rest NEW_DEFENSE_FIT; NEW_DEFENSE_FIT subroles by group (salt "critic"):
        CRITIC_FIT u < 0.70, CRITIC_VAL 0.70 <= u < 0.85, CONTROLLER_CALIB u >= 0.85.
  This study (osf), unchanged fitting/selection roles:
    OSF_DEFENSE_FIT = smf NEW_DEFENSE_FIT; HEAD_VALIDATION, AUDIT_FIT, INNER_SELECTION = the smf roles;
    subroles CRITIC_FIT, CRITIC_VAL, DIAGNOSTIC_CALIB (= smf CONTROLLER_CALIB; no controller is trained).
  OSF_DEVELOPMENT_ASSESSMENT = the fixed union of four previously used pools
    ORIG_ASSESSMENT = old assessment; RGJ_DEV = rgj DEVELOPMENT_ASSESSMENT; SMF_DEV = smf NEW_DEVELOPMENT_ASSESSMENT;
    CERT = old cert, only if custody establishes eligibility (--cert-eligibility; both variants are published)
  minus every exact-record group that also has a row in OSF_DEFENSE_FIT, HEAD_VALIDATION, AUDIT_FIT,
  INNER_SELECTION, excluded_exposure or excluded_dup. A group is never split.
Numeric refit: the 5 numeric columns are re-standardised (population sd) on OSF_DEFENSE_FIT rows only. Raw integers are
regenerated with the pinned loader (jcv.data.raw_frames(); raw files hash-checked there and here; only the 10
permitted source columns are kept from the loader's frames, no label column is read) and tied to the admitted rows by
bitwise equality of all 83 admitted columns. Checked for every kept row, including the newly kept assessment pools.

Usage (from the worktree root, OMP_NUM_THREADS=1, PYTHONPATH=.):
  ~/PCRL/.venv/bin/python results/pcrl_online_strength_frontier_v1/provenance/role_check.py
      -> ROLE_MANIFEST.json (independent sections)
  ... role_check.py --cert-eligibility
      -> separate, label-free custody step: code pins, release row sets, head scaler moments, LEACE fit-row and
         fit-feature hashes, FARE tree fit-row fingerprints of every eligible smf model; writes "cert_eligibility" and
         "assessment_exclusion_from_eligible_models" (and cross-references ADMISSION.json when it exists)
  ... role_check.py --compare-loader
      -> separate step: reruns the independent step, then imports osf.data, runs its sealed load(), and records an
         exact comparison (role/pool sets, counts and row-id hashes, X bitwise for every kept row, numeric refit,
         label sealing, allowlist guard, the cert_eligible=False variant of the pure role function, fitting-tensor
         identity with smf) in "loader_comparison"
Publishes counts, hashes and booleans only. No row id, record key, group id or label statistic of any role is written.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
WT = PKG.parents[1]
PRIVATE_CACHE = Path.home() / "PCRL_eval_cache_private"
SRC = PRIVATE_CACHE / "jcv_v1" / "inputs" / "adult_jcv.npz"
SRC_PUBLIC = "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz"
SRC_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
PRED_ADMISSION = WT / "results" / "pcrl_joint_complete_view_method_v1" / "DATA_ADMISSION.json"
PRED_ADMISSION_SHA = "fa49b4fd1cc72b497fc7f0268d1748bd6bb8da623a46575a0b71d75247591c55"
SMF_PKG = WT / "results" / "pcrl_strength_matched_feedback_v1"
SMF_MANIFEST = SMF_PKG / "ROLE_MANIFEST.json"
RGJ_MANIFEST = WT / "results" / "pcrl_refreshed_guarded_joint_v1" / "ROLE_MANIFEST.json"
SOURCE_PIN = "a9951ed2fed9943d445a208a8a7e456a56f39114"         # smf closed evidence (prompt section 2)
RGJ_PIN = "ccdcc5a372d538cd12332e496e08f7753f7c9922"
RAW_SHA = {"adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
           "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"}
OUT = PKG / "ROLE_MANIFEST.json"
SMF_UNITS = PRIVATE_CACHE / "smf_v1" / "run" / "units"
SMF_FARE = PRIVATE_CACHE / "smf_v1" / "fare_cache"
SMF_ACTIVITY = PRIVATE_CACHE / "smf_v1" / "run" / "ACTIVITY_LOG.jsonl"

RGJ_SEED, SEED = 20261004, 20261005
HEAD_SHARE = Fraction(3, 10)
ASSESS_SHARE = Fraction(1, 5)
CRIT_A, CRIT_B = Fraction(7, 10), Fraction(17, 20)
RGJ_KEPT = {"defense_train": "DEFENSE_FIT", "attacker_fit": "AUDIT_FIT", "attacker_val": "INNER_SELECTION"}
EXCLUSIONS = ["excluded_exposure", "excluded_dup"]
FIT_ROLES = ["OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"]
ASSESS = "OSF_DEVELOPMENT_ASSESSMENT"
ROLES = ["OSF_DEFENSE_FIT", ASSESS, "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"]
SUBROLES = ["CRITIC_FIT", "CRITIC_VAL", "DIAGNOSTIC_CALIB"]
SMF_SUBROLES = ["CRITIC_FIT", "CRITIC_VAL", "CONTROLLER_CALIB"]
RGJ_SUBROLES = ["CRITIC_FIT", "CRITIC_VAL", "CALIB"]
POOLS = ["ORIG_ASSESSMENT", "RGJ_DEV", "SMF_DEV", "CERT"]
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "relationship", "native-country"]
EXCLUDED_SOURCE = ["sex", "race", "income", "occupation", "fnlwgt", "row id / record key / group id"]
LABEL_KEYS = ("sex", "race", "y_income", "y_occupation_group")
SEEDS = (0, 1, 2)
FARE_IDS = (1, 2, 3, 4, 5, 6)

RULE_TEXT = (
    "Fitting/selection roles are the strength-matched feedback study's, unchanged: OSF_DEFENSE_FIT = smf "
    "NEW_DEFENSE_FIT (rgj DEFENSE_FIT = old defense_train groups with u = int(sha256('20261005|assess|<unit>')[:16], 16)"
    " / 2**64 >= 0.20); HEAD_VALIDATION (old defense_val groups with u('20261004|dev') < 0.30), AUDIT_FIT (old "
    "attacker_fit) and INNER_SELECTION (old attacker_val) unchanged; OSF_DEFENSE_FIT subroles by group, salt 'critic', "
    "seed 20261005: CRITIC_FIT u < 0.70, CRITIC_VAL 0.70 <= u < 0.85, DIAGNOSTIC_CALIB u >= 0.85 (= smf "
    "CONTROLLER_CALIB). OSF_DEVELOPMENT_ASSESSMENT is the fixed union of ORIG_ASSESSMENT (old assessment), RGJ_DEV (rgj "
    "DEVELOPMENT_ASSESSMENT = old defense_val groups with u('20261004|dev') >= 0.30), SMF_DEV (smf "
    "NEW_DEVELOPMENT_ASSESSMENT) and CERT (old cert, only if custody establishes it never entered the eligible models' "
    "fitting or selection), minus every exact-record group that also has a row in a fitting role or in "
    "excluded_exposure / excluded_dup; groups are never split. The allocation reads only the old role array and the "
    "group ids; no label is read. The 5 numeric columns are re-standardised (population sd) on OSF_DEFENSE_FIT rows only "
    "and applied to every kept row; the 78 one-hot columns use the loader's fixed category sets.")


# ----------------------------------------------------------------------------------------------- helpers
def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rowid_hash(r) -> str:
    """Convention shared with jcv/rgj/smf manifests: sha256 of the sorted row ids as int64 raw bytes."""
    return sha_bytes(np.ascontiguousarray(np.sort(np.asarray(r)).astype(np.int64)).tobytes())


def unit_set_hash(u) -> str:
    return sha_bytes(np.ascontiguousarray(np.unique(np.asarray(u)).astype(np.int64)).tobytes())


def sha_arrays(*arrays) -> str:
    """oar.fare_official._sha256_arrays convention (dtype + shape + bytes), re-implemented."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def u_int(seed: int, salt: str, unit: int) -> int:
    return int(hashlib.sha256(f"{seed}|{salt}|{int(unit)}".encode()).hexdigest()[:16], 16)


def split_groups(units, seed, salt, cuts_exact, cuts_float, names):
    """Group -> class by its hash, float rule as written AND exact rational rule; returns (map, disagreements)."""
    out, disagree = {}, 0
    for g in np.unique(units).tolist():
        n = u_int(seed, salt, g)
        uf, ux = n / 2.0 ** 64, Fraction(n, 2 ** 64)
        kf = next((names[i] for i, c in enumerate(cuts_float) if uf < c), names[-1])
        kx = next((names[i] for i, c in enumerate(cuts_exact) if ux < c), names[-1])
        disagree += int(kf != kx)
        out[g] = kf
    return out, disagree


def role_record(ix, row_id, unit):
    return {"rows": int(len(ix)), "groups": int(len(np.unique(unit[ix]))), "row_id_sha256": rowid_hash(row_id[ix]),
            "group_id_set_sha256": unit_set_hash(unit[ix])}


def git(*a):
    return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True).stdout.strip()


def git_blob_sha256(commit, path):
    r = subprocess.run(["git", "-C", str(WT), "show", f"{commit}:{path}"], capture_output=True)
    return sha_bytes(r.stdout) if r.returncode == 0 else None


def jload(p):
    return json.loads(Path(p).read_text())


def write_manifest(man):
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=1) + "\n")
    tmp.replace(OUT)


# ----------------------------------------------------------------------------------------------- partition
def allocate(old_role, unit):
    """Per-row inherited roles, osf fitting roles/subroles and pool membership (pure function of role array + groups)."""
    n = len(old_role)
    rgj = np.array([RGJ_KEPT.get(r, "") for r in old_role.tolist()], dtype=object)
    dv = old_role == "defense_val"
    gm, d1 = split_groups(unit[dv], RGJ_SEED, "dev", [HEAD_SHARE], [0.30], ["HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT"])
    rgj[dv] = [gm[g] for g in unit[dv].tolist()]
    rgj_sub = np.array([""] * n, dtype=object)
    df = rgj == "DEFENSE_FIT"
    sm, d2 = split_groups(unit[df], RGJ_SEED, "critic", [CRIT_A, CRIT_B], [0.70, 0.85], RGJ_SUBROLES)
    rgj_sub[df] = [sm[g] for g in unit[df].tolist()]

    smf = np.array([""] * n, dtype=object)
    am, d3 = split_groups(unit[df], SEED, "assess", [ASSESS_SHARE], [0.20], ["NEW_DEVELOPMENT_ASSESSMENT",
                                                                            "NEW_DEFENSE_FIT"])
    smf[df] = [am[g] for g in unit[df].tolist()]
    for r in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
        smf[rgj == r] = r
    smf_sub = np.array([""] * n, dtype=object)
    nf = smf == "NEW_DEFENSE_FIT"
    cm, d4 = split_groups(unit[nf], SEED, "critic", [CRIT_A, CRIT_B], [0.70, 0.85], SMF_SUBROLES)
    smf_sub[nf] = [cm[g] for g in unit[nf].tolist()]

    fit = np.array([""] * n, dtype=object)
    fit[smf == "NEW_DEFENSE_FIT"] = "OSF_DEFENSE_FIT"
    for r in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
        fit[smf == r] = r
    sub = np.array([{"CONTROLLER_CALIB": "DIAGNOSTIC_CALIB"}.get(s, s) for s in smf_sub.tolist()], dtype=object)
    pool = np.array([""] * n, dtype=object)
    pool[old_role == "assessment"] = "ORIG_ASSESSMENT"
    pool[rgj == "DEVELOPMENT_ASSESSMENT"] = "RGJ_DEV"
    pool[smf == "NEW_DEVELOPMENT_ASSESSMENT"] = "SMF_DEV"
    pool[old_role == "cert"] = "CERT"
    disagree = {"rgj_dev_split": d1, "rgj_critic_split": d2, "smf_assess_split": d3, "smf_critic_split": d4}
    return {"rgj": rgj.astype(str), "rgj_sub": rgj_sub.astype(str), "smf": smf.astype(str),
            "smf_sub": smf_sub.astype(str), "fit": fit.astype(str), "sub": sub.astype(str), "pool": pool.astype(str),
            "disagree": disagree}


def consolidate(A, old_role, unit, cert_eligible):
    """Final role per row for one CERT variant. Returns (role, pool_kept, per-pool admission info)."""
    fit, pool = A["fit"], A["pool"]
    cand = (pool != "") & ((pool != "CERT") | bool(cert_eligible))
    blocked = np.unique(unit[(fit != "") | np.isin(old_role, EXCLUSIONS)])
    overlap = cand & np.isin(unit, blocked)
    keep_pool = cand & ~overlap
    role = fit.copy().astype(object)
    role[keep_pool] = ASSESS
    info = {}
    for p in POOLS:
        nom = pool == p
        info[p] = {"nominal_rows": int(nom.sum()), "nominal_groups": int(len(np.unique(unit[nom]))),
                   "nominal_row_id_sha256": None,
                   "eligible": bool(cert_eligible) if p == "CERT" else True,
                   "excluded_rows_group_overlap": int((overlap & nom).sum()),
                   "excluded_rows_ineligible_pool": int(nom.sum()) if (p == "CERT" and not cert_eligible) else 0,
                   "kept_rows": int((keep_pool & nom).sum()),
                   "kept_groups": int(len(np.unique(unit[keep_pool & nom])))}
    return role.astype(str), np.where(keep_pool, pool, ""), info


# ----------------------------------------------------------------------------------------------- numeric refit
def raw_permitted(n_expected):
    """Raw values of the 10 permitted source columns, row-aligned with adult_jcv.npz (row_id = position in
    concat[test, train]). The pinned loader's frames are reduced to the permitted columns immediately."""
    root = Path.home() / "PCRL" / "data" / "adult"
    raw_hashes = {f: sha_file(root / f) for f in RAW_SHA}
    assert raw_hashes == RAW_SHA, f"raw Adult files changed: {raw_hashes}"
    sys.path.insert(0, str(WT))
    import pandas as pd
    from jcv.data import raw_frames            # pinned loader wrapper (hash-checks the raw files itself)
    from pcrl.data.adult import AdultDataset   # fixed category sets only
    te, tr = raw_frames()
    raw = pd.concat([te[NUMERIC + CATEGORICAL], tr[NUMERIC + CATEGORICAL]], ignore_index=True)
    del te, tr
    assert len(raw) == n_expected, (len(raw), n_expected)
    return (raw, AdultDataset.CATEGORY_VALUES, raw_hashes, sha_file(WT / "pcrl" / "data" / "adult.py"),
            sha_file(WT / "jcv" / "data.py"))


def numeric_check(X, fn, raw, cats, old_role, fit_mask, keep, groups_of_rows):
    """Admitted-normalisation inversion vs raw integers (every row), one-hot vs raw indicators (every row), expected
    refit on OSF_DEFENSE_FIT applied to every kept row. groups_of_rows: name -> boolean row mask for per-role reports."""
    adm = jload(PRED_ADMISSION)
    R = {}
    for c in NUMERIC:
        v = raw[c].to_numpy()
        assert np.issubdtype(v.dtype, np.integer), f"{c}: raw dtype {v.dtype} is not integer"
        R[c] = v.astype(np.int64)
    onehot_equal, onehot_rows_ok = {}, np.ones(len(X), bool)
    for c in CATEGORICAL:
        vals = raw[c].to_numpy().astype(str)
        exp = (vals[:, None] == np.array(cats[c], dtype=str)[None, :]).astype(np.float32)
        cols = [fn.index(f"{c}={v}") for v in cats[c]]
        eq_rows = np.all(X[:, cols] == exp, axis=1)
        onehot_rows_ok &= eq_rows
        onehot_equal[c] = bool(eq_rows.all())
    old_fit = old_role == "defense_train"
    per, expected = {}, {}
    inv_ok_rows = np.ones(len(X), bool)
    nf = np.flatnonzero(fit_mask)
    for c in NUMERIC:
        j = fn.index(c)
        r = R[c].astype(np.float64)
        mu_o, sd_o = float(r[old_fit].mean()), float(r[old_fit].std())
        mu_a, sd_a = adm["numeric_norm"][c]
        x_old = ((r - mu_o) / sd_o).astype(np.float32)
        inv = X[:, j].astype(np.float64) * sd_a + mu_a
        rnd = np.round(inv)
        inv_ok_rows &= (rnd == r) & (np.abs(inv - r) < 0.05)
        m, s = float(r[nf].mean()), float(r[nf].std())
        expected[c] = ((r - m) / s).astype(np.float32)
        per[c] = {
            "admitted_norm_rederived_from_raw_on_old_defense_train": bool(mu_o == mu_a and sd_o == sd_a),
            "admitted_column_equals_raw_standardised_bitwise_all_rows": bool(np.array_equal(x_old, X[:, j])),
            "inversion_max_abs_err_all_rows": float(np.abs(inv - r).max()),
            "inversion_max_abs_err_kept_rows": float(np.abs(inv - r)[keep].max()),
            "rounded_inversion_equals_raw_integers_all_rows": bool(np.array_equal(rnd, r)),
            "mean_osf_fit": m, "sd_osf_fit": s, "osf_fit_rows": int(len(nf)),
            "refit_differs_from_admitted_norm": bool(m != mu_a or s != sd_a)}
    by_group = {g: {"rows": int(msk.sum()),
                    "raw_numerics_recovered_exactly": int((inv_ok_rows & msk).sum()),
                    "onehot_equals_raw_indicators": int((onehot_rows_ok & msk).sum())}
                for g, msk in groups_of_rows.items()}
    out = {"raw_numeric_columns_integer": True, "onehot_columns_equal_raw_indicators_all_rows": onehot_equal,
           "per_column": per, "recovered_raw_numerics_by_role_and_pool": by_group}
    out["all_ok"] = bool(all(onehot_equal.values()) and all(
        v["admitted_norm_rederived_from_raw_on_old_defense_train"]
        and v["admitted_column_equals_raw_standardised_bitwise_all_rows"]
        and v["rounded_inversion_equals_raw_integers_all_rows"] and v["inversion_max_abs_err_all_rows"] < 0.05
        for v in per.values()) and all(v["rows"] == v["raw_numerics_recovered_exactly"] == v["onehot_equals_raw_indicators"]
                                       for v in by_group.values()))
    Xexp = X.copy()
    for c in NUMERIC:
        Xexp[:, fn.index(c)] = expected[c]
    return out, Xexp


# ----------------------------------------------------------------------------------------------- independent step
def independent(write=True):
    src_sha = sha_file(SRC)
    assert src_sha == SRC_SHA, "REFUSED: adult_jcv.npz hash mismatch"
    adm_sha = sha_file(PRED_ADMISSION)
    assert adm_sha == PRED_ADMISSION_SHA, "REFUSED: predecessor DATA_ADMISSION.json changed"
    z = np.load(SRC, allow_pickle=False)
    files = list(z.files)
    old_role, unit, row_id = z["role"].astype(str), z["unit"].astype(np.int64), z["row_id"].astype(np.int64)
    fn = [str(f) for f in z["feature_names"]]
    X = z["X"]
    del z
    n = len(old_role)
    assert np.array_equal(row_id, np.arange(n)), "row ids are not 0..n-1"

    adm = jload(PRED_ADMISSION)
    assert adm["private_inputs_sha256"] == SRC_SHA
    old = {r: role_record(np.flatnonzero(old_role == r), row_id, unit) for r in sorted(set(old_role.tolist()))}
    old_match = {r: bool(adm["roles"].get(r) and all(adm["roles"][r][k] == old[r][k]
                                                     for k in ("rows", "groups", "row_id_sha256"))) for r in old}
    assert set(old) == set(adm["roles"]) and all(old_match.values()), old_match
    x_match = sha_bytes(np.ascontiguousarray(X).tobytes()) == adm["X_sha256"]
    assert x_match, "X differs from predecessor DATA_ADMISSION"

    A = allocate(old_role, unit)
    rgj, rgj_sub, smf, smf_sub, fit, sub, pool = (A[k] for k in ("rgj", "rgj_sub", "smf", "smf_sub", "fit", "sub",
                                                                 "pool"))

    # ---- inherited roles must equal their published manifests (blobs pinned at the source commits)
    rgj_blob_ok = {"at_rgj_pin": git("rev-parse", f"{RGJ_PIN}:results/pcrl_refreshed_guarded_joint_v1/ROLE_MANIFEST.json")
                   == git("hash-object", str(RGJ_MANIFEST)),
                   "at_source_pin": git("rev-parse", f"{SOURCE_PIN}:results/pcrl_refreshed_guarded_joint_v1/"
                                                     "ROLE_MANIFEST.json") == git("hash-object", str(RGJ_MANIFEST))}
    smf_blob_ok = git("rev-parse", f"{SOURCE_PIN}:results/pcrl_strength_matched_feedback_v1/ROLE_MANIFEST.json") == \
        git("hash-object", str(SMF_MANIFEST))
    rm, sm = jload(RGJ_MANIFEST), jload(SMF_MANIFEST)
    rgj_pub = {**rm["roles"], **rm["defense_fit_subroles"]}
    rgj_mine = {r: role_record(np.flatnonzero(rgj == r), row_id, unit)
                for r in ["DEFENSE_FIT", "HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT", "AUDIT_FIT", "INNER_SELECTION"]}
    rgj_mine.update({r: role_record(np.flatnonzero(rgj_sub == r), row_id, unit) for r in RGJ_SUBROLES})
    keys4 = ("rows", "groups", "row_id_sha256", "group_id_set_sha256")
    rgj_match = {r: all(rgj_mine[r][k] == rgj_pub[r][k] for k in keys4) for r in rgj_mine}
    smf_pub = {**sm["roles"], **sm["new_defense_fit_subroles"]}
    smf_mine = {r: role_record(np.flatnonzero(smf == r), row_id, unit)
                for r in ["NEW_DEFENSE_FIT", "NEW_DEVELOPMENT_ASSESSMENT", "HEAD_VALIDATION", "AUDIT_FIT",
                          "INNER_SELECTION"]}
    smf_mine.update({r: role_record(np.flatnonzero(smf_sub == r), row_id, unit) for r in SMF_SUBROLES})
    smf_match = {r: all(smf_mine[r][k] == smf_pub[r][k] for k in keys4) for r in smf_mine}
    assert all(rgj_blob_ok.values()) and smf_blob_ok and all(rgj_match.values()) and all(smf_match.values()), \
        (rgj_blob_ok, smf_blob_ok, rgj_match, smf_match)

    # ---- the two CERT variants
    variants = {}
    for ce in (True, False):
        role, pkeep, info = consolidate(A, old_role, unit, ce)
        for p in POOLS:
            info[p]["nominal_row_id_sha256"] = rowid_hash(row_id[pool == p])
            info[p]["kept_row_id_sha256"] = rowid_hash(row_id[pkeep == p])
        a = np.flatnonzero(role == ASSESS)
        variants[ce] = {"role": role, "pool": pkeep, "info": info,
                        "assessment": {**role_record(a, row_id, unit),
                                       "by_pool": {p: {"rows": info[p]["kept_rows"], "groups": info[p]["kept_groups"],
                                                       "row_id_sha256": info[p]["kept_row_id_sha256"]} for p in POOLS},
                                       "dropped_rows_total": int((role == "").sum()),
                                       "dropped_rows_by_old_role": {r: int(((role == "") & (old_role == r)).sum())
                                                                    for r in sorted(set(old_role[role == ""].tolist()))}}}

    # published pool identities: old assessment / old cert (smf manifest old_roles), RGJ_DEV and SMF_DEV
    pool_pub = {"ORIG_ASSESSMENT": sm["old_roles"]["per_role"]["assessment"],
                "RGJ_DEV": rgj_pub["DEVELOPMENT_ASSESSMENT"], "SMF_DEV": smf_pub["NEW_DEVELOPMENT_ASSESSMENT"],
                "CERT": sm["old_roles"]["per_role"]["cert"]}
    pool_rec = {p: role_record(np.flatnonzero(pool == p), row_id, unit) for p in POOLS}
    pool_match = {p: all(pool_rec[p][k] == pool_pub[p][k] for k in keys4) for p in POOLS}
    assert all(pool_match.values()), pool_match

    # ---- assertions (both variants)
    A_ = {}
    blocked = np.unique(unit[(fit != "") | np.isin(old_role, EXCLUSIONS)])
    for ce, V in variants.items():
        role = V["role"]
        tag = "with_CERT" if ce else "without_CERT"
        idx = {r: np.flatnonzero(role == r) for r in ROLES}
        pairs = [(a, b) for i, a in enumerate(ROLES) for b in ROLES[i + 1:]]
        A_[f"{tag}: roles_row_disjoint"] = all(len(np.intersect1d(idx[a], idx[b])) == 0 for a, b in pairs)
        A_[f"{tag}: roles_group_disjoint"] = all(len(np.intersect1d(unit[idx[a]], unit[idx[b]])) == 0 for a, b in pairs)
        keep = role != ""
        A_[f"{tag}: no_kept_group_occurs_in_a_dropped_row"] = len(np.intersect1d(unit[keep], unit[~keep])) == 0
        A_[f"{tag}: no_assessment_group_overlaps_a_fitting_role_or_exclusion"] = bool(
            len(np.intersect1d(unit[role == ASSESS], unit[(fit != "") | np.isin(old_role, EXCLUSIONS)])) == 0)
        A_[f"{tag}: assessment_rows_come_only_from_the_named_pools"] = bool(np.all(V["pool"][role == ASSESS] != ""))
        overlap = (pool != "") & ((pool != "CERT") | ce) & np.isin(unit, blocked)
        exp_drop = np.isin(old_role, EXCLUSIONS) | ((pool == "CERT") & (not ce)) | overlap
        A_[f"{tag}: dropped_rows_are_exactly_exclusions_plus_ineligible_or_overlapping_pool_rows"] = bool(
            np.array_equal(~keep, exp_drop))
        A_[f"{tag}: no_exclusion_row_kept"] = bool(not np.any(keep & np.isin(old_role, EXCLUSIONS)))
    A_["fitting_roles_equal_smf_roles"] = all(np.array_equal(np.flatnonzero(fit == f), np.flatnonzero(smf == s))
                                              for f, s in (("OSF_DEFENSE_FIT", "NEW_DEFENSE_FIT"),
                                                           ("HEAD_VALIDATION", "HEAD_VALIDATION"),
                                                           ("AUDIT_FIT", "AUDIT_FIT"),
                                                           ("INNER_SELECTION", "INNER_SELECTION")))
    A_["subroles_equal_smf_subroles"] = all(np.array_equal(np.flatnonzero(sub == a), np.flatnonzero(smf_sub == b))
                                            for a, b in zip(SUBROLES, SMF_SUBROLES))
    A_["subroles_partition_OSF_DEFENSE_FIT"] = bool(np.array_equal(np.flatnonzero(sub != ""),
                                                                   np.flatnonzero(fit == "OSF_DEFENSE_FIT")))
    sp = [(a, b) for i, a in enumerate(SUBROLES) for b in SUBROLES[i + 1:]]
    A_["subroles_row_and_group_disjoint"] = all(
        len(np.intersect1d(unit[sub == a], unit[sub == b])) == 0 for a, b in sp)
    A_["pools_are_row_disjoint_and_hold_no_fitting_row"] = bool(np.all(fit[pool != ""] == ""))
    A_["pool_sources_equal_published_inherited_manifests"] = all(pool_match.values())
    A_["inherited_roles_recomputed_equal_published_manifests"] = bool(all(rgj_match.values()) and all(smf_match.values()))
    A_["float_rule_equals_exact_rational_rule_for_every_group"] = all(v == 0 for v in A["disagree"].values())
    assert all(A_.values()), {k: v for k, v in A_.items() if not v}

    # group sharing between pools (allowed: both sides are assessment rows; reported, never split)
    pg = {p: set(unit[pool == p].tolist()) for p in POOLS}
    shared = {f"{a}&{b}": len(pg[a] & pg[b]) for i, a in enumerate(POOLS) for b in POOLS[i + 1:]}

    # ---- main variant = custody determination (published separately); default nominal union with CERT
    prev = jload(OUT) if OUT.exists() else {}
    verdict = (prev.get("cert_eligibility") or {}).get("verdict")
    ce_main = verdict != "NOT_ESTABLISHED"
    V = variants[ce_main]
    role = V["role"]
    idx = {r: np.flatnonzero(role == r) for r in ROLES}
    idx.update({r: np.flatnonzero(sub == r) for r in SUBROLES})
    roles = {r: role_record(idx[r], row_id, unit) for r in ROLES + SUBROLES}
    use = {"OSF_DEFENSE_FIT": "encoders, warm starts, training heads, critics (subroles), numeric standardisation, "
                              "official LEACE maps and FARE / compression trees, deployed-head fitting",
           ASSESS: "withheld from every fitting/selection procedure; labels masked (-1) at load until the pushed "
                   "EVALUATION_LOCK; scored once after it (consolidated, historically exposed benchmark)",
           "HEAD_VALIDATION": "deployed task-head C selection only",
           "AUDIT_FIT": "inner and final attacker fitting",
           "INNER_SELECTION": "candidate evaluation, nomination, attacker and orientation selection",
           "CRITIC_FIT": "critic fitting (encoders also train on these rows)",
           "CRITIC_VAL": "held out from critic fitting (diagnostics); encoders train on these rows",
           "DIAGNOSTIC_CALIB": "held out from critic fitting (diagnostic calibration; no controller); encoders train "
                               "on these rows"}
    src_txt = {"OSF_DEFENSE_FIT": "smf NEW_DEFENSE_FIT (unchanged)",
               ASSESS: "union of ORIG_ASSESSMENT, RGJ_DEV, SMF_DEV" + (", CERT" if ce_main else "") +
                       " minus groups overlapping a fitting role or an exclusion",
               "HEAD_VALIDATION": "smf/rgj HEAD_VALIDATION (unchanged)", "AUDIT_FIT": "smf/rgj AUDIT_FIT (unchanged)",
               "INNER_SELECTION": "smf/rgj INNER_SELECTION (unchanged)"}
    for r in ROLES + SUBROLES:
        roles[r]["source"] = src_txt.get(r, "OSF_DEFENSE_FIT (smf group-hash split, seed 20261005, salt 'critic')")
        roles[r]["permitted_use"] = use[r]
    roles[ASSESS]["by_pool"] = V["assessment"]["by_pool"]
    for r, s in zip(["OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"] + SUBROLES,
                    ["NEW_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"] + SMF_SUBROLES):
        roles[r]["equals_smf_published_role"] = s

    # ---- permitted columns (input-only checks)
    base = [f.split("=")[0] for f in fn]
    per_source = {c: base.count(c) for c in dict.fromkeys(base)}
    forbidden_tokens = ["sex", "race", "income", "occupation", "fnlwgt", "row_id", "record", "unit"]
    forbidden_present = [f for f in fn if f.split("=")[0].strip().lower() in forbidden_tokens]
    keep = role != ""
    blocks = {c: X[:, [i for i, b in enumerate(base) if b == c]].sum(1) for c in CATEGORICAL}
    exact_onehot_rows = np.all(np.isin(X[:, len(NUMERIC):], (0.0, 1.0)), axis=1) & np.all(
        np.stack([blocks[c] == 1 for c in CATEGORICAL], 1), axis=1)
    masks = {r: role == r for r in ROLES}
    masks.update({f"{ASSESS}:{p}": V["pool"] == p for p in POOLS})
    masks["CERT (both variants, nominal)"] = pool == "CERT"
    unseen = {g: int((~exact_onehot_rows & m).sum()) for g, m in masks.items()}

    # ---- numeric refit from the pinned loader's raw rows, every kept row
    raw, cats, raw_hashes, loader_sha, jcv_data_sha = raw_permitted(n)
    num, Xexp = numeric_check(X, fn, raw, cats, old_role, fit == "OSF_DEFENSE_FIT", keep, masks)
    assert num["all_ok"], num
    del raw
    smf_num = sm["numeric_refit"]["per_column"]
    num["equals_smf_published_refit"] = {c: bool(num["per_column"][c]["mean_osf_fit"] == smf_num[c]["mean_new_fit"]
                                                 and num["per_column"][c]["sd_osf_fit"] == smf_num[c]["sd_new_fit"])
                                         for c in NUMERIC}
    assert all(num["equals_smf_published_refit"].values())

    # ---- permitted-input duplicates (exact equality of the 83 inputs; invariant under the per-column refit)
    Xb = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
    fit_set = set(Xb[idx["OSF_DEFENSE_FIT"]].tolist())
    x_dup = {g: int(sum(x in fit_set for x in Xb[m].tolist())) for g, m in masks.items() if g != "OSF_DEFENSE_FIT"}

    fingerprint = sha_bytes("|".join(f"{r}:{roles[r]['row_id_sha256']}" for r in ROLES + SUBROLES).encode())
    man = {
        "schema": "osf-role-manifest-v1",
        "study": "pcrl_online_strength_frontier_v1",
        "produced_by": "results/pcrl_online_strength_frontier_v1/provenance/role_check.py (independent step imports "
                       "no osf, smf or rgj module)",
        "role_check_py_sha256": sha_file(Path(__file__)),
        "all_rows_historically_exposed": True,
        "exposure_note": ("Every row in every role is a previously used UCI Adult row (EXPOSURE_LEDGER.md). "
                          "OSF_DEVELOPMENT_ASSESSMENT is a consolidated, reused benchmark of four previously used "
                          "pools; it is not fresh data, its scoring is not confirmation and not an independent "
                          "replication, and earlier outcomes on these pools influenced this research direction."),
        "source": {"file": SRC_PUBLIC, "sha256": src_sha, "sha256_matches_admitted": True,
                   "arrays_present": files,
                   "arrays_read_by_independent_step": ["role", "unit", "row_id", "feature_names", "X"],
                   "label_arrays_read_by_independent_step": [],
                   "rows_total": n, "groups_total": int(len(np.unique(unit))),
                   "predecessor_admission": "results/pcrl_joint_complete_view_method_v1/DATA_ADMISSION.json",
                   "predecessor_admission_sha256": adm_sha,
                   "X_sha256_matches_predecessor_admission": x_match,
                   "source_pin": SOURCE_PIN,
                   "smf_role_manifest_blob_equals_source_pin": smf_blob_ok,
                   "rgj_role_manifest_blob_equals_pins": rgj_blob_ok},
        "rule": {"text": RULE_TEXT, "seed_smf": SEED, "seed_rgj": RGJ_SEED, "smf_assessment_share": 0.20,
                 "rgj_head_share": 0.30, "critic_split": [0.70, 0.85],
                 "hash": "sha256 of the UTF-8 string '<seed>|<salt>|<unit as decimal int>', first 16 hex digits / 2**64",
                 "salts": {"rgj defense_val split (seed 20261004)": "dev",
                           "smf assessment split of rgj DEFENSE_FIT (seed 20261005)": "assess",
                           "OSF_DEFENSE_FIT subroles (seed 20261005; = smf subroles)": "critic"},
                 "analysis_unit": "exact-record group `unit` (de-duplicated full raw record incl. labels and fnlwgt; "
                                  "Adult has no household identifiers); a group is never split; rows identical on the "
                                  "83 inputs but different groups are different people and are never pooled",
                 "labels_read_to_form_roles": False,
                 "float_vs_exact_rational_disagreements": A["disagree"]},
        "old_roles": {"definition": "oar-roles-v1 as admitted by jcv", "per_role": old,
                      "matches_predecessor_DATA_ADMISSION": old_match},
        "inherited_roles_recomputed": {
            "rgj": {"per_role": rgj_mine, "matches_published_manifest": rgj_match},
            "smf": {"per_role": smf_mine, "matches_published_manifest": smf_match}},
        "cert_variant_in_roles": "with_CERT" if ce_main else "without_CERT",
        "cert_variant_basis": ("custody determination in 'cert_eligibility' (verdict "
                               f"{verdict or 'not yet recorded; nominal union shown'})"),
        "roles": {r: roles[r] for r in ROLES},
        "defense_fit_subroles": {r: roles[r] for r in SUBROLES},
        "subrole_note": ("Subroles are the smf draw unchanged (CONTROLLER_CALIB renamed DIAGNOSTIC_CALIB; no controller "
                         "is trained). Encoders and training heads train on all OSF_DEFENSE_FIT rows; the subroles "
                         "are held out only from critic/diagnostic fitting."),
        "assessment_pools": {p: {**V["info"][p], "source": {"ORIG_ASSESSMENT": "old assessment (oar-roles-v1)",
                                                            "RGJ_DEV": "rgj DEVELOPMENT_ASSESSMENT",
                                                            "SMF_DEV": "smf NEW_DEVELOPMENT_ASSESSMENT",
                                                            "CERT": "old cert (oar-cert-v1 carve-out of attacker_fit)"}[p],
                                 "nominal_set_equals_published_source": pool_match[p]}
                             for p in POOLS},
        "consolidated_assessment_variants": {"with_CERT": variants[True]["assessment"],
                                             "without_CERT": variants[False]["assessment"]},
        "groups_shared_between_pools": shared,
        "role_partition_fingerprint_sha256": fingerprint,
        "assertions": A_,
        "permitted_columns": {
            "n_columns": len(fn), "feature_names_sha256": sha_bytes("\n".join(fn).encode()),
            "source_columns_kept": NUMERIC + CATEGORICAL, "columns_per_source": per_source,
            "excluded_source_columns": EXCLUDED_SOURCE, "forbidden_column_names_present": forbidden_present,
            "relationship_Husband_present": "relationship=Husband" in fn,
            "relationship_Wife_present": "relationship=Wife" in fn,
            "proxy_note": ("relationship=Husband / relationship=Wife are retained permitted proxies (prompt section 6: "
                           "do not remove difficult proxies); see EXPOSURE_LEDGER.md."),
            "rows_not_exact_one_hot_by_role_and_pool": unseen,
            "categorical_handling": ("fixed loader category sets (pcrl.data.adult.AdultDataset.CATEGORY_VALUES); no "
                                     "fitted vocabulary; an unseen value gives an all-zero block (none occurs in any "
                                     "kept row)")},
        "numeric_refit": {
            "rule": ("raw integer values recovered from the admitted normalisation and re-standardised with population "
                     "mean/sd computed on OSF_DEFENSE_FIT rows only; the transform is applied to all kept rows"),
            "raw_source": ("jcv.data.raw_frames() -> pinned pcrl.data.adult loader on hash-checked adult.data / "
                           "adult.test (frames reduced to the 10 permitted source columns at once); row alignment by "
                           "bitwise equality of the 78 one-hot columns and of the 5 admitted numeric columns "
                           "re-derived from the raw values"),
            "raw_files_sha256": raw_hashes, "pcrl_data_adult_py_sha256": loader_sha, "jcv_data_py_sha256": jcv_data_sha,
            **num},
        "input_duplicate_note": {
            "what": ("Rows per role/pool whose 83-column input vector also occurs in OSF_DEFENSE_FIT (different exact-"
                     "record groups; a property of Adult's mostly categorical inputs, not a split defect)."),
            "counts": x_dup},
        "label_dtypes_from_npy_headers": npz_label_dtypes(),
        "withheld": ("No label statistic of any role is published. OSF_DEVELOPMENT_ASSESSMENT labels are sealed until "
                     "the pushed EVALUATION_LOCK. No row ids, record keys or group ids are published (hashes only)."),
    }
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("osf", "smf", "rgj"))
    assert not loaded, f"independence violated: {loaded}"
    man["independence"] = {"forbidden_packages": ["osf", "smf", "rgj"], "loaded_during_independent_step": loaded,
                           "imports_used": ["numpy", "pandas (raw frames)", "jcv.data.raw_frames", "pcrl.data.adult"]}
    for k in ("cert_eligibility", "assessment_exclusion_from_eligible_models", "loader_comparison"):
        if k in prev:
            man[k] = prev[k]
    if write:
        write_manifest(man)
    ctx = {"old_role": old_role, "unit": unit, "row_id": row_id, "fn": fn, "X_adm": X, "Xexp": Xexp, "A": A,
           "variants": variants, "ce_main": ce_main}
    return man, ctx


def npz_label_dtypes():
    """dtype of the label arrays from the npy headers only (values are not read)."""
    out = {}
    with zipfile.ZipFile(SRC) as zf:
        for k in LABEL_KEYS:
            with zf.open(k + ".npy") as f:
                ver = np.lib.format.read_magic(f)
                rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dt = rd(f)
                out[k] = {"dtype": str(dt), "signed_integer": bool(np.issubdtype(dt, np.signedinteger)),
                          "shape": list(shape)}
    return out


# ----------------------------------------------------------------------------------------------- custody: CERT
def eligible_units():
    """The eligible current smf models (prompt / lead instruction) by kind."""
    u = {"warm": [f"warm__s{k}" for k in SEEDS],
         "U": [f"tl__s{k}__e40" for k in SEEDS],
         "RAW": [f"raw__s{k}__RAW-{t}__b{b}__e{e}" for k in SEEDS for t in "JL" for b in ("0.1", "0.3") for e in (20, 40)],
         "RAW_run_receipts": [f"run__raw__s{k}__RAW-{t}__b{b}" for k in SEEDS for t in "JL" for b in ("0.1", "0.3")],
         "LEACE": [f"lc__s{k}__E" for k in SEEDS],
         "FARE": [f"fare__s{k}__p{i}__{c}" for k in SEEDS for i in (0, 1) for c in [f"c{j}" for j in FARE_IDS] + ["Z1"]]}
    trees = [f"smf__s{k}__p{i}__{c}" for k in SEEDS for i in (0, 1) for c in [f"c{j}" for j in FARE_IDS] + ["Z1"]]
    return u, trees


def complete_ok(d: Path):
    c = d / "COMPLETE.json"
    if not c.exists():
        return False, None
    files = jload(c)["files"]
    listed = set(files)
    present = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()} - {"COMPLETE.json"}
    ok = all((d / f).exists() and sha_file(d / f) == h for f, h in files.items())
    return bool(ok and listed <= present), sha_file(c)


def fare_row_hashes(X64):
    """oar.fare_official.row_hashes, re-implemented: sorted unique blake2b-8 (little-endian) of float64 rows."""
    out = np.empty(X64.shape[0], dtype=np.uint64)
    for i in range(X64.shape[0]):
        out[i] = int.from_bytes(hashlib.blake2b(X64[i].tobytes(), digest_size=8).digest(), "little")
    return np.unique(out)


def scaler_matches(head, R_fit):
    """Compare a saved deployed head's StandardScaler with the moments of R_fit (sklearn's own estimator, same version);
    a pure moment computation, not a model fit."""
    from sklearn.preprocessing import StandardScaler
    sc = head.steps[0][1]
    ref = StandardScaler().fit(R_fit)
    return {"n_samples_seen": int(np.asarray(sc.n_samples_seen_).max()),
            "n_equals_fit_rows": bool(np.all(np.asarray(sc.n_samples_seen_) == len(R_fit))),
            "mean_bitwise": bool(np.array_equal(sc.mean_, ref.mean_)),
            "var_bitwise": bool(np.array_equal(sc.var_, ref.var_))}


def cert_eligibility():
    """Label-free custody receipts of every eligible smf model; writes 'cert_eligibility' and
    'assessment_exclusion_from_eligible_models'."""
    import joblib
    import torch
    man, ctx = independent(write=True)
    unit, row_id, Xexp = ctx["unit"], ctx["row_id"], ctx["Xexp"]
    A = ctx["A"]
    fit_rows = np.flatnonzero(A["fit"] == "OSF_DEFENSE_FIT")
    hv_rows = np.flatnonzero(A["fit"] == "HEAD_VALIDATION")
    smf_kept = np.flatnonzero(A["smf"] != "")
    cat = np.where(A["fit"] != "", A["fit"], np.where(A["pool"] != "", A["pool"], "EXCLUDED"))
    units, trees = eligible_units()
    rec = {"units": {}, "trees": {}}
    fails = []

    def compose(rid):
        c = cat[np.asarray(rid, dtype=np.int64)]
        return {k: int((c == k).sum()) for k in FIT_ROLES + POOLS + ["EXCLUDED"]}

    # ---- every unit: COMPLETE, release row composition, heads' scaler moments
    for kind, names in units.items():
        for nm in names:
            d = SMF_UNITS / nm
            ok, csha = complete_ok(d)
            r = {"kind": kind, "complete_ok": ok, "complete_json_sha256": csha}
            if not ok:
                fails.append(f"{nm}: COMPLETE.json does not verify")
            if (d / "release.npz").exists():
                z = np.load(d / "release.npz", allow_pickle=False)
                rid = z["row_id"].astype(np.int64)
                r["release_rows_by_category"] = compose(rid)
                r["release_row_ids_equal_smf_kept_rows"] = bool(np.array_equal(rid, smf_kept))
                pos = np.searchsorted(rid, fit_rows)
                assert np.array_equal(rid[pos], fit_rows)
                if kind == "FARE":
                    h = joblib.load(d / "head.joblib")
                    r["head_scaler_vs_OSF_DEFENSE_FIT"] = {"head": scaler_matches(h, z["r"][pos])}
                else:
                    r["head_scaler_vs_OSF_DEFENSE_FIT"] = {f"head_{i}": scaler_matches(joblib.load(d / f"head_{i}.joblib"),
                                                                                       z[f"r{i + 1}"][pos]) for i in (0, 1)}
                for hk, hv in r["head_scaler_vs_OSF_DEFENSE_FIT"].items():
                    if not (hv["n_equals_fit_rows"] and hv["mean_bitwise"] and hv["var_bitwise"]):
                        fails.append(f"{nm}: {hk} scaler moments differ from OSF_DEFENSE_FIT")
                c = r["release_rows_by_category"]
                if any(c[p] for p in ("ORIG_ASSESSMENT", "RGJ_DEV", "CERT", "EXCLUDED")) or not r[
                        "release_row_ids_equal_smf_kept_rows"]:
                    fails.append(f"{nm}: release covers rows outside the smf roles")
            rec["units"][nm] = r

    # sensitivity of the scaler check (one unit): adding the CERT rows to the fit rows changes the moments
    z = np.load(SMF_UNITS / "tl__s0__e40" / "release.npz")
    rid = z["row_id"].astype(np.int64)
    pos = np.searchsorted(rid, fit_rows)
    cert_rows = np.flatnonzero(A["pool"] == "CERT")
    hv_pos = np.searchsorted(rid, hv_rows)
    neg = scaler_matches(joblib.load(SMF_UNITS / "tl__s0__e40" / "head_0.joblib"),
                         np.vstack([z["r1"][pos], z["r1"][hv_pos]]))
    sensitivity = {"what": "head_0 of tl__s0__e40 compared with the moments of OSF_DEFENSE_FIT + HEAD_VALIDATION rows "
                           "(a wrong fit set); CERT rows have no features in any smf release",
                   "matches_wrong_set": bool(neg["mean_bitwise"] and neg["n_equals_fit_rows"]),
                   "cert_rows_with_smf_release_features": int(np.isin(cert_rows, rid).sum())}

    # ---- LEACE: fit-row id hash, fit-feature hash from U's release, n_fit
    leace = {}
    fit_ids_hash_inorder = sha_bytes(np.ascontiguousarray(row_id[fit_rows].astype("<i8")).tobytes())
    for k in SEEDS:
        nm = f"lc__s{k}__E"
        lr = jload(SMF_UNITS / nm / "record.json")
        zu = np.load(SMF_UNITS / lr["source_unit"] / "release.npz")
        p = np.searchsorted(zu["row_id"], fit_rows)
        out = {"source_unit": lr["source_unit"],
               "source_model_pt_sha256_matches_U": lr["source_model_pt_sha256"] == sha_file(
                   SMF_UNITS / f"tl__s{k}__e40" / "model.pt")}
        for i in (0, 1):
            mj = jload(SMF_UNITS / nm / f"leace_{i}" / "leace_map.json")
            H = np.ascontiguousarray(zu[f"r{i + 1}"][p], dtype=np.float64)
            out[f"map_{i}"] = {"n_fit": mj["n_fit"], "n_fit_equals_OSF_DEFENSE_FIT": mj["n_fit"] == len(fit_rows),
                               "fit_row_ids_sha256_equals_OSF_DEFENSE_FIT": mj["fit_row_ids_sha256"] == fit_ids_hash_inorder,
                               "H_fit_sha256_equals_U_features_on_OSF_DEFENSE_FIT": mj["H_fit_sha256"] == sha_bytes(H.tobytes()),
                               "npz_sha256_matches": mj["npz_sha256"] == sha_file(SMF_UNITS / nm / f"leace_{i}" / "leace_map.npz")}
            if not all(v for kk, v in out[f"map_{i}"].items() if kk != "n_fit"):
                fails.append(f"{nm}: LEACE map {i} fit provenance differs from OSF_DEFENSE_FIT")
        if not out["source_model_pt_sha256_matches_U"]:
            fails.append(f"{nm}: source U model hash differs")
        leace[nm] = out

    # ---- FARE trees: n_fit, fit-row fingerprint (ordered), per-row hash set, model fingerprint, encode coverage
    X64 = np.ascontiguousarray(Xexp[fit_rows], dtype=np.float64)
    fit_sha = sha_arrays(X64)
    fit_rh = fare_row_hashes(X64)
    for uid in trees:
        d = SMF_FARE / uid
        ok, csha = complete_ok(d)
        fr = jload(d / "rec.json")
        meta = jload(d / "model" / "model.json")
        mf = jload(d / "model" / "manifest.json")
        rh = np.load(d / "model" / "fit_row_hashes.npy", allow_pickle=False)
        h = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
        h.update(np.asarray(rh, dtype=np.uint64).tobytes())
        cells = np.load(d / "cells.npy", allow_pickle=False)
        t = {"complete_ok": ok, "complete_json_sha256": csha, "n_fit": fr["n_fit"], "n_all": fr["n_all"],
             "n_fit_equals_OSF_DEFENSE_FIT": fr["n_fit"] == len(fit_rows),
             "fit_rows_sha256_equals_OSF_DEFENSE_FIT_inputs": fr["fit_rows_sha256"] == fit_sha,
             "fit_row_hash_set_equals_OSF_DEFENSE_FIT_inputs": bool(np.array_equal(np.sort(rh), fit_rh)),
             "model_fingerprint_consistent": bool(h.hexdigest() == mf["fingerprint"] == fr["model_fingerprint"]),
             "tree_pkl_sha256_matches_manifest": sha_file(d / "model" / "tree.pkl") == mf["tree_pkl_sha256"],
             "encode_rows_equal_smf_kept_rows": fr["n_all"] == len(smf_kept) == len(cells)}
        if not all(v for kk, v in t.items() if kk not in ("n_fit", "n_all", "complete_json_sha256")):
            fails.append(f"{uid}: FARE tree fit provenance check failed")
        rec["trees"][uid] = t
    for nm in units["FARE"]:
        fr = jload(SMF_UNITS / nm / "record.json")["provenance"]
        uid = fr["fare_uid"]
        rec["units"][nm]["fare_tree"] = uid
        rec["units"][nm]["fare_cache_complete_sha256_matches"] = fr["fare_cache_complete_sha256"] == \
            rec["trees"][uid]["complete_json_sha256"]
        z = np.load(SMF_UNITS / nm / "release.npz")
        rec["units"][nm]["release_cells_equal_tree_cells"] = bool(
            np.array_equal(z["cells"], np.load(SMF_FARE / uid / "cells.npy").astype(np.int64)))
        if not (rec["units"][nm]["fare_cache_complete_sha256_matches"] and rec["units"][nm]["release_cells_equal_tree_cells"]):
            fails.append(f"{nm}: FARE unit does not match its tree")

    # ---- RAW: run receipts (recipe facts) and critic head = warm head
    raw_runs = {}
    for nm in units["RAW_run_receipts"]:
        dg = jload(SMF_UNITS / nm / "record.json")["diag"]
        k = int(nm.split("__")[2][1:])
        fin = torch.load(SMF_UNITS / nm / "final.pt", weights_only=False)
        warm = torch.load(SMF_UNITS / f"warm__s{k}" / "warm.pt")
        ch = fin["theta_T_minus_1"]["critic_head"]
        head_eq = all(torch.equal(ch[i][0], warm[f"head.{i}.weight"].float()) and torch.equal(ch[i][1], warm[f"head.{i}.bias"].float())
                      for i in (0, 1))
        e40 = nm.replace("run__raw__", "raw__") + "__e40"
        st40 = torch.load(SMF_UNITS / e40 / "model.pt")
        final_eq = all(torch.equal(fin["theta_T"][q], st40[q]) for q in st40) and set(fin["theta_T"]) == set(st40)
        raw_runs[nm] = {"stage": dg["stage"], "arm": dg["arm"], "beta": dg["beta"], "lr": dg["lr"],
                        "encoder_updates": dg["encoder_updates"], "nonfinite": dg["nonfinite"],
                        "rescued": "rescue" in dg, "critic_online_updates": dg["critic_online_updates"],
                        "critic_head_equals_warm_head": head_eq, "theta_T_equals_e40_model": final_eq}
        if not (head_eq and final_eq and dg["stage"] == "B" and dg["encoder_updates"] == 40 * 61 and not dg["nonfinite"]):
            fails.append(f"{nm}: RAW run receipt differs from the recipe")

    # ---- code evidence at the source pin: loaders drop the old pools; fitting uses DEFENSE_FIT only
    def lock_hash(lock, f):
        d = jload(SMF_PKG / lock)
        return (d.get("code_files") or d.get("locked_code_files") or {}).get(f)
    code_files = ["smf/data.py", "rgj/data.py", "smf/run.py", "smf/baselines.py", "smf/train.py", "rgj/train.py",
                  "rgj/finalize.py", "jcv/train.py", "jcv/finalize.py", "jcv/run.py", "oar/fare_official.py",
                  "stored_model_eval/defenses.py"]
    code = {}
    for f in code_files:
        pin = git_blob_sha256(SOURCE_PIN, f)
        code[f] = {"sha256_at_source_pin": pin,
                   "worktree_equals_source_pin": (WT / f).exists() and sha_file(WT / f) == pin,
                   "DATA_AND_ENGINEERING_LOCK": lock_hash("DATA_AND_ENGINEERING_LOCK.json", f) == pin,
                   "PHASE_B_PROTOCOL_LOCK": lock_hash("PHASE_B_PROTOCOL_LOCK.json", f) == pin,
                   "EVALUATION_LOCK": lock_hash("EVALUATION_LOCK.json", f) == pin}
    smf_src = subprocess.run(["git", "-C", str(WT), "show", f"{SOURCE_PIN}:smf/data.py"], capture_output=True,
                             text=True).stdout
    rgj_src = subprocess.run(["git", "-C", str(WT), "show", f"{SOURCE_PIN}:rgj/data.py"], capture_output=True,
                             text=True).stdout
    textual = {
        "rgj/data.py DROPPED includes cert and assessment": 'DROPPED = ("assessment", "cert", "excluded_exposure", '
                                                           '"excluded_dup")' in rgj_src,
        "smf/data.py keeps only rows of the five smf roles": 'keep = np.flatnonzero(new != "")' in smf_src,
        "smf/data.py docstring: old cert dropped at load": "old assessment, old cert" in smf_src,
    }
    smf_lc = jload(SMF_MANIFEST)["loader_comparison"]
    behaviour = {
        "cert_rows_receive_no_smf_role_in_independent_recomputation": bool(np.all(A["smf"][A["pool"] == "CERT"] == "")),
        "orig_assessment_rows_receive_no_smf_role": bool(np.all(A["smf"][A["pool"] == "ORIG_ASSESSMENT"] == "")),
        "rgj_dev_rows_receive_no_smf_role": bool(np.all(A["smf"][A["pool"] == "RGJ_DEV"] == "")),
        "smf_published_loader_comparison_verdict": smf_lc["verdict"],
        "smf_loader_rows": smf_lc["loader_rows"],
        "smf_kept_rows_recomputed": int(len(smf_kept)),
    }

    # ---- lock timing of the eligible units (smf activity log, private; times only)
    produced = {}
    if SMF_ACTIVITY.exists():
        for line in SMF_ACTIVITY.read_text().splitlines():
            e = json.loads(line)
            if e.get("event") == "unit complete":
                produced[e["unit"]] = e["at"]
    locks = {L: jload(SMF_PKG / f"{L}.json")["written_at"] for L in ("DATA_AND_ENGINEERING_LOCK",
                                                                    "PHASE_A_PROTOCOL_LOCK",
                                                                    "PHASE_B_PROTOCOL_LOCK", "EVALUATION_LOCK")}
    import time as _t
    timing = {}
    for kind, names in units.items():
        ts = [produced[n] for n in names if n in produced]
        mt = [_t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime((SMF_UNITS / n / "COMPLETE.json").stat().st_mtime))
              for n in names]
        last = max(ts + mt)
        timing[kind] = {"units_with_completion_event": len(ts),
                        "first_event": min(ts) if ts else None, "last_event": max(ts) if ts else None,
                        "complete_json_mtime_utc_range": [min(mt), max(mt)],
                        "all_before_EVALUATION_LOCK_written_at": last < locks["EVALUATION_LOCK"]}
    smf_iv = jload(SMF_PKG / "INDEPENDENT_VERIFICATION.json")["checks"]["releases"]["per_unit"]
    verifier = {}
    for kind in ("U", "RAW", "LEACE", "FARE"):
        rs = [smf_iv.get(n) for n in units[kind]]
        verifier[kind] = {"units": len(rs), "replayed_by_smf_verifier": sum(r is not None for r in rs),
                          "PASS": sum(1 for r in rs if r and r.get("status") == "PASS"),
                          "dropped_row_ids_present_total": sum((r or {}).get("dropped_row_ids_present", 0) for r in rs)}

    # ---- configuration banks fixed before any of these fits (selection did not involve the pools)
    oar_lock = jload(WT / "results" / "combined_output_aware_removal_v1" / "EXECUTION_LOCK.json")
    sys.path.insert(0, str(WT))
    from jcv.run import FARE_GRID   # registered grid (jcv imports no osf/smf/rgj module)
    grid_eq = [{k: v for k, v in g.items() if k != "range"} for g in oar_lock["fare"]["grid"]] == [dict(g) for g in FARE_GRID]
    a3 = (WT / "results" / "pcrl_joint_complete_view_method_v1" / "AMENDMENT_A3_2026-10-03.md").read_text()
    selection = {
        "deployed_head_C": "jcv.finalize.HEAD_C on HEAD_VALIDATION log loss (rgj.finalize aliases); no pool row",
        "raw_betas_registered_in_smf_DATA_AND_ENGINEERING_LOCK": jload(SMF_PKG / "DATA_AND_ENGINEERING_LOCK.json")["HP"].get(
            "raw_betas"),
        "fare_grid_equals_oar_EXECUTION_LOCK_grid": grid_eq,
        "oar_EXECUTION_LOCK_commit": "3130c8d4e48e4446854d0744e34b22997d985430 (frozen before any oar fit)",
        "oar_fare_selection_rule_uses": "attacker_val (val) only; certificate config is descriptive",
        "jcv_amendment_A3_states_no_registered_decision_uses_certificates":
            "No registered decision uses certificates" in a3,
        "smf_selection_roles": "INNER_SELECTION (selection_A / selection_B), AUDIT_FIT attackers; smf FARE config "
                               "selection on INNER_SELECTION; none of the four pools",
        "osf_reselects": "this study reselects every configuration on INNER_SELECTION only (prompt section 13)",
    }

    verdict = "ESTABLISHED" if not fails else "NOT_ESTABLISHED"
    adm_x = None
    adm_path = PKG / "ADMISSION.json"
    if adm_path.exists():
        ad = jload(adm_path)
        adm_x = {"file": "ADMISSION.json", "sha256": sha_file(adm_path), "verdict": ad.get("verdict"),
                 "admitted_units": len(ad.get("admitted", {})), "checks_failed": ad.get("checks_failed")}
    kinds = ("warm", "U", "RAW", "LEACE", "FARE")
    man = jload(OUT)
    man["cert_eligibility"] = {
        "question": ("Did any CERT row (old certification pool) ever enter the fitting, head selection, critic fitting "
                     "or selection of the eligible current models: smf warm starts, U (tl e40), RAW beta 0.1/0.3 "
                     "(e20/e40), official LEACE lc__s{k}__E, official FARE trees/units, and models this study fits on "
                     "OSF_DEFENSE_FIT?"),
        "rule": "prompt section 6: fit provenance takes precedence over the pool's historical name",
        "verdict": verdict,
        "CERT_ELIGIBLE": verdict == "ESTABLISHED",
        "failures": fails,
        "evidence": {
            "code_at_source_pin": {"files": code, "text": textual, "behaviour": behaviour},
            "receipts": {
                "units_checked": {k: len(units[k]) for k in units},
                "fare_trees_checked": len(trees),
                "complete_json_ok": sum(1 for r in rec["units"].values() if r["complete_ok"]) +
                                    sum(1 for t in rec["trees"].values() if t["complete_ok"]),
                "release_rows_from_CERT_ORIG_RGJ_DEV_or_exclusions": sum(
                    sum(r["release_rows_by_category"][p] for p in ("CERT", "ORIG_ASSESSMENT", "RGJ_DEV", "EXCLUDED"))
                    for r in rec["units"].values() if "release_rows_by_category" in r),
                "head_scalers_equal_OSF_DEFENSE_FIT_moments": sum(
                    1 for r in rec["units"].values() for h in r.get("head_scaler_vs_OSF_DEFENSE_FIT", {}).values()
                    if h["n_equals_fit_rows"] and h["mean_bitwise"] and h["var_bitwise"]),
                "head_scalers_checked": sum(len(r.get("head_scaler_vs_OSF_DEFENSE_FIT", {})) for r in rec["units"].values()),
                "scaler_check_sensitivity": sensitivity,
                "leace": leace,
                "fare_trees": {"n": len(trees),
                               "all_fit_provenance_ok": all(all(v for kk, v in t.items() if kk not in
                                                                ("n_fit", "n_all", "complete_json_sha256"))
                                                            for t in rec["trees"].values()),
                               "OSF_DEFENSE_FIT_float64_input_fingerprint": fit_sha},
                "raw_run_receipts": raw_runs,
                "per_unit": rec["units"], "per_tree": rec["trees"]},
            "lock_timing": {"locks_written_at": locks, "units": timing},
            "smf_independent_verifier_release_replays": verifier,
            "configuration_banks_and_selection": selection,
            "admission_cross_check": adm_x,
            "encoder_fit_rows_note": ("Encoder/critic fitting rows of warm starts, U and RAW rest on the pinned code "
                                      "(rgj.train.TData = DEFENSE_FIT rows of the smf loader, which never contains a "
                                      "CERT, ORIG_ASSESSMENT or RGJ_DEV row; critics on CRITIC_FIT within it) and on "
                                      "the receipts above (heads, releases, critic head, final state). Receipt-level "
                                      "re-derivation of the encoder weights is the lead's bitwise replay stage "
                                      "(osf.run replay: U and RAW from the admitted warm starts on OSF_DEFENSE_FIT); "
                                      "a warm-start bitwise replay is recommended to close the same gap for warm.pt.")},
        "historical_uses_of_CERT": ("PCRL v2 lineage: part of adult.test; pilot/bench: part of attacker_fit "
                                    "(attackers fitted on it); oar: carved out as the FARE certification set (20% of "
                                    "exposure-cleaned attacker_fit groups, salt oar-cert-v1) and used for descriptive "
                                    "FARE certificates; odx/jcv: FARE certificates (jcv amendment A3: vacuous or "
                                    "unavailable; no registered decision used them); rgj, smf: dropped at load. "
                                    "See EXPOSURE_LEDGER.md. These are historical exposures, not fitting or selection "
                                    "of the eligible models."),
        "scope_note": ("Eligibility here concerns fit/selection provenance of the eligible models only. CERT rows remain "
                       "historically exposed benchmark rows; their inclusion is a reused-benchmark measurement, not "
                       "fresh or independent evidence."),
    }
    man["assessment_exclusion_from_eligible_models"] = {
        "pools": POOLS,
        "fitting_rows_from_pool_by_model_kind": {
            k: {p: 0 for p in POOLS} for k in kinds} if not fails else "SEE cert_eligibility.failures",
        "basis": {"warm": "code at source pin (rgj.train.TData = smf DEFENSE_FIT) + warm head = RAW critic head",
                  "U": "code + heads' scaler moments = OSF_DEFENSE_FIT + release rows",
                  "RAW": "code + run receipts + heads' scaler moments + release rows",
                  "LEACE": "map fit-row id hash and fit-feature hash = OSF_DEFENSE_FIT + heads + release rows",
                  "FARE": "tree n_fit / ordered fit-row fingerprint / fit-row hash set = OSF_DEFENSE_FIT inputs + heads"},
        "rows_encoded_in_smf_releases_by_pool": {
            p: int(np.isin(np.flatnonzero(A["pool"] == p), smf_kept).sum()) for p in POOLS},
        "SMF_DEV_note": ("SMF_DEV rows were encoded in smf releases and scored once after the smf EVALUATION_LOCK "
                         "(scoring, not fitting); their labels were masked during every smf fit and selection."),
        "new_fits_of_this_study": ("Prospective: osf.data masks every OSF_DEVELOPMENT_ASSESSMENT label to -1 at load and "
                                   "its allowlist refuses assessment labels to training, heads, inner audit and "
                                   "selection (loader_comparison); new units must show the same receipts (heads' "
                                   "scaler moments, LEACE/FARE fit hashes) before EVALUATION_LOCK."),
    }
    write_manifest(man)
    print(json.dumps({"verdict": verdict, "failures": fails[:20],
                      "units": {k: len(v) for k, v in units.items()}, "trees": len(trees)}, indent=1))
    return verdict


# ----------------------------------------------------------------------------------------------- compare step
def compare_loader():
    man, ctx = independent(write=True)
    old_role, unit, row_id, fn, Xexp = (ctx[k] for k in ("old_role", "unit", "row_id", "fn", "Xexp"))
    sys.path.insert(0, str(WT))
    import osf.data as O  # noqa: E402  (imported only in this step)
    D = O.load()          # sealed (unseal=False)
    ce_loader = bool(O.CERT_ELIGIBLE)
    V = ctx["variants"][ce_loader]
    role_mine = V["role"]
    keep_mine = np.flatnonzero(role_mine != "")
    out = {"loader": "osf.data.load() with its defaults (verify=True, unseal=False), imported in a separate step after "
                     "the independent computation",
           "osf_data_py_sha256": sha_file(WT / "osf" / "data.py"),
           "loader_CERT_ELIGIBLE": ce_loader,
           "custody_CERT_ELIGIBLE": (man.get("cert_eligibility") or {}).get("CERT_ELIGIBLE"),
           "loader_flag_agrees_with_custody": ce_loader == (man.get("cert_eligibility") or {}).get("CERT_ELIGIBLE")}
    out["loader_rows"] = int(len(D["row_id"]))
    out["loader_rows_equal_independent_kept_rows_in_order"] = bool(np.array_equal(D["row_id"], row_id[keep_mine]))
    exact = {}
    for r in ROLES:
        exact[r] = bool(np.array_equal(np.sort(D["row_id"][D["idx"][r]]), row_id[role_mine == r]))
    sub_mine = ctx["A"]["sub"]
    for r in SUBROLES:
        exact[r] = bool(np.array_equal(np.sort(D["row_id"][D["idx"][r]]), row_id[sub_mine == r]))
    out["row_id_sets_identical"] = exact
    pool_l = {p: bool(np.array_equal(np.sort(D["row_id"][D["pool"] == p]), row_id[V["pool"] == p])) for p in POOLS}
    out["pool_row_id_sets_identical"] = pool_l
    lm = O.manifest(D)
    pub = {**man["roles"], **man["defense_fit_subroles"]} if man["cert_variant_in_roles"] == (
        "with_CERT" if ce_loader else "without_CERT") else None
    if pub is None:
        pub = {r: role_record(np.flatnonzero(role_mine == r), row_id, unit) for r in ROLES}
        pub.update({r: role_record(np.flatnonzero(sub_mine == r), row_id, unit) for r in SUBROLES})
    out["manifest_counts_and_hashes_identical"] = {r: bool(all(lm[r][k] == pub[r][k] for k in ("rows", "groups",
                                                                                              "row_id_sha256")))
                                                   for r in ROLES + SUBROLES}
    bp = lm[ASSESS]["by_pool"]
    out["by_pool_counts_identical"] = {p: bool(bp[p]["rows"] == V["info"][p]["kept_rows"] and
                                              bp[p]["groups"] == V["info"][p]["kept_groups"]) for p in POOLS}
    pa = lm["pool_admission"]
    out["pool_admission_identical"] = {p: bool(pa[p]["nominal_rows"] == V["info"][p]["nominal_rows"] and
                                              pa[p]["excluded_rows_group_overlap"] == V["info"][p]["excluded_rows_group_overlap"])
                                       for p in POOLS}
    out["aliases_point_to_osf_roles"] = {a: bool(np.array_equal(D["idx"][a], D["idx"][b])) for a, b in O.ALIASES.items()}
    out["feature_names_identical"] = [str(f) for f in D["feature_names"]] == fn
    Xe = Xexp[keep_mine]
    out["X_dtype"] = str(D["X"].dtype)
    out["X_all_kept_rows_bitwise_equal_independent"] = bool(D["X"].dtype == np.float32 and np.array_equal(D["X"], Xe))
    out["X_rows_differing"] = int((~np.all(D["X"] == Xe, axis=1)).sum())
    out["X_bitwise_by_role_and_pool"] = {
        **{r: bool(np.array_equal(D["X"][D["role"] == r], Xe[role_mine[keep_mine] == r])) for r in ROLES},
        **{f"{ASSESS}:{p}": bool(np.array_equal(D["X"][D["pool"] == p], Xe[V["pool"][keep_mine] == p])) for p in POOLS}}
    mine_num = man["numeric_refit"]["per_column"]
    out["numeric_refit_mean_sd_identical"] = {c: bool(D["numeric_refit"][c]["mean_new_fit"] == mine_num[c]["mean_osf_fit"]
                                                      and D["numeric_refit"][c]["sd_new_fit"] == mine_num[c]["sd_osf_fit"])
                                              for c in NUMERIC}
    out["numeric_inversion_err_identical_on_kept_rows"] = {
        c: bool(D["numeric_refit"][c]["inversion_max_abs_err"] == mine_num[c]["inversion_max_abs_err_kept_rows"])
        for c in NUMERIC}
    # sealing (membership checks only; no label value or statistic is computed or written)
    a = D["idx"][ASSESS]
    rest = np.setdiff1d(np.arange(len(D["row_id"])), a)
    sealed = {k: bool(np.all(D[k][a] == -1)) for k in LABEL_KEYS}
    sealed["y_dict_masked"] = bool(all(np.all(D["y"][t][a] == -1) for t in D["y"]))
    sealed["loader_flag_sealed"] = bool(D.get("sealed"))
    out["assessment_labels_masked_to_minus_one"] = sealed
    out["no_minus_one_outside_assessment"] = {k: bool(not np.any(D[k][rest] == -1)) for k in LABEL_KEYS}
    # allowlist guard: every (procedure, role) pair, sealed; then with the sealed flag cleared on a shallow copy whose
    # label arrays stay masked (tests the allowlist alone, no label is unsealed)
    procs = list(O.ALLOW)
    probe_roles = ROLES + ["DEFENSE_FIT", "DEVELOPMENT_ASSESSMENT"] + SUBROLES + ["CALIB"]

    def guard(Dx):
        res = {}
        for p in procs:
            res[p] = {}
            for r in probe_roles:
                try:
                    O.labels_for(Dx, p, r)
                    res[p][r] = "allowed"
                except PermissionError as e:
                    res[p][r] = "refused: sealed" if "sealed" in str(e) else "refused: allowlist"
        return res
    g_sealed = guard(D)
    Dflag = dict(D)
    Dflag["sealed"] = False
    g_flag = guard(Dflag)
    refuse_all = all(g_sealed[p][r].startswith("refused") for p in procs for r in (ASSESS, "DEVELOPMENT_ASSESSMENT"))
    only_assessment_proc = [p for p in procs if g_flag[p][ASSESS] == "allowed"]
    out["allowlist_guard"] = {
        "sealed_loader": g_sealed, "allowlist_only (sealed flag cleared; labels still masked)": g_flag,
        "assessment_labels_refused_to_every_procedure_while_sealed": refuse_all,
        "procedures_allowed_assessment_labels_by_allowlist": only_assessment_proc,
        "assessment_labels_refused_to_training_heads_inner_audit_selection_by_allowlist": all(
            g_flag[p][ASSESS].startswith("refused") and g_flag[p]["DEVELOPMENT_ASSESSMENT"].startswith("refused")
            for p in procs if p != "assessment"),
        "unseal_path": "not exercised here (osf.load(unseal=True) would unmask labels; reserved for osf.assess after "
                       "the pushed EVALUATION_LOCK)",
        "observation": ("the guard keys on role names: subroles of OSF_DEFENSE_FIT (CRITIC_FIT, CRITIC_VAL, "
                        "DIAGNOSTIC_CALIB, alias CALIB) are refused to every procedure via labels_for; the training code "
                        "reads them through rgj.train.TData (DEFENSE_FIT tensors), not through labels_for. The "
                        "'selection' procedure is refused OSF_DEFENSE_FIT labels, which the fitting-prior constant of "
                        "the task gates needs; the 'assessment' procedure is refused HEAD_VALIDATION. Both are "
                        "conservative refusals, not leakage; callers needing them must use an allowed procedure.")}
    # the pure role function's cert_eligible=False variant equals the independent without-CERT variant
    r0, s0, p0, i0 = O.assign_roles(old_role, unit, cert_eligible=False)
    out["cert_eligible_False_variant_equals_independent"] = bool(
        np.array_equal(np.asarray(r0).astype(str), ctx["variants"][False]["role"]) and
        np.array_equal(np.asarray(p0).astype(str), ctx["variants"][False]["pool"]))
    out["cert_eligible_False_variant_assessment_rows"] = int((np.asarray(r0) == ASSESS).sum())
    # fitting tensors identical to smf's (the admitted checkpoints saw exactly these inputs)
    import smf.data as S  # noqa: E402
    DS = S.load()
    out["fit_tensor_sha_osf"] = O.fit_tensor_sha(D)
    out["fit_tensor_sha_smf"] = O.fit_tensor_sha(DS)
    out["fit_tensor_identical_to_smf"] = out["fit_tensor_sha_osf"] == out["fit_tensor_sha_smf"]
    common = {r: bool(np.array_equal(D["row_id"][D["idx"][r]], DS["row_id"][DS["idx"][s]]) and
                      np.array_equal(D["X"][D["idx"][r]], DS["X"][DS["idx"][s]]))
              for r, s in (("OSF_DEFENSE_FIT", "NEW_DEFENSE_FIT"), ("HEAD_VALIDATION", "HEAD_VALIDATION"),
                           ("AUDIT_FIT", "AUDIT_FIT"), ("INNER_SELECTION", "INNER_SELECTION"),
                           ("CRITIC_FIT", "CRITIC_FIT"), ("CRITIC_VAL", "CRITIC_VAL"),
                           ("DIAGNOSTIC_CALIB", "CONTROLLER_CALIB"))}
    smf_dev_same = bool(np.array_equal(D["row_id"][D["pool"] == "SMF_DEV"], DS["row_id"][DS["idx"]["NEW_DEVELOPMENT_ASSESSMENT"]])
                        and np.array_equal(D["X"][D["pool"] == "SMF_DEV"], DS["X"][DS["idx"]["NEW_DEVELOPMENT_ASSESSMENT"]]))
    out["rows_and_X_identical_to_smf_loader"] = {**common, "SMF_DEV (= smf NEW_DEVELOPMENT_ASSESSMENT)": smf_dev_same}
    del D, DS, Dflag
    out["note"] = ("Only counts, hashes, bitwise equalities, masking and guard behaviour were compared. No label value or "
                   "label statistic of any role was computed or written.")
    ok = (out["loader_rows_equal_independent_kept_rows_in_order"] and all(exact.values()) and all(pool_l.values())
          and all(out["manifest_counts_and_hashes_identical"].values()) and all(out["by_pool_counts_identical"].values())
          and all(out["pool_admission_identical"].values()) and all(out["aliases_point_to_osf_roles"].values())
          and out["feature_names_identical"] and out["X_all_kept_rows_bitwise_equal_independent"]
          and all(out["numeric_refit_mean_sd_identical"].values())
          and all(out["numeric_inversion_err_identical_on_kept_rows"].values())
          and all(sealed.values()) and all(out["no_minus_one_outside_assessment"].values())
          and refuse_all and only_assessment_proc == ["assessment"]
          and out["cert_eligible_False_variant_equals_independent"] and out["fit_tensor_identical_to_smf"]
          and all(out["rows_and_X_identical_to_smf_loader"].values()))
    out["verdict"] = "MATCH" if ok else "MISMATCH"
    if out["loader_flag_agrees_with_custody"] is False:
        out["verdict"] += " (LOADER CERT_ELIGIBLE DISAGREES WITH CUSTODY DETERMINATION)"
    man = jload(OUT)
    man["loader_comparison"] = out
    write_manifest(man)
    print(json.dumps({k: v for k, v in out.items() if k != "allowlist_guard"}, indent=1))
    return out


if __name__ == "__main__":
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("set OMP_NUM_THREADS=1")
    if "--compare-loader" in sys.argv:
        compare_loader()
    elif "--cert-eligibility" in sys.argv:
        cert_eligibility()
    else:
        m = independent()[0]
        print(json.dumps({"roles": {r: (v["rows"], v["groups"]) for r, v in m["roles"].items()},
                          "by_pool": m["roles"][ASSESS]["by_pool"],
                          "variants": {k: (v["rows"], v["groups"]) for k, v in m["consolidated_assessment_variants"].items()},
                          "subroles": {r: (v["rows"], v["groups"]) for r, v in m["defense_fit_subroles"].items()},
                          "assertions_ok": all(m["assertions"].values()),
                          "numeric_ok": m["numeric_refit"]["all_ok"],
                          "groups_shared_between_pools": m["groups_shared_between_pools"],
                          "x_dup": m["input_duplicate_note"]["counts"]}, indent=1))
