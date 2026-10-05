"""Independent recomputation of the strength-matched feedback study's role partition and numeric refit.

Owner: data and provenance role. Written from the rule text in smf/data.py's docstring and the study prompt (section 5)
only. The independent step does NOT import smf or rgj (or oar). It reads the admitted input file's non-label arrays
(role, unit, row_id, feature_names, X) and never reads sex, race, y_income or y_occupation_group.

Rule (seed 20261005):
  Start from the refreshed (rgj) study's roles, recomputed here from their own rule (seed 20261004):
      rgj DEFENSE_FIT = old defense_train; rgj AUDIT_FIT = old attacker_fit; rgj INNER_SELECTION = old attacker_val;
      old defense_val by group: u = int(sha256("20261004|dev|<unit>")[:16], 16) / 2**64;
      rgj HEAD_VALIDATION iff u < 0.30, else rgj DEVELOPMENT_ASSESSMENT.
  New roles:
      rgj DEFENSE_FIT groups: u = int(sha256("20261005|assess|<unit>")[:16], 16) / 2**64;
          NEW_DEVELOPMENT_ASSESSMENT iff u < 0.20, else NEW_DEFENSE_FIT.
      HEAD_VALIDATION, AUDIT_FIT, INNER_SELECTION = the rgj roles of the same name (unchanged).
      NEW_DEFENSE_FIT subroles by group, salt "critic", seed 20261005:
          CRITIC_FIT u < 0.70; CRITIC_VAL 0.70 <= u < 0.85; CONTROLLER_CALIB u >= 0.85.
  Dropped from every operation: rgj DEVELOPMENT_ASSESSMENT (3,397 rows), old assessment, cert, excluded_exposure,
  excluded_dup.
Numeric refit: the 5 numeric columns are re-standardised (population sd) on NEW_DEFENSE_FIT rows only. Here the raw
integer values are regenerated from the pinned loader (jcv.data.raw_frames(); raw files hash-checked there and again
here), tied to the admitted rows by exact equality of all 83 admitted columns, and the expected refitted columns are
computed from them. The step also re-derives the admitted (old) normalisation and checks the inversion that smf uses.

Usage (from the worktree root, OMP_NUM_THREADS=1):
  ~/PCRL/.venv/bin/python results/pcrl_strength_matched_feedback_v1/provenance/role_check.py
      -> writes results/pcrl_strength_matched_feedback_v1/ROLE_MANIFEST.json (independent sections)
  ~/PCRL/.venv/bin/python results/pcrl_strength_matched_feedback_v1/provenance/role_check.py --compare-loader
      -> separate step: reruns the independent step, then imports smf.data, runs its sealed load(), and records an
         exact comparison (role sets, counts/hashes, X bitwise, numeric refit, label sealing) in ROLE_MANIFEST.json
  ~/PCRL/.venv/bin/python results/pcrl_strength_matched_feedback_v1/provenance/role_check.py --checkpoint-exposure
      -> read-only, label-free: confirms the predecessor FARE trees were fitted on all old defense_train rows (which
         contain NEW_DEVELOPMENT_ASSESSMENT) and counts NEW_DEVELOPMENT_ASSESSMENT rows by their rgj critic subrole;
         recorded in ROLE_MANIFEST.json "historical_checkpoint_exposure"

Publishes counts and hashes only. No row id, record key or label statistic of any role is written.
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
RGJ_MANIFEST = WT / "results" / "pcrl_refreshed_guarded_joint_v1" / "ROLE_MANIFEST.json"
SOURCE_PIN = "ccdcc5a372d538cd12332e496e08f7753f7c9922"
RAW_SHA = {"adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
           "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"}
OUT = PKG / "ROLE_MANIFEST.json"

RGJ_SEED, SEED = 20261004, 20261005
HEAD_SHARE = Fraction(3, 10)
ASSESS_SHARE = Fraction(1, 5)
CRIT_A, CRIT_B = Fraction(7, 10), Fraction(17, 20)
RGJ_KEPT = {"defense_train": "DEFENSE_FIT", "attacker_fit": "AUDIT_FIT", "attacker_val": "INNER_SELECTION"}
OLD_DROPPED = ["assessment", "cert", "excluded_exposure", "excluded_dup"]
NEW_ROLES = ["NEW_DEFENSE_FIT", "NEW_DEVELOPMENT_ASSESSMENT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"]
SUBROLES = ["CRITIC_FIT", "CRITIC_VAL", "CONTROLLER_CALIB"]
RGJ_SUBROLES = ["CRITIC_FIT", "CRITIC_VAL", "CALIB"]
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "relationship", "native-country"]
EXCLUDED_SOURCE = ["sex", "race", "income", "occupation", "fnlwgt", "row id / record key / group id"]
SEALED = "NEW_DEVELOPMENT_ASSESSMENT"

RULE_TEXT = (
    "Start from the refreshed study's roles (seed 20261004; recomputed here from its rule). Its DEFENSE_FIT (= old "
    "defense_train) is split by exact-record group: u = int(sha256('20261005|assess|<unit>')[:16], 16) / 2**64; "
    "NEW_DEVELOPMENT_ASSESSMENT iff u < 0.20, else NEW_DEFENSE_FIT. HEAD_VALIDATION, AUDIT_FIT and INNER_SELECTION are "
    "kept unchanged. NEW_DEFENSE_FIT is split by group with salt 'critic' (seed 20261005): CRITIC_FIT u < 0.70, "
    "CRITIC_VAL 0.70 <= u < 0.85, CONTROLLER_CALIB u >= 0.85; encoders and training heads still train on all "
    "NEW_DEFENSE_FIT rows. The refreshed DEVELOPMENT_ASSESSMENT, old assessment, cert, excluded_exposure and "
    "excluded_dup rows are dropped from every operation. The allocation reads only the old role array and the group "
    "ids; no label is read. The 5 numeric columns are re-standardised (population sd) on NEW_DEFENSE_FIT rows only; "
    "the 78 one-hot columns use the loader's fixed category sets.")


# ----------------------------------------------------------------------------------------------- helpers
def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rowid_hash(r) -> str:
    """Convention shared with jcv/rgj/smf manifests: sha256 of the sorted row ids as int64 raw bytes."""
    return hashlib.sha256(np.ascontiguousarray(np.sort(np.asarray(r)).astype(np.int64)).tobytes()).hexdigest()


def unit_set_hash(u) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.unique(np.asarray(u)).astype(np.int64)).tobytes()).hexdigest()


def u_int(seed: int, salt: str, unit: int) -> int:
    return int(hashlib.sha256(f"{seed}|{salt}|{int(unit)}".encode()).hexdigest()[:16], 16)


def split_groups(units, seed, salt, cuts_exact, cuts_float, names):
    """Map each group to a class by its hash; float rule as written AND exact rational rule. Returns
    (dict group->name, number of float/exact disagreements)."""
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


# ----------------------------------------------------------------------------------------------- partition
def allocate(old_role, unit):
    """Per-row rgj role, rgj subrole, new role, new subrole. Every split is per group, so no group can be split."""
    n = len(old_role)
    rgj = np.array([RGJ_KEPT.get(r, "") for r in old_role.tolist()], dtype=object)
    dv = old_role == "defense_val"
    gm, d1 = split_groups(unit[dv], RGJ_SEED, "dev", [HEAD_SHARE], [0.30],
                          ["HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT"])
    rgj[dv] = [gm[g] for g in unit[dv].tolist()]
    rgj_sub = np.array([""] * n, dtype=object)
    df = rgj == "DEFENSE_FIT"
    sm, d2 = split_groups(unit[df], RGJ_SEED, "critic", [CRIT_A, CRIT_B], [0.70, 0.85], RGJ_SUBROLES)
    rgj_sub[df] = [sm[g] for g in unit[df].tolist()]

    new = np.array([""] * n, dtype=object)
    am, d3 = split_groups(unit[df], SEED, "assess", [ASSESS_SHARE], [0.20], [SEALED, "NEW_DEFENSE_FIT"])
    new[df] = [am[g] for g in unit[df].tolist()]
    for r in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
        new[rgj == r] = r
    sub = np.array([""] * n, dtype=object)
    nf = new == "NEW_DEFENSE_FIT"
    cm, d4 = split_groups(unit[nf], SEED, "critic", [CRIT_A, CRIT_B], [0.70, 0.85], SUBROLES)
    sub[nf] = [cm[g] for g in unit[nf].tolist()]
    disagree = {"rgj_dev_split": d1, "rgj_critic_split": d2, "new_assess_split": d3, "new_critic_split": d4}
    return rgj.astype(str), rgj_sub.astype(str), new.astype(str), sub.astype(str), disagree


# ----------------------------------------------------------------------------------------------- numeric refit
def raw_permitted(n_expected):
    """Raw values of the 10 permitted source columns, row-aligned with adult_jcv.npz (row_id = position in
    concat[test, train]). The pinned loader holds all columns in memory; only the permitted ones are kept."""
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
    loader_sha = sha_file(WT / "pcrl" / "data" / "adult.py")
    jcv_data_sha = sha_file(WT / "jcv" / "data.py")
    return raw, AdultDataset.CATEGORY_VALUES, raw_hashes, loader_sha, jcv_data_sha


def numeric_check(X, fn, raw, cats, old_role, new):
    adm = json.loads(PRED_ADMISSION.read_text())
    R = {}
    for c in NUMERIC:
        v = raw[c].to_numpy()
        assert np.issubdtype(v.dtype, np.integer), f"{c}: raw dtype {v.dtype} is not integer"
        R[c] = v.astype(np.int64)
    # (a) row alignment: every one-hot column equals the raw category indicator, bitwise
    onehot_equal = {}
    for c in CATEGORICAL:
        vals = raw[c].to_numpy().astype(str)
        exp = (vals[:, None] == np.array(cats[c], dtype=str)[None, :]).astype(np.float32)
        cols = [fn.index(f"{c}={v}") for v in cats[c]]
        onehot_equal[c] = bool(np.array_equal(X[:, cols], exp))
    old_fit = old_role == "defense_train"
    out = {"raw_numeric_columns_integer": True, "onehot_columns_equal_raw_indicators": onehot_equal}
    per = {}
    nf = np.flatnonzero(new == "NEW_DEFENSE_FIT")
    keep = np.flatnonzero(new != "")
    expected = {}
    for c in NUMERIC:
        j = fn.index(c)
        r = R[c].astype(np.float64)
        mu_o, sd_o = float(r[old_fit].mean()), float(r[old_fit].std())
        mu_a, sd_a = adm["numeric_norm"][c]
        x_old = ((r - mu_o) / sd_o).astype(np.float32)
        inv = X[:, j].astype(np.float64) * sd_a + mu_a
        rnd = np.round(inv)
        m, s = float(r[nf].mean()), float(r[nf].std())
        expected[c] = ((r[keep] - m) / s).astype(np.float32)
        per[c] = {
            "admitted_norm_rederived_from_raw_on_old_defense_train": bool(mu_o == mu_a and sd_o == sd_a),
            "admitted_column_equals_raw_standardised_bitwise": bool(np.array_equal(x_old, X[:, j])),
            "inversion_max_abs_err": float(np.abs(inv - r).max()),
            "rounded_inversion_equals_raw_integers_all_rows": bool(np.array_equal(rnd, r)),
            "mean_new_fit": m, "sd_new_fit": s,
            "new_fit_rows": int(len(nf)),
            "refit_differs_from_admitted_norm": bool(m != mu_a or s != sd_a)}
    out["per_column"] = per
    out["all_ok"] = bool(all(onehot_equal.values()) and all(
        v["admitted_norm_rederived_from_raw_on_old_defense_train"] and v["admitted_column_equals_raw_standardised_bitwise"]
        and v["rounded_inversion_equals_raw_integers_all_rows"] and v["inversion_max_abs_err"] < 0.05
        for v in per.values()))
    return out, expected


# ----------------------------------------------------------------------------------------------- independent step
def independent():
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
    assert len(np.unique(row_id)) == n and np.array_equal(row_id, np.arange(n)), "row ids not 0..n-1"

    # ---- old roles must match the predecessor admission
    adm = json.loads(PRED_ADMISSION.read_text())
    assert adm["private_inputs_sha256"] == SRC_SHA
    old = {r: role_record(np.flatnonzero(old_role == r), row_id, unit) for r in sorted(set(old_role.tolist()))}
    old_match = {r: bool(adm["roles"].get(r) and all(adm["roles"][r][k] == old[r][k] for k in ("rows", "groups", "row_id_sha256")))
                 for r in old}
    assert set(old) == set(adm["roles"]) and all(old_match.values()), old_match
    x_match = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest() == adm["X_sha256"]
    assert x_match, "X differs from predecessor DATA_ADMISSION"

    # ---- allocation
    rgj, rgj_sub, new, sub, disagree = allocate(old_role, unit)

    # ---- rgj roles recomputed here must equal the published rgj manifest at the source pin
    rgj_man_blob_ok = git("rev-parse", f"{SOURCE_PIN}:results/pcrl_refreshed_guarded_joint_v1/ROLE_MANIFEST.json") == \
        git("hash-object", str(RGJ_MANIFEST))
    rm = json.loads(RGJ_MANIFEST.read_text())
    rgj_pub = {**rm["roles"], **rm["defense_fit_subroles"]}
    rgj_mine = {r: role_record(np.flatnonzero(rgj == r), row_id, unit)
                for r in ["DEFENSE_FIT", "HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT", "AUDIT_FIT", "INNER_SELECTION"]}
    rgj_mine.update({r: role_record(np.flatnonzero(rgj_sub == r), row_id, unit) for r in RGJ_SUBROLES})
    rgj_match = {r: all(rgj_mine[r][k] == rgj_pub[r][k] for k in ("rows", "groups", "row_id_sha256", "group_id_set_sha256"))
                 for r in rgj_mine}
    assert rgj_man_blob_ok and all(rgj_match.values()), (rgj_man_blob_ok, rgj_match)

    keep = new != ""
    idx = {r: np.flatnonzero(new == r) for r in NEW_ROLES}
    idx.update({r: np.flatnonzero(sub == r) for r in SUBROLES})
    roles = {r: role_record(idx[r], row_id, unit) for r in NEW_ROLES + SUBROLES}
    src = {"NEW_DEFENSE_FIT": "refreshed DEFENSE_FIT (= old defense_train), groups with u_assess >= 0.20",
           SEALED: "refreshed DEFENSE_FIT (= old defense_train), groups with u_assess < 0.20",
           "HEAD_VALIDATION": "refreshed HEAD_VALIDATION, unchanged (30% of old defense_val groups)",
           "AUDIT_FIT": "refreshed AUDIT_FIT, unchanged (= old attacker_fit)",
           "INNER_SELECTION": "refreshed INNER_SELECTION, unchanged (= old attacker_val)"}
    use = {"NEW_DEFENSE_FIT": ("encoders, warm starts, training heads, numeric standardisation, official LEACE maps and "
                               "FARE / zero-fairness trees (all retrained or refitted from scratch on these rows)"),
           SEALED: ("withheld from this procedure until EVALUATION_LOCK.json is pushed; labels masked at load; counts "
                    "and row-id hashes only in this manifest (no label statistic)"),
           "HEAD_VALIDATION": "deployed task-head C selection only",
           "AUDIT_FIT": "inner and final attacker fitting",
           "INNER_SELECTION": "candidate evaluation, nomination, attacker and orientation selection",
           "CRITIC_FIT": "critic fitting and controller-probe fitting",
           "CRITIC_VAL": "bounded-refit early stopping / attempt choice; controller-probe selection",
           "CONTROLLER_CALIB": "controller calibration AUC (feedback targets and weight updates)"}
    for r in NEW_ROLES + SUBROLES:
        roles[r]["source"] = src.get(r, "NEW_DEFENSE_FIT (group-hash split, seed 20261005, salt 'critic')")
        roles[r]["permitted_use"] = use[r]
    for r in SUBROLES:
        roles[r]["share_of_NEW_DEFENSE_FIT_rows"] = round(roles[r]["rows"] / roles["NEW_DEFENSE_FIT"]["rows"], 4)

    # ---- assertions
    A = {}
    pairs = [(a, b) for i, a in enumerate(NEW_ROLES) for b in NEW_ROLES[i + 1:]]
    A["new_roles_row_disjoint"] = all(len(np.intersect1d(row_id[idx[a]], row_id[idx[b]])) == 0 for a, b in pairs)
    A["new_roles_group_disjoint"] = all(len(np.intersect1d(unit[idx[a]], unit[idx[b]])) == 0 for a, b in pairs)
    sp = [(a, b) for i, a in enumerate(SUBROLES) for b in SUBROLES[i + 1:]]
    A["subroles_row_and_group_disjoint"] = all(
        len(np.intersect1d(unit[idx[a]], unit[idx[b]])) == 0 and len(np.intersect1d(row_id[idx[a]], row_id[idx[b]])) == 0
        for a, b in sp)
    A["subroles_partition_NEW_DEFENSE_FIT"] = bool(
        np.array_equal(np.sort(np.concatenate([idx[r] for r in SUBROLES])), idx["NEW_DEFENSE_FIT"]))
    A["new_fit_and_new_assessment_partition_refreshed_DEFENSE_FIT_and_old_defense_train"] = bool(
        np.array_equal(np.sort(np.concatenate([idx["NEW_DEFENSE_FIT"], idx[SEALED]])), np.flatnonzero(rgj == "DEFENSE_FIT"))
        and np.array_equal(np.flatnonzero(rgj == "DEFENSE_FIT"), np.flatnonzero(old_role == "defense_train")))
    A["kept_roles_identical_to_refreshed_roles"] = all(
        np.array_equal(idx[r], np.flatnonzero(rgj == r)) for r in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"))
    dropped_lab = np.where(rgj == "DEVELOPMENT_ASSESSMENT", "DROPPED:refreshed_DEVELOPMENT_ASSESSMENT",
                           "DROPPED:old_" + old_role)
    lab = np.where(keep, new, dropped_lab)
    span = {}
    for r, g in zip(lab.tolist(), unit.tolist()):
        span.setdefault(g, set()).add(r)
    multi = {g: s for g, s in span.items() if len(s) > 1}
    kept_multi = [g for g, s in multi.items() if any(not x.startswith("DROPPED:") for x in s)]
    A["no_group_spans_two_new_roles_or_a_new_role_and_a_dropped_pool"] = len(kept_multi) == 0
    A["refreshed_DEVELOPMENT_ASSESSMENT_rows_receive_no_new_role"] = bool(np.all(new[rgj == "DEVELOPMENT_ASSESSMENT"] == ""))
    A["old_dropped_pools_receive_no_new_role"] = bool(np.all(new[np.isin(old_role, OLD_DROPPED)] == ""))
    A["no_new_role_group_occurs_in_any_dropped_pool"] = bool(
        len(np.intersect1d(unit[keep], unit[~keep])) == 0)
    A["dropped_rows_are_exactly_refreshed_assessment_plus_old_dropped_pools"] = bool(
        np.array_equal(np.flatnonzero(~keep),
                       np.flatnonzero((rgj == "DEVELOPMENT_ASSESSMENT") | np.isin(old_role, OLD_DROPPED))))
    A["float_rule_equals_exact_rational_rule_for_every_group"] = all(v == 0 for v in disagree.values())
    A["refreshed_roles_recomputed_equal_published_refreshed_manifest"] = bool(rgj_man_blob_ok and all(rgj_match.values()))
    assert all(A.values()), A

    # ---- realised allocation shares (groups)
    n_df_groups = len(np.unique(unit[rgj == "DEFENSE_FIT"]))
    alloc = {"refreshed_DEFENSE_FIT_groups": n_df_groups,
             "NEW_DEVELOPMENT_ASSESSMENT_group_share": round(roles[SEALED]["groups"] / n_df_groups, 4),
             "NEW_DEVELOPMENT_ASSESSMENT_row_share": round(roles[SEALED]["rows"] / int((rgj == "DEFENSE_FIT").sum()), 4),
             "nominal_share": 0.20}

    # ---- re-draw relative to the refreshed critic subroles (counts only)
    cross = {r: {s: int(((new == r) & (rgj_sub == s)).sum()) for s in RGJ_SUBROLES} for r in ("NEW_DEFENSE_FIT", SEALED)}
    cross_sub = {s: {t: int(((sub == s) & (rgj_sub == t)).sum()) for t in RGJ_SUBROLES} for s in SUBROLES}

    # ---- permitted columns (input-only checks)
    base = [f.split("=")[0] for f in fn]
    per_source = {c: base.count(c) for c in dict.fromkeys(base)}
    forbidden_tokens = ["sex", "race", "income", "occupation", "fnlwgt", "row_id", "record", "unit"]
    forbidden_present = [f for f in fn if f.split("=")[0].strip().lower() in forbidden_tokens]
    onehot = X[:, len(NUMERIC):]
    blocks_ok = bool(np.all(np.isin(onehot, (0.0, 1.0)))) and all(
        np.all(X[:, [i for i, b in enumerate(base) if b == c]].sum(1) == 1) for c in CATEGORICAL)

    # ---- numeric refit, from the pinned loader's raw rows
    raw, cats, raw_hashes, loader_sha, jcv_data_sha = raw_permitted(n)
    num, expected = numeric_check(X, fn, raw, cats, old_role, new)
    assert num["all_ok"], num
    del raw

    # ---- permitted-input duplicates (exact equality of the 83 inputs; invariant under the per-column refit)
    Xb = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
    fit_set = set(Xb[idx["NEW_DEFENSE_FIT"]].tolist())
    x_dup = {r: int(sum(x in fit_set for x in Xb[idx[r]].tolist())) for r in NEW_ROLES if r != "NEW_DEFENSE_FIT"}

    fingerprint = hashlib.sha256("|".join(f"{r}:{roles[r]['row_id_sha256']}" for r in NEW_ROLES + SUBROLES)
                                 .encode()).hexdigest()
    man = {
        "schema": "smf-role-manifest-v1",
        "study": "pcrl_strength_matched_feedback_v1",
        "produced_by": "results/pcrl_strength_matched_feedback_v1/provenance/role_check.py (independent of smf and rgj)",
        "role_check_py_sha256": sha_file(Path(__file__)),
        "all_rows_historically_exposed": True,
        "exposure_note": ("Every row in every role below is a previously used UCI Adult row (EXPOSURE_LEDGER.md). "
                          "NEW_DEVELOPMENT_ASSESSMENT consists of old defense_train rows that trained every predecessor "
                          "encoder, warm start, LEACE map and FARE tree. It is withheld from this procedure only; it is "
                          "not untouched and its results cannot count as fresh confirmation."),
        "source": {"file": SRC_PUBLIC, "sha256": src_sha, "sha256_matches_admitted": True,
                   "arrays_present": files,
                   "arrays_read_by_independent_step": ["role", "unit", "row_id", "feature_names", "X"],
                   "label_arrays_read_by_independent_step": [],
                   "rows_total": n, "groups_total": int(len(np.unique(unit))),
                   "predecessor_admission": "results/pcrl_joint_complete_view_method_v1/DATA_ADMISSION.json",
                   "predecessor_admission_sha256": adm_sha,
                   "X_sha256_matches_predecessor_admission": x_match,
                   "source_pin": SOURCE_PIN,
                   "refreshed_role_manifest_blob_equals_source_pin": rgj_man_blob_ok},
        "rule": {"text": RULE_TEXT, "seed": SEED, "refreshed_seed": RGJ_SEED,
                 "assessment_share": 0.20, "critic_split": [0.70, 0.85],
                 "hash": "sha256 of the UTF-8 string '<seed>|<salt>|<unit as decimal int>', first 16 hex digits / 2**64",
                 "salts": {"refreshed defense_val split (seed 20261004)": "dev",
                           "new assessment split of refreshed DEFENSE_FIT (seed 20261005)": "assess",
                           "NEW_DEFENSE_FIT critic/controller subroles (seed 20261005)": "critic"},
                 "analysis_unit": "exact-record group `unit` (de-duplicated full raw record incl. labels and fnlwgt; "
                                  "Adult has no household identifiers)",
                 "labels_read_to_form_roles": False,
                 "float_vs_exact_rational_disagreements": disagree,
                 "realised_allocation": alloc},
        "old_roles": {"definition": "oar-roles-v1 as admitted by jcv", "per_role": old,
                      "matches_predecessor_DATA_ADMISSION": old_match},
        "refreshed_roles_recomputed": {"per_role": rgj_mine, "matches_published_refreshed_manifest": rgj_match},
        "roles": {r: roles[r] for r in NEW_ROLES},
        "new_defense_fit_subroles": {r: roles[r] for r in SUBROLES},
        "subrole_note": ("The critic/controller subroles are a new draw (seed 20261005) inside NEW_DEFENSE_FIT, not the "
                         "refreshed study's CRITIC_FIT/CRITIC_VAL/CALIB. Encoders and training heads train on all "
                         "NEW_DEFENSE_FIT rows, including the CRITIC_VAL and CONTROLLER_CALIB labels; those subroles are "
                         "held out from critic/controller fitting only."),
        "overlap_with_refreshed_critic_subroles_rows": {"by_new_role": cross, "by_new_subrole": cross_sub},
        "role_partition_fingerprint_sha256": fingerprint,
        "dropped_pools": {
            "pools": ["refreshed DEVELOPMENT_ASSESSMENT"] + [f"old {p}" for p in OLD_DROPPED],
            "rows": {"refreshed DEVELOPMENT_ASSESSMENT": int((rgj == "DEVELOPMENT_ASSESSMENT").sum()),
                     **{f"old {p}": old[p]["rows"] for p in OLD_DROPPED}},
            "row_id_sha256": {"refreshed DEVELOPMENT_ASSESSMENT": rgj_mine["DEVELOPMENT_ASSESSMENT"]["row_id_sha256"],
                              **{f"old {p}": old[p]["row_id_sha256"] for p in OLD_DROPPED}},
            "rows_total": int((~keep).sum()),
            "treatment": ("excluded from every new fitting, selection, inference and reporting call; smf.data.load() "
                          "drops them before any array leaves the loader")},
        "assertions": A,
        "groups_spanning_dropped_pools_only": int(len(multi) - len(kept_multi)),
        "permitted_columns": {
            "n_columns": len(fn), "feature_names_sha256": hashlib.sha256("\n".join(fn).encode()).hexdigest(),
            "source_columns_kept": NUMERIC + CATEGORICAL, "columns_per_source": per_source,
            "excluded_source_columns": EXCLUDED_SOURCE,
            "forbidden_column_names_present": forbidden_present,
            "relationship_Husband_present": "relationship=Husband" in fn,
            "relationship_Wife_present": "relationship=Wife" in fn,
            "proxy_note": ("relationship=Husband / relationship=Wife are kept permitted proxies; in the predecessor's "
                           "shortcut audit (DATA_ADMISSION.json) they determine SEX for about 46% of rows. Deliberately "
                           "not dropped (prompt section 5); see EXPOSURE_LEDGER.md."),
            "categorical_blocks_are_exact_one_hot_all_rows": blocks_ok,
            "unseen_category_rows": 0 if blocks_ok else "see blocks check",
            "categorical_handling": ("fixed loader category sets (pcrl.data.adult.AdultDataset.CATEGORY_VALUES); no "
                                     "fitted vocabulary; an unseen value would give an all-zero block (none occurs)")},
        "numeric_refit": {
            "rule": ("raw integer values recovered from the admitted normalisation and re-standardised with population "
                     "mean/sd computed on NEW_DEFENSE_FIT rows only; the transform is then applied to all kept rows"),
            "raw_source": ("jcv.data.raw_frames() -> pinned pcrl.data.adult loader on hash-checked adult.data / "
                           "adult.test; row alignment proven by bitwise equality of all 78 one-hot columns and of the 5 "
                           "admitted numeric columns re-derived from the raw values"),
            "raw_files_sha256": raw_hashes, "pcrl_data_adult_py_sha256": loader_sha, "jcv_data_py_sha256": jcv_data_sha,
            **num},
        "input_duplicate_note": {
            "what": ("Groups are exact full-record duplicates. Rows identical on the 83 permitted inputs but differing in "
                     "a removed column are different groups and can sit in different roles. Rows per role whose input "
                     "vector also occurs in NEW_DEFENSE_FIT:"),
            "counts": x_dup},
        "withheld": ("No label statistic of any role is published here. NEW_DEVELOPMENT_ASSESSMENT labels are sealed "
                     "until EVALUATION_LOCK.json. No row ids, record keys or group ids are published (hashes only)."),
    }
    if OUT.exists():
        prev = json.loads(OUT.read_text())
        for k in ("loader_comparison", "historical_checkpoint_exposure"):
            if k in prev:
                man[k] = prev[k]
    OUT.write_text(json.dumps(man, indent=1) + "\n")
    mine = {r: np.sort(row_id[idx[r]]) for r in NEW_ROLES + SUBROLES}
    return man, mine, expected, X, fn, keep


# ----------------------------------------------------------------------------------------------- compare step
def npz_label_dtypes():
    """dtype/shape of label arrays from the npy headers only (values are not read)."""
    out = {}
    with zipfile.ZipFile(SRC) as zf:
        for k in ("sex", "race", "y_income", "y_occupation_group"):
            with zf.open(k + ".npy") as f:
                ver = np.lib.format.read_magic(f)
                rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dt = rd(f)
                out[k] = {"dtype": str(dt), "signed_integer": bool(np.issubdtype(dt, np.signedinteger))}
    return out


def compare_loader():
    man, mine, expected, X_adm, fn, keep = independent()
    sys.path.insert(0, str(WT))
    import smf.data as S  # noqa: E402  (imported only in this step)
    D = S.load()          # sealed (unseal=False) - assessment labels are masked by the loader
    roles_all = list(S.ROLES) + list(S.SUBROLES)
    assert set(roles_all) == set(NEW_ROLES + SUBROLES), roles_all
    theirs = {r: np.sort(D["row_id"][D["idx"][r]]) for r in roles_all}
    exact = {r: bool(np.array_equal(mine[r], theirs[r])) for r in NEW_ROLES + SUBROLES}
    lm = S.manifest(D)
    pub = {**man["roles"], **man["new_defense_fit_subroles"]}
    hashes = {r: bool(all(lm[r][k] == pub[r][k] for k in ("rows", "groups", "row_id_sha256"))) for r in NEW_ROLES + SUBROLES}
    kept_rows = int(len(D["row_id"]))
    keep_ix = np.flatnonzero(keep)
    rows_in_order = bool(np.array_equal(D["row_id"], keep_ix))
    aliases = {a: bool(np.array_equal(D["idx"][a], D["idx"][b])) for a, b in S.ALIASES.items()}
    # X: one-hot columns unchanged from the admitted file; numeric columns equal the independently expected refit
    onehot_cols = [i for i, f in enumerate(fn) if f not in NUMERIC]
    x_onehot = bool(np.array_equal(D["X"][:, onehot_cols], X_adm[keep_ix][:, onehot_cols]))
    x_num = {c: bool(D["X"].dtype == np.float32 and np.array_equal(D["X"][:, fn.index(c)], expected[c])) for c in NUMERIC}
    x_num_maxdiff = {c: float(np.abs(D["X"][:, fn.index(c)].astype(np.float64) - expected[c].astype(np.float64)).max())
                     for c in NUMERIC}
    mine_num = man["numeric_refit"]["per_column"]
    refit_stats = {c: bool(D["numeric_refit"][c]["mean_new_fit"] == mine_num[c]["mean_new_fit"]
                           and D["numeric_refit"][c]["sd_new_fit"] == mine_num[c]["sd_new_fit"]
                           and D["numeric_refit"][c]["inversion_max_abs_err"] == mine_num[c]["inversion_max_abs_err"])
                   for c in NUMERIC}
    feature_names_same = [str(f) for f in D["feature_names"]] == fn
    # sealing: every label array is -1 on all NEW_DEVELOPMENT_ASSESSMENT rows (checks masking only; no statistic)
    a = D["idx"][SEALED]
    sealed = {k: bool(np.all(D[k][a] == -1)) for k in ("sex", "race", "y_income", "y_occupation_group")}
    sealed["y_dict_masked"] = bool(all(np.all(D["y"][t][a] == -1) for t in D["y"]))
    sealed["loader_flag_sealed"] = bool(D.get("sealed"))
    # non-assessment rows carry real labels (no -1 outside the sealed role); a membership check, no statistic
    unsealed_ok = {k: bool(not np.any(D[k][np.setdiff1d(np.arange(kept_rows), a)] == -1))
                   for k in ("sex", "race", "y_income", "y_occupation_group")}
    del D
    man["loader_comparison"] = {
        "loader": "smf.data.load() with its defaults (verify=True, unseal=False), imported in a separate step after the "
                  "independent computation",
        "smf_data_py_sha256": sha_file(WT / "smf" / "data.py"),
        "row_id_sets_identical": exact,
        "smf_manifest_counts_and_hashes_identical": hashes,
        "loader_keeps_only_the_five_roles": kept_rows == sum(man["roles"][r]["rows"] for r in NEW_ROLES),
        "loader_rows": kept_rows,
        "loader_rows_in_ascending_admitted_order": rows_in_order,
        "aliases_point_to_new_roles": aliases,
        "feature_names_identical": feature_names_same,
        "X_onehot_columns_unchanged_bitwise": x_onehot,
        "X_numeric_columns_equal_independent_refit_bitwise": x_num,
        "X_numeric_max_abs_diff": x_num_maxdiff,
        "numeric_refit_mean_sd_and_inversion_error_identical": refit_stats,
        "label_dtypes_from_npy_headers": npz_label_dtypes(),
        "assessment_labels_masked_to_minus_one": sealed,
        "no_minus_one_outside_assessment": unsealed_ok,
        "note": ("Only counts, hashes, bitwise equalities and masking were compared. No label value or label statistic "
                 "of any role was computed or written."),
    }
    ok = (all(exact.values()) and all(hashes.values()) and rows_in_order and all(aliases.values()) and feature_names_same
          and x_onehot and all(x_num.values()) and all(refit_stats.values()) and all(sealed.values())
          and all(unsealed_ok.values()) and man["loader_comparison"]["loader_keeps_only_the_five_roles"])
    man["loader_comparison"]["verdict"] = "MATCH" if ok else "MISMATCH"
    OUT.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man["loader_comparison"], indent=1))


# ----------------------------------------------------------------------------------------------- checkpoint exposure
def _sha_arrays(*arrays) -> str:
    """Fingerprint convention of oar.fare_official._sha256_arrays (dtype + shape + bytes), re-implemented."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def checkpoint_exposure():
    """Label-free, read-only: (1) predecessor jcv FARE trees fitted on all old defense_train inputs; (2) how many
    NEW_DEVELOPMENT_ASSESSMENT rows sat in each refreshed critic subrole; (3) unit counts in predecessor stores."""
    z = np.load(SRC, allow_pickle=False)
    old_role, unit, X = z["role"].astype(str), z["unit"].astype(np.int64), z["X"]
    del z
    rgj, rgj_sub, new, sub, _ = allocate(old_role, unit)
    tr_old = np.flatnonzero(old_role == "defense_train")
    fit_sha = _sha_arrays(np.ascontiguousarray(X[tr_old], dtype=np.float64))
    cache = PRIVATE_CACHE / "jcv_v1" / "fare_cache"
    trees = sorted(p for p in cache.iterdir() if (p / "rec.json").exists())
    fare = {"trees": len(trees), "n_fit_19230": 0, "fit_rows_sha256_equals_old_defense_train_inputs": 0}
    for p in trees:
        rec = json.loads((p / "rec.json").read_text())
        fare["n_fit_19230"] += int(rec["n_fit"] == 19230)
        fare["fit_rows_sha256_equals_old_defense_train_inputs"] += int(rec["fit_rows_sha256"] == fit_sha)
    a = new == SEALED
    units = {}
    for study in ("jcv_v1", "pnx_v1", "rgj_v1"):
        d = PRIVATE_CACHE / study / "run" / "units"
        if d.exists():
            kinds = {}
            for q in d.iterdir():
                k = q.name.split("__")[0]
                kinds[k] = kinds.get(k, 0) + 1
            units[study] = dict(sorted(kinds.items()))
    out = {
        "method": "read-only; label-free; predecessor FARE tree-cache fingerprints vs adult_jcv.npz inputs",
        "NEW_DEVELOPMENT_ASSESSMENT_rows_inside_old_defense_train": int(np.isin(np.flatnonzero(a), tr_old).sum()),
        "NEW_DEVELOPMENT_ASSESSMENT_rows_by_refreshed_subrole": {s: int((a & (rgj_sub == s)).sum()) for s in RGJ_SUBROLES},
        "jcv_fare_trees": fare,
        "predecessor_unit_directories_by_kind": units,
        "reading": ("Every jcv FARE tree (reused by pnx and rgj) was fitted on all 19,230 old defense_train rows, so on "
                    "every NEW_DEVELOPMENT_ASSESSMENT row. The refreshed study trained its critics on, early-stopped on, "
                    "or calibrated its multipliers on these rows according to the subrole counts above."),
    }
    man = json.loads(OUT.read_text())
    man["historical_checkpoint_exposure"] = out
    OUT.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("set OMP_NUM_THREADS=1")
    if "--compare-loader" in sys.argv:
        compare_loader()
    elif "--checkpoint-exposure" in sys.argv:
        checkpoint_exposure()
    else:
        m = independent()[0]
        print(json.dumps({"roles": {r: (v["rows"], v["groups"]) for r, v in m["roles"].items()},
                          "subroles": {r: (v["rows"], v["groups"]) for r, v in m["new_defense_fit_subroles"].items()},
                          "assertions": m["assertions"], "alloc": m["rule"]["realised_allocation"],
                          "disagree": m["rule"]["float_vs_exact_rational_disagreements"],
                          "numeric_ok": m["numeric_refit"]["all_ok"],
                          "x_dup": m["input_duplicate_note"]["counts"]}, indent=1))
