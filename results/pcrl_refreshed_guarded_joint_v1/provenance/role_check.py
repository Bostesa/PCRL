"""Independent recomputation of the refreshed guarded joint study's role partition (data/provenance owner).

Written from the rule text only; it does NOT import rgj.data (or jcv/oar). It reads the admitted input file's
non-label arrays (role, unit, row_id, feature_names, X) and never touches sex, race, income or occupation_group.

Rule (as registered in rgj/data.py docstring, seed 20261004):
  DEFENSE_FIT = old defense_train; AUDIT_FIT = old attacker_fit; INNER_SELECTION = old attacker_val.
  Old defense_val is split by exact-record group `unit`:
      u = int(sha256("20261004|dev|<unit>")[:16], 16) / 2**64 ; HEAD_VALIDATION iff u < 0.30 else DEVELOPMENT_ASSESSMENT
  DEFENSE_FIT subroles by group, salt "critic": CRITIC_FIT u < 0.70, CRITIC_VAL 0.70 <= u < 0.85, CALIB u >= 0.85.
  Old assessment, cert, excluded_exposure, excluded_dup are dropped.

Usage (from the worktree root):
  OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/pcrl_refreshed_guarded_joint_v1/provenance/role_check.py
      -> writes results/pcrl_refreshed_guarded_joint_v1/ROLE_MANIFEST.json (independent section)
  OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/pcrl_refreshed_guarded_joint_v1/provenance/role_check.py --compare-loader
      -> separate step: imports rgj.data, runs its load(), and records an exact comparison in ROLE_MANIFEST.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
WT = PKG.parents[1]
SRC = Path.home() / "PCRL_eval_cache_private" / "jcv_v1" / "inputs" / "adult_jcv.npz"
SRC_PUBLIC = "~/PCRL_eval_cache_private/jcv_v1/inputs/adult_jcv.npz"
SRC_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
PRED_ADMISSION = WT / "results" / "pcrl_joint_complete_view_method_v1" / "DATA_ADMISSION.json"
OUT = PKG / "ROLE_MANIFEST.json"

SEED = 20261004
HEAD_SHARE = Fraction(3, 10)
CRIT_A, CRIT_B = Fraction(7, 10), Fraction(17, 20)
KEPT_OLD = {"defense_train": "DEFENSE_FIT", "attacker_fit": "AUDIT_FIT", "attacker_val": "INNER_SELECTION"}
SPLIT_OLD = "defense_val"
DROPPED = ["assessment", "cert", "excluded_exposure", "excluded_dup"]
NEW_ROLES = ["DEFENSE_FIT", "HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT", "AUDIT_FIT", "INNER_SELECTION"]
SUBROLES = ["CRITIC_FIT", "CRITIC_VAL", "CALIB"]
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
SOURCE_COLUMNS_KEPT = NUMERIC + ["workclass", "education", "marital-status", "relationship", "native-country"]
EXCLUDED_SOURCE = ["sex", "race", "income", "occupation", "fnlwgt", "row id / record key"]

RULE_TEXT = (
    "DEFENSE_FIT = old defense_train; AUDIT_FIT = old attacker_fit; INNER_SELECTION = old attacker_val. "
    "Old defense_val is allocated by exact-record group (unit): u = int(sha256('20261004|dev|<unit>')[:16], 16) / 2**64; "
    "HEAD_VALIDATION iff u < 0.30, else DEVELOPMENT_ASSESSMENT. DEFENSE_FIT is split by group with the same hash and "
    "salt 'critic': CRITIC_FIT u < 0.70, CRITIC_VAL 0.70 <= u < 0.85, CALIB u >= 0.85 (encoders still train on all of "
    "DEFENSE_FIT). Old assessment, cert, excluded_exposure and excluded_dup rows are dropped at load. The allocation "
    "reads only the old role array and the group ids; no label is read.")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rowid_hash(r) -> str:
    """Same convention as the predecessor DATA_ADMISSION: sha256 of sorted int64 row ids (raw bytes)."""
    return hashlib.sha256(np.ascontiguousarray(np.sort(np.asarray(r)).astype(np.int64)).tobytes()).hexdigest()


def unit_set_hash(u) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.unique(np.asarray(u)).astype(np.int64)).tobytes()).hexdigest()


def u_int(salt: str, unit: int) -> int:
    return int(hashlib.sha256(f"{SEED}|{salt}|{int(unit)}".encode()).hexdigest()[:16], 16)


def allocate(old_role, unit):
    """Per-row new role and subrole, computed per GROUP (so a group can never be split by construction), with both
    the float rule as written and an exact rational check of every threshold comparison."""
    new = np.array([KEPT_OLD.get(r, "") for r in old_role.tolist()], dtype=object)
    sub = np.array([""] * len(old_role), dtype=object)
    float_vs_exact_disagreements = 0
    dv_units = np.unique(unit[old_role == SPLIT_OLD])
    gmap = {}
    for g in dv_units.tolist():
        n = u_int("dev", g)
        uf = n / 2.0 ** 64
        head_f = uf < 0.30
        head_x = Fraction(n, 2 ** 64) < HEAD_SHARE
        float_vs_exact_disagreements += int(head_f != head_x)
        gmap[g] = "HEAD_VALIDATION" if head_f else "DEVELOPMENT_ASSESSMENT"
    m = old_role == SPLIT_OLD
    new[m] = [gmap[g] for g in unit[m].tolist()]
    df = new == "DEFENSE_FIT"
    smap = {}
    for g in np.unique(unit[df]).tolist():
        n = u_int("critic", g)
        uf = n / 2.0 ** 64
        sf = "CRITIC_FIT" if uf < 0.70 else ("CRITIC_VAL" if uf < 0.85 else "CALIB")
        ux = Fraction(n, 2 ** 64)
        sx = "CRITIC_FIT" if ux < CRIT_A else ("CRITIC_VAL" if ux < CRIT_B else "CALIB")
        float_vs_exact_disagreements += int(sf != sx)
        smap[g] = sf
    sub[df] = [smap[g] for g in unit[df].tolist()]
    return new.astype(str), sub.astype(str), float_vs_exact_disagreements


def role_record(ix, row_id, unit):
    return {"rows": int(len(ix)), "groups": int(len(np.unique(unit[ix]))), "row_id_sha256": rowid_hash(row_id[ix]),
            "group_id_set_sha256": unit_set_hash(unit[ix])}


def independent():
    src_sha = sha_file(SRC)
    assert src_sha == SRC_SHA, "REFUSED: adult_jcv.npz hash mismatch"
    z = np.load(SRC, allow_pickle=False)
    files = list(z.files)
    old_role, unit, row_id = z["role"].astype(str), z["unit"].astype(np.int64), z["row_id"].astype(np.int64)
    fn = [str(f) for f in z["feature_names"]]
    X = z["X"]
    n = len(old_role)
    assert len(np.unique(row_id)) == n, "row ids not unique"

    # ---- predecessor role manifest (DATA_ADMISSION.json) must match the file before anything else is trusted
    adm = json.loads(PRED_ADMISSION.read_text())
    assert adm["private_inputs_sha256"] == SRC_SHA
    old = {}
    old_match = {}
    for r in sorted(set(old_role.tolist())):
        ix = np.flatnonzero(old_role == r)
        rec = role_record(ix, row_id, unit)
        old[r] = rec
        a = adm["roles"].get(r)
        old_match[r] = bool(a and a["rows"] == rec["rows"] and a["groups"] == rec["groups"]
                            and a["row_id_sha256"] == rec["row_id_sha256"])
    assert set(old) == set(adm["roles"]), "old role set differs from the predecessor admission"
    assert all(old_match.values()), f"old roles differ from predecessor DATA_ADMISSION: {old_match}"
    xsha = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
    x_match = xsha == adm["X_sha256"]
    assert x_match, "X hash differs from predecessor DATA_ADMISSION"

    # ---- old-role group structure (before reallocation)
    old_span = {}
    for r, g in zip(old_role.tolist(), unit.tolist()):
        old_span.setdefault(g, set()).add(r)
    old_multi = {g: s for g, s in old_span.items() if len(s) > 1}
    old_multi_patterns = {}
    for s in old_multi.values():
        k = "+".join(sorted(s))
        old_multi_patterns[k] = old_multi_patterns.get(k, 0) + 1

    # ---- new allocation
    new, sub, disagree = allocate(old_role, unit)
    keep = new != ""
    assert set(np.unique(new[keep]).tolist()) == set(NEW_ROLES)
    assert set(old_role[~keep].tolist()) <= set(DROPPED) and not (set(old_role[keep].tolist()) & set(DROPPED))
    assert int(keep.sum()) + sum(old[r]["rows"] for r in DROPPED if r in old) == n

    idx = {r: np.flatnonzero(new == r) for r in NEW_ROLES}
    idx.update({r: np.flatnonzero(sub == r) for r in SUBROLES})
    roles = {r: role_record(idx[r], row_id, unit) for r in NEW_ROLES + SUBROLES}
    for r in NEW_ROLES:
        roles[r]["source"] = {v: k for k, v in KEPT_OLD.items()}.get(r, "old defense_val (group-hash split)")
    for r in SUBROLES:
        roles[r]["source"] = "DEFENSE_FIT (group-hash split, salt 'critic')"
    roles["HEAD_VALIDATION"]["permitted_use"] = "task-head C selection only"
    roles["DEVELOPMENT_ASSESSMENT"]["permitted_use"] = ("sealed until EVALUATION_LOCK.json is pushed; counts and row-id "
                                                        "hashes only in this manifest (no label statistic)")
    roles["DEFENSE_FIT"]["permitted_use"] = "encoders, training heads, critics, LEACE maps, FARE trees, input preprocessing"
    roles["AUDIT_FIT"]["permitted_use"] = "inner and final attacker fitting"
    roles["INNER_SELECTION"]["permitted_use"] = "candidate evaluation, nomination, attacker selection"
    roles["CRITIC_FIT"]["permitted_use"] = "critic fitting"
    roles["CRITIC_VAL"]["permitted_use"] = "bounded-refit early stopping and attempt choice"
    roles["CALIB"]["permitted_use"] = "constraint / multiplier calibration rows"

    # ---- assertions
    A = {}
    pairs = [(a, b) for i, a in enumerate(NEW_ROLES) for b in NEW_ROLES[i + 1:]]
    A["new_roles_row_disjoint"] = all(len(np.intersect1d(row_id[idx[a]], row_id[idx[b]])) == 0 for a, b in pairs)
    A["new_roles_group_disjoint"] = all(len(np.intersect1d(unit[idx[a]], unit[idx[b]])) == 0 for a, b in pairs)
    sp = [(a, b) for i, a in enumerate(SUBROLES) for b in SUBROLES[i + 1:]]
    A["subroles_row_and_group_disjoint"] = all(
        len(np.intersect1d(unit[idx[a]], unit[idx[b]])) == 0 and len(np.intersect1d(row_id[idx[a]], row_id[idx[b]])) == 0
        for a, b in sp)
    A["subroles_partition_DEFENSE_FIT"] = bool(
        np.array_equal(np.sort(np.concatenate([idx[r] for r in SUBROLES])), idx["DEFENSE_FIT"]))
    A["defense_val_fully_allocated"] = bool(
        np.array_equal(np.sort(np.concatenate([idx["HEAD_VALIDATION"], idx["DEVELOPMENT_ASSESSMENT"]])),
                       np.flatnonzero(old_role == SPLIT_OLD)))
    lab = np.where(keep, new, "DROPPED:" + old_role)
    span = {}
    for r, g in zip(lab.tolist(), unit.tolist()):
        span.setdefault(g, set()).add(r)
    multi = {g: s for g, s in span.items() if len(s) > 1}
    kept_multi = [g for g, s in multi.items() if any(not x.startswith("DROPPED:") for x in s)]
    A["no_group_spans_two_new_roles_or_a_new_role_and_a_dropped_pool"] = len(kept_multi) == 0
    A["dropped_rows_receive_no_new_role"] = bool(np.all(new[np.isin(old_role, DROPPED)] == ""))
    A["float_rule_equals_exact_rational_rule_for_every_group"] = disagree == 0
    assert all(A.values()), A

    # ---- permitted columns (input-only checks; no label read)
    base = [f.split("=")[0] for f in fn]
    per_source = {c: base.count(c) for c in dict.fromkeys(base)}
    forbidden_tokens = ["sex", "race", "income", "occupation", "fnlwgt", "row_id", "record", "unit"]
    forbidden_present = [f for f in fn if f.split("=")[0].strip().lower() in forbidden_tokens]
    df_ix = idx["DEFENSE_FIT"]
    Xn = X[:, [fn.index(c) for c in NUMERIC]].astype(np.float64)
    std_check = {c: {"defense_fit_mean": round(float(Xn[df_ix, j].mean()), 6),
                     "defense_fit_std": round(float(Xn[df_ix, j].std()), 6)} for j, c in enumerate(NUMERIC)}
    std_ok = all(abs(v["defense_fit_mean"]) < 1e-4 and abs(v["defense_fit_std"] - 1) < 1e-4 for v in std_check.values())
    onehot = X[:, len(NUMERIC):]
    onehot_ok = bool(np.all(np.isin(onehot, (0.0, 1.0))))
    blocks_ok = all(int(X[:, [i for i, b in enumerate(base) if b == c]].sum(1).max()) == 1
                    and int(X[:, [i for i, b in enumerate(base) if b == c]].sum(1).min()) == 1
                    for c in SOURCE_COLUMNS_KEPT[len(NUMERIC):])
    # permitted-input duplicates across roles (exact X equality; X is an input, not a label)
    Xb = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()
    df_set = set(Xb[df_ix].tolist())
    x_dup = {r: {"rows_whose_X_also_occurs_in_DEFENSE_FIT": int(sum(x in df_set for x in Xb[idx[r]].tolist()))}
             for r in NEW_ROLES if r != "DEFENSE_FIT"}

    man = {
        "schema": "rgj-role-manifest-v1",
        "study": "pcrl_refreshed_guarded_joint_v1",
        "produced_by": "results/pcrl_refreshed_guarded_joint_v1/provenance/role_check.py (independent of rgj.data)",
        "all_rows_historically_exposed": True,
        "exposure_note": ("Every row in every role below has been used by earlier studies (see EXPOSURE_LEDGER.md). "
                          "DEVELOPMENT_ASSESSMENT is a new development partition, not fresh confirmation."),
        "source": {"file": SRC_PUBLIC, "sha256": src_sha, "sha256_matches_admitted": src_sha == SRC_SHA,
                   "arrays_present": files,
                   "arrays_read_by_this_check": ["role", "unit", "row_id", "feature_names", "X"],
                   "label_arrays_read_by_this_check": [],
                   "rows_total": n, "groups_total": int(len(np.unique(unit))),
                   "predecessor_admission": "results/pcrl_joint_complete_view_method_v1/DATA_ADMISSION.json",
                   "X_sha256_matches_predecessor_admission": x_match},
        "rule": {"text": RULE_TEXT, "seed": SEED, "head_share": 0.30, "critic_split": [0.70, 0.85],
                 "hash": "sha256 of the UTF-8 string '<seed>|<salt>|<unit as decimal int>', first 16 hex digits / 2**64",
                 "salts": {"defense_val split": "dev", "DEFENSE_FIT critic subroles": "critic"},
                 "analysis_unit": "exact-record group `unit` (de-duplicated full raw record incl. labels and fnlwgt; "
                                  "Adult has no household identifiers)",
                 "labels_read_to_form_roles": False},
        "old_roles": {"definition": "oar-roles-v1 as admitted by jcv (defense_val = oar head holdout)",
                      "per_role": old, "matches_predecessor_DATA_ADMISSION": old_match,
                      "groups_spanning_old_roles": old_multi_patterns},
        "roles": {r: roles[r] for r in NEW_ROLES},
        "defense_fit_subroles": {r: roles[r] for r in SUBROLES},
        "dropped_pools": {"pools": DROPPED,
                          "rows": {r: old[r]["rows"] for r in DROPPED if r in old},
                          "rows_total": int((~keep).sum()),
                          "treatment": ("excluded from every new fitting, selection, inference and reporting call; "
                                        "their labels are not read and no new prediction is produced for them")},
        "assertions": A,
        "groups_spanning_dropped_pools_only": int(len(multi) - len(kept_multi)),
        "permitted_columns": {
            "n_columns": len(fn), "source_columns_kept": SOURCE_COLUMNS_KEPT, "columns_per_source": per_source,
            "excluded_source_columns": EXCLUDED_SOURCE,
            "forbidden_column_names_present": forbidden_present,
            "relationship_Husband_present": "relationship=Husband" in fn,
            "relationship_Wife_present": "relationship=Wife" in fn,
            "proxy_note": ("relationship=Husband / relationship=Wife are kept permitted proxies: in the predecessor's "
                           "shortcut audit (DATA_ADMISSION.json) they determine SEX for about 46% of rows. They are "
                           "deliberately not dropped; see EXPOSURE_LEDGER.md."),
            "numeric_standardisation_fit_on_DEFENSE_FIT": std_ok, "numeric_check": std_check,
            "categorical_blocks_are_exact_one_hot": bool(onehot_ok and blocks_ok)},
        "input_duplicate_note": {
            "what": ("Groups are exact full-record duplicates. Rows that are identical on the 83 permitted inputs but "
                     "differ in a removed column (labels, fnlwgt, sex, race) are different groups and can sit in "
                     "different roles. Count of rows per role whose input vector also occurs in DEFENSE_FIT:"),
            "counts": x_dup},
        "withheld": ("No label statistic of any role is published here; DEVELOPMENT_ASSESSMENT labels are sealed "
                     "until EVALUATION_LOCK.json. No record keys or row ids are published (hashes only)."),
    }
    if OUT.exists():
        prev = json.loads(OUT.read_text())
        if "loader_comparison" in prev:
            man["loader_comparison"] = prev["loader_comparison"]
    OUT.write_text(json.dumps(man, indent=1) + "\n")
    return man, {r: np.sort(row_id[idx[r]]) for r in NEW_ROLES + SUBROLES}


def compare_loader():
    """Separate step: run the lead's loader and compare role membership exactly (row-id sets per role/subrole)."""
    man, mine = independent()
    sys.path.insert(0, str(WT))
    import rgj.data as R  # noqa: E402  (imported only in this step)
    D = R.load()
    theirs = {r: np.sort(D["row_id"][D["idx"][r]]) for r in R.ROLES + R.SUBROLES}
    exact = {r: bool(np.array_equal(mine[r], theirs[r])) for r in NEW_ROLES + SUBROLES}
    lm = R.manifest(D)
    hashes = {r: bool(lm[r]["row_id_sha256"] == man["roles" if r in NEW_ROLES else "defense_fit_subroles"][r]["row_id_sha256"]
                      and lm[r]["rows"] == man["roles" if r in NEW_ROLES else "defense_fit_subroles"][r]["rows"]
                      and lm[r]["groups"] == man["roles" if r in NEW_ROLES else "defense_fit_subroles"][r]["groups"])
              for r in NEW_ROLES + SUBROLES}
    dev_withheld = lm["DEVELOPMENT_ASSESSMENT"].get("sex_male_share") == "withheld until EVALUATION_LOCK"
    kept_rows = int(len(D["row_id"]))
    no_dropped = kept_rows == sum(man["roles"][r]["rows"] for r in NEW_ROLES)
    man["loader_comparison"] = {
        "loader": "rgj.data.load() (imported in a separate step, after the independent computation)",
        "rgj_data_py_sha256": sha_file(WT / "rgj" / "data.py"),
        "row_id_sets_identical": exact,
        "rgj_manifest_counts_and_hashes_identical": hashes,
        "loader_keeps_only_the_five_roles": no_dropped,
        "loader_rows": kept_rows,
        "loader_partition_info": D["partition_info"],
        "rgj_manifest_withholds_DEVELOPMENT_ASSESSMENT_label_share": dev_withheld,
        "note": ("rgj.data.manifest() publishes a sex_male_share for non-assessment roles; this manifest does not "
                 "reproduce it. Only counts and hashes were compared."),
        "verdict": "MATCH" if all(exact.values()) and all(hashes.values()) and no_dropped else "MISMATCH"}
    OUT.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man["loader_comparison"], indent=1))


def _sha_arrays(*arrays) -> str:
    """Same fingerprint convention as oar.fare_official._sha256_arrays (dtype + shape + bytes), re-implemented."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _complete(d: Path):
    """(status, n_files): status OK iff COMPLETE.json exists and every listed file re-hashes to its recorded sha256."""
    c = d / "COMPLETE.json"
    if not c.exists():
        return "MISSING_RECEIPT", 0
    files = json.loads(c.read_text())["files"]
    bad = [f for f, h in files.items() if not (d / f).exists() or sha_file(d / f) != h]
    return ("OK" if not bad else f"HASH_MISMATCH:{bad}"), len(files)


def reuse_audit():
    """Read-only check of the predecessor (jcv) warm-start and FARE units against the new roles. No fit, no predict.
    Labels are touched only through rgj.data.load() and only for DEFENSE_FIT rows (to re-derive a label hash)."""
    sys.path.insert(0, str(WT))
    import rgj.data as R  # noqa: E402
    units = Path.home() / "PCRL_eval_cache_private" / "jcv_v1" / "run" / "units"
    cache = Path.home() / "PCRL_eval_cache_private" / "jcv_v1" / "fare_cache"
    z = np.load(SRC, allow_pickle=False)
    old_role, X = z["role"].astype(str), z["X"]
    tr_old = np.flatnonzero(old_role == "defense_train")
    D = R.load()
    df = D["idx"]["DEFENSE_FIT"]
    assert np.array_equal(D["row_id"][df], z["row_id"][tr_old]), "DEFENSE_FIT order differs from old defense_train"
    Xfit = np.ascontiguousarray(X[tr_old], dtype=np.float64)
    Xall = np.ascontiguousarray(X, dtype=np.float64)
    fit_rows_sha, all_rows_sha = _sha_arrays(Xfit), _sha_arrays(Xall)
    out = {"warm": {}, "fare": {}}
    src_txt = (WT / "jcv" / "train.py").read_text()
    for k in (0, 1, 2):
        d = units / f"warm__s{k}"
        st, nf = _complete(d)
        rec = json.loads((d / "record.json").read_text())
        out["warm"][f"warm__s{k}"] = {
            "complete_json": st, "files": nf,
            "warm_pt_sha256": json.loads((d / "COMPLETE.json").read_text())["files"].get("warm.pt"),
            "epochs_logged": len(rec["log"]), "HP_warm_epochs": rec["HP"]["warm_epochs"],
            "HP_warm_lr_adam": rec["HP"]["warm_lr_adam"], "HP_guard_seed": rec["HP"]["guard_seed"],
            "HP_guard_size": rec["HP"]["guard_size"],
            "log_has_validation_entries": any(("val" in kk) for e in rec["log"] for kk in e)}
    out["warm_code"] = {"jcv_train_py_sha256_at_base": hashlib.sha256(src_txt.encode()).hexdigest(),
                        "warm_start_reads_validation": ("defense_val" in src_txt.split("def warm_start")[1].split("def guard_losses")[0])}
    for k in (0, 1, 2):
        for i in (0, 1):
            t = D["y"]["income" if i == 0 else "occupation_group"][df].astype(np.int64)
            s = D["sex"][df].astype(np.int64)
            lab_sha = _sha_arrays(t, s)
            t_sha, s_sha = _sha_arrays(t), _sha_arrays(s)
            for tag in [f"c{c}" for c in range(1, 7)] + ["Z1"]:
                name = f"fare__s{k}__p{i}__{tag}"
                d = units / name
                if not d.exists():
                    continue
                st, _ = _complete(d)
                rec = json.loads((d / "record.json").read_text())
                cd = cache / rec["fare_uid"]
                cst, _ = _complete(cd)
                cr = json.loads((cd / "rec.json").read_text())
                cfg_canon = json.dumps({kk: cr["cfg"][kk] for kk in sorted(cr["cfg"])}, sort_keys=True, default=str)
                inputs = hashlib.sha256("|".join([fit_rows_sha, t_sha, s_sha, all_rows_sha, cfg_canon,
                                                  str(int(cr["seed"]))]).encode()).hexdigest()
                fh = np.load(cd / "model" / "fit_row_hashes.npy")
                out["fare"][name] = {
                    "unit_complete_json": st, "tree_cache_uid": rec["fare_uid"], "tree_cache_complete_json": cst,
                    "n_fit": cr["n_fit"], "n_fit_equals_DEFENSE_FIT_rows": cr["n_fit"] == len(df),
                    "fit_rows_sha256_equals_DEFENSE_FIT_inputs": cr["fit_rows_sha256"] == fit_rows_sha,
                    "fit_labels_sha256_equals_DEFENSE_FIT_task_and_sex": cr["fit_labels_sha256"] == lab_sha,
                    "inputs_sha256_recomputed_equal": cr["inputs_sha256"] == inputs,
                    "n_fit_row_hashes": int(len(fh)),
                    "encoded_rows_n_all": cr["n_all"], "encoded_all_39205_rows_incl_dropped_pools": cr["n_all"] == len(old_role),
                    "config": cr["cfg"].get("name"), "gamma": cr["cfg_official"]["gamma"], "seed": cr["seed"],
                    "random_state": cr["random_state"],
                    "head_selected_C": rec["head"]["selected_C"],
                    "head_selected_on": [kk for kk in rec["head"]["table"][0] if kk != "C"],
                    "amendment_A1": bool(rec.get("amendment_A1"))}
    return out


if __name__ == "__main__":
    if "--reuse-audit" in sys.argv:
        print(json.dumps(reuse_audit(), indent=1))
    elif "--compare-loader" in sys.argv:
        compare_loader()
    else:
        m, _ = independent()
        print(json.dumps({"roles": {r: (v["rows"], v["groups"]) for r, v in m["roles"].items()},
                          "subroles": {r: (v["rows"], v["groups"]) for r, v in m["defense_fit_subroles"].items()},
                          "assertions": m["assertions"], "old_spans": m["old_roles"]["groups_spanning_old_roles"],
                          "x_dup": m["input_duplicate_note"]["counts"],
                          "cols": m["permitted_columns"]["columns_per_source"],
                          "std_ok": m["permitted_columns"]["numeric_standardisation_fit_on_DEFENSE_FIT"]}, indent=1))
