"""Loader wrapper, calibration roles and the label allowlist of the held-out calibration study (hcal; role A).

load() returns lra.data.load()'s D UNCHANGED (pinned osf -> dpc -> qpc -> lra loaders, sealed: assessment labels are -1)
and attaches D["hcal"] = the new role split. Nothing pinned is edited or monkeypatched. Only hcal.assess may request
unsealed labels, through load(unseal=True), whose gate requires (a) caller module hcal.assess and (b) this study's
EVALUATION_LOCK.json committed and byte-identical on origin/research/pcrl-heldout-calibration-v1. The lra gate is never
used (it opens only for lra.assess); after the hcal gate passes, the pinned osf.data.load(unseal=True) is called directly
with the pinned loader-blob and role checks, exactly as lra.data does for lra.

ROLE SPLIT (prompt section 5; assigned from group identities only, before any label, SEX frequency or calibration
performance is inspected). Canonical group ID = D["unit"] (the osf exact-record group integer), serialised as its decimal
string. Rank key = sha256(UTF-8 (salt + canonical ID)).hexdigest(); ties by the integer ID. One representative per
selected group = its smallest row ID (D["row_id"]).
  CALIBRATION_HELDOUT        first 2,000 AUDIT_FIT groups under salt "hcal-v1|heldout|20261011|" (2,000 representatives)
  ATTACK_FIT_NEW             every row of the remaining AUDIT_FIT groups (expected 4,061 groups)
  CALIBRATION_TRAIN_MATCHED  first 2,000 OSF_DEFENSE_FIT groups under salt "hcal-v1|train-matched|20261011|"
                             (2,000 representatives; diagnostic only)
Each selected group must lie wholly inside its parent role (checked). The split is deterministic and receipt-hashed.

LABEL ALLOWLIST (labels_for(D, procedure, role) -> row positions; sealed labels are refused):
  calibration       task labels of CALIBRATION_HELDOUT and CALIBRATION_TRAIN_MATCHED representatives
  selection         task labels of INNER_SELECTION (inherited qpc procedure)
  fitting           task labels of OSF_DEFENSE_FIT (inherited; the source constant predictor only)
  attack            SEX of ATTACK_FIT_NEW (fresh readers), AUDIT_FIT (admitted legacy readers, refit for replay /
                    assessment) and INNER_SELECTION (reader selection)
  attack_diagnostic SEX of OSF_DEFENSE_FIT (plug-in MI diagnostic only; never an objective)
  assessment        OSF_DEVELOPMENT_ASSESSMENT, only on an unsealed D (hcal.assess after the pushed EVALUATION_LOCK)
HEAD_VALIDATION labels are never readable by any hcal procedure (it selected deployed heads). Inherited procedures are
delegated to qpc.data.labels_for unchanged.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np

from hcal import ids as I

SALT_HELDOUT = "hcal-v1|heldout|20261011|"
SALT_TRAIN = "hcal-v1|train-matched|20261011|"
N_CAL = 2000
EXPECTED_ATTACK_NEW_GROUPS = 4061
UNSEAL_CALLER = "hcal.assess"
EVALUATION_LOCK = I.PKG / "EVALUATION_LOCK.json"
HCAL_ROLES = ("CALIBRATION_HELDOUT", "ATTACK_FIT_NEW", "CALIBRATION_TRAIN_MATCHED")
ALLOW = {"calibration": ("CALIBRATION_HELDOUT", "CALIBRATION_TRAIN_MATCHED"),
         "attack": ("ATTACK_FIT_NEW", "AUDIT_FIT", "INNER_SELECTION"),
         "attack_diagnostic": ("OSF_DEFENSE_FIT",)}
INHERITED = {"selection": ("INNER_SELECTION",), "fitting": ("OSF_DEFENSE_FIT",)}
FORBIDDEN = ("HEAD_VALIDATION",)


def sha_hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha_arr(a):
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def canonical_group_id(u):
    return str(int(u))


def rank_groups(group_ids, salt):
    """Groups sorted by (sha256(salt + canonical id), integer id). Returns the list of integer ids and their keys."""
    keyed = sorted((sha_hex(salt + canonical_group_id(u)), int(u)) for u in group_ids)
    return [u for _, u in keyed], [h for h, _ in keyed]


def _groups_of(D, role):
    pos = np.asarray(D["idx"][role], dtype=np.int64)
    return pos, np.asarray(D["unit"], dtype=np.int64)[pos]


def _select(D, parent, salt, n_sel):
    pos, grp = _groups_of(D, parent)
    uniq = np.unique(grp)
    order, keys = rank_groups(uniq.tolist(), salt)
    chosen = np.asarray(order[:n_sel], dtype=np.int64)
    rest = np.asarray(order[n_sel:], dtype=np.int64)
    all_grp = np.asarray(D["unit"], dtype=np.int64)
    rid = np.asarray(D["row_id"], dtype=np.int64)
    in_chosen = np.isin(all_grp, chosen)
    member_pos = np.flatnonzero(in_chosen)
    # purity: every member row of a chosen group lies in the parent role
    impure = np.setdiff1d(member_pos, pos)
    if impure.size:
        raise SystemExit(f"REFUSED: {impure.size} rows of selected {parent} groups lie outside {parent}")
    reps = []
    for u in chosen:                                # chosen is in rank order; representatives listed in that order
        m = member_pos[all_grp[member_pos] == u]
        reps.append(int(m[np.argmin(rid[m])]))
    reps = np.asarray(reps, dtype=np.int64)
    rest_pos = np.flatnonzero(np.isin(all_grp, rest))
    if np.setdiff1d(rest_pos, pos).size:
        raise SystemExit(f"REFUSED: rows of remaining {parent} groups lie outside {parent}")
    return {"groups": chosen, "group_keys": keys[:n_sel], "reps": reps, "member_rows": member_pos,
            "rest_groups": rest, "rest_rows": rest_pos, "parent_groups": int(uniq.size)}


def build_roles(D):
    """The hcal role split from group identities and row ids only (no label is read). Deterministic."""
    h = _select(D, "AUDIT_FIT", SALT_HELDOUT, N_CAL)
    t = _select(D, "OSF_DEFENSE_FIT", SALT_TRAIN, N_CAL)
    roles = {"CALIBRATION_HELDOUT": np.sort(h["reps"]), "CALIBRATION_HELDOUT_GROUP_ROWS": np.sort(h["member_rows"]),
             "ATTACK_FIT_NEW": np.sort(h["rest_rows"]), "CALIBRATION_TRAIN_MATCHED": np.sort(t["reps"]),
             "CALIBRATION_TRAIN_MATCHED_GROUP_ROWS": np.sort(t["member_rows"])}
    meta = {"heldout": h, "train": t}
    return roles, meta


def role_receipt(D, roles, meta):
    """Counts, hashes and disjointness proofs of the new roles (JSON-safe; no labels)."""
    rid = np.asarray(D["row_id"], dtype=np.int64)
    grp = np.asarray(D["unit"], dtype=np.int64)
    idx = {r: np.asarray(v, dtype=np.int64) for r, v in D["idx"].items()}

    def gset(pos):
        return set(grp[pos].tolist())

    def desc(pos):
        pos = np.asarray(pos, dtype=np.int64)
        return {"rows": int(pos.size), "groups": int(len(gset(pos))), "row_id_sha256": sha_arr(np.sort(rid[pos])),
                "group_id_sha256": sha_arr(np.unique(grp[pos]))}
    out = {"rule": {"canonical_group_id": "D['unit'] (osf exact-record group integer) as a decimal string",
                    "rank_key": "sha256(UTF-8(salt + canonical id)).hexdigest(), ties by integer id",
                    "salt_heldout": SALT_HELDOUT, "salt_train_matched": SALT_TRAIN, "n_selected_groups": N_CAL,
                    "representative": "smallest row_id of the selected group",
                    "labels_read": "none (group ids and row ids only)"},
           "roles": {}, "disjointness": {}, "inherited": {}}
    for r in ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION", "OSF_DEVELOPMENT_ASSESSMENT"):
        out["inherited"][r] = desc(idx[r])
    for r, pos in roles.items():
        out["roles"][r] = desc(pos)
    out["roles"]["CALIBRATION_HELDOUT"]["selected_groups_in_rank_order_sha256"] = sha_arr(meta["heldout"]["groups"])
    out["roles"]["CALIBRATION_HELDOUT"]["representatives_in_rank_order_sha256"] = sha_arr(rid[meta["heldout"]["reps"]])
    out["roles"]["CALIBRATION_TRAIN_MATCHED"]["selected_groups_in_rank_order_sha256"] = sha_arr(meta["train"]["groups"])
    out["roles"]["CALIBRATION_TRAIN_MATCHED"]["representatives_in_rank_order_sha256"] = \
        sha_arr(rid[meta["train"]["reps"]])
    out["parent_groups"] = {"AUDIT_FIT": meta["heldout"]["parent_groups"],
                            "OSF_DEFENSE_FIT": meta["train"]["parent_groups"]}
    cal = roles["CALIBRATION_HELDOUT_GROUP_ROWS"]
    tm = roles["CALIBRATION_TRAIN_MATCHED_GROUP_ROWS"]
    att = roles["ATTACK_FIT_NEW"]
    pairs = {"CALIBRATION_HELDOUT vs OSF_DEFENSE_FIT (teacher parameter fitting)": (cal, idx["OSF_DEFENSE_FIT"]),
             "CALIBRATION_HELDOUT vs HEAD_VALIDATION": (cal, idx["HEAD_VALIDATION"]),
             "CALIBRATION_HELDOUT vs ATTACK_FIT_NEW": (cal, att),
             "CALIBRATION_HELDOUT vs INNER_SELECTION": (cal, idx["INNER_SELECTION"]),
             "CALIBRATION_HELDOUT vs OSF_DEVELOPMENT_ASSESSMENT": (cal, idx["OSF_DEVELOPMENT_ASSESSMENT"]),
             "ATTACK_FIT_NEW vs INNER_SELECTION": (att, idx["INNER_SELECTION"]),
             "ATTACK_FIT_NEW vs OSF_DEVELOPMENT_ASSESSMENT": (att, idx["OSF_DEVELOPMENT_ASSESSMENT"]),
             "ATTACK_FIT_NEW vs OSF_DEFENSE_FIT": (att, idx["OSF_DEFENSE_FIT"]),
             "ATTACK_FIT_NEW vs HEAD_VALIDATION": (att, idx["HEAD_VALIDATION"]),
             "CALIBRATION_TRAIN_MATCHED vs INNER_SELECTION": (tm, idx["INNER_SELECTION"]),
             "CALIBRATION_TRAIN_MATCHED vs OSF_DEVELOPMENT_ASSESSMENT": (tm, idx["OSF_DEVELOPMENT_ASSESSMENT"]),
             "CALIBRATION_TRAIN_MATCHED vs AUDIT_FIT": (tm, idx["AUDIT_FIT"]),
             "CALIBRATION_TRAIN_MATCHED vs HEAD_VALIDATION": (tm, idx["HEAD_VALIDATION"])}
    for name, (a, b) in pairs.items():
        out["disjointness"][name] = {"shared_rows": int(np.intersect1d(a, b).size),
                                     "shared_groups": int(len(gset(a) & gset(b)))}
    out["containment"] = {
        "CALIBRATION_HELDOUT groups inside AUDIT_FIT": bool(np.isin(cal, idx["AUDIT_FIT"]).all()),
        "ATTACK_FIT_NEW inside AUDIT_FIT": bool(np.isin(att, idx["AUDIT_FIT"]).all()),
        "CALIBRATION_HELDOUT + ATTACK_FIT_NEW = AUDIT_FIT": bool(np.array_equal(np.sort(np.concatenate([cal, att])),
                                                                                np.sort(idx["AUDIT_FIT"]))),
        "CALIBRATION_TRAIN_MATCHED groups inside OSF_DEFENSE_FIT": bool(np.isin(tm, idx["OSF_DEFENSE_FIT"]).all())}
    out["ok"] = (all(v["shared_rows"] == 0 and v["shared_groups"] == 0 for v in out["disjointness"].values())
                 and all(out["containment"].values())
                 and out["roles"]["CALIBRATION_HELDOUT"]["rows"] == N_CAL
                 and out["roles"]["CALIBRATION_HELDOUT"]["groups"] == N_CAL
                 and out["roles"]["CALIBRATION_TRAIN_MATCHED"]["rows"] == N_CAL
                 and out["roles"]["CALIBRATION_TRAIN_MATCHED"]["groups"] == N_CAL)
    out["attack_fit_new_groups_expected"] = EXPECTED_ATTACK_NEW_GROUPS
    out["attack_fit_new_groups_match_expected"] = out["roles"]["ATTACK_FIT_NEW"]["groups"] == EXPECTED_ATTACK_NEW_GROUPS
    return out


# ------------------------------------------------------------------ unseal gate (hcal only)
def _caller_module(depth):
    f = sys._getframe(depth)
    name = f.f_globals.get("__name__")
    if name == "__main__":
        spec = f.f_globals.get("__spec__")
        name = getattr(spec, "name", None) or "__main__"
    return name


def evaluation_lock_pushed(lock=None, branch=I.BRANCH):
    lock = lock or EVALUATION_LOCK
    rel = f"{I.REL}/EVALUATION_LOCK.json"
    if not lock.exists():
        return False, "EVALUATION_LOCK.json does not exist"
    subprocess.run(["git", "-C", str(I.WT), "fetch", "-q", "origin", branch], capture_output=True)
    r = subprocess.run(["git", "-C", str(I.WT), "show", f"origin/{branch}:{rel}"], capture_output=True)
    if r.returncode != 0:
        return False, "EVALUATION_LOCK.json is not on origin"
    if r.stdout != lock.read_bytes():
        return False, "local EVALUATION_LOCK.json differs from origin"
    return True, "EVALUATION_LOCK.json byte-identical on origin"


def unseal_gate(caller):
    if caller != UNSEAL_CALLER:
        raise PermissionError(f"REFUSED: only {UNSEAL_CALLER} may unseal assessment labels (caller: {caller})")
    ok, why = evaluation_lock_pushed()
    if not ok:
        raise PermissionError(f"REFUSED: assessment labels stay sealed: {why}")
    return why


def load(verify=True, unseal=False):
    """D with the hcal roles attached. Sealed unless called from hcal.assess after the pushed EVALUATION_LOCK."""
    from osf import data as OD
    from qpc import data as QD
    from lra import data as LD
    if unseal:
        gate = unseal_gate(_caller_module(2))
        if verify:
            bad = [f for f, ok in QD.loader_blob_checks().items() if not ok]
            if bad:
                raise SystemExit(f"REFUSED: pinned loader differs from its blob at the pin: {bad}")
        D = OD.load(verify=verify, unseal=True)
        if verify:
            _, bad = QD.verify_D(D)
            if bad:
                raise SystemExit(f"REFUSED: roles differ from the pinned manifests: {bad}")
    else:
        gate = None
        D = LD.load(verify=verify, unseal=False)
        a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
        assert D["sealed"] and all(np.all(np.asarray(D[k])[a] == -1) for k in QD.LABEL_KEYS), "not sealed"
    roles, meta = build_roles(D)
    D["hcal"] = {"roles": roles, "receipt": role_receipt(D, roles, meta)}
    if not D["hcal"]["receipt"]["ok"]:
        raise SystemExit("REFUSED: hcal role split fails its disjointness/containment receipt")
    D["study"] = I.STUDY
    D["unseal_gate"] = gate
    return D


def role_rows(D, role):
    if role in HCAL_ROLES:
        return np.asarray(D["hcal"]["roles"][role], dtype=np.int64)
    return np.asarray(D["idx"][role], dtype=np.int64)


def labels_for(D, procedure, role):
    """Row positions whose labels `procedure` may read on `role` (the hcal allowlist; module docstring)."""
    if role in FORBIDDEN:
        raise PermissionError(f"REFUSED: no hcal procedure may read labels of {role}")
    if procedure in ALLOW:
        if role not in ALLOW[procedure]:
            raise PermissionError(f"REFUSED: procedure {procedure!r} may not read labels of {role}")
        rows = role_rows(D, role)
    elif procedure in INHERITED:
        if role not in INHERITED[procedure]:
            raise PermissionError(f"REFUSED: procedure {procedure!r} may not read labels of {role}")
        from qpc import data as QD
        rows = np.asarray(QD.labels_for(D, procedure, role), dtype=np.int64)
    elif procedure == "assessment":
        if role != "OSF_DEVELOPMENT_ASSESSMENT" or D.get("sealed", True):
            raise PermissionError("REFUSED: assessment labels are sealed until hcal.assess opens them")
        rows = role_rows(D, role)
    else:
        raise PermissionError(f"REFUSED: unknown procedure {procedure!r}")
    if procedure != "assessment":
        a = np.asarray(D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"], dtype=np.int64)
        if np.intersect1d(rows, a).size:
            raise PermissionError("REFUSED: assessment rows requested outside hcal.assess")
    return rows


def task_labels(D, task, procedure, role):
    """(positions, int64 labels) of one task ("income" / "occupation") through the allowlist."""
    rows = labels_for(D, procedure, role)
    key = {"income": "y_income", "occupation": "y_occupation_group"}[task]
    y = np.asarray(D[key], dtype=np.int64)[rows]
    if np.any(y < 0):
        raise PermissionError(f"REFUSED: sealed labels on {role}")
    return rows, y


def sex_labels(D, procedure, role):
    rows = labels_for(D, procedure, role)
    s = np.asarray(D["sex"], dtype=np.int64)[rows]
    if np.any(s < 0):
        raise PermissionError(f"REFUSED: sealed SEX labels on {role}")
    return rows, s


def write_role_manifest(D, path):
    rec = {"schema": "hcal-role-manifest-v1", "study": I.STUDY, **D["hcal"]["receipt"],
           "allowlist": {**{k: list(v) for k, v in ALLOW.items()}, **{k: list(v) for k, v in INHERITED.items()},
                         "assessment": ["OSF_DEVELOPMENT_ASSESSMENT (hcal.assess after the pushed EVALUATION_LOCK)"],
                         "forbidden": list(FORBIDDEN)}}
    path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec
