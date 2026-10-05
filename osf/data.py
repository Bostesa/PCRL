"""Role admission for the online-strength frontier study (consolidated exposed benchmark; no fresh-data claim).

Source: admitted input adult_jcv.npz (sha256 pinned; 39,205 historically exposed Adult rows). The fitting/selection roles
are the strength-matched feedback study's (smf.data, pinned rule, unchanged):
  OSF_DEFENSE_FIT   = smf NEW_DEFENSE_FIT       encoders, training heads, critics, LEACE maps, FARE trees
  HEAD_VALIDATION   = smf HEAD_VALIDATION       deployed-head C selection only
  AUDIT_FIT         = smf AUDIT_FIT             attacker fitting
  INNER_SELECTION   = smf INNER_SELECTION       candidate evaluation, nomination, attacker selection
  subroles of OSF_DEFENSE_FIT (smf rule): CRITIC_FIT, CRITIC_VAL, DIAGNOSTIC_CALIB (= smf CONTROLLER_CALIB; no
  controller here). Encoders train on all OSF_DEFENSE_FIT rows; subroles are held out only from critic/diagnostic fits.
OSF_DEVELOPMENT_ASSESSMENT = the fixed union of four previously used pools (all eligible groups, no favourable subset):
  ORIG_ASSESSMENT  old "assessment" pool (5,243 rows)
  RGJ_DEV          refreshed study DEVELOPMENT_ASSESSMENT (3,397 rows; 70% of old defense_val groups)
  SMF_DEV          smf NEW_DEVELOPMENT_ASSESSMENT (3,796 rows; 20% of old defense_train groups)
  CERT             old certification pool (1,500 rows) iff CERT_ELIGIBLE (custody: no eligible model's fitting or
                   selection ever received it - smf.data and rgj.data drop it at load; see ROLE_MANIFEST.json)
  minus every exact-record group that also has a row in OSF_DEFENSE_FIT, HEAD_VALIDATION, AUDIT_FIT, INNER_SELECTION or
  the inherited exclusions (excluded_exposure, excluded_dup). A group is never split.
Labels of OSF_DEVELOPMENT_ASSESSMENT are masked (-1) at load until the pushed EVALUATION_LOCK (osf.assess unseals).
Preprocessing: smf.data.refit_numeric (exact integer inversion of the admitted normalisation, re-standardised on
OSF_DEFENSE_FIT only) applied to every kept row; the 78 one-hot columns keep the loader's fixed category sets (an unseen
category is an all-zero block). 83 permitted columns, order unchanged. Internal aliases (pinned rgj/smf helpers run
unchanged): DEFENSE_FIT = OSF_DEFENSE_FIT, DEVELOPMENT_ASSESSMENT = OSF_DEVELOPMENT_ASSESSMENT, CALIB =
DIAGNOSTIC_CALIB.
Role allowlist (ALLOW): which procedure may read labels of which role; the assessment appears only under "assessment".
"""
from __future__ import annotations

import hashlib

import numpy as np

from rgj import data as RD
from smf import data as SD

SRC, SRC_SHA = RD.SRC, RD.SRC_SHA
ROLES = ("OSF_DEFENSE_FIT", "OSF_DEVELOPMENT_ASSESSMENT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
SUBROLES = ("CRITIC_FIT", "CRITIC_VAL", "DIAGNOSTIC_CALIB")
FIT_ROLES = ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
POOLS = ("ORIG_ASSESSMENT", "RGJ_DEV", "SMF_DEV", "CERT")
EXCLUSIONS = ("excluded_exposure", "excluded_dup")
CERT_ELIGIBLE = True
ALIASES = {"DEFENSE_FIT": "OSF_DEFENSE_FIT", "DEVELOPMENT_ASSESSMENT": "OSF_DEVELOPMENT_ASSESSMENT",
           "CALIB": "DIAGNOSTIC_CALIB"}
ALLOW = {"training": ("OSF_DEFENSE_FIT",), "heads": ("OSF_DEFENSE_FIT", "HEAD_VALIDATION"),
         "inner_audit": ("OSF_DEFENSE_FIT", "AUDIT_FIT", "INNER_SELECTION"), "selection": ("INNER_SELECTION",),
         "assessment": ("OSF_DEFENSE_FIT", "AUDIT_FIT", "INNER_SELECTION", "OSF_DEVELOPMENT_ASSESSMENT")}
N_PERMITTED = 83
LABEL_KEYS = ("sex", "race", "y_income", "y_occupation_group")


def assign_roles(old_role, unit, cert_eligible=CERT_ELIGIBLE):
    """Pure function of the old role array and the group ids (no labels). Returns (role, subrole, pool, info)."""
    smf_role, smf_sub, rgj_role = SD.assign_roles(old_role, unit)
    role = np.full(len(old_role), "", dtype="<U32")
    sub = np.full(len(old_role), "", dtype="<U32")
    pool = np.full(len(old_role), "", dtype="<U32")
    role[smf_role == "NEW_DEFENSE_FIT"] = "OSF_DEFENSE_FIT"
    for r in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
        role[smf_role == r] = r
    sub[smf_sub == "CRITIC_FIT"] = "CRITIC_FIT"
    sub[smf_sub == "CRITIC_VAL"] = "CRITIC_VAL"
    sub[smf_sub == "CONTROLLER_CALIB"] = "DIAGNOSTIC_CALIB"
    pool[old_role == "assessment"] = "ORIG_ASSESSMENT"
    pool[rgj_role == "DEVELOPMENT_ASSESSMENT"] = "RGJ_DEV"
    pool[smf_role == "NEW_DEVELOPMENT_ASSESSMENT"] = "SMF_DEV"
    if cert_eligible:
        pool[old_role == "cert"] = "CERT"
    blocked = set(np.unique(unit[np.isin(role, FIT_ROLES) | np.isin(old_role, EXCLUSIONS)]).tolist())
    cand = pool != ""
    assert not np.any(role[cand] != ""), "a pool row already has a fitting role"
    bad = cand & np.isin(unit, list(blocked))
    info = {p: {"nominal_rows": int((pool == p).sum()), "excluded_rows_group_overlap": int((bad & (pool == p)).sum())}
            for p in POOLS}
    info["CERT"]["eligible"] = bool(cert_eligible)
    if not cert_eligible:
        info["CERT"]["nominal_rows"] = int((old_role == "cert").sum())
        info["CERT"]["excluded_rows_ineligible_pool"] = info["CERT"]["nominal_rows"]
    pool[bad] = ""
    role[pool != ""] = "OSF_DEVELOPMENT_ASSESSMENT"
    return role, sub, pool, info


def check_partition(role, sub, pool, unit, old_role):
    keep = role != ""
    assert set(np.unique(role[keep])) == set(ROLES)
    seen = {}
    for r, u in zip(np.where(keep, role, "DROPPED:" + old_role), unit):
        seen.setdefault(int(u), set()).add(r)
    span = [u for u, s in seen.items() if len(s) > 1 and any(not x.startswith("DROPPED:") for x in s)]
    assert not span, f"{len(span)} groups span a kept role and another role/pool"
    assert np.all(np.isin(old_role[role == "OSF_DEVELOPMENT_ASSESSMENT"], ("assessment", "cert", "defense_val",
                                                                         "defense_train")))
    assert not np.any(np.isin(old_role[keep], EXCLUSIONS))
    df = role == "OSF_DEFENSE_FIT"
    assert set(np.unique(sub[df])) == set(SUBROLES) and np.all(sub[~df] == "")
    assert np.all((pool != "") == (role == "OSF_DEVELOPMENT_ASSESSMENT"))


def load(verify=True, unseal=False):
    """D for this study (rows of the five roles). unseal=True only from osf.assess after the pushed EVALUATION_LOCK."""
    if verify and RD.sha_file(SRC) != SRC_SHA:
        raise SystemExit("REFUSED: adult_jcv.npz does not match the admitted hash")
    z = np.load(SRC, allow_pickle=False)
    old_role, unit = z["role"], z["unit"]
    role, sub, pool, info = assign_roles(old_role, unit)
    check_partition(role, sub, pool, unit, old_role)
    keep = np.flatnonzero(role != "")
    D = {"row_id": z["row_id"][keep], "unit": unit[keep], "role": role[keep], "subrole": sub[keep],
         "pool": pool[keep], "feature_names": z["feature_names"], "pool_info": info}
    X, norm = SD.refit_numeric(z["X"][keep], z["feature_names"], role[keep] == "OSF_DEFENSE_FIT")
    D["X"], D["numeric_refit"] = X, norm
    for k in LABEL_KEYS:
        D[k] = z[k][keep].copy()
    del z
    D["sealed"] = not unseal
    if not unseal:
        assess = D["role"] == "OSF_DEVELOPMENT_ASSESSMENT"
        for k in LABEL_KEYS:
            D[k][assess] = -1          # sealed: any accidental use fails loudly or is visibly wrong
    assert D["X"].shape[1] == N_PERMITTED
    low = [f.lower() for f in D["feature_names"]]
    assert not [f for f in low if f.split("=")[0] in ("sex", "race", "income", "occupation", "fnlwgt")]
    D["idx"] = {r: np.flatnonzero(D["role"] == r) for r in ROLES}
    D["idx"].update({r: np.flatnonzero(D["subrole"] == r) for r in SUBROLES})
    for a, b in ALIASES.items():
        D["idx"][a] = D["idx"][b]
    D["y"] = {"income": D["y_income"], "occupation_group": D["y_occupation_group"]}
    return D


def labels_for(D, procedure, role):
    """Allowlist guard: refuses a procedure reading labels of a role outside ALLOW (and sealed labels always)."""
    if role not in ALLOW[procedure] and ALIASES.get(role) not in ALLOW[procedure]:
        raise PermissionError(f"{procedure} may not read labels of {role}")
    ix = D["idx"][role]
    if D["sealed"] and np.any(D["role"][ix] == "OSF_DEVELOPMENT_ASSESSMENT"):
        raise PermissionError("assessment labels are sealed until the pushed EVALUATION_LOCK")
    return ix


def manifest(D):
    out = {}
    for r in ROLES + SUBROLES:
        ix = D["idx"][r]
        out[r] = {"rows": int(len(ix)), "groups": int(len(np.unique(D["unit"][ix]))),
                  "row_id_sha256": RD.sha_arr(np.sort(D["row_id"][ix]).astype(np.int64))}
    a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    out["OSF_DEVELOPMENT_ASSESSMENT"]["by_pool"] = {
        p: {"rows": int((D["pool"][a] == p).sum()), "groups": int(len(np.unique(D["unit"][a][D["pool"][a] == p])))}
        for p in POOLS}
    out["pool_admission"] = D["pool_info"]
    return out


def fit_tensor_sha(D):
    """Hash of the training tensors (X, task labels, SEX of OSF_DEFENSE_FIT in source order): equals the smf hash iff
    the smf checkpoints saw exactly these fitting inputs."""
    tr = D["idx"]["DEFENSE_FIT"]
    h = hashlib.sha256()
    for a in (D["X"][tr], D["y_income"][tr], D["y_occupation_group"][tr], D["sex"][tr], D["row_id"][tr]):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
