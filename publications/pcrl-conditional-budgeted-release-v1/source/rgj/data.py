"""Role admission for the refreshed guarded joint study (new development partition inside already-exposed Adult rows).

Source: the admitted predecessor input file adult_jcv.npz (sha256 pinned below; 83 permitted columns, preprocessing
fitted on the old defense_train role only, which is preserved here as DEFENSE_FIT).

New roles (fixed before any score; analysis unit = exact-record group `unit`):
  DEFENSE_FIT             = old defense_train   encoders, training heads, critics, LEACE maps, FARE trees
  HEAD_VALIDATION         = 30% of old defense_val groups   task-head C selection only
  DEVELOPMENT_ASSESSMENT  = 70% of old defense_val groups   untouched until EVALUATION_LOCK.json is pushed
  AUDIT_FIT               = old attacker_fit    inner and final attacker fitting
  INNER_SELECTION         = old attacker_val    candidate evaluation, nomination, attacker selection
Allocation of old defense_val groups: u = int(sha256("20261004|dev|<unit>")[:16], 16) / 2**64; HEAD_VALIDATION iff
u < 0.30. DEFENSE_FIT is further split by group for the critics (same hash, salt "critic"): CRITIC_FIT u < 0.70,
CRITIC_VAL 0.70 <= u < 0.85 (bounded-refit early stopping and attempt choice), CALIB u >= 0.85 (constraint/
multiplier calibration rows). Encoders still train on all of DEFENSE_FIT.

The old assessment, cert and excluded rows are dropped at load: no function in this package receives their inputs,
labels or row ids, and no prediction is ever produced for them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

HOME = Path.home()
SRC = HOME / "PCRL_eval_cache_private" / "jcv_v1" / "inputs" / "adult_jcv.npz"
SRC_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
SEED = 20261004
HEAD_SHARE = 0.30
CRITIC_SPLIT = (0.70, 0.85)
OLD_TO_NEW = {"defense_train": "DEFENSE_FIT", "attacker_fit": "AUDIT_FIT", "attacker_val": "INNER_SELECTION"}
DROPPED = ("assessment", "cert", "excluded_exposure", "excluded_dup")
ROLES = ("DEFENSE_FIT", "HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT", "AUDIT_FIT", "INNER_SELECTION")
SUBROLES = ("CRITIC_FIT", "CRITIC_VAL", "CALIB")
N_PERMITTED = 83
EXPECTED_OLD = {"defense_train": 19230, "defense_val": 4897, "attacker_fit": 6065, "attacker_val": 2235,
                "assessment": 5243, "cert": 1500, "excluded_exposure": 17, "excluded_dup": 18}
FORBIDDEN_FEATURE_TOKENS = ("sex", "race", "income", "occupation", "fnlwgt", "row_id", "record")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def group_u(units, salt):
    return np.array([int(hashlib.sha256(f"{SEED}|{salt}|{int(u)}".encode()).hexdigest()[:16], 16) / 2.0 ** 64
                     for u in units])


def assign_roles(old_role, unit):
    """Pure function of the old role array and the group ids (no labels). Returns (new role array, subrole array)."""
    new = np.full(len(old_role), "", dtype="<U32")
    for o, n in OLD_TO_NEW.items():
        new[old_role == o] = n
    dv = old_role == "defense_val"
    u = group_u(unit[dv], "dev")
    new[np.flatnonzero(dv)] = np.where(u < HEAD_SHARE, "HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT")
    sub = np.full(len(old_role), "", dtype="<U32")
    df = new == "DEFENSE_FIT"
    uc = group_u(unit[df], "critic")
    sub[np.flatnonzero(df)] = np.where(uc < CRITIC_SPLIT[0], "CRITIC_FIT",
                                       np.where(uc < CRITIC_SPLIT[1], "CRITIC_VAL", "CALIB"))
    return new, sub


def check_partition(new, sub, unit, old_role):
    counts = {r: int((old_role == r).sum()) for r in EXPECTED_OLD}
    assert counts == EXPECTED_OLD, f"old role counts changed: {counts}"
    keep = new != ""
    assert set(np.unique(new[keep])) == set(ROLES)
    assert np.all(np.isin(old_role[~keep], DROPPED)) and np.all(~np.isin(old_role[keep], DROPPED))
    # a group never spans two new roles, nor a new role and a dropped pool
    seen = {}
    for r, u in zip(np.where(keep, new, "DROPPED:" + old_role), unit):
        seen.setdefault(int(u), set()).add(r)
    span = {u: s for u, s in seen.items() if len(s) > 1}
    kept_span = {u: s for u, s in span.items() if any(not x.startswith("DROPPED:") for x in s)}
    assert not kept_span, f"{len(kept_span)} groups span a new role and another role"
    df = new == "DEFENSE_FIT"
    assert set(np.unique(sub[df])) == set(SUBROLES) and np.all(sub[~df] == "")
    sseen = {}
    for r, u in zip(sub[df], unit[df]):
        sseen.setdefault(int(u), set()).add(r)
    assert all(len(s) == 1 for s in sseen.values()), "a group spans two critic subroles"
    return {"groups_spanning_dropped_pools_only": len(span)}


def load(verify=True):
    """D for the new study: only rows of the five new roles; idx arrays index into these rows."""
    if verify and sha_file(SRC) != SRC_SHA:
        raise SystemExit("REFUSED: adult_jcv.npz does not match the admitted hash")
    z = np.load(SRC, allow_pickle=False)
    old_role, unit = z["role"], z["unit"]
    new, sub = assign_roles(old_role, unit)
    info = check_partition(new, sub, unit, old_role)
    keep = np.flatnonzero(new != "")
    D = {"row_id": z["row_id"][keep], "unit": unit[keep], "role": new[keep], "subrole": sub[keep],
         "X": z["X"][keep], "feature_names": z["feature_names"]}
    for k in ("sex", "race", "y_income", "y_occupation_group"):
        D[k] = z[k][keep]        # labels of dropped rows are never retained or passed on
    del z
    assert D["X"].shape[1] == N_PERMITTED
    low = [f.lower() for f in D["feature_names"]]
    bad = [f for f in low if any(f.split("=")[0].strip() == t or f.startswith(t + "=") for t in FORBIDDEN_FEATURE_TOKENS)]
    assert not bad, f"forbidden input columns present: {bad}"
    D["idx"] = {r: np.flatnonzero(D["role"] == r) for r in ROLES}
    D["idx"].update({r: np.flatnonzero(D["subrole"] == r) for r in SUBROLES})
    D["y"] = {"income": D["y_income"], "occupation_group": D["y_occupation_group"]}
    D["partition_info"] = info
    return D


def manifest(D):
    """Aggregate role manifest (counts and hashes only; no record keys)."""
    out = {}
    for r in ROLES + SUBROLES:
        ix = D["idx"][r]
        out[r] = {"rows": int(len(ix)), "groups": int(len(np.unique(D["unit"][ix]))),
                  "row_id_sha256": sha_arr(np.sort(D["row_id"][ix]).astype(np.int64)),
                  "sex_male_share": float(D["sex"][ix].mean()) if r != "DEVELOPMENT_ASSESSMENT" else "withheld until EVALUATION_LOCK"}
    return out
