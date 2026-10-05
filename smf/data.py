"""Role admission for the strength-matched feedback study (new development partition inside already-exposed Adult rows).

Source: admitted predecessor input adult_jcv.npz (sha256 pinned). Roles (analysis unit = exact-record group `unit`):
  NEW_DEFENSE_FIT            80% of the refreshed study's DEFENSE_FIT groups (= old defense_train)
  NEW_DEVELOPMENT_ASSESSMENT 20% of those groups; labels MASKED (-1) at load until EVALUATION_LOCK.json is pushed
  HEAD_VALIDATION            preserved from the refreshed study (30% of old defense_val groups, seed 20261004 rule)
  AUDIT_FIT, INNER_SELECTION preserved (old attacker_fit / attacker_val)
Allocation: u = int(sha256("20261005|assess|<unit>")[:16], 16) / 2**64; NEW_DEVELOPMENT_ASSESSMENT iff u < 0.20.
NEW_DEFENSE_FIT subroles (salt "critic", seed 20261005): CRITIC_FIT u < 0.70, CRITIC_VAL 0.70 <= u < 0.85,
CONTROLLER_CALIB u >= 0.85. Encoders and training heads train on ALL NEW_DEFENSE_FIT rows (including the critic and
controller subroles' labels); the subroles are held out only from critic/controller fitting.
Excluded from every operation (dropped at load): old assessment, old cert, exposure/duplicate exclusions, and the
refreshed study's DEVELOPMENT_ASSESSMENT (3,397 rows).
Preprocessing: the admitted file standardised the 5 numeric columns on the old defense_train, which contains the new
assessment rows. They are refitted here on NEW_DEFENSE_FIT only: raw integer values are recovered exactly from the
admitted normalisation (DATA_ADMISSION.json numeric_norm; float32 inversion error asserted < 0.05, so rounding to the
integers is unambiguous) and re-standardised (population
sd). The 78 one-hot columns use the loader's fixed category sets (no fitted vocabulary). 83 columns, order unchanged.
Internal aliases: idx["DEFENSE_FIT"] = NEW_DEFENSE_FIT, idx["DEVELOPMENT_ASSESSMENT"] = NEW_DEVELOPMENT_ASSESSMENT,
idx["CALIB"] = CONTROLLER_CALIB, so pinned rgj helpers run unchanged on the new roles.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rgj import data as RD

HOME = Path.home()
SRC, SRC_SHA = RD.SRC, RD.SRC_SHA
WT = Path(__file__).resolve().parents[1]
ADMISSION = WT / "results" / "pcrl_joint_complete_view_method_v1" / "DATA_ADMISSION.json"
SEED = 20261005
ASSESS_SHARE = 0.20
CRITIC_SPLIT = (0.70, 0.85)
ROLES = ("NEW_DEFENSE_FIT", "NEW_DEVELOPMENT_ASSESSMENT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
SUBROLES = ("CRITIC_FIT", "CRITIC_VAL", "CONTROLLER_CALIB")
ALIASES = {"DEFENSE_FIT": "NEW_DEFENSE_FIT", "DEVELOPMENT_ASSESSMENT": "NEW_DEVELOPMENT_ASSESSMENT",
           "CALIB": "CONTROLLER_CALIB"}
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
N_PERMITTED = 83
LABEL_KEYS = ("sex", "race", "y_income", "y_occupation_group")


def group_u(units, salt):
    return np.array([int(hashlib.sha256(f"{SEED}|{salt}|{int(u)}".encode()).hexdigest()[:16], 16) / 2.0 ** 64
                     for u in units])


def assign_roles(old_role, unit):
    """Pure function of the old role array and group ids (no labels)."""
    rgj_role, _ = RD.assign_roles(old_role, unit)
    new = np.full(len(old_role), "", dtype="<U32")
    df = rgj_role == "DEFENSE_FIT"
    u = group_u(unit[df], "assess")
    new[np.flatnonzero(df)] = np.where(u < ASSESS_SHARE, "NEW_DEVELOPMENT_ASSESSMENT", "NEW_DEFENSE_FIT")
    for r in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
        new[rgj_role == r] = r
    sub = np.full(len(old_role), "", dtype="<U32")
    nf = new == "NEW_DEFENSE_FIT"
    uc = group_u(unit[nf], "critic")
    sub[np.flatnonzero(nf)] = np.where(uc < CRITIC_SPLIT[0], "CRITIC_FIT",
                                       np.where(uc < CRITIC_SPLIT[1], "CRITIC_VAL", "CONTROLLER_CALIB"))
    return new, sub, rgj_role


def check_partition(new, sub, unit, rgj_role):
    keep = new != ""
    assert set(np.unique(new[keep])) == set(ROLES)
    assert not np.any(rgj_role[keep] == "DEVELOPMENT_ASSESSMENT"), "refreshed assessment rows leaked into a new role"
    seen = {}
    for r, u in zip(np.where(keep, new, "DROPPED"), unit):
        seen.setdefault(int(u), set()).add(r)
    span = [u for u, s in seen.items() if len(s) > 1 and any(x != "DROPPED" for x in s)]
    assert not span, f"{len(span)} groups span a new role and another role/pool"
    nf = new == "NEW_DEFENSE_FIT"
    assert set(np.unique(sub[nf])) == set(SUBROLES) and np.all(sub[~nf] == "")
    ss = {}
    for r, u in zip(sub[nf], unit[nf]):
        ss.setdefault(int(u), set()).add(r)
    assert all(len(s) == 1 for s in ss.values())


def refit_numeric(X, names, fit_mask):
    """Recover raw integer numerics from the admitted normalisation and re-standardise on fit_mask rows only."""
    norm = json.loads(ADMISSION.read_text())["numeric_norm"]
    X = X.astype(np.float64).copy()
    out = {}
    for c in NUMERIC:
        j = list(names).index(c)
        mu, sd = norm[c]
        raw = X[:, j] * sd + mu
        r = np.round(raw)
        err = float(np.abs(raw - r).max())
        # float32 storage: |X| <= ~15 and sd <= ~7e3 give errors <= ~0.01; 0.05 keeps rounding unambiguous (ints)
        assert err < 0.05, f"{c}: raw values are not integers after inversion (max err {err})"
        m, s = float(r[fit_mask].mean()), float(r[fit_mask].std())
        X[:, j] = (r - m) / s
        out[c] = {"mean_new_fit": m, "sd_new_fit": s, "inversion_max_abs_err": err}
    return X.astype(np.float32), out


def load(verify=True, unseal=False):
    """D for the new study (only rows of the five new roles). unseal=True is for rgj-style assessment code only and
    must be called after the pushed EVALUATION_LOCK.json has been verified by the caller (smf.assess does this)."""
    if verify and RD.sha_file(SRC) != SRC_SHA:
        raise SystemExit("REFUSED: adult_jcv.npz does not match the admitted hash")
    z = np.load(SRC, allow_pickle=False)
    old_role, unit = z["role"], z["unit"]
    new, sub, rgj_role = assign_roles(old_role, unit)
    check_partition(new, sub, unit, rgj_role)
    keep = np.flatnonzero(new != "")
    D = {"row_id": z["row_id"][keep], "unit": unit[keep], "role": new[keep], "subrole": sub[keep],
         "feature_names": z["feature_names"]}
    X, norm = refit_numeric(z["X"][keep], z["feature_names"], new[keep] == "NEW_DEFENSE_FIT")
    D["X"], D["numeric_refit"] = X, norm
    for k in LABEL_KEYS:
        D[k] = z[k][keep].copy()
    del z
    assess = D["role"] == "NEW_DEVELOPMENT_ASSESSMENT"
    D["sealed"] = not unseal
    if not unseal:
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


def manifest(D):
    out = {}
    for r in ROLES + SUBROLES:
        ix = D["idx"][r]
        out[r] = {"rows": int(len(ix)), "groups": int(len(np.unique(D["unit"][ix]))),
                  "row_id_sha256": RD.sha_arr(np.sort(D["row_id"][ix]).astype(np.int64))}
    return out
