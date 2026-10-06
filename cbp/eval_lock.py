"""EVALUATION_LOCK.json writer (committed and pushed before the single assessment opening; no manual substitution).

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.eval_lock results/pcrl_confidence_budgeted_privacy_v1/EVALUATION_LOCK.json

Adapted from qpc/eval_lock.py at d0c8a45. Freezes: the inner-only selection statuses and resolved configurations (P*, J*,
T*, C_rate, C_global, Q), the scored list (PROTOCOL.md section 12), every scored unit with the sha256 of every unit
file (release + inner), the composed source-reader bank (the 38 registered composition codes of each seed, which
contains every frozen composed winner), a hash of the OSF_DEFENSE_FIT SEX prior counts, the assessment role manifest,
attacker settings, lock / amendment hashes, locked code files, the 37 endpoints, the data-dependent role aliases
(alias_of_by_role) and the technical-validity inputs of the label truth table (real-data controls, endpoint parity,
admission). Nothing here reads an assessment value.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from cbp import family as FAM
from cbp import lock as LK
from cbp import run as R

ROLES = ("P*", "J*", "T*", "C_rate", "C_global", "Q")
SOURCE_L01 = ("U|JOINT|i8o64|l0.1", "U|SEQ-12|i8o64|l0.1", "U|SEQ-21|i8o64|l0.1")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prior_hash(D):
    tr = D["idx"]["DEFENSE_FIT"]
    c = np.bincount(np.asarray(D["sex"])[tr], minlength=2).astype(np.int64)
    return hashlib.sha256(c.tobytes()).hexdigest()


def _cfg(s):
    return (s or {}).get("config") or (s or {}).get("descriptive_config")


def scored_labels(S):
    """PROTOCOL.md section 12, fixed from the inner selection only: the six roles (or their registered fallbacks),
    U continuous and CLASS-ONLY, the source lambda 0.1 JOINT and both sequential lambda 0.1 controls, the strongest
    ordinary privacy winner without headroom, each family's headroom winner (or fallback), RAW-J, FARE (REF|F),
    no-fairness FARE (REF|F0) and LEACE (REF|E). Duplicates are scored once."""
    st, d = S["statuses"], S["diagnostics"]
    labels = [_cfg(st.get(x)) for x in ROLES]
    labels += ["SRC|U", R.config_id("CLASS")] + list(SOURCE_L01)
    labels.append(_cfg(d["ordinary_privacy_winner_no_headroom"]))
    labels += [_cfg(d["family_headroom_winners"][f]) for f in R.PRIVACY]
    labels += ["SRC|RAW-J_b0.3", "REF|F", "REF|F0", "REF|E"]
    return list(dict.fromkeys(x for x in labels if x))


def role_aliases(resolved):
    """Data-dependent aliases between role configurations (SEL-N1): every pair of roles resolved to one config."""
    out = {}
    for i, a in enumerate(ROLES):
        for b in ROLES[i + 1:]:
            if resolved.get(a) and resolved.get(a) == resolved.get(b):
                out[f"{a}=={b}"] = resolved[a]
    return out


def alias_of_by_role(aliases):
    """Slot-level aliases implied by role aliases (the slots are kept; prose never counts one comparison twice)."""
    by = {}
    claims = {c: (n, m) for c, (n, m) in FAM.CLAIMS.items()}
    for e in FAM.PRIMARY:
        if e["claim"] == "Q":
            continue
        for e2 in FAM.PRIMARY:
            if e2["claim"] in ("Q", e["claim"]) or e2["id"] >= e["id"] or e2["kind"] != e["kind"]:
                continue
            if e.get("task") != e2.get("task") or e.get("view") != e2.get("view"):
                continue
            n1, m1 = claims[e["claim"]]
            n2, m2 = claims[e2["claim"]]
            same_nom = n1 == n2 or f"{n2}=={n1}" in aliases or f"{n1}=={n2}" in aliases
            same_ref = e["kind"] not in ("coalition", "local") or m1 == m2 or f"{m2}=={m1}" in aliases or \
                f"{m1}=={m2}" in aliases
            if same_nom and same_ref and e["id"] not in by:
                by[e["id"]] = e2["id"]
    return by


def technical_validity():
    """Global technical validity: real-data controls (AUDIT_PRELOCK_CHECKS.json), reused-endpoint parity, admission."""
    out = {"ok": True, "failures": []}
    p = R.PKG / "AUDIT_PRELOCK_CHECKS.json"
    if not p.exists():
        out["failures"].append("AUDIT_PRELOCK_CHECKS.json missing")
    else:
        v = json.loads(p.read_text()).get("verdict", {})
        out["controls_verdict_sha256"] = sha(p)
        if not v.get("all_ok"):
            out["failures"].append({"controls": v.get("failures")})
    ep = R.RUN / "endpoint_parity.json"
    if not ep.exists():
        out["failures"].append("endpoint_parity.json missing")
    else:
        par = json.loads(ep.read_text())
        bad = [n for n, r in par.items() if not r.get("ok")]
        out["endpoint_parity"] = {"units": len(par), "failed": bad, "sha256": sha(ep)}
        if bad or len(par) != 3 * len(R.reused_ids()):
            out["failures"].append({"endpoint_parity": bad or f"{len(par)} units"})
    ad = R.PRIV / "admitted" / "ADMISSION_RECEIPT.json"
    if not ad.exists() or json.loads(ad.read_text()).get("verdict") != "ADMITTED":
        out["failures"].append("admission receipt missing or not ADMITTED")
    out["ok"] = not out["failures"]
    out["scope"] = "global: a failure here touches every claim (LABEL_TRUTH_TABLE.json failure_scope)"
    return out


def build():
    from osf import data as OD
    from cbp import data as DA
    S = json.loads((R.RUN / "selection.json").read_text())
    D = DA.load()
    labels = scored_labels(S)
    resolved = {x: _cfg(S["statuses"].get(x)) for x in ROLES}
    al = role_aliases(resolved)
    comp = R.composition_ids()
    lock = {"schema": "cbp-evaluation-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": LK.git("rev-parse", "HEAD"),
            "locks_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("*_LOCK.json"))
                             if p.name != "EVALUATION_LOCK.json"},
            "amendments_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("AMENDMENT_A*.json"))},
            "locked_code_files": LK.locked_files(LK.latest()),
            "selection_sha256": sha(R.RUN / "selection.json"), "selection_public_sha256": sha(R.PKG / "SELECTION.json"),
            "statuses": {x: {kk: v.get(kk) for kk in ("status", "config", "descriptive_config", "descriptive_only",
                                                      "winning_family", "config_family", "reason", "aliases")}
                         for x, v in S["statuses"].items()},
            "resolved": resolved, "role_aliases": al, "alias_of_by_role": alias_of_by_role(al),
            "scored_labels": labels, "technical_validity": technical_validity(),
            "sex_prior_defense_fit_sha256": prior_hash(D),
            "assessment_role": {"name": "OSF_DEVELOPMENT_ASSESSMENT", **OD.manifest(D)["OSF_DEVELOPMENT_ASSESSMENT"]},
            "attackers": {"slate": "cbp.audit: pinned FINAL slate + defence-aware readers on every primary contract; "
                                   "finite codes add cell-conditional readers on the exact token identity / tuple",
                          "fit": "AUDIT_FIT", "selection": "INNER_SELECTION (AUC primary, CE separate)",
                          "refits": [0, 1, 2], "coalition_bank": "pair + ignore-recipient-1 + ignore-recipient-2",
                          "composed_source_readers": "SRC|U composes over the 38 registered composition codes of the "
                                                     "same seed (27 cbp codes + 11 composition-only qpc maps), which "
                                                     "contain every frozen composed winner"},
            "endpoints": {"primary": [e["id"] for e in FAM.PRIMARY], "z": FAM.Z_PRIMARY, "B": FAM.B,
                          "boot_seed": FAM.BOOT_SEED, "size": FAM.PRIMARY_SIZE},
            "seeds": {}}
    for k in R.SEEDS:
        score, files = {}, {}
        for c in labels:
            u = R.unit_for(k, c)
            score[c] = {"kind": R.parse_id(c)["kind"], "unit": u, "cid": c}
            for n in (u, f"inner__{u}"):
                files[n] = json.loads((R.U(n) / "COMPLETE.json").read_text())["files"]
        for n in [f"tea__s{k}__U"] + [R.unit_for(k, c) for c in comp]:
            files.setdefault(n, json.loads((R.U(n) / "COMPLETE.json").read_text())["files"])
        lock["seeds"][str(k)] = {"score": score, "unit_file_sha256": files, "u_label": "SRC|U",
                                 "composed_policies": [R.unit_for(k, c) for c in comp]}
    return R._finite(lock)


if __name__ == "__main__":
    L = build()
    Path(sys.argv[1]).write_text(json.dumps(L, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"statuses": {x: [v["status"], v.get("config") or v.get("descriptive_config")]
                                   for x, v in L["statuses"].items()},
                      "n_labels": len(L["scored_labels"]), "role_aliases": L["role_aliases"],
                      "technical_validity": L["technical_validity"]["ok"]}, indent=1))
