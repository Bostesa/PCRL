"""EVALUATION_LOCK.json writer for lcr (committed and pushed before the single assessment opening).

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m lcr.eval_lock results/pcrl_learned_decoder_constrained_release_v1/EVALUATION_LOCK.json

Adapted from cbp/eval_lock.py at 7f3ec67. Freezes the inner-only statuses and resolved roles (P*, N*, J*, T*, C*,
C_pair*, Q), the scored list (prompt section 13), every scored unit with the sha256 of every unit file (release, decoder,
policy, inner), the composed source-reader bank (all 83 registered codes of each seed, containing every frozen composed
winner), the SEX prior hash, the assessment role manifest, attackers, lock/amendment hashes, locked code, the 37
endpoints, data-dependent role aliases and the global technical-validity inputs. Nothing here reads an assessment value.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from lcr import family as FAM
from lcr import lock as LK
from lcr import run as R

ROLES = ("P*", "N*", "J*", "T*", "C*", "C_pair*", "Q")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prior_hash(D):
    tr = D["idx"]["DEFENSE_FIT"]
    c = np.bincount(np.asarray(D["sex"])[tr], minlength=2).astype(np.int64)
    return hashlib.sha256(c.tobytes()).hexdigest()


def _cfg(s):
    return (s or {}).get("config") or (s or {}).get("descriptive_config")


def scored_labels(S):
    """PROTOCOL.md section 13 (prompt section 13), fixed from the inner selection only: the seven roles (or their
    registered fallbacks); U continuous and decisions alone; the best D1 fixed-map privacy control and its paired D0
    release; the best weighted privacy control; C-TASK; the five constrained arms; RAW-J, FARE, F0 and LEACE.
    Duplicates (exact config aliases) are scored once; every role mapping is kept."""
    st, d = S["statuses"], S["diagnostics"]
    labels = [_cfg(st.get(x)) for x in ROLES]
    labels += ["SRC|U", R.d0_id("CLASS")]
    labels += [_cfg(d["best_d1_fixed_privacy"]), d["best_d1_fixed_privacy"].get("paired_d0")]
    labels += [_cfg(d["best_weighted_privacy"]), R.ctask_id()] + R.constrained_ids()
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
    """Global technical validity: real-data controls (AUDIT_PRELOCK_CHECKS.json), the registered fixture gate, admission."""
    out = {"ok": True, "failures": []}
    p = R.PKG / "AUDIT_PRELOCK_CHECKS.json"
    if not p.exists():
        out["failures"].append("AUDIT_PRELOCK_CHECKS.json missing")
    else:
        v = json.loads(p.read_text()).get("verdict", {})
        out["controls_verdict_sha256"] = sha(p)
        if not v.get("all_ok"):
            out["failures"].append({"controls": v.get("failures")})
    fx = R.PKG / "FIXTURE_GATE.json"
    if not fx.exists():
        out["failures"].append("FIXTURE_GATE.json missing")
    else:
        gate = json.loads(fx.read_text())
        out["fixture_gate"] = {"verdict": gate.get("verdict"), "sha256": sha(fx)}
        if gate.get("verdict") not in ("GATE_MET",):
            out["failures"].append({"fixture_gate": gate.get("verdict")})
    sel = R.RUN / "selection.json"
    iv = json.loads(sel.read_text()).get("inner_validation") if sel.exists() else None
    out["inner_validation"] = iv
    if not (iv or {}).get("ok"):
        out["failures"].append({"inner_validation": iv})
    ad = R.PRIV / "admitted" / "ADMISSION_RECEIPT.json"
    if not ad.exists() or json.loads(ad.read_text()).get("verdict") != "ADMITTED":
        out["failures"].append("admission receipt missing or not ADMITTED")
    out["ok"] = not out["failures"]
    out["scope"] = "global: a failure here touches every claim (LABEL_TRUTH_TABLE.json failure_scope)"
    return out


def build():
    from osf import data as OD
    from lcr import data as DA
    S = json.loads((R.RUN / "selection.json").read_text())
    D = DA.load()
    labels = scored_labels(S)
    resolved = {x: _cfg(S["statuses"].get(x)) for x in ROLES}
    al = role_aliases(resolved)
    comp = R.code_ids()
    lock = {"schema": "lcr-evaluation-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": LK.git("rev-parse", "HEAD"),
            "locks_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("*_LOCK.json"))
                             if p.name != "EVALUATION_LOCK.json"},
            "amendments_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("AMENDMENT_A*.json"))},
            "locked_code_files": LK.locked_files(LK.latest()),
            "selection_sha256": sha(R.RUN / "selection.json"), "selection_public_sha256": sha(R.PKG / "SELECTION.json"),
            "statuses": {x: {kk: v.get(kk) for kk in ("status", "config", "descriptive_config", "descriptive_only",
                                                      "winning", "config_arm", "reason", "aliases",
                                                      "fallback_rank_status", "missing_guards")}
                         for x, v in S["statuses"].items()},
            "resolved": resolved, "role_aliases": al, "alias_of_by_role": alias_of_by_role(al),
            "decoder_ablation_pair": [_cfg(S["diagnostics"]["best_d1_fixed_privacy"]),
                                      S["diagnostics"]["best_d1_fixed_privacy"].get("paired_d0")],
            "scored_labels": labels, "technical_validity": technical_validity(),
            "sex_prior_defense_fit_sha256": prior_hash(D),
            "assessment_role": {"name": "OSF_DEVELOPMENT_ASSESSMENT", **OD.manifest(D)["OSF_DEVELOPMENT_ASSESSMENT"]},
            "attackers": {"slate": "lcr.audit: pinned FINAL slate + defence-aware readers on every primary contract; "
                                   "finite codes add cell-conditional readers on the exact token identity / tuple",
                          "fit": "AUDIT_FIT", "selection": "INNER_SELECTION (AUC primary, CE separate)",
                          "refits": [0, 1, 2], "coalition_bank": "pair + ignore-recipient-1 + ignore-recipient-2",
                          "composed_source_readers": "SRC|U composes, at the assessment, over every frozen source-winning "
                                                     "composed reader of the inner bank (all 83 registered codes per seed "
                                                     "were composed at the inner stage) plus every scored code; codes "
                                                     "that lost every inner comparison are not re-scored (exact: the final "
                                                     "audit re-selects on the same seed-0 inner fits)"},
            "endpoints": {"primary": [e["id"] for e in FAM.PRIMARY], "z": FAM.Z_PRIMARY, "B": FAM.B,
                          "boot_seed": FAM.BOOT_SEED, "size": FAM.PRIMARY_SIZE},
            "seeds": {}}
    for k in R.SEEDS:
        score, files = {}, {}
        for c in labels:
            u = R.unit_for(k, c)
            score[c] = {"kind": R.parse_id(c)["kind"], "unit": u, "cid": c}
            for n in (u, R.inner_name(k, c)):
                files[n] = json.loads((R.U(n) / "COMPLETE.json").read_text())["files"]
        for n in [f"tea__s{k}__U"] + [R.unit_for(k, c) for c in comp]:
            files.setdefault(n, json.loads((R.U(n) / "COMPLETE.json").read_text())["files"])
        from lcr import audit as AU
        frozen = list(AU.composed_freeze_list(k, "U"))
        scored_codes = [c for c in labels if R.parse_id(c)["kind"] == "policy"]
        comp_k = [c for c in comp if c in set(frozen) | set(scored_codes)]
        for c in comp_k:
            files.setdefault(R.unit_for(k, c), json.loads((R.U(R.unit_for(k, c)) / "COMPLETE.json").read_text())["files"])
        lock["seeds"][str(k)] = {"score": score, "unit_file_sha256": files, "u_label": "SRC|U",
                                 "composed_frozen_winners": frozen,
                                 "composed_policies": [R.unit_for(k, c) for c in comp_k]}
    return R._finite(lock)


if __name__ == "__main__":
    L = build()
    Path(sys.argv[1]).write_text(json.dumps(L, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"statuses": {x: [v["status"], v.get("config") or v.get("descriptive_config")]
                                   for x, v in L["statuses"].items()},
                      "n_labels": len(L["scored_labels"]), "role_aliases": L["role_aliases"],
                      "technical_validity": L["technical_validity"]["ok"]}, indent=1))
