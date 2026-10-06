"""EVALUATION_LOCK.json writer (committed and pushed before the single assessment opening; no manual substitution).

    PYTHONPATH=. <python> -m qpc.eval_lock results/pcrl_confidence_capacity_v1/EVALUATION_LOCK.json

Freezes: inner-only selection statuses and resolved configurations (J*, C_rate, C_global, P*, T*, Q*), the scored label
list (PROTOCOL.md section 13), every scored unit with the sha256 of every unit file, the composed source-reader
policies, a hash of the OSF_DEFENSE_FIT SEX prior counts, the assessment role manifest, attacker settings, lock and
amendment hashes, locked code files, the endpoint list, and the validity inputs of the label truth table
(stage_a_valid, gate_met, technical_validity from the real-data controls).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from qpc import family as FAM
from qpc import lock as LK
from qpc import run as R

PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prior_hash(D):
    tr = D["idx"]["DEFENSE_FIT"]
    c = np.bincount(np.asarray(D["sex"])[tr], minlength=2).astype(np.int64)
    return hashlib.sha256(c.tobytes()).hexdigest()


def scored_labels(S):
    """PROTOCOL.md section 13: fixed before the opening from the inner selection only."""
    res = S["resolved"]
    labels = [res[x] for x in ("J*", "C_rate", "C_global", "P*", "T*", "Q*") if res.get(x)]
    labels += R.stagea_ids()                                             # task-only capacity curve (all 8 rates)
    labels += ["SRC|U", "SRC|RAW-J_b0.3", R.config_id("CLASS"), "REF|E", "REF|F", "REF|F0"]
    j = res.get("J*")
    if j:                                                                # the J* cell (rate and lambda)
        p = R.parse_id(j)
        labels.append(R.config_id("FINE-TASK", p["m1"], p["m2"]))
        labels += [R.config_id(f, p["m1"], p["m2"], p["lam"]) for f in PRIVACY]
    return list(dict.fromkeys(labels))


def technical_validity():
    """Real-data controls verdict (qpc.audit writes AUDIT_PRELOCK_CHECKS.json); missing file -> invalid."""
    p = R.PKG / "AUDIT_PRELOCK_CHECKS.json"
    if not p.exists():
        return {"ok": False, "reason": "AUDIT_PRELOCK_CHECKS.json missing"}
    v = json.loads(p.read_text()).get("verdict", {})
    return {"ok": bool(v.get("all_ok")), "controls_verdict_sha256": sha(p), "failures": v.get("failures")}


def build():
    from osf import data as OD
    from qpc import data as DA
    S = json.loads((R.RUN / "selection.json").read_text())
    G = json.loads((R.PKG / "CAPACITY_GATE.json").read_text())
    D = DA.load()
    labels = scored_labels(S)
    comp = [c for c in R.scored_ids() if R.parse_id(c)["kind"] == "policy"]
    lock = {"schema": "qpc-evaluation-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": LK.git("rev-parse", "HEAD"),
            "locks_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("*_LOCK.json")) if p.name != "EVALUATION_LOCK.json"},
            "amendments_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("AMENDMENT_A*.json"))},
            "locked_code_files": LK.locked_files(LK.latest()),
            "selection_sha256": sha(R.RUN / "selection.json"), "selection_public_sha256": sha(R.PKG / "SELECTION.json"),
            "capacity_gate_sha256": sha(R.PKG / "CAPACITY_GATE.json"),
            "statuses": {x: {kk: v.get(kk) for kk in ("status", "config", "descriptive_config", "winning_family",
                                                      "cell", "reason")} for x, v in S["statuses"].items()},
            "resolved": S["resolved"], "scored_labels": labels,
            "stage_a_valid": True, "gate_met": G["decision"]["gate"] == "CAPACITY_GATE_MET",
            "technical_validity": technical_validity(),
            "sex_prior_defense_fit_sha256": prior_hash(D),
            "assessment_role": {"name": "OSF_DEVELOPMENT_ASSESSMENT", **OD.manifest(D)["OSF_DEVELOPMENT_ASSESSMENT"]},
            "attackers": {"slate": "qpc.audit: pinned smf.audit FINAL slate + defence-aware readers on every primary "
                                   "contract; finite codes add cell-conditional readers {0.5, 1, 5} on the exact token "
                                   "identity / tuple with prior and validation-chosen pair fallbacks",
                          "fit": "AUDIT_FIT", "selection": "INNER_SELECTION (AUC primary, CE separate)",
                          "refits": [0, 1, 2], "coalition_bank": "pair + ignore-recipient-1 + ignore-recipient-2",
                          "composed_source_readers": "SRC|U composes over EVERY fitted code of this study (same seed)"},
            "endpoints": {"primary": [e["id"] for e in FAM.PRIMARY], "z": FAM.Z_PRIMARY, "B": FAM.B,
                          "boot_seed": FAM.BOOT_SEED, "size": FAM.PRIMARY_SIZE},
            "seeds": {}}
    for k in R.SEEDS:
        score, files = {}, {}
        for c in labels:
            u = R.unit_for(k, c)
            score[c] = {"kind": R.parse_id(c)["kind"], "unit": u, "cid": c}
            files[u] = json.loads((R.U(u) / "COMPLETE.json").read_text())["files"]
            iu = f"inner__{u}"
            files[iu] = json.loads((R.U(iu) / "COMPLETE.json").read_text())["files"]
        lock["seeds"][str(k)] = {"score": score, "unit_file_sha256": files, "u_label": "SRC|U",
                                 "composed_policies": [R.unit_for(k, c) for c in comp]}
    return lock


if __name__ == "__main__":
    L = build()
    Path(sys.argv[1]).write_text(json.dumps(L, indent=1, default=float) + "\n")
    print(json.dumps({"statuses": L["statuses"], "n_labels": len(L["scored_labels"]),
                      "technical_validity": L["technical_validity"]}, indent=1))
