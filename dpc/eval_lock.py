"""EVALUATION_LOCK.json writer (committed and pushed before the single assessment opening; no manual substitution).

    PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.eval_lock results/pcrl_decision_preserving_compression_v1/EVALUATION_LOCK.json

Freezes: inner-only selection statuses and resolved configurations (J*, P*, T*, C_global, C_match), the scored label
set (dpc.select.scored_labels), every scored unit with the sha256 of every unit file (policies: policy + prototypes +
release; teachers; references), the composed source-reader policies, a hash of the OSF_DEFENSE_FIT SEX prior counts,
the assessment role manifest, attacker settings, lock/amendment hashes and locked code files, and the endpoint list.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from dpc import data as DA
from dpc import family as FAM
from dpc import lock as LK
from dpc import run as R


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prior_hash(D):
    tr = D["idx"]["DEFENSE_FIT"]
    c = np.bincount(D["sex"][tr], minlength=2).astype(np.int64)
    return hashlib.sha256(c.tobytes()).hexdigest()


def build():
    S = json.loads((R.RUN / "selection.json").read_text())
    D = DA.load()
    statuses = {x: {kk: v.get(kk) for kk in ("status", "config", "descriptive_config", "missing_guards")}
                for x, v in S["statuses"].items()}
    resolved = {x: (s.get("config") or s.get("descriptive_config")) for x, s in S["statuses"].items()}
    labels = S["scored_labels"]
    composed = {t: [R.unit_for(0, c).replace("pol__s0__", "") for c in labels
                    if R.parse_id(c)["kind"] == "policy" and R.parse_id(c)["teacher"] == t] for t in R.TEACHERS}
    lock = {"schema": "dpc-evaluation-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": LK.git("rev-parse", "HEAD"),
            "locks_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("*_LOCK.json")) if p.name != "EVALUATION_LOCK.json"},
            "amendments_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("AMENDMENT_A*.json"))},
            "locked_code_files": LK.locked_files(LK.latest()),
            "selection_sha256": sha(R.RUN / "selection.json"), "selection_public_sha256": sha(R.PKG / "SELECTION.json"),
            "statuses": statuses, "resolved": resolved, "deployable_compact": S["deployable_compact"],
            "U_valid": S["U_valid"], "bank": R.locked_bank(), "scored_labels": labels,
            "sex_prior_defense_fit_sha256": prior_hash(D),
            "assessment_role": {"name": "OSF_DEVELOPMENT_ASSESSMENT", **DA.manifest(D)["OSF_DEVELOPMENT_ASSESSMENT"]},
            "attackers": {"slate": "dpc.audit: pinned smf.audit FINAL slate (LR x5, MLP x4, HGB x4, DA_LR, DA_MLP) on every "
                                   "primary contract; finite codes add cell-conditional readers {0.5, 1, 5} on the exact "
                                   "token identity / tuple with prior and validation-chosen pair fallbacks",
                          "fit": "AUDIT_FIT", "selection": "INNER_SELECTION (AUC primary, CE separate)", "refits": [0, 1, 2],
                          "coalition_bank": "pair + ignore-recipient-1 + ignore-recipient-2",
                          "composed_source_readers": "code readers of the locked scored policies of the same teacher/seed"},
            "endpoints": {"primary": [e["id"] for e in FAM.PRIMARY], "z": FAM.Z_PRIMARY, "B": FAM.B,
                          "boot_seed": FAM.BOOT_SEED, "size": FAM.PRIMARY_SIZE},
            "seeds": {}}
    for k in R.SEEDS:
        score, files = {}, {}
        for c in labels:
            u = R.unit_for(k, c)
            score[c] = {"kind": R.parse_id(c)["kind"], "unit": u, "cid": c}
            files[u] = json.loads((R.U(u) / "COMPLETE.json").read_text())["files"]
        lock["seeds"][str(k)] = {"score": score, "unit_file_sha256": files, "u_label": "SRC|U",
                                 "composed_policies": {t: [f"pol__s{k}__{x}" for x in composed[t]] for t in R.TEACHERS}}
    needed = {v for v in resolved.values() if v}
    assert needed <= set(labels), needed - set(labels)
    return lock


if __name__ == "__main__":
    L = build()
    Path(sys.argv[1]).write_text(json.dumps(L, indent=1, default=float) + "\n")
    print(json.dumps({"statuses": L["statuses"], "n_labels": len(L["scored_labels"])}, indent=1))
