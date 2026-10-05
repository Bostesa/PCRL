"""EVALUATION_LOCK.json writer (pushed before the single NEW_DEVELOPMENT_ASSESSMENT scoring; no manual substitution).

    PYTHONPATH=. ~/PCRL/.venv/bin/python -m smf.eval_lock results/pcrl_strength_matched_feedback_v1/EVALUATION_LOCK.json

Per seed: the frozen unit (or FARE pair) behind every scored label (smf.family.SCORED) with its status, C*, the
J-F nomination guards, controller targets, the sha256 of every file of every scored unit, a hash of the NEW_DEFENSE_FIT SEX
prior counts, attacker settings, lock/amendment hashes and the exact endpoint lists.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from smf import data as DA
from smf import family as FAM
from smf import lock as LK
from smf import run as R


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prior_hash(D):
    tr = D["idx"]["DEFENSE_FIT"]
    c = np.bincount(D["sex"][tr], minlength=2).astype(np.int64)
    return hashlib.sha256(c.tobytes()).hexdigest()


def build():
    SA = json.loads((R.RUN / "selection_A.json").read_text())
    SB = json.loads((R.RUN / "selection_B.json").read_text())
    D = DA.load()
    tr = D["idx"]["DEFENSE_FIT"]
    lock = {"schema": "smf-evaluation-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": LK.git("rev-parse", "HEAD"),
            "locks_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("*_LOCK.json"))},
            "amendments_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("AMENDMENT_A*.json"))},
            "locked_code_files": LK.locked_files(LK.latest()),
            "selection_A_sha256": sha(R.RUN / "selection_A.json"), "selection_B_sha256": sha(R.RUN / "selection_B.json"),
            "phase_A_schedule": SA["schedule"],
            # only a hash: the full-precision NEW_DEFENSE_FIT prior minus the refreshed study's published prior would
            # reveal the sealed assessment's SEX share (provenance review); smf.infer recomputes and checks it
            "sex_prior_defense_fit_sha256": prior_hash(D),
            "assessment_role": {"name": "NEW_DEVELOPMENT_ASSESSMENT", **DA.manifest(D)["NEW_DEVELOPMENT_ASSESSMENT"]},
            "attackers": {"slate": "smf.audit.final_slate (LR grid, MLPs, HGB grid, defense-aware float64 scaled reader)",
                          "fit": "AUDIT_FIT", "selection": "INNER_SELECTION: AUC (fixed orientation) primary, CE separate",
                          "refits": [0, 1, 2], "coalition_bank": "own pair + ignore-recipient-1 + ignore-recipient-2"},
            "endpoints": {"primary": [e["id"] for e in FAM.PRIMARY], "secondary": [e["id"] for e in FAM.SECONDARY],
                          "z_primary": FAM.Z_PRIMARY, "z_secondary": FAM.Z_SECONDARY, "B": FAM.B, "boot_seed": FAM.BOOT_SEED},
            "seeds": {}}
    sched = SA["schedule"]["selected"]
    for k in R.SEEDS:
        s = str(k)
        sb = SB[s]
        if not sb.get("valid_reference"):
            lock["seeds"][s] = {"valid_reference": False, "score": {}, "status": {}}
            continue
        arms = sb["arms"]
        score, status = {}, {}
        for lab in ("J-F", "L-F", "J-N", "L-N", "RAW-J", "RAW-L", "E"):
            score[lab], status[lab] = {"kind": "neural", "unit": arms[lab]["unit"]}, arms[lab]["status"]
        score["U"], status["U"] = {"kind": "neural", "unit": sb["U"]["unit"]}, arms["U"]["status"]
        for lab in ("F", "F0"):
            score[lab], status[lab] = {"kind": "fare", "units": arms[lab]["units"]}, arms[lab]["status"]
        for lab, sc in (("A-ONLINE@r0.75", "ONLINE"), ("A-REFRESHED@r0.75", "REFRESHED"), ("A-MATCHED@r0.75", "ONLINE_MATCHED")):
            score[lab], status[lab] = {"kind": "neural", "unit": R.a_name(k, "NJ", sc, 0.75)}, "FIXED_RHO_COMPONENT"
        for arm in ("J-F", "L-F", "J-N", "L-N"):
            score[f"{arm}@r0.75"] = {"kind": "neural", "unit": R.b_name(k, arm, 0.75)}
            status[f"{arm}@r0.75"] = "FIXED_RHO_COMPONENT"
        files = {}
        for spec in score.values():
            for u in ([spec["unit"]] if spec["kind"] == "neural" else spec["units"]):
                files[u] = json.loads((R.U(u) / "COMPLETE.json").read_text())["files"]
        lock["seeds"][s] = {"valid_reference": True, "reference": sb["reference"], "schedule": sched, "score": score,
                            "status": status, "comparator": sb["comparator"],
                            "J-F_nomination_guards": arms["J-F"].get("nomination_guards"),
                            "controller_targets": R.rec(f"calib__s{k}")["b"], "unit_file_sha256": files}
        assert set(FAM.SCORED) <= set(score), set(FAM.SCORED) - set(score)
    return lock


if __name__ == "__main__":
    L = build()
    Path(sys.argv[1]).write_text(json.dumps(L, indent=1, default=float) + "\n")
    print({k: {lab: v["status"].get(lab) for lab in ("J-F", "L-F")} | {"C*": v.get("comparator", {}).get("arm")}
           for k, v in L["seeds"].items()})
