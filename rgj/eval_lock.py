"""EVALUATION_LOCK.json writer (pushed before the single DEVELOPMENT_ASSESSMENT scoring; no manual substitution).

    ~/PCRL/.venv/bin/python -m rgj.eval_lock results/pcrl_refreshed_guarded_joint_v1/EVALUATION_LOCK.json

Per seed: the frozen unit (or FARE unit pair) behind every scored label of rgj.family.SCORED with its selection status,
the per-seed comparator C*, the sha256 of every file of every scored unit (model, heads, critics/transforms, release),
the DEFENSE_FIT SEX prior, attacker settings, code-lock hashes and the exact endpoint lists.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from rgj import data as DA
from rgj import family as FAM
from rgj import lock as LK
from rgj import run as R


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def unit_hashes(name):
    return json.loads((R.U(name) / "COMPLETE.json").read_text())["files"]


def build():
    SB, SC = R.selB(), json.loads((R.RUN / "selection_C.json").read_text())
    D = DA.load()
    tr = D["idx"]["DEFENSE_FIT"]
    lock = {"schema": "rgj-evaluation-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": LK.git("rev-parse", "HEAD"),
            "code_lock_sha256": sha(LK.PKG / "CODE_LOCK.json"),
            "code_lock_amendments_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("CODE_LOCK_A*.json"))},
            "locked_code_files": LK.locked_files(json.loads((LK.PKG / "CODE_LOCK.json").read_text())),
            "selection_B_sha256": sha(R.RUN / "selection_B.json"), "selection_C_sha256": sha(R.RUN / "selection_C.json"),
            "sex_prior_defense_fit": (np.bincount(D["sex"][tr], minlength=2) / len(tr)).tolist(),
            "assessment_role": {"name": "DEVELOPMENT_ASSESSMENT", "rows": int(len(D["idx"]["DEVELOPMENT_ASSESSMENT"])),
                                "row_id_sha256": DA.manifest(D)["DEVELOPMENT_ASSESSMENT"]["row_id_sha256"]},
            "attackers": {"final_slate": "rgj.audit.final_slate (LR C grid, MLP x4, HGB x4, DA canonical MLP; finite: +CC)",
                          "fit": "AUDIT_FIT", "selection": "INNER_SELECTION: primary by AUC (fixed orientation), "
                                                           "proper-loss by cross-entropy", "refits": [0, 1, 2],
                          "coalition_bank": "own pair attackers + ignore-recipient-1 + ignore-recipient-2"},
            "endpoints": {"primary": [e["id"] for e in FAM.PRIMARY], "secondary": [e["id"] for e in FAM.SECONDARY],
                          "z_primary": FAM.Z_PRIMARY, "z_secondary": FAM.Z_SECONDARY, "B": FAM.B,
                          "boot_seed": FAM.BOOT_SEED},
            "seeds": {}}
    for k in R.SEEDS:
        s = str(k)
        sc = SC[s]
        if not sc.get("valid_reference", False):
            lock["seeds"][s] = {"valid_reference": False, "score": {}, "status": {}}
            continue
        arms = sc["arms"]
        score, status = {}, {}
        for lab in ("J-G", "L-G", "J-R", "J-O", "L-R", "L-O", "U", "E"):
            a = arms[lab]
            score[lab] = {"kind": "neural", "unit": a["unit"]}
            status[lab] = a["status"]
        score["U"] = {"kind": "neural", "unit": sc["U"]["unit"]}
        status["U"] = arms["U"]["status"]
        for lab in ("F", "F0"):
            score[lab] = {"kind": "fare", "units": arms[lab]["units"]}
            status[lab] = arms[lab]["status"]
        for lab, arm in (("J-G@fx", "J-G"), ("J-R@fx", "J-R"), ("J-O@fx", "J-O")):
            score[lab] = {"kind": "neural", "unit": R.ck_name("C", k, arm, 0.1, 20)}
            status[lab] = "FIXED_BETA_COMPONENT (secondary)"
        files = {}
        for lab, spec in score.items():
            for u in ([spec["unit"]] if spec["kind"] == "neural" else spec["units"]):
                files[u] = unit_hashes(u)
        lock["seeds"][s] = {"valid_reference": True, "score": score, "status": status,
                            "comparator": sc["comparator"], "L-R_reference": sc["L-R_reference"],
                            "stage_B_L-R": {kk: SB[s]["L-R"].get(kk) for kk in ("status", "unit", "beta", "epoch")},
                            "calibrated_budgets": R.rec(f"calib__s{k}")["c"], "unit_file_sha256": files,
                            "J-G_nomination_guard": arms["J-G"].get("nomination_guard")}
    assert set(FAM.SCORED) <= set(next(iter(lock["seeds"].values()))["score"]) or not lock["seeds"]
    return lock


if __name__ == "__main__":
    L = build()
    Path(sys.argv[1]).write_text(json.dumps(L, indent=1, default=float) + "\n")
    print({k: {lab: v["status"].get(lab) for lab in ("J-G", "L-G")} | {"C*": v.get("comparator", {}).get("arm")}
           for k, v in L["seeds"].items()})
