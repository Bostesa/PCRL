"""EVALUATION_LOCK.json writer (committed and pushed before the single consolidated assessment; no manual substitution).

    PYTHONPATH=. ~/PCRL/.venv/bin/python -m osf.eval_lock results/pcrl_online_strength_frontier_v1/EVALUATION_LOCK.json

Freezes: the inner-only selection (statuses and global configurations of L*, C*, N*, R*, descriptive fallbacks, the
deployable best model), every scored label per seed (all locked bank configurations + E, F, F0) with the unit(s) behind
it and the sha256 of every unit file, the exact scalar grid, heads (inside the unit hashes), a hash of the
OSF_DEFENSE_FIT SEX prior counts (osf.infer recomputes it), attacker settings, the assessment role manifest (counts and
row-id hash), lock/amendment hashes, locked code files and the endpoint lists. The scored grid is the descriptive
frontier; it is never a post-assessment selection bank.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from osf import data as DA
from osf import family as FAM
from osf import lock as LK
from osf import run as R
from osf import train as T


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prior_hash(D):
    tr = D["idx"]["DEFENSE_FIT"]
    c = np.bincount(D["sex"][tr], minlength=2).astype(np.int64)
    return hashlib.sha256(c.tobytes()).hexdigest()


def resolve(statuses, x):
    s = statuses[x]
    return s.get("config") or s.get("descriptive_config")


def build():
    from osf import baselines as BL
    sel_path = R.RUN / "selection.json"
    S = json.loads(sel_path.read_text())
    D = DA.load()
    kind, ids = R.locked_bank()
    statuses = {x: {kk: v.get(kk) for kk in ("status", "config", "descriptive_config", "guards_used", "missing_guards")}
                for x, v in S["statuses"].items()}
    lock = {"schema": "osf-evaluation-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": LK.git("rev-parse", "HEAD"),
            "locks_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("*_LOCK.json")) if p.name != "EVALUATION_LOCK.json"},
            "amendments_sha256": {p.name: sha(p) for p in sorted(LK.PKG.glob("AMENDMENT_A*.json"))},
            "locked_code_files": LK.locked_files(LK.latest()),
            "selection_sha256": sha(sel_path), "selection_public_sha256": sha(R.PKG / "SELECTION.json"),
            "statuses": statuses, "resolved": {x: resolve(S["statuses"], x) for x in statuses},
            "deployable_best": S["deployable_best"], "U_valid": S["U_valid"],
            "bank": {"kind": kind, "configs": ids, "params": {c: T.parse_id(c) for c in ids}},
            "sex_prior_defense_fit_sha256": prior_hash(D),
            "assessment_role": {"name": "OSF_DEVELOPMENT_ASSESSMENT", **DA.manifest(D)["OSF_DEVELOPMENT_ASSESSMENT"]},
            "attackers": {"slate": "osf.audit (= pinned smf.audit final slate: LR grid, MLPs, HGB grid, defense-aware "
                                   "float64 scaled reader DA_LR/DA_MLP; finite releases + cell-conditional)",
                          "fit": "AUDIT_FIT", "selection": "INNER_SELECTION: AUC (fixed orientation) primary, CE separate",
                          "refits": [0, 1, 2], "coalition_bank": "own pair + ignore-recipient-1 + ignore-recipient-2"},
            "endpoints": {"primary": [e["id"] for e in FAM.PRIMARY], "secondary": [e["id"] for e in FAM.SECONDARY],
                          "z_primary": FAM.Z_PRIMARY, "z_secondary": FAM.Z_SECONDARY, "B": FAM.B,
                          "boot_seed": FAM.BOOT_SEED, "primary_size": FAM.PRIMARY_SIZE,
                          "secondary_size": FAM.SECONDARY_SIZE},
            "seeds": {}}
    for k in R.SEEDS:
        score = {cid: {"kind": "release", "unit": R.rel_name(k, cid)} for cid in ids}
        rc = BL.reference_candidates(k)
        for lab in ("E", "F", "F0"):
            c = rc[lab]
            score[lab] = ({"kind": "fare", "units": list(c["units"])} if c.get("units") else
                          {"kind": "release", "unit": c["unit"]})
        files = {}
        for spec in score.values():
            for u in ([spec["unit"]] if spec["kind"] == "release" else spec["units"]):
                files[u] = json.loads((R.U(u) / "COMPLETE.json").read_text())["files"]
        lock["seeds"][str(k)] = {"score": score, "unit_file_sha256": files,
                                 "reference_status": {lab: rc[lab].get("status") for lab in ("E", "F", "F0")}}
    needed = {resolve(S["statuses"], x) for x in statuses} - {None}
    assert needed <= set(lock["seeds"]["0"]["score"]), needed - set(lock["seeds"]["0"]["score"])
    return lock


if __name__ == "__main__":
    L = build()
    Path(sys.argv[1]).write_text(json.dumps(L, indent=1, default=float) + "\n")
    print(json.dumps({"statuses": L["statuses"], "deployable_best": L["deployable_best"]}, indent=1))
