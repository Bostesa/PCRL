"""EVALUATION_LOCK of the held-out calibration study (hcal; role A; prompt section 10 / 11 E).

Built after calibration, inner audits, control checks, selection, the pre-lock replay and role E's independent inner
replay. Binds every nominee / comparator / Ucal* / diagnostic binding, the scored releases, the frozen attack winners
(per seed and partition key: view, criterion, bank, attacker, fit role, stored-prediction key, view fingerprint), every
unit file the assessment reads (COMPLETE.json file maps), decoder-table and frozen-bank hashes, the software hashes
(every code file), role hashes, thresholds and endpoint definitions. No science code changes after it is pushed; only
hcal.assess may then unseal (hcal.data gate).

    PYTHONPATH=. <python> -m hcal.eval_lock write
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

from hcal import family as FAM
from hcal import ids as I
from hcal import lock as LK
from hcal import select as SEL

LOCK_REL = f"{I.REL}/EVALUATION_LOCK.json"
E_INNER = I.PKG / "verification" / "E_INNER_REPLAY.json"


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scored_releases(sel, ucal_rid):
    out = [I.U_ID, f"{I.U_ID}|H-GLOBAL-TEMP", f"{I.U_ID}|H-CLASS-TEMP"]
    for role in ("T*", "P*"):
        rid = (sel.get(role) or {}).get("release")
        if rid:
            out.append(rid)
    for p in I.DIAGNOSTIC_PARTITIONS:
        out += [I.release_id(p, d) for d in I.decoders_of(p)]
    assert ucal_rid in out
    return list(dict.fromkeys(out))


def attack_keys(scored):
    keys = []
    for rid in scored:
        keys.append("SRC|U" if rid.startswith(I.U_ID) else I.parse_release(rid)[0])
    return list(dict.fromkeys(keys))


def state(role_rec):
    st = (role_rec or {}).get("status")
    if st == "NOMINEE":
        return "ELIGIBLE"
    if st in ("TECHNICAL_FAILURE",):
        return "TECHNICAL_FAILURE"
    return "NO_ELIGIBLE"


def build():
    from hcal import run as R
    from hcal import stages as ST
    sel = json.loads((R.RUN / "selection.json").read_text())
    plan = json.loads((R.RUN / "audit_plan.json").read_text())
    ctl = json.loads((R.RUN / "controls.json").read_text())
    rep = json.loads((R.RUN / "replay.json").read_text())
    eng = json.loads((I.PKG / "ENGINEERING_RESULT.json").read_text())
    e_inner = json.loads(E_INNER.read_text()) if E_INNER.exists() else None
    adm = R.admission()
    ucal = sel["ucal_star"]["release"]
    scored = scored_releases(sel, ucal)
    keys = attack_keys(scored)
    seeds = {}
    for k in I.SEEDS:
        units, winners, banks = {}, {}, {}
        need = [ST.util_name(k), ST.calu_name(k)]
        for key in keys:
            need.append(ST.com_name(k, key))
            com = R.rec(ST.com_name(k, key))["recovery"]
            winners[key] = {w: {c: com["winner_detail"][w][c] for c in ("auc", "ce")} for w in ("v1", "v2", "pair")}
            for w in winners[key].values():
                for d in w.values():
                    if d["kind"] == "fresh":
                        need.append(d["unit"])
            if key != "SRC|U":
                need.append(ST.cal_name(k, key))
                banks[key] = adm["bank"][f"{k}|{key}"]["bank_sha256"]
                need.append(ST.fam_name(k, key))
        for u in dict.fromkeys(need):
            units[u] = json.loads((R.U(u) / "COMPLETE.json").read_text())["files"]
        seeds[str(k)] = {"unit_file_sha256": units, "winners": winners, "bank_sha256": banks}
    pres = {}
    for k in I.SEEDS:
        rel = R.rec(ST.util_name(k))["releases"]
        pres[str(k)] = {"all_registered_releases_preserved": all(all(v["preserved"].values()) for v in rel.values()),
                        "scored_preserved": {r: rel[r]["preserved"] for r in scored},
                        "all_finite": all(v["finite"] for v in rel.values())}
    tv = {"engineering_gate": eng.get("verdict"), "engineering_result_sha256": _sha(I.PKG / "ENGINEERING_RESULT.json"),
          "admission": adm.get("verdict"), "controls_all_ok": SEL.controls_all_ok(ctl),
          "pre_lock_replay_all_ok": bool(rep.get("all_ok")),
          "independent_inner_replay": (e_inner or {}).get("verdict"),
          "decision_preservation_all_seeds": all(v["all_registered_releases_preserved"] for v in pres.values())}
    tv["ok"] = (tv["engineering_gate"] == "ENGINEERING_READY" and tv["admission"] == "ADMITTED" and tv["controls_all_ok"]
                and tv["pre_lock_replay_all_ok"] and tv["independent_inner_replay"] == "PASS"
                and tv["decision_preservation_all_seeds"])
    D = None
    from hcal import data as HD
    D = HD.load()
    import numpy as np
    a = np.asarray(D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"])
    rid = np.asarray(D["row_id"])[a]
    lock = {"schema": "hcal-evaluation-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": LK.git("rev-parse", "HEAD"),
            "locks_sha256": {n: _sha(I.PKG / f"{n}.json") for n in LK.ORDER if (I.PKG / f"{n}.json").exists()},
            "amendments_sha256": {p.name: _sha(p) for p in sorted(I.PKG.glob("AMENDMENT_A*.json"))},
            "locked_code_files": LK.code_files(),
            "selection_sha256": _sha(R.RUN / "selection.json"), "audit_plan_sha256": _sha(R.RUN / "audit_plan.json"),
            "controls_sha256": _sha(R.RUN / "controls.json"), "replay_sha256": _sha(R.RUN / "replay.json"),
            "independent_inner_replay_sha256": _sha(E_INNER) if E_INNER.exists() else None,
            "resolved": {"P*": sel["P*"]["release"], "T*": sel["T*"]["release"], "Ucal*": ucal, "U0": I.U_ID},
            "statuses": {"P*": {"state": state(sel["P*"]), "status": sel["P*"]["status"],
                                "release": sel["P*"]["release"], "disclosures": sel["P*"].get("disclosures")},
                         "T*": {"state": state(sel["T*"]), "status": sel["T*"]["status"],
                                "release": sel["T*"]["release"]},
                         "Ucal*": {"family": sel["ucal_star"]["family"], "release": ucal}},
            "scored_releases": scored, "attack_keys": keys, "audited_partitions": plan["audited"],
            "role_manifest_sha256": _sha(I.PKG / "ROLE_MANIFEST.json"),
            "assessment_role": {"rows": int(a.size), "groups": int(np.unique(np.asarray(D["unit"])[a]).size),
                                "row_id_sha256": HD.sha_arr(rid)},
            "family": {"size": FAM.PRIMARY_SIZE, "z": FAM.Z_PRIMARY, "B": FAM.B, "bootstrap_seed": FAM.BOOT_SEED,
                       "slots": FAM.PRIMARY},
            "technical_validity": tv, "decision_preservation_receipt": pres, "seeds": seeds,
            "statement": LK.STATEMENT}
    return lock


def write():
    lock = build()
    (I.PKG / "EVALUATION_LOCK.json").write_text(json.dumps(lock, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"resolved": lock["resolved"], "technical_validity": lock["technical_validity"],
                      "scored": len(lock["scored_releases"]), "keys": lock["attack_keys"]}, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        write()
