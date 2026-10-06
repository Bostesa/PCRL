"""Source admission for the confidence-budgeted privacy study (cbp). Lead-owned; locked in SOURCE_ADMISSION_LOCK.

Admits the hash-pinned qpc artifacts by VERIFIED COPY (no fit, no refit, nothing written into the qpc store):
  per seed k in {0, 1, 2}:
    tea__s{k}__U, tea__s{k}__RAW-J_b0.3            teacher outputs (row_id, p1, p2, d1, d2, c1, c2, r1, r2)
    ref__s{k}__{E, F, F0}                           LEACE / FARE / no-fairness FARE score-only releases
    fine__s{k}                                      fine partitions (income 32, occupation 128 per predicted class)
    pol__s{k}__U_DIRECT-TASK_i8o64, pol__s{k}__U_FINE-TASK_i8o64, pol__s{k}__U_CLASS_i1o1
    pol__s{k}__U_{LOCAL,SEQ-12,SEQ-21,JOINT}_i8o64_l{0.01,0.1}     the 24 reusable endpoint privacy maps
    pol__s{k}__U_DIRECT-TASK_i{4,8}o{8,16,32,64} (7 rates other than i8o64) and U_{LOCAL,...,JOINT}_i8o64_l1:
                                                    composition-only public maps for the U source bank (never candidates)
  admitted/rel__s{k}__{U,RAW-J_b0.3}                deployed teacher model.pt + heads (deployment and parity)
  inputs/deploy_input.npz, inputs/schema.json       the authorised 83-column deployment input (qpc reconstruction)

Every copied file must hash-match BOTH (i) the unit's own COMPLETE.json in the live qpc store and (ii) the independent
qpc same-device copy v2 SHA256SUMS (written and re-read uncached at the qpc closeout, 2026-10-06). Scored qpc units
must also match qpc's EVALUATION_LOCK unit_file_sha256 at the pinned source commit.

Parity (all bitwise):
  teachers: own forward application (pinned dpc.deploy.load_teacher / teacher_probs on D["X"]) of the admitted
            model.pt + heads reproduces p1, p2 and argmax decisions of tea__s{k}__U and tea__s{k}__RAW-J_b0.3;
  codes:    every admitted pol__ unit re-encoded from its policy.json and the admitted teacher (qpc.release) equals its
            stored release.npz on all rows (tokens, decoded probabilities, decisions, alphabets).
The objective-receipt parity of the reused endpoint maps (D, I terms recomputed from the deployed fitting-row release)
is checked in the fit stage by cbp.fit.endpoint_parity (FIT_LOCK), before any new fit.

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.admit plan       hash-only precheck (no data load, no copy)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
QPC = HOME / "PCRL_eval_cache_private" / "qpc_v1"
QPC_COPY = HOME / "PCRL_eval_cache_private" / "qpc_v1_local_copy_20261006_v2"
CBP = HOME / "PCRL_eval_cache_private" / "cbp_v1"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_confidence_budgeted_privacy_v1"
QPC_EVIDENCE = "9dd06da6b64e558e1c079f76e43982b60b327e63"
QPC_EVAL_LOCK_REL = "results/pcrl_confidence_capacity_v1/EVALUATION_LOCK.json"
SEEDS = (0, 1, 2)
TEACHERS = ("U", "RAW-J_b0.3")
REFS = ("E", "F", "F0")
ENDPOINT_LAMS = ("0.01", "0.1")
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")


def units_for(k):
    u = [f"tea__s{k}__{t}" for t in TEACHERS] + [f"ref__s{k}__{r}" for r in REFS] + [f"fine__s{k}"]
    u += [f"pol__s{k}__U_DIRECT-TASK_i8o64", f"pol__s{k}__U_FINE-TASK_i8o64", f"pol__s{k}__U_CLASS_i1o1"]
    u += [f"pol__s{k}__U_{f}_i8o64_l{lam}" for lam in ENDPOINT_LAMS for f in PRIVACY]
    u += [f"pol__s{k}__U_DIRECT-TASK_i{a}o{b}" for a in (4, 8) for b in (8, 16, 32, 64) if (a, b) != (8, 64)]
    u += [f"pol__s{k}__U_{f}_i8o64_l1" for f in PRIVACY]          # composition-only public maps (cbp.run extras)
    return u


def admitted_dirs():
    return [f"rel__s{k}__{t}" for k in SEEDS for t in TEACHERS]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def copy_sums():
    out = {}
    for line in (QPC_COPY / "SHA256SUMS").read_text().splitlines():
        h, rel = line.split("  ", 1)
        out[rel] = h
    return out


def qpc_eval_lock_files():
    r = subprocess.run(["git", "-C", str(WT), "show", f"{QPC_EVIDENCE}:{QPC_EVAL_LOCK_REL}"], capture_output=True)
    L = json.loads(r.stdout)
    files = {}
    for s in L["seeds"].values():
        files.update(s["unit_file_sha256"])
    return files


def check_dir(src: Path, rel_prefix: str, sums: dict, complete=True):
    """(file -> sha) after checking every file against COMPLETE.json (if present) and the copy SHA256SUMS."""
    got, bad = {}, []
    comp = json.loads((src / "COMPLETE.json").read_text())["files"] if complete and (src / "COMPLETE.json").exists() else None
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(src))
        h = sha_file(p)
        got[rel] = h
        if comp is not None and rel != "COMPLETE.json" and comp.get(rel) != h:
            bad.append(f"{src.name}/{rel}: COMPLETE.json mismatch")
        ref = sums.get(f"{rel_prefix}/{rel}")
        if ref is None:
            bad.append(f"{src.name}/{rel}: absent from the qpc copy SHA256SUMS")
        elif ref != h:
            bad.append(f"{src.name}/{rel}: differs from the qpc copy SHA256SUMS")
    return got, bad


def plan():
    sums, lockf = copy_sums(), qpc_eval_lock_files()
    out = {"units": {}, "admitted": {}, "inputs": {}, "bad": []}
    for k in SEEDS:
        for u in units_for(k):
            got, bad = check_dir(QPC / "run" / "units" / u, f"qpc_v1/run/units/{u}", sums)
            if u in lockf:
                bad += [f"{u}/{f}: differs from qpc EVALUATION_LOCK" for f, h in lockf[u].items() if got.get(f) != h]
            out["units"][u] = {"files": got, "in_qpc_evaluation_lock": u in lockf}
            out["bad"] += bad
    for a in admitted_dirs():
        got, bad = check_dir(QPC / "admitted" / a, f"qpc_v1/admitted/{a}", sums)
        out["admitted"][a] = got
        out["bad"] += bad
    got, bad = check_dir(QPC / "inputs", "qpc_v1/inputs", sums, complete=False)
    out["inputs"] = got
    out["bad"] += bad
    return out


def _copy(src: Path, dst: Path):
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        raise SystemExit(f"REFUSED: {dst.name} already admitted (resume skips verified units)")
    tmp.rename(dst)


def teacher_parity(k, t, D):
    from dpc import deploy as DD
    model, heads, msha = DD.load_teacher(CBP / "admitted" / f"rel__s{k}__{t}", seed=k)
    p = DD.teacher_probs(model, heads, D["X"])
    z = np.load(CBP / "run" / "units" / f"tea__s{k}__{t}" / "teacher.npz")
    res = {"model_sha256": msha}
    for i in (1, 2):
        res[f"p{i}_bitwise"] = bool(np.array_equal(p[i - 1], z[f"p{i}"]))
        res[f"d{i}_bitwise"] = bool(np.array_equal(p[i - 1].argmax(1), z[f"d{i}"]))
    res["row_id_equal"] = bool(np.array_equal(z["row_id"], D["row_id"]))
    res["ok"] = all(v for kk, v in res.items() if kk.endswith(("_bitwise", "_equal")))
    return res


def reencode_parity(unit, k):
    from qpc import release as RL
    pair = RL.load_policy(CBP / "run" / "units" / unit / "policy.json")
    T = np.load(CBP / "run" / "units" / f"tea__s{k}__U" / "teacher.npz")
    out = RL.release_arrays(pair, T["row_id"], T["p1"], T["d1"], T["p2"], T["d2"])
    z = np.load(CBP / "run" / "units" / unit / "release.npz")
    res = {x: bool(np.array_equal(np.asarray(out[x]), z[x])) for x in z.files}
    res["keys_equal"] = sorted(out) == sorted(z.files)
    res["decisions_equal_teacher"] = bool(np.array_equal(z["hard1"], T["d1"]) and np.array_equal(z["hard2"], T["d2"]))
    res["ok"] = all(res.values())
    return res


def run(D):
    """Verified copies + parity. Refuses (SystemExit) on any mismatch. Writes the private receipt and returns it."""
    pl = plan()
    if pl["bad"]:
        raise SystemExit("ADMISSION REFUSED (hash mismatch): " + "; ".join(pl["bad"][:10]))
    (CBP / "run" / "units").mkdir(parents=True, exist_ok=True)
    (CBP / "admitted").mkdir(parents=True, exist_ok=True)
    copied = []
    for k in SEEDS:
        for u in units_for(k):
            dst = CBP / "run" / "units" / u
            if not dst.exists():
                _copy(QPC / "run" / "units" / u, dst)
                copied.append(u)
            if {str(p.relative_to(dst)): sha_file(p) for p in dst.rglob("*") if p.is_file()} != pl["units"][u]["files"]:
                raise SystemExit(f"ADMISSION REFUSED: copy of {u} differs from its source")
    for a in admitted_dirs():
        dst = CBP / "admitted" / a
        if not dst.exists():
            _copy(QPC / "admitted" / a, dst)
            copied.append(a)
    if not (CBP / "inputs").exists():
        _copy(QPC / "inputs", CBP / "inputs")
    rec = {"schema": "cbp-admission-v1", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source_evidence": QPC_EVIDENCE, "hash_sources": ["qpc unit COMPLETE.json",
                                                             "qpc same-device copy v2 SHA256SUMS",
                                                             "qpc EVALUATION_LOCK unit_file_sha256 (scored units)"],
           "copied_now": copied, "teachers": {}, "codes": {}, "units": sorted(pl["units"])}
    for k in SEEDS:
        for t in TEACHERS:
            r = teacher_parity(k, t, D)
            rec["teachers"][f"{t}|{k}"] = r
            if not r["ok"]:
                raise SystemExit(f"ADMISSION REFUSED: teacher parity {t} seed {k}: {r}")
        for u in units_for(k):
            if u.startswith("pol__"):
                r = reencode_parity(u, k)
                rec["codes"][u] = r
                if not r["ok"]:
                    raise SystemExit(f"ADMISSION REFUSED: re-encode parity {u}: {r}")
    rec["verdict"] = "ADMITTED"
    (CBP / "admitted" / "ADMISSION_RECEIPT.json").write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n")
    return rec


def admission_record():
    p = CBP / "admitted" / "ADMISSION_RECEIPT.json"
    return json.loads(p.read_text()) if p.exists() else {}


if __name__ == "__main__":
    if sys.argv[1:] == ["plan"]:
        pl = plan()
        print(json.dumps({"units": len(pl["units"]), "admitted_dirs": len(pl["admitted"]), "inputs": sorted(pl["inputs"]),
                          "bad": pl["bad"][:20], "precheck_ok": not pl["bad"]}, indent=1))
