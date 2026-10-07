"""The single locked assessment opening on OSF_DEVELOPMENT_ASSESSMENT of the held-out calibration study (hcal; role A;
prompt section 10 / 11 F). This is the ONLY hcal module that indexes assessment rows or reads their labels.

GATE (adapted from lra/assess.py at 9762025). REFUSES unless results/pcrl_heldout_calibration_v1/EVALUATION_LOCK.json is
committed, unmodified, contained in origin/research/pcrl-heldout-calibration-v1 and byte-identical there; every file of
the lock's locked_code_files matches its hash now; every worktree module loaded by this process is locked; and the lock's
technical_validity is ok (ENGINEERING_READY bound and still true, admission ADMITTED, controls all_ok, pre-lock replay
all_ok, role E's independent inner replay PASS, decision preservation on every registered release and seed). Only then
does load_unsealed() call hcal.data.load(unseal=True) FROM THIS MODULE (hcal.data's caller gate) and check the unsealed
assessment role against the lock. There is no override flag.

PER SEED (jobs = scored releases + attack keys from the lock; resumable atomic units):
  oprob__s{k}__<release>  assess_row_id, assess_unit (exact-record group), sex, y_income, y_occ, const_class and the
                          release's prob1 / prob2 / hard1 / hard2 on the assessment rows (frozen tables; no refit)
  oatt__s{k}__<key>       P_auc_<view> / P_ce_<view> (3, n, 2): the frozen common-bank (or U composed) winners refit at
                          attacker seeds 0-2 on their OWN fit rows (fresh: ATTACK_FIT_NEW; legacy / lra: AUDIT_FIT) and
                          scored on the assessment rows; every refit must first reproduce its stored INNER_SELECTION
                          predictions bitwise (hcal.bank.refit_selected), and the winner must equal the lock's.

    <PRIVATE_CACHE>/hcal_v1/run/work.sh EVALUATION_LOCK assess [i/n]   (via python -m hcal.assess --evaluation-lock ...)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from hcal import ids as I

LOCK_NAME = "EVALUATION_LOCK.json"
LOCK_REL = f"{I.REL}/{LOCK_NAME}"
_OPENED = None


def _git(*args, text=True):
    return subprocess.run(["git", "-C", str(I.WT), *args], capture_output=True, text=text)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lock_is_pushed(path, fetch=True):
    p = Path(path).resolve()
    if p.name != LOCK_NAME or not p.is_file():
        return {"ok": False, "reason": "lock file missing or misnamed"}
    try:
        rel = str(p.relative_to(I.WT.resolve()))
    except ValueError:
        return {"ok": False, "reason": "lock file is outside the study repository"}
    if rel != LOCK_REL:
        return {"ok": False, "reason": f"lock file is not the registered {LOCK_REL}"}
    if fetch:
        _git("fetch", "-q", "origin", I.BRANCH)
    commit = _git("log", "-1", "--format=%H", "--", rel).stdout.strip()
    if not commit:
        return {"ok": False, "reason": "lock file is not committed"}
    blob = _git("show", f"{commit}:{rel}", text=False)
    if blob.returncode != 0 or blob.stdout != p.read_bytes():
        return {"ok": False, "reason": "working-tree lock differs from its last commit"}
    if _git("status", "--porcelain", "--", rel).stdout.strip():
        return {"ok": False, "reason": "lock file has uncommitted changes"}
    if f"origin/{I.BRANCH}" not in _git("branch", "-r", "--contains", commit).stdout.split():
        return {"ok": False, "reason": f"lock commit {commit[:12]} is not on origin/{I.BRANCH}"}
    rb = _git("show", f"origin/{I.BRANCH}:{rel}", text=False)
    if rb.returncode != 0 or rb.stdout != p.read_bytes():
        return {"ok": False, "reason": "the lock on origin differs from the working-tree lock"}
    return {"ok": True, "commit": commit, "sha256": _sha(p), "path": str(p)}


def verify_code(lock):
    from hcal import lock as LK
    listed = lock.get("locked_code_files") or {}
    now = LK.code_files()
    changed = sorted(f for f, h in listed.items() if now.get(f) != h)
    if changed:
        raise SystemExit(f"REFUSED: code changed after EVALUATION_LOCK: {changed[:10]}")
    bad = LK.check_loaded_modules(listed)
    if bad:
        raise SystemExit(f"REFUSED: loaded worktree modules are not locked: {bad[:10]}")
    return {"locked_files": len(listed), "all_match": True}


def verify_validity(lock):
    from hcal import run as R
    tv = lock.get("technical_validity") or {}
    if tv.get("ok") is not True:
        raise SystemExit(f"REFUSED: the evaluation lock records a technical failure: {json.dumps(tv)[:800]}")
    ok, why = R.engineering_ready()
    if not ok or tv.get("engineering_gate") != "ENGINEERING_READY":
        raise SystemExit(f"REFUSED: the engineering gate is not ready: {why}")
    if _sha(I.PKG / "ENGINEERING_RESULT.json") != tv.get("engineering_result_sha256"):
        raise SystemExit("REFUSED: ENGINEERING_RESULT.json differs from the version bound in the evaluation lock")
    for n, h in (lock.get("locks_sha256") or {}).items():
        if _sha(I.PKG / f"{n}.json") != h:
            raise SystemExit(f"REFUSED: {n} differs from the evaluation lock")
    for key, path in (("selection_sha256", R.RUN / "selection.json"), ("controls_sha256", R.RUN / "controls.json"),
                      ("replay_sha256", R.RUN / "replay.json"), ("audit_plan_sha256", R.RUN / "audit_plan.json")):
        if _sha(path) != lock.get(key):
            raise SystemExit(f"REFUSED: {path.name} differs from the evaluation lock")
    if R.admission().get("verdict") != "ADMITTED":
        raise SystemExit("REFUSED: admission is not ADMITTED")
    return {"technical_validity": True}


def open_assessment(lock_path, fetch=True):
    global _OPENED
    v = lock_is_pushed(lock_path, fetch)
    if not v["ok"]:
        raise SystemExit(f"REFUSED: {LOCK_NAME} is not committed and pushed ({v['reason']})")
    lock = json.loads(Path(lock_path).read_text())
    v["code"] = verify_code(lock)
    v["validity"] = verify_validity(lock)
    v["lock"] = lock
    _OPENED = v
    return lock


def _require_open():
    if _OPENED is None:
        raise SystemExit("REFUSED: the development assessment is sealed (no verified pushed EVALUATION_LOCK)")
    v = lock_is_pushed(_OPENED["path"], fetch=False)
    if not v["ok"] or v["sha256"] != _OPENED["sha256"] or v["commit"] != _OPENED["commit"]:
        raise SystemExit("REFUSED: EVALUATION_LOCK changed since it was opened")
    return _OPENED


def load_unsealed():
    opened = _require_open()
    from hcal import data as HD
    D = HD.load(unseal=True)
    if D.get("sealed", True):
        raise SystemExit("REFUSED: hcal.data did not unseal the assessment labels")
    a = np.asarray(D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"])
    ar = opened["lock"]["assessment_role"]
    got = {"rows": int(a.size), "groups": int(np.unique(np.asarray(D["unit"])[a]).size),
           "row_id_sha256": HD.sha_arr(np.asarray(D["row_id"])[a])}
    if got != ar:
        raise SystemExit(f"REFUSED: the unsealed assessment role differs from the lock: {got} vs {ar}")
    return D


# ------------------------------------------------------------------ units
def base_arrays(D):
    from hcal import data as HD
    from qpc import utility as UT
    a = np.asarray(D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"], dtype=np.int64)
    r1, y1 = HD.task_labels(D, "income", "assessment", "OSF_DEVELOPMENT_ASSESSMENT")
    r2, y2 = HD.task_labels(D, "occupation", "assessment", "OSF_DEVELOPMENT_ASSESSMENT")
    assert np.array_equal(r1, a) and np.array_equal(r2, a)
    sex = np.asarray(D["sex"], dtype=np.int64)[a]
    if (sex < 0).any():
        raise SystemExit("REFUSED: sealed SEX on assessment rows")
    return a, {"assess_row_id": np.asarray(D["row_id"])[a], "assess_unit": np.asarray(D["unit"])[a], "sex": sex,
               "y_income": y1, "y_occ": y2,
               "const_class": np.array([UT.constant_class(D, 0), UT.constant_class(D, 1)], dtype=np.int64)}


def jobs_from_lock(lock):
    out = []
    for k in I.SEEDS:
        out += [("prob", k, rid) for rid in lock["scored_releases"]]
        out += [("att", k, key) for key in lock["attack_keys"]]
    return out


def run_job(job, D, lock, a, base):
    from hcal import infer as INF
    from hcal import run as R
    from hcal import stages as ST
    _require_open()
    kind, k, x = job
    t0, c0 = time.time(), time.process_time()
    if kind == "prob":
        name = INF.prob_name(k, x)
        if R.done(name):
            return
        _, q, hard = ST.release_arrays(k, x)
        arr = {**base}
        for i in (1, 2):
            arr[f"prob{i}"] = np.asarray(q[i], dtype=np.float64)[a]
            arr[f"hard{i}"] = np.asarray(hard[i], dtype=np.int64)[a]
            if not np.all(np.isfinite(arr[f"prob{i}"])):
                raise SystemExit(f"TECHNICAL FAILURE: nonfinite probabilities for {x} s{k}")
        R.save(name, {"preds.npz": arr}, {"schema": "hcal-oprob-v1", "seed": k, "release": x,
                                          "lock_sha256": _OPENED["sha256"], "lock_commit": _OPENED["commit"],
                                          "wall_s": time.time() - t0})
        return
    name = INF.att_name(k, x)
    if R.done(name):
        return
    com = R.rec(ST.com_name(k, x))["recovery"]
    want = lock["seeds"][str(k)]["winners"][x]
    arr, info = {"assess_row_id": base["assess_row_id"]}, {}
    for w in ("v1", "v2", "pair"):
        for crit in ("auc", "ce"):
            if json.dumps(com["winner_detail"][w][crit], sort_keys=True) != json.dumps(want[w][crit], sort_keys=True):
                raise SystemExit(f"REFUSED: the {x} s{k} {w}/{crit} winner differs from the evaluation lock")
            _, P = ST.refit_winner(k, x, w, crit, D, score_rows=a)
            if not np.all(np.isfinite(P)):
                raise SystemExit(f"TECHNICAL FAILURE: nonfinite attacker scores {x} s{k} {w}/{crit}")
            arr[f"P_{crit}_{w}"] = P
            d = want[w][crit]
            info[f"{crit}_{w}"] = {"bank": d["bank"], "kind": d["kind"], "attacker": d.get("attacker"),
                                   "view": d.get("view"), "inner_refit_bitwise": True}
    R.save(name, {"preds.npz": arr}, {"schema": "hcal-oatt-v1", "seed": k, "key": x, "winners": info,
                                      "lock_sha256": _OPENED["sha256"], "lock_commit": _OPENED["commit"],
                                      "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0})


def main(argv=None):
    from hcal import run as R
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-lock", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    lock = open_assessment(Path(a.evaluation_lock))
    import hcal.stages, hcal.bank, hcal.infer, hcal.calib            # noqa: E401,F401  (locked scoring chain)
    from hcal import lock as LK
    bad = LK.check_loaded_modules(lock["locked_code_files"])
    if bad:
        raise SystemExit("REFUSED: unlocked or changed code loaded: " + "; ".join(bad[:10]))
    D = load_unsealed()
    rows, base = base_arrays(D)
    R.event("start assess", shard=a.shard, pid=os.getpid(), lock_commit=_OPENED["commit"])
    t0, c0 = time.time(), time.process_time()
    for job in R.shard(jobs_from_lock(lock), a.shard):
        run_job(job, D, lock, rows, base)
    bad = LK.check_loaded_modules(lock["locked_code_files"])
    if bad:
        raise SystemExit("REFUSED after assessment (lazy import of unlocked code): " + "; ".join(bad[:10]))
    R.ledger("assess", a.shard, time.time() - t0, time.process_time() - c0)
    R.event("end assess", shard=a.shard, pid=os.getpid())


if __name__ == "__main__":
    main()
