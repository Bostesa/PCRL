"""Runner of the held-out calibration study (hcal; atomic hash-verified units; resumable; <= 2 workers).

Adapted from lra/run.py at 9762025 (same unit format, save/resume, lock and loaded-module checks).

    <PRIVATE_CACHE>/hcal_v1/run/work.sh <LOCK> <stage> [i/n]
      == python -m hcal.sema --label A:<stage> -- env OMP_NUM_THREADS=1 ... <python> -m hcal.run --lock <LOCK> --stage ...

Stages (each refuses unless the latest named lock + amendments verify and are on origin, and every study module the
stage loads is locked with its hash):
  SOURCE_ADMISSION_LOCK  admit        verified copies, parity, frozen bank, role receipt (hcal.admit; no label)
  ENGINEERING_LOCK       engineering  synthetic correctness checks only (hcal.engineering)
  SCIENCE_LOCK           calibrate    H-TOKEN32 / H-GLOBAL-TEMP / H-CLASS-TEMP (+ T-TOKEN32, U calibrations)
                         utility      utility tables of the full decoder bank; Ucal*; the audit plan (pruning rule)
                         audit        fresh banks + common records of the audited partitions (hcal.bank)
                         compose      continuous-U composed record per seed
                         controls     real-data controls (hcal.controls)
                         select       nomination (hcal.select)
                         replay       seed-0 legacy-bank refit replay of the selected legacy readers (hcal.bank)
Assessment labels stay sealed (hcal.data); only hcal.assess unseals, after the pushed EVALUATION_LOCK.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import resource
import time

import numpy as np

from hcal import ids as I

UNITS = I.UNITS
RUN = I.RUN
LATE = {"engineering": ("hcal.engineering", "stage_engineering"), "calibrate": ("hcal.stages", "stage_calibrate"),
        "utility": ("hcal.stages", "stage_utility"), "audit": ("hcal.stages", "stage_audit"),
        "compose": ("hcal.stages", "stage_compose"), "controls": ("hcal.stages", "stage_controls"),
        "select": ("hcal.select", "select_all"), "replay": ("hcal.stages", "stage_replay")}
SCIENCE_STAGES = ("calibrate", "utility", "audit", "compose", "controls", "select", "replay")
STAGE_MODULES = {"admit": ["hcal.admit"], "engineering": ["hcal.engineering", "hcal.calib", "hcal.bank", "hcal.controls",
                                                          "hcal.select", "hcal.family"],
                 "calibrate": ["hcal.stages", "hcal.calib"], "utility": ["hcal.stages", "hcal.calib"],
                 "audit": ["hcal.stages", "hcal.bank"], "compose": ["hcal.stages", "hcal.bank"],
                 "controls": ["hcal.stages", "hcal.controls", "hcal.bank"], "select": ["hcal.select", "hcal.family"],
                 "replay": ["hcal.stages", "hcal.bank"]}
GATE_RESULT = "ENGINEERING_RESULT.json"


def _finite(o):
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _finite(o.tolist())
    return o


def event(msg, **kw):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "ACTIVITY_LOG.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": msg, **_finite(kw)},
                           allow_nan=False) + "\n")


def ledger(stage, shard, wall, cpu):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "COMPUTE_LEDGER.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, "shard": shard,
                            "pid": os.getpid(), "wall_s": wall, "cpu_s": cpu, "maxrss_bytes":
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}) + "\n")


def U(name):
    return UNITS / name


def done(name):
    from jcv.finalize import unit_complete
    return unit_complete(U(name))


def rec(name):
    return json.loads((U(name) / "record.json").read_text())


def npz(name, file):
    z = np.load(U(name) / file, allow_pickle=False)
    return {x: z[x] for x in z.files}


def save(name, files, record):
    """files: {fname: callable(path) | dict of arrays (-> npz) | dict/list (-> json) | str (-> text)}."""
    from jcv.finalize import save_unit
    UNITS.mkdir(parents=True, exist_ok=True)

    def writer(v, fname):
        if callable(v):
            return v
        if isinstance(v, dict) and fname.endswith(".npz"):
            return lambda p: np.savez_compressed(p, **{k: np.asarray(a) for k, a in v.items()})
        if isinstance(v, (dict, list)):
            return lambda p: p.write_text(json.dumps(_finite(v), allow_nan=False))
        return lambda p: p.write_text(str(v))
    save_unit(U(name), {f: writer(v, f) for f, v in files.items()}, _finite(record))
    event("unit complete", unit=name)


def shard(jobs, spec):
    if not spec:
        return jobs
    i, n = map(int, spec.split("/"))
    return [j for t, j in enumerate(jobs) if t % n == i]


def bank(k, p):
    """Frozen-bank arrays of partition p at seed k (admitted; hash-checked against the admission receipt)."""
    from hcal import admit as AD
    path = AD.bank_path(k, p)
    want = admission()["bank"][f"{k}|{p}"]["bank_sha256"]
    if AD.sha_file(path) != want:
        raise SystemExit(f"REFUSED: frozen bank {path.name} differs from its admitted hash")
    z = np.load(path, allow_pickle=False)
    return {x: z[x] for x in z.files}


_ADM = {}


def admission():
    if "rec" not in _ADM:
        r = json.loads((I.ADM / "ADMISSION_RECEIPT.json").read_text())
        if r.get("verdict") != "ADMITTED":
            raise SystemExit("REFUSED: admission receipt is not ADMITTED")
        _ADM["rec"] = r
    return _ADM["rec"]


def teacher(k, t="U"):
    z = np.load(I.ADM_UNITS / f"tea__s{k}__{t}" / "teacher.npz", allow_pickle=False)
    return {x: z[x] for x in z.files}


# ------------------------------------------------------------------ admission (SOURCE_ADMISSION_LOCK)
def stage_admit(D, shard_spec=None):
    from hcal import admit as AD
    from hcal import data as HD
    r = AD.run(D)
    pub = AD.public_record(r, D)
    pub["role_receipt"] = D["hcal"]["receipt"]
    (RUN / "SOURCE_ADMISSION.public.json").write_text(json.dumps(_finite(pub), indent=1, allow_nan=False) + "\n")
    HD.write_role_manifest(D, RUN / "ROLE_MANIFEST.public.json")
    event("admission", verdict=r["verdict"], copied=len(r["copied_now"]), bank=len(r["bank"]))


def engineering_ready():
    """Every Adult science stage requires the engineering stage's ENGINEERING_READY result, pushed and bound (sha256)
    into the latest named lock's documents. Never hard-coded."""
    from hcal import lock as LK
    p = I.PKG / GATE_RESULT
    if not p.exists():
        return False, f"{GATE_RESULT} missing"
    r = json.loads(p.read_text())
    if r.get("verdict") != "ENGINEERING_READY":
        return False, f"verdict is {r.get('verdict')!r}, not ENGINEERING_READY"
    lat = LK.latest()
    if lat["documents_sha256"].get(GATE_RESULT) != LK.sha_file(p):
        return False, f"{GATE_RESULT} is not the version bound in {lat['name']}"
    if not LK.on_origin(f"{LK.REL}/{GATE_RESULT}"):
        return False, f"{GATE_RESULT} is not on origin"
    return True, "ENGINEERING_READY"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from hcal import lock as LK
    from pathlib import Path
    v = LK.verify_lock(Path(a.lock), stage=a.stage)
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    if a.stage in SCIENCE_STAGES:
        ok, why = engineering_ready()
        if not ok:
            raise SystemExit("REFUSED: Adult science stages require ENGINEERING_READY: " + why)
    D = None
    if a.stage != "engineering":                 # engineering is synthetic: no real data is loaded
        from hcal import data as HD
        D = HD.load()
        assert D["sealed"]
    if a.stage == "admit":
        f = stage_admit
    elif a.stage in LATE:
        mod, name = LATE[a.stage]
        f = getattr(importlib.import_module(mod), name)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    for m in STAGE_MODULES.get(a.stage, []):
        importlib.import_module(m)
    bad = LK.check_loaded_modules(v["locked_files"])
    if bad:
        raise SystemExit("REFUSED: unlocked or changed code loaded: " + "; ".join(bad[:10]))
    event(f"start {a.stage}", shard=a.shard, pid=os.getpid(), lock=v["lock"])
    t0, c0 = time.time(), time.process_time()
    f(D, a.shard)
    bad = LK.check_loaded_modules(v["locked_files"])
    if bad:
        raise SystemExit("REFUSED after stage (lazy import of unlocked code): " + "; ".join(bad[:10]))
    ledger(a.stage, a.shard, time.time() - t0, time.process_time() - c0)
    event(f"end {a.stage}", shard=a.shard, pid=os.getpid())


if __name__ == "__main__":
    main()
