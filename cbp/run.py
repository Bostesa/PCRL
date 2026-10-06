"""Runner for the confidence-budgeted privacy study (atomic hash-verified units; resumable; at most two workers).

Adapted from qpc/run.py at d0c8a45 (same unit format, same save/resume and lock checks).

    <PRIVATE_CACHE>/cbp_v1/run/work.sh <LOCK> <stage> [i/n]
      == python -m cbp.sema --label A:<stage> -- env OMP_NUM_THREADS=1 ... <python> -m cbp.run --lock <LOCK> --stage ...

Stages (each refuses unless the latest named lock + amendments verify and are on origin, and every study module the
stage loads is locked with its hash -- cbp.lock.check_loaded_modules):
  SOURCE_ADMISSION_LOCK     admit      verified copies of the qpc teachers, references, fine partitions, DIRECT-TASK,
                                       FINE-TASK, CLASS-ONLY and the 24 endpoint privacy maps (lambda 0.01, 0.1) +
                                       teacher forward parity + release re-encode parity (cbp.admit)
  FIT_LOCK (+ AUDIT_AND_SELECTION_LOCK, both pushed before any new fit)
                            fit        endpoint objective parity of every reused code (cbp.fit.endpoint_parity), then
                                       the 48 new intermediate-lambda units (cbp.fit.fit_unit = qpc.compress.fit_unit)
  AUDIT_AND_SELECTION_LOCK  inner      inner audits of every code and reference (cbp.audit), plus the 11 composition-only
                                       qpc public maps (COMPOSED_EXTRA_IDS; never selection candidates)
                            inner_src  continuous-source inner audits composed over EVERY code of the same seed
                            controls   real-data null and planted controls (cbp.audit)
                            select     cbp.select.select_all
Assessment labels stay sealed (cbp.data); only cbp.assess unseals, after the pushed EVALUATION_LOCK.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private" / "cbp_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_confidence_budgeted_privacy_v1"
SEEDS = (0, 1, 2)
TEACHERS = ("U", "RAW-J_b0.3")
REFS = ("E", "F", "F0")
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
RATE = (8, 64)
LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
ENDPOINT_LAMS = (0.01, 0.1)
NEW_LAMS = (0.025, 0.04, 0.06, 0.08)
LATE = {"inner": ("cbp.run", "stage_inner"), "inner_src": ("cbp.run", "stage_inner_src"),
        "controls": ("cbp.audit", "stage_controls"), "select": ("cbp.select", "select_all")}


# ------------------------------------------------------------------ configuration ids (qpc scheme)
def g(x):
    return f"{x:g}"


def config_id(family, m1=None, m2=None, lam=None, teacher="U"):
    if family == "CLASS":
        return f"{teacher}|CLASS|i1o1"
    return f"{teacher}|{family}|i{m1}o{m2}" + (f"|l{g(lam)}" if family in PRIVACY else "")


def parse_id(cid):
    if cid.startswith("SRC|"):
        return {"kind": "source", "teacher": cid.split("|")[1]}
    if cid.startswith("REF|"):
        return {"kind": "reference", "label": cid.split("|")[1]}
    t, fam, rate, *rest = cid.split("|")
    m1, m2 = rate[1:].split("o")
    return {"kind": "policy", "teacher": t, "family": fam, "m1": int(m1), "m2": int(m2),
            "lam": float(rest[0][1:]) if rest else None}


def safe(cid):
    return cid.replace("|", "_")


def unit_for(k, cid):
    p = parse_id(cid)
    if p["kind"] == "source":
        return f"tea__s{k}__{p['teacher']}"
    if p["kind"] == "reference":
        return f"ref__s{k}__{p['label']}"
    return f"pol__s{k}__{safe(cid)}"


def privacy_ids(lams=LAMS):
    return [config_id(f, *RATE, lam) for lam in lams for f in PRIVACY]


def task_ids():
    return [config_id("DIRECT-TASK", *RATE), config_id("FINE-TASK", *RATE), config_id("CLASS")]


def code_ids():
    """Every code of the registered bank (27 per seed): task-only references, then the 24 privacy maps."""
    return task_ids() + privacy_ids()


# composition-only public maps (prompt sec. 10: retain valid admitted source attack candidates where compatible): the
# 11 qpc codes of the source composed bank that are not in the cbp bank. They are audited (inner units) so SRC|U can
# apply them as public maps; they are NEVER selection candidates and never scored as releases.
COMPOSED_EXTRA_IDS = ([config_id("DIRECT-TASK", a, b) for a in (4, 8) for b in (8, 16, 32, 64) if (a, b) != RATE] +
                      [config_id(f, *RATE, 1.0) for f in PRIVACY])


def composition_ids():
    """Every public code map the continuous U source composes with: the 27 cbp codes, then the 11 qpc extras."""
    return code_ids() + COMPOSED_EXTRA_IDS


def scored_ids(protocol=None):
    """Every inner-selection candidate (32 per seed): codes, continuous sources, references."""
    return code_ids() + [f"SRC|{t}" for t in TEACHERS] + [f"REF|{r}" for r in REFS]


def reused_ids():
    return task_ids() + privacy_ids(ENDPOINT_LAMS)


def new_ids():
    return privacy_ids(NEW_LAMS)


# ------------------------------------------------------------------ units, events, compute ledger
def _finite(o):
    """Recursively replace nonfinite floats by None so every new JSON is finite-or-null (prompt sec. 6)."""
    import math
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
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": msg, **kw},
                           allow_nan=False) + "\n")


def ledger(stage, shard, wall, cpu):
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
    z = np.load(U(name) / file)
    return {x: z[x] for x in z.files}


def save(name, files, record):
    """files: {fname: callable(path) | dict of arrays (-> npz) | dict/list (-> json) | str (-> text)}."""
    from jcv.finalize import save_unit

    def writer(v, fname):
        if callable(v):
            return v
        if isinstance(v, dict) and fname.endswith(".npz"):
            return lambda p: np.savez_compressed(p, **{k: np.asarray(a) for k, a in v.items()})
        if isinstance(v, (dict, list)):
            return lambda p: p.write_text(json.dumps(v, default=float))
        return lambda p: p.write_text(str(v))
    save_unit(U(name), {f: writer(v, f) for f, v in files.items()}, record)
    event("unit complete", unit=name)


def shard(jobs, spec):
    if not spec:
        return jobs
    i, n = map(int, spec.split("/"))
    return [j for t, j in enumerate(jobs) if t % n == i]


def teacher(k, t="U"):
    return npz(f"tea__s{k}__{t}", "teacher.npz")


def fit_rows(D):
    return np.asarray(D["idx"]["DEFENSE_FIT"])


def bind_meta(k, cid, D, t="U"):
    from qpc import deploy as DP
    trec = rec(f"tea__s{k}__{t}")
    return {"teacher": t, "seed": k, "config": cid, "teacher_model_sha256": trec["model_sha256"],
            "feature_names_sha256": DP.schema_sha256([str(x) for x in D["feature_names"]])}


# ------------------------------------------------------------------ admission (SOURCE_ADMISSION_LOCK)
def stage_admit(D, shard_spec=None):
    from cbp import admit as AD
    r = AD.run(D)
    event("admission", verdict=r["verdict"], copied=len(r["copied_now"]), teachers=len(r["teachers"]),
          codes=len(r["codes"]))


# ------------------------------------------------------------------ fit (FIT_LOCK; AUDIT_AND_SELECTION_LOCK pushed too)
def fit_jobs():
    """(seed, lambda) groups for the NEW lambdas; inside a group LOCAL, SEQ-12, SEQ-21, then JOINT (its witnesses:
    the admitted FINE-TASK and the same-lambda LOCAL / SEQ-12 / SEQ-21 of the group)."""
    return [(k, lam) for k in SEEDS for lam in NEW_LAMS]


def stage_fit(D, shard_spec=None):
    from cbp import fit as FT
    tr = fit_rows(D)
    S_fit = np.asarray(D["sex"][tr])
    assert (S_fit >= 0).all()
    # 1. objective / deployment parity of every reused code (written once; refuses on any mismatch)
    par_path = RUN / "endpoint_parity.json"
    if not par_path.exists():
        par = {}
        for k in SEEDS:
            T = teacher(k)
            fine = json.loads((U(f"fine__s{k}") / "fine.json").read_text())
            for cid in reused_ids():
                n = unit_for(k, cid)
                r = FT.endpoint_parity(U(n), T, tr, S_fit, fine)
                par[n] = r
                if not r.get("ok"):
                    raise SystemExit(f"REUSE PARITY FAILED {n}: {r}")
        par_path.write_text(json.dumps(_finite(par), indent=1, allow_nan=False) + "\n")
        event("endpoint parity", units=len(par), all_ok=True)
    # 2. the 48 new intermediate-lambda units
    for k, lam in shard(fit_jobs(), shard_spec):
        T = teacher(k)
        fine = json.loads((U(f"fine__s{k}") / "fine.json").read_text())
        for fam in PRIVACY:
            cid = config_id(fam, *RATE, lam)
            n = unit_for(k, cid)
            if done(n):
                continue
            wit = None
            if fam == "JOINT":
                wit = {f: json.loads((U(unit_for(k, config_id(f, *RATE, None if f == "FINE-TASK" else lam)))
                                      / "policy.json").read_text()) for f in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")}
            t0, c0 = time.time(), time.process_time()
            r, files = FT.fit_unit(fam, fine, T, tr, S_fit, lam, bind_meta(k, cid, D), witnesses=wit)
            r.update({"seed": k, "config": cid, "cfg": parse_id(cid), "wall_s": time.time() - t0,
                      "cpu_s": time.process_time() - c0, "origin": "NEW_FIT"})
            save(n, files, _finite(r))


# ------------------------------------------------------------------ inner audits (AUDIT_AND_SELECTION_LOCK)
def _inner(D, shard_spec, kinds):
    from cbp import audit as AU
    ids = scored_ids() + (COMPOSED_EXTRA_IDS if "policy" in kinds else [])
    jobs = [(k, c) for k in SEEDS for c in ids if parse_id(c)["kind"] in kinds]
    for k, cid in shard(jobs, shard_spec):
        n = f"inner__{unit_for(k, cid)}"
        if done(n):
            continue
        t0, c0 = time.time(), time.process_time()
        r = AU.inner_unit(parse_id(cid)["kind"], k, cid, D)
        files = r.pop("_files", {})
        r.update({"of": unit_for(k, cid), "config": cid, "seed": k, "wall_s": time.time() - t0,
                  "cpu_s": time.process_time() - c0})
        save(n, files, _finite(r))


def stage_inner(D, shard_spec=None):
    _inner(D, shard_spec, ("policy", "reference"))


def stage_inner_src(D, shard_spec=None):
    missing = [f"inner__{unit_for(k, c)}" for k in SEEDS for c in composition_ids()
               if not done(f"inner__{unit_for(k, c)}")]
    if missing:
        raise SystemExit(f"REFUSED: composed source banks need every code's inner unit first; missing {missing[:5]}")
    _inner(D, shard_spec, ("source",))


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from cbp import lock as LK
    v = LK.verify_lock(Path(a.lock), stage=a.stage)
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    from cbp import data as DA
    D = DA.load()
    assert D["sealed"]
    fn = {"admit": stage_admit, "fit": stage_fit}
    if a.stage in fn:
        f = fn[a.stage]
    elif a.stage in LATE:
        mod, name = LATE[a.stage]
        f = getattr(importlib.import_module(mod), name)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    for m in {"admit": ["cbp.admit"], "fit": ["cbp.fit", "qpc.deploy"], "inner": ["cbp.audit"],
              "inner_src": ["cbp.audit"], "controls": ["cbp.audit"], "select": ["cbp.select"]}.get(a.stage, []):
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
