"""Runner for the learned-decoder constrained-release study (lcr; atomic hash-verified units; resumable; <= 2 workers).

Adapted from cbp/run.py at 7f3ec67 (same unit format, same save/resume and lock checks).

    <PRIVATE_CACHE>/lcr_v1/run/work.sh <LOCK> <stage> [i/n]
      == python -m lcr.sema --label A:<stage> -- env OMP_NUM_THREADS=1 ... <python> -m lcr.run --lock <LOCK> --stage ...

Stages (each refuses unless the latest named lock + amendments verify and are on origin, and every study module the
stage loads is locked with its hash -- lcr.lock.check_loaded_modules):
  SOURCE_ADMISSION_LOCK  admit      verified copies of the cbp teachers, references, fine partitions, the D0 bank
                                    (DIRECT-TASK, FINE-TASK, CLASS-ONLY, 24 privacy maps x 3 seeds) and their cbp inner
                                    audits + teacher forward parity + release re-encode parity (lcr.admit)
  FIXTURE_LOCK           fixture    the four registered fixture families: exhaustive references + every arm (lcr.fixtures)
  SCIENCE_LOCK           d1         D1 learned decoders on the EXACT fixed D0 maps (calibration-only controls)
                         fit        C-TASK, 72 weighted controls, 15 constrained units (lcr.mapper)
                         inner      inner audits of every new release (lcr.audit)
                         inner_src  continuous-source inner audits composed over the COMPLETE registered bank
                         controls   real-data null and planted controls
                         select     lcr.select.select_all
Assessment labels stay sealed (lcr.data); only lcr.assess unseals, after the pushed EVALUATION_LOCK.

CONFIG IDS (shared contract; every module uses these helpers):
  D0 (admitted, unchanged)   U|DIRECT-TASK|i8o64, U|FINE-TASK|i8o64, U|CLASS|i1o1, U|{LOCAL,SEQ-12,SEQ-21,JOINT}|i8o64|l{lam}
  D1 fixed-map controls      <D0 id>|D1 for DIRECT-TASK, FINE-TASK and the 24 privacy maps (assignments unchanged)
  C-TASK                     U|C-TASK|i8o64|D1
  weighted controls          U|W-{LOCAL,SEQ-12,SEQ-21,JOINT}|i8o64|l{lam}|D1
  constrained arms           U|K-{LOCAL,SEQ-12,SEQ-21,JOINT-SINGLE,JOINT-PAIR}|i8o64|D1
  sources / references       SRC|U, SRC|RAW-J_b0.3, REF|E, REF|F, REF|F0
UNITS: pol__s{k}__<safe> (D0, admitted), dec__s{k}__<safe> (D1 fixed-map), new__s{k}__<safe> (fitted), tea__, ref__,
fine__, inner__<unit>.
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
PRIV = HOME / "PCRL_eval_cache_private" / "lcr_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_learned_decoder_constrained_release_v1"
SEEDS = (0, 1, 2)
TEACHERS = ("U", "RAW-J_b0.3")
REFS = ("E", "F", "F0")
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
RATE = (8, 64)
LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
CONSTRAINED = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT-SINGLE", "JOINT-PAIR")
LATE = {"d1": ("lcr.run", "stage_d1"), "fit": ("lcr.run", "stage_fit"), "inner": ("lcr.run", "stage_inner"),
        "inner_src": ("lcr.run", "stage_inner_src"), "controls": ("lcr.audit", "stage_controls"),
        "select": ("lcr.select", "select_all"), "fixture": ("lcr.fixtures", "stage_fixture")}


def g(x):
    return f"{x:g}"


# ------------------------------------------------------------------ config ids
def d0_id(family, lam=None):
    if family == "CLASS":
        return "U|CLASS|i1o1"
    if family in ("DIRECT-TASK", "FINE-TASK"):
        return f"U|{family}|i8o64"
    return f"U|{family}|i8o64|l{g(lam)}"


def d1_id(family, lam=None):
    return d0_id(family, lam) + "|D1"


def ctask_id():
    return "U|C-TASK|i8o64|D1"


def weighted_id(family, lam):
    return f"U|W-{family}|i8o64|l{g(lam)}|D1"


def constrained_id(arm):
    assert arm in CONSTRAINED
    return f"U|K-{arm}|i8o64|D1"


def d0_ids():
    return [d0_id("DIRECT-TASK"), d0_id("FINE-TASK"), d0_id("CLASS")] + [d0_id(f, lam) for lam in LAMS for f in PRIVACY]


def d0_privacy_ids():
    return [d0_id(f, lam) for lam in LAMS for f in PRIVACY]


def d1_fixed_ids():
    return [d1_id("DIRECT-TASK"), d1_id("FINE-TASK")] + [d1_id(f, lam) for lam in LAMS for f in PRIVACY]


def weighted_ids():
    return [weighted_id(f, lam) for lam in LAMS for f in PRIVACY]


def constrained_ids():
    return [constrained_id(a) for a in CONSTRAINED]


def new_fit_ids():
    """The 30 new mapping-pair fits per seed (90 total): C-TASK, 24 weighted controls, 5 constrained arms."""
    return [ctask_id()] + weighted_ids() + constrained_ids()


def code_ids():
    """Every code release of the registered bank (one seed): D0 (27), D1 fixed-map (26), new fits (30) = 83."""
    return d0_ids() + d1_fixed_ids() + new_fit_ids()


def scored_ids():
    """Every inner-selection candidate / comparator (one seed): the code bank, both continuous sources, 3 references."""
    return code_ids() + [f"SRC|{t}" for t in TEACHERS] + [f"REF|{r}" for r in REFS]


def parse_id(cid):
    if cid.startswith("SRC|"):
        return {"kind": "source", "teacher": cid.split("|")[1]}
    if cid.startswith("REF|"):
        return {"kind": "reference", "label": cid.split("|")[1]}
    parts = cid.split("|")
    t, fam, rate = parts[0], parts[1], parts[2]
    rest = parts[3:]
    decoder = "D1" if rest and rest[-1] == "D1" else "D0"
    lam = next((float(x[1:]) for x in rest if x.startswith("l")), None)
    m1, m2 = rate[1:].split("o")
    if fam.startswith("W-"):
        arm, base = "weighted", fam[2:]
    elif fam.startswith("K-"):
        arm, base = "constrained", fam[2:]
    elif fam == "C-TASK":
        arm, base = "ctask", fam
    elif decoder == "D1":
        arm, base = "d1_fixed", fam
    else:
        arm, base = "d0", fam
    return {"kind": "policy", "teacher": t, "family": fam, "base_family": base, "arm": arm, "decoder": decoder,
            "m1": int(m1), "m2": int(m2), "lam": lam,
            "privacy_trained": base in PRIVACY + CONSTRAINED and fam != "C-TASK"}


def safe(cid):
    return cid.replace("|", "_")


def unit_for(k, cid):
    p = parse_id(cid)
    if p["kind"] == "source":
        return f"tea__s{k}__{p['teacher']}"
    if p["kind"] == "reference":
        return f"ref__s{k}__{p['label']}"
    prefix = {"d0": "pol", "d1_fixed": "dec"}.get(p["arm"], "new")
    return f"{prefix}__s{k}__{safe(cid)}"


# ------------------------------------------------------------------ units, events, compute ledger
def _finite(o):
    """Recursively replace nonfinite floats by None so every new JSON is finite-or-null (prompt sec. 5)."""
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
            return lambda p: p.write_text(json.dumps(_finite(v), allow_nan=False))
        return lambda p: p.write_text(str(v))
    save_unit(U(name), {f: writer(v, f) for f, v in files.items()}, _finite(record))
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
    from lcr import admit as AD
    r = AD.run(D)
    event("admission", verdict=r["verdict"], copied=len(r["copied_now"]), teachers=len(r["teachers"]),
          codes=len(r["codes"]))


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from lcr import lock as LK
    v = LK.verify_lock(Path(a.lock), stage=a.stage)
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    D = None
    if a.stage != "fixture":                     # fixtures are synthetic known laws: no real data is loaded
        from lcr import data as DA
        D = DA.load()
        assert D["sealed"]
    fn = {"admit": stage_admit}
    if a.stage in fn:
        f = fn[a.stage]
    elif a.stage in LATE:
        mod, name = LATE[a.stage]
        f = getattr(importlib.import_module(mod), name)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    for m in {"admit": ["lcr.admit"], "fixture": ["lcr.fixtures", "lcr.decoder", "lcr.mapper"],
              "d1": ["lcr.decoder"], "fit": ["lcr.mapper", "lcr.decoder"], "inner": ["lcr.audit"],
              "inner_src": ["lcr.audit"], "controls": ["lcr.audit"], "select": ["lcr.select"]}.get(a.stage, []):
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
