"""Runner for the decision-preserving compression study (atomic hash-verified units; resumable; 2-worker sharding).

    env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.run --lock <lock json> --stage <stage> [--shard i/n]

Stages (each refuses unless the latest named lock + amendments verify and are on origin; dpc.lock.STAGE_MIN_LOCK):
  ENGINEERING_LOCK          admit      verified teacher outputs (tea__s{k}__<teacher>) and reference score releases
                                       (ref__s{k}__<label>) from the admitted copies (dpc.admit)
  TRAINING_LOCK             partition  fine KL partitions per teacher/seed/recipient on OSF_DEFENSE_FIT (fine__s{k}__<t>)
                            fit        every locked bank configuration (pol__s{k}__<config>) + class-only m=1
  SELECTION_AND_AUDIT_LOCK  inner      inner audits + INNER_SELECTION utility of every policy, source and reference
                            controls   real-data nulls and planted code leaks (dpc.controls / dpc.audit)
                            select     dpc.select.select_all
Assessment labels stay sealed (dpc.data); only dpc.assess unseals after the pushed EVALUATION_LOCK.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from dpc import data as DA
from rgj import finalize as FN

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private" / "dpc_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_decision_preserving_compression_v1"
SEEDS = (0, 1, 2)
TEACHERS = ("U", "RAW-J_b0.3")
REFS = ("E", "F", "F0")
TASK_ONLY = ("FINE-TASK", "DIRECT-TASK")
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
BANKS = {"full": {"rates": (2, 4, 8), "lams": (0.1, 1.0, 10.0)}, "reduced": {"rates": (4, 8), "lams": (0.1, 1.0)}}
RATES = BANKS["full"]["rates"]                       # rebound from the TRAINING_LOCK by locked_bank_ids()
LATE = {"inner": ("dpc.run", "stage_inner"), "controls": ("dpc.controls", "run_controls"),
        "select": ("dpc.select", "select_all")}


def g(x):
    return f"{x:g}"


def config_id(teacher, family, m, lam=None):
    if family == "CLASS":
        return f"{teacher}|CLASS|m1"
    return f"{teacher}|{family}|m{m}" + (f"|l{g(lam)}" if family in PRIVACY else "")


def parse_id(cid):
    if cid.startswith("SRC|"):
        return {"kind": "source", "teacher": cid.split("|")[1]}
    if cid.startswith("REF|"):
        return {"kind": "reference", "label": cid.split("|")[1]}
    t, fam, m, *rest = cid.split("|")
    return {"kind": "policy", "teacher": t, "family": fam, "m": int(m[1:]),
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


def bank_ids(kind="full"):
    b = BANKS[kind]
    ids = []
    for t in TEACHERS:
        for m in b["rates"]:
            ids += [config_id(t, f, m) for f in TASK_ONLY]
            ids += [config_id(t, f, m, lam) for lam in b["lams"] for f in PRIVACY]
        ids.append(config_id(t, "CLASS", 1))
    return ids


def locked_bank():
    L = json.loads((PKG / "TRAINING_LOCK.json").read_text())
    return L["protocol"]["bank"]


def locked_bank_ids():
    """Every selection candidate: the locked policy bank + continuous sources + references."""
    global RATES
    kind = locked_bank()
    RATES = BANKS[kind]["rates"]
    return bank_ids(kind) + [f"SRC|{t}" for t in TEACHERS] + [f"REF|{r}" for r in REFS]


# ------------------------------------------------------------------ units, events, compute ledger
def event(msg, **kw):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "ACTIVITY_LOG.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": msg, **kw}) + "\n")


def ledger(stage, shard, wall, cpu):
    with open(RUN / "COMPUTE_LEDGER.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, "shard": shard,
                            "pid": os.getpid(), "wall_s": wall, "cpu_s": cpu, "maxrss_bytes":
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}) + "\n")


def U(name):
    return UNITS / name


def done(name):
    return FN.unit_complete(U(name))


def rec(name):
    return json.loads((U(name) / "record.json").read_text())


def shard(jobs, spec):
    if not spec:
        return jobs
    i, n = map(int, spec.split("/"))
    return [j for t, j in enumerate(jobs) if t % n == i]


def npz_writer(arrs):
    return lambda p: np.savez_compressed(p, **arrs)


def teacher(k, t):
    z = np.load(U(f"tea__s{k}__{t}") / "teacher.npz")
    return {x: z[x] for x in z.files}


def sha_arrays(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ admission (ENGINEERING_LOCK)
def stage_admit(D, shard_spec=None):
    from dpc import admit as AD
    for k in shard(list(SEEDS), shard_spec):
        for t in TEACHERS:
            n = f"tea__s{k}__{t}"
            if done(n):
                continue
            T = AD.teacher(t, k)
            assert np.array_equal(T["row_id"], D["row_id"]), "teacher rows differ from the osf rows"
            for i in (1, 2):
                assert np.array_equal(T[f"d{i}"], np.asarray(T[f"p{i}"]).argmax(1)), "decision != argmax p"
            FN.save_unit(U(n), {"teacher.npz": npz_writer({x: np.asarray(T[x]) for x in
                                                            ("row_id", "p1", "p2", "d1", "d2", "c1", "c2", "r1", "r2")})},
                         {"seed": k, "teacher": t, "admission": AD.admission_record().get("teachers", {}).get(f"{t}|{k}"),
                          "p_sha256": sha_arrays(T["p1"], T["p2"])})
            event("unit complete", unit=n)
        for lab in REFS:
            n = f"ref__s{k}__{lab}"
            if done(n):
                continue
            Rf = AD.reference(lab, k)
            arrs = {x: np.asarray(v) for x, v in Rf.items() if isinstance(v, np.ndarray)}
            FN.save_unit(U(n), {"reference.npz": npz_writer(arrs)},
                         {"seed": k, "label": lab, "arrays": sorted(arrs),
                          "admission": {x: v for x, v in Rf.items() if not isinstance(v, np.ndarray)}})
            event("unit complete", unit=n)


# ------------------------------------------------------------------ partitions and policies (TRAINING_LOCK)
def fit_rows(D):
    return D["idx"]["DEFENSE_FIT"]


def stage_partition(D, shard_spec=None):
    from dpc import partition as PT
    tr = fit_rows(D)
    for k, t in shard([(k, t) for k in SEEDS for t in TEACHERS], shard_spec):
        n = f"fine__s{k}__{t}"
        if done(n):
            continue
        T = teacher(k, t)
        t0, c0 = time.time(), time.process_time()
        fine1, fine2, receipts = PT.fit_fine_pair(T["p1"][tr], T["d1"][tr], T["p2"][tr], T["d2"][tr])
        a1 = PT.assign_fine(T["p1"], T["d1"], fine1)
        a2 = PT.assign_fine(T["p2"], T["d2"], fine2)
        FN.save_unit(U(n), {"fine.json": lambda p: p.write_text(json.dumps({"fine1": PT.to_dict(fine1),
                                                                             "fine2": PT.to_dict(fine2)})),
                            "assign.npz": npz_writer({"row_id": D["row_id"], "f1": a1, "f2": a2})},
                     {"seed": k, "teacher": t, "receipts": receipts, "wall_s": time.time() - t0,
                      "cpu_s": time.process_time() - c0, "fit_rows": int(len(tr))})
        event("unit complete", unit=n)


def stage_fit(D, shard_spec=None):
    from dpc import compress as CP
    from dpc import deploy as DP
    from dpc import partition as PT
    from dpc import release as RL
    tr = fit_rows(D)
    S_fit = np.asarray(D["sex"][tr])
    assert (S_fit >= 0).all()
    ids = [c for c in locked_bank_ids() if parse_id(c)["kind"] == "policy"]
    jobs = [(k, c) for k in SEEDS for c in ids]
    for k, cid in shard(jobs, shard_spec):
        n = unit_for(k, cid)
        if done(n):
            continue
        c = parse_id(cid)
        T = teacher(k, c["teacher"])
        F = json.loads((U(f"fine__s{k}__{c['teacher']}") / "fine.json").read_text())
        fine1, fine2 = PT.from_dict(F["fine1"]), PT.from_dict(F["fine2"])
        t0, c0 = time.time(), time.process_time()
        trec = rec(f"tea__s{k}__{c['teacher']}")
        meta = {"teacher": c["teacher"], "seed": k, "config": cid,           # binds the policy for dpc.deploy
                "teacher_model_sha256": trec["admission"]["complete_files_sha256"]["model.pt"],
                "feature_names_sha256": DP.schema_sha256([str(x) for x in D["feature_names"]])}
        if c["family"] == "CLASS":
            pair, receipts = CP.fit_class_only(fine1, fine2, T["p1"][tr], T["d1"][tr], T["p2"][tr], T["d2"][tr], S_fit,
                                               meta=meta)
        else:                       # JOINT recomputes its four witnesses deterministically (verified against units)
            pair, receipts = CP.fit_policy_pair(c["family"], fine1, fine2, T["p1"][tr], T["d1"][tr], T["p2"][tr],
                                                T["d2"][tr], S_fit, c["m"], c["lam"], meta=meta)
        out = {"row_id": D["row_id"]}
        for i in (1, 2):
            tok, q, dec = RL.encode(pair[i - 1], T[f"p{i}"], T[f"d{i}"])
            out.update({f"tok{i}": tok, f"q{i}": q, f"hard{i}": dec,
                        f"alpha{i}": np.int64(RL.alphabet_size(pair[i - 1]))})
        cp = {f"recipient_{i}": bool(np.array_equal(out[f"hard{i}"], T[f"d{i}"])) for i in (1, 2)}
        if not all(cp.values()):
            raise SystemExit(f"CLASS PRESERVATION FAILED {n}: {cp}")
        pol = RL.policy_pair_to_dict(pair)
        FN.save_unit(U(n), {"policy.json": lambda p: p.write_text(json.dumps(pol)),
                            "release.npz": npz_writer(out)},
                     {"seed": k, "config": cid, "cfg": c, "receipts": receipts, "class_preservation_all_rows": cp,
                      "fingerprint": RL.fingerprint(pair), "token_states_fit": {
                          i: int(len(np.unique(out[f"tok{i}"][tr]))) for i in (1, 2)},
                      "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0})
        event("unit complete", unit=n)


# ------------------------------------------------------------------ inner audits (SELECTION_AND_AUDIT_LOCK)
def stage_inner(D, shard_spec=None):
    from dpc import audit as AU
    jobs = [(k, c) for k in SEEDS for c in locked_bank_ids()]
    for k, cid in shard(jobs, shard_spec):
        n = f"inner__{unit_for(k, cid)}"
        if done(n):
            continue
        t0, c0 = time.time(), time.process_time()
        r = AU.inner_unit(parse_id(cid)["kind"], k, cid, D)
        files = r.pop("_files", {})
        r.update({"of": unit_for(k, cid), "config": cid, "seed": k, "wall_s": time.time() - t0,
                  "cpu_s": time.process_time() - c0})
        FN.save_unit(U(n), files, r)
        event("unit complete", unit=n)


def parity_ok():
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from dpc.lock import verify_lock
    v = verify_lock(Path(a.lock), stage=a.stage)
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    D = DA.load()
    assert D["sealed"]
    event(f"start {a.stage}", shard=a.shard, pid=os.getpid(), lock=v["lock"])
    t0, c0 = time.time(), time.process_time()
    fn = {"admit": stage_admit, "partition": stage_partition, "fit": stage_fit}
    if a.stage in fn:
        fn[a.stage](D, a.shard)
    elif a.stage in LATE:
        mod, f = LATE[a.stage]
        getattr(importlib.import_module(mod), f)(D, a.shard)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    ledger(a.stage, a.shard, time.time() - t0, time.process_time() - c0)
    event(f"end {a.stage}", shard=a.shard, pid=os.getpid())


if __name__ == "__main__":
    main()
