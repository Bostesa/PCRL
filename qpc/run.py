"""Runner for the confidence-capacity study (atomic hash-verified units; resumable; at most two sharded workers).

    <PRIVATE_CACHE>/qpc_v1/run/work.sh <LOCK> <stage> [i/n]
      == python -m qpc.sema --label A:<stage> -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.run
             --lock results/pcrl_confidence_capacity_v1/<LOCK>.json --stage <stage> [--shard i/n]

Stages (each refuses unless the latest named lock + amendments verify and are on origin, and every study module the
stage loaded is locked with its hash -- qpc.lock.check_loaded_modules):
  SOURCE_ADMISSION_LOCK     admit      teacher outputs tea__s{k}__{U,RAW-J_b0.3} rebuilt by qpc.admit with parity
                                       against the admitted dpc outputs; references ref__s{k}__{E,F,F0}
  STAGE_A_LOCK              stagea     A1 a1__s{k}; per-recipient DIRECT-TASK fits dir__s{k}__r{i}__m{m};
                                       mapping pairs pol__s{k}__U_DIRECT-TASK_i{m1}o{m2} (8 rates x 3 seeds)
                            gate       A3 INNER_SELECTION utility of every Stage A code + A4 rate decision (qpc.gate)
  STAGE_B_LOCK              partition  fine__s{k} (income <= 32, occupation <= 128 per predicted class)
                            fit        pol__s{k}__<Stage B config> at the locked rates / lambdas + U|CLASS|i1o1
  AUDIT_AND_SELECTION_LOCK  inner      inner audits of every policy and reference (qpc.audit)
                            inner_src  continuous-source inner audits with composition over EVERY fitted code
                            controls   real-data null and positive controls (qpc.audit)
                            select     qpc.select.select_all
Assessment labels stay sealed (qpc.data); only qpc.assess unseals, after the pushed EVALUATION_LOCK.
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

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private" / "qpc_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_confidence_capacity_v1"
DPC_UNITS = HOME / "PCRL_eval_cache_private" / "dpc_v1" / "run" / "units"
SEEDS = (0, 1, 2)
TEACHERS = ("U", "RAW-J_b0.3")
REFS = ("E", "F", "F0")
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
STAGE_B_FAMILIES = ("FINE-TASK",) + PRIVACY
LAMS_FULL = (0.01, 0.1, 1.0)
LATE = {"inner": ("qpc.run", "stage_inner"), "inner_src": ("qpc.run", "stage_inner_src"),
        "controls": ("qpc.audit", "stage_controls"), "select": ("qpc.select", "select_all")}


# ------------------------------------------------------------------ configuration ids
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


def stagea_ids():
    from qpc import gate as GT
    return GT.stagea_ids()


def lock_protocol():
    from qpc import lock as LK
    return LK.latest().get("protocol") or {}


def stage_b_ids(protocol=None):
    """Stage B bank from the STAGE_B_LOCK protocol: {"rates": [[m1, m2], ...], "lams": [...]}."""
    sb = (protocol or lock_protocol()).get("stage_b")
    if not sb:
        return []
    ids = []
    for m1, m2 in sb["rates"]:
        ids.append(config_id("FINE-TASK", m1, m2))
        ids += [config_id(f, m1, m2, lam) for lam in sb["lams"] for f in PRIVACY]
    return ids + [config_id("CLASS")]


def scored_ids(protocol=None):
    """Every inner-selection candidate: Stage A codes, Stage B codes, continuous sources and references."""
    return stagea_ids() + stage_b_ids(protocol) + [f"SRC|{t}" for t in TEACHERS] + [f"REF|{r}" for r in REFS]


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
    from qpc import admit as AD
    AD.run(D)                           # verified copies + receipts (admission_record), inputs; refuses on any mismatch
    for k in shard(list(SEEDS), shard_spec):
        for t in TEACHERS:
            n = f"tea__s{k}__{t}"
            if done(n):
                continue
            T = AD.teacher(t, k)
            assert np.array_equal(T["row_id"], D["row_id"]), "teacher rows differ from the pinned rows"
            for i in (1, 2):
                assert np.array_equal(T[f"d{i}"], np.asarray(T[f"p{i}"]).argmax(1)), "decision != argmax p"
            par = AD.parity_with_source(t, k, T)
            if not par.get("ok"):
                raise SystemExit(f"ADMISSION PARITY FAILED {n}: {par}")
            arrs = {x: np.asarray(T[x]) for x in ("row_id", "p1", "p2", "d1", "d2", "c1", "c2", "r1", "r2") if x in T}
            save(n, {"teacher.npz": arrs},
                 {"seed": k, "teacher": t, "parity": par, "model_sha256": T["model_sha256"],
                  "admission": AD.admission_record().get("teachers", {}).get(f"{t}|{k}"),
                  "p_sha256": hashlib.sha256(np.ascontiguousarray(T["p1"]).tobytes() +
                                             np.ascontiguousarray(T["p2"]).tobytes()).hexdigest()})
        for lab in REFS:
            n = f"ref__s{k}__{lab}"
            if done(n):
                continue
            Rf = AD.reference(lab, k)
            arrs = {x: np.asarray(v) for x, v in Rf.items() if isinstance(v, np.ndarray)}
            save(n, {"reference.npz": arrs}, {"seed": k, "label": lab, "arrays": sorted(arrs),
                                              "admission": {x: v for x, v in Rf.items()
                                                            if not isinstance(v, np.ndarray)}})


# ------------------------------------------------------------------ Stage A (STAGE_A_LOCK)
def stage_stagea(D, shard_spec=None):
    from qpc import gate as GT
    from qpc import stagea as SA
    tr = fit_rows(D)
    for k in shard(list(SEEDS), shard_spec):
        T = teacher(k)
        n = f"a1__s{k}"
        if not done(n):
            from jcv.finalize import unit_complete
            src_dir = DPC_UNITS / f"pol__s{k}__U_DIRECT-TASK_m8"           # admitted source DIRECT-TASK (8, 8) policy
            if not unit_complete(src_dir):
                raise SystemExit(f"A1 source policy missing or not hash-complete: dpc pol__s{k}__U_DIRECT-TASK_m8")
            z = np.load(src_dir / "release.npz")
            src = {x: z[x] for x in z.files}
            t0, c0 = time.time(), time.process_time()
            r, files = SA.a1_unit(T, tr, src, bind_meta(k, "U|DIRECT-TASK|i8o8", D))
            r.update({"seed": k, "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0})
            if r.get("ENGINEERING_BLOCKER") or not r["parity_with_admitted_release"]["ok"]:
                save(f"{n}__FAILED", files, r)                      # failed output preserved, then stop
                raise SystemExit(f"A1 REPRODUCTION MISMATCH (engineering blocker) seed {k}: "
                                 f"{r['parity_with_admitted_release']}")
            save(n, files, r)
        pols = {}
        for i, (K, rates) in ((1, (2, GT.RATES_I)), (2, (6, GT.RATES_O))):
            for m in rates:
                n = f"dir__s{k}__r{i}__m{m}"
                if not done(n):
                    t0, c0 = time.time(), time.process_time()
                    pol, receipts = SA.recipient_fit(T[f"p{i}"][tr], T[f"d{i}"][tr], K, m, i)
                    save(n, {"policy.json": pol}, {"seed": k, "recipient": i, "m": m, "receipts": receipts,
                                                   "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0})
                pols[(i, m)] = json.loads((U(n) / "policy.json").read_text())
        for m1 in GT.RATES_I:
            for m2 in GT.RATES_O:
                cid = GT.config_id(m1, m2)
                n = unit_for(k, cid)
                if done(n):
                    continue
                t0, c0 = time.time(), time.process_time()
                r, files = SA.pair_unit(pols[(1, m1)], pols[(2, m2)], T, bind_meta(k, cid, D), tr=tr)
                r.update({"seed": k, "config": cid, "per_recipient_units": [f"dir__s{k}__r1__m{m1}",
                                                                            f"dir__s{k}__r2__m{m2}"],
                          "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0})
                save(n, files, r)


def stage_gate(D, shard_spec=None):
    """A3 + A4: INNER_SELECTION utility of every Stage A code (and the A1 versions); no attackers, no SEX."""
    from qpc import gate as GT
    from qpc import utility as UT
    per_cfg_seed, summary, a1_rows = {}, {}, []
    for cid in GT.stagea_ids():
        per = {}
        for k in SEEDS:
            T = teacher(k)
            Uu = UT.release_inner_utility({1: T["p1"], 2: T["p2"]}, {1: T["d1"], 2: T["d2"]}, D)
            z = npz(unit_for(k, cid), "release.npz")
            pres = {i: bool(np.array_equal(z[f"hard{i}"], T[f"d{i}"])) for i in (1, 2)}
            cu = UT.release_inner_utility({1: z["q1"], 2: z["q2"]}, {1: z["hard1"], 2: z["hard2"]}, D)
            gr = UT.gate_record(cu, Uu, pres)
            per[k] = {"gate": gr, "util": cu, "alpha1": int(z["alpha1"]), "alpha2": int(z["alpha2"])}
        summary[cid] = GT.summarize(per)
        per_cfg_seed[cid] = {str(k): {"gate": per[k]["gate"], "alpha1": per[k]["alpha1"], "alpha2": per[k]["alpha2"],
                                      "util": per[k]["util"]} for k in SEEDS}
    for k in SEEDS:
        T = teacher(k)
        Uu = UT.release_inner_utility({1: T["p1"], 2: T["p2"]}, {1: T["d1"], 2: T["d2"]}, D)
        a1 = rec(f"a1__s{k}")
        for ver in ("src20", "r200"):
            z = npz(f"a1__s{k}", f"release_{ver}.npz")
            pres = {i: bool(np.array_equal(z[f"hard{i}"], T[f"d{i}"])) for i in (1, 2)}
            cu = UT.release_inner_utility({1: z["q1"], 2: z["q2"]}, {1: z["hard1"], 2: z["hard2"]}, D)
            gr = UT.gate_record(cu, Uu, pres)
            for i, t in ((1, "income"), (2, "occupation")):
                fit = a1[ver]["per_recipient"][str(i)] if str(i) in a1[ver]["per_recipient"] else \
                    a1[ver]["per_recipient"][i]
                cls = [c for c in fit["per_class"] if not c["fallback"]]
                win = [next(s for s in c["starts"] if s["start"] == c["winner"]) for c in cls]
                a1_rows.append({"seed": k, "version": ver, "recipient": t,
                                "fit_objective_total": fit["objective_total"], "fit_mean_kl": fit["mean_kl_fit"],
                                "all_converged": fit["all_converged"],
                                "classes_fitted": len(cls), "classes_converged": sum(bool(w["converged"]) for w in win),
                                "rounds_used_max": max(int(w["rounds_used"]) for w in win),
                                "rounds_used_sum": sum(int(w["rounds_used"]) for w in win),
                                "stop_reasons": ";".join(sorted({str(w["stop_reason"]) for w in win})),
                                "a1_parity_ids_rule": a1["parity_with_admitted_release"]["ids_rule"],
                                "a1_parity_q_rule": a1["parity_with_admitted_release"]["q_rule"],
                                "inner_logloss": cu[t]["logloss"], "inner_brier": cu[t]["brier"],
                                "inner_ll_excess": gr[t]["ll_excess"], "inner_brier_excess": gr[t]["brier_excess"],
                                "eligible_seed_both_tasks": bool(gr["eligible"]),
                                "cpu_s_unit": a1.get("cpu_s")})
    decision = GT.select_rates(summary)
    GT.write_outputs(decision, summary, a1_rows, per_cfg_seed, PKG, RUN)
    event("gate decision", decision=decision["gate"], selected=decision["selected_rates"])
    print(json.dumps(decision, indent=1))


# ------------------------------------------------------------------ Stage B (STAGE_B_LOCK)
def stage_partition(D, shard_spec=None):
    from qpc import partition as PT
    tr = fit_rows(D)
    for k in shard(list(SEEDS), shard_spec):
        n = f"fine__s{k}"
        if done(n):
            continue
        t0, c0 = time.time(), time.process_time()
        r, files = PT.fine_unit(teacher(k), tr, caps={1: 32, 2: 128})
        r.update({"seed": k, "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0})
        save(n, files, r)


def fit_jobs(protocol=None):
    """(seed, rate-group) jobs; inside a group: CLASS (once per seed, rate group 0), FINE-TASK, then LOCAL, SEQ-12,
    SEQ-21 for every lambda, then JOINT for every lambda (witnesses are completed units of the same group)."""
    sb = (protocol or lock_protocol())["stage_b"]
    jobs = []
    for k in SEEDS:
        for gi, (m1, m2) in enumerate(sb["rates"]):
            cids = ([config_id("CLASS")] if gi == 0 else []) + [config_id("FINE-TASK", m1, m2)]
            cids += [config_id(f, m1, m2, lam) for lam in sb["lams"] for f in ("LOCAL", "SEQ-12", "SEQ-21")]
            cids += [config_id("JOINT", m1, m2, lam) for lam in sb["lams"]]
            jobs.append((k, cids))
    return jobs


def stage_fit(D, shard_spec=None):
    from qpc import compress as CP
    tr = fit_rows(D)
    S_fit = np.asarray(D["sex"][tr])
    assert (S_fit >= 0).all()
    for k, cids in shard(fit_jobs(), shard_spec):
        T = teacher(k)
        fine = json.loads((U(f"fine__s{k}") / "fine.json").read_text())
        for cid in cids:
            n = unit_for(k, cid)
            if done(n):
                continue
            c = parse_id(cid)
            wit = None
            if c["family"] == "JOINT":
                wit = {f: json.loads((U(unit_for(k, config_id(f, c["m1"], c["m2"], c["lam"]))) / "policy.json")
                                     .read_text()) for f in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")}
            t0, c0 = time.time(), time.process_time()
            r, files = CP.fit_unit(c["family"], fine, T, tr, S_fit, c["m1"], c["m2"], c["lam"],
                                   bind_meta(k, cid, D), witnesses=wit)
            r.update({"seed": k, "config": cid, "cfg": c, "wall_s": time.time() - t0,
                      "cpu_s": time.process_time() - c0})
            save(n, files, r)


# ------------------------------------------------------------------ inner audits (AUDIT_AND_SELECTION_LOCK)
def _inner(D, shard_spec, kinds):
    from qpc import audit as AU
    jobs = [(k, c) for k in SEEDS for c in scored_ids() if parse_id(c)["kind"] in kinds]
    for k, cid in shard(jobs, shard_spec):
        n = f"inner__{unit_for(k, cid)}"
        if done(n):
            continue
        t0, c0 = time.time(), time.process_time()
        r = AU.inner_unit(parse_id(cid)["kind"], k, cid, D)
        files = r.pop("_files", {})
        r.update({"of": unit_for(k, cid), "config": cid, "seed": k, "wall_s": time.time() - t0,
                  "cpu_s": time.process_time() - c0})
        save(n, files, r)


def stage_inner(D, shard_spec=None):
    _inner(D, shard_spec, ("policy", "reference"))


def stage_inner_src(D, shard_spec=None):
    missing = [f"inner__{unit_for(k, c)}" for k in SEEDS for c in scored_ids() if parse_id(c)["kind"] == "policy"
               and not done(f"inner__{unit_for(k, c)}")]
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
    from qpc import lock as LK
    stage_key = "inner" if a.stage == "inner_src" else a.stage
    v = LK.verify_lock(Path(a.lock), stage=stage_key)
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    from qpc import data as DA
    D = DA.load()
    assert D["sealed"]
    fn = {"admit": stage_admit, "stagea": stage_stagea, "gate": stage_gate, "partition": stage_partition,
          "fit": stage_fit}
    if a.stage in fn:
        f = fn[a.stage]
    elif a.stage in LATE:
        mod, name = LATE[a.stage]
        f = getattr(importlib.import_module(mod), name)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    # import the stage's code up front, then refuse if any loaded study module is unlocked or changed
    for m in {"admit": ["qpc.admit"], "stagea": ["qpc.stagea", "qpc.gate", "qpc.deploy"],
              "gate": ["qpc.gate", "qpc.utility"], "partition": ["qpc.partition"],
              "fit": ["qpc.compress", "qpc.deploy"], "inner": ["qpc.audit"], "inner_src": ["qpc.audit"],
              "controls": ["qpc.audit"], "select": ["qpc.select"]}.get(a.stage, []):
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
