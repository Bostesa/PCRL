"""Runner for the strength-matched feedback study (atomic hash-verified units; resumable; 2-worker sharding).

    env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m smf.run --lock <lock json> --stage <stage> [--shard i/n]

Stages: warm | parity | taskline | raw | phaseA | inner | selectA | calibrate | preflight | phaseB | baselines | selectB |
tracking. The new assessment is never touched here (labels are sealed at load; smf.assess unseals after the pushed
EVALUATION_LOCK.json).
Units: warm__s{k}; tl__s{k}__e{20,40} (U); raw__s{k}__{RAW-J|RAW-L}__b{beta}__e{20,40} (+ run__raw__...);
A__s{k}__{NJ|NL}__{SCHED}__r{rho}__e20 (+ run__A__...); calib__s{k}; preflight__s{k};
B__s{k}__{J-F|L-F|J-N|L-N}__r{rho}__e20 (+ run__B__...); tplB__s{k}__... (ONLINE_MATCHED templates, if selected);
lc__s{k}__E and fare__* (smf.baselines); inner__<unit>; track__*.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from jcv import train as JT
from rgj import finalize as FN
from rgj import train as RT
from smf import data as DA
from smf import train as T

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private" / "smf_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_strength_matched_feedback_v1"
SEEDS = (0, 1, 2)
RHOS = T.HP["rhos"]
RAW_BETAS = T.HP["raw_betas"]
SCHEDS = T.SCHEDULES
BASES = {"NJ": "joint", "NL": "local"}
B_ARMS = {"J-F": ("joint", True), "L-F": ("local", True), "J-N": ("joint", False), "L-N": ("local", False)}
STEPS_PER_EPOCH = 61
MID_STEP = 10 * STEPS_PER_EPOCH + 1


def event(msg, **kw):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "ACTIVITY_LOG.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": msg, **kw}) + "\n")


def U(name):
    return UNITS / name


def done(name):
    return FN.unit_complete(U(name))


def rec(name):
    return json.loads((U(name) / "record.json").read_text())


def g(x):
    return f"{x:g}"


def shard(jobs, spec):
    if not spec:
        return jobs
    i, n = map(int, spec.split("/"))
    return [j for t, j in enumerate(jobs) if t % n == i]


def model_from(state, k):
    m = T.Model(T.D_IN, T.KS, k)
    m.load_state_dict(state)
    return m


def load_warm(k):
    return torch.load(U(f"warm__s{k}") / "warm.pt")


def save_release_unit(name, state, k, D, record, critics=None):
    out, heads, meta = FN.finalize_model(model_from(state, k), D)
    files = {"model.pt": lambda p: torch.save(state, p),
             "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out)}
    for i, h in heads.items():
        files[f"head_{i}.joblib"] = (lambda p, h=h: joblib.dump(h, p))
    if critics is not None:
        files["critics.pt"] = lambda p: torch.save(critics, p)
    FN.save_unit(U(name), files, {**record, "unit": name, "seed": k, "heads": meta, "map": "identity (no erasure)"})
    event("unit complete", unit=name)


def save_run_unit(name, diag, captures, final, record):
    files = {"final.pt": lambda p: torch.save(final, p)}
    if captures:
        files["captures.pt"] = lambda p: torch.save(captures, p)
    FN.save_unit(U(name), files, {**record, "unit": name, "diag": diag})
    event("unit complete", unit=name)


# ------------------------------------------------------------------ warm starts (fresh, NEW_DEFENSE_FIT only)
def stage_warm(D, shard_spec=None):
    data = RT.TData(D)
    for k in shard(list(SEEDS), shard_spec):
        if done(f"warm__s{k}"):
            continue
        t0 = time.time()
        log = []
        m = JT.warm_start(T.D_IN, T.KS, data, k, log=log.append)
        st = m.state_dict()
        FN.save_unit(U(f"warm__s{k}"), {"warm.pt": lambda p: torch.save(st, p)},
                     {"seed": k, "rows": "NEW_DEFENSE_FIT (15,434)", "schedule": "jcv.train.warm_start (Adam 1e-3, 20 "
                      "epochs, batch 256, rng([seed, 999, ep]))", "log": log, "wall_s": time.time() - t0})
        event("unit complete", unit=f"warm__s{k}")


def task_spec():
    return {"base": "none", "schedule": "ONLINE", "update": "task"}


# ------------------------------------------------------------------ engineering parity (rho = 0 / beta = 0)
def stage_parity(D, shard_spec=None):
    data = RT.TData(D)
    jobs = [(k, x) for k in SEEDS for x in ["RAW-J", "RAW-L"] + [f"{b}|{s}" for b in BASES for s in SCHEDS] + list(B_ARMS)]
    for k, x in shard(jobs, shard_spec):
        name = f"parity__s{k}__{x.replace('|', '__')}"
        if done(name):
            continue
        warm = load_warm(k)
        head = T.head_of(warm)
        stage = "B" if x in B_ARMS else "A"
        tm, _, _, _, _ = T.train_run(task_spec(), 0.0, warm, data, k, stage, head, n_epochs=4, ckpt_epochs=())
        if x.startswith("RAW"):
            m, d, _, _, _ = RT.train_run({"RAW-J": "J-O", "RAW-L": "L-O"}[x], 0.0, warm, data, k, "B", n_epochs=4,
                                         critic_head=head, ckpt_epochs=())
        elif x in B_ARMS:
            base, fb = B_ARMS[x]
            ctrl = {"receipt0": T.probe_receipt(model_from(warm, k), data, head, k, "parity"), "b": [0.5, 0.5]}
            m, d, _, _, _ = T.train_run({"base": base, "schedule": "ONLINE", "feedback": fb}, 0.0, warm, data, k, "B",
                                        head, n_epochs=4, ckpt_epochs=(), controller=ctrl)
        else:
            b, s = x.split("|")
            cnt = {f"{v}|{kk}": 50 for v in RT.VIEWS for kk in RT.KINDS}
            m, d, _, _, _ = T.train_run({"base": BASES[b], "schedule": s}, 0.0, warm, data, k, "A", head, n_epochs=4,
                                        ckpt_epochs=(), matched_counts=cnt if s == "ONLINE_MATCHED" else None)
        ok = all(torch.equal(m.state_dict()[q], tm.state_dict()[q]) for q in tm.state_dict())
        FN.save_unit(U(name), {}, {"seed": k, "arm": x, "pass": bool(ok), "check": "rho=0 (beta=0) equals the task-only "
                                   "continuation from the same initialisation, 4 epochs, with critics/controller/extra "
                                   "updates running on their own RNG streams"})
        event("parity", unit=name, ok=bool(ok))
        if not ok:
            raise SystemExit(f"PARITY FAILED {name}")


def parity_passed():
    xs = ["RAW-J", "RAW-L"] + [f"{b}__{s}" for b in BASES for s in SCHEDS] + list(B_ARMS)
    return all(done(f"parity__s{k}__{x}") and rec(f"parity__s{k}__{x}")["pass"] for k in SEEDS for x in xs)


# ------------------------------------------------------------------ U and raw controls (40 epochs)
def stage_taskline(D, shard_spec=None):
    data = RT.TData(D)
    for k in shard(list(SEEDS), shard_spec):
        names = [f"tl__s{k}__e{e}" for e in (20, 40)]
        if all(done(n) for n in names):
            continue
        m, d, ck, _, _ = T.train_run(task_spec(), 0.0, load_warm(k), data, k, "A", None, n_epochs=40, ckpt_epochs=(20, 40))
        for e, n in zip((20, 40), names):
            if not done(n):
                save_release_unit(n, ck[e]["model"], k, D, {"arm": "U", "epoch": e, "role": "U (Phase A ref)" if e == 20
                                                            else "U (Phase B / final utility reference)"})


def stage_raw(D, shard_spec=None):
    data = RT.TData(D)
    jobs = [(k, a, b) for k in SEEDS for a in ("RAW-J", "RAW-L") for b in RAW_BETAS]
    for k, a, b in shard(jobs, shard_spec):
        names = [f"raw__s{k}__{a}__b{g(b)}__e{e}" for e in (20, 40)]
        rn = f"run__raw__s{k}__{a}__b{g(b)}"
        if done(rn) and all(done(n) for n in names):
            continue
        warm = load_warm(k)
        t0 = time.time()
        m, d, ck, cap, fin = RT.train_run({"RAW-J": "J-O", "RAW-L": "L-O"}[a], b, warm, RT.TData(D), k, "B",
                                          n_epochs=40, critic_head=T.head_of(warm), ckpt_epochs=(20, 40))
        if d["nonfinite"]:
            m, d, ck, cap, fin = RT.train_run({"RAW-J": "J-O", "RAW-L": "L-O"}[a], b, warm, RT.TData(D), k, "B",
                                              n_epochs=40, critic_head=T.head_of(warm), ckpt_epochs=(20, 40),
                                              lr=T.HP["sgd_lr"] / 2)
            d["rescue"] = "half-lr retry after nonfinite"
        for e, n in zip((20, 40), names):
            if not done(n):
                save_release_unit(n, ck[e]["model"], k, D, {"arm": a, "beta": b, "epoch": e,
                                                            "recipe": "rgj.train J-O/L-O unchanged (raw penalty)"},
                                  critics={kk: ck[e][kk] for kk in ("critics", "transforms", "lam") if kk in ck[e]})
        save_run_unit(rn, d, None, fin, {"arm": a, "beta": b, "seed": k, "wall_s": time.time() - t0})


# ------------------------------------------------------------------ Phase A normalized arms
def a_name(k, b, s, r):
    return f"A__s{k}__{b}__{s}__r{g(r)}__e20"


def a_run(k, b, s, r):
    return f"run__A__s{k}__{b}__{s}__r{g(r)}"


def wait_for(name, timeout=3600):
    t0 = time.time()
    while not done(name):
        if time.time() - t0 > timeout:
            raise SystemExit(f"timed out waiting for {name}")
        time.sleep(5)


def do_norm_run(D, data, stage, k, spec, r, init_state, run_name, ck_name, init_critics=None, matched_counts=None,
                controller=None, capture=False, record=None):
    if done(run_name) and (ck_name is None or done(ck_name)):
        return
    head = T.head_of(load_warm(k))
    t0, c0 = time.time(), time.process_time()
    kw = dict(n_epochs=20, init_critics=init_critics, ckpt_epochs=(20,), matched_counts=matched_counts,
              controller=controller, capture_steps=(1, MID_STEP) if capture else ())
    m, d, ck, cap, fin = T.train_run(spec, r, init_state, data, k, stage, head, **kw)
    if d["nonfinite"]:
        m, d, ck, cap, fin = T.train_run(spec, r, init_state, data, k, stage, head, lr=T.HP["sgd_lr"] / 2, **kw)
        d["rescue"] = "half-lr retry after nonfinite"
    base = {"stage": stage, "spec": spec, "rho": r, "seed": k, **(record or {})}
    if ck_name is not None and not done(ck_name):
        c = ck[20]
        save_release_unit(ck_name, c["model"], k, D, {**base, "epoch": 20, "w": c.get("w"),
                                                      "critics_alignment": "critics last updated on theta_{step-1}"},
                          critics={kk: c[kk] for kk in ("critics", "transforms", "critic_head", "w") if kk in c})
    save_run_unit(run_name, d, cap, fin, {**base, "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0})


def stage_A(D, shard_spec=None):
    data = RT.TData(D)
    order = {"REFRESHED": 0, "ONLINE": 1, "ONLINE_MATCHED": 2}
    jobs = sorted([(k, b, s, r) for k in SEEDS for b in BASES for s in SCHEDS for r in RHOS], key=lambda j: order[j[2]])
    for k, b, s, r in shard(jobs, shard_spec):
        cnt = None
        if s == "ONLINE_MATCHED":
            tpl = a_run(k, b, "REFRESHED", r)
            wait_for(tpl)
            cnt = T.refit_counts(rec(tpl)["diag"])
        do_norm_run(D, data, "A", k, {"base": BASES[b], "schedule": s}, r, load_warm(k), a_run(k, b, s, r),
                    a_name(k, b, s, r), matched_counts=cnt, capture=(r == 0.75 and b == "NJ"),
                    record={"arm": b, "schedule": s, "matched_template": tpl if cnt else None})


# ------------------------------------------------------------------ inner audit
def stage_inner(D, shard_spec=None):
    from smf import audit as AU
    pref = ("tl__", "raw__", "A__", "B__", "lc__")
    todo = [d.name for d in sorted(UNITS.glob("*")) if d.name.startswith(pref) and done(d.name)
            and not done(f"inner__{d.name}")]
    for n in shard(todo, shard_spec):
        z = np.load(U(n) / "release.npz")
        assert np.array_equal(z["row_id"], D["row_id"])
        V = FN.views_from_release(z)
        t0 = time.time()
        r = {"of": n, "recovery": AU.inner_audit({w: V[w] for w in ("v1", "v2", "pair")}, D),
             "utility": inner_utility(V["out"], D)}
        r["wall_s"] = time.time() - t0
        FN.save_unit(U(f"inner__{n}"), {}, r)
        event("unit complete", unit=f"inner__{n}")


def inner_utility(o, D):
    v, tr = D["idx"]["INNER_SELECTION"], D["idx"]["DEFENSE_FIT"]
    res = {}
    for i, t in enumerate(FN.TASKS):
        y = D["y"][t]
        const = int(np.argmax(np.bincount(y[tr])))
        res[i] = {"acc": float((o[f"hard{i + 1}"][v] == y[v]).mean()), "const_acc": float((y[v] == const).mean())}
    return res


# ------------------------------------------------------------------ Phase B
def selA():
    return json.loads((RUN / "selection_A.json").read_text())


def reference(k):
    s = selA()["seeds"][str(k)]["reference"]
    crit = torch.load(U(s["unit"]) / "critics.pt") if s["status"] == "NOMINEE" else None
    return torch.load(U(s["unit"]) / "model.pt"), crit, s


def stage_calibrate(D):
    data = RT.TData(D)
    for k in SEEDS:
        name = f"calib__s{k}"
        if done(name):
            continue
        st, _, s = reference(k)
        receipt = T.probe_receipt(model_from(st, k), data, T.head_of(load_warm(k)), k, "reference")
        b = T.targets_from(receipt)
        FN.save_unit(U(name), {}, {"seed": k, "reference": s, "receipt": receipt, "b": b,
                                   "floor_active": [x == T.HP["target_floor"] for x in b],
                                   "rule": "b_i = max(0.5, calib AUC_i(reference) - 0.01)"})
        event("unit complete", unit=name, b=b)


def b_name(k, arm, r):
    return f"B__s{k}__{arm}__r{g(r)}__e20"


def b_run(k, arm, r, tpl=False):
    return f"{'tplB' if tpl else 'run__B'}__s{k}__{arm}__r{g(r)}"


def stage_B(D, shard_spec=None, arms=tuple(B_ARMS), rhos=RHOS):
    data = RT.TData(D)
    sched = selA()["schedule"]["selected"]
    jobs = [(k, a, r) for k in SEEDS for a in arms for r in rhos]
    for k, a, r in shard(jobs, shard_spec):
        if selA()["seeds"][str(k)]["reference"]["status"] == "NO_VALID_REFERENCE":
            continue
        st, crit, sref = reference(k)
        c = rec(f"calib__s{k}")
        ctrl = {"receipt0": c["receipt"], "b": c["b"]}
        base, fb = B_ARMS[a]
        cnt, tpl = None, None
        if sched == "ONLINE_MATCHED":       # registered template: REFRESHED twin on the same init / rho / arm rule
            tpl = b_run(k, a, r, tpl=True)
            do_norm_run(D, data, "B", k, {"base": base, "schedule": "REFRESHED", "feedback": fb}, r, st, tpl, None,
                        init_critics=crit, controller=ctrl, record={"arm": a, "role": "ONLINE_MATCHED template"})
            cnt = T.refit_counts(rec(tpl)["diag"])
        do_norm_run(D, data, "B", k, {"base": base, "schedule": sched, "feedback": fb}, r, st, b_run(k, a, r),
                    b_name(k, a, r), init_critics=crit, matched_counts=cnt, controller=ctrl, capture=True,
                    record={"arm": a, "schedule": sched, "reference": sref["unit"], "matched_template": tpl})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from smf.lock import verify_lock
    v = verify_lock(Path(a.lock), stage=a.stage)
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    D = DA.load()
    assert D["sealed"]
    if a.stage not in ("warm", "parity") and not parity_passed():
        raise SystemExit("REFUSED: engineering parity has not passed")
    event(f"start {a.stage}", shard=a.shard, pid=os.getpid())
    fn = {"warm": stage_warm, "parity": stage_parity, "taskline": stage_taskline, "raw": stage_raw, "phaseA": stage_A,
          "inner": stage_inner, "phaseB": stage_B}
    if a.stage in fn:
        fn[a.stage](D, a.shard)
    elif a.stage == "selectA":
        from smf.select import select_A
        select_A(D)
    elif a.stage == "calibrate":
        stage_calibrate(D)
    elif a.stage == "preflight":
        from smf.preflight import run_preflight
        run_preflight(D)
    elif a.stage == "baselines":
        from smf.select import run_baselines
        run_baselines(D, a.shard)
    elif a.stage == "selectB":
        from smf.select import select_B
        select_B(D)
    elif a.stage == "tracking":
        from smf.track import run_tracking
        run_tracking(D, a.shard)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    event(f"end {a.stage}", shard=a.shard, pid=os.getpid())


if __name__ == "__main__":
    main()
