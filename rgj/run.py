"""Runner for the refreshed guarded joint study (atomic, hash-verified units; resumable; job sharding for 2 workers).

    env OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m rgj.run --lock results/pcrl_refreshed_guarded_joint_v1/CODE_LOCK.json \
        --stage <stage> [--shard i/n]

Stages (PROTOCOL.md section 5):
  parity     engineering receipts (beta = 0, lambda = 0 identity; environment parity vs the predecessor's U)
  taskline   task-only line from the shared warm start, 40 epochs, checkpoints every 5 (U-B = epoch 20; U = 20 + e_LR)
  stageB     L-O and L-R x beta x seed (20 epochs, checkpoints 5/10/15/20)
  inner      inner audit (AUDIT_FIT -> INNER_SELECTION) of every complete candidate release lacking one
  selectB    freeze L-R (and L-O) per seed
  calibrate  local surrogate budgets c_i from the frozen L-R (fresh critics, CALIB rows)
  stageC     J-G, L-G, J-R, J-O x beta x seed from the frozen L-R checkpoint + task-only/beta=0 receipts
  baselines  official LEACE on U, FARE re-heads (rgj.baselines, audit/baseline owner)
  selectC    controls, C*, J-G nomination (rgj.select)
  tracking   critic-gap diagnosis at aligned frozen snapshots (rgj.critic_track)
  whiten     inner-only transform diagnostic (rgj.whiten_diag)
  ablation   optional registered diagnostic: J-G with the capped transform, beta 0.1, 3 seeds
The development assessment is never touched here (rgj.assess refuses without a pushed EVALUATION_LOCK.json).
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

from rgj import data as DA
from rgj import finalize as FN
from rgj import train as T

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private" / "rgj_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
PRED = HOME / "PCRL_eval_cache_private" / "jcv_v1" / "run" / "units"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_refreshed_guarded_joint_v1"
SEEDS = (0, 1, 2)
BETAS = T.HP["betas"]
STAGE_B_ARMS = ("L-O", "L-R")
STAGE_C_ARMS = ("J-G", "L-G", "J-R", "J-O")
TASKLINE_EPOCHS = 40
MID_STEP = 10 * 76 + 1
WARM_SHA = {0: None, 1: None, 2: None}   # verified against the predecessor COMPLETE.json at load


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


def bname(b):
    return f"b{b:g}"


def ck_name(stage, k, arm, b, e):
    return f"ck__{stage}__s{k}__{arm}__{bname(b)}__e{e}"


def run_name(stage, k, arm, b):
    return f"run__{stage}__s{k}__{arm}__{bname(b)}"


def load_warm(k):
    d = PRED / f"warm__s{k}"
    if not FN.unit_complete(d):
        raise SystemExit(f"REFUSED: predecessor warm start {d.name} fails its COMPLETE.json hashes")
    return torch.load(d / "warm.pt")


def model_from(state, k):
    m = T.Model(T.D_IN, T.KS, k)
    m.load_state_dict(state)
    return m


# ------------------------------------------------------------------ unit writers
def save_release_unit(name, state, k, D, record, critics=None):
    model = model_from(state, k)
    out, heads, meta = FN.finalize_model(model, D)
    files = {"model.pt": lambda p: torch.save(state, p),
             "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out)}
    for i, h in heads.items():
        files[f"head_{i}.joblib"] = (lambda p, h=h: joblib.dump(h, p))
    if critics is not None:
        files["critics.pt"] = lambda p: torch.save(critics, p)
    record = {**record, "unit": name, "seed": k, "heads": meta, "map": "identity (no erasure)"}
    FN.save_unit(U(name), files, record)
    event("unit complete", unit=name)


def save_run_unit(name, diag, captures, final, record):
    files = {"final.pt": lambda p: torch.save(final, p)}
    if captures:
        files["captures.pt"] = lambda p: torch.save(captures, p)
    FN.save_unit(U(name), files, {**record, "unit": name, "diag": diag})
    event("unit complete", unit=name)


def shard(jobs, spec):
    if not spec:
        return jobs
    i, n = map(int, spec.split("/"))
    return [j for t, j in enumerate(jobs) if t % n == i]


# ------------------------------------------------------------------ parity (engineering)
def stage_parity(D, shard_spec=None):
    data = T.TData(D)
    jobs = [(k, a) for k in SEEDS for a in ("ENV",) + STAGE_B_ARMS + STAGE_C_ARMS]
    for k, a in shard(jobs, shard_spec):
        name = f"parity__s{k}__{a}"
        if done(name):
            continue
        warm = load_warm(k)
        t0 = time.time()
        if a == "ENV":
            m, diag, _ = T.task_line(warm, data, k, 20)
            ref = torch.load(PRED / f"nn__s{k}__U" / "model.pt")
            ok = all(torch.equal(m.state_dict()[x], ref[x]) for x in ref)
            r = {"check": "task line (20 epochs, stage-B order) == predecessor U (environment)", "model_bitwise": ok}
        else:
            stage = "B" if a in STAGE_B_ARMS else "C"
            tm, _, _ = T.task_line(warm, data, k, 20, stage=stage)
            m, diag, _, _, _ = T.train_run(a, 0.0, warm, data, k, stage, budgets=(0.0, 0.0),
                                           critic_head=T.head_of(warm))
            ok = all(torch.equal(m.state_dict()[x], tm.state_dict()[x]) for x in tm.state_dict())
            r = {"check": f"{a} at beta=0, lambda=0 == task-only continuation from the same initialisation "
                          f"(warm start, stage-{stage} data order; initialisation-specific receipt)",
                 "model_bitwise": ok, "critic_online_updates": diag["critic_online_updates"],
                 "critic_refit_updates": diag["critic_refit_updates"], "lambda_trace": diag["lambda_trace"]}
        r.update({"seed": k, "arm": a, "pass": bool(ok), "wall_s": time.time() - t0})
        FN.save_unit(U(name), {}, r)
        event("parity", unit=name, ok=bool(ok))
        if not ok:
            raise SystemExit(f"PARITY FAILED: {name}")


def parity_passed():
    return all(done(f"parity__s{k}__{a}") and rec(f"parity__s{k}__{a}")["pass"]
               for k in SEEDS for a in ("ENV",) + STAGE_B_ARMS + STAGE_C_ARMS)


# ------------------------------------------------------------------ task line
def stage_taskline(D, shard_spec=None):
    data = T.TData(D)
    for k in shard(list(SEEDS), shard_spec):
        names = [f"tl__s{k}__e{e}" for e in range(5, TASKLINE_EPOCHS + 1, 5)]
        if all(done(n) for n in names):
            continue
        t0, c0 = time.time(), time.process_time()
        m, diag, ck = T.task_line(load_warm(k), data, k, TASKLINE_EPOCHS)
        for e, n in zip(range(5, TASKLINE_EPOCHS + 1, 5), names):
            if not done(n):
                save_release_unit(n, ck[e]["model"], k, D, {"arm": "TASK", "beta": 0.0, "epoch": e,
                                                             "role": "U-B" if e == 20 else "task-only line",
                                                             "from": f"warm__s{k} (predecessor, hash-verified)"})
        event("taskline done", seed=k, wall_s=time.time() - t0, cpu_s=time.process_time() - c0)


# ------------------------------------------------------------------ protection runs
def do_run(D, data, stage, k, arm, b, init_state, init_critics=None, budgets=None, init_desc="", tkind="floored",
           capture=False, tag=""):
    rn = run_name(stage, k, arm, b) + tag
    cks = [ck_name(stage, k, arm, b, e) + tag for e in T.HP["checkpoints"]]
    if done(rn) and all(done(c) for c in cks):
        return
    t0, c0 = time.time(), time.process_time()
    head = T.head_of(load_warm(k))
    model, diag, ck, cap, final = T.train_run(arm, b, init_state, data, k, stage, init_critics=init_critics,
                                              budgets=budgets, tkind=tkind, critic_head=head,
                                              capture_steps=(1, MID_STEP) if capture else ())
    rescue = None
    if diag["nonfinite"] > 0:   # registered technical retry: one half-learning-rate rerun, kept receipts
        rescue = {"reason": "nonfinite", "first_attempt_nonfinite": diag["nonfinite"]}
        model, diag, ck, cap, final = T.train_run(arm, b, init_state, data, k, stage, init_critics=init_critics,
                                                  budgets=budgets, tkind=tkind, lr=T.HP["sgd_lr"] / 2,
                                                  critic_head=head, capture_steps=(1, MID_STEP) if capture else ())
    diag["rescue"] = rescue
    base = {"stage": stage, "arm": arm, "beta": b, "seed": k, "init": init_desc, "tkind": tkind, "budgets": budgets}
    for e, cn in zip(T.HP["checkpoints"], cks):
        if not done(cn):
            c = ck[e]
            save_release_unit(cn, c["model"], k, D, {**base, "epoch": e, "step": c["step"], "lam": c.get("lam"),
                                                     "critics_alignment": "critics.pt holds the critics in use at this "
                                                     "checkpoint; they were last updated on theta_{step-1} views, i.e. "
                                                     "before the epoch's final encoder step (review A5)"},
                              critics={kk: c[kk] for kk in ("critics", "transforms", "lam") if kk in c})
    save_run_unit(rn, diag, cap, final, {**base, "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0})


def stage_B(D, shard_spec=None):
    data = T.TData(D)
    jobs = [(k, a, b) for k in SEEDS for a in STAGE_B_ARMS for b in BETAS]
    for k, a, b in shard(jobs, shard_spec):
        do_run(D, data, "B", k, a, b, load_warm(k), init_desc=f"warm__s{k}")


def selB():
    return json.loads((RUN / "selection_B.json").read_text())


def lr_init(k):
    """Frozen L-R checkpoint of seed k -> (model state, critics dict or None, description, e_LR)."""
    s = selB()[str(k)]["L-R"]
    if s["status"] == "TASK_ONLY_ALIAS":
        return torch.load(U(s["unit"]) / "model.pt"), None, s["unit"], 20
    crit = torch.load(U(s["unit"]) / "critics.pt")
    return torch.load(U(s["unit"]) / "model.pt"), crit, s["unit"], s["epoch"]


def budgets_of(k):
    r = rec(f"calib__s{k}")
    return tuple(r["c"])


def stage_C(D, shard_spec=None, arms=STAGE_C_ARMS, betas=BETAS, tkind="floored", tag=""):
    data = T.TData(D)
    jobs = [(k, a, b) for k in SEEDS for a in arms for b in betas]
    if not tag:
        jobs = [(k, "TC", 0.0) for k in SEEDS] + jobs
    for k, a, b in shard(jobs, shard_spec):
        if selB()[str(k)]["L-R"]["status"] == "NO_VALID_REFERENCE":
            event("stage C skipped: no valid reference", seed=k)
            continue
        st, crit, desc, e_lr = lr_init(k)
        if a == "TC":
            name = f"tc__s{k}"
            if done(name):
                continue
            tm, _, tck = T.task_line(st, data, k, 20, stage="C")
            parity = {}
            for arm in STAGE_C_ARMS:
                m0, _, _, _, _ = T.train_run(arm, 0.0, st, data, k, "C", init_critics=crit, budgets=(0.0, 0.0),
                                             critic_head=T.head_of(load_warm(k)))
                parity[arm] = all(torch.equal(m0.state_dict()[x], tm.state_dict()[x]) for x in tm.state_dict())
            save_release_unit(name, tm.state_dict(), k, D, {
                "arm": "TC", "beta": 0.0, "from": desc, "epoch": 20,
                "note": "task-only continuation from the frozen L-R checkpoint (stage-C order); initialisation-"
                        "specific receipt, not U and not an alias of the original warm start",
                "beta0_parity_from_LR_init": parity, "pass": all(parity.values())})
            if not all(parity.values()):
                raise SystemExit(f"PARITY FAILED (stage C beta=0 from L-R init), seed {k}: {parity}")
            continue
        do_run(D, data, "C", k, a, b, st, init_critics=crit, budgets=budgets_of(k) if T.ARMS[a]["guard"] else None,
               init_desc=desc, tkind=tkind, capture=True, tag=tag)


# ------------------------------------------------------------------ inner audit
def candidate_units():
    out = []
    for d in sorted(UNITS.glob("*")):
        n = d.name
        if (n.startswith("ck__") or n.startswith("tl__") or n.startswith("lc__") or n.startswith("tc__")) and done(n):
            out.append(n)
    return out


def inner_utility(z, D):
    v = D["idx"]["INNER_SELECTION"]
    tr = D["idx"]["DEFENSE_FIT"]
    res = {}
    for i, t in enumerate(FN.TASKS):
        y = D["y"][t]
        const = int(np.argmax(np.bincount(y[tr])))
        res[i] = {"acc": float((z[f"hard{i + 1}"][v] == y[v]).mean()), "const_acc": float((y[v] == const).mean())}
    return res


def stage_inner(D, shard_spec=None):
    from rgj import audit as AU
    todo = [n for n in candidate_units() if not done(f"inner__{n}")]
    for n in shard(todo, shard_spec):
        z = np.load(U(n) / "release.npz")
        assert np.array_equal(z["row_id"], D["row_id"])
        V = FN.views_from_release(z)
        t0 = time.time()
        r = {"of": n, "recovery": AU.inner_audit({w: V[w] for w in ("v1", "v2", "pair")}, D),
             "utility": inner_utility(V["out"], D), "wall_s": None}
        r["wall_s"] = time.time() - t0
        FN.save_unit(U(f"inner__{n}"), {}, r)
        event("unit complete", unit=f"inner__{n}")


# ------------------------------------------------------------------ calibration of local budgets
def stage_calibrate(D):
    data = T.TData(D)
    H = T.entropy(data.prior)
    logprior = torch.tensor(np.log(data.prior), dtype=torch.float32)
    for k in SEEDS:
        name = f"calib__s{k}"
        if done(name):
            continue
        s = selB()[str(k)]["L-R"]
        if s["status"] == "NO_VALID_REFERENCE":
            continue
        st, _, desc, _ = lr_init(k)
        model = model_from(st, k)
        banks = {v: {kk: T.new_critic(k, v, kk, "calib") for kk in T.KINDS} for v in T.VIEWS}
        Tm = {v: None for v in T.VIEWS}
        head = T.head_of(load_warm(k))
        receipts = T.refit_banks(model, banks, Tm, data, k, 999, "floored", logprior, H, head, tag="calib")
        cal = T.calib_recovery(model, banks, Tm, data, logprior, H, head)
        c = [cal["v1"]["R"] + T.HP["c_margin"], cal["v2"]["R"] + T.HP["c_margin"]]
        FN.save_unit(U(name), {}, {"seed": k, "reference": desc, "status": s["status"], "calib": cal,
                                   "c": c, "margin": T.HP["c_margin"], "receipts": receipts,
                                   "note": "fresh critics (bounded refit on CRITIC_FIT, choice on CRITIC_VAL) of the "
                                           "frozen L-R release views; R on CALIB rows; c_i = R_i + 0.005"})
        event("unit complete", unit=name, c=c)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--shard", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from rgj.lock import verify_lock
    require = {"baselines": ["rgj/baselines.py"], "selectC": ["rgj/baselines.py"], "tracking": ["rgj/critic_track.py"],
               "whiten": ["rgj/whiten_diag.py"]}.get(a.stage, [])
    v = verify_lock(Path(a.lock), require=require)
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    D = DA.load()
    if a.stage not in ("parity",) and not parity_passed():
        raise SystemExit("REFUSED: engineering parity has not passed")
    event(f"start {a.stage}", shard=a.shard, pid=os.getpid())
    if a.stage == "parity":
        stage_parity(D, a.shard)
    elif a.stage == "taskline":
        stage_taskline(D, a.shard)
    elif a.stage == "stageB":
        stage_B(D, a.shard)
    elif a.stage == "inner":
        stage_inner(D, a.shard)
    elif a.stage == "selectB":
        from rgj.select import select_B
        select_B(D)
    elif a.stage == "calibrate":
        stage_calibrate(D)
    elif a.stage == "stageC":
        stage_C(D, a.shard)
    elif a.stage == "baselines":
        from rgj.select import run_baselines
        run_baselines(D, a.shard)
    elif a.stage == "selectC":
        from rgj.select import select_C
        select_C(D)
    elif a.stage == "tracking":
        from rgj.critic_track import run_tracking
        run_tracking(D, a.shard)
    elif a.stage == "whiten":
        from rgj.whiten_diag import run_whiten
        run_whiten(D)
    elif a.stage == "ablation":
        stage_C(D, a.shard, arms=("J-G",), betas=(0.1,), tkind="capped", tag="__capped")
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    event(f"end {a.stage}", shard=a.shard, pid=os.getpid())


if __name__ == "__main__":
    main()
