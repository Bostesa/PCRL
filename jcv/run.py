"""Development ladder runner for the joint complete-view study (units with atomic receipts; resumable).

    env OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m jcv.run --lock <LOCK.json> --stage warm|train|fare|inner|select --seeds 0 1 2

Stages (PROTOCOL.md, running ladder A-E):
  warm    task-only warm start per seed (shared bitwise by every neural arm of that seed)
  train   U (task-only continuation), E (U + final LEACE), L/J/JP/S12/S21 x beta in {0.1, 1, 10}; finalize each
  fare    official FARE per purpose x registered grid on the permitted raw inputs (defense_train), heads on cells
  inner   inner audit of every candidate: utility on attacker_val (deployed head) + inner-slate recovery on attacker_val
  select  registered selection -> F0 fit at F's selected budget -> comparator C* -> SELECTION_LOCK candidates
Outer scoring is a separate module (jcv.outer) that refuses to run without a pushed SELECTION_LOCK.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from jcv import audit as AU
from jcv import data as DA
from jcv import finalize as FN
from jcv import train as T

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private" / "jcv_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_joint_complete_view_method_v1"
KS = [2, 6]
NEURAL_ARMS = ["L", "J", "JP", "S12", "S21"]
FARE_GRID = [  # admitted grid (output-aware study EXECUTION_LOCK.json "fare.grid"), unchanged
    {"id": 1, "name": "F1_task_first_k100_g0.3", "max_leaf_nodes": 100, "min_samples_leaf": 100, "gamma": 0.3, "criterion": "fair_gini_dp"},
    {"id": 2, "name": "F2_task_first_k50_g0.7", "max_leaf_nodes": 50, "min_samples_leaf": 100, "gamma": 0.7, "criterion": "fair_gini_dp"},
    {"id": 3, "name": "F3_balanced_k20_g0.85", "max_leaf_nodes": 20, "min_samples_leaf": 100, "gamma": 0.85, "criterion": "fair_gini_dp"},
    {"id": 4, "name": "F4_protect_k10_g0.9_ni1000", "max_leaf_nodes": 10, "min_samples_leaf": 1000, "gamma": 0.9, "criterion": "fair_gini_dp"},
    {"id": 5, "name": "F5_protect_k5_g0.95_ni1000", "max_leaf_nodes": 5, "min_samples_leaf": 1000, "gamma": 0.95, "criterion": "fair_gini_dp"},
    {"id": 6, "name": "F6_protect_k3_g0.95_ni1000", "max_leaf_nodes": 3, "min_samples_leaf": 1000, "gamma": 0.95, "criterion": "fair_gini_dp"},
]
FARE_SEED_BASE = 0


def event(msg, **kw):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "ACTIVITY_LOG.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": msg, **kw}) + "\n")


def U(name):
    return UNITS / name


def done(name):
    return FN.unit_complete(U(name))


def tensors(D):
    tr = D["idx"]["defense_train"]
    S = D["sex"][tr]
    g = np.random.default_rng(T.HP["guard_seed"]).choice(len(tr), T.HP["guard_size"], replace=False)
    return T.Data(torch.from_numpy(D["X"][tr]), {0: torch.from_numpy(D["y"]["income"][tr]),
                                                  1: torch.from_numpy(D["y"]["occupation_group"][tr])},
                  torch.from_numpy(S), np.bincount(S, minlength=2) / len(S), np.sort(g), 0)


# ------------------------------------------------------------------ warm start
def stage_warm(D, k):
    name = f"warm__s{k}"
    if done(name):
        return
    data = tensors(D)
    data.seed = k
    log = []
    t0, c0 = time.time(), time.process_time()
    m = T.warm_start(D["X"].shape[1], KS, data, k, log=log.append)
    B = T.guard_losses(m, data, [None, None])
    st = m.state_dict()
    rec = {"seed": k, "guard_losses": B, "budgets": {i: B[i] + T.HP["budget_slack_nats"] for i in B},
           "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0, "params": m.n_params(), "log": log,
           "init": "torch default (kaiming-uniform) under per-(seed, module) manual seeds", "HP": T.HP}
    FN.save_unit(U(name), {"warm.pt": lambda p: torch.save(st, p)}, rec)
    event("unit complete", unit=name)


def load_warm(k):
    st = torch.load(U(f"warm__s{k}") / "warm.pt")
    rec = json.loads((U(f"warm__s{k}") / "record.json").read_text())
    return st, {int(i): v for i, v in rec["budgets"].items()}


# ------------------------------------------------------------------ neural arms
def _save_release(name, model, arm, beta, k, D, diag, extra=None, erasure=None, cpu=None, wall=None):
    spec_erasure = T.arm_spec(arm)["erasure"] if erasure is None else erasure
    out, maps, heads, meta = FN.finalize_neural(model, spec_erasure, D["X"], D, KS)
    files = {"model.pt": lambda p: torch.save(model.state_dict(), p),
             "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out)}
    for i, m in maps.items():
        files[f"leace_{i}/.keep"] = (lambda p, m=m: (m.save(p.parent), p.write_text("")))
    for i, h in heads.items():
        files[f"head_{i}.joblib"] = (lambda p, h=h: joblib.dump(h, p))
    rec = {"unit": name, "arm": arm, "beta": beta, "seed": k, "erasure": spec_erasure, "diag": diag, "finalize": meta,
           "cpu_s": cpu, "wall_s": wall, "params": model.n_params(), **(extra or {})}
    FN.save_unit(U(name), files, rec)
    event("unit complete", unit=name)


def stage_train(D, k, arms=None, betas=None):
    data = tensors(D)
    data.seed = k
    st, budgets = load_warm(k)
    d_in = D["X"].shape[1]
    todo = [("U", 0.0)] + [(a, b) for a in (arms or NEURAL_ARMS) for b in (betas or T.HP["betas"])]
    for arm, beta in todo:
        name = f"nn__s{k}__{arm}" + ("" if arm == "U" else f"__b{beta:g}")
        if done(name):
            continue
        t0, c0 = time.time(), time.process_time()
        model, diag = T.train_arm(arm, beta, st, d_in, KS, data, k, budgets)
        rescue = None
        if arm != "U" and (diag["nonfinite"] > 0 or (diag["protection_steps_attempted"] > 0 and
                                                     diag["protection_steps_rejected"] == diag["protection_steps_attempted"])):
            rescue = {"reason": "nonfinite" if diag["nonfinite"] else "all protection steps rejected", "first_attempt": diag}
            model, diag = T.train_arm(arm, beta, st, d_in, KS, data, k, budgets, lr=T.HP["sgd_lr"] / 2)
        diag["rescue"] = rescue
        _save_release(name, model, arm, beta, k, D, diag, cpu=time.process_time() - c0, wall=time.time() - t0)
    # E = U's encoders + final official LEACE + refitted heads (no privacy training)
    name = f"nn__s{k}__E"
    if not done(name):
        model = T.Model(d_in, KS, k)
        model.load_state_dict(torch.load(U(f"nn__s{k}__U") / "model.pt"))
        _save_release(name, model, "E", 0.0, k, D, {"from": f"nn__s{k}__U"}, erasure=True)


# ------------------------------------------------------------------ FARE (official, permitted raw inputs)
def fare_env():
    os.environ.setdefault("OAR_RUN_UNITS", str(PRIV / "fare_cache"))
    from oar import fare_official as FO
    return FO


def _fare_release(name, k, i, cfg, D, auth, zero=False):
    if done(name):
        return
    FO = fare_env()
    tr = D["idx"]["defense_train"]
    y = D["y"]["income" if i == 0 else "occupation_group"]
    c = FO.zero_fairness(cfg) if zero else cfg
    uid = f"jcv__s{k}__p{i}__{'Z' if zero else 'c'}{cfg['id']}"
    t0 = time.time()
    model, cells, frec = FO.fit_encode_cached(uid, D["X"][tr].astype(np.float64), y[tr], D["sex"][tr],
                                              D["X"].astype(np.float64), c, seed=FARE_SEED_BASE + k, auth=auth,
                                              synthetic=False, ledger=lambda *a: None)
    ncell = int(cells.max()) + 1
    R = np.eye(ncell)[cells]
    head, hm = FN.fit_head(R, y, tr, D["idx"]["defense_val"], KS[i])
    cen, P, hard = FN.outputs(head, R)
    rec = {"unit": name, "arm": "F0" if zero else "F", "purpose": i, "config": c, "seed": k, "fare_uid": uid,
           "n_cells": ncell, "fare_rec": {kk: v for kk, v in frec.items() if kk != "cells"}, "head": hm,
           "wall_s": time.time() - t0}
    FN.save_unit(U(name), {"release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], cells=cells, r=R, c=cen,
                                                                            p=P, hard=hard),
                           "head.joblib": lambda p: joblib.dump(head, p)}, rec)
    event("unit complete", unit=name)


def stage_fare(D, k):
    from stored_model_eval.guards import FitAuthorization
    auth = FitAuthorization(execute_scientific_fits=True)
    for i in (0, 1):
        for cfg in FARE_GRID:
            _fare_release(f"fare__s{k}__p{i}__c{cfg['id']}", k, i, cfg, D, auth)


# ------------------------------------------------------------------ releases and views
def release_views(name):
    """Primary views of a two-recipient release unit: v1, v2, pair = [r_i, centred logits_i]."""
    z = np.load(U(name) / "release.npz")
    v1 = np.hstack([z["r1"], z["c1"]])
    v2 = np.hstack([z["r2"], z["c2"]])
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2]),
            "out": {"p1": z["p1"], "p2": z["p2"], "hard1": z["hard1"], "hard2": z["hard2"]}}


def fare_pair_views(n1, n2):
    a, b = np.load(U(n1) / "release.npz"), np.load(U(n2) / "release.npz")
    v1 = np.hstack([a["r"], a["c"]])
    v2 = np.hstack([b["r"], b["c"]])
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2]),
            "out": {"p1": a["p"], "p2": b["p"], "hard1": a["hard"], "hard2": b["hard"]}}


def constants(D):
    tr = D["idx"]["defense_train"]
    return {i: int(np.argmax(np.bincount(D["y"][t][tr]))) for i, t in enumerate(("income", "occupation_group"))}


# ------------------------------------------------------------------ inner audit (attacker_fit -> attacker_val)
def inner_recovery(V, D, which=("v1", "v2", "pair"), finite=False):
    f, v = D["idx"]["attacker_fit"], D["idx"]["attacker_val"]
    S = D["sex"]
    out, sel = {}, {}
    for w in which:
        if w == "pair":
            continue
        s = AU.fit_view(V[w][f], S[f], V[w][v], S[v], "inner", finite=finite)
        sel[w] = s
        out[w] = AU.macro_auc(S[v], s["Pv"], [0, 1])
    if "pair" in which:
        s = AU.fit_view(V["pair"][f], S[f], V["pair"][v], S[v], "inner", finite=finite)
        cands = [("pair:" + s["selected"], s["val_log_loss"], s["Pv"])]
        for w in ("v1", "v2"):  # ignore-other-view candidates
            if w in sel:
                cands.append((f"ignore_other:{w}:" + sel[w]["selected"], sel[w]["val_log_loss"], sel[w]["Pv"]))
        best = min(cands, key=lambda c: c[1])
        out["pair"] = AU.macro_auc(S[v], best[2], [0, 1])
        out["pair_selected"] = best[0]
    return out


def inner_utility(out, D):
    v = D["idx"]["attacker_val"]
    cst = constants(D)
    res = {}
    for i, t in enumerate(("income", "occupation_group")):
        y = D["y"][t][v]
        res[i] = {"acc": float((out[f"hard{i + 1}"][v] == y).mean()), "const_acc": float((y == cst[i]).mean())}
    return res


def stage_inner(D, k):
    names = [f"nn__s{k}__U", f"nn__s{k}__E"] + [f"nn__s{k}__{a}__b{b:g}" for a in NEURAL_ARMS for b in T.HP["betas"]]
    for nm in names:
        iname = f"inner__{nm}"
        if done(iname) or not done(nm):
            continue
        V = release_views(nm)
        rec = {"unit": iname, "of": nm, "recovery": inner_recovery(V, D), "utility": inner_utility(V["out"], D)}
        FN.save_unit(U(iname), {}, rec)
        event("unit complete", unit=iname)
    for i in (0, 1):
        for cfg in FARE_GRID:
            nm = f"fare__s{k}__p{i}__c{cfg['id']}"
            iname = f"inner__{nm}"
            if done(iname) or not done(nm):
                continue
            z = np.load(U(nm) / "release.npz")
            V = {"v": np.hstack([z["r"], z["c"]])}
            f, v = D["idx"]["attacker_fit"], D["idx"]["attacker_val"]
            s = AU.fit_view(V["v"][f], D["sex"][f], V["v"][v], D["sex"][v], "inner", finite=True)
            y = D["y"]["income" if i == 0 else "occupation_group"][v]
            rec = {"unit": iname, "of": nm, "purpose": i, "recovery_local": AU.macro_auc(D["sex"][v], s["Pv"], [0, 1]),
                   "utility": {"acc": float((z["hard"][v] == y).mean()), "const_acc": float((y == constants(D)[i]).mean())}}
            FN.save_unit(U(iname), {}, rec)
            event("unit complete", unit=iname)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", required=True, choices=("warm", "train", "fare", "inner", "select"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--arms", nargs="*")
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from jcv.lock import verify_lock
    v = verify_lock(Path(a.lock))
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    D = DA.load()
    event(f"start {a.stage}", seeds=a.seeds, pid=os.getpid())
    for k in a.seeds:
        if a.stage == "warm":
            stage_warm(D, k)
        elif a.stage == "train":
            stage_train(D, k, arms=a.arms)
        elif a.stage == "fare":
            stage_fare(D, k)
        elif a.stage == "inner":
            stage_inner(D, k)
    if a.stage == "select":
        from jcv.select import run_selection
        run_selection(D, a.seeds)
    event(f"end {a.stage}", seeds=a.seeds, pid=os.getpid())


if __name__ == "__main__":
    main()
