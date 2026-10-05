"""Controller preflight on the frozen Phase A references (fitting-only rows; counted as scientific work; PROTOCOL 5).

Per seed, from calib__s{k} (reference probe receipt and targets):
  1. intended initial violation: v_i = AUC_i(ref) - b_i (= +0.01 wherever the target is above the 0.5 floor);
  2. the epoch-0 controller update (w after one update);
  3. one encoder step at the reference state on a fixed minibatch with the reference critics: per-encoder relative
     strengths ||q_i||/||t_i|| and directions under w = (1, 1), the epoch-0 w, an asymmetric w = (2, 1) and w = (2, 2)
     for joint and local P -> allocation change, joint local/pair direction change, exact local common-weight symmetry;
  4. selectivity: a hypothetical update where only recipient 1 violates leaves w_2 unchanged;
  5. a short real J-F vs J-N and L-F vs L-N run (2 epochs, rho 0.75) from the reference: weights applied vs recorded,
     and the parameter difference between the feedback arm and its twin (is the logged weight change acting?).
Writes preflight__s{k} units and PREFLIGHT.json (public aggregate).
"""
from __future__ import annotations

import json

import numpy as np
import torch

from jcv.train import flat_grad, task_losses
from rgj import finalize as FN
from rgj import train as RT
from smf import run as R
from smf import train as T


def one_step(model, data, crit, head, base, w, rho, seed):
    idx = torch.from_numpy(np.sort(np.random.default_rng([seed, 4242]).choice(data.n, 256, replace=False)))
    enc = [list(model.enc[i].parameters()) for i in (0, 1)]
    params = enc[0] + enc[1] + [p for i in (0, 1) for p in model.head[i].parameters()]
    for p in params:
        p.requires_grad_(True)
    n0 = sum(p.numel() for p in enc[0])
    ne = n0 + sum(p.numel() for p in enc[1])
    Lt = task_losses(model, data.X[idx], {i: data.Y[i][idx] for i in (0, 1)}, [None, None], [0, 1])
    g = flat_grad(sum(Lt.values()), params)
    banks = {v: {} for v in RT.VIEWS}
    for v in RT.VIEWS:
        for k in RT.KINDS:
            c = RT.critic(k, RT.DV[v])
            c.load_state_dict(crit["critics"][v][k])
            banks[v][k] = c
    Tm = {v: RT.Transform(state=crit["transforms"][v]) for v in RT.VIEWS}
    coef = T.coefficients(base, w)
    active = [v for v in RT.VIEWS if coef[v] > 0]
    V = RT.critic_views(model, data.X[idx], active, head, grad=True)
    H = RT.entropy(data.prior)
    lp = torch.tensor(np.log(data.prior), dtype=torch.float32)
    P = sum(coef[v] * RT.recovery([banks[v][k](Tm[v](V[v])) for k in RT.KINDS], data.S[idx], lp, H)[0] for v in active)
    p = flat_grad(P, enc[0] + enc[1])
    s = T.allocation(*w)
    out, qs = {}, []
    for i, (lo, hi) in enumerate(((0, n0), (n0, ne))):
        q, info = T.normalized_direction(g[lo:hi], p[lo:hi], rho * s[i])
        qs.append(q)
        out[f"ratio_{i + 1}"] = info.get("ratio")
        out[f"p_dir_{i + 1}"] = (p[lo:hi] / p[lo:hi].norm()).detach()
    out["rms_ratio"] = float(np.sqrt(np.mean([out["ratio_1"] ** 2, out["ratio_2"] ** 2])))
    out["alloc"] = list(s)
    out["q"] = [q.detach() for q in qs]
    for p_ in params:
        p_.requires_grad_(False)
    return out


def cos(a, b):
    return float(torch.dot(a.double().flatten(), b.double().flatten()) / (a.double().norm() * b.double().norm()))


def run_preflight(D):
    data = RT.TData(D)
    pub = {"schema": "smf-preflight-v1", "seeds": {}}
    for k in R.SEEDS:
        name = f"preflight__s{k}"
        if R.done(name):
            pub["seeds"][str(k)] = R.rec(name)["summary"]
            continue
        st, crit, sref = R.reference(k)
        head = T.head_of(R.load_warm(k))
        c = R.rec(f"calib__s{k}")
        if crit is None:   # task-only reference: bounded fresh critics on the frozen reference (same refit procedure)
            m = R.model_from(st, k)
            banks = {v: {kk: RT.new_critic(k, v, kk, "init") for kk in RT.KINDS} for v in RT.VIEWS}
            Tm = {v: None for v in RT.VIEWS}
            RT.refit_banks(m, banks, Tm, data, k, 0, "floored", torch.tensor(np.log(data.prior), dtype=torch.float32),
                           RT.entropy(data.prior), head, tag="preflight")
            crit = RT.snapshot_state(m, banks, Tm, [1.0, 1.0], head)
        aucs = [c["receipt"][v]["calib_auc"] for v in ("v1", "v2")]
        v0 = [aucs[i] - c["b"][i] for i in (0, 1)]
        w0 = [T.controller_update(1.0, aucs[i], c["b"][i])[0] for i in (0, 1)]
        res = {"seed": k, "reference": sref.get("unit"), "b": c["b"], "floor_active": c["floor_active"],
               "initial_violation": v0, "w_after_epoch0": w0}
        steps = {}
        for base in ("joint", "local"):
            for tag, w in (("w11", (1.0, 1.0)), ("w_epoch0", tuple(w0)), ("w21", (2.0, 1.0)), ("w22", (2.0, 2.0))):
                steps[(base, tag)] = one_step(R.model_from(st, k), data, crit, head, base, w, 0.75, k)
        rows = {}
        for base in ("joint", "local"):
            b11 = steps[(base, "w11")]
            for tag in ("w_epoch0", "w21", "w22"):
                x = steps[(base, tag)]
                rows[f"{base}|{tag}"] = {"alloc": x["alloc"], "ratio_1": x["ratio_1"], "ratio_2": x["ratio_2"],
                                         "rms_ratio": x["rms_ratio"],
                                         "cos_q1_vs_w11": cos(x["q"][0], b11["q"][0]), "cos_q2_vs_w11": cos(x["q"][1], b11["q"][1]),
                                         "max_abs_q_diff_vs_w11": max(float((x["q"][i] - b11["q"][i]).abs().max()) for i in (0, 1))}
        res["one_step"] = rows
        res["checks"] = {
            "initial_violation_where_above_floor": all(abs(v0[i] - 0.01) < 1e-12 for i in (0, 1) if not c["floor_active"][i]),
            "weights_change_after_epoch0": any(x != 1.0 for x in w0),
            "local_common_weight_exact_symmetry": rows["local|w22"]["max_abs_q_diff_vs_w11"] == 0.0,
            "joint_common_weight_changes_direction": rows["joint|w22"]["cos_q1_vs_w11"] < 1 - 1e-9,
            "asymmetric_weights_reallocate": rows["local|w21"]["ratio_1"] > rows["local|w21"]["ratio_2"] + 0.1,
            "rms_identity": all(abs(r["rms_ratio"] - 0.75) < 1e-5 for r in rows.values()),
            "selectivity": T.controller_update(1.0, 0.5, 0.6)[0] == 0.25 and T.controller_update(1.0, 0.7, 0.6)[0] == 2.0}
        short = {}
        ctrl = {"receipt0": c["receipt"], "b": c["b"]}
        sched = R.selA()["schedule"]["selected"]
        for base, (fa, na) in (("joint", ("J-F", "J-N")), ("local", ("L-F", "L-N"))):
            out = {}
            for arm, fb in ((fa, True), (na, False)):
                kw = {}
                if sched == "ONLINE_MATCHED":
                    kw["matched_counts"] = {f"{v}|{kk}": 0 for v in RT.VIEWS for kk in RT.KINDS}
                m, d, *_ = T.train_run({"base": base, "schedule": "ONLINE" if sched == "ONLINE_MATCHED" else sched,
                                        "feedback": fb}, 0.75, st, data, k, "B", head, n_epochs=2, init_critics=crit,
                                       ckpt_epochs=(), controller=ctrl, **kw)
                out[arm] = {"w_trace": [e["w_after"] for e in d["controller"]], "auc_trace": [e["auc"] for e in d["controller"]],
                            "model": m.state_dict()}
            diff = max(float((out[fa]["model"][q] - out[na]["model"][q]).abs().max()) for q in out[fa]["model"])
            short[base] = {fa: {kk: v for kk, v in out[fa].items() if kk != "model"},
                           na: {kk: v for kk, v in out[na].items() if kk != "model"},
                           "max_abs_param_diff_feedback_vs_twin": diff}
        res["short_run"] = short
        res["checks"]["feedback_changes_parameters_joint"] = short["joint"]["max_abs_param_diff_feedback_vs_twin"] > 0
        summary = {kk: res[kk] for kk in ("b", "floor_active", "initial_violation", "w_after_epoch0", "checks")}
        summary["one_step"] = rows
        summary["short_run_w"] = {b: {a: v["w_trace"] for a, v in short[b].items() if isinstance(v, dict)} for b in short}
        summary["short_run_param_diff"] = {b: short[b]["max_abs_param_diff_feedback_vs_twin"] for b in short}
        FN.save_unit(R.U(name), {}, {**res, "summary": summary})
        R.event("unit complete", unit=name)
        pub["seeds"][str(k)] = summary
    pub["all_checks"] = {kk: all(pub["seeds"][s]["checks"][kk] for s in pub["seeds"]) for kk in
                         next(iter(pub["seeds"].values()))["checks"]}
    (R.PKG / "PREFLIGHT.json").write_text(json.dumps(pub, indent=1, default=float) + "\n")
    return pub
