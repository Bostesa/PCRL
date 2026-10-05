"""Strength-matched training engine with active AUC feedback (registered in PROTOCOL.md / METHOD_CARD.md).

Inherited (pinned rgj.train, unchanged): encoders 83-64-64-16 + affine training heads (jcv.train.Model), warm-start
critic head fixed per seed, critic views v_i = [r_i, centred(W_w r_i + b_w)], pair = [v1, v2], critic kinds A/B per view,
recovery surrogate R = (CE_const - min(CE_const, CE_A, CE_B)) / H on the same rows, floored ZCA transforms, bounded
refits with transport, SGD 0.05, global-norm clip 5, batch 256. Raw-penalty controls (RAW-J / RAW-L) are rgj.train's
J-O / L-O called unchanged (base weights beta/3 x3 or beta/2 x2, online per-step transform).

Norm-controlled update (new). Per encoder i on the encoder minibatch:
    t_i = d L_task / d enc_i  (only recipient i's task loss reaches encoder i: separate encoders)
    p_i = d P / d enc_i,  P = sum_v c_v R_v with c normalised to sum 1:
        joint  c ∝ (w1, w2, 1) on (R_1, R_2, R_pair);   local  c ∝ (w1, w2) on (R_1, R_2); pair bank = shadow (no grad)
    a_i = stop_grad(rho * s_i * ||t_i|| / ||p_i||) in float64, capped at A_MAX = 100;  q_i = a_i p_i
    q_i = 0 (event logged) if ||p_i|| <= ZERO_TOL or ||t_i|| == 0
    update u = -(t + q) on encoders, -(task grad) on heads; global-norm clip 5; params += lr * u
Phase A: s_1 = s_2 = 1 (w = (1, 1)). Phase B: s_i = w_i / sqrt((w_1^2 + w_2^2)/2); before caps/clipping
sqrt(mean((||q_i||/||t_i||)^2)) = rho. rho = 0 gives q = 0 and reproduces the task-only continuation bitwise.
Schedules: ONLINE (rgj online: per-step floored transform on a fixed 4096-row CRITIC_FIT subset, 5 critic steps per
encoder step); REFRESHED (rgj refits at epochs 0, 4, 8, 12, 16 + diagnostic 20; 5 online steps between, block transform);
ONLINE_MATCHED (ONLINE + extra Adam updates per bank equal to the paired REFRESHED unit's refit receipt for epochs
0-16 (continued + restart attempts), spread over encoder steps by floor(E*s/S) - floor(E*(s-1)/S); own RNG stream).
Controller (Phase B; feedback arms apply it, no-feedback twins record it): at the start of epochs 0..19 (and a
diagnostic measurement at 20) fit probes on frozen local views (scale-aware whitened LR and an MLP; CRITIC_FIT ->
CRITIC_VAL selection with orientation chosen there -> AUC on CONTROLLER_CALIB); v_i = AUC_i - b_i;
w_i <- clip(w_i + clip(v_i / 0.01, -1, 1), 0.25, 8). Epoch 0 reuses the frozen reference's probe receipt.
RNG streams: task minibatches rng([seed, salt, ep]); critics rng([seed, salt, ep, 7]); matched extras
rng([seed, salt, ep, 13]); controller probes seeded by (seed, epoch, view) -> diagnostic work never perturbs task order.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from jcv.train import Model, assign_add, entropy, flat_grad, task_losses
from rgj import train as RT
from rgj.train import (KINDS, VIEWS, Transform, critic_views, fit_bounded, frozen_views, head_of, new_critic,
                       recovery, refit_banks, snapshot_state, transport_first_layer)

torch.set_num_threads(1)

HP = {**{k: v for k, v in RT.HP.items() if k not in ("betas", "c_margin", "lambda_cap_mult", "checkpoints")},
      "A_MAX": 100.0, "ZERO_TOL": 1e-12, "rhos": (0.25, 0.75, 1.5), "raw_betas": (0.1, 0.3),
      "ctrl_gain_div": 0.01, "w_min": 0.25, "w_max": 8.0, "target_offset": 0.01, "target_floor": 0.5}
D_IN, KS = RT.D_IN, RT.KS
SCHEDULES = ("ONLINE", "REFRESHED", "ONLINE_MATCHED")
SALT = {"A": 0, "B": 1}


# ------------------------------------------------------------------ update-strength primitives
def normalized_direction(t, p, rho_i, a_max=None, zero_tol=None):
    """q = a p with a = rho_i ||t|| / ||p|| (float64 norms, no gradient), capped at a_max. Returns (q, info)."""
    a_max = HP["A_MAX"] if a_max is None else a_max
    zero_tol = HP["ZERO_TOL"] if zero_tol is None else zero_tol
    with torch.no_grad():
        nt = float(t.double().norm())
        npn = float(p.double().norm())
        info = {"t_norm": nt, "p_norm": npn, "zero": None, "cap": False}
        if rho_i == 0:
            info.update(a=0.0, ratio=0.0)
            return torch.zeros_like(p), info
        if npn <= zero_tol or nt == 0.0:
            info.update(a=0.0, ratio=None, zero="p" if npn <= zero_tol else "t")
            return torch.zeros_like(p), info
        a = rho_i * nt / npn
        if a > a_max:
            a, info["cap"] = a_max, True
        q = (p.double() * a).to(p.dtype)
        info.update(a=a, ratio=float(q.double().norm()) / nt,
                    cos=float(torch.dot(t.double(), p.double()) / (nt * npn)))
        return q, info


def allocation(w1, w2):
    rms = math.sqrt((w1 * w1 + w2 * w2) / 2.0)
    return w1 / rms, w2 / rms


def controller_update(w, auc, b):
    v = auc - b
    step = max(-1.0, min(1.0, v / HP["ctrl_gain_div"]))
    return float(min(HP["w_max"], max(HP["w_min"], w + step))), v


def coefficients(base, w):
    if base == "joint":
        s = w[0] + w[1] + 1.0
        return {"v1": w[0] / s, "v2": w[1] / s, "pair": 1.0 / s}
    if base == "local":
        s = w[0] + w[1]
        return {"v1": w[0] / s, "v2": w[1] / s, "pair": 0.0}
    return {"v1": 0.0, "v2": 0.0, "pair": 0.0}


def spread(total, S, s):
    """Extra updates at encoder step s (1-indexed) so that the S steps receive exactly `total`, evenly."""
    return (total * s) // S - (total * (s - 1)) // S


# ------------------------------------------------------------------ controller probes
CTRL_REL_TOL = 1e-9   # review R2: the whitened LR reads the r block only (16 columns; the centred logits are affine in r,
                      # so the linear function class is identical) - no float32 logit-block null directions to exclude;
                      # 1e-9 x s_max keeps a rotated 1e-6 clue (relative singular value ~1e-7) and drops only
                      # numerically singular directions.
R_DIM = 16


def _whiten_fit(Z):
    mu = Z.mean(0)
    U, s, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    tol = s.max() * CTRL_REL_TOL
    keep = s > tol
    return mu, Vt[keep].T / s[keep] * math.sqrt(len(Z))


def probe_auc(V, S, data, seed, tag):
    """Controller probe on one frozen local view: whitened LR on the r block (scale-aware, float64, rank tol 1e-9) and
    an MLP on the full view,
    fitted on CRITIC_FIT, selected (with orientation) by CRITIC_VAL AUC, evaluated on CONTROLLER_CALIB."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    V = np.asarray(V, dtype=np.float64)
    cf, cv, cal = data.cf, data.cv, data.cal
    Rb = V[:, :R_DIM]
    mu, W = _whiten_fit(Rb[cf])
    Zf, Zv, Zc = (Rb[cf] - mu) @ W, (Rb[cv] - mu) @ W, (Rb[cal] - mu) @ W
    rs = int(RT._seed("rgj-ctrl", seed, tag) % (2 ** 31))
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(Zf, S[cf])
    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-4, max_iter=200,
                                                        early_stopping=True, validation_fraction=0.1,
                                                        n_iter_no_change=10, random_state=rs)).fit(V[cf], S[cf])
    cands = {"whitened_LR": (lr.predict_proba(Zv)[:, 1], lr.predict_proba(Zc)[:, 1]),
             "MLP_64": (mlp.predict_proba(V[cv])[:, 1], mlp.predict_proba(V[cal])[:, 1])}
    table = {}
    for name, (pv, pc) in cands.items():
        a = roc_auc_score(S[cv], pv)
        sign = 1.0 if a >= 0.5 else -1.0
        table[name] = {"val_auc_oriented": max(a, 1 - a), "orientation": sign,
                       "calib_auc": float(roc_auc_score(S[cal], sign * pc))}
    best = max(table, key=lambda n: (table[n]["val_auc_oriented"], n == "whitened_LR"))
    return {"selected": best, "calib_auc": table[best]["calib_auc"], "table": table}


def probe_receipt(model, data, head, seed, tag):
    Vl = frozen_views(model, data.X, head)
    S = data.S.numpy()
    return {v: probe_auc(Vl[v].numpy(), S, data, seed, f"{tag}|{v}") for v in ("v1", "v2")}


def targets_from(receipt):
    return [max(HP["target_floor"], receipt[v]["calib_auc"] - HP["target_offset"]) for v in ("v1", "v2")]


# ------------------------------------------------------------------ one run
def train_run(spec, rho, init_state, data, seed, stage, critic_head, n_epochs=20, init_critics=None,
              ckpt_epochs=(20,), matched_counts=None, controller=None, capture_steps=(), lr=None, log=None):
    """spec: {"base": joint|local|none, "schedule": ONLINE|REFRESHED|ONLINE_MATCHED, "feedback": bool,
    "phase": "A"|"B"}. controller (phase B): {"receipt0": reference probe receipt, "b": [b1, b2]}.
    Returns (model, diag, checkpoints, captures, final)."""
    lr = HP["sgd_lr"] if lr is None else lr
    base, sched = spec["base"], spec.get("schedule", "ONLINE")
    model = Model(D_IN, KS, seed)
    model.load_state_dict(init_state)
    enc = [list(model.enc[i].parameters()) for i in (0, 1)]
    enc_params = enc[0] + enc[1]
    head_params = [p for i in (0, 1) for p in model.head[i].parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in enc_params + head_params:
        p.requires_grad_(True)
    n0 = sum(p.numel() for p in enc[0])
    ne = n0 + sum(p.numel() for p in enc[1])
    H = entropy(data.prior)
    logprior = torch.tensor(np.log(data.prior), dtype=torch.float32)
    has_critics = base != "none"
    head = critic_head
    banks, opts, T = {}, {}, {v: None for v in VIEWS}
    if has_critics:
        for v in VIEWS:
            banks[v] = {}
            for k in KINDS:
                c = new_critic(seed, v, k, "init")
                if init_critics is not None:
                    c.load_state_dict(init_critics["critics"][v][k])
                banks[v][k] = c
            if init_critics is not None and init_critics["transforms"][v] is not None:
                T[v] = Transform(state=init_critics["transforms"][v])
    T_inherited = dict(T)

    def reset_opts():
        for v in banks:
            opts[v] = {k: torch.optim.Adam(banks[v][k].parameters(), lr=HP["critic_lr"]) for k in KINDS}
    reset_opts()
    w = [1.0, 1.0]
    w_hyp = [1.0, 1.0]
    diag = {"spec": spec, "rho": rho, "lr": lr, "seed": seed, "stage": stage, "encoder_updates": 0,
            "critic_online_updates": 0, "critic_matched_extra_updates": 0, "critic_refit_updates": 0,
            "refits": [], "nonfinite": 0, "clip_hits": 0, "cap_hits": [0, 0], "zero_events": [0, 0],
            "epochs": [], "controller": [], "matched_counts": matched_counts, "early_stops": 0,
            "restart_chosen": 0, "continued_chosen": 0}
    checkpoints, captures, final = {}, {}, {}
    salt = SALT[stage]
    n = data.n
    S_steps = n_epochs * math.ceil(n / HP["batch"])
    step = 0
    t0 = time.time()
    for ep in range(n_epochs + 1):
        if ep in ckpt_epochs and ep > 0:
            checkpoints[ep] = snap_full(model, banks, T, w, head, opts, w_hyp) if has_critics else {"model": copy.deepcopy(model.state_dict())}
            checkpoints[ep]["step"], checkpoints[ep]["w"] = step, list(w)
        if has_critics and sched == "REFRESHED" and ep in HP["refit_epochs"]:
            rr = refit_banks(model, banks, T, data, seed, ep, "floored", logprior, H, head)
            reset_opts()
            entry = {"epoch": ep, "step": step, "receipts": rr, "diagnostic_only": ep == n_epochs}
            if ep < n_epochs:
                for v in rr:
                    for k in rr[v]["kinds"]:
                        q = rr[v]["kinds"][k]
                        diag["critic_refit_updates"] += q["continued"]["updates"] + q["restart"]["updates"]
                        diag["early_stops"] += int(q["continued"]["early_stopped"]) + int(q["restart"]["early_stopped"])
                        diag["restart_chosen" if q["choice"] == "restart" else "continued_chosen"] += 1
            diag["refits"].append(entry)
            if ep == n_epochs:
                final["refit_critics"] = snap_full(model, banks, T, w, head, opts, w_hyp)
        if controller is not None and ep <= n_epochs:      # measurement at the start of epoch ep (20 = diagnostic)
            rec = controller["receipt0"] if ep == 0 else probe_receipt(model, data, head, seed, f"{stage}|{ep}")
            aucs = [rec["v1"]["calib_auc"], rec["v2"]["calib_auc"]]
            b = controller["b"]
            new_hyp, vs = [], []
            for i in (0, 1):
                nw, v = controller_update(w_hyp[i], aucs[i], b[i])
                new_hyp.append(nw)
                vs.append(v)
            entry = {"epoch": ep, "auc": aucs, "b": b, "v": vs, "w_before": list(w), "diagnostic_only": ep == n_epochs,
                     "probe": {v: {"selected": rec[v]["selected"]} for v in ("v1", "v2")}, "reused_reference": ep == 0}
            if ep < n_epochs:
                w_hyp = new_hyp
                if spec.get("feedback"):
                    w = list(new_hyp)
            entry["w_after"], entry["w_hypothetical"] = list(w), list(w_hyp)
            entry["allocation"] = allocation(*w)
            diag["controller"].append(entry)
        if ep == n_epochs:
            break
        perm = np.random.default_rng([seed, salt, ep]).permutation(n)
        crng = np.random.default_rng([seed, salt, ep, 7])
        xrng = np.random.default_rng([seed, salt, ep, 13])
        coef = coefficients(base, w) if spec.get("update") != "task" else {"v1": 0, "v2": 0, "pair": 0}
        s_alloc = allocation(*w)
        ep_log = {"epoch": ep, "ratio": [[], []], "cos": [[], []], "R": {v: [] for v in VIEWS}, "clip": 0,
                  "t_norm": [[], []], "q_norm": [[], []], "p_norm": [[], []], "a": [[], []], "update_norm_post_clip": [],
                  "post_clip_enc_norm": [[], []], "w": list(w), "alloc": list(s_alloc), "n_steps": 0,
                  "zero": [0, 0], "cap": [0, 0], "nonfinite": 0, "realized_ratio": [[], []]}
        for s in range(0, n, HP["batch"]):
            b_idx = torch.from_numpy(perm[s:s + HP["batch"]])
            step += 1
            if has_critics:
                if sched in ("ONLINE", "ONLINE_MATCHED"):
                    Vref = critic_views(model, data.X[torch.from_numpy(data.ref)], list(VIEWS), head)
                    T_step = {v: Transform(Vref[v], "floored") for v in VIEWS}
                    if step == 1 and init_critics is not None:
                        for v in VIEWS:
                            if T_inherited.get(v) is not None:
                                for k in KINDS:
                                    transport_first_layer(banks[v][k], T_inherited[v], T_step[v])
                        reset_opts()
                    T = T_step
                for _ in range(HP["critic_steps_per_step"]):
                    cb = torch.from_numpy(crng.choice(data.cf, HP["batch"], replace=False))
                    V = critic_views(model, data.X[cb], list(VIEWS), head)
                    for v in VIEWS:
                        for k in KINDS:
                            loss = F.cross_entropy(banks[v][k](T[v](V[v])), data.S[cb])
                            opts[v][k].zero_grad()
                            loss.backward()
                            opts[v][k].step()
                            diag["critic_online_updates"] += 1
                if sched == "ONLINE_MATCHED":
                    need = {(v, k): spread(matched_counts[f"{v}|{k}"], S_steps, step) for v in VIEWS for k in KINDS}
                    for j in range(max(need.values())):
                        cb = torch.from_numpy(xrng.choice(data.cf, HP["batch"], replace=False))
                        V = critic_views(model, data.X[cb], list(VIEWS), head)
                        for (v, k), m in need.items():
                            if m > j:
                                loss = F.cross_entropy(banks[v][k](T[v](V[v])), data.S[cb])
                                opts[v][k].zero_grad()
                                loss.backward()
                                opts[v][k].step()
                                diag["critic_matched_extra_updates"] += 1
            if step in capture_steps or step == S_steps:
                snap = snap_full(model, banks, T, w, head, opts, w_hyp) if has_critics else {"model": copy.deepcopy(model.state_dict())}
                if step in capture_steps:
                    captures[step] = snap
                if step == S_steps:
                    final["theta_T_minus_1"] = snap
            Yb = {i: data.Y[i][b_idx] for i in (0, 1)}
            Lt = task_losses(model, data.X[b_idx], Yb, [None, None], [0, 1])
            g = flat_grad(sum(Lt.values()), enc_params + head_params)       # task gradient (rgj ordering)
            d_enc = torch.zeros(ne)
            active = [v for v in VIEWS if coef[v] > 0]
            ep_log["n_steps"] += 1
            protect = has_critics and bool(active) and rho > 0
            step_ratio = [0.0, 0.0] if protect else None
            if protect:
                V = critic_views(model, data.X[b_idx], active, head, grad=True)
                P = 0.0
                for v in active:
                    Rv, _, _ = recovery([banks[v][k](T[v](V[v])) for k in KINDS], data.S[b_idx], logprior, H)
                    ep_log["R"][v].append(float(Rv))
                    P = P + coef[v] * Rv
                if not (isinstance(P, torch.Tensor) and P.requires_grad):     # review R1: constant won everywhere
                    for i in (0, 1):
                        diag["zero_events"][i] += 1
                        ep_log["zero"][i] += 1
                else:
                    pvec = flat_grad(P, enc_params)
                    if not torch.isfinite(pvec).all():
                        diag["nonfinite"] += 1
                        ep_log["nonfinite"] += 1
                    else:
                        parts = []
                        for i, (lo, hi) in enumerate(((0, n0), (n0, ne))):
                            qi, info = normalized_direction(g[lo:hi], pvec[lo:hi], rho * s_alloc[i])
                            parts.append(qi)
                            diag["cap_hits"][i] += int(info["cap"])
                            ep_log["cap"][i] += int(info["cap"])
                            ep_log["p_norm"][i].append(info["p_norm"])
                            if info["zero"]:
                                diag["zero_events"][i] += 1
                                ep_log["zero"][i] += 1
                            if info.get("ratio") is not None:
                                step_ratio[i] = info["ratio"]
                                ep_log["ratio"][i].append(info["ratio"])
                                ep_log["t_norm"][i].append(info["t_norm"])
                                ep_log["q_norm"][i].append(info["ratio"] * info["t_norm"])
                                ep_log["a"][i].append(info["a"])
                            if info.get("cos") is not None:
                                ep_log["cos"][i].append(info["cos"])
                        d_enc = torch.cat(parts)
            if step_ratio is not None:
                for i in (0, 1):
                    ep_log["realized_ratio"][i].append(step_ratio[i])
            u = torch.cat([-(g[:ne] + d_enc), -g[ne:]])
            nrm = float(u.norm())
            if not math.isfinite(nrm):
                diag["nonfinite"] += 1
                continue
            kappa = 1.0
            if nrm > HP["clip"]:
                kappa = HP["clip"] / nrm
                u = u * kappa
                diag["clip_hits"] += 1
                ep_log["clip"] += 1
            ep_log["update_norm_post_clip"].append(min(nrm, HP["clip"]))
            ep_log["post_clip_enc_norm"][0].append(float(u[:n0].norm()))
            ep_log["post_clip_enc_norm"][1].append(float(u[n0:ne].norm()))
            ep_log.setdefault("kappa", []).append(kappa)
            assign_add(enc_params + head_params, u, lr)
            diag["encoder_updates"] += 1
        def m(x):
            return float(np.mean(x)) if x else None
        summ = {"epoch": ep, "w": ep_log["w"], "alloc": ep_log["alloc"], "clip": ep_log["clip"], "n_steps": ep_log["n_steps"],
                "nonfinite": ep_log["nonfinite"], "update_norm_post_clip_mean": m(ep_log["update_norm_post_clip"]),
                "kappa_mean": m(ep_log.get("kappa", [])), "kappa_min": min(ep_log.get("kappa", [1.0]))}
        for i in (0, 1):
            r = ep_log["ratio"][i]
            rr = ep_log["realized_ratio"][i]
            summ[f"ratio_mean_{i + 1}"] = m(r)                       # conditional on a nonzero direction
            summ[f"ratio_rms_{i + 1}"] = float(np.sqrt(np.mean(np.square(r)))) if r else None
            summ[f"realized_ratio_mean_{i + 1}"] = m(rr)             # zeros counted (review R1)
            summ[f"zero_{i + 1}"], summ[f"cap_{i + 1}"] = ep_log["zero"][i], ep_log["cap"][i]
            summ[f"cos_mean_{i + 1}"] = m(ep_log["cos"][i])
            summ[f"t_norm_mean_{i + 1}"] = m(ep_log["t_norm"][i])
            summ[f"q_norm_mean_{i + 1}"] = m(ep_log["q_norm"][i])
            summ[f"p_norm_mean_{i + 1}"] = m(ep_log["p_norm"][i])
            summ[f"a_min_{i + 1}"] = min(ep_log["a"][i]) if ep_log["a"][i] else None
            summ[f"a_max_{i + 1}"] = max(ep_log["a"][i]) if ep_log["a"][i] else None
            summ[f"post_clip_enc_norm_mean_{i + 1}"] = m(ep_log["post_clip_enc_norm"][i])
        summ["R"] = {v: (float(np.mean(x)) if x else None) for v, x in ep_log["R"].items()}
        diag["epochs"].append(summ)
        if log:
            log(summ)
    for p in model.parameters():
        p.requires_grad_(False)
    final["theta_T"] = copy.deepcopy(model.state_dict())
    diag["wall_s"] = time.time() - t0
    return model, diag, checkpoints, captures, final


def snap_full(model, banks, T, w, head, opts, w_hyp):
    """Review R4: theta, critic states, critic-view head, transforms, critic Adam states and both weight vectors."""
    sn = snapshot_state(model, banks, T, w, head)
    sn["opts"] = {v: {k: copy.deepcopy(opts[v][k].state_dict()) for k in KINDS} for v in opts}
    sn["w_hyp"] = list(w_hyp)
    return sn


def refit_counts(diag):
    """Per-bank optimizer-update counts of a REFRESHED unit's refits at epochs 0..16 (continued + restart attempts;
    the diagnostic final refit excluded) -> the ONLINE_MATCHED template."""
    out = {f"{v}|{k}": 0 for v in VIEWS for k in KINDS}
    for e in diag["refits"]:
        if e["diagnostic_only"]:
            continue
        for v, rv in e["receipts"].items():
            for k, q in rv["kinds"].items():
                out[f"{v}|{k}"] += q["continued"]["updates"] + q["restart"]["updates"]
    return out
