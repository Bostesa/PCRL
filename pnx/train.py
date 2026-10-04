"""Training for the focused no-erasure penalty study (arms PN, LN), reusing the predecessor engine.

PN: joint ordinary penalty, P = R_v1 + R_v2 + R_pair, update u = -grad(L1 + L2) - beta grad P (JP's update), NO erasure
    at any point (representation maps are the identity in training and finalisation).
LN: local ordinary penalty, P = R_v1 + R_v2, local banks {A, B, B2} per view (equal critic-update budget), NO erasure.
Every other component (encoders, heads, warm starts, data order, critics, whitening, 5 critic steps, SGD 0.05, clip 5,
20 protection epochs) is the predecessor's, imported from jcv.train. beta = 0 reproduces U exactly (registered check).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from jcv.project import project
from jcv.train import (HP, Model, Whitener, _seed, assign_add, critic, entropy, fit_maps, flat_grad, guard_losses,
                       restore, snapshot, task_losses, views)
from jcv.train import arm_spec as _predecessor_arm_spec


def arm_spec(name):
    if name == "PN":
        return {"erasure": False, "stages": [{"train": [0, 1], "banks": {"v1": ["A", "B"], "v2": ["A", "B"], "pair": ["A", "B"]},
                                              "P": ["v1", "v2", "pair"], "guards": [], "mode": "penalty"}]}
    if name == "LN":
        return {"erasure": False, "stages": [{"train": [0, 1], "banks": {"v1": ["A", "B", "B2"], "v2": ["A", "B", "B2"]},
                                              "P": ["v1", "v2"], "guards": [], "mode": "penalty"}]}
    return _predecessor_arm_spec(name)


def train_arm(arm, beta, warm_state, d_in, Ks, data, seed, budgets, log=None, lr=None, masks=None):
    """Verbatim copy of jcv.train.train_arm (predecessor, locked) with additions that never change the update: the
    PN/LN arm specs (arm_spec below); the return of the final online critics, their whiteners and the model state at
    the last critic update (theta_{T-1}; review R1); and logging of task/penalty norms and clip hits (review A4).
    Fidelity is checked bitwise against predecessor units (JP, U) before any new fit."""
    spec = arm_spec(arm)
    lr = HP["sgd_lr"] if lr is None else lr
    model = Model(d_in, Ks, seed, masks=masks)
    model.load_state_dict(warm_state)
    maps = [None, None]
    official = {}
    n = data.X.shape[0]
    H = entropy(data.prior)
    logprior = torch.tensor(np.log(data.prior), dtype=torch.float32)
    diag = {"arm": arm, "beta": beta, "lr": lr, "steps": 0, "protection_steps_attempted": 0,
            "protection_steps_rejected": 0, "backtracks": 0, "projection_active": {"0": 0, "1": 0, "2": 0},
            "max_projection_violation": 0.0, "nonfinite": 0, "leace_refits": 0, "stage_logs": []}
    gstate = torch.Generator()
    for si, st in enumerate(spec["stages"]):
        train_idx = st["train"]
        frozen = tuple(i for i in (0, 1) if i not in train_idx)
        enc_params = [p for i in train_idx for p in model.enc[i].parameters()]
        head_params = [p for i in train_idx for p in model.head[i].parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        for p in enc_params + head_params:
            p.requires_grad_(True)
        # critics (fresh per stage; identical initialisation across arms for the same seed/view/kind)
        dv = {"v1": 16 + Ks[0], "v2": 16 + Ks[1], "pair": 32 + Ks[0] + Ks[1]}
        banks = {}
        for v, kinds in st["banks"].items():
            banks[v] = []
            for k in kinds:
                torch.manual_seed(_seed("critic", seed, v, k))
                c = critic("A" if k == "A" else "B", dv[v])
                banks[v].append((c, torch.optim.Adam(c.parameters(), lr=HP["critic_lr"])))
        if spec["erasure"]:
            # maps of frozen encoders: fit once (they do not change during this stage)
            need = [i for i in frozen if maps[i] is None]
            official.update(fit_maps(model, data, need, maps))
        for ep in range(HP["prot_epochs"]):
            if spec["erasure"]:
                official.update(fit_maps(model, data, train_idx, maps))
                diag["leace_refits"] += 1
            ref = torch.from_numpy(data.guard_idx)
            perm = np.random.default_rng([seed, si, ep]).permutation(n)
            crng = np.random.default_rng([seed, si, ep, 7])
            grng = np.random.default_rng([seed, si, ep, 11])
            ep_log = {"phase": f"stage{si}", "epoch": ep, "critic_ce": {}, "R": {}, "active": 0, "rejected": 0}
            for s in range(0, n, HP["batch"]):
                b = torch.from_numpy(perm[s:s + HP["batch"]])
                diag["steps"] += 1
                # ---- critic updates (fitting role, detached views)
                if banks:
                    with torch.no_grad():
                        Vref = views(model, data.X[ref], maps, list(banks), frozen=(0, 1))
                    wh = {v: Whitener(Vref[v]) for v in banks}
                    for _ in range(HP["critic_steps_per_step"]):
                        cb = torch.from_numpy(crng.choice(n, HP["batch"], replace=False))
                        with torch.no_grad():
                            V = views(model, data.X[cb], maps, list(banks), frozen=(0, 1))
                        for v, bank in banks.items():
                            for c, opt in bank:
                                loss = F.cross_entropy(c(wh[v](V[v])), data.S[cb])
                                opt.zero_grad()
                                loss.backward()
                                opt.step()
                    state_at_critic = {kk: x.detach().clone() for kk, x in model.state_dict().items()}  # R1 (diagnostic only)
                Yb = {i: data.Y[i][b] for i in (0, 1)}
                # ---- task direction
                Lt = task_losses(model, data.X[b], Yb, maps, train_idx)
                ltask = sum(Lt.values())
                t = flat_grad(ltask, enc_params + head_params)
                ne = sum(p.numel() for p in enc_params)
                t = -t
                d_enc = torch.zeros(ne)
                if st["mode"] in ("projected", "penalty") and beta > 0:
                    diag["protection_steps_attempted"] += 1
                    V = views(model, data.X[b], maps, st["P"], frozen=frozen)
                    P = 0.0
                    for v in st["P"]:
                        ces = [F.cross_entropy(c(wh[v](V[v])), data.S[b]) for c, _ in banks[v]]
                        ce_const = F.nll_loss(logprior.expand(len(b), 2), data.S[b])
                        allc = ces + [ce_const]
                        j = int(np.argmin([float(x) for x in allc]))
                        Rv = 1.0 - allc[j] / H
                        P = P + Rv
                        ep_log["R"].setdefault(v, []).append(float(Rv))
                    if isinstance(P, torch.Tensor) and P.requires_grad:
                        pvec = -beta * flat_grad(P, enc_params)
                    else:
                        pvec = torch.zeros(ne)
                    if not torch.isfinite(pvec).all():
                        diag["nonfinite"] += 1
                        pvec = torch.zeros(ne)
                    if st["mode"] == "projected":
                        cur = guard_losses(model, data, maps, st["guards"])
                        act = [i for i in st["guards"] if cur[i] > budgets[i] - HP["active_margin_nats"]]
                        A = []
                        if act:
                            gb = torch.from_numpy(data.guard_idx[grng.choice(len(data.guard_idx), HP["guard_minibatch"],
                                                                             replace=False)])
                            La = task_losses(model, data.X[gb], {i: data.Y[i][gb] for i in (0, 1)}, maps, act)
                            for i in act:
                                A.append(flat_grad(La[i], enc_params, retain=True).double().numpy())
                        dvec, info = project(pvec.double().numpy(), np.array(A) if A else np.zeros((0, ne)))
                        diag["projection_active"][str(len(A))] += 1
                        diag["max_projection_violation"] = max(diag["max_projection_violation"],
                                                               info["max_constraint_violation"])
                        ep_log["active"] += int(len(A) > 0)
                        d_enc = torch.from_numpy(dvec).float()
                    else:
                        d_enc = pvec
                u = torch.cat([t[:ne] + d_enc, t[ne:]])
                nrm = float(u.norm())
                diag.setdefault("norms", {"t": [], "penalty": [], "clipped": 0, "steps": 0})
                diag["norms"]["steps"] += 1
                if diag["norms"]["steps"] % 20 == 0:
                    diag["norms"]["t"].append(round(float(t.norm()), 4))
                    diag["norms"]["penalty"].append(round(float(d_enc.norm()), 4))
                diag["norms"]["clipped"] += int(math.isfinite(nrm) and nrm > HP["clip"])
                if not math.isfinite(nrm):
                    diag["nonfinite"] += 1
                    continue
                if nrm > HP["clip"]:
                    u = u * (HP["clip"] / nrm)
                params = enc_params + head_params
                if st["mode"] == "projected" and beta > 0:
                    before = guard_losses(model, data, maps, st["guards"])
                    snap = snapshot(params)
                    ok = False
                    for k, sc in enumerate(HP["backtrack"]):
                        assign_add(params, u, lr * sc)
                        after = guard_losses(model, data, maps, st["guards"])
                        if all(after[i] <= max(budgets[i], before[i]) + HP["accept_tol"] for i in st["guards"]):
                            ok = True
                            diag["backtracks"] += k
                            break
                        restore(params, snap)
                    if not ok:  # reject the protection component: task-only step
                        diag["protection_steps_rejected"] += 1
                        ep_log["rejected"] += 1
                        tt = t.clone()
                        tn = float(tt.norm())
                        if tn > HP["clip"]:
                            tt = tt * (HP["clip"] / tn)
                        assign_add(params, tt, lr)
                else:
                    assign_add(params, u, lr)
            if banks:
                ep_log["R_trace"] = {v: [round(float(y), 4) for y in x] for v, x in ep_log["R"].items()}
                ep_log["R"] = {v: float(np.mean(x)) for v, x in ep_log["R"].items()}
            ep_log["guard"] = guard_losses(model, data, maps, (0, 1))
            if log:
                log(ep_log)
            diag["stage_logs"].append(ep_log)
        if spec["erasure"]:  # end-of-stage refit for the trained encoders (frozen afterwards)
            official.update(fit_maps(model, data, train_idx, maps))
    for p in model.parameters():
        p.requires_grad_(False)
    final_critics = {"model_state_at_last_critic_update": state_at_critic if banks else None,
                     "critics": {v: [c.state_dict() for c, _ in bank] for v, bank in banks.items()},
                     "kinds": {v: list(spec["stages"][-1]["banks"][v]) for v in banks},
                     "whiteners": {v: {"mu": w.mu.clone(), "W": w.W.clone()} for v, w in (wh.items() if banks else [])}}
    return model, diag, final_critics
