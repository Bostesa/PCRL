"""Joint complete-view training engine (registered in METHOD_CARD.md / PROTOCOL.md).

Networks (per purpose i in {0: income, 1: occupation_group}):
  g_i : Linear(d_in,64)-ReLU-Linear(64,64)-ReLU-Linear(64,16)   (separate encoders, optimised together)
  h_i : Linear(16, K_i)                                            (affine task head)
  release r_i = LEACE_i(g_i(X)) for erasure arms (map frozen between registered refits), else g_i(X)
  view v_i = [r_i, centred logits of h_i(r_i)];  coalition view = [v_1, v_2]
Critics (per view bank): kind "A" = Linear(dv,32)-ReLU-Linear(32,2); kind "B" = Linear(dv,64)-ReLU-Linear(64,64)-ReLU-
  Linear(64,2) (interaction-capable); plus the constant fitting-prior predictor. Critics minimise SEX cross-entropy (Adam).
Surrogate: R_v = 1 - min(CE_A, CE_B, ..., CE_const) / H_fit(S) on the step's minibatch (fitting role only).
  P_L = R_v1 + R_v2;  P_J = R_v1 + R_v2 + R_pair.
Encoder/head step (plain SGD so that the projection statement holds for the step direction):
  task direction  t = -grad(L_1 + L_2)                         (encoders and heads)
  privacy dir.    p = -beta * grad P   (encoders only; head weights detached inside the views)
  projected arms  d = Proj(p; active guards a_j = grad_enc L_j(guard minibatch))   [jcv.project]
  penalty arm JP  d = p
  update u = t + d (encoder part), t (head part); global-norm clip; projected arms: backtracking on the guard set,
  accept if every guard loss <= max(budget_j, previous) ; else task-only step (counted as a rejected protection step).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jcv.project import project

torch.set_num_threads(1)
DEV = torch.device("cpu")

# ------------------------------------------------------------------ registered schedule (PROTOCOL.md)
HP = {
    "width": 64, "rep_dim": 16, "batch": 256, "warm_epochs": 20, "warm_lr_adam": 1e-3,
    "prot_epochs": 20, "sgd_lr": 0.05, "clip": 5.0, "critic_lr": 3e-3, "critic_steps_per_step": 5,
    "guard_size": 4096, "guard_minibatch": 512, "guard_seed": 20261003, "budget_slack_nats": 0.01,
    "active_margin_nats": 0.002, "backtrack": [1.0, 0.5, 0.25, 0.125], "accept_tol": 1e-6,
    "betas": [0.1, 1.0, 10.0], "whiten_floor": 1e-8,
}


def mlp_encoder(d_in, width, out):
    return nn.Sequential(nn.Linear(d_in, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, out))


def critic(kind, dv):
    if kind == "A":
        return nn.Sequential(nn.Linear(dv, 32), nn.ReLU(), nn.Linear(32, 2))
    return nn.Sequential(nn.Linear(dv, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))


class Whitener:
    """Critic-input whitening (part of the critic, not of the release): ZCA whitening of a view, statistics from a fixed
    fitting-role reference subset, refreshed before every encoder step and treated as a constant within the step;
    eigenvalues floored at whiten_floor x max so the dimension is fixed.
    Repairs 2026-10-03 (XOR fixture, before any real fit): (1) unwhitened critics could not detect low-variance
    encodings; (2) per-epoch PCA whitening amplified encoder drift in near-degenerate directions."""

    def __init__(self, V):
        V = V.double()
        self.mu = V.mean(0)
        C = torch.cov((V - self.mu).T)
        ev, U = torch.linalg.eigh(C)
        ev = torch.clamp(ev, min=float(ev.max()) * HP["whiten_floor"])
        self.W = ((U / torch.sqrt(ev)) @ U.T).float()     # ZCA: continuous in the covariance (no sign flips)
        self.mu = self.mu.float()

    def __call__(self, V):
        return (V - self.mu) @ self.W


def _seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class Model(nn.Module):
    def __init__(self, d_in, Ks, seed, width=64, rep=16, masks=None):
        """masks (fixtures only): per-encoder input column subsets; the real study passes None (full X to both)."""
        super().__init__()
        self.masks = masks
        encs, heads = [], []
        for i, K in enumerate(Ks):
            torch.manual_seed(_seed("enc", seed, i))
            encs.append(mlp_encoder(d_in if masks is None else len(masks[i]), width, rep))
            torch.manual_seed(_seed("head", seed, i))
            heads.append(nn.Linear(rep, K))
        self.enc = nn.ModuleList(encs)
        self.head = nn.ModuleList(heads)
        self.Ks = list(Ks)

    def encode(self, i, X):
        return self.enc[i](X if self.masks is None else X[:, self.masks[i]])

    def n_params(self):
        return {"encoder_each": sum(p.numel() for p in self.enc[0].parameters()),
                "heads": [sum(p.numel() for p in h.parameters()) for h in self.head]}


class TorchMap:
    """Frozen affine LEACE map in float32 for the training graph: r = h - ((h - mu) @ Pr^T) @ Pl^T (official form)."""

    def __init__(self, leace_map):
        self.mu = torch.tensor(leace_map.mean_x, dtype=torch.float32)
        self.Pl = torch.tensor(leace_map.proj_left, dtype=torch.float32)
        self.Pr = torch.tensor(leace_map.proj_right, dtype=torch.float32)

    def __call__(self, h):
        return h - ((h - self.mu) @ self.Pr.T) @ self.Pl.T


def release(model, X, maps, i, detach_head=True, grad=True):
    h = model.encode(i, X)
    r = maps[i](h) if maps[i] is not None else h
    W, b = model.head[i].weight, model.head[i].bias
    if detach_head:
        W, b = W.detach(), b.detach()
    lg = F.linear(r, W, b)
    return r, lg


def view(model, X, maps, i):
    r, lg = release(model, X, maps, i)
    return torch.cat([r, lg - lg.mean(1, keepdim=True)], 1)


def views(model, X, maps, which, frozen=()):
    out = {}
    v = {}
    for i in (0, 1):
        if any(w in which for w in (f"v{i + 1}", "pair")):
            if i in frozen:
                with torch.no_grad():
                    v[i] = view(model, X, maps, i)
            else:
                v[i] = view(model, X, maps, i)
    for w in which:
        out[w] = v[0] if w == "v1" else v[1] if w == "v2" else torch.cat([v[0], v[1]], 1)
    return out


def task_losses(model, X, Y, maps, idxs=(0, 1)):
    out = {}
    for i in idxs:
        h = model.encode(i, X)
        r = maps[i](h) if maps[i] is not None else h
        out[i] = F.cross_entropy(model.head[i](r), Y[i])
    return out


# ------------------------------------------------------------------ arms
def arm_spec(name):
    full = {"v1": ["A", "B"], "v2": ["A", "B"], "pair": ["A", "B"]}
    if name == "U":
        return {"erasure": False, "stages": [{"train": [0, 1], "banks": {}, "P": [], "guards": [], "mode": "task"}]}
    if name == "L":
        return {"erasure": True, "stages": [{"train": [0, 1], "banks": {"v1": ["A", "B", "B2"], "v2": ["A", "B", "B2"]},
                                             "P": ["v1", "v2"], "guards": [0, 1], "mode": "projected"}]}
    if name == "J":
        return {"erasure": True, "stages": [{"train": [0, 1], "banks": full, "P": ["v1", "v2", "pair"], "guards": [0, 1],
                                             "mode": "projected"}]}
    if name == "JP":
        return {"erasure": True, "stages": [{"train": [0, 1], "banks": full, "P": ["v1", "v2", "pair"], "guards": [0, 1],
                                             "mode": "penalty"}]}
    if name in ("S12", "S21"):
        a, b = (0, 1) if name == "S12" else (1, 0)
        va, vb = f"v{a + 1}", f"v{b + 1}"
        return {"erasure": True, "stages": [
            {"train": [a], "banks": {va: ["A", "B"]}, "P": [va], "guards": [a], "mode": "projected"},
            {"train": [b], "banks": {vb: ["A", "B"], "pair": ["A", "B"]}, "P": [vb, "pair"], "guards": [b],
             "mode": "projected"}]}
    raise ValueError(name)


# ------------------------------------------------------------------ helpers
def flat_grad(loss, params, retain=False):
    gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])


def assign_add(params, vec, scale):
    with torch.no_grad():
        o = 0
        for p in params:
            n = p.numel()
            p.add_(vec[o:o + n].view_as(p), alpha=scale)
            o += n


def snapshot(params):
    return [p.detach().clone() for p in params]


def restore(params, snap):
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.copy_(s)


def entropy(prior):
    prior = np.asarray(prior, float)
    return float(-(prior * np.log(prior)).sum())


@dataclass
class Data:
    X: torch.Tensor          # defense_train inputs
    Y: dict                  # purpose -> LongTensor
    S: torch.Tensor          # sex
    prior: np.ndarray        # fitting prior of S
    guard_idx: np.ndarray    # fixed guard subset of defense_train
    seed: int


def fit_maps(model, data, idxs, maps):
    """Refit official LEACE (float64, defaults) for encoders idxs on defense_train representations."""
    from stored_model_eval.defenses import fit_leace
    Z = np.eye(2)[data.S.numpy()]
    out = {}
    with torch.no_grad():
        for i in idxs:
            H = model.encode(i, data.X).double().numpy()
            m = fit_leace(H, Z)
            maps[i] = TorchMap(m)
            out[i] = m
    return out


def warm_start(d_in, Ks, data, seed, log=None, masks=None):
    """Task-only warm start (Adam), shared bitwise by every arm of this seed."""
    model = Model(d_in, Ks, seed, masks=masks)
    opt = torch.optim.Adam(model.parameters(), lr=HP["warm_lr_adam"])
    n = data.X.shape[0]
    maps = [None, None]
    for ep in range(HP["warm_epochs"]):
        perm = np.random.default_rng([seed, 999, ep]).permutation(n)
        tot = 0.0
        for s in range(0, n, HP["batch"]):
            b = torch.from_numpy(perm[s:s + HP["batch"]])
            L = task_losses(model, data.X[b], {i: data.Y[i][b] for i in (0, 1)}, maps)
            loss = L[0] + L[1]
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if log:
            log({"phase": "warm", "epoch": ep, "train_task_loss": tot / n})
    return model


def guard_losses(model, data, maps, idxs=(0, 1)):
    g = torch.from_numpy(data.guard_idx)
    with torch.no_grad():
        L = task_losses(model, data.X[g], {i: data.Y[i][g] for i in (0, 1)}, maps, idxs)
    return {i: float(v) for i, v in L.items()}


def train_arm(arm, beta, warm_state, d_in, Ks, data, seed, budgets, log=None, lr=None, masks=None):
    """Run the protection phase of one arm from the shared warm start. Returns (model, maps_official, diagnostics)."""
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
    return model, diag
