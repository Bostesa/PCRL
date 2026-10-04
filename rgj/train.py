"""Refreshed guarded joint training engine (registered in METHOD_CARD.md / PROTOCOL.md).

Networks are the predecessor's (jcv.train.Model): per recipient i an encoder 83-64-64-16 (ReLU) and an affine training
head; release r_i = g_i(X) (identity map, no erasure anywhere). Critic views are v_i = [r_i, centred (W_w r_i + b_w)]
with ONE FIXED per-seed head (W_w, b_w) = the warm-start training head, pair = [v_1, v_2] (review R1-a, 2026-10-04,
before any nonzero fit: with the drifting training head the logit block left the snapshot's null space and the floored
block transform amplified it ~3e3x, degrading the online critics within a block; the fixed head keeps the views'
information identical - logits are affine in r - and makes the null space constant and transport exact). Every non-task arm carries the same
three banks (v1, v2, pair) x two kinds (A: 32-unit MLP, B: 64-64 MLP) with separate deterministic seeds; a local arm's
pair bank is a shadow bank (trained, never in the encoder gradient).

Recovery surrogate (all terms on the same rows; constant = DEFENSE_FIT prior):
    R_v = (CE_const - min(CE_const, CE_A, CE_B)) / H_fit(S)        (>= 0; exactly 0, no gradient, if the constant wins)
Objective (encoder gradient; heads get the task gradient only):
    L_task + beta (R_1 + R_2 + R_0)/3 + lambda_1 (R_1 - c_1) + lambda_2 (R_2 - c_2)
    joint arms R_0 = R_pair; local arms R_0 = (R_1 + R_2)/2  -> weights beta/2 on R_1, R_2 and none on the pair.
Update: u = -grad(L_1 + L_2) - grad P (encoder part), global-norm clip 5, plain SGD 0.05 (predecessor schedule).

Critic schedules:
  online (L-O, J-O; inherited): before every encoder step the floored ZCA transform is recomputed on a fixed 4096-row
      CRITIC_FIT reference subset, then 5 Adam steps per critic on CRITIC_FIT minibatches.
  refreshed (L-R, J-R, J-G, L-G): at protection epochs 0, 4, 8, 12, 16 and 20 the model is frozen, views are computed,
      a new transform is fitted on CRITIC_FIT rows and held fixed for the whole block; per view and kind a continued
      critic (first layer transported so its function is unchanged) and a deterministic fresh restart are trained
      (bounded refit: <= 15 epochs, patience 3, batch 256, Adam 3e-3, CRITIC_FIT) and the lower CRITIC_VAL
      cross-entropy wins. Between refits: 5 online Adam steps per encoder step with the block transform.
Guarded arms (J-G, L-G) update lambda_i <- clip(lambda_i + beta (R_i - c_i), 0, 3 beta) at each refit from CALIB rows
with the refitted critics. All critic rows are inside DEFENSE_FIT; no other role is touched here.
beta = 0 (hence lambda = 0) reproduces the task-only continuation bitwise from the same initialisation (registered check).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from jcv.train import Model, _seed, assign_add, critic, entropy, flat_grad, task_losses

torch.set_num_threads(1)

HP = {
    "width": 64, "rep_dim": 16, "batch": 256, "prot_epochs": 20, "sgd_lr": 0.05, "clip": 5.0, "critic_lr": 3e-3,
    "critic_steps_per_step": 5, "whiten_floor": 1e-8, "capped_ridge": 1e-4,
    "refit_epochs": (0, 4, 8, 12, 16, 20), "refit_max_epochs": 15, "refit_patience": 3, "refit_batch": 256,
    "checkpoints": (5, 10, 15, 20), "betas": (0.03, 0.1, 0.3), "lambda_cap_mult": 3.0, "c_margin": 0.005,
    "online_ref_size": 4096, "online_ref_seed": 20261004,
}
KS = [2, 6]
D_IN = 83
KINDS = ("A", "B")
VIEWS = ("v1", "v2", "pair")
DV = {"v1": 16 + KS[0], "v2": 16 + KS[1], "pair": 32 + KS[0] + KS[1]}
ARMS = {
    "TASK": {"base": "none", "refresh": False, "guard": False},
    "L-O": {"base": "local", "refresh": False, "guard": False},
    "L-R": {"base": "local", "refresh": True, "guard": False},
    "J-G": {"base": "joint", "refresh": True, "guard": True},
    "L-G": {"base": "local", "refresh": True, "guard": True},
    "J-R": {"base": "joint", "refresh": True, "guard": False},
    "J-O": {"base": "joint", "refresh": False, "guard": False},
}
STAGE_ORDER_SALT = {"B": 0, "C": 1}     # data order: rng([seed, salt, epoch]); stage B == task-only line epochs 0..19


# ------------------------------------------------------------------ critic-input transforms
class Transform:
    """Frozen critic-input transform T(V) = (V - mu) @ W, statistics from fixed reference rows of one snapshot.
    floored: ZCA with eigenvalues floored at 1e-8 x max (predecessor); capped: ZCA of C + 1e-4 max(ev) I (amplification
    <= 100x the top direction; invertible); raw: centring only (W = I). Every kind keeps all coordinates."""

    def __init__(self, V=None, kind="floored", state=None):
        if state is not None:
            self.kind, self.mu64, self.W64, self.ev = state["kind"], state["mu"], state["W"], state["ev"]
        else:
            V = V.double()
            mu = V.mean(0)
            C = torch.cov((V - mu).T)
            ev, U = torch.linalg.eigh(C)
            if kind == "floored":
                W = (U / torch.sqrt(torch.clamp(ev, min=float(ev.max()) * HP["whiten_floor"]))) @ U.T
            elif kind == "capped":
                W = (U / torch.sqrt(torch.clamp(ev, min=0.0) + float(ev.max()) * HP["capped_ridge"])) @ U.T
            elif kind == "raw":
                W = torch.eye(V.shape[1], dtype=torch.float64)
            else:
                raise ValueError(kind)
            self.kind, self.mu64, self.W64, self.ev = kind, mu, W, ev
        self.mu = self.mu64.float()
        self.W = self.W64.float()

    def __call__(self, V):
        return (V - self.mu) @ self.W

    def state(self):
        return {"kind": self.kind, "mu": self.mu64.clone(), "W": self.W64.clone(), "ev": self.ev.clone()}

    def spectrum(self):
        s = torch.linalg.svdvals(self.W64)
        return {"kind": self.kind, "cov_eig_max": float(self.ev.max()), "cov_eig_min": float(self.ev.min()),
                "W_sv_max": float(s.max()), "W_sv_min": float(s.min()), "W_condition": float(s.max() / s.min())}


def transport_first_layer(c, T_old, T_new):
    """Rewrite the first Linear layer of critic c so that c_new(T_new(V)) == c_old(T_old(V)) for every V.
    Row-vector convention x = (V - mu) W (W symmetric), layer y = x A^T + b:
        A' = A W_old W_new^{-1},   b' = b + (mu_new - mu_old) W_old A^T   (float64 algebra, stored as float32)."""
    lin = c[0]
    A = lin.weight.detach().double()
    b = lin.bias.detach().double()
    WoAt = T_old.W64 @ A.T
    At_new = torch.linalg.solve(T_new.W64, WoAt)
    b_new = b + (T_new.mu64 - T_old.mu64) @ WoAt
    with torch.no_grad():
        lin.weight.copy_(At_new.T.float())
        lin.bias.copy_(b_new.float())
    return c


# ------------------------------------------------------------------ surrogate, coefficients, multipliers
def recovery(logits, S, logprior, H):
    """R = (CE_const - min(CE_const, CE_1, ..., CE_m)) / H on the same rows. Returns (R tensor, index, CE floats);
    index == len(logits) means the constant won (R is an exact zero with no critic/encoder gradient path)."""
    ces = [F.cross_entropy(l, S) for l in logits]
    ce_const = F.nll_loss(logprior.expand(len(S), 2), S)
    allc = ces + [ce_const]
    vals = [float(x) for x in allc]
    j = int(np.argmin(vals))
    R = (ce_const - allc[j]) / H
    return R, j, vals


def coefficients(base, beta, lam):
    """Encoder-gradient weights on (R_v1, R_v2, R_pair)."""
    if base == "joint":
        return {"v1": beta / 3 + lam[0], "v2": beta / 3 + lam[1], "pair": beta / 3}
    if base == "local":
        return {"v1": beta / 2 + lam[0], "v2": beta / 2 + lam[1], "pair": 0.0}
    return {"v1": 0.0, "v2": 0.0, "pair": 0.0}


def base_coefficients(base, beta):
    return coefficients(base, beta, (0.0, 0.0))


def dual_update(lam, R, c, beta):
    return float(min(max(lam + beta * (R - c), 0.0), HP["lambda_cap_mult"] * beta))


# ------------------------------------------------------------------ data
class TData:
    """DEFENSE_FIT tensors and the fixed critic subroles (indices local to DEFENSE_FIT)."""

    def __init__(self, D):
        tr = D["idx"]["DEFENSE_FIT"]
        pos = {int(r): j for j, r in enumerate(tr)}
        self.X = torch.from_numpy(D["X"][tr])
        self.Y = {0: torch.from_numpy(D["y"]["income"][tr]), 1: torch.from_numpy(D["y"]["occupation_group"][tr])}
        self.S = torch.from_numpy(D["sex"][tr])
        self.n = len(tr)
        self.prior = np.bincount(D["sex"][tr], minlength=2) / len(tr)
        self.cf = np.array([pos[int(i)] for i in D["idx"]["CRITIC_FIT"]])
        self.cv = np.array([pos[int(i)] for i in D["idx"]["CRITIC_VAL"]])
        self.cal = np.array([pos[int(i)] for i in D["idx"]["CALIB"]])
        self.ref = np.sort(np.random.default_rng(HP["online_ref_seed"]).choice(self.cf, min(HP["online_ref_size"],
                                                                                         len(self.cf)), replace=False))


# ------------------------------------------------------------------ critics
def head_of(state):
    """Fixed critic-view head of a seed: the warm-start training heads (detached float32 copies)."""
    return {i: (state[f"head.{i}.weight"].detach().clone().float(), state[f"head.{i}.bias"].detach().clone().float())
            for i in (0, 1)}


def critic_views(model, X, which, head, grad=False):
    """v_i = [r_i, centred(W_w r_i + b_w)] with the fixed per-seed head; pair = [v1, v2]. grad=True keeps the encoder
    graph (penalty); the head is a constant either way."""
    need = set()
    for w in which:
        need |= {0, 1} if w == "pair" else {int(w[1]) - 1}
    v = {}
    with (torch.enable_grad() if grad else torch.no_grad()):
        for i in sorted(need):
            r = model.encode(i, X)
            lg = F.linear(r, head[i][0], head[i][1])
            v[i] = torch.cat([r, lg - lg.mean(1, keepdim=True)], 1)
        return {w: (v[0] if w == "v1" else v[1] if w == "v2" else torch.cat([v[0], v[1]], 1)) for w in which}


def new_critic(seed, view, kind, tag):
    torch.manual_seed(_seed("rgj-critic", seed, view, kind, tag))
    return critic(kind, DV[view])


def ce_of(c, Z, S):
    with torch.no_grad():
        return float(F.cross_entropy(c(Z), S))


def fit_bounded(c, Zf, Sf, Zv, Sv, rng_seed):
    """Bounded refit: Adam(critic_lr), batch 256, <= 15 epochs, patience 3 on validation CE; keeps the best state
    (including the starting state). Returns (critic, receipt)."""
    opt = torch.optim.Adam(c.parameters(), lr=HP["critic_lr"])
    best = ce_of(c, Zv, Sv)
    start = best
    best_state = copy.deepcopy(c.state_dict())
    bad, ep_run, upd = 0, 0, 0
    n = len(Sf)
    for ep in range(HP["refit_max_epochs"]):
        perm = np.random.default_rng(list(rng_seed) + [ep]).permutation(n)
        for s in range(0, n, HP["refit_batch"]):
            b = torch.from_numpy(perm[s:s + HP["refit_batch"]])
            loss = F.cross_entropy(c(Zf[b]), Sf[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            upd += 1
        ep_run += 1
        v = ce_of(c, Zv, Sv)
        if v < best - 1e-9:
            best, bad, best_state = v, 0, copy.deepcopy(c.state_dict())
        else:
            bad += 1
            if bad >= HP["refit_patience"]:
                break
    c.load_state_dict(best_state)
    return c, {"start_val_ce": start, "best_val_ce": best, "epochs": ep_run, "updates": upd,
               "early_stopped": bool(bad >= HP["refit_patience"])}


def frozen_views(model, X, head):
    return critic_views(model, X, list(VIEWS), head, grad=False)


def snapshot_state(model, banks, T, lam, head=None):
    return {"model": copy.deepcopy(model.state_dict()),
            "critic_head": ({i: (head[i][0].clone(), head[i][1].clone()) for i in head} if head is not None else None),
            "critics": {v: {k: copy.deepcopy(banks[v][k].state_dict()) for k in KINDS} for v in banks},
            "transforms": {v: (T[v].state() if T.get(v) is not None else None) for v in VIEWS},
            "lam": list(lam)}


def calib_recovery(model, banks, T, data, logprior, H, head):
    Vc = frozen_views(model, data.X[torch.from_numpy(data.cal)], head)
    Sc = data.S[torch.from_numpy(data.cal)]
    out = {}
    with torch.no_grad():
        for v in VIEWS:
            R, j, vals = recovery([banks[v][k](T[v](Vc[v])) for k in KINDS], Sc, logprior, H)
            out[v] = {"R": float(R), "winner": (list(KINDS) + ["const"])[j], "ce": vals}
    return out


def refit_banks(model, banks, T, data, seed, ep, tkind, logprior, H, head, tag="refit"):
    """Frozen-snapshot bounded refit of every bank. Mutates banks/T; returns receipts."""
    V = frozen_views(model, data.X, head)
    cf, cv = torch.from_numpy(data.cf), torch.from_numpy(data.cv)
    rec = {}
    for v in VIEWS:
        T_new = Transform(V[v][cf], tkind)
        Zf, Zv = T_new(V[v][cf]), T_new(V[v][cv])
        rec[v] = {"transform": T_new.spectrum(), "kinds": {}}
        for k in KINDS:
            cont = copy.deepcopy(banks[v][k])
            if T.get(v) is not None:
                transport_first_layer(cont, T[v], T_new)
            rest = new_critic(seed, v, k, f"{tag}-restart-{ep}")
            cont, rc = fit_bounded(cont, Zf, data.S[cf], Zv, data.S[cv], [seed, _seed(v, k), ep, 0])
            rest, rr = fit_bounded(rest, Zf, data.S[cf], Zv, data.S[cv], [seed, _seed(v, k), ep, 1])
            choice = "continued" if rc["best_val_ce"] <= rr["best_val_ce"] else "restart"
            banks[v][k] = cont if choice == "continued" else rest
            rec[v]["kinds"][k] = {"continued": rc, "restart": rr, "choice": choice,
                                  "online_val_ce_before_refit": rc["start_val_ce"],
                                  "gap_online_minus_refit": rc["start_val_ce"] - min(rc["best_val_ce"], rr["best_val_ce"])}
        T[v] = T_new
    return rec


# ------------------------------------------------------------------ one run
def train_run(arm, beta, init_state, data, seed, stage, init_critics=None, n_epochs=None, tkind="floored",
              capture_steps=(), log=None, lr=None, budgets=None, ckpt_epochs=None, critic_head=None):
    """One protection run from init_state. Returns (model, diag, checkpoints, captures, final).
    checkpoints[e]: state after e epochs (model, critics in use, transforms, lambda); captures[s]: aligned snapshot at
    global step s (theta_{s-1}, critics after their step-s updates, transforms); final: theta_{T-1}/theta_T, the
    online critics aligned with theta_{T-1} and, for refreshed arms, the epoch-20 refit critics aligned with theta_T."""
    spec = ARMS[arm]
    n_epochs = HP["prot_epochs"] if n_epochs is None else n_epochs
    lr = HP["sgd_lr"] if lr is None else lr
    model = Model(D_IN, KS, seed)
    model.load_state_dict(init_state)
    enc_params = [p for i in (0, 1) for p in model.enc[i].parameters()]
    head_params = [p for i in (0, 1) for p in model.head[i].parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in enc_params + head_params:
        p.requires_grad_(True)
    ne = sum(p.numel() for p in enc_params)
    H = entropy(data.prior)
    logprior = torch.tensor(np.log(data.prior), dtype=torch.float32)
    has_critics = spec["base"] != "none"
    head = critic_head
    if has_critics and head is None:
        raise ValueError("critic arms need the fixed per-seed critic-view head (head_of(warm state))")
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
    lam = [0.0, 0.0]
    ckpt_epochs = HP["checkpoints"] if ckpt_epochs is None else tuple(ckpt_epochs)
    c_t = list(budgets) if (spec["guard"] and budgets is not None) else None
    if spec["guard"] and c_t is None:
        raise ValueError("guarded arm needs calibrated budgets c")
    diag = {"arm": arm, "beta": beta, "lr": lr, "seed": seed, "stage": stage, "tkind": tkind, "encoder_updates": 0,
            "critic_online_updates": 0, "critic_refit_updates": 0, "refits": [], "lambda_trace": [], "nonfinite": 0,
            "clip_hits": 0, "norms": [], "epochs": [], "early_stops": 0, "restart_chosen": 0, "continued_chosen": 0,
            "coefficients_base": base_coefficients(spec["base"], beta), "c": c_t}
    checkpoints, captures, final = {}, {}, {}
    salt = STAGE_ORDER_SALT[stage]
    n = data.n
    step = 0
    total_steps = n_epochs * math.ceil(n / HP["batch"])
    t0 = time.time()
    for ep in range(n_epochs + 1):
        if ep in ckpt_epochs and ep > 0:
            checkpoints[ep] = snapshot_state(model, banks, T, lam, head) if has_critics else {"model": copy.deepcopy(model.state_dict())}
            checkpoints[ep]["step"] = step
        if has_critics and spec["refresh"] and ep in HP["refit_epochs"] and ep > 0:   # review A4: block input scale
            Vb = frozen_views(model, data.X[torch.from_numpy(data.cf)], head)
            with torch.no_grad():
                diag.setdefault("block_end_max_abs_z", []).append(
                    {"epoch": ep, **{v: float(T[v](Vb[v]).abs().max()) for v in VIEWS}})
        if has_critics and spec["refresh"] and ep in HP["refit_epochs"]:
            rr = refit_banks(model, banks, T, data, seed, ep, tkind, logprior, H, head)
            reset_opts()
            cal = calib_recovery(model, banks, T, data, logprior, H, head)
            entry = {"epoch": ep, "step": step, "receipts": rr, "calib": cal}
            for v in rr:
                for k in rr[v]["kinds"]:
                    q = rr[v]["kinds"][k]
                    diag["critic_refit_updates"] += q["continued"]["updates"] + q["restart"]["updates"]
                    diag["early_stops"] += int(q["continued"]["early_stopped"]) + int(q["restart"]["early_stopped"])
                    diag["restart_chosen" if q["choice"] == "restart" else "continued_chosen"] += 1
            if spec["guard"]:
                old = list(lam)
                if ep < n_epochs:
                    lam = [dual_update(lam[i], cal[f"v{i + 1}"]["R"], c_t[i], beta) for i in (0, 1)]
                entry["lambda_before"], entry["lambda_after"] = old, list(lam)
                diag["lambda_trace"].append({"epoch": ep, "R_calib": [cal["v1"]["R"], cal["v2"]["R"]], "c": c_t,
                                             "lambda": list(lam), "effective_weights": coefficients(spec["base"], beta, lam)})
            diag["refits"].append(entry)
            if ep == n_epochs:
                final["refit_critics"] = snapshot_state(model, banks, T, lam, head)
        if ep == n_epochs:
            break
        perm = np.random.default_rng([seed, salt, ep]).permutation(n)
        crng = np.random.default_rng([seed, salt, ep, 7])
        ep_log = {"epoch": ep, "R": {v: [] for v in VIEWS}, "clip": 0}
        coef = coefficients(spec["base"], beta, lam)
        for s in range(0, n, HP["batch"]):
            b = torch.from_numpy(perm[s:s + HP["batch"]])
            step += 1
            # ---- critic updates (CRITIC_FIT rows, detached views)
            if has_critics:
                if not spec["refresh"]:
                    Vref = critic_views(model, data.X[torch.from_numpy(data.ref)], list(VIEWS), head)
                    T_step = {v: Transform(Vref[v], "floored") for v in VIEWS}
                    if step == 1 and init_critics is not None:   # inherited critics: one transport into the first transform
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
            if step in capture_steps or step == total_steps:
                snap = snapshot_state(model, banks, T, lam, head) if has_critics else {"model": copy.deepcopy(model.state_dict())}
                if step in capture_steps:
                    captures[step] = snap
                if step == total_steps:
                    final["theta_T_minus_1"] = snap
            Yb = {i: data.Y[i][b] for i in (0, 1)}
            Lt = task_losses(model, data.X[b], Yb, [None, None], [0, 1])
            ltask = sum(Lt.values())
            t = flat_grad(ltask, enc_params + head_params)
            t = -t
            d_enc = torch.zeros(ne)
            pnorm = 0.0
            active = [v for v in VIEWS if coef[v] > 0]
            if has_critics and active:
                V = critic_views(model, data.X[b], active, head, grad=True)
                P = 0.0
                for v in active:
                    R, j, _ = recovery([banks[v][k](T[v](V[v])) for k in KINDS], data.S[b], logprior, H)
                    ep_log["R"][v].append(float(R))
                    P = P + coef[v] * R
                if isinstance(P, torch.Tensor) and P.requires_grad:
                    d_enc = -flat_grad(P, enc_params)
                if not torch.isfinite(d_enc).all():
                    diag["nonfinite"] += 1
                    d_enc = torch.zeros(ne)
                pnorm = float(d_enc.norm())
            u = torch.cat([t[:ne] + d_enc, t[ne:]])
            nrm = float(u.norm())
            if not math.isfinite(nrm):
                diag["nonfinite"] += 1
                continue
            if nrm > HP["clip"]:
                u = u * (HP["clip"] / nrm)
                diag["clip_hits"] += 1
                ep_log["clip"] += 1
            if step % 20 == 0:
                diag["norms"].append({"step": step, "task": float(t[:ne].norm()), "penalty": pnorm, "total": nrm})
            assign_add(enc_params + head_params, u, lr)
            diag["encoder_updates"] += 1
        ep_log["R"] = {v: (float(np.mean(x)) if x else None) for v, x in ep_log["R"].items()}
        ep_log["lambda"] = list(lam)
        diag["epochs"].append(ep_log)
        if log:
            log(ep_log)
    for p in model.parameters():
        p.requires_grad_(False)
    final["theta_T"] = copy.deepcopy(model.state_dict())
    diag["wall_s"] = time.time() - t0
    return model, diag, checkpoints, captures, final


def task_line(init_state, data, seed, n_epochs, stage="B", save_every=5):
    """Task-only continuation through train_run's own update path (exact parity with every arm at beta = 0), with
    checkpoints every save_every epochs."""
    model, diag, ck, _, _ = train_run("TASK", 0.0, init_state, data, seed, stage, n_epochs=n_epochs,
                                      ckpt_epochs=range(save_every, n_epochs + 1, save_every))
    return model, diag, ck
