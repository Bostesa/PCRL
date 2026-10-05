"""Online strength-frontier training engine (registered in PROTOCOL.md / METHOD_CARD.md).

Inherited unchanged (pinned rgj.train / jcv.train): two encoders 83-64-64-16 (ReLU) with affine task heads
(jcv.train.Model), source task losses, SGD 0.05, global update-norm clip 5, batch 256; critic views
v_i = [r_i, centred(W_w r_i + b_w)] with the FIXED warm-start training head (W_w, b_w) of the seed, pair = [v1, v2];
critic kinds A (32-unit MLP) and B (64-64 MLP) per view with deterministic init seeds (rgj.train.new_critic, tag "init");
ONLINE schedule only: before every encoder step the floored ZCA transform is recomputed on the fixed 4096-row
CRITIC_FIT reference subset, then 5 Adam(3e-3) steps per critic on CRITIC_FIT minibatches. No refits, restarts,
feedback, learned budgets or projections exist in this module.

Recovery surrogate (all terms on the encoder minibatch, constant = OSF_DEFENSE_FIT SEX prior):
    R_v = (CE_const - min(CE_const, CE_A, CE_B)) / H        (exactly 0 with no gradient path if the constant wins)
Protection proxy:  joint P = (R_v1 + R_v2 + R_pair)/3;   local P = (R_v1 + R_v2)/2 (pair bank = trained shadow bank,
zero weight). t_i = d L_task / d enc_i (only recipient i's task loss reaches encoder i), p_i = d P / d enc_i.
    RAW  (beta):     q_i = beta p_i, computed exactly as rgj.train J-O / L-O: d_enc = -grad(sum_v c_v R_v) with
                     c = beta/3 (joint) or beta/2 on v1, v2 (local) - same tensors, same operation order (bitwise).
    NORM (rho, a):   q_i = stop_grad(rho s_i ||t_i|| / ||p_i||) p_i via smf.train.normalized_direction (float64 norms,
                     scalar capped at 100, q_i = 0 and a logged event if ||p_i|| <= 1e-12 or ||t_i|| = 0);
                     s_income = a / sqrt((a^2+1)/2), s_occupation = 1 / sqrt((a^2+1)/2)  (RMS of s = 1).
    TASK:            q = 0, no critics.
Update (every mode): t = -grad(L_task) over encoders + heads; u = [t_enc - q, t_heads]; clip ||u|| <= 5;
params += lr u. Heads receive the task gradient only.
RNG (common convention, explicit numeric salt, default 0 = the raw stage-B order of rgj = smf stage A):
task minibatches rng([seed, salt, ep]); critic minibatches rng([seed, salt, ep, 7]); nothing else draws randomness;
instrumentation, snapshots and receipts never consume RNG. beta = 0 / rho = 0 reproduce TASK bitwise (critics still
train) - registered parity check.
Per-step receipts (steps.npz): t/p/q norms per encoder, q/t ratios (realized: zero steps count 0), combined
||q_enc||/||t_enc||, RMS of the two ratios, cos(t_i, p_i), scale, cap, zero reason, pre/post-clip total, encoder and
head update norms, clip factor, R per active view, cumulative critic updates, epoch/step, minibatch and transform
fingerprints. Zero reasons: 0 none, 1 P has no gradient path (constant won on every active view), 2 ||p_i|| <= tol,
3 ||t_i|| = 0, 4 nonfinite p, 5 no protection term (TASK or strength 0).
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

import numpy as np
import torch

from jcv.train import Model, assign_add, entropy, flat_grad, task_losses
from rgj import train as RT
from rgj.train import KINDS, VIEWS, Transform, critic_views, head_of, new_critic, recovery, snapshot_state  # noqa: F401
from smf.train import normalized_direction

torch.set_num_threads(1)

HP = {"width": 64, "rep_dim": 16, "batch": 256, "sgd_lr": 0.05, "clip": 5.0, "critic_lr": RT.HP["critic_lr"],
      "critic_steps_per_step": RT.HP["critic_steps_per_step"], "whiten_floor": RT.HP["whiten_floor"],
      "online_ref_size": RT.HP["online_ref_size"], "online_ref_seed": RT.HP["online_ref_seed"], "epochs": 40,
      "A_MAX": 100.0, "ZERO_TOL": 1e-12, "salt": 0, "ckpt_epochs": (20, 40),
      "progress_fractions": (0.0, 0.25, 0.5, 0.75, 1.0)}
D_IN, KS = RT.D_IN, RT.KS
TREAT = {"J": "joint", "L": "local"}
Z_NONE, Z_NOGRAD, Z_P, Z_T, Z_NONFINITE, Z_OFF = 0, 1, 2, 3, 4, 5


def g(x):
    return f"{x:g}"


# ------------------------------------------------------------------ registered bank
def config_id(c):
    if c["mode"] == "TASK":
        return "U"
    if c["mode"] == "RAW":
        return f"RAW-{c['treat']}|b{g(c['beta'])}"
    return f"NORM-{c['treat']}|r{g(c['rho'])}|a{g(c['a'])}"


def parse_id(cid):
    if cid == "U":
        return {"mode": "TASK", "treat": None}
    fam, *rest = cid.split("|")
    mode, treat = fam.split("-")
    kv = {r[0]: float(r[1:]) for r in rest}
    if mode == "RAW":
        return {"mode": "RAW", "treat": treat, "beta": kv["b"]}
    return {"mode": "NORM", "treat": treat, "rho": kv["r"], "a": kv["a"]}


def bank(kind="full"):
    betas = (0.1, 0.3, 0.6) if kind == "full" else (0.1, 0.3)
    sym = (1.5, 3.0, 5.0) if kind == "full" else (1.5, 3.0)
    alloc_rhos = (3.0, 5.0) if kind == "full" else (3.0,)
    out = [{"mode": "TASK", "treat": None}]
    for t in ("J", "L"):
        out += [{"mode": "RAW", "treat": t, "beta": b} for b in betas]
        out += [{"mode": "NORM", "treat": t, "rho": r, "a": 1.0} for r in sym]
        out += [{"mode": "NORM", "treat": t, "rho": r, "a": a} for r in alloc_rhos for a in (0.5, 2.0)]
    return out


def allocation(a):
    """(s_income, s_occupation) with RMS 1: s_1 = a / sqrt((a^2+1)/2), s_2 = 1 / sqrt((a^2+1)/2)."""
    r = math.sqrt((a * a + 1.0) / 2.0)
    return a / r, 1.0 / r


def proxy_coefficients(c):
    """Encoder-gradient weights on (R_v1, R_v2, R_pair). RAW: rgj.train.coefficients(base, beta, (0, 0)) exactly.
    NORM: the unit proxy P (scale is removed by the normalisation; relative view weights only)."""
    if c["mode"] == "RAW":
        return RT.coefficients(TREAT[c["treat"]], c["beta"], (0.0, 0.0))
    if c["mode"] == "NORM":
        if c["treat"] == "J":
            return {"v1": 1.0 / 3, "v2": 1.0 / 3, "pair": 1.0 / 3}
        return {"v1": 0.5, "v2": 0.5, "pair": 0.0}
    return {"v1": 0.0, "v2": 0.0, "pair": 0.0}


def strength(c):
    return c.get("beta", 0.0) if c["mode"] == "RAW" else c.get("rho", 0.0) if c["mode"] == "NORM" else 0.0


# ------------------------------------------------------------------ fingerprints (no RNG)
def fp64(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return int.from_bytes(h.digest()[:8], "little") >> 1


def state_sha(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def critics_sha(banks):
    return state_sha({f"{v}|{k}|{n}": t for v in banks for k in banks[v] for n, t in banks[v][k].state_dict().items()})


def transform_fp(T):
    return fp64(*[np.concatenate([T[v].mu64.numpy().ravel(), T[v].W64.numpy().ravel()]) for v in VIEWS])


def snap_full(model, banks, T, head, opts):
    sn = snapshot_state(model, banks, T, [0.0, 0.0], head)
    sn["opts"] = {v: {k: copy.deepcopy(opts[v][k].state_dict()) for k in KINDS} for v in opts}
    return sn


def capture_steps_for(n_epochs, n):
    S = n_epochs * math.ceil(n / HP["batch"])
    return {f: max(1, int(round(f * S))) for f in HP["progress_fractions"]}


# ------------------------------------------------------------------ one run
STEP_FIELDS = ("epoch", "step", "mb_fp", "tf_fp", "t_norm", "p_norm", "q_norm", "ratio", "realized_ratio", "cos",
               "scale", "cap", "zero", "comb_ratio", "rms_ratio", "pre_total", "post_total", "kappa", "pre_enc",
               "post_enc", "pre_head", "R", "critic_updates")


def _alloc_steps(S):
    z2 = lambda dt: np.zeros((S, 2), dt)    # noqa: E731
    return {"epoch": np.zeros(S, np.int16), "step": np.zeros(S, np.int32), "mb_fp": np.zeros(S, np.int64),
            "tf_fp": np.zeros(S, np.int64), "t_norm": z2(np.float64), "p_norm": np.full((S, 2), np.nan),
            "q_norm": z2(np.float64), "ratio": np.full((S, 2), np.nan), "realized_ratio": z2(np.float64),
            "cos": np.full((S, 2), np.nan), "scale": z2(np.float64), "cap": z2(np.bool_), "zero": z2(np.int8),
            "comb_ratio": np.zeros(S), "rms_ratio": np.zeros(S), "pre_total": np.zeros(S), "post_total": np.zeros(S),
            "kappa": np.ones(S), "pre_enc": z2(np.float64), "post_enc": z2(np.float64), "pre_head": np.zeros(S),
            "R": np.full((S, 3), np.nan), "critic_updates": np.zeros(S, np.int64), "applied": np.zeros(S, np.bool_)}


def train_run(c, init_state, data, seed, critic_head, n_epochs=None, salt=None, ckpt_epochs=None, capture_steps=(),
              lr=None, log=None):
    """One protection/task continuation from init_state. Returns (model, diag, checkpoints, captures, final, steps).
    checkpoints[e]: state after e epochs; captures[s]: aligned snapshot at global step s (theta_{s-1}, critics after
    their step-s updates, the transform in use); final: theta_{T-1} snapshot and theta_T."""
    n_epochs = HP["epochs"] if n_epochs is None else n_epochs
    capture_steps = set(capture_steps.values()) if isinstance(capture_steps, dict) else set(capture_steps)  # review A3
    salt = HP["salt"] if salt is None else salt
    lr = HP["sgd_lr"] if lr is None else lr
    ckpt_epochs = HP["ckpt_epochs"] if ckpt_epochs is None else tuple(ckpt_epochs)
    mode = c["mode"]
    assert mode in ("TASK", "RAW", "NORM")
    model = Model(D_IN, KS, seed)
    model.load_state_dict(init_state)
    enc_params = [p for i in (0, 1) for p in model.enc[i].parameters()]
    head_params = [p for i in (0, 1) for p in model.head[i].parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in enc_params + head_params:
        p.requires_grad_(True)
    n0 = sum(p.numel() for p in model.enc[0].parameters())
    ne = sum(p.numel() for p in enc_params)
    blocks = ((0, n0), (n0, ne))
    H = entropy(data.prior)
    logprior = torch.tensor(np.log(data.prior), dtype=torch.float32)
    has_critics = mode != "TASK"
    head = critic_head
    if has_critics and head is None:
        raise ValueError("critic modes need the fixed per-seed critic-view head (head_of(warm state))")
    banks, opts, T = {}, {}, {v: None for v in VIEWS}
    if has_critics:
        for v in VIEWS:
            banks[v] = {k: new_critic(seed, v, k, "init") for k in KINDS}
            opts[v] = {k: torch.optim.Adam(banks[v][k].parameters(), lr=HP["critic_lr"]) for k in KINDS}
    coef = proxy_coefficients(c)
    st = strength(c)
    s_alloc = allocation(c.get("a", 1.0)) if mode == "NORM" else (1.0, 1.0)
    active = [v for v in VIEWS if coef[v] > 0]
    protect = has_critics and bool(active) and st > 0
    diag = {"config": config_id(c), "cfg": c, "seed": seed, "salt": salt, "lr": lr, "n_epochs": n_epochs,
            "coefficients": coef, "allocation": list(s_alloc), "encoder_updates": 0, "critic_online_updates": 0,
            "nonfinite": 0, "clip_hits": 0, "cap_hits": [0, 0], "zero_events": [0, 0], "epochs": [],
            "undefined_ratio_steps": [0, 0],
            "init_state_sha256": state_sha(init_state),
            "critic_init_sha256": critics_sha(banks) if has_critics else None,
            "fixed_head_sha256": (state_sha({f"{i}|{j}": head[i][j] for i in head for j in (0, 1)})
                                  if head is not None else None)}
    checkpoints, captures, final = {}, {}, {}
    n = data.n
    per_ep = math.ceil(n / HP["batch"])
    S = n_epochs * per_ep
    rec = _alloc_steps(S)
    step = 0
    t0 = time.time()
    for ep in range(n_epochs + 1):
        if ep in ckpt_epochs and ep > 0:
            checkpoints[ep] = snap_full(model, banks, T, head, opts) if has_critics else {"model": copy.deepcopy(model.state_dict())}
            checkpoints[ep]["step"] = step
        if ep == n_epochs:
            break
        perm = np.random.default_rng([seed, salt, ep]).permutation(n)
        crng = np.random.default_rng([seed, salt, ep, 7])
        for s in range(0, n, HP["batch"]):
            bi = perm[s:s + HP["batch"]]
            b = torch.from_numpy(bi)
            step += 1
            j = step - 1
            rec["epoch"][j], rec["step"][j], rec["mb_fp"][j] = ep, step, fp64(bi.astype(np.int64))
            # ---- critic updates (CRITIC_FIT rows, detached views), rgj online order
            if has_critics:
                Vref = critic_views(model, data.X[torch.from_numpy(data.ref)], list(VIEWS), head)
                T = {v: Transform(Vref[v], "floored") for v in VIEWS}
                rec["tf_fp"][j] = transform_fp(T)
                for _ in range(HP["critic_steps_per_step"]):
                    cb = torch.from_numpy(crng.choice(data.cf, HP["batch"], replace=False))
                    V = critic_views(model, data.X[cb], list(VIEWS), head)
                    for v in VIEWS:
                        for k in KINDS:
                            loss = torch.nn.functional.cross_entropy(banks[v][k](T[v](V[v])), data.S[cb])
                            opts[v][k].zero_grad()
                            loss.backward()
                            opts[v][k].step()
                            diag["critic_online_updates"] += 1
            rec["critic_updates"][j] = diag["critic_online_updates"]
            if step in capture_steps or step == S:
                snap = snap_full(model, banks, T, head, opts) if has_critics else {"model": copy.deepcopy(model.state_dict())}
                snap["step"] = step
                if step in capture_steps:
                    captures[step] = snap
                if step == S:
                    final["theta_T_minus_1"] = snap
            # ---- task gradient (encoders + heads), rgj ordering
            Yb = {i: data.Y[i][b] for i in (0, 1)}
            Lt = task_losses(model, data.X[b], Yb, [None, None], [0, 1])
            gt = flat_grad(sum(Lt.values()), enc_params + head_params)
            t = -gt
            d_enc = torch.zeros(ne)
            zero = [Z_OFF, Z_OFF] if not protect else [Z_NONE, Z_NONE]
            infos = [None, None]
            if protect:
                V = critic_views(model, data.X[b], active, head, grad=True)
                P = 0.0
                for vi, v in enumerate(VIEWS):
                    if v in active:
                        Rv, _, _ = recovery([banks[v][k](T[v](V[v])) for k in KINDS], data.S[b], logprior, H)
                        rec["R"][j, vi] = float(Rv)
                        P = P + coef[v] * Rv
                if not (isinstance(P, torch.Tensor) and P.requires_grad):
                    zero = [Z_NOGRAD, Z_NOGRAD]
                elif mode == "RAW":                     # rgj.train J-O / L-O, unchanged operations
                    d_enc = -flat_grad(P, enc_params)
                    if not torch.isfinite(d_enc).all():
                        diag["nonfinite"] += 1
                        d_enc = torch.zeros(ne)
                        zero = [Z_NONFINITE, Z_NONFINITE]
                else:                                    # NORM
                    pvec = flat_grad(P, enc_params)
                    if not torch.isfinite(pvec).all():
                        diag["nonfinite"] += 1
                        zero = [Z_NONFINITE, Z_NONFINITE]
                    else:
                        parts = []
                        for i, (lo, hi) in enumerate(blocks):
                            qi, info = normalized_direction(gt[lo:hi], pvec[lo:hi], st * s_alloc[i],
                                                            a_max=HP["A_MAX"], zero_tol=HP["ZERO_TOL"])
                            parts.append(qi)
                            infos[i] = info
                            if info["zero"]:
                                zero[i] = Z_P if info["zero"] == "p" else Z_T
                        d_enc = -torch.cat(parts)
            # ---- receipts (pure functions of the tensors above; no RNG)
            with torch.no_grad():
                q = -d_enc
                tq2, tt2 = 0.0, 0.0
                for i, (lo, hi) in enumerate(blocks):
                    gi64, qi64 = gt[lo:hi].double(), q[lo:hi].double()
                    tn, qn = float(gi64.norm()), float(qi64.norm())
                    rec["t_norm"][j, i], rec["q_norm"][j, i] = tn, qn
                    tt2 += tn * tn
                    tq2 += qn * qn
                    if mode == "RAW" and zero[i] == Z_NONE:
                        if qn == 0.0:
                            zero[i] = Z_P
                        else:
                            rec["p_norm"][j, i] = qn / c["beta"]
                            rec["cos"][j, i] = float(torch.dot(gi64, qi64)) / (tn * qn) if tn > 0 else np.nan
                            rec["scale"][j, i] = c["beta"]
                    elif mode == "NORM" and infos[i] is not None:
                        rec["p_norm"][j, i] = infos[i]["p_norm"]
                        if infos[i].get("cos") is not None:
                            rec["cos"][j, i] = infos[i]["cos"]
                        rec["scale"][j, i] = infos[i]["a"]
                        rec["cap"][j, i] = infos[i]["cap"]
                        diag["cap_hits"][i] += int(infos[i]["cap"])
                    if tn > 0 and zero[i] == Z_NONE:
                        rec["ratio"][j, i] = qn / tn
                        rec["realized_ratio"][j, i] = qn / tn
                    elif tn == 0 and qn > 0:            # review A4: protection applied with no task gradient
                        rec["ratio"][j, i] = rec["realized_ratio"][j, i] = np.inf
                        diag["undefined_ratio_steps"][i] += 1
                    if zero[i] != Z_NONE and zero[i] != Z_OFF:
                        diag["zero_events"][i] += 1
                    rec["zero"][j, i] = zero[i]
                rec["comb_ratio"][j] = math.sqrt(tq2 / tt2) if tt2 > 0 else np.nan
                rec["rms_ratio"][j] = math.sqrt(float(np.mean(np.square(rec["realized_ratio"][j]))))
            u = torch.cat([t[:ne] + d_enc, t[ne:]])
            nrm = float(u.norm())
            with torch.no_grad():
                rec["pre_total"][j] = nrm
                for i, (lo, hi) in enumerate(blocks):
                    rec["pre_enc"][j, i] = float(u[lo:hi].double().norm())
                rec["pre_head"][j] = float(u[ne:].double().norm())
            if not math.isfinite(nrm):
                diag["nonfinite"] += 1
                continue
            kappa = 1.0
            if nrm > HP["clip"]:
                kappa = HP["clip"] / nrm
                u = u * kappa
                diag["clip_hits"] += 1
            with torch.no_grad():
                rec["kappa"][j] = kappa
                rec["post_total"][j] = float(u.double().norm())
                for i, (lo, hi) in enumerate(blocks):
                    rec["post_enc"][j, i] = float(u[lo:hi].double().norm())
            assign_add(enc_params + head_params, u, lr)
            rec["applied"][j] = True
            diag["encoder_updates"] += 1
        diag["epochs"].append(epoch_summary(rec, ep, per_ep))
        if log:
            log(diag["epochs"][-1])
    for p in model.parameters():
        p.requires_grad_(False)
    final["theta_T"] = copy.deepcopy(model.state_dict())
    diag["summary"] = run_summary(rec)
    diag["wall_s"] = time.time() - t0
    return model, diag, checkpoints, captures, final, rec


# ------------------------------------------------------------------ summaries
def _q(x, p):
    return float(np.percentile(x, p)) if len(x) else None


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "rms": float(np.sqrt(np.mean(x * x))), "median": _q(x, 50),
            "p10": _q(x, 10), "p90": _q(x, 90)}


def epoch_summary(rec, ep, per_ep):
    sl = slice(ep * per_ep, (ep + 1) * per_ep)
    out = {"epoch": ep, "clip": int((rec["kappa"][sl] < 1).sum()), "comb_ratio_mean": float(np.nanmean(rec["comb_ratio"][sl]))
           if np.isfinite(rec["comb_ratio"][sl]).any() else None, "rms_ratio_mean": float(rec["rms_ratio"][sl].mean())}
    for i in (0, 1):
        z = rec["zero"][sl, i]
        out[f"realized_ratio_mean_{i + 1}"] = float(rec["realized_ratio"][sl, i].mean())
        out[f"zero_{i + 1}"] = int(((z != Z_NONE) & (z != Z_OFF)).sum())
        out[f"cap_{i + 1}"] = int(rec["cap"][sl, i].sum())
        out[f"t_norm_mean_{i + 1}"] = float(rec["t_norm"][sl, i].mean())
        out[f"q_norm_mean_{i + 1}"] = float(rec["q_norm"][sl, i].mean())
        out[f"cos_mean_{i + 1}"] = float(np.nanmean(rec["cos"][sl, i])) if np.isfinite(rec["cos"][sl, i]).any() else None
    out["R_mean"] = [float(np.nanmean(rec["R"][sl, v])) if np.isfinite(rec["R"][sl, v]).any() else None for v in range(3)]
    return out


def run_summary(rec):
    """Realized statistics count zero steps as 0; 'conditional' statistics (nonzero steps only) are labelled as such."""
    ap = rec["applied"]
    out = {"steps": int(len(ap)), "applied": int(ap.sum()), "clip_fraction": float((rec["kappa"] < 1).mean()),
           "combined_ratio": _stats(rec["comb_ratio"][np.isfinite(rec["comb_ratio"])]),
           "rms_ratio": _stats(rec["rms_ratio"])}
    for i in (0, 1):
        z = rec["zero"][:, i]
        nz = rec["ratio"][:, i]
        out[f"enc{i + 1}"] = {"realized_ratio": _stats(rec["realized_ratio"][:, i]),
                              "conditional_ratio_nonzero_steps": _stats(nz[np.isfinite(nz)]),
                              "zero_fraction": float(((z != Z_NONE) & (z != Z_OFF)).mean()),
                              "zero_reasons": {str(r): int((z == r).sum()) for r in range(6) if (z == r).any()},
                              "cap_fraction": float(rec["cap"][:, i].mean()),
                              "cos": _stats(rec["cos"][:, i][np.isfinite(rec["cos"][:, i])]),
                              "t_norm": _stats(rec["t_norm"][:, i]), "q_norm": _stats(rec["q_norm"][:, i])}
    return out


# ------------------------------------------------------------------ frozen-minibatch equivalence (algebra check)
EQUIV_RTOL = 1e-4      # declared float32 tolerance: ||q_norm - q_raw||_2 <= 1e-4 ||q_raw||_2 per encoder (review A1)


def frozen_equivalence(model_state, data, seed, head, treat, beta, batch_idx, rtol=EQUIV_RTOL, rho_override=None,
                       a_max=None, snapshot=None):
    """On one frozen minibatch (critics/transforms from `snapshot`, else fresh init critics and the transform of the
    reference rows): RAW q_i = beta p_i (rgj operations) versus the NORM expression with rho_i := r_i =
    ||beta p_i|| / ||t_i||. Per encoder: 'equivalent' iff t_i, p_i nonzero and ||q_norm - q_raw|| <= rtol ||q_raw||;
    an encoder with p_i = 0 is 'not_applicable' (both expressions give 0). rho_override (a common rho for both
    encoders) and a_max exercise the registered failure cases. Pure function: no RNG."""
    model = Model(D_IN, KS, seed)
    model.load_state_dict(model_state)
    enc_params = [p for i in (0, 1) for p in model.enc[i].parameters()]
    head_params = [p for i in (0, 1) for p in model.head[i].parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in enc_params + head_params:
        p.requires_grad_(True)
    n0 = sum(p.numel() for p in model.enc[0].parameters())
    ne = sum(p.numel() for p in enc_params)
    H = entropy(data.prior)
    logprior = torch.tensor(np.log(data.prior), dtype=torch.float32)
    banks = {v: {k: new_critic(seed, v, k, "init") for k in KINDS} for v in VIEWS}
    if snapshot is not None:
        for v in VIEWS:
            for k in KINDS:
                banks[v][k].load_state_dict(snapshot["critics"][v][k])
        T = {v: Transform(state=snapshot["transforms"][v]) for v in VIEWS}
    else:
        Vref = critic_views(model, data.X[torch.from_numpy(data.ref)], list(VIEWS), head)
        T = {v: Transform(Vref[v], "floored") for v in VIEWS}
    b = torch.from_numpy(np.asarray(batch_idx))
    Lt = task_losses(model, data.X[b], {i: data.Y[i][b] for i in (0, 1)}, [None, None], [0, 1])
    gt = flat_grad(sum(Lt.values()), enc_params + head_params)

    def proxy(cf):
        active = [v for v in VIEWS if cf[v] > 0]
        V = critic_views(model, data.X[b], active, head, grad=True)
        P = 0.0
        for v in active:
            Rv, _, _ = recovery([banks[v][k](T[v](V[v])) for k in KINDS], data.S[b], logprior, H)
            P = P + cf[v] * Rv
        return flat_grad(P, enc_params) if isinstance(P, torch.Tensor) and P.requires_grad else torch.zeros(ne)

    q_raw = proxy(RT.coefficients(TREAT[treat], beta, (0.0, 0.0)))
    p_unit = proxy(proxy_coefficients({"mode": "NORM", "treat": treat}))
    out = {"equivalent": True, "applicable": 0, "encoders": []}
    for i, (lo, hi) in enumerate(((0, n0), (n0, ne))):
        tn = float(gt[lo:hi].double().norm())
        qn = float(q_raw[lo:hi].double().norm())
        r_i = qn / tn if tn > 0 else float("nan")
        rho_i = r_i if rho_override is None else rho_override
        qi, info = normalized_direction(gt[lo:hi], p_unit[lo:hi], rho_i if math.isfinite(rho_i) else 0.0,
                                        a_max=HP["A_MAX"] if a_max is None else a_max, zero_tol=HP["ZERO_TOL"])
        err = float((qi.double() - q_raw[lo:hi].double()).norm())          # relative L2 (review A1)
        applicable = qn > 0
        ok = bool(err <= rtol * qn) if applicable else None
        out["encoders"].append({"t_norm": tn, "q_raw_norm": qn, "r_i": r_i, "rho_used": rho_i, "l2_err": err,
                                "rel_err": err / qn if qn > 0 else None, "cap": info["cap"], "zero": info["zero"],
                                "equivalent": ok if applicable else "not_applicable (p_i = 0)"})
        if applicable:
            out["applicable"] += 1
            out["equivalent"] &= ok
    out["equivalent"] = bool(out["equivalent"] and out["applicable"] > 0)
    return out
