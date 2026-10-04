"""Critic-gap diagnosis on frozen final training snapshots of PN and LN (inner roles only; no outer rows).

For each PN/LN unit and each critic view (v1, v2, and pair for PN):
  snapshot            = theta_{T-1}: the model state at the LAST CRITIC UPDATE (the online critics and their whitener
                        belong to this state; review R1: evaluating them at the final state theta_T through the floored
                        whitener amplifies tiny changes ~10^4 and is a whitening artefact, not critic weakness)
  training-time view  = [g_i(X), centred training-head logits] at theta_{T-1} (identity map), what the critics saw;
  fixed input map     = the SAVED final-step whitener (ZCA, fitted on the defense_train reference rows at theta_{T-1});
  metric              = SEX cross-entropy (natural log) on the SAME attacker_val rows for every predictor.
PRIMARY statistic (registered, review R2): mean over the bank's critics of CE(online critic j) - CE(fresh_def critic
  j), paired by critic index (same kind), on attacker_val at theta_{T-1}. Best-of-bank values and the sklearn slate
  are secondary (they choose among predictors on the evaluation rows and are optimistic).
Sensitivity rows: theta_T with a ZCA whitener REFITTED on the same defense_train reference rows (online critics and the
  same fresh-critic protocol), never the saved whitener applied to theta_T or to deployed views.
Predictors compared like for like:
  prior        fitting-prior constant (defense_train SEX rate)
  online       each saved online critic (best of the bank reported), i.e. the critic state at the end of training
  fresh_def    same architectures, freshly initialised, trained on the SAME defense_train rows of the frozen view
               (Adam 3e-3, batch 256, up to 3000 steps, early stopping on a fixed 20% split of defense_train)
  fresh_att    same, trained on attacker_fit rows instead (early stopping on 20% of attacker_fit)
  slate        the inner sklearn slate (LR / MLP / HGB) on the unwhitened training-time view, fit on attacker_fit
Also recorded: covariance spectrum and per-feature scales of the view (attacker_fit rows), and the inner slate's log
loss on the DEPLOYED view (refitted head). A large online-vs-fresh gap on the same frozen view means the information
was present but the online critics did not read it at the end of training; small gaps with losses near the prior
mean little readable information remained for these predictor families.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m pnx.critic_gap --lock <LOCK.json>
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jcv import audit as AU
from jcv import train as T
from pnx import run as R

torch.set_num_threads(1)


def ce(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def train_fresh(kind, dv, Z, y, seed):
    """Fresh critic on fixed (whitened) inputs; early stopping on a fixed 20% split of the training rows."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Z))
    k = len(Z) // 5
    va, tr = idx[:k], idx[k:]
    torch.manual_seed(seed)
    c = T.critic("A" if kind == "A" else "B", dv)
    opt = torch.optim.Adam(c.parameters(), lr=T.HP["critic_lr"])
    best, best_state, bad = np.inf, None, 0
    Zt, yt = torch.from_numpy(Z).float(), torch.from_numpy(y)
    for step in range(3000):
        b = torch.from_numpy(rng.choice(tr, T.HP["batch"], replace=False))
        loss = F.cross_entropy(c(Zt[b]), yt[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                v = float(F.cross_entropy(c(Zt[va]), yt[va]))
            if v < best - 1e-5:
                best, best_state, bad = v, {kk: x.clone() for kk, x in c.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 5:
                    break
    c.load_state_dict(best_state)
    return c


def proba_t(c, Z):
    with torch.no_grad():
        return torch.softmax(c(torch.from_numpy(Z).float()), 1).double().numpy()


def _views_at(state, D, which, k):
    model = T.Model(D["X"].shape[1], R.KS, k)
    model.load_state_dict(state)
    with torch.no_grad():
        return {w: x.double().numpy() for w, x in T.views(model, torch.from_numpy(D["X"]), [None, None], which,
                                                            frozen=(0, 1)).items()}


def _online_and_fresh(crit, w, Zall, S, tr, af, av, name, tag_prefix, fresh=True):
    dv = Zall.shape[1]
    r = {}
    online = []
    for kind, st in zip(crit["kinds"][w], crit["critics"][w]):
        c = T.critic("A" if kind == "A" else "B", dv)
        c.load_state_dict(st)
        online.append({"kind": kind, "ce": ce(proba_t(c, Zall[av]), S[av])})
    r["online"] = online
    r["online_best_ce"] = min(o["ce"] for o in online)
    if fresh:
        for tag, rows in (("fresh_def", tr), ("fresh_att", af)):
            fr = []
            for j, kind in enumerate(crit["kinds"][w]):
                c = train_fresh(kind, dv, Zall[rows], S[rows], T._seed("fresh", tag_prefix, tag, name, w, j))
                fr.append({"kind": kind, "ce": ce(proba_t(c, Zall[av]), S[av])})
            r[tag] = fr
            r[f"{tag}_best_ce"] = min(o["ce"] for o in fr)
        r["primary_mean_paired_online_minus_fresh_def"] = float(np.mean([o["ce"] - f["ce"] for o, f in zip(online, r["fresh_def"])]))
    return r


def diagnose(name, D):
    d = R.U(name)
    rec = json.loads((d / "record.json").read_text())
    k = rec["seed"]
    crit = torch.load(d / "critics_final.pt")
    S = D["sex"]
    tr, af, av = D["idx"]["defense_train"], D["idx"]["attacker_fit"], D["idx"]["attacker_val"]
    ref = tr[R.JR.tensors(D).guard_idx]           # the defense_train reference rows the training whiteners used
    prior = np.bincount(S[tr], minlength=2) / len(tr)
    which = list(crit["critics"])
    V = _views_at(crit["model_state_at_last_critic_update"], D, which, k)        # theta_{T-1}
    VT = _views_at(torch.load(d / "model.pt"), D, which, k)                      # theta_T (sensitivity)
    dep = R.release_views(name)
    out = {"unit": name, "arm": rec["arm"], "beta": rec["beta"], "seed": k, "snapshot": "theta_{T-1} (last critic update)",
           "views": {}}
    for w in which:
        wh = crit["whiteners"][w]
        mu, W = wh["mu"].double().numpy(), wh["W"].double().numpy()
        Zall = (V[w] - mu) @ W
        r = {"prior_ce": ce(np.tile(prior, (len(av), 1)), S[av])}
        r.update(_online_and_fresh(crit, w, Zall, S, tr, af, av, name, "Tm1"))
        # sensitivity: theta_T with a refitted ZCA whitener on the same reference rows (online critics only + fresh_def)
        wT = T.Whitener(torch.from_numpy(VT[w][ref]).float())
        ZT = (torch.from_numpy(VT[w]).float() - wT.mu) @ wT.W
        sens = _online_and_fresh(crit, w, ZT.double().numpy(), S, tr, af, av, name, "T")
        r["sensitivity_thetaT_refit_whitener"] = {kk: sens[kk] for kk in ("online_best_ce", "fresh_def_best_ce",
                                                                          "primary_mean_paired_online_minus_fresh_def")}
        sel = AU.fit_view(V[w][af], S[af], V[w][av], S[av], "inner")
        r["slate_selected"] = sel["selected"]
        r["slate_ce"] = sel["val_log_loss"]
        r["slate_auc"] = AU.macro_auc(S[av], sel["Pv"], [0, 1])
        dsel = AU.fit_view(dep[w][af], S[af], dep[w][av], S[av], "inner")
        r["deployed_view_slate_ce"] = dsel["val_log_loss"]
        r["deployed_view_slate_auc"] = AU.macro_auc(S[av], dsel["Pv"], [0, 1])
        ev = np.linalg.eigvalsh(np.cov(V[w][af].T))
        r["spectrum"] = {"eigenvalues_desc": [float(x) for x in ev[::-1]], "condition": float(ev.max() / max(ev.min(), 1e-300))}
        sd = V[w][af].std(0)
        r["feature_scale"] = {"min": float(sd.min()), "median": float(np.median(sd)), "max": float(sd.max())}
        H = r["prior_ce"]
        r["R_like"] = {kk: 1 - r[kk] / H for kk in ("online_best_ce", "fresh_def_best_ce", "fresh_att_best_ce", "slate_ce")}
        out["views"][w] = r
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args(argv)
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    from pnx.lock import verify_lock
    v = verify_lock(Path(a.lock))
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:5]))
    D = R.load_D()
    for k in a.seeds:
        for arm in R.NEW_ARMS:
            for b in R.BETAS:
                nm = R.unit_name(k, arm, b)
                cname = f"critic__{nm}"
                if R.done(cname) or not R.done(nm):
                    continue
                t0 = time.time()
                res = diagnose(nm, D)
                res["wall_s"] = time.time() - t0
                R.FN.save_unit(R.U(cname), {}, res)
                R.event("unit complete", unit=cname)


if __name__ == "__main__":
    main()
