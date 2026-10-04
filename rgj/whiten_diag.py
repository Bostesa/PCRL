"""Inner-only critic-input transform diagnostic (PROTOCOL.md section 8). Not used for selection.

Snapshots: the frozen L-R reference and the selected J-G checkpoint of each seed (training views; DEFENSE_FIT rows).
Per view and transform kind (floored = main arms; raw = centring only; capped = ZCA of C + 1e-4 max(ev) I):
  covariance spectrum, singular values and condition number of W;
  fresh-critic fit quality (best CRITIC_VAL CE of kinds A and B after a bounded refit) and its CALIB CE;
  mean per-row input-gradient norm |dCE/dV| of the fitted critic through the transform on CALIB rows (the size of the
  signal an encoder would receive from that critic).
Planted fixture (real frozen views): append a clue column 1e-6 * (2S - 1) (+ 1e-7 noise) to each view and refit each
transform kind; also a standardised logistic regression on the clue alone (a suitably scaled classifier). A transform
whose finite critic cannot detect the clue is reported as BLIND: blindness is an optimisation property, never evidence
of less sensitive information. Every transform keeps all coordinates; nothing is projected out of any release.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rgj import finalize as FN
from rgj import run as R
from rgj import train as T

KINDS_T = ("floored", "raw", "capped")


def fit_eval(V, data, k, tkind, tag):
    cf, cv, cal = (torch.from_numpy(a) for a in (data.cf, data.cv, data.cal))
    Tm = T.Transform(V[cf], tkind)
    out = {"spectrum": Tm.spectrum(), "kinds": {}}
    for kind in T.KINDS:
        torch.manual_seed(T._seed("rgj-whiten", k, tag, tkind, kind))
        c = T.critic(kind, V.shape[1])
        c, rc = T.fit_bounded(c, Tm(V[cf]), data.S[cf], Tm(V[cv]), data.S[cv], [k, T._seed(tag, tkind, kind), 31])
        Vc = V[cal].clone().requires_grad_(True)
        logits = c(Tm(Vc))
        ce = F.cross_entropy(logits, data.S[cal], reduction="sum")
        g = torch.autograd.grad(ce, Vc)[0]
        with torch.no_grad():
            p = torch.softmax(c(Tm(V[cal])), 1)[:, 1].numpy()
        out["kinds"][kind] = {"best_val_ce": rc["best_val_ce"], "epochs": rc["epochs"],
                              "calib_ce": T.ce_of(c, Tm(V[cal]), data.S[cal]),
                              "calib_auc": float(roc_auc_score(data.S[cal].numpy(), p)),
                              "input_grad_norm_mean": float(g.norm(dim=1).mean())}
    return out


def run_whiten(D):
    name = "whiten__diag"
    if R.done(name):
        return
    data = T.TData(D)
    sc = json.loads((R.RUN / "selection_C.json").read_text())
    res = {"snapshots": {}, "planted": {}}
    for k in R.SEEDS:
        units = {"L-R": R.selB()[str(k)]["L-R"]["unit"], "J-G": sc[str(k)]["arms"]["J-G"]["unit"]}
        for lab, unit in units.items():
            model = R.model_from(torch.load(R.U(unit) / "model.pt"), k)
            V = T.frozen_views(model, data.X, T.head_of(R.load_warm(k)))
            res["snapshots"][f"s{k}|{lab}|{unit}"] = {v: {t: fit_eval(V[v], data, k, t, f"{k}{lab}{v}") for t in KINDS_T}
                                                     for v in T.VIEWS}
            if lab == "J-G":
                rng = np.random.default_rng([k, 1006])
                S = data.S.numpy()
                for v in ("v1",):
                    clue = torch.from_numpy((1e-6 * (2 * S - 1) + 1e-7 * rng.normal(size=len(S))).astype(np.float32))
                    Vp = torch.cat([V[v], clue[:, None]], 1)
                    pr = {t: fit_eval(Vp, data, k, t, f"{k}planted{v}") for t in KINDS_T}
                    cf, cal = data.cf, data.cal
                    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                    lr.fit(clue.numpy()[cf, None].astype(np.float64), S[cf])
                    auc_lr = float(roc_auc_score(S[cal], lr.predict_proba(clue.numpy()[cal, None].astype(np.float64))[:, 1]))
                    base = res["snapshots"][f"s{k}|{lab}|{unit}"][v]
                    for t in KINDS_T:
                        best_auc = max(pr[t]["kinds"][kk]["calib_auc"] for kk in T.KINDS)
                        pr[t]["verdict"] = "DETECTS" if best_auc > 0.99 else "BLIND"
                        pr[t]["auc_gain_vs_unplanted"] = best_auc - max(base[t]["kinds"][kk]["calib_auc"] for kk in T.KINDS)
                    res["planted"][f"s{k}|{v}"] = {"transforms": pr, "standardised_LR_on_clue_auc": auc_lr,
                                                   "clue_amplitude": 1e-6}
    FN.save_unit(R.U(name), {}, res)
    R.event("unit complete", unit=name)
