#!/usr/bin/env python
"""Independent verifier replay for results/pcrl_penalty_no_erasure_v1 (focused no-erasure penalty study, "pnx").

Every check is rebuilt from the raw saved arrays and manifests (private inputs npz, unit files, records, the locks).
The study code is never imported: a sys.meta_path guard refuses pnx, jcv, oar, odx, cap, stored_model_eval, report and
pcrl, and the script asserts at the end that none of them was loaded (also not through a pickle). Only numpy, scipy,
scikit-learn, torch, joblib and the standard library are used. Nothing is trained; the only fits are (a) refits of
already-selected attacker configurations, (b) the inner slate on two decisive releases, (c) one shuffled/planted control
and (d) the deployed-head C grid, all to replay recorded numbers. One task-only loop (seed 0, beta = 0) and one final
penalty step per PN/LN unit are replayed from saved states to check the update rule; neither result is kept.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/pcrl_penalty_no_erasure_v1/verification/replay_pnx.py

Writes results/pcrl_penalty_no_erasure_v1/INDEPENDENT_VERIFICATION.json (aggregates only; no per-person values, no
absolute paths).
"""
from __future__ import annotations

import importlib.abc
import sys

FORBIDDEN = ("pnx", "jcv", "oar", "odx", "cap", "stored_model_eval", "report", "pcrl")
BLOCKED = []


class _Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in FORBIDDEN:
            BLOCKED.append(name)
            raise ImportError(f"verifier guard: import of study module '{name}' is forbidden")
        return None


sys.meta_path.insert(0, _Guard())
assert not [m for m in sys.modules if m.split(".")[0] in FORBIDDEN], "a forbidden module was loaded before the guard"

import csv  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss as sk_log_loss  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

HOME = Path.home()
WT = Path(__file__).resolve().parents[3]
PKG = WT / "results" / "pcrl_penalty_no_erasure_v1"
PRIV = HOME / "PCRL_eval_cache_private"
INPUTS = PRIV / "jcv_v1" / "inputs" / "adult_jcv.npz"
RUN = PRIV / "pnx_v1" / "run"
NEW = RUN / "units"
PRED = PRIV / "jcv_v1" / "run" / "units"
OUT = PKG / "INDEPENDENT_VERIFICATION.json"
INPUT_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
SEL_COMMIT = "b0b9256e3a3ffbd02324fcd7609c88c3b4dc9cf4"
BRANCH = "research/pcrl-penalty-no-erasure-v1"
SEEDS = (0, 1, 2)
BETAS = (0.1, 1.0, 10.0)
BSTR = ("0.1", "1", "10")
LABELS = ["U", "E", "F", "F0"] + [f"{a}_b{b}" for a in ("JP", "PN", "LN") for b in BSTR]
KS = (2, 6)
VIEW_KEY = {("prim", "v1"): "P_v1", ("prim", "v2"): "P_v2", ("prim", "pair"): "P_pair",
            ("prob", "v1"): "P_p1", ("prob", "v2"): "P_p2", ("prob", "pair"): "P_ppair",
            ("hard", "v1"): "P_h1", ("hard", "v2"): "P_h2", ("hard", "pair"): "P_hpair"}
TOL_PT = 1e-10          # recomputed point / SE vs runner JSON (same draws; only summation order differs)
TOL_CSV = 6e-7          # vs 6-decimal CSV
CHECKS = []


def add(cid, status, detail, max_abs_diff=None, **extra):
    e = {"id": cid, "status": status, "detail": detail}
    if max_abs_diff is not None:
        e["max_abs_diff"] = float(max_abs_diff)
    e.update(extra)
    CHECKS.append(e)
    tail = "" if max_abs_diff is None else f" [max_abs_diff={float(max_abs_diff):.3g}]"
    print(f"[{status}] {cid}: {detail}{tail}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rjson(p):
    return json.loads(Path(p).read_text())


def udir(name):
    d = NEW / name
    a = d / "ALIAS.json"
    if a.exists():
        src = rjson(a)["source"]
        assert src.startswith("~/"), "alias source is not home-relative"
        return HOME / src[2:]
    return d


def complete_ok(d):
    c = d / "COMPLETE.json"
    if not c.exists():
        return False, ["no COMPLETE.json"], 0
    files = rjson(c)["files"]
    bad = [f for f, h in files.items() if not (d / f).exists() or sha(d / f) != h]
    on_disk = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()} - {"COMPLETE.json"}
    extra = sorted(on_disk - set(files))
    return (not bad and not extra), bad + [f"unlisted:{x}" for x in extra], len(files)


# =============================================================================== numerics
def wauc_matrix(score, pos, W, chunk=400):
    """Weighted Mann-Whitney AUC (ties count 1/2) of score for pos vs not pos, one value per weight column."""
    score = np.asarray(score, np.float64)
    pos = np.asarray(pos, bool)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    starts = np.flatnonzero(np.concatenate([[True], s[1:] != s[:-1]]))
    pp = pos[order][:, None]
    out = np.empty(W.shape[1])
    for c0 in range(0, W.shape[1], chunk):
        Wo = W[order, c0:c0 + chunk]
        Wp = np.where(pp, Wo, 0.0)
        Wn = np.where(pp, 0.0, Wo)
        gp = np.add.reduceat(Wp, starts, axis=0)
        gn = np.add.reduceat(Wn, starts, axis=0)
        below = np.cumsum(gn, 0) - gn
        num = (gp * (below + 0.5 * gn)).sum(0)
        den = gp.sum(0) * gn.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[c0:c0 + chunk] = np.where(den > 0, num / den, np.nan)
    return out


def auc_plain(y_pos, score):
    return float(wauc_matrix(score, y_pos, np.ones((len(score), 1)))[0])


def logloss(y, P):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


# =============================================================================== attackers (own implementation)
class DAttack(BaseEstimator, ClassifierMixin):
    """Defense-aware attacker: PCA whitening fitted on the training rows, drop variance <= 1e-9 x max, MLP(128,128)."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        self.pca_ = PCA(whiten=True, random_state=self.seed).fit(X)
        ev = self.pca_.explained_variance_
        self.keep_ = ev > 1e-9 * ev.max()
        self.m_ = MLPClassifier(hidden_layer_sizes=(128, 128), alpha=1e-4, max_iter=300, early_stopping=True,
                                validation_fraction=0.1, n_iter_no_change=15, random_state=self.seed)
        self.m_.fit(self.pca_.transform(X)[:, self.keep_], y)
        self.classes_ = self.m_.classes_
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(self.pca_.transform(X)[:, self.keep_])


class CellCond:
    def __init__(self, alpha, K=2):
        self.alpha, self.K = alpha, K

    def fit(self, X, y):
        self.K = max(self.K, int(np.max(y)) + 1)
        self.prior_ = np.bincount(y, minlength=self.K) / len(y)
        self.t_ = {}
        for r, s in zip(np.ascontiguousarray(np.round(X, 9)), y):
            self.t_.setdefault(r.tobytes(), np.zeros(self.K))[s] += 1
        self.classes_ = np.arange(self.K)
        return self

    def predict_proba(self, X):
        out = np.empty((len(X), self.K))
        for i, r in enumerate(np.ascontiguousarray(np.round(X, 9))):
            c = self.t_.get(r.tobytes())
            out[i] = self.prior_ if c is None else (c + self.alpha * self.prior_) / (c.sum() + self.alpha)
        return out


def F_LR(C):
    return lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def F_MLP(h):
    return lambda s: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=h, alpha=1e-4, max_iter=300,
                                                                   early_stopping=True, validation_fraction=0.1,
                                                                   n_iter_no_change=15, random_state=s))


def F_HGB(lr, lv):
    return lambda s: HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=lv, max_iter=200,
                                                    early_stopping=False, random_state=s)


def slate_final(finite):
    out = [(f"LR_C{C}", F_LR(C)) for C in (0.01, 0.1, 1.0, 10.0, 100.0)]
    out += [("MLP_" + "x".join(map(str, h)), F_MLP(h)) for h in ((64,), (128,), (64, 64), (128, 128))]
    out += [(f"HGB_{lr}_{lv}", F_HGB(lr, lv)) for lr in (0.05, 0.1) for lv in (15, 31)]
    out += [("DA_canonical_MLP", lambda s: DAttack(s))]
    if finite:
        out += [(f"CC_alpha{a}", (lambda s, a=a: CellCond(a))) for a in (0.1, 1.0, 10.0)]
    return out


def slate_inner():
    return [("LR_C1", F_LR(1.0)), ("MLP_64x64", F_MLP((64, 64))), ("HGB_0.1_31", F_HGB(0.1, 31))]


def slate_secondary(finite):
    out = slate_inner() + [("DA_canonical_MLP", lambda s: DAttack(s))]
    if finite:
        out += [(f"CC_alpha{a}", (lambda s, a=a: CellCond(a))) for a in (0.1, 1.0, 10.0)]
    return out


def proba(m, X, K):
    P = m.predict_proba(X)
    out = np.zeros((len(X), K))
    for j, c in enumerate(getattr(m, "classes_", range(P.shape[1]))):
        out[:, int(c)] = P[:, j]
    return out


def fit_select(Xf, yf, Xv, yv, slate, K):
    best, table = None, []
    for name, fac in slate:
        m = fac(0).fit(Xf, yf)
        Pv = proba(m, Xv, K)
        ll = logloss(yv, Pv)
        table.append({"attacker": name, "attacker_val_log_loss": ll})
        if best is None or ll < best[0] - 1e-12:
            best = (ll, name, fac, m, Pv)
    return {"val_log_loss": best[0], "selected": best[1], "factory": best[2], "model0": best[3], "Pv": best[4],
            "table": table}


# =============================================================================== torch models (own definitions)
def _enc():
    return nn.Sequential(nn.Linear(83, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([_enc(), _enc()])
        self.head = nn.ModuleList([nn.Linear(16, 2), nn.Linear(16, 6)])


def load_net(state):
    m = Net()
    m.load_state_dict(state, strict=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def critic_net(kind, dv):
    if kind == "A":
        return nn.Sequential(nn.Linear(dv, 32), nn.ReLU(), nn.Linear(32, 2))
    return nn.Sequential(nn.Linear(dv, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))


def train_views(m, X, which):
    """[g_i(X), centred training-head logits] with the training head detached (identity map); pair = [v1, v2]."""
    v = {}
    for i in (0, 1):
        if f"v{i + 1}" in which or "pair" in which:
            h = m.enc[i](X)
            lg = F.linear(h, m.head[i].weight.detach(), m.head[i].bias.detach())
            v[i] = torch.cat([h, lg - lg.mean(1, keepdim=True)], 1)
    return {w: (v[0] if w == "v1" else v[1] if w == "v2" else torch.cat([v[0], v[1]], 1)) for w in which}


def flat_grad(loss, params):
    gs = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])


def add_vec(params, vec, scale):
    with torch.no_grad():
        o = 0
        for p in params:
            n = p.numel()
            p.add_(vec[o:o + n].view_as(p), alpha=scale)
            o += n


# =============================================================================== shared state
class Ctx:
    pass


C = Ctx()


def unit_label(u):
    if isinstance(u, list):
        raise ValueError("pair units have no single label")
    p = u.split("__")
    if p[0] == "nn" and p[-1] in ("U", "E"):
        return p[-1]
    assert p[3].startswith("b")
    return f"{p[2]}_b{p[3][1:]}"


# =============================================================================== 1. inputs, roles, aliases
def check_inputs():
    h = sha(INPUTS)
    z = np.load(INPUTS, allow_pickle=False)
    D = {k: z[k] for k in z.files}
    D["idx"] = {r: np.flatnonzero(D["role"] == r) for r in np.unique(D["role"])}
    C.D = D
    keys = ["row_id", "unit", "role", "X", "feature_names", "sex", "race", "y_income", "y_occupation_group"]
    ok = h == INPUT_SHA and sorted(z.files) == sorted(keys) and D["X"].shape[1] == 83 and D["X"].dtype == np.float32
    add("01-input-hash", "PASS" if ok else "FAIL",
        f"sha256 {'matches' if h == INPUT_SHA else 'DIFFERS from'} the admitted hash; keys {sorted(z.files)}; X {D['X'].shape} {D['X'].dtype}")
    SL = C.SL
    bad, sizes = [], {}
    for r, hh in SL["role_row_hashes"].items():
        ix = D["idx"].get(r, np.array([], int))
        mine = hashlib.sha256(np.sort(D["row_id"][ix]).astype(np.int64).tobytes()).hexdigest()
        sizes[r] = int(len(ix))
        if mine != hh:
            bad.append(r)
    allrows = np.concatenate(list(D["idx"].values()))
    disjoint = len(allrows) == len(np.unique(allrows)) == len(D["row_id"])
    fit_roles = ("defense_train", "defense_val", "attacker_fit", "attacker_val")
    a_units = set(D["unit"][D["idx"]["assessment"]].tolist())
    shared = {r: len(a_units & set(D["unit"][D["idx"][r]].tolist())) for r in fit_roles}
    add("02-roles", "PASS" if not bad and disjoint else "FAIL",
        f"8 role row-id hashes vs SELECTION_LOCK: {8 - len(bad)}/8 match{(' (bad: ' + ','.join(bad) + ')') if bad else ''}; "
        f"roles partition all {len(D['row_id'])} rows: {disjoint}; sizes {sizes}; assessment groups = rows "
        f"({len(a_units)} groups); record groups shared between assessment and fitting roles: {shared}")
    # alias integrity
    n_alias, problems = 0, []
    for d in sorted(NEW.iterdir()):
        a = d / "ALIAS.json"
        if not a.exists():
            continue
        n_alias += 1
        rec = rjson(a)
        own_ok, own_bad, _ = complete_ok(d)
        if not own_ok:
            problems.append(f"{d.name}: alias COMPLETE {own_bad}")
        src = udir(d.name)
        if not src.is_dir():
            problems.append(f"{d.name}: source missing")
            continue
        if sha(src / "COMPLETE.json") != rec["source_COMPLETE_sha256"]:
            problems.append(f"{d.name}: source COMPLETE sha mismatch")
        sok, sbad, _ = complete_ok(src)
        if not sok:
            problems.append(f"{d.name}: source files {sbad[:3]}")
        if src.name != d.name:
            problems.append(f"{d.name}: source name {src.name}")
    C.n_alias = n_alias
    add("03-alias-integrity", "PASS" if not problems and n_alias else "FAIL",
        f"{n_alias} alias records: ALIAS.json hashed in the alias COMPLETE.json, source COMPLETE.json sha256 equals the "
        f"recorded source_COMPLETE_sha256, and every source file hash listed there verifies; same unit name; problems: "
        f"{problems[:5] or 'none'}")


# =============================================================================== 2. identity-map deployment replay
def pn_units():
    return [f"pn__s{k}__{a}__b{b}" for k in SEEDS for a in ("PN", "LN") for b in BSTR]


def check_deployment():
    D = C.D
    X = torch.from_numpy(D["X"])
    tr, va = D["idx"]["defense_train"], D["idx"]["defense_val"]
    mx = {k: 0.0 for k in ("r", "c", "p")}
    hard_bad, struct, f64 = [], [], 0.0
    head_C_bad, head_coef = [], 0.0
    head_tab = 0.0
    for nm in pn_units():
        d = NEW / nm
        if (d / "ALIAS.json").exists() or any(d.glob("leace_*")):
            struct.append(f"{nm}: alias or leace_* present")
        rec = rjson(d / "record.json")
        if rec.get("erasure") is not False or rec.get("maps") != "identity" or rec["finalize"].get("leace") not in ({}, None):
            struct.append(f"{nm}: record says erasure/maps {rec.get('erasure')}/{rec.get('maps')}")
        m = load_net(torch.load(d / "model.pt", weights_only=True))
        z = np.load(d / "release.npz")
        if not np.array_equal(z["row_id"], D["row_id"]):
            struct.append(f"{nm}: row_id order")
        with torch.no_grad():
            Hs = [m.enc[i](X).double().numpy() for i in (0, 1)]
        for i in (0, 1):
            # float64 forward as an independent arithmetic check of the same weights
            st = {k: v.double().numpy() for k, v in m.enc[i].state_dict().items()}
            h64 = np.maximum(D["X"].astype(np.float64) @ st["0.weight"].T + st["0.bias"], 0)
            h64 = np.maximum(h64 @ st["2.weight"].T + st["2.bias"], 0)
            h64 = h64 @ st["4.weight"].T + st["4.bias"]
            f64 = max(f64, float(np.abs(h64 - Hs[i]).max()))
            head = joblib.load(d / f"head_{i}.joblib")
            steps = [s[0] for s in head.steps]
            lr = head.steps[-1][1]
            if steps != ["standardscaler", "logisticregression"] or head.n_features_in_ != 16 or \
                    head.steps[0][1].n_features_in_ != 16 or lr.coef_.shape[1] != 16:
                struct.append(f"{nm}/head_{i}: structure {steps} n_features_in {head.n_features_in_}")
            Z = np.asarray(head.decision_function(Hs[i]), np.float64)
            if Z.ndim == 1:
                Z = np.stack([np.zeros_like(Z), Z], 1)
            cen = Z - Z.mean(1, keepdims=True)
            P = head.predict_proba(Hs[i])
            hard = P.argmax(1)
            mx["r"] = max(mx["r"], float(np.abs(Hs[i] - z[f"r{i + 1}"]).max()))
            mx["c"] = max(mx["c"], float(np.abs(cen - z[f"c{i + 1}"]).max()))
            mx["p"] = max(mx["p"], float(np.abs(P - z[f"p{i + 1}"]).max()))
            if not np.array_equal(hard, z[f"hard{i + 1}"]):
                hard_bad.append(f"{nm}/{i}")
            # deployed-head rule: StandardScaler + LR, C grid fitted on defense_train, chosen by defense_val log loss
            y = D["y_income"] if i == 0 else D["y_occupation_group"]
            best, tab = None, []
            for Cc in (0.01, 0.1, 1.0, 10.0, 100.0):
                hm = make_pipeline(StandardScaler(), LogisticRegression(C=Cc, max_iter=3000)).fit(Hs[i][tr], y[tr])
                ll = sk_log_loss(y[va], hm.predict_proba(Hs[i][va]), labels=list(range(KS[i])))
                tab.append(ll)
                if best is None or ll < best[0] - 1e-12:
                    best = (ll, Cc, hm)
            rh = rec["finalize"]["heads"][str(i)]
            head_tab = max(head_tab, max(abs(a - b["defense_val_log_loss"]) for a, b in zip(tab, rh["table"])))
            if best[1] != rh["selected_C"] or best[1] != lr.C:
                head_C_bad.append(f"{nm}/{i}: refit C {best[1]} vs record {rh['selected_C']} vs saved {lr.C}")
            head_coef = max(head_coef, float(np.abs(best[2].steps[-1][1].coef_ - lr.coef_).max()))
    worst = max(mx.values())
    ok = not hard_bad and not struct and worst <= 1e-12
    add("04-deployment-identity-replay", "PASS" if ok else "FAIL",
        f"18 PN/LN units: r_i = g_i(X) (own float32 torch forward, then float64), centred decision-function logits "
        f"(binary -> (0, d)), predict_proba, argmax vs release.npz: max |diff| r {mx['r']:.3g}, c {mx['c']:.3g}, "
        f"p {mx['p']:.3g}; hard decisions identical on all rows: {not hard_bad}; float64 re-forward of the same weights "
        f"differs from the float32 release by at most {f64:.3g} (rounding only)", max_abs_diff=worst)
    add("05-no-map-heads-16", "PASS" if not struct else "FAIL",
        f"no leace_* directory, not an alias, record erasure False / maps identity / no LEACE meta, heads = "
        f"StandardScaler+LR taking exactly 16 inputs (r_i only): {'all 18 units' if not struct else struct[:4]}")
    add("06-deployed-head-refit", "PASS" if not head_C_bad and head_coef < 1e-6 else "FAIL",
        f"head rule replayed on defense_train/defense_val for 36 heads: selected C equal to record and saved head "
        f"({'all' if not head_C_bad else head_C_bad[:3]}); defense_val log-loss table max |diff| {head_tab:.3g}; "
        f"refit coefficient max |diff| {head_coef:.3g}", max_abs_diff=max(head_tab, head_coef))


# =============================================================================== 3. update rule / beta = 0 parity
def tr_tensors():
    D = C.D
    tr = D["idx"]["defense_train"]
    S = D["sex"][tr]
    return (torch.from_numpy(D["X"][tr]), {0: torch.from_numpy(D["y_income"][tr]), 1: torch.from_numpy(D["y_occupation_group"][tr])},
            torch.from_numpy(S), np.bincount(S, minlength=2) / len(S))


def check_last_step():
    """theta_T = theta_{T-1} + lr * clip(-grad(L1+L2) - beta grad P), P from the saved online critics and whitener."""
    Xt, Y, S, prior = tr_tensors()
    n = Xt.shape[0]
    H = float(-(prior * np.log(prior)).sum())
    logprior = torch.tensor(np.log(prior), dtype=torch.float32)
    worst, nonbit, info = 0.0, [], []
    for nm in pn_units():
        d = NEW / nm
        rec = rjson(d / "record.json")
        k, arm, beta = rec["seed"], rec["arm"], float(rec["beta"])
        crit = torch.load(d / "critics_final.pt", weights_only=True)
        final = torch.load(d / "model.pt", weights_only=True)
        m = load_net(crit["model_state_at_last_critic_update"])
        enc_p = [p for i in (0, 1) for p in m.enc[i].parameters()]
        head_p = [p for i in (0, 1) for p in m.head[i].parameters()]
        for p in enc_p + head_p:
            p.requires_grad_(True)
        perm = np.random.default_rng([k, 0, 19]).permutation(n)
        last = list(range(0, n, 256))[-1]
        b = torch.from_numpy(perm[last:last + 256])
        Lt = {i: F.cross_entropy(m.head[i](m.enc[i](Xt[b])), Y[i][b]) for i in (0, 1)}
        t = -flat_grad(sum(Lt.values()), enc_p + head_p)
        ne = sum(p.numel() for p in enc_p)
        which = ["v1", "v2", "pair"] if arm == "PN" else ["v1", "v2"]
        expect_kinds = {"v1": ["A", "B"], "v2": ["A", "B"], "pair": ["A", "B"]} if arm == "PN" else \
            {"v1": ["A", "B", "B2"], "v2": ["A", "B", "B2"]}
        if crit["kinds"] != expect_kinds:
            nonbit.append(f"{nm}: critic kinds {crit['kinds']}")
        V = train_views(m, Xt[b], which)
        Pobj, chosen = 0.0, {}
        for v in which:
            wh = crit["whiteners"][v]
            cs = []
            for kind, st in zip(crit["kinds"][v], crit["critics"][v]):
                c = critic_net(kind, V[v].shape[1])
                c.load_state_dict(st)
                cs.append(F.cross_entropy(c((V[v] - wh["mu"]) @ wh["W"]), S[b]))
            allc = cs + [F.nll_loss(logprior.expand(len(b), 2), S[b])]
            j = int(np.argmin([float(x) for x in allc]))
            chosen[v] = (crit["kinds"][v] + ["const"])[j]
            Pobj = Pobj + (1.0 - allc[j] / H)
        pvec = -beta * flat_grad(Pobj, enc_p) if isinstance(Pobj, torch.Tensor) and Pobj.requires_grad else torch.zeros(ne)
        u = torch.cat([t[:ne] + pvec, t[ne:]])
        nrm = float(u.norm())
        if nrm > 5.0:
            u = u * (5.0 / nrm)
        add_vec(enc_p + head_p, u.detach(), 0.05)
        st = m.state_dict()
        diff = max(float((st[kk] - final[kk]).abs().max()) for kk in final)
        bit = all(torch.equal(st[kk], final[kk]) for kk in final) and set(st) == set(final)
        # the counterfactual "no penalty" step must NOT reproduce theta_T (the penalty actually acted)
        m0 = load_net(crit["model_state_at_last_critic_update"])
        u0 = torch.cat([t[:ne], t[ne:]])
        n0 = float(u0.norm())
        u0 = u0 * (5.0 / n0) if n0 > 5.0 else u0
        add_vec([p for i in (0, 1) for p in m0.enc[i].parameters()] + [p for i in (0, 1) for p in m0.head[i].parameters()],
                u0.detach(), 0.05)
        d0 = max(float((m0.state_dict()[kk] - final[kk]).abs().max()) for kk in final)
        worst = max(worst, diff)
        if not bit:
            nonbit.append(f"{nm}: max {diff:.3g}")
        info.append({"unit": nm, "bitwise": bit, "max_abs_diff": diff, "task_only_step_max_abs_diff": d0,
                     "selected_surrogates": chosen, "clipped": nrm > 5.0})
    C.last_step = info
    pen_matters = sum(1 for x in info if x["task_only_step_max_abs_diff"] > 0)
    add("07-last-step-update-rule", "PASS" if not nonbit else ("WARN" if worst < 1e-6 else "FAIL"),
        f"for all 18 PN/LN units the final encoder/head step was replayed from the saved theta_(T-1), saved online "
        f"critics and saved ZCA whitener on the last minibatch of epoch 19 (identity maps, P = R_v1+R_v2(+R_pair for PN), "
        f"u = -grad(L1+L2) - beta grad P, clip 5, SGD 0.05): {18 - len(nonbit)}/18 reproduce model.pt bitwise; critic "
        f"banks PN {{A,B}}x3 / LN {{A,B,B2}}x2 as registered; the task-only counterfactual step differs from model.pt on "
        f"{pen_matters}/18 units (the penalty acted); {nonbit[:3] or ''}", max_abs_diff=worst)


def check_u_replication():
    """Task-only (beta = 0) protection phase, seed 0, from the hashed warm start; vs the predecessor U model."""
    Xt, Y, S, prior = tr_tensors()
    n = Xt.shape[0]
    k = 0
    warm = torch.load(udir(f"warm__s{k}") / "warm.pt", weights_only=True)
    m = Net()
    m.load_state_dict(warm, strict=True)
    enc_p = [p for i in (0, 1) for p in m.enc[i].parameters()]
    head_p = [p for i in (0, 1) for p in m.head[i].parameters()]
    params = enc_p + head_p
    ne = sum(p.numel() for p in enc_p)
    clipped = 0
    t0 = time.time()
    for ep in range(20):
        perm = np.random.default_rng([k, 0, ep]).permutation(n)
        for s in range(0, n, 256):
            b = torch.from_numpy(perm[s:s + 256])
            L = {i: F.cross_entropy(m.head[i](m.enc[i](Xt[b])), Y[i][b]) for i in (0, 1)}
            t = -flat_grad(sum(L.values()), params)
            u = torch.cat([t[:ne] + torch.zeros(ne), t[ne:]])
            nrm = float(u.norm())
            if nrm > 5.0:
                u = u * (5.0 / nrm)
                clipped += 1
            add_vec(params, u.detach(), 0.05)
    ref = torch.load(PRED / f"nn__s{k}__U" / "model.pt", weights_only=True)
    st = m.state_dict()
    bit = all(torch.equal(st[kk], ref[kk]) for kk in ref) and set(st) == set(ref)
    diff = max(float((st[kk] - ref[kk]).abs().max()) for kk in ref)
    add("08-beta0-task-only-loop-replication", "PASS" if bit else ("WARN" if diff < 1e-5 else "FAIL"),
        f"own re-implementation of the beta = 0 path (no penalty; critics, if any, never touch the encoders) for seed 0: "
        f"20 epochs x 76 SGD steps from the hashed warm start, perm = default_rng([seed, 0, epoch]), clip 5 "
        f"({clipped} clipped steps), lr 0.05 -> predecessor nn__s0__U model.pt {'BITWISE equal' if bit else 'differs'} "
        f"({time.time() - t0:.0f}s)", max_abs_diff=diff)


def check_parity_receipts():
    files = sorted(RUN.glob("PARITY_*.json"))
    recs = [rjson(p) for p in files]
    unit = NEW / "parity__0_1_2"
    uok, ubad, _ = complete_ok(unit)
    urec = rjson(unit / "record.json")
    same = json.dumps(urec, sort_keys=True) == json.dumps(recs[-1], sort_keys=True) if recs else False
    probs = []
    for r in recs:
        for c in r["checks"]:
            if not c["pass"]:
                probs.append(c["check"])
            if "release_max_abs_diff" in c and (max(c["release_max_abs_diff"].values()) > 1e-12 or not c["hard_equal"]
                                                 or not c["identity_maps"] or not c["model_bitwise"]):
                probs.append(f"{c['check']} seed {c.get('seed')}: claimed values violate the tolerance")
            if c["check"].endswith("== U") and c.get("protection_steps") != 0:
                probs.append(f"{c['check']}: protection steps {c.get('protection_steps')} at beta 0")
    kinds = sorted({c["check"] for r in recs for c in r["checks"]})
    n_beta0 = sum(1 for r in recs for c in r["checks"] if c["check"].endswith("== U"))
    # model files: the predecessor U equals what the U alias points at
    alias_same = all(sha(udir(f"nn__s{k}__U") / "model.pt") == sha(PRED / f"nn__s{k}__U" / "model.pt") for k in SEEDS)
    ok = recs and all(r["all_pass"] for r in recs) and not probs and uok and same and n_beta0 == 6 and alias_same
    add("09-beta0-parity-receipts", "PASS" if ok else "FAIL",
        f"{len(files)} PARITY receipt(s), unit parity__0_1_2 hash-complete={uok} and identical to the receipt={same}; "
        f"checks {kinds}: all pass={all(r['all_pass'] for r in recs) if recs else False}; 6 PN/LN beta=0 claims report "
        f"model bitwise, release max diff 0, hard equal, no map, 0 protection steps; problems {probs or 'none'}. "
        f"The beta=0 PN/LN models themselves were not kept (aliases of U), so they cannot be re-hashed; check 08 "
        f"independently reproduces U from the warm start and check 07 the penalty step.")
    add("09b-parity-jp-replication-claim", "INFO",
        "the receipt's 'copied loop reproduces predecessor JP beta=1 seed 0 bitwise' cannot be replayed here without the "
        "study's official LEACE wrapper (forbidden import); recorded as reported (model_bitwise True)")


# =============================================================================== 4. selection replay
def inner_rec(name):
    return rjson(udir(f"inner__{name}") / "record.json")


def gates(util, uU):
    ok, margins = True, []
    for i in (0, 1):
        a, aU, c = util[i]["acc"], uU[i]["acc"], util[i]["const_acc"]
        g = (a - (aU - 0.01), (a - c) - 0.8 * (aU - c), (a - c) - 0.03)
        margins.append(min(g))
        ok = ok and all(x >= 0 for x in g)
    return ok, min(margins)


def gate1(u, uU):
    a, aU, c = u["acc"], uU["acc"], u["const_acc"]
    g = (a - (aU - 0.01), (a - c) - 0.8 * (aU - c), (a - c) - 0.03)
    return all(x >= 0 for x in g), min(g)


def my_selection(k):
    uU = {int(i): v for i, v in inner_rec(f"nn__s{k}__U")["utility"].items()}

    def cand(unit, beta):
        r = inner_rec(unit)
        ok, sh = gates({int(i): v for i, v in r["utility"].items()}, uU)
        return {"unit": unit, "beta": beta, "gates_ok": ok, "worst_gate_margin": sh,
                **{f"R_{w}": r["recovery"][w] for w in ("v1", "v2", "pair")}}

    def closest(Cs):
        nz = [c for c in Cs if c["beta"] != 0] or Cs
        return max(nz, key=lambda c: (c["worst_gate_margin"], -c["beta"]))

    out = {"valid_reference": all(uU[i]["acc"] - uU[i]["const_acc"] >= 0.03 for i in (0, 1)), "arms": {}}
    near = []
    Cln = [cand(f"nn__s{k}__U", 0.0)] + [cand(f"pn__s{k}__LN__b{b:g}", b) for b in BETAS]
    feas = [c for c in Cln if c["gates_ok"]]
    if feas:
        s = min(feas, key=lambda c: (max(c["R_v1"], c["R_v2"]), (c["R_v1"] + c["R_v2"]) / 2, c["beta"]))
        out["arms"]["LN"] = {"status": "TASK_ONLY_ALIAS" if s["beta"] == 0 else "NOMINEE", **s}
    else:
        out["arms"]["LN"] = {"status": "NO_FEASIBLE_NOMINEE", **closest(Cln)}
    if not out["valid_reference"]:
        out["arms"]["LN"]["status"] = "NO_VALID_REFERENCE"
    LN = out["arms"]["LN"]
    ln_ok = LN["status"] in ("NOMINEE", "TASK_ONLY_ALIAS")
    allow = lambda c: ln_ok and c["R_v1"] <= LN["R_v1"] + 0.01 and c["R_v2"] <= LN["R_v2"] + 0.01  # noqa: E731
    tables = {"LN": Cln}
    for arm, cands in (("PN", [cand(f"nn__s{k}__U", 0.0)] + [cand(f"pn__s{k}__PN__b{b:g}", b) for b in BETAS]),
                       ("JP", [cand(f"nn__s{k}__JP__b{b:g}", b) for b in BETAS])):
        tables[arm] = cands
        feas = [c for c in cands if c["gates_ok"] and allow(c)]
        if feas:
            s = min(feas, key=lambda c: (c["R_pair"], c["beta"]))
            out["arms"][arm] = {"status": "TASK_ONLY_ALIAS" if s["beta"] == 0 else "NOMINEE", **s}
        else:
            out["arms"][arm] = {"status": "NO_FEASIBLE_NOMINEE", "descriptive": "INFEASIBLE", **closest(cands)}
    for c in [x for t in tables.values() for x in t]:
        if abs(c["worst_gate_margin"]) < 1e-12:
            near.append(c["unit"])
    for arm in ("U", "E"):
        r = inner_rec(f"nn__s{k}__{arm}")
        ok, sh = gates({int(i): v for i, v in r["utility"].items()}, uU)
        out["arms"][arm] = {"status": "REFERENCE" if arm == "U" else ("NOMINEE" if ok else "GATES_FAILED"),
                            "unit": f"nn__s{k}__{arm}", "beta": 0.0, "gates_ok": ok, "worst_gate_margin": sh,
                            **{f"R_{w}": r["recovery"][w] for w in ("v1", "v2", "pair")}}
    fsel = {}
    for i in (0, 1):
        rows = []
        for cfg in range(1, 7):
            r = inner_rec(f"fare__s{k}__p{i}__c{cfg}")
            ok, sh = gate1(r["utility"], uU[i])
            rows.append({"config": cfg, "unit": f"fare__s{k}__p{i}__c{cfg}", "gate_ok": ok, "margin": sh,
                         "R_local": r["recovery_local"], "acc": r["utility"]["acc"]})
        adm = [x for x in rows if x["gate_ok"]]
        b = min(adm, key=lambda x: (round(x["R_local"], 12), x["config"])) if adm else \
            max(rows, key=lambda x: (x["margin"], -x["config"]))
        fsel[i] = {"status": "NOMINEE" if adm else "NO_FEASIBLE_NOMINEE", **b}
    for arm, tag in (("F", "c"), ("F0", "Z")):
        cfgs = [fsel[0]["config"], fsel[1]["config"]] if arm == "F" else [1, 1]
        units = [f"fare__s{k}__p0__{tag}{cfgs[0]}", f"fare__s{k}__p1__{tag}{cfgs[1]}"]
        r = inner_rec(f"pair__s{k}__{arm}")
        if r["of"] != units:
            raise AssertionError(f"inner pair record of {arm} seed {k} is for {r['of']}, expected {units}")
        ok, sh = gates({int(i): v for i, v in r["utility"].items()}, uU)
        if arm == "F":
            st = "NOMINEE" if ok and all(fsel[i]["status"] == "NOMINEE" for i in (0, 1)) else "NO_FEASIBLE_NOMINEE"
        else:
            st = "NOMINEE" if ok else "GATES_FAILED"
        out["arms"][arm] = {"status": st, "units": units, "configs": cfgs, "gates_ok": ok, "worst_gate_margin": sh,
                            **{f"R_{w}": r["recovery"][w] for w in ("v1", "v2", "pair")}}
        if arm == "F0":
            out["arms"][arm]["own_gate_status"] = "PASS" if ok else "FAIL"
    order = ["LN", "U", "E", "JP", "F", "F0"]
    cands = []
    for arm in order:
        a = out["arms"][arm]
        if a["status"] in ("NOMINEE", "TASK_ONLY_ALIAS", "REFERENCE") and a["gates_ok"] and allow(a):
            cands.append((a["R_pair"], order.index(arm), arm))
    if not out["valid_reference"]:
        cands = []
        for arm in ("PN", "JP", "E", "F", "F0"):
            if out["arms"][arm]["status"] in ("NOMINEE", "TASK_ONLY_ALIAS"):
                out["arms"][arm]["status"] = "NO_VALID_REFERENCE"
    win = min(cands)[2] if cands else None
    wa = out["arms"].get(win, {}) if win else {}
    out["comparator"] = {"arm": win, "candidates": [c[2] for c in cands], "status": wa.get("status"),
                         "unit": wa.get("unit", wa.get("units")),
                         "is_task_only_release": bool(win) and (win == "U" or wa.get("status") == "TASK_ONLY_ALIAS")}
    out["fare_selection"] = fsel
    out["_tables"] = tables
    out["_near_boundary"] = near
    return out


def check_selection():
    SL = C.SL
    mism, nmax, n_fields = [], 0.0, 0
    C.mysel = {}
    near = []
    for k in SEEDS:
        me = my_selection(k)
        C.mysel[k] = me
        near += me["_near_boundary"]
        L = SL["seeds"][str(k)]
        if me["valid_reference"] != L["valid_reference"]:
            mism.append(f"s{k} valid_reference")
        for arm, a in me["arms"].items():
            la = L["arms"][arm]
            for f in ("status", "unit", "units", "beta", "gates_ok", "configs", "descriptive", "own_gate_status"):
                if f in a or f in la:
                    if f == "descriptive" and la.get("status") != "NO_FEASIBLE_NOMINEE":
                        continue
                    n_fields += 1
                    if a.get(f) != la.get(f):
                        mism.append(f"s{k}/{arm}/{f}: mine {a.get(f)} lock {la.get(f)}")
            for f in ("worst_gate_margin", "R_v1", "R_v2", "R_pair"):
                n_fields += 1
                dd = abs(a[f] - la[f])
                nmax = max(nmax, dd)
                if dd > 1e-15:
                    mism.append(f"s{k}/{arm}/{f} diff {dd:.3g}")
        for f in ("arm", "candidates", "status", "unit", "is_task_only_release"):
            n_fields += 1
            if me["comparator"][f] != L["comparator"][f]:
                mism.append(f"s{k}/C*/{f}: mine {me['comparator'][f]} lock {L['comparator'][f]}")
        for i in (0, 1):
            lf = L["fare_selection"][str(i)]
            for f in ("status", "config", "unit", "gate_ok"):
                n_fields += 1
                if me["fare_selection"][i][f] != lf[f]:
                    mism.append(f"s{k}/F p{i}/{f}")
            nmax = max(nmax, abs(me["fare_selection"][i]["margin"] - lf["margin"]))
        # the run-dir selection.json tables (beta = 0 candidates) must be what the lock summarises
        S = rjson(RUN / "selection.json")[str(k)]
        for arm in ("LN", "PN", "JP"):
            for mine, theirs in zip(me["_tables"][arm], S["arms"][arm]["table"]):
                n_fields += 1
                if mine["unit"] != theirs["unit"] or mine["gates_ok"] != theirs["gates_ok"] or \
                        abs(mine["R_pair"] - theirs["R_pair"]) > 1e-15:
                    mism.append(f"s{k}/{arm} table row {mine['unit']}")
    add("10-selection-replay", "PASS" if not mism else "FAIL",
        f"PROTOCOL section 4 re-implemented from the inner records (gates G1-G3 vs U on attacker_val; LN first over "
        f"beta {{0=U,0.1,1,10}} by (worse local, mean local, beta); PN and JP with the +0.01 local allowance vs frozen LN by "
        f"(R_pair, beta); E gates; F per purpose by (R_local, id); F0 own gates; C* over LN,U,E,JP,F,F0; nonzero-beta "
        f"closest-utility fallback; valid_reference): {n_fields} fields compared with SELECTION_LOCK.json and the run "
        f"tables, {len(mism)} mismatches {mism[:5] or ''}; gate margins within 1e-12 of 0: {near or 'none'}",
        max_abs_diff=nmax)
    # seed-by-seed summary
    summ = {k: {a: (C.mysel[k]["arms"][a]["status"], C.mysel[k]["arms"][a].get("unit", C.mysel[k]["arms"][a].get("units")))
                for a in ("LN", "PN", "JP", "E", "F", "F0")} | {"C*": C.mysel[k]["comparator"]["arm"]} for k in SEEDS}
    add("10b-selection-outcome", "INFO", f"replayed outcome: {json.dumps(summ)}")
    # seed 0 PN allowance
    me = C.mysel[0]
    pn = [c for c in me["_tables"]["PN"] if c["beta"] == 0.1][0]
    LN = me["arms"]["LN"]
    lim = LN["R_v2"] + 0.01
    ok = pn["gates_ok"] and pn["R_v2"] > lim and pn["R_v1"] <= LN["R_v1"] + 0.01 and me["arms"]["PN"]["status"] == "NO_FEASIBLE_NOMINEE"
    others = [(c["unit"], c["gates_ok"], c["R_v1"] <= LN["R_v1"] + 0.01 and c["R_v2"] <= lim) for c in me["_tables"]["PN"]]
    add("11-seed0-PN-allowance", "PASS" if ok else "FAIL",
        f"seed 0: PN beta=0.1 passes the gates (worst margin {pn['worst_gate_margin']:.4f}) but R_v2 {pn['R_v2']:.4f} > "
        f"R_v2(LN) {LN['R_v2']:.4f} + 0.01 = {lim:.4f} (excess {pn['R_v2'] - lim:.4f}); R_v1 {pn['R_v1']:.4f} within "
        f"{LN['R_v1'] + 0.01:.4f}; other candidates (unit, gates, allowance): {others} -> NO_FEASIBLE_NOMINEE, "
        f"descriptive {me['arms']['PN']['unit']}")
    # beta = 0 candidates point to U aliases
    probs = []
    for k in SEEDS:
        for arm in ("LN", "PN"):
            r0 = [c for c in C.mysel[k]["_tables"][arm] if c["beta"] == 0]
            if len(r0) != 1 or r0[0]["unit"] != f"nn__s{k}__U":
                probs.append(f"s{k}/{arm}")
        for nm in (f"nn__s{k}__U", f"inner__nn__s{k}__U"):
            a = NEW / nm / "ALIAS.json"
            if not a.exists() or rjson(a)["source"] != f"~/PCRL_eval_cache_private/jcv_v1/run/units/{nm}":
                probs.append(nm)
        if any((NEW / f"pn__s{k}__{a}__b0").exists() for a in ("PN", "LN")):
            probs.append(f"s{k}: a separate beta=0 PN/LN unit exists")
    aliased = [(k, a) for k in SEEDS for a in ("LN", "PN", "JP")
               if C.mysel[k]["arms"][a]["status"] == "TASK_ONLY_ALIAS" or C.mysel[k]["arms"][a]["beta"] == 0]
    add("12-beta0-candidates-are-U-aliases", "PASS" if not probs else "FAIL",
        f"on every seed the beta=0 candidate of LN and PN is nn__s{{k}}__U, whose unit and inner record are ALIAS.json "
        f"records of the predecessor's U (no separate beta=0 model is kept); selections or fallbacks at beta=0 / "
        f"TASK_ONLY_ALIAS: {aliased or 'none'}; problems {probs or 'none'}")


def check_inner_replay():
    """Independent inner recovery (attacker_fit -> attacker_val, inner slate, ignore-other bank) for the decisive seed-0
    releases, and inner utility for every unit the selection reads."""
    D = C.D
    f, v = D["idx"]["attacker_fit"], D["idx"]["attacker_val"]
    S = D["sex"]
    worst, rows = 0.0, []
    for nm in pn_units():
        z = np.load(NEW / nm / "release.npz")
        V = {"v1": np.hstack([z["r1"], z["c1"]]), "v2": np.hstack([z["r2"], z["c2"]])}
        V["pair"] = np.hstack([V["v1"], V["v2"]])
        sel, rec = {}, {}
        for w in ("v1", "v2"):
            sel[w] = fit_select(V[w][f], S[f], V[w][v], S[v], slate_inner(), 2)
            rec[w] = auc_plain(S[v] == 1, sel[w]["Pv"][:, 1])
        sp = fit_select(V["pair"][f], S[f], V["pair"][v], S[v], slate_inner(), 2)
        cands = [("pair:" + sp["selected"], sp["val_log_loss"], sp["Pv"])] + \
                [(f"ignore_other:{w}:" + sel[w]["selected"], sel[w]["val_log_loss"], sel[w]["Pv"]) for w in ("v1", "v2")]
        best = min(cands, key=lambda c: c[1])
        rec["pair"] = auc_plain(S[v] == 1, best[2][:, 1])
        r = inner_rec(nm)
        for w in ("v1", "v2", "pair"):
            worst = max(worst, abs(rec[w] - r["recovery"][w]))
        rows.append({"unit": nm, **{w: round(rec[w], 6) for w in ("v1", "v2", "pair")}, "pair_selected": best[0],
                     "record_pair_selected": r["recovery"]["pair_selected"]})
    sel_ok = all(x["pair_selected"] == x["record_pair_selected"] for x in rows)
    dec = [x for x in rows if x["unit"] in ("pn__s0__PN__b0.1", "pn__s0__LN__b0.1", "pn__s1__PN__b0.1",
                                            "pn__s1__LN__b0.1", "pn__s2__PN__b0.1", "pn__s2__LN__b0.1")]
    add("13-inner-recovery-replay", "PASS" if worst < 1e-9 and sel_ok else "FAIL",
        f"inner slate (LR_C1 | MLP_64x64 | HGB_0.1_31 fitted on attacker_fit, attacker_val log-loss selection, "
        f"ignore-other bank for the pair) refitted on all 18 new PN/LN releases: inner R_v1/R_v2/R_pair reproduced "
        f"(max |diff| {worst:.2g}), pair bank choice identical on {sum(x['pair_selected'] == x['record_pair_selected'] for x in rows)}/18; "
        f"decisive beta=0.1 rows: {dec}", max_abs_diff=worst, rows=rows)
    # inner utility for every unit the selection reads (deployed hard decisions on attacker_val)
    cst = {i: int(np.argmax(np.bincount((D["y_income"] if i == 0 else D["y_occupation_group"])[D["idx"]["defense_train"]])))
           for i in (0, 1)}
    worst, n = 0.0, 0
    for k in SEEDS:
        names = [f"nn__s{k}__U", f"nn__s{k}__E"] + [f"nn__s{k}__JP__b{b}" for b in BSTR] + \
                [f"pn__s{k}__{a}__b{b}" for a in ("PN", "LN") for b in BSTR]
        for nm in names:
            z = np.load(udir(nm) / "release.npz")
            r = inner_rec(nm)
            for i in (0, 1):
                y = (D["y_income"] if i == 0 else D["y_occupation_group"])[v]
                worst = max(worst, abs(float((z[f"hard{i + 1}"][v] == y).mean()) - r["utility"][str(i)]["acc"]),
                            abs(float((y == cst[i]).mean()) - r["utility"][str(i)]["const_acc"]))
                n += 1
        for i in (0, 1):
            for cfg in list(range(1, 7)):
                z = np.load(udir(f"fare__s{k}__p{i}__c{cfg}") / "release.npz")
                r = inner_rec(f"fare__s{k}__p{i}__c{cfg}")
                y = (D["y_income"] if i == 0 else D["y_occupation_group"])[v]
                worst = max(worst, abs(float((z["hard"][v] == y).mean()) - r["utility"]["acc"]))
                n += 1
        for arm in ("F", "F0"):
            r = inner_rec(f"pair__s{k}__{arm}")
            for i, u in enumerate(r["of"]):
                z = np.load(udir(u) / "release.npz")
                y = (D["y_income"] if i == 0 else D["y_occupation_group"])[v]
                worst = max(worst, abs(float((z["hard"][v] == y).mean()) - r["utility"][str(i)]["acc"]))
                n += 1
    add("14-inner-utility-replay", "PASS" if worst == 0.0 else "FAIL",
        f"{n} inner utility entries (deployed hard decisions vs labels on attacker_val; constant = defense_train majority "
        f"{cst}) recomputed from the release files", max_abs_diff=worst)


# =============================================================================== 5. outer recomputation
class Boot:
    def __init__(self, units, B, seed):
        uniq, inv = np.unique(np.asarray(units), return_inverse=True)
        rng = np.random.default_rng(seed)
        p = np.full(len(uniq), 1.0 / len(uniq))
        counts = np.empty((len(uniq), B), np.int64)
        for b in range(B):
            counts[:, b] = rng.multinomial(len(uniq), p)
        self.W = counts[inv].astype(np.float64)
        self.n_units = len(uniq)


class Stat:
    __slots__ = ("pt", "reps")

    def __init__(self, pt, reps):
        self.pt, self.reps = pt, reps


def s_mean(xs):
    return Stat(float(np.mean([x.pt for x in xs])), np.mean(np.stack([x.reps for x in xs]), 0))


def s_diff(a, b):
    return Stat(a.pt - b.pt, a.reps - b.reps)


def s_max(a, b):
    return Stat(max(a.pt, b.pt), np.maximum(a.reps, b.reps))


class Outer:
    def __init__(self, SL, boot):
        self.SL, self.boot = SL, boot
        self.preds, self.cache = {}, {}
        z0 = None
        for k in SEEDS:
            for lab in LABELS:
                z = np.load(NEW / f"outer__s{k}__{lab}" / "preds.npz")
                self.preds[(k, lab)] = {x: z[x] for x in z.files}
                if z0 is None:
                    z0 = self.preds[(k, lab)]
                for key in ("assess_row_id", "assess_unit", "sex", "race_row_id", "race", "y_income", "y_occ"):
                    assert np.array_equal(z[key], z0[key]), f"{key} differs across outer units"
        self.z0 = z0
        self.sex = z0["sex"]
        pos = {r: i for i, r in enumerate(z0["assess_row_id"].tolist())}
        self.race_pos = np.array([pos[r] for r in z0["race_row_id"].tolist()])
        self.race = z0["race"]
        self.Wr = boot.W[self.race_pos]
        D = C.D
        tr = D["idx"]["defense_train"]
        self.const_cls = {0: int(np.argmax(np.bincount(D["y_income"][tr]))),
                          1: int(np.argmax(np.bincount(D["y_occupation_group"][tr])))}
        self.const_correct = {0: (z0["y_income"] == self.const_cls[0]).astype(float),
                              1: (z0["y_occ"] == self.const_cls[1]).astype(float)}

    def _auc(self, key, score, posmask, W):
        if key not in self.cache:
            pt = float(wauc_matrix(score, posmask, np.ones((len(score), 1)))[0])
            self.cache[key] = Stat(pt, wauc_matrix(score, posmask, W) if W is not None else None)
        return self.cache[key]

    def rec(self, k, lab, fmt, view, boot=True):
        P = self.preds[(k, lab)][VIEW_KEY[(fmt, view)]]
        xs = [self._auc(("sex", k, lab, fmt, view, s, boot), P[s][:, 1], self.sex == 1, self.boot.W if boot else None)
              for s in range(3)]
        return xs if not boot else s_mean(xs)

    def rrace(self, k, lab, view, boot=True):
        P3 = self.preds[(k, lab)][f"Prace_{view}"]
        cls = sorted(set(self.race.tolist()))
        per = []
        for s in range(3):
            per.append([self._auc(("race", k, lab, view, s, c, boot), P3[s][:, c], self.race == c, self.Wr if boot else None)
                        for c in cls])
        if not boot:
            return per
        return s_mean([s_mean(x) for x in per])

    def acc(self, k, lab, j):
        p = self.preds[(k, lab)]
        y = p["y_income"] if j == 0 else p["y_occ"]
        c = (p[f"hard{j + 1}"] == y).astype(float)
        W = self.boot.W
        return Stat(float(c.mean()), (c @ W) / W.sum(0))

    def const(self, j):
        c = self.const_correct[j]
        W = self.boot.W
        return Stat(float(c.mean()), (c @ W) / W.sum(0))

    def lab(self, k, arm):
        S = self.SL["seeds"][str(k)]
        if arm == "C*":
            arm = S["comparator"]["arm"]
            if arm is None:
                return None
        if arm in ("F", "F0"):
            return arm
        return unit_label(S["arms"][arm]["unit"])


def endpoint(O, e):
    kind = e["kind"]
    per = []
    for k in SEEDS:
        if kind in ("coalition", "local", "acc", "retain", "useful"):
            pn = O.lab(k, "PN")
            if kind in ("coalition", "local"):
                ref = O.lab(k, e["ref"])
                if ref is None:
                    return None
                if kind == "coalition":
                    per.append(s_diff(O.rec(k, ref, "prim", "pair"), O.rec(k, pn, "prim", "pair")))
                else:
                    per.append(s_diff(O.rec(k, pn, "prim", e["view"]), O.rec(k, ref, "prim", e["view"])))
            elif kind == "acc":
                per.append(s_diff(O.acc(k, pn, e["task"]), O.acc(k, "U", e["task"])))
            elif kind == "retain":
                a, u, c = O.acc(k, pn, e["task"]), O.acc(k, "U", e["task"]), O.const(e["task"])
                per.append(Stat(a.pt - 0.8 * u.pt - 0.2 * c.pt, a.reps - 0.8 * u.reps - 0.2 * c.reps))
            else:
                per.append(s_diff(O.acc(k, pn, e["task"]), O.const(e["task"])))
        elif kind == "erase_acc":
            per += [s_diff(O.acc(k, f"PN_b{b}", e["task"]), O.acc(k, f"JP_b{b}", e["task"])) for b in BSTR]
        elif kind == "erase_rec":
            per += [s_diff(O.rec(k, f"PN_b{b}", "prim", e["view"]), O.rec(k, f"JP_b{b}", "prim", e["view"])) for b in BSTR]
        elif kind == "out":
            per.append(s_diff(O.rec(k, O.lab(k, "LN"), e["fmt"], e["view"]), O.rec(k, O.lab(k, "PN"), e["fmt"], e["view"])))
        elif kind == "race":
            per.append(s_diff(O.rrace(k, O.lab(k, "LN"), e["view"]), O.rrace(k, O.lab(k, "PN"), e["view"])))
        elif kind == "rec":
            per.append(s_diff(O.rec(k, O.lab(k, e["a"]), "prim", "pair"), O.rec(k, O.lab(k, "PN"), "prim", "pair")))
        elif kind == "synergy":
            lab = O.lab(k, e["arm"])
            per.append(s_diff(O.rec(k, lab, "prim", "pair"), s_max(O.rec(k, lab, "prim", "v1"), O.rec(k, lab, "prim", "v2"))))
        elif kind == "beta_matched":
            b = e["beta"]
            per.append(s_diff(O.rec(k, f"LN_b{b}", "prim", "pair"), O.rec(k, f"PN_b{b}", "prim", "pair")))
        else:
            raise ValueError(kind)
    return s_mean(per), len(per)


def decide(e, lo, hi):
    if e["side"] == "lower>":
        return "PASS" if lo > e["target"] else "NOT_ESTABLISHED"
    if e["side"] == "upper<":
        return "PASS" if hi < e["target"] else "NOT_ESTABLISHED"
    return "ABOVE" if lo > e["target"] else ("BELOW" if hi < e["target"] else "NOT_RESOLVED")


def read_csv(p):
    with open(p, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def check_outer():
    LOCK = C.LOCK
    fam = LOCK["families"]
    z1 = float(norm.ppf(1 - 0.05 / 36))
    z2 = float(norm.ppf(1 - 0.05 / 60))
    zok = round(z1, 6) == fam["z_primary"] and round(z2, 6) == fam["z_secondary"] and fam["B"] == 1999 and \
        fam["boot_seed"] == 20261015 and len(fam["primary"]) == 18 and len(fam["secondary"]) == 30
    t0 = time.time()
    O0 = None
    z0 = np.load(NEW / "outer__s0__U" / "preds.npz")
    boot = Boot(z0["assess_unit"], fam["B"], fam["boot_seed"])
    O = Outer(C.SL, boot)
    O0 = O
    C.O = O
    inf = rjson(RUN / "inference.json")
    csvs = {"primary": read_csv(PKG / "PRIMARY_ENDPOINTS.csv"), "secondary": read_csv(PKG / "SECONDARY_ENDPOINTS.csv")}
    rows = {"primary": [], "secondary": []}
    worst = {"json_point": 0.0, "json_se": 0.0, "json_bounds": 0.0, "csv": 0.0}
    dec_mm, rep_mm = [], []
    for famkey, z in (("primary", z1), ("secondary", z2)):
        theirs = {e["id"]: e for e in inf[famkey]}
        for e in fam[famkey]:
            res = endpoint(O, e)
            th = theirs[e["id"]]
            if res is None:
                row = {"id": e["id"], "point": None, "decision": "NOT_ESTABLISHED", "runner_decision": th["decision"]}
                if th["decision"] != "NOT_ESTABLISHED" or th.get("point") is not None:
                    dec_mm.append(e["id"])
                rows[famkey].append(row)
                continue
            st, npairs = res
            r = st.reps[np.isfinite(st.reps)]
            se = float(np.std(r, ddof=1))
            lo, hi = st.pt - z * se, st.pt + z * se
            dec = decide(e, lo, hi)
            dp, ds = abs(st.pt - th["point"]), abs(se - th["se"])
            db = max(abs(lo - th["lower"]), abs(hi - th["upper"]))
            worst["json_point"] = max(worst["json_point"], dp)
            worst["json_se"] = max(worst["json_se"], ds)
            worst["json_bounds"] = max(worst["json_bounds"], db)
            cr = csvs[famkey][e["id"]]
            dc = max(abs(float(cr[x]) - y) for x, y in (("point", st.pt), ("se", se), ("lower", lo), ("upper", hi)))
            worst["csv"] = max(worst["csv"], dc)
            if dec != th["decision"] or dec != cr["decision"]:
                dec_mm.append(f"{e['id']}: mine {dec} json {th['decision']} csv {cr['decision']}")
            if len(r) != th.get("n_finite_replicates", len(r)):
                rep_mm.append(e["id"])
            rows[famkey].append({"id": e["id"], "stat": e["stat"], "side": e["side"], "target": e["target"],
                                 "n_per_seed_or_pair_terms": npairs, "point": st.pt, "se": se, "lower": lo, "upper": hi,
                                 "z": z, "decision": dec, "n_finite_replicates": int(len(r)),
                                 "runner_point": th["point"], "runner_se": th["se"], "runner_lower": th["lower"],
                                 "runner_upper": th["upper"], "runner_decision": th["decision"],
                                 "csv_decision": cr["decision"], "abs_diff_point": dp, "abs_diff_se": ds,
                                 "abs_diff_bounds": db, "abs_diff_vs_csv_max": dc, "agree": dec == th["decision"] == cr["decision"]})
    C.rows = rows
    okj = worst["json_point"] <= TOL_PT and worst["json_se"] <= TOL_PT and worst["json_bounds"] <= 1e-9
    add("15-outer-endpoints", "PASS" if okj and not dec_mm and not rep_mm and worst["csv"] <= TOL_CSV and zok else "FAIL",
        f"18 primary + 30 secondary endpoints rebuilt from the 39 preds.npz files only (SEX AUC with ties 1/2 averaged over "
        f"3 attacker seeds; race macro OvR over present classes {sorted(set(O.race.tolist()))} on {len(O.race)} rows; "
        f"deployed accuracy; const = defense_train majority {O.const_cls}; own multinomial bootstrap of {boot.n_units} "
        f"record groups, B=1999, seed 20261015, SE ddof 1; z {z1:.6f}/{z2:.6f} equal to LOCK: {zok}): max |diff| vs "
        f"inference.json point {worst['json_point']:.2g}, SE {worst['json_se']:.2g}, bounds {worst['json_bounds']:.2g}; "
        f"vs the 6-decimal CSVs {worst['csv']:.2g}; decision mismatches {dec_mm or 'none'}; replicate-count mismatches "
        f"{rep_mm or 'none'} ({time.time() - t0:.0f}s)",
        max_abs_diff=max(worst["json_point"], worst["json_se"], worst["json_bounds"]), max_abs_diff_vs_csv=worst["csv"])
    # per-seed breakdown of the claim-A recovery clauses (descriptive; the family uses the seed means)
    br = {}
    for k in SEEDS:
        pn, ln = O.lab(k, "PN"), O.lab(k, "LN")
        r = {w: (O.rec(k, pn, "prim", w).pt, O.rec(k, ln, "prim", w).pt) for w in ("v1", "v2", "pair")}
        br[k] = {"PN": pn, "PN_status": C.SL["seeds"][str(k)]["arms"]["PN"]["status"], "LN": ln,
                 "R_pair(LN)-R_pair(PN)": round(r["pair"][1] - r["pair"][0], 4),
                 "R_v1(PN)-R_v1(LN)": round(r["v1"][0] - r["v1"][1], 4),
                 "R_v2(PN)-R_v2(LN)": round(r["v2"][0] - r["v2"][1], 4)}
    C.per_seed = br
    add("15c-claimA-per-seed", "INFO", f"outer per-seed values of P01/P02/P03 (selected PN, seed-0 PN is the INFEASIBLE "
        f"descriptive configuration): {json.dumps(br)}")
    # labels scored per seed
    used = {k: {a: O.lab(k, a) for a in ("PN", "LN", "JP", "C*")} for k in SEEDS}
    add("15b-outer-label-map", "INFO", f"selected units mapped to scored labels per seed: {json.dumps(used)}")
    # claims
    st = {k: C.SL["seeds"][str(k)] for k in SEEDS}
    valid = all(st[k]["valid_reference"] for k in SEEDS)
    pn_nz = all(st[k]["arms"]["PN"]["status"] == "NOMINEE" and st[k]["arms"]["PN"]["beta"] != 0 for k in SEEDS) and valid
    ln_nz = all(st[k]["arms"]["LN"]["status"] == "NOMINEE" and st[k]["arms"]["LN"]["beta"] != 0 for k in SEEDS) and valid
    has_c = all(st[k]["comparator"]["arm"] is not None for k in SEEDS) and valid
    claims = {}
    for claim, req in (("A", pn_nz and ln_nz), ("B", pn_nz and has_c)):
        cl = [r for r in rows["primary"] if r["id"] in [e["id"] for e in fam["primary"] if e["claim"] == claim]]
        allp = all(r["decision"] == "PASS" for r in cl)
        failing = [r["id"] for r in cl if r["decision"] != "PASS"]
        th = inf[f"claim{claim}"]
        claims[claim] = {"clauses_passing": sum(r["decision"] == "PASS" for r in cl), "failing_clauses": failing,
                         "all_nine_pass": allp, "status_requirements_met": req,
                         "pn_nonzero_nominee_per_seed": {k: st[k]["arms"]["PN"]["status"] for k in SEEDS},
                         "ln_nonzero_nominee_per_seed": {k: st[k]["arms"]["LN"]["status"] for k in SEEDS},
                         "comparator_per_seed": {k: st[k]["comparator"]["arm"] for k in SEEDS},
                         "decision": "PASS" if allp and req else "NOT_ESTABLISHED",
                         "runner": {x: th[x] for x in ("all_nine_pass", "status_requirements_met", "decision", "clauses_passing")}}
    C.claims = claims
    agree = all(claims[c]["decision"] == claims[c]["runner"]["decision"] and
                claims[c]["clauses_passing"] == claims[c]["runner"]["clauses_passing"] and
                claims[c]["all_nine_pass"] == claims[c]["runner"]["all_nine_pass"] and
                claims[c]["status_requirements_met"] == claims[c]["runner"]["status_requirements_met"] for c in claims)
    add("16-claims", "PASS" if agree else "FAIL",
        f"claim A: {claims['A']['clauses_passing']}/9 clauses pass (failing {claims['A']['failing_clauses']}), status "
        f"requirement met {claims['A']['status_requirements_met']} -> {claims['A']['decision']}; claim B: "
        f"{claims['B']['clauses_passing']}/9 (failing {claims['B']['failing_clauses']}), requirement met "
        f"{claims['B']['status_requirements_met']} -> {claims['B']['decision']}; runner agrees: {agree}. Both fail twice "
        f"over: P02/P11 (local v1 allowance, upper bound above 0.01) and PN is NO_FEASIBLE_NOMINEE on seed 0")
    # levels (points for all, SE where my bootstrap computed the same quantity)
    lv = inf["levels"]
    worst_pt, worst_se, n_se = 0.0, 0.0, 0
    for name, v in lv.items():
        p = name.split("|")
        if p[0] == "R":
            k, lab, fmt, view = int(p[1]), p[2], p[3], p[4]
            per = O.rec(k, lab, fmt, view, boot=False)
            mine = float(np.mean([x.pt for x in per]))
            key0 = ("sex", k, lab, fmt, view, 0, True)
            if key0 in O.cache:
                se = float(np.std(O.rec(k, lab, fmt, view).reps, ddof=1))
                worst_se = max(worst_se, abs(se - v["se"]))
                n_se += 1
        elif p[0] == "Rrace":
            k, lab, view = int(p[1]), p[2], p[3]
            per = O.rrace(k, lab, view, boot=False)
            mine = float(np.mean([np.mean([x.pt for x in cl]) for cl in per]))
        elif p[0] == "acc":
            mine = O.acc(int(p[1]), p[2], int(p[3])).pt
        elif p[0] == "Rmean":
            mine = float(np.mean([np.mean([x.pt for x in O.rec(k, p[1], "prim", p[2], boot=False)]) for k in SEEDS]))
        elif p[0] == "accmean":
            mine = float(np.mean([O.acc(k, p[1], int(p[2])).pt for k in SEEDS]))
        else:
            continue
        worst_pt = max(worst_pt, abs(mine - v["point"]))
    add("17-outer-levels", "PASS" if worst_pt <= TOL_PT and worst_se <= TOL_PT else "FAIL",
        f"all {len(lv)} per-seed / per-label level points in inference.json recomputed (max |diff| {worst_pt:.2g}); "
        f"{n_se} level SEs with my bootstrap (max |diff| {worst_se:.2g})", max_abs_diff=max(worst_pt, worst_se))
    # ACTUAL_TASK_UTILITY.csv accuracy / constant
    worst_u = 0.0
    with open(PKG / "ACTUAL_TASK_UTILITY.csv", newline="") as fh:
        ur = list(csv.DictReader(fh))
    for r in ur:
        j = 0 if r["task"] == "income" else 1
        a = O.acc(int(r["seed"]), r["label"], j).pt
        worst_u = max(worst_u, abs(a - float(r["accuracy"])), abs(O.const(j).pt - float(r["const_accuracy"])),
                      abs(a - O.const(j).pt - float(r["useful_gain"])))
    add("18-utility-csv", "PASS" if worst_u <= 6e-6 and len(ur) == 78 else "FAIL",
        f"{len(ur)} rows of ACTUAL_TASK_UTILITY.csv: deployed accuracy, constant accuracy and useful gain recomputed "
        f"from preds (6 significant digits in the CSV)", max_abs_diff=worst_u)
    # preds labels vs raw inputs; outer preds vs release files on assessment rows
    D = C.D
    a = D["idx"]["assessment"]
    pz = O.z0
    labs_ok = np.array_equal(pz["assess_row_id"], D["row_id"][a]) and np.array_equal(pz["assess_unit"], D["unit"][a]) and \
        np.array_equal(pz["sex"], D["sex"][a]) and np.array_equal(pz["y_income"], D["y_income"][a]) and \
        np.array_equal(pz["y_occ"], D["y_occupation_group"][a])
    sup = [c for c in range(int(D["race"].max()) + 1)
           if all((D["race"][D["idx"][r]] == c).sum() >= 30 for r in ("attacker_fit", "attacker_val", "assessment"))]
    ra = a[np.isin(D["race"][a], sup)]
    race_ok = np.array_equal(pz["race_row_id"], D["row_id"][ra]) and np.array_equal(pz["race"], D["race"][ra])
    rel_bad, rel_max = [], 0.0
    for k in SEEDS:
        for lab in LABELS:
            p = O.preds[(k, lab)]
            rec = rjson(NEW / f"outer__s{k}__{lab}" / "record.json")
            src = rec["source"]
            if isinstance(src, list):
                rz = [np.load(udir(u) / "release.npz") for u in src]
                hard = [rz[0]["hard"], rz[1]["hard"]]
                probs = [rz[0]["p"], rz[1]["p"]]
            else:
                rz = np.load(udir(src) / "release.npz")
                hard = [rz["hard1"], rz["hard2"]]
                probs = [rz["p1"], rz["p2"]]
                if unit_label(src) != lab:
                    rel_bad.append(f"s{k}/{lab}: source {src}")
            for i in (0, 1):
                if not np.array_equal(p[f"hard{i + 1}"], hard[i][a]):
                    rel_bad.append(f"s{k}/{lab}/hard{i + 1}")
                rel_max = max(rel_max, float(np.abs(p[f"p{i + 1}"] - probs[i][a]).max()))
            if rec["race"]["supported_classes"] != sup:
                rel_bad.append(f"s{k}/{lab}: race classes {rec['race']['supported_classes']}")
            sel = C.SL["seeds"][str(k)]["arms"]
            if lab in ("F", "F0") and src != sel[lab]["units"]:
                rel_bad.append(f"s{k}/{lab}: units {src} vs lock {sel[lab]['units']}")
    add("19-outer-inputs-consistency", "PASS" if labs_ok and race_ok and not rel_bad and rel_max == 0 else "FAIL",
        f"outer preds carry exactly the assessment rows / groups / SEX / labels of the admitted inputs ({labs_ok}), race "
        f"subset = supported classes {sup} ({race_ok}); every outer unit's deployed hard decisions and probabilities equal "
        f"its source release on assessment rows (max |dp| {rel_max:.2g}), labels match sources, F/F0 units match the lock: "
        f"{rel_bad or 'all 39'}", max_abs_diff=rel_max)


# =============================================================================== 6. coalition bank provenance
def check_provenance():
    probs, n = [], 0
    bitwise_src = []
    for k in SEEDS:
        for lab in LABELS:
            r = rjson(NEW / f"outer__s{k}__{lab}" / "record.json")
            p = np.load(NEW / f"outer__s{k}__{lab}" / "preds.npz")
            for blk, ckey, locs, pk in ((r["primary"], "pair", ("v1", "v2"), "P_"),
                                        (r["secondary_prob"], "ppair", ("p1", "p2"), "P_"),
                                        (r["secondary_hard"], "hpair", ("h1", "h2"), "P_"),
                                        (r["race"]["audit"], "pair", ("v1", "v2"), "Prace_")):
                n += 1
                c = blk[ckey]
                tag = f"s{k}/{lab}/{ckey}"
                if not c.get("own_table") or set(c.get("ignore_other_tables", {})) != set(locs) or not c.get("bank"):
                    probs.append(f"{tag}: missing table/bank")
                    continue
                if any(not c["ignore_other_tables"][o] for o in locs):
                    probs.append(f"{tag}: empty ignore table")
                for o in locs:
                    if c["ignore_other_tables"][o] != blk[o]["own_table"]:
                        probs.append(f"{tag}: ignore table {o} != the local record")

                def argmin_table(tb):
                    best = None
                    for row in tb:
                        if best is None or row["attacker_val_log_loss"] < best["attacker_val_log_loss"] - 1e-12:
                            best = row
                    return best
                own = argmin_table(c["own_table"])
                if own["attacker"] != c["own_selected"] or own["attacker_val_log_loss"] != c["own_val_log_loss"]:
                    probs.append(f"{tag}: own selection not the table argmin")
                cands = [(c["own_val_log_loss"], 0, ckey)]
                for j, o in enumerate(locs):
                    lo = argmin_table(c["ignore_other_tables"][o])
                    cands.append((lo["attacker_val_log_loss"], j + 1, o))
                    if blk[o]["selected"] != lo["attacker"]:
                        probs.append(f"{tag}: local {o} selection")
                win = min(cands)
                bank_views = [b["view"] for b in c["bank"]]
                bank_ll = [b["val_log_loss"] for b in c["bank"]]
                if bank_views != [ckey] + list(locs) or bank_ll != [x[0] for x in cands]:
                    probs.append(f"{tag}: bank rows")
                if c["selected_view"] != win[2] or c["val_log_loss"] != win[0]:
                    probs.append(f"{tag}: selected {c['selected_view']} vs bank argmin {win[2]}")
                if win[2] != ckey:
                    same = np.array_equal(p[f"{pk}{ckey}"], p[f"{pk}{win[2]}"])
                    bitwise_src.append(same)
                    if not same:
                        probs.append(f"{tag}: coalition preds differ from the winning local view's preds")
    add("20-coalition-bank-provenance", "PASS" if not probs else "FAIL",
        f"{n} coalition banks (primary pair, prob ppair, hard hpair, race pair x 39 outer units): own_table, both "
        f"ignore_other_tables (identical to the local views' own tables) and bank present; own/local selections are the "
        f"table argmins (ties to slate order within 1e-12); selected view = bank argmin (ties coalition, v1, v2); "
        f"{len(bitwise_src)} banks won by a local view carry that view's preds bitwise; problems {probs[:5] or 'none'}")


# =============================================================================== 7. predecessor reproducibility
def check_predecessor():
    res, probs = [], []
    for k in SEEDS:
        prev_jp = rjson(PRED / f"outer__s{k}__JP" / "record.json")["source"]
        pairs = [(lab, lab) for lab in ("U", "E", "F", "F0")] + [(unit_label(prev_jp), "JP")]
        for new_lab, old in pairs:
            a = np.load(NEW / f"outer__s{k}__{new_lab}" / "preds.npz")
            b = np.load(PRED / f"outer__s{k}__{old}" / "preds.npz")
            ra = rjson(NEW / f"outer__s{k}__{new_lab}" / "record.json")
            rb = rjson(PRED / f"outer__s{k}__{old}" / "record.json")
            if ra["source"] != rb["source"]:
                probs.append(f"s{k}/{new_lab}: source {ra['source']} vs {rb['source']}")
            keys_same = sorted(a.files) == sorted(b.files)
            neq = [x for x in a.files if x in b.files and not np.array_equal(a[x], b[x])]
            mx = max([float(np.abs(a[x].astype(float) - b[x].astype(float)).max()) for x in neq] or [0.0])
            sel_same = all(ra[blk][w]["selected"] == rb[blk][w]["selected"] for blk, ws in
                           (("primary", ("v1", "v2", "pair")), ("secondary_prob", ("p1", "p2", "ppair")),
                            ("secondary_hard", ("h1", "h2", "hpair"))) for w in ws)
            res.append({"seed": k, "new": new_lab, "predecessor": old, "keys_same": keys_same, "bitwise": not neq,
                        "n_arrays": len(a.files), "max_abs_diff": mx, "selected_attackers_same": sel_same})
            if not keys_same or neq or not sel_same:
                probs.append(f"s{k}/{new_lab}: {neq[:3]} max {mx:.3g}")
    jp = {k: unit_label(rjson(PRED / f"outer__s{k}__JP" / "record.json")["source"]) for k in SEEDS}
    worst = max(r["max_abs_diff"] for r in res)
    add("21-predecessor-reproducibility", "PASS" if not probs else "FAIL",
        f"{len(res)} re-scored references (U, E, F, F0 and the predecessor's descriptive JP = {jp}) vs the predecessor's "
        f"outer preds: {sum(r['bitwise'] for r in res)}/{len(res)} bitwise equal on all arrays, same sources and same "
        f"selected attackers; problems {probs or 'none'}", max_abs_diff=worst, rows=res)


# =============================================================================== 8. attacker replay
def views_of(src):
    if isinstance(src, list):
        a, b = (np.load(udir(u) / "release.npz") for u in src)
        v1, v2 = np.hstack([a["r"], a["c"]]), np.hstack([b["r"], b["c"]])
        out = {"p1": a["p"], "p2": b["p"], "hard1": a["hard"], "hard2": b["hard"]}
        finite = True
    else:
        z = np.load(udir(src) / "release.npz")
        v1, v2 = np.hstack([z["r1"], z["c1"]]), np.hstack([z["r2"], z["c2"]])
        out = {"p1": z["p1"], "p2": z["p2"], "hard1": z["hard1"], "hard2": z["hard2"]}
        finite = False
    V = {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2]), "p1": out["p1"], "p2": out["p2"],
         "ppair": np.hstack([out["p1"], out["p2"]]), "h1": np.eye(2)[out["hard1"]], "h2": np.eye(6)[out["hard2"]]}
    V["hpair"] = np.hstack([V["h1"], V["h2"]])
    return V, finite


def check_attackers():
    D = C.D
    f, v, a = D["idx"]["attacker_fit"], D["idx"]["attacker_val"], D["idx"]["assessment"]
    S = D["sex"]
    picks = [("outer__s1__PN_b0.1", ("pair", "hpair")), ("outer__s0__LN_b0.1", ("pair", "hpair")),
             ("outer__s1__JP_b1", ("pair", "hpair")), ("outer__s0__F", ("pair", "hpair")),
             ("outer__s2__PN_b0.1", ("pair", "h2"))]
    rows, worst, probs = [], 0.0, []
    for unit, ws in picks:
        rec = rjson(NEW / unit / "record.json")
        p = np.load(NEW / unit / "preds.npz")
        V, finite = views_of(rec["source"])
        for w in ws:
            blk = rec["primary"] if w in ("v1", "v2", "pair") else rec["secondary_hard"]
            r = blk[w]
            src, name = r["selected_view"], r["selected"]
            if w in ("v1", "v2", "pair"):
                sl = dict(slate_final(finite))
            else:
                sl = dict(slate_secondary(True))
            m = sl[name](0).fit(V[src][f], S[f])
            P = proba(m, V[src][a], 2)
            Pv = proba(m, V[src][v], 2)
            d = float(np.abs(P - p[f"P_{w}"][0]).max())
            dll = abs(logloss(S[v], Pv) - r["val_log_loss"])
            worst = max(worst, d)
            rows.append({"unit": unit, "view": w, "selected_view": src, "attacker": name, "max_abs_diff_P0": d,
                         "val_log_loss_abs_diff": dll})
            if d > 1e-9 or dll > 1e-12:
                probs.append(f"{unit}/{w}/{name}: P {d:.3g}, val ll {dll:.3g}")
    # full own-table replay for the seed-1 PN nominee coalition view
    rec = rjson(NEW / "outer__s1__PN_b0.1" / "record.json")
    V, finite = views_of(rec["source"])
    fs = fit_select(V["pair"][f], S[f], V["pair"][v], S[v], slate_final(finite), 2)
    tdiff = max(abs(x["attacker_val_log_loss"] - y["attacker_val_log_loss"]) for x, y in zip(fs["table"], rec["primary"]["pair"]["own_table"]))
    names_same = [x["attacker"] for x in fs["table"]] == [y["attacker"] for y in rec["primary"]["pair"]["own_table"]]
    if tdiff > 1e-12 or not names_same or fs["selected"] != rec["primary"]["pair"]["own_selected"]:
        probs.append(f"own_table replay: diff {tdiff:.3g}, selected {fs['selected']}")
    add("22-attacker-replay", "PASS" if not probs else ("WARN" if worst < 1e-6 else "FAIL"),
        f"{len(rows)} recorded selected attackers refitted at attacker seed 0 from their names with my own slate "
        f"(final: LR C grid, MLP grid, HGB grid, DA = PCA whiten + drop var <= 1e-9 max + MLP(128,128); secondary: inner "
        f"+ DA + cell-conditional): P_view[0] on assessment max |diff| {worst:.3g}, attacker_val log loss reproduced; "
        f"full 14-attacker final-slate table for outer__s1__PN_b0.1/pair reproduced (max |dll| {tdiff:.2g}, same "
        f"argmin {fs['selected']}); problems {probs or 'none'}", max_abs_diff=max(worst, tdiff), rows=rows)


# =============================================================================== 9. critic gap
def check_critic_gap():
    D = C.D
    S = D["sex"]
    av = D["idx"]["attacker_val"]
    X = torch.from_numpy(D["X"])
    with open(PKG / "CRITIC_GAP.csv", newline="") as fh:
        cg = {(r["unit"], r["view"]): r for r in csv.DictReader(fh)}
    worst_rec, worst_csv_cols = 0.0, 0.0
    prim, cells, missing = [], {}, []
    gap_cols = {"primary_mean_paired_online_minus_fresh_def": 0.0, "best_of_bank_online_minus_fresh_def": 0.0,
                "sensitivity_thetaT_refit_whitener_primary": 0.0}
    per_unit = {}
    n = 0
    for nm in pn_units():
        r = rjson(NEW / f"critic__{nm}" / "record.json")
        assert r["snapshot"].startswith("theta_{T-1}")
        vals = []
        for w, v in r["views"].items():
            n += 1
            mine = float(np.mean([o["ce"] - fr["ce"] for o, fr in zip(v["online"], v["fresh_def"])]))
            kinds_ok = [o["kind"] for o in v["online"]] == [fr["kind"] for fr in v["fresh_def"]]
            assert kinds_ok
            worst_rec = max(worst_rec, abs(mine - v["primary_mean_paired_online_minus_fresh_def"]),
                            abs(min(o["ce"] for o in v["online"]) - v["online_best_ce"]),
                            abs(min(o["ce"] for o in v["fresh_def"]) - v["fresh_def_best_ce"]))
            prim.append(mine)
            vals += [o["ce"] - fr["ce"] for o, fr in zip(v["online"], v["fresh_def"])]
            cells[(nm, w)] = {"primary": mine, "best": v["online_best_ce"] - v["fresh_def_best_ce"],
                              "sens": v["sensitivity_thetaT_refit_whitener"]["primary_mean_paired_online_minus_fresh_def"],
                              "online_best": v["online_best_ce"], "fresh_def_best": v["fresh_def_best_ce"],
                              "prior": v["prior_ce"]}
            row = cg.get((nm, w))
            if row is None:
                missing.append(f"{nm}/{w}")
                continue
            for col in ("prior_ce", "online_best_ce", "fresh_def_best_ce", "fresh_att_best_ce", "slate_ce", "slate_auc"):
                worst_csv_cols = max(worst_csv_cols, abs(float(row[col]) - v[col]) / max(1.0, abs(v[col])))
            for col, ref in (("primary_mean_paired_online_minus_fresh_def", mine),
                             ("best_of_bank_online_minus_fresh_def", cells[(nm, w)]["best"]),
                             ("sensitivity_thetaT_refit_whitener_primary", cells[(nm, w)]["sens"])):
                if col not in row:
                    missing.append(f"column {col}")
                    continue
                gap_cols[col] = max(gap_cols[col], abs(float(row[col]) - ref))
        per_unit[nm] = float(np.mean(vals))
    overall = float(np.mean([x for nm in pn_units() for x in
                             [o["ce"] - fr["ce"] for w, v in rjson(NEW / f"critic__{nm}" / "record.json")["views"].items()
                              for o, fr in zip(v["online"], v["fresh_def"])]]))
    npos = sum(1 for x in per_unit.values() if x > 0)
    add("23-critic-gap-primary-statistic", "PASS" if worst_rec < 1e-15 else "FAIL",
        f"registered primary (mean over the bank of paired CE(online j) - CE(fresh_def j), theta_(T-1), saved whitener, "
        f"attacker_val) recomputed from the stored per-critic CEs for {n} unit-views: equals the record field "
        f"(max |diff| {worst_rec:.2g}); per unit-view range [{min(prim):.4f}, {max(prim):.4f}], positive in "
        f"{sum(x > 0 for x in prim)}/{n}; per-unit mean positive in {npos}/18; overall mean over all paired critics "
        f"{overall:.4f} nats (prediction 6 'positive for most units, a few hundredths': direction confirmed, size larger "
        f"at beta >= 1)", max_abs_diff=worst_rec)
    # ---- check 24: the public CRITIC_GAP.csv (corrected schema) against the records and my recomputation
    def git(*a):
        return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True).stdout
    rel = "results/pcrl_penalty_no_erasure_v1/CRITIC_GAP.csv"
    old_cols = list(cg.values())[0].keys() if cg else []
    hist = "history unavailable"
    old_txt = git("show", f"ec1ae45:{rel}")
    if old_txt:
        old = {(r["unit"], r["view"]): r for r in csv.DictReader(old_txt.splitlines())}
        if old and "gap_online_minus_fresh_def" in next(iter(old.values())):
            ob = max(abs(float(r["gap_online_minus_fresh_def"]) - cells[key]["best"]) for key, r in old.items())
            op = max(abs(float(r["gap_online_minus_fresh_def"]) - cells[key]["primary"]) for key, r in old.items())
            hist = (f"history: the first verifier run returned WARN because the CSV committed in ec1ae45 had a single "
                    f"column 'gap_online_minus_fresh_def' that is the best-of-bank difference (max |diff| to "
                    f"online_best - fresh_def_best {ob:.2g}) under a name suggesting the registered statistic (max |diff| "
                    f"to it {op:.3g}); root cause report.py critic_gap(). Corrected after that run")
    state = "uncommitted working-tree change vs HEAD" if git("status", "--porcelain", "--", rel).strip() else "committed"
    tol = 1e-6
    ok24 = not missing and len(cg) == len(cells) == 45 and "gap_online_minus_fresh_def" not in old_cols and \
        all(x <= tol for x in gap_cols.values())
    add("24-critic-gap-csv", "PASS" if ok24 else "FAIL",
        f"corrected CRITIC_GAP.csv ({len(cg)} rows, {state}): 'primary_mean_paired_online_minus_fresh_def' equals my "
        f"recomputed registered mean-paired statistic (max |diff| {gap_cols['primary_mean_paired_online_minus_fresh_def']:.2g}); "
        f"'best_of_bank_online_minus_fresh_def' equals online_best_ce - fresh_def_best_ce (max |diff| "
        f"{gap_cols['best_of_bank_online_minus_fresh_def']:.2g}); 'sensitivity_thetaT_refit_whitener_primary' equals the "
        f"record's sensitivity field (max |diff| {gap_cols['sensitivity_thetaT_refit_whitener_primary']:.2g}); tolerance "
        f"{tol:g} for 6 significant digits; other columns max rel diff {worst_csv_cols:.2g}; ambiguous old column absent "
        f"{'gap_online_minus_fresh_def' not in old_cols}; missing {missing or 'none'}. {hist}.",
        max_abs_diff=max(gap_cols.values()))
    # ---- check 24b: the per-beta/view seed-mean table and summary numbers of CRITIC_GAP_DIAGNOSIS.md
    md = (PKG / "CRITIC_GAP_DIAGNOSIS.md").read_text()
    table = {}
    for line in md.splitlines():
        if not (line.startswith("| PN ") or line.startswith("| LN ")):
            continue
        cl = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        arm, b = cl[0].split()
        ws = [x.strip() for x in cl[1].split("/")]
        cols = [[float(x.strip()) for x in cl[j].split("/")] for j in range(2, 6)]
        for i, w in enumerate(ws):
            table[(arm, b, w)] = cols[0][i], cols[1][i], cols[2][i], cols[3][i]
    names = ("online_best", "fresh_def_best", "primary", "sens")
    mm, big, worst_md, n_md = [], [], 0.0, 0
    for arm in ("PN", "LN"):
        for b in BSTR:
            for w in (("v1", "v2", "pair") if arm == "PN" else ("v1", "v2")):
                seedvals = {x: [cells[(f"pn__s{k}__{arm}__b{b}", w)][x] for k in SEEDS] for x in names}
                mine = [float(np.mean(seedvals[x])) for x in names]
                if (arm, b, w) not in table:
                    mm.append(f"{arm} {b} {w}: row missing")
                    big.append(f"{arm} {b} {w}")
                    continue
                for x, mv, tv in zip(names, mine, table[(arm, b, w)]):
                    n_md += 1
                    worst_md = max(worst_md, abs(mv - tv))
                    if abs(round(mv, 3) - tv) > 1e-9:
                        dbl = round(float(np.mean([round(y, 3) for y in seedvals[x]])), 3)
                        mm.append(f"{arm} {b} {w} {x}: table {tv:+.3f}, recomputed seed mean {mv:+.5f} (rounds to "
                                  f"{round(mv, 3):+.3f}); mean of the per-seed values rounded to 3 decimals gives "
                                  f"{dbl:+.3f} ({'= table: double rounding' if abs(dbl - tv) < 1e-9 else 'unexplained'})")
                        if abs(mv - tv) > 0.001 + 1e-12 or abs(dbl - tv) > 1e-9:
                            big.append(f"{arm} {b} {w} {x}")
    extra = sorted(set(table) - {(a, b, w) for a in ("PN", "LN") for b in BSTR
                                 for w in (("v1", "v2", "pair") if a == "PN" else ("v1", "v2"))})
    pv = [c["primary"] for c in cells.values()]
    bv = [c["best"] for c in cells.values()]
    sd_cell = max(abs(c["primary"] - c["sens"]) for c in cells.values())
    sd_mean = max(abs(t[2] - t[3]) for t in table.values()) if table else float("nan")
    priors = sorted({round(c["prior"], 3) for c in cells.values()})
    pn01v1 = float(np.mean([cells[(f"pn__s{k}__PN__b0.1", "v1")]["primary"] for k in SEEDS]))
    summary = (f"text claims recomputed: primary positive {sum(x > 0 for x in pv)}/{len(pv)} cells, mean of the 45 cell "
               f"values {np.mean(pv):+.4f} (doc '+0.047'); best-of-bank positive {sum(x > 0 for x in bv)}/{len(bv)}, mean "
               f"{np.mean(bv):+.4f} (doc '+0.063'); prior CE {priors} (doc 0.629 every row); |primary - sensitivity| max "
               f"{sd_mean:.3f} over the table's seed means and {sd_cell:.3f} over single cells (doc 'within about 0.01'); "
               f"PN 0.1 v1 seed-mean primary {pn01v1:+.5f} (doc rounding fix '+0.026, not +0.027')")
    txt_mm = []
    if sum(x > 0 for x in pv) != 45 or round(float(np.mean(pv)), 3) != 0.047:
        txt_mm.append("primary 45/45 / +0.047")
    if sum(x > 0 for x in bv) != 45 or round(float(np.mean(bv)), 3) != 0.063:
        txt_mm.append("best-of-bank 45/45 / +0.063")
    if priors != [0.629]:
        txt_mm.append("prior 0.629")
    if round(pn01v1, 3) != 0.026:
        txt_mm.append("PN 0.1 v1 rounding")
    sens_note = (f"'within about 0.01' holds for the table's seed means (max {sd_mean:.3f}) but not for single cells "
                 f"(max {sd_cell:.3f})")
    if sd_mean > 0.015:
        txt_mm.append(f"sensitivity agreement 'within about 0.01' (max {sd_mean:.3f} over seed means)")
    # FAIL only for a substantive error (missing row, > 1 unit in the last printed digit, unexplained value or a wrong
    # text claim); last-digit errors fully explained by double rounding are WARN
    st24b = "FAIL" if big or extra or txt_mm else ("WARN" if mm else "PASS")
    add("24b-critic-gap-diagnosis-table", st24b,
        f"CRITIC_GAP_DIAGNOSIS.md results table: {len(table)} arm/beta/view rows x 4 columns (best-of-bank online CE, "
        f"best-of-bank fresh_def CE, primary mean-paired gap, theta_T sensitivity) compared with my seed means over the "
        f"3 seeds of the critic__ records: {n_md - len(mm)}/{n_md} printed values equal the correct 3-decimal rounding "
        f"(max |recomputed - printed| {worst_md:.4f}); mismatches {mm or 'none'}. Root cause of the mismatches: those "
        f"cells were averaged from per-seed values already rounded to 3 decimals (double rounding), the same artefact as "
        f"the documented PN 0.1 v1 fix, which the correction pass did not catch here; each is off by one unit in the "
        f"third decimal and no conclusion changes. Unexpected rows {extra or 'none'}; {summary}; {sens_note}; "
        f"substantive text-claim discrepancies {txt_mm or 'none'}", max_abs_diff=worst_md)
    # independent re-evaluation of the online critics at theta_(T-1)
    worst_on, worst_f32, worst_T, n_eval = 0.0, 0.0, 0.0, 0
    Sav = S[av]
    for nm in pn_units():
        crit = torch.load(NEW / nm / "critics_final.pt", weights_only=True)
        r = rjson(NEW / f"critic__{nm}" / "record.json")
        m = load_net(crit["model_state_at_last_critic_update"])
        with torch.no_grad():
            Vt = train_views(m, X, list(crit["critics"]))
        for w in crit["critics"]:
            wh = crit["whiteners"][w]
            V64 = Vt[w].double().numpy()
            Z = (V64 - wh["mu"].double().numpy()) @ wh["W"].double().numpy()
            with torch.no_grad():
                Z32 = ((Vt[w] - wh["mu"]) @ wh["W"]).numpy()
            for j, (kind, st) in enumerate(zip(crit["kinds"][w], crit["critics"][w])):
                c = critic_net(kind, Z.shape[1])
                c.load_state_dict(st)
                with torch.no_grad():
                    P = torch.softmax(c(torch.from_numpy(Z[av]).float()), 1).double().numpy()
                    P32 = torch.softmax(c(torch.from_numpy(Z32[av])), 1).double().numpy()
                ce = logloss(Sav, P)
                worst_on = max(worst_on, abs(ce - r["views"][w]["online"][j]["ce"]))
                worst_f32 = max(worst_f32, abs(logloss(Sav, P32) - ce))
                n_eval += 1
    add("25-online-critic-reevaluation", "PASS" if worst_on < 1e-9 else ("WARN" if worst_on < 1e-5 else "FAIL"),
        f"all {n_eval} saved online critics of the 18 PN/LN units re-evaluated with my own critic/encoder definitions: "
        f"training-time views [g_i(X), centred training-head logits] at the saved theta_(T-1), saved whitener "
        f"(V - mu) @ W, kinds A = Linear(dv,32)-ReLU-Linear(32,2), B/B2 = 64-64; SEX CE on attacker_val vs the stored "
        f"online CEs: max |diff| {worst_on:.2g} (whitening in float32 instead of float64 changes CE by at most "
        f"{worst_f32:.2g})", max_abs_diff=worst_on)


# =============================================================================== 10. controls
def check_controls():
    cj = rjson(RUN / "controls.json")
    pub = rjson(PKG / "AUDIT_CONTROLS.json")
    nulls = [v["null_val_auc"] for v in cj.values()]
    plants = [v["planted_val_auc"] for v in cj.values()]
    flags_ok = all((v["null_val_auc"] > 0.55) == v["null_flag_above_0.55"] and
                   (v["planted_val_auc"] > 0.75) == v["planted_detected_above_0.75"] for v in cj.values())
    ok = cj == pub and len(cj) == 9 and max(nulls) <= 0.55 and min(plants) > 0.75 and flags_ok
    add("26-controls", "PASS" if ok else "FAIL",
        f"{len(cj)} controls (PN, LN seed-0 selected units and U; v1, v2, pair): shuffled-label null AUC max "
        f"{max(nulls):.4f} <= 0.55, planted-leak AUC min {min(plants):.4f} > 0.75; flags consistent {flags_ok}; public "
        f"AUDIT_CONTROLS.json identical to the private file {cj == pub}")
    # independent replay of one control (LN seed 0 pair)
    D = C.D
    f, v = D["idx"]["attacker_fit"], D["idx"]["attacker_val"]
    S = D["sex"]
    V, finite = views_of("pn__s0__LN__b0.1")
    X = V["pair"]
    rng = np.random.default_rng(20261014)
    Sp = S.copy()
    Sp[f] = rng.permutation(S[f])
    Sp[v] = rng.permutation(S[v])
    sel = fit_select(X[f], Sp[f], X[v], Sp[v], slate_secondary(finite), 2)
    null = auc_plain(Sp[v] == 1, proba(sel["model0"], X[v], 2)[:, 1])
    noisy = S.copy()
    flip = rng.random(len(S)) < 0.2
    noisy[flip] = rng.integers(0, 2, flip.sum())
    Xp = np.hstack([X, np.eye(2)[noisy]])
    sel2 = fit_select(Xp[f], S[f], Xp[v], S[v], slate_secondary(finite), 2)
    planted = auc_plain(S[v] == 1, proba(sel2["model0"], Xp[v], 2)[:, 1])
    th = cj["LN(pn__s0__LN__b0.1)/pair"]
    d = max(abs(null - th["null_val_auc"]), abs(planted - th["planted_val_auc"]))
    add("26b-control-replay", "PASS" if d < 1e-9 else "FAIL",
        f"LN(pn__s0__LN__b0.1)/pair control replayed (SEX permuted within attacker_fit and attacker_val, seed 20261014; "
        f"planted one-hot SEX with 20% random replacement; secondary slate): null {null:.4f}, planted {planted:.4f}",
        max_abs_diff=d)


# =============================================================================== 11. F0 status
def check_f0():
    probs = []
    for k in SEEDS:
        a = C.SL["seeds"][str(k)]["arms"]["F0"]
        me = C.mysel[k]["arms"]["F0"]
        fsel = C.SL["seeds"][str(k)]["fare_selection"]
        own = "PASS" if a["gates_ok"] else "FAIL"
        exp_status = "NOMINEE" if a["gates_ok"] else "GATES_FAILED"
        if a.get("own_gate_status") != own or a["status"] != exp_status or me["status"] != a["status"]:
            probs.append(f"s{k}: status {a['status']} own {a.get('own_gate_status')} mine {me['status']}")
        if "pairing_to_F" not in a or not all(f"p{i} {fsel[str(i)]['status']} cfg {fsel[str(i)]['config']}" in a["pairing_to_F"]
                                               for i in (0, 1)):
            probs.append(f"s{k}: pairing field")
        if a["units"] != [f"fare__s{k}__p0__Z1", f"fare__s{k}__p1__Z1"]:
            probs.append(f"s{k}: units")
    pre = {k: rjson(PRED / f"outer__s{k}__F0" / "record.json")["status"] for k in SEEDS}
    st = {k: (C.SL["seeds"][str(k)]["arms"]["F0"]["status"], C.SL["seeds"][str(k)]["arms"]["F"]["status"]) for k in SEEDS}
    add("27-F0-status", "PASS" if not probs else "FAIL",
        f"F0 status is its own gate status on every seed (F0, F) = {st}; own_gate_status recorded; pairing_to_F carried "
        f"separately and names F's per-purpose outcome; F0 units frozen at Z1 x 2. (The predecessor's outer records "
        f"labelled F0 {pre}, the mislabel this study repairs.) problems {probs or 'none'}")


# =============================================================================== 12. integrity
def check_integrity():
    n, bad = 0, []
    for d in sorted(NEW.iterdir()):
        if not d.is_dir():
            continue
        n += 1
        if d.name.endswith((".tmp", ".quarantined")):
            bad.append(f"stray {d.name}")
            continue
        ok, b, _ = complete_ok(d)
        if not ok:
            bad.append(f"{d.name}: {b[:2]}")
    add("28-complete-hashes", "PASS" if not bad else "FAIL",
        f"{n} unit directories in the run: every COMPLETE.json lists exactly the files on disk and every sha256 verifies "
        f"(aliases' sources verified in check 03); problems {bad[:5] or 'none'}")
    LOCK = C.LOCK
    mism = [f for f, h in LOCK["code_files"].items() if not (WT / f).exists() or sha(WT / f) != h]
    globs = ["pnx/*.py", "pnx/tests/*.py", "jcv/*.py", "stored_model_eval/defenses.py", "stored_model_eval/bench_infer.py",
             "stored_model_eval/pilot_infer.py", "stored_model_eval/guards.py", "oar/fare_official.py", "oar/study.py",
             "pcrl/data/adult.py"]
    now = {str(Path(p).relative_to(WT)) for g in globs for p in glob.glob(str(WT / g))}
    added = sorted(now - set(LOCK["code_files"]))
    docs = {d: sha(PKG / d) == h for d, h in LOCK["documents_sha256"].items()}
    add("29-lock-code-hashes", "PASS" if not mism and not added and all(docs.values()) else "FAIL",
        f"{len(LOCK['code_files'])} LOCK.json code_files re-hashed from the worktree: {len(LOCK['code_files']) - len(mism)} "
        f"match (mismatch {mism or 'none'}); files added under the locked globs since: {added or 'none'}; locked "
        f"documents unchanged {docs}")
    reused = LOCK["admitted"]["reused_predecessor_units"]
    rbad = [p for p, h in reused.items() if sha(HOME / p[2:]) != h]
    add("29b-lock-reused-units", "PASS" if not rbad else "FAIL",
        f"{len(reused)} reused predecessor COMPLETE.json hashes in LOCK.json verify: {len(reused) - len(rbad)} "
        f"(bad {rbad[:3] or 'none'}); input path/hash in LOCK equal the admitted ones "
        f"{LOCK['admitted']['private_inputs_sha256'] == INPUT_SHA}")
    nart, abad = 0, []
    for k, s in C.SL["seeds"].items():
        for arm, a in s["arms"].items():
            for unit, files in a.get("artifact_sha256", {}).items():
                for rel, h in files.items():
                    nart += 1
                    if sha(udir(unit) / rel) != h:
                        abad.append(f"{unit}/{rel}")
    add("30-selection-lock-artifacts", "PASS" if not abad and nart else "FAIL",
        f"{nart} artifact_sha256 entries of SELECTION_LOCK.json verified against the unit files (aliases resolved to the "
        f"predecessor files): bad {abad or 'none'}")
    # pushed and unchanged
    def git(*a):
        return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True).stdout.strip()
    rel = "results/pcrl_penalty_no_erasure_v1/SELECTION_LOCK.json"
    last = git("log", "-1", "--format=%H", "--", rel)
    remotes = git("branch", "-r", "--contains", SEL_COMMIT)
    blob_commit = git("rev-parse", f"{SEL_COMMIT}:{rel}")
    blob_origin = git("rev-parse", f"origin/{BRANCH}:{rel}")
    blob_now = git("hash-object", str(PKG / "SELECTION_LOCK.json"))
    dirty = git("status", "--porcelain", "--", rel)
    ok = last == SEL_COMMIT and f"origin/{BRANCH}" in remotes and blob_commit == blob_origin == blob_now and not dirty
    lock_commit_in = C.SL["lock"].endswith(git("log", "-1", "--format=%H", "--", "results/pcrl_penalty_no_erasure_v1/LOCK.json"))
    add("31-selection-lock-pushed", "PASS" if ok and lock_commit_in else "FAIL",
        f"SELECTION_LOCK.json last changed in {last[:12]} (= {SEL_COMMIT[:12]}: {last == SEL_COMMIT}), commit contained in "
        f"origin/{BRANCH}: {f'origin/{BRANCH}' in remotes}; blob at the commit = blob on origin = working file: "
        f"{blob_commit == blob_origin == blob_now}; uncommitted changes: {bool(dirty)}; its 'lock' field names the LOCK.json "
        f"commit {lock_commit_in}")
    # outer scoring started after the selection commit
    ct = git("show", "-s", "--format=%cI", SEL_COMMIT)
    cdt = datetime.fromisoformat(ct).astimezone(timezone.utc).replace(tzinfo=None)
    starts = []
    for line in (RUN / "ACTIVITY_LOG.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("unit", "").startswith("outer__"):
            end = datetime.strptime(e["at"], "%Y-%m-%dT%H:%M:%SZ")
            w = rjson(NEW / e["unit"] / "record.json")["wall_s"]
            starts.append(end - timedelta(seconds=w))
    first = min(starts)
    sel_done = [json.loads(x)["at"] for x in (RUN / "ACTIVITY_LOG.jsonl").read_text().splitlines()
                if json.loads(x)["event"] == "selection done"]
    crit_after = all(datetime.strptime(json.loads(x)["at"], "%Y-%m-%dT%H:%M:%SZ") > cdt for x in
                     (RUN / "ACTIVITY_LOG.jsonl").read_text().splitlines() if json.loads(x).get("unit", "").startswith("critic__"))
    add("32-outer-after-selection-lock", "PASS" if first > cdt else "WARN",
        f"selection computed {sel_done}; SELECTION_LOCK commit at {cdt.isoformat()}Z; earliest outer unit start "
        f"(completion minus wall time) {first.isoformat(timespec='seconds')}Z ({(first - cdt).total_seconds():.0f}s later); "
        f"the outer runner refuses unless the lock commit is on origin; critic-gap units all completed after the "
        f"selection commit {crit_after}. Push time itself is not recorded locally.")
    # dependencies of this replay vs the lock
    import importlib.metadata as md
    import platform
    import scipy
    import sklearn
    mine = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
            "machine": platform.machine()}
    try:
        mine["concept-erasure"] = md.version("concept-erasure")
    except Exception:
        mine["concept-erasure"] = None
    dd = {k: (v, LOCK["dependencies"].get(k)) for k, v in mine.items() if v != LOCK["dependencies"].get(k)}
    add("33-environment", "PASS" if not dd else "WARN",
        f"replay environment versus LOCK.json dependencies: {'identical' if not dd else dd}; torch threads 1, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    rs = rjson(PKG / "RUN_STATUS.json")
    rs_claims = rs.get("claims", {})
    mine_cl = {c: getattr(C, "claims", {}).get(c, {}).get("decision") for c in ("A", "B")}
    rs_locks = rs.get("locks", {})
    rs_ok = rs.get("stage") == "COMPLETE" and rs_claims == mine_cl and \
        rs_locks.get("selection", SEL_COMMIT) == SEL_COMMIT and \
        rs_locks.get("protocol", C.SL["lock"].split("@")[-1].strip()) == C.SL["lock"].split("@")[-1].strip()
    add("34-run-status-file", "PASS" if rs_ok else "WARN",
        f"public RUN_STATUS.json: stage {rs.get('stage')!r} (updated {rs.get('updated_at')}), claims {rs_claims} vs "
        f"replayed {mine_cl}, lock commits {rs_locks or 'not recorded'}. History: the first verifier run returned WARN "
        f"because the file still said 'LOCKED_BEFORE_FITS' after the study had finished; corrected after that run "
        f"(status file only, no effect on results)")
    # training diagnostics (finite training, no rescue)
    rows = []
    for nm in pn_units():
        g = rjson(NEW / nm / "record.json")["diag"]
        rows.append((g["nonfinite"], bool(g.get("rescue")), g["steps"], g["protection_steps_attempted"], g["leace_refits"]))
    ok = all(r == (0, False, 1520, 1520, 0) for r in rows)
    add("35-training-records", "PASS" if ok else "FAIL",
        f"18 PN/LN records: nonfinite 0, no rescue retry, 1520 encoder updates, 1520 penalty steps, 0 LEACE refits: {ok}")


def check_jp_e_release():
    """E/JP: r = h - ((h - mean_x) @ proj_right.T) @ proj_left.T in float64 from the saved LEACE npz (descriptive)."""
    D = C.D
    X = torch.from_numpy(D["X"])
    worst, n = 0.0, 0
    for k in SEEDS:
        for nm in [f"nn__s{k}__E"] + [f"nn__s{k}__JP__b{b}" for b in BSTR]:
            d = udir(nm)
            m = load_net(torch.load(d / "model.pt", weights_only=True))
            z = np.load(d / "release.npz")
            for i in (0, 1):
                with torch.no_grad():
                    h = m.enc[i](X).double().numpy()
                L = np.load(d / f"leace_{i}" / "leace_map.npz")
                r = h - ((h - L["mean_x"]) @ L["proj_right"].T) @ L["proj_left"].T
                worst = max(worst, float(np.abs(r - z[f"r{i + 1}"]).max()))
                n += 1
    add("36-erased-reference-release-replay", "PASS" if worst < 1e-9 else "INFO",
        f"{n} E/JP recipient releases rebuilt from model.pt + saved LEACE npz (float64 affine form): max |diff| "
        f"{worst:.2g} (erasure references are deployed with a map; PN/LN are not)", max_abs_diff=worst)


def check_fare_seed_identity():
    same = []
    for i in (0, 1):
        for cfg in ("c1", "Z1"):
            hs = [sha(udir(f"fare__s{k}__p{i}__{cfg}") / "release.npz") for k in SEEDS]
            arrs = [np.load(udir(f"fare__s{k}__p{i}__{cfg}") / "release.npz") for k in SEEDS]
            eq = all(np.array_equal(arrs[0][x], arrs[j][x]) for j in (1, 2) for x in arrs[0].files)
            same.append(f"p{i}/{cfg}: releases identical across encoder seeds={eq}")
    add("37-fare-seed-dependence", "INFO",
        f"FARE references reused from the predecessor: {same}. Identical releases across seeds make the F/F0 'seed' "
        f"replicates partly non-independent (explains identical inner R values across seeds in SELECTION_LOCK); "
        f"predecessor-disclosed, no effect on the PN/LN contrasts")


# =============================================================================== main
def main():
    t0 = time.time()
    C.SL = rjson(PKG / "SELECTION_LOCK.json")
    C.LOCK = rjson(PKG / "LOCK.json")
    steps = [check_inputs, check_deployment, check_last_step, check_u_replication, check_parity_receipts,
             check_selection, check_inner_replay, check_outer, check_provenance, check_predecessor, check_attackers,
             check_critic_gap, check_controls, check_f0, check_integrity, check_jp_e_release, check_fare_seed_identity]
    for fn in steps:
        try:
            fn()
        except Exception as e:  # recorded as a failure of the verifier step, never hidden
            add(f"ERROR-{fn.__name__}", "FAIL", f"{type(e).__name__}: {e} | {traceback.format_exc().splitlines()[-3:]}")
    loaded = sorted({m.split(".")[0] for m in sys.modules} & set(FORBIDDEN))
    add("00-import-guard", "PASS" if not loaded and not BLOCKED else "FAIL",
        f"sys.meta_path guard active for {list(FORBIDDEN)}; forbidden modules loaded at the end: {loaded or 'none'}; "
        f"blocked import attempts: {BLOCKED or 'none'}")
    assert not loaded, f"forbidden modules loaded: {loaded}"
    counts = {s: sum(1 for c in CHECKS if c["status"] == s) for s in ("PASS", "FAIL", "WARN", "INFO")}
    maxd = {c["id"]: c["max_abs_diff"] for c in CHECKS if "max_abs_diff" in c}
    out = {"schema": "pnx-independent-verification-v1",
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "study": "results/pcrl_penalty_no_erasure_v1 (focused method study: ordinary penalty without erasure)",
           "verifier_script": "results/pcrl_penalty_no_erasure_v1/verification/replay_pnx.py",
           "forbidden_imports": list(FORBIDDEN), "import_guard": True,
           "inputs": {"private_inputs": "~/PCRL_eval_cache_private/jcv_v1/inputs/adult_jcv.npz", "sha256": INPUT_SHA},
           "selection_lock_commit": SEL_COMMIT,
           "checks": sorted(CHECKS, key=lambda c: c["id"]),
           "primary": C.rows["primary"] if hasattr(C, "rows") else [],
           "secondary": C.rows["secondary"] if hasattr(C, "rows") else [],
           "claims": getattr(C, "claims", {}),
           "last_step_replay": getattr(C, "last_step", []),
           "summary": {"counts": counts, "n_checks": len(CHECKS), "max_abs_diffs": maxd,
                       "failing": [c["id"] for c in CHECKS if c["status"] == "FAIL"],
                       "warnings": [c["id"] for c in CHECKS if c["status"] == "WARN"],
                       "runtime_s": round(time.time() - t0, 1)}}
    txt = json.dumps(out, indent=1, default=float)
    assert "/Users/" not in txt and str(HOME) not in txt, "absolute path leaked into the verification JSON"
    OUT.write_text(txt + "\n")
    print(json.dumps(out["summary"], indent=1, default=float))


if __name__ == "__main__":
    main()
