#!/opt/homebrew/bin/python3
"""Synthetic validation of replay.py (ROLE 3), before any real-data replay.

Builds, in a scratch directory, a private-input directory and a run_v1 directory that follow
notes/FROZEN_DESIGN.md (labels with record keys/units/roles, cache, releases, manifests,
units/<id>/{preds.npz, supported.json, fit_records.json, COMPLETE.json}, run_v1/inputs/{task_labels_v1.npz,
manifest_v2_*}; full-K probability columns indexed by class label; surfaces rep/outputs/repPLUSoutputs;
LO_P Laplace alpha=1; s_prior_fit/t_prior_fit) with known ground truth:

  * direct signal          income_prediction__sex       (P2 must be RECOVERY_OUTSIDE_SCOPE_ESTABLISHED, C2 and C3)
  * null                   education_assessment__income (P2 BELOW_BAR_ESTABLISHED; held-out G1 < 0 -> GENERALISES)
  * multiclass + unsupported classes   *__race          (classes 0 and 3 unsupported; macro over {1,2,4})
  * not estimable          employment_analysis__marital_status (1 supported class -> P1/P2 NE, C5_NE)
  * borderline             education_assessment__sex    (true AUC = 0.55 -> UNRESOLVED expected)
  * ties                   income_prediction__race      (probabilities rounded to 2 dp)
  * duplicates             400 record units carry two identical rows (some land in assessment)
  * negative held-out R²   analytic case pred = prior - 0.5 (Y - prior) -> R² = 1 - 1.5² = -1.25 exactly
  * 18 noise units with release seeds (seed aggregation = mean over seeds within replicate)

A synthetic "runner" report (PRIMARY_ENDPOINTS / DECOMPOSITION / UTILITY / SUPPORT_COVERAGE CSVs) is
produced by an implementation inside this file that is independent of replay.py: sklearn AUC/log-loss,
sklearn Ridge, and an index-resampling cluster bootstrap with scipy rankdata and a different RNG.
The tables use the runner's real layouts (headers read from the real report directory on 2026-10-02).

Mutation runs check that replay.py detects: wrong supported.json, an assessment ID swapped for a fit row,
a missing unit, a perturbed point estimate, a flipped decision, and a G1_pred that is not the ridge fit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay as R  # noqa: E402  (the module under test; not a repo import)

DEFAULT_WORK = Path("/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/synth_replay")

PURPOSES = ["income_prediction", "employment_analysis", "education_assessment"]
TASK_K = {"income_prediction": 2, "employment_analysis": 6, "education_assessment": 4}
SIGNAL = {  # per untreated unit: attacker signal strength a (see make_P)
    "income_prediction__race": 0.8, "income_prediction__sex": 2.0,
    "employment_analysis__race": 0.3, "employment_analysis__age_group": 0.5,
    "employment_analysis__marital_status": 1.0,
    "education_assessment__sex": float(norm.ppf(0.55)),   # binary true AUC = Phi(a) = 0.55
    "education_assessment__race": 0.0, "education_assessment__income": 0.0,
}


def own_role(key):  # second, separately typed copy of the frozen role rule
    h = hashlib.sha256(f"pilot-roles-v1|{key}".encode()).hexdigest()
    u = int(h[:8], 16) / 4294967296.0
    return "attacker_fit" if u < 0.5 else ("attacker_val" if u < 0.65 else "assessment")


def fmt(x):
    return repr(float(x))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- data generation
def generate(work, seed=7):
    rng = np.random.default_rng(seed)
    inp, run = work / "inputs", work / "run_v1"
    if work.exists():
        shutil.rmtree(work)
    (inp / "cache").mkdir(parents=True)
    (inp / "releases").mkdir()
    (run / "units").mkdir(parents=True)
    (run / "inputs").mkdir()          # actual layout: task labels + v2 manifests under run_v1/inputs/

    n_units, n_dup = 6000, 400
    keys = np.array([hashlib.sha1(f"synthetic-{i}".encode()).hexdigest()[:20] for i in range(n_units)])
    unit_ids = rng.permutation(np.arange(100000, 100000 + n_units))
    dup = rng.choice(n_units, n_dup, replace=False)
    src = np.concatenate([np.arange(n_units), dup])
    src = src[rng.permutation(len(src))]
    N = len(src)
    row_id = np.arange(N, dtype=np.int64)

    def per_unit(probs):
        return rng.choice(len(probs), size=n_units, p=probs)
    attrs_u = {
        "sex": per_unit([0.33, 0.67]),
        "race": per_unit([0.012, 0.07, 0.10, 0.008, 0.81]),
        "age_group": per_unit([0.2, 0.47, 0.25, 0.08]),
        "marital_status": per_unit([0.98, 0.02]),
        "income": per_unit([0.75, 0.25]),
    }
    d = 16
    base = rng.normal(size=(n_units, d))
    rep_u = {
        "rep_p0": base + 1.2 * np.outer(attrs_u["sex"] - 0.67, np.eye(d)[0])
        + 0.6 * np.outer((attrs_u["race"] == 2).astype(float), np.eye(d)[1]),
        "rep_p1": rng.normal(size=(n_units, d)) + 0.25 * np.outer(attrs_u["age_group"] == 1, np.eye(d)[2]),
        "rep_p2": rng.normal(size=(n_units, d)),
    }
    heads = {p: rng.normal(size=(d, TASK_K[p])) * 0.5 for p in PURPOSES}
    task_u = {
        "income_prediction": attrs_u["income"],
        "employment_analysis": per_unit([0.3, 0.2, 0.2, 0.15, 0.145, 0.005]),   # class 5 unsupported
        "education_assessment": per_unit([0.4, 0.3, 0.2, 0.1]),
    }
    lab = {"row_id": row_id, "unit": unit_ids[src].astype(np.int64), "record_key": keys[src],
           "role": np.array([own_role(k) for k in keys[src]])}
    for a, v in attrs_u.items():
        lab[a] = v[src].astype(np.int64)
    np.savez(inp / "labels.npz", **lab)   # D1 #5: labels.npz unchanged; task labels in a separate file
    tl = {"row_id": row_id[::-1].copy()}   # deliberately stored in a different row order
    for p in PURPOSES:
        lab[f"task_{p}"] = task_u[p][src].astype(np.int64)
        tl[f"y_task_{p}"] = lab[f"task_{p}"][::-1].copy()
    np.savez(run / "inputs" / "task_labels_v1.npz", **tl)
    cache = {"row_id": row_id}
    for k, v in rep_u.items():
        cache[k] = v[src].astype(np.float32)
    for i, p in enumerate(PURPOSES):
        cache[f"logits_{p}"] = (cache[f"rep_p{i}"] @ heads[p]).astype(np.float32)
    np.savez(inp / "cache" / "synth_test.npz", **cache)

    def manifest(uid, rep_file_key, rep_key, attr, purpose, release=None):
        m = {"schema": "stored_model_eval.manifest/v1", "synthetic": True, "min_class_support": 100,
             "files": {"fwd": {"path": str(inp / "cache" / "synth_test.npz"), "sha256": "synthetic"},
                       "lab": {"path": str(inp / "labels.npz"), "sha256": "synthetic"}},
             "arrays": {"representations": {"file": rep_file_key, "key": rep_key, "ids": "row_id"},
                        "outputs": {"file": "fwd", "key": f"logits_{purpose}", "ids": "row_id"},
                        "labels": {"file": "lab", "key": attr, "ids": "row_id"}}}
        if release:
            m["files"]["rel"] = {"path": release, "sha256": "synthetic"}
            m["release"] = {"kind": "gaussian_noise"}
        (run / "inputs" / f"manifest_v2_{uid}.json").write_text(json.dumps(m, indent=1))

    roles = lab["role"]
    fm, vm, am = roles == "attacker_fit", roles == "attacker_val", roles == "assessment"
    ai = np.flatnonzero(am)                       # assessment rows in row_id order
    truth = {"units": {}, "N": int(N), "n_assess": int(am.sum()),
             "n_assess_clusters": int(len(np.unique(lab["unit"][am])))}

    def ridge_pred(H, Y, lam):
        r = Ridge(alpha=lam, fit_intercept=True, solver="cholesky").fit(H[fm], Y[fm])
        return r.predict(H[am])

    def make_P(y, cols, a, gen, ties=False):
        Z = gen.normal(size=(len(y), len(cols))) + a * (y[:, None] == np.array(cols)[None, :])
        P = softmax(Z)
        if ties:
            P = np.round(P, 2)
            P = P / P.sum(axis=1, keepdims=True)
        return P

    untreated_outputs = {}

    def write_unit(uid, purpose, attr, H, U1_logits, P_dict, extra_fit=None):
        y = lab[attr]
        K = int(max(y)) + 1
        fit_classes = list(range(K))        # actual format: columns = class labels 0..K-1
        Y = np.eye(K)[y]
        g1 = ridge_pred(H.astype(np.float64), Y, 1e-6)
        lam2 = 0.02 * np.trace(np.cov(H[fm].T)) * fm.sum()
        g2 = ridge_pred(H.astype(np.float64), Y, lam2)
        prior = Y[fm].mean(axis=0)
        t = lab[f"task_{purpose}"]
        LO = np.zeros((am.sum(), K))
        for tv in np.unique(t[am]):          # Laplace alpha=1 on attacker_fit
            sel = fm & (t == tv)
            LO[t[am] == tv] = [(np.sum(y[sel] == c) + 1.0) / (sel.sum() + K) for c in range(K)]
        W2 = np.random.default_rng(int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)).normal(size=(H.shape[1], TASK_K[purpose])) * 0.3
        U2 = softmax(H[am].astype(np.float64) @ (heads[purpose] + W2))
        gr = np.random.default_rng(int(hashlib.sha256(("rho" + uid).encode()).hexdigest()[:8], 16))
        ru = gr.normal(size=am.sum())
        rv = 0.4 * ru + gr.normal(size=am.sum())          # population rho^2 = 0.16/1.16
        preds = {"RHO_u": ru, "RHO_v": rv}
        preds.update({"assess_row_id": row_id[am], "assess_unit": lab["unit"][am], "y_s": y[am],
                 "y_task": t[am], "G1_pred": g1, "G2_pred": g2, "G1_prior": prior, "G2_prior": prior,
                 "LO_P": LO, "U1_logits": U1_logits, "U2_P": U2, "s_prior_fit": prior,
                 "t_prior_fit": np.array([np.mean(t[fm] == c) for c in range(TASK_K[purpose])])})
        for k, v in P_dict.items():
            preds[k] = v
        ud = run / "units" / uid
        ud.mkdir(parents=True)
        np.savez(ud / "preds.npz", **preds)
        counts = {str(c): {r: int(np.sum((y == c) & (roles == r))) for r in R.SUPPORT_MIN}
                  for c in sorted(np.unique(y).tolist())}
        supc = [int(c) for c, cc in counts.items() if all(cc[r] >= R.SUPPORT_MIN[r] for r in R.SUPPORT_MIN)]
        supp = [[a_, b_] for i_, a_ in enumerate(supc) for b_ in supc[i_ + 1:]]
        (ud / "supported.json").write_text(json.dumps(
            {"unit": uid, "counts": counts, "supported_classes": supc, "supported_pairs": supp}, indent=1))
        (ud / "fit_records.json").write_text(json.dumps({"synthetic": True, **(extra_fit or {})}))
        (ud / "models").mkdir()
        (ud / "COMPLETE.json").write_text(json.dumps({"unit": uid, "files": {
            n_: hashlib.sha256((ud / n_).read_bytes()).hexdigest()
            for n_ in ("preds.npz", "supported.json", "fit_records.json")}}, indent=1))
        truth["units"][uid] = {"fit_classes": fit_classes, "supported_classes": supc}
        return preds

    gen = np.random.default_rng(seed + 1)
    for p, a in R.UNTREATED_PAIRS:
        uid = f"{p}__{a}"
        pi = PURPOSES.index(p)
        H = cache[f"rep_p{pi}"]
        y = lab[a]
        cols = list(range(int(max(y)) + 1))
        s = SIGNAL[uid]
        ties = uid == "income_prediction__race"
        yA = y[am]
        Pd = {"P__rep__L": make_P(yA, cols, 0.8 * s, gen, ties),
              "P__rep__GBT": make_P(yA, cols, s, gen, ties),
              "P__rep__MLP": make_P(yA, cols, 0.9 * s, gen, ties)}
        Pd["P__rep__NL"] = Pd["P__rep__GBT"].copy()
        for surf, f_ in (("outputs", 0.6), ("repPLUSoutputs", 1.05)):
            Pd[f"P__{surf}__L"] = make_P(yA, cols, 0.8 * f_ * s, gen, ties)
            Pd[f"P__{surf}__GBT"] = make_P(yA, cols, 0.95 * f_ * s, gen, ties)
            Pd[f"P__{surf}__MLP"] = make_P(yA, cols, f_ * s, gen, ties)
            Pd[f"P__{surf}__NL"] = Pd[f"P__{surf}__MLP"].copy()
        if uid == "income_prediction__sex":
            untreated_outputs = {k: v for k, v in Pd.items() if k.startswith("P__outputs__")}
        write_unit(uid, p, a, H, cache[f"logits_{p}"][am], Pd)
        manifest(uid, "fwd", f"rep_p{pi}", a, p)
    for s in R.SIGMAS:
        for k in R.SEEDS:
            uid = f"income_prediction__sex__p0_sigma{s}_seed{k}"
            sig = float(s)
            g = np.random.default_rng(1000 + int(sig * 100) * 10 + k)
            Hn = (cache["rep_p0"] + g.normal(scale=sig, size=cache["rep_p0"].shape)).astype(np.float32)
            relp = inp / "releases" / f"p0_sigma{s}_seed{k}.npz"
            np.savez(relp, row_id=row_id, rep=Hn)
            y = lab["sex"]
            yA = y[am]
            a_s = 2.0 / (1.0 + sig)
            Pd = {"P__rep__L": make_P(yA, [0, 1], 0.8 * a_s, g), "P__rep__GBT": make_P(yA, [0, 1], a_s, g),
                  "P__rep__MLP": make_P(yA, [0, 1], 0.7 * a_s, g),
                  "P__repPLUSoutputs__NL": make_P(yA, [0, 1], max(a_s, 1.2), g),
                  "P__rep__LRT_A2": make_P(yA, [0, 1], 0.9 * a_s, g),
                  "P__rep__LRT_A4": make_P(yA, [0, 1], 1.3 * a_s, g)}
            Pd["P__rep__NL"] = Pd["P__rep__GBT"].copy()
            Pd.update({k_: v_.copy() for k_, v_ in untreated_outputs.items()})   # reused outputs surface
            write_unit(uid, "income_prediction", "sex", Hn, (Hn[am] @ heads["income_prediction"]), Pd,
                       {"release_contract": {"noise": "gaussian", "sigma": sig, "persistent": True}})
            manifest(uid, "rel", "rep", "sex", "income_prediction", str(relp))
    return inp, run, truth


# ---------------------------------------------------------------- independent reference ("synthetic runner")
def ref_macro_auc(y, P, cols, supc):
    return float(np.mean([roc_auc_score(y == c, P[:, cols.index(c)]) for c in supc]))


def mw_auc(score, pos):
    r = rankdata(score)
    n1 = pos.sum()
    n0 = len(pos) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ll12(y, P, labels):   # frozen convention: clip probabilities at 1e-12 (EFFECTIVE_PROTOCOL log_loss_clip)
    idx = np.searchsorted(np.asarray(labels), y)
    return float(np.mean(-np.log(np.clip(np.asarray(P, dtype=np.float64)[np.arange(len(y)), idx], 1e-12, 1 - 1e-12))))


def ref_r2(Y, pred, prior):
    return 1.0 - np.sum((Y - pred) ** 2) / np.sum((Y - prior[None, :]) ** 2)


def build_reference_report(inp, run, report, truth, B=20000, B_expl=2000):
    """Independent 'synthetic runner': writes the seven runner tables in the real layouts (headers as read
    from results/combined_stored_model_pilot_v1 on 2026-10-02). Point values by sklearn/numpy; intervals by an
    index-resampling cluster bootstrap with RandomState(777) and scipy rankdata (not replay.py's code)."""
    lab = dict(np.load(inp / "labels.npz"))
    roles = lab["role"]
    fm, am = roles == "attacker_fit", roles == "assessment"
    tl = dict(np.load(run / "inputs" / "task_labels_v1.npz"))
    order = np.argsort(tl["row_id"])
    task_all = {p: tl[f"y_task_{p}"][order] for p in PURPOSES}
    task_sup = {p: [int(c) for c in np.unique(t_all)
                    if all(np.sum((t_all == c) & (roles == r_)) >= R.SUPPORT_MIN[r_] for r_ in R.SUPPORT_MIN)]
                for p, t_all in task_all.items()}
    units = sorted(p.name for p in (run / "units").iterdir())
    alpha = 0.05 / 16
    first = dict(np.load(run / "units" / units[0] / "preds.npz"))
    au = first["assess_unit"]
    uniq = np.unique(au)
    members = [np.flatnonzero(au == u) for u in uniq]
    rs = np.random.RandomState(777)
    draws = [np.concatenate([members[j] for j in rs.randint(0, len(uniq), len(uniq))]) for _ in range(B)]
    dex = draws[:B_expl]
    cache = dict(np.load(inp / "cache" / "synth_test.npz"))

    def boot(fn, ds):
        return np.array([fn(ix) for ix in ds])

    def ci90(v):
        return np.quantile(v, [0.05, 0.95])

    def macro(y, P, cols, supc, ix=None):
        ix = np.arange(len(y)) if ix is None else ix
        return float(np.mean([mw_auc(P[ix, cols.index(c)], y[ix] == c) for c in supc]))

    def r2fn(Y, pred, prior):
        res = ((Y - pred) ** 2).sum(1)
        tot = ((Y - prior[None, :]) ** 2).sum(1)
        return lambda ix: 1 - res[ix].sum() / tot[ix].sum()

    def mixed_r2(H, Y):   # own float32-centring/Gram, float64 solve
        Hf = H.astype(np.float32)
        Hc = Hf - Hf.mean(axis=0, keepdims=True)
        G = Hc.T @ Hc + 1e-6 * np.eye(Hc.shape[1])
        Zc = Y - Y.mean(0, keepdims=True)
        W = np.linalg.solve(G.astype(np.float64), (Hc.T @ Zc).astype(np.float64))
        return 1 - np.sum((Zc - Hc @ W) ** 2) / np.sum(Zc ** 2)

    def n1_cols(H, Yall):   # within-assessment in-sample native statistic
        Ha, Ya = H[am], Yall[am]
        r_ = Ridge(alpha=1e-6, solver="cholesky").fit(Ha.astype(np.float64), Ya)
        return {"N1_float64_raw": fmt(ref_r2(Ya, r_.predict(Ha.astype(np.float64)), Ya.mean(0))),
                "N1_mixed_raw": fmt(mixed_r2(Ha, Ya))}

    def dec(lo, hi, t, incl):
        return "ESTABLISHED_ABOVE" if lo > t else ("ESTABLISHED_BELOW" if (hi <= t if incl else hi < t) else "UNRESOLVED")

    rows_p, rows_e, rows_d, rows_u, rows_s, rows_n = [], [], [], [], [], []
    ref = {}
    n0 = {}
    for p, a in R.UNTREATED_PAIRS:
        uid = f"{p}__{a}"
        H = cache[f"rep_p{PURPOSES.index(p)}"]
        Yall = np.eye(int(lab[a].max()) + 1)[lab[a]]
        r = Ridge(alpha=1e-6, solver="cholesky").fit(H.astype(np.float64), Yall)
        n0[uid] = (ref_r2(Yall, r.predict(H.astype(np.float64)), Yall.mean(0)), mixed_r2(H, Yall))
    for p, a in R.UNTREATED_PAIRS:
        uid = f"{p}__{a}"
        pr = dict(np.load(run / "units" / uid / "preds.npz"))
        y = pr["y_s"]
        cols = truth["units"][uid]["fit_classes"]
        supc = truth["units"][uid]["supported_classes"]
        Y = np.eye(len(cols))[y]
        f_g1 = r2fn(Y, pr["G1_pred"], pr["G1_prior"])
        g1 = f_g1(np.arange(len(y)))
        ref[uid] = {"G1": g1, "N0": n0[uid][0]}
        c1 = n0[uid][1] > 0.05
        base = {"alpha_each": fmt(alpha), "B": B, "seed": 20261003, "n_ne_replicates": 0,
                "native_N0_category": "C1" if c1 else "historical check passes"}
        if len(supc) < 2:
            for fam in ("P1", "P2"):
                rows_p.append({"id": f"{fam}-{uid}", "unit": uid, "point": "", "lower": "", "upper": "",
                               "decision": "NE", "category": "C5", **base})
        else:
            lo, hi = np.quantile(boot(f_g1, draws), [alpha, 1 - alpha])
            d1 = dec(lo, hi, 0.05, True)
            rows_p.append({"id": f"P1-{uid}", "unit": uid, "point": fmt(g1), "lower": fmt(lo), "upper": fmt(hi),
                           "decision": d1, "category": {"UNRESOLVED": "C5", "ESTABLISHED_BELOW": "none (established below)"}
                           .get(d1, "C1" if c1 else "C2"), **base})
            P = pr["P__rep__NL"]
            a2 = macro(y, P, cols, supc)
            lo, hi = np.quantile(boot(lambda ix: macro(y, P, cols, supc, ix), draws), [alpha, 1 - alpha])
            d2 = dec(lo, hi, 0.55, False)
            rows_p.append({"id": f"P2-{uid}", "unit": uid, "point": fmt(a2), "lower": fmt(lo), "upper": fmt(hi),
                           "decision": d2, "category": {"UNRESOLVED": "C5", "ESTABLISHED_BELOW": "none (established below)"}
                           .get(d2, "C1 (native check fails as historically defined)" if c1 else "C3"), **base})
            ref[uid]["P2"] = a2
        # exploratory: G1 with tau grid, rep NL macro AUC with bars, LL quantities
        lo, hi = ci90(boot(f_g1, dex))
        rows_e.append({"id": f"{uid}|G1|r2", "unit": uid, "kind": "untreated", "quantity": "G1", "surface": "rep",
                       "recipe": "G1", "metric": "r2", "point": fmt(g1), "lower90": fmt(lo), "upper90": fmt(hi),
                       "decisions": json.dumps({f"tau={t}": dec(lo, hi, t, True) for t in (0.01, 0.02, 0.05, 0.1)}),
                       "boot_B": B_expl, "boot_seed": 20261002})
        # decomposition
        rows_d.append({"unit": uid, "step": "F0", "point": fmt(max(0.0, n0[uid][1])), "point_raw_mixed": fmt(n0[uid][1]),
                       "point_float64": fmt(max(0.0, n0[uid][0])), "historical_r2_onehot": "",
                       "category": "C1" if c1 else "historical check passes"})
        rows_d.append({"unit": uid, "step": "F1", "point": fmt(g1), "lower90": fmt(lo), "upper90": fmt(hi)})
        if len(supc) >= 2:
            yfit = lab[a][fm]
            prior = np.array([np.mean(yfit == c) for c in cols])
            for k_ in ("P__rep__NL", "P__rep__L"):
                Pk = pr[k_]
                v = boot(lambda ix: macro(y, Pk, cols, supc, ix), dex)
                lo, hi = ci90(v)
                pt = macro(y, Pk, cols, supc)
                rows_e.append({"id": f"{uid}|{k_}|macro_auc", "unit": uid, "kind": "untreated", "quantity": "recovery",
                               "surface": "rep", "recipe": k_.split("__")[-1], "metric": "macro_auc", "point": fmt(pt),
                               "lower90": fmt(lo), "upper90": fmt(hi), "boot_B": B_expl, "boot_seed": 20261002,
                               "decisions": json.dumps({f"{b:.2f}": dec(lo, hi, b, False) for b in (0.52, 0.55, 0.60)})})
                ll = ll12(y, Pk, cols)
                ll0 = ll12(y, np.tile(prior, (len(y), 1)), cols)
                for met, val in (("LLR_nats", ll0 - ll), ("LL_skill", 1 - ll / ll0)):
                    rows_e.append({"id": f"{uid}|{k_}|{met}", "unit": uid, "kind": "untreated", "quantity": "recovery",
                                   "surface": "rep", "recipe": k_.split("__")[-1], "metric": met, "point": fmt(val),
                                   "boot_B": B_expl, "boot_seed": 20261002})
            Fk = {"F2": pr["G1_pred"], "F3": pr["P__rep__L"], "F4": pr["P__rep__NL"], "F5": pr["P__outputs__NL"],
                  "F6": pr["P__repPLUSoutputs__NL"]}
            fv = {f_: boot(lambda ix, P_=P_: macro(y, P_, cols, supc, ix), dex) for f_, P_ in Fk.items()}
            for f_, P_ in Fk.items():
                lo, hi = ci90(fv[f_])
                row = {"unit": uid, "step": f_, "point": fmt(macro(y, P_, cols, supc)), "lower90": fmt(lo), "upper90": fmt(hi)}
                dn = {"F3": "F2", "F4": "F3", "F5": "F4", "F6": "F4"}.get(f_)
                if dn:
                    dl, dh = ci90(fv[f_] - fv[dn])
                    row.update({"delta_name": f"{f_}_minus_{dn}",
                                "delta_point": fmt(macro(y, P_, cols, supc) - macro(y, Fk[dn], cols, supc)),
                                "delta_lower90": fmt(dl), "delta_upper90": fmt(dh)})
                rows_d.append(row)
        rows_n.append({"unit": uid, "N0_status": "REPRODUCED", "N0_mixed_clamped": fmt(max(0.0, n0[uid][1])),
                       "N0_mixed_raw": fmt(n0[uid][1]), "N0_float64_clamped": fmt(max(0.0, n0[uid][0])),
                       "N0_rows": len(lab[a]),
                       **n1_cols(cache[f"rep_p{PURPOSES.index(p)}"], np.eye(int(lab[a].max()) + 1)[lab[a]])})
    # seed aggregate G1 (mean over seeds within replicate)
    for s in R.SIGMAS:
        fns, pts = [], []
        for k in R.SEEDS:
            pr = dict(np.load(run / "units" / f"income_prediction__sex__p0_sigma{s}_seed{k}" / "preds.npz"))
            Y = np.eye(2)[pr["y_s"]]
            f_ = r2fn(Y, pr["G1_pred"], pr["G1_prior"])
            fns.append(f_)
            pts.append(f_(np.arange(len(Y))))
        lo, hi = ci90(boot(lambda ix: np.mean([f_(ix) for f_ in fns]), dex))
        rows_e.append({"id": f"income_prediction__sex__p0_sigma{s}__seedmean|G1|r2",
                       "unit": f"income_prediction__sex__p0_sigma{s}__seedmean", "kind": "noise_seed_aggregate",
                       "quantity": "G1", "surface": "rep", "recipe": "G1", "metric": "r2", "point": fmt(np.mean(pts)),
                       "lower90": fmt(lo), "upper90": fmt(hi), "seed_sd": fmt(np.std(pts, ddof=1)),
                       "per_seed_points": json.dumps([float(x) for x in pts]), "boot_B": B_expl, "boot_seed": 20261002})
    refu = None
    for uid in units:
        pr = dict(np.load(run / "units" / uid / "preds.npz"))
        t = pr["y_task"]
        K = pr["U1_logits"].shape[1]
        purpose = uid.split("__")[0]
        tsup = task_sup[purpose]
        tfit = task_all[purpose][fm]
        maj = np.bincount(tfit).argmax()
        u1p = softmax(pr["U1_logits"].astype(np.float64))
        vals = {("Uconst", "accuracy"): float(np.mean(t == maj))}
        for name, P in (("U1", u1p), ("U2", pr["U2_P"])):
            vals[(name, "accuracy")] = accuracy_score(t, P.argmax(1))
            vals[(name, "log_loss")] = ll12(t, P, list(range(K)))
            vals[(name, "macro_auc")] = float(np.mean([roc_auc_score(t == c, P[:, c]) for c in tsup]))
            vals[(name, "macro_f1")] = f1_score(t, P.argmax(1), labels=tsup, average="macro", zero_division=0)
            vals[(f"{name}_lift", "accuracy_lift_over_constant")] = vals[(name, "accuracy")] - vals[("Uconst", "accuracy")]
        if uid == "income_prediction__sex":
            refu = vals
        for (kind, met), v in vals.items():
            row = {"unit": uid, "kind": kind, "metric": met, "point": fmt(v)}
            if "sigma" in uid and kind in ("U1", "U2"):
                row.update({"reference_unit": "income_prediction__sex", "diff_point": fmt(v - refu[(kind, met)])})
            rows_u.append(row)
        if "sigma" in uid:
            for name in ("U1", "U2"):
                k_ = (f"{name}_lift", "accuracy_lift_over_constant")
                rows_u.append({"unit": uid, "kind": f"{name}_normalised_lift", "metric": "normalised_accuracy_lift",
                               "point": fmt(vals[k_] / refu[k_])})
            Hn = dict(np.load(inp / "releases" / f"p0_{uid.split('__p0_')[1]}.npz"))["rep"]
            rows_n.append({"unit": uid, "N0_status": "unverified", **n1_cols(Hn, np.eye(2)[lab["sex"]])})
        a = uid.split("__")[1]
        for what, y_all, sup_ in (("sensitive", lab[a], truth["units"][uid]["supported_classes"]),
                                  ("task", task_all[purpose], tsup)):
            for c in sorted(np.unique(y_all).tolist()):
                cnt = {r_: int(np.sum((y_all == c) & (roles == r_))) for r_ in R.SUPPORT_MIN}
                rows_s.append({"unit": uid, "what": what, "class": c, **{f"n_{k}": v for k, v in cnt.items()},
                               "supported": c in sup_,
                               "unit_status": "ESTIMABLE" if len(truth["units"][uid]["supported_classes"]) >= 2 else "NE"})
    report.mkdir(parents=True, exist_ok=True)
    fam = [{"id": f"{f}-{p}__{a}", "unit": f"{p}__{a}", "statistic": "G1_r2" if f == "P1" else "rep__NL__macro_auc",
            "bar": 0.05 if f == "P1" else 0.55} for f in ("P1", "P2") for p, a in R.UNTREATED_PAIRS]
    (report / "PRIMARY_FAMILY.json").write_text(json.dumps(
        {"family": fam, "family_size": 16, "alpha_each": alpha, "B": B, "seed": 20261003, "quantile_method": "linear"}))
    for name, rows in (("PRIMARY_ENDPOINTS.csv", rows_p), ("EXPLORATORY_ENDPOINTS.csv", rows_e),
                       ("DECOMPOSITION.csv", rows_d), ("UTILITY.csv", rows_u), ("SUPPORT_COVERAGE.csv", rows_s),
                       ("NATIVE_CHECKS.csv", rows_n)):
        cols_ = []
        for r_ in rows:
            cols_ += [c for c in r_ if c not in cols_]
        with open(report / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols_)
            w.writeheader()
            w.writerows(rows)
    return ref


# ---------------------------------------------------------------- checks
class Checks:
    def __init__(self):
        self.rows = []

    def __call__(self, name, ok, detail=None):
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail is not None else ""), flush=True)


def unit_tests(C):
    rng = np.random.default_rng(3)
    # weighted AUC == brute-force pairwise == sklearn, with ties and integer weights
    s = np.round(rng.normal(size=300), 1)
    y = rng.random(300) < 0.4
    w = rng.integers(0, 3, size=300).astype(float)
    got = R.WAUC(s, y)(w[None, :])[0]
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    num = sum(w[i] * w[j] * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0) for i in pos for j in neg)
    brute = num / (w[pos].sum() * w[neg].sum())
    C("unit: weighted AUC with ties == brute-force pairwise", abs(got - brute) < 1e-12, f"{got:.12f} vs {brute:.12f}")
    sk = roc_auc_score(y, s, sample_weight=w)
    C("unit: weighted AUC == sklearn roc_auc_score(sample_weight)", abs(got - sk) < 1e-12, f"{got:.12f} vs {sk:.12f}")
    # cluster bootstrap: every unit duplicated -> SE of a mean is sqrt(p(1-p)/n_units), sqrt(2) x row bootstrap
    n_u = 2000
    x_u = (rng.random(n_u) < 0.3).astype(float)
    units = np.repeat(np.arange(n_u), 2)
    x = x_u[units]
    inv, n_cl = R.cluster_index(units)
    st = {"m": R.wmean(x)}
    bc = R.run_bootstrap(st, inv, n_cl, 4000, 11)["m"]
    inv_r, n_r = R.cluster_index(np.arange(len(x)))
    br = R.run_bootstrap(st, inv_r, n_r, 4000, 11)["m"]
    p = x_u.mean()
    se_true = np.sqrt(p * (1 - p) / n_u)
    C("unit: cluster bootstrap SE matches unit-level SE (duplicates collapse)",
      abs(bc.std() / se_true - 1) < 0.08, f"ratio={bc.std() / se_true:.3f}")
    C("unit: cluster SE ~ sqrt(2) x naive row SE when every unit is doubled",
      abs(bc.std() / br.std() - np.sqrt(2)) < 0.12, f"ratio={bc.std() / br.std():.3f}")
    # 90% percentile interval of a mean ~ normal theory
    lo, hi = np.quantile(bc, [0.05, 0.95])
    C("unit: 90% percentile interval half-width ~ 1.645 SE", abs((hi - lo) / 2 / (1.645 * se_true) - 1) < 0.1,
      f"{(hi - lo) / 2:.5f} vs {1.645 * se_true:.5f}")
    # MC SE of a far-tail percentile is positive and small
    mc = R.mc_se_quantile(bc, 0.05 / 16)
    C("unit: MC SE of alpha=0.05/16 percentile finite and < 0.5 SE", 0 < mc < 0.5 * se_true, f"{mc:.6f}")
    # MC SE estimate of a far-tail percentile is calibrated: spread of the bound over independent seeds
    lows, ses = [], []
    for sd in range(8):
        bb = R.run_bootstrap(st, inv, n_cl, 20000, 100 + sd)["m"]
        lows.append(np.quantile(bb, 0.05 / 16, method="linear"))
        ses.append(R.mc_se_quantile(bb, 0.05 / 16))
    ratio = np.std(lows, ddof=1) / np.mean(ses)
    C("unit: MC SE of the B=20000 alpha=0.05/16 bound matches seed-to-seed spread (ratio in [0.5, 2])",
      0.5 <= ratio <= 2.0, f"empirical sd={np.std(lows, ddof=1):.6f} mean mcse={np.mean(ses):.6f} ratio={ratio:.2f}")
    # analytic negative R²
    Y = np.eye(3)[rng.integers(0, 3, 500)]
    prior = np.array([0.2, 0.3, 0.5])
    pred = prior[None, :] - 0.5 * (Y - prior[None, :])
    res, tot = ((Y - pred) ** 2).sum(1), ((Y - prior) ** 2).sum(1)
    C("unit: analytic R² = 1 - 1.5^2 = -1.25 (unclamped)", abs(1 - res.sum() / tot.sum() + 1.25) < 1e-12)
    # role rule cross-check against this file's own copy
    ks = [hashlib.sha1(str(i).encode()).hexdigest()[:20] for i in range(2000)]
    C("unit: role rule matches independent copy", all(R.role_of_key(k) == own_role(k) for k in ks))


def run(work, b_prim, b_expl, b_ref):
    C = Checks()
    t0 = time.time()
    unit_tests(C)
    inp, rundir, truth = generate(work)
    report = work / "report"
    ref = build_reference_report(inp, rundir, report, truth, B=b_ref)
    print(f"[test] synthetic data + reference report built in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    payload, res = R.replay(rundir, inp, None, report, work / "IV_clean.json", b_expl=b_expl, b_prim=b_prim,
                            expected_role_counts=None, historical_json=None,
                            results_path=work / "replay_results_clean.json", verbose=True)
    t_clean = time.time() - t1
    items = payload["items"]
    fails = [i for i in items if i["status"] == "FAIL"]
    C("clean synthetic run: no FAIL items", not fails, [f"{i['check']}: {i['runner_value']} vs {i['replay_value']}"
                                                         for i in fails][:10])
    C("clean synthetic run: runner CSV comparisons were actually made",
      all(sum(i["scope"] == sc for i in items) >= 16 for sc in ("primary", "exploratory", "decomposition", "utility",
                                                                 "support", "native", "categories")),
      dict(__import__("collections").Counter(i["scope"] for i in items)))
    border = [i["check"] for i in items if i["status"] == "MC_BORDERLINE"]
    C("clean synthetic run: MC-borderline decisions listed (informational)", True, border)
    prim = {r["endpoint"]: r for r in res["primary"]}
    C("26 units / 16 endpoints", len(prim) == 16 and len(res["support"]) == 26)
    C("direct signal: P2 income_prediction__sex RECOVERY_OUTSIDE_SCOPE_ESTABLISHED",
      prim["P2-income_prediction__sex"]["decision"] == "RECOVERY_OUTSIDE_SCOPE_ESTABLISHED",
      prim["P2-income_prediction__sex"].get("lower"))
    C("direct signal: P1 income_prediction__sex FAILS_TO_GENERALISE (G1 lower > 0.05)",
      prim["P1-income_prediction__sex"]["decision"] == "FAILS_TO_GENERALISE", prim["P1-income_prediction__sex"]["estimate"])
    cs = res["categories"]["income_prediction__sex"]
    C("direct signal: C1 (N0 > 0.05) and C3; failing P1 NOT read as C2 when C1 holds (D1 #8)",
      {"C1", "C3"} <= set(cs["categories"]) and "C2" not in cs["categories"] and cs["notes"], cs)
    for args, want in (((0.01, 0.01, "FAILS_TO_GENERALISE", "BELOW_BAR_ESTABLISHED", None), ["historical_check_passes", "C2"]),
                       ((0.06, 0.06, "FAILS_TO_GENERALISE", "UNRESOLVED", None), ["C1", "C5_UNRESOLVED"]),
                       ((0.01, 0.01, "GENERALISES", "BELOW_BAR_ESTABLISHED", 0.60), ["historical_check_passes", "C3"]),
                       ((0.049, 0.051, "NE", "NE", None), ["historical_check_passes", "C1_float32_float64_disagree", "C5_NE"])):
        got = R.assign_categories(*args)[0]
        C(f"category rule {args} -> {want}", got == want, got)
    u1 = res["exploratory"]["income_prediction__sex|RHO1SQ_heldout"]["estimate"]
    prr = dict(np.load(rundir / "units" / "income_prediction__sex" / "preds.npz"))
    C("held-out rho1^2 from RHO_u/RHO_v == np.corrcoef^2", abs(u1 - np.corrcoef(prr["RHO_u"], prr["RHO_v"])[0, 1] ** 2) < 1e-12, u1)
    C("tau sensitivity grid reported for G1", set(res["tau_sensitivity_G1"]["income_prediction__sex"]) == {"0.01", "0.02", "0.05", "0.1"},
      res["tau_sensitivity_G1"]["income_prediction__sex"])
    C("unsupported task class excluded from utility AUC/macro-F1 (employment class 5)",
      res["exploratory"]["employment_analysis__race|U1|AUC"]["estimate"] != res["exploratory"]["employment_analysis__race|U1|AUC_allpresent"]["estimate"])
    C("normalised-lift flag computed from clean lift", "U1" in res["normalised_lift_flags"], res["normalised_lift_flags"])
    C("null: P2 education_assessment__income BELOW_BAR_ESTABLISHED",
      prim["P2-education_assessment__income"]["decision"] == "BELOW_BAR_ESTABLISHED",
      prim["P2-education_assessment__income"].get("upper"))
    g1n = prim["P1-education_assessment__income"]
    C("null: held-out G1 negative and GENERALISES", g1n["estimate"] < 0 and g1n["decision"] == "GENERALISES",
      g1n["estimate"])
    C("borderline (true AUC 0.55): P2 education_assessment__sex UNRESOLVED",
      prim["P2-education_assessment__sex"]["decision"] == "UNRESOLVED", prim["P2-education_assessment__sex"]["estimate"])
    ne = res["categories"]["employment_analysis__marital_status"]["categories"]
    C("NE unit: P1 and P2 NE and category C5_NE (not UNRESOLVED)",
      prim["P1-employment_analysis__marital_status"]["decision"] == "NE"
      and prim["P2-employment_analysis__marital_status"]["decision"] == "NE" and "C5_NE" in ne
      and "C5_UNRESOLVED" not in ne, ne)
    C("unsupported classes: race supported = {1,2,4}",
      res["support"]["income_prediction__race"]["supported_classes"] == [1, 2, 4],
      res["support"]["income_prediction__race"]["supported_classes"])
    # macro over supported only, against sklearn, for the tied-probability unit
    pr = dict(np.load(rundir / "units" / "income_prediction__race" / "preds.npz"))
    cols = truth["units"]["income_prediction__race"]["fit_classes"]
    skm = ref_macro_auc(pr["y_s"], pr["P__rep__NL"], cols, [1, 2, 4])
    mine = res["exploratory"]["income_prediction__race|P|rep|NL|AUC_macro"]["estimate"]
    C("ties + unsupported: macro AUC over supported classes == sklearn", abs(skm - mine) < 1e-12, f"{mine} vs {skm}")
    allm = ref_macro_auc(pr["y_s"], pr["P__rep__NL"], cols, cols)
    C("macro AUC differs from all-class macro (unsupported excluded)", abs(allm - mine) > 1e-6, f"{allm} vs {mine}")
    # worst pair against an explicit reference
    P = pr["P__rep__NL"]
    wp = []
    for j, k in ((1, 2), (1, 4), (2, 4)):
        rows = (pr["y_s"] == j) | (pr["y_s"] == k)
        pj, pk = P[rows, cols.index(j)], P[rows, cols.index(k)]
        wp.append(roc_auc_score(pr["y_s"][rows] == k, pk / (pj + pk)))   # protocol orientation
    got = res["exploratory"]["income_prediction__race|P|rep|NL|AUC_worst_pair"]["estimate"]
    C("worst-pair AUC (max over supported pairs; p_j/(p_j+p_k), rows of the two classes) == sklearn reference",
      abs(max(wp) - got) < 1e-12, f"{got} vs {max(wp)}")
    got_min = res["exploratory"]["income_prediction__race|P|rep|NL|AUC_min_pair"]["estimate"]
    C("min-pair AUC kept as a separate statistic", abs(min(wp) - got_min) < 1e-12)
    # log-loss reduction and Brier skill against sklearn / numpy, prior = attacker_fit frequencies
    lab = dict(np.load(inp / "labels.npz"))
    yfit = lab["sex"][lab["role"] == "attacker_fit"]
    prior = np.array([np.mean(yfit == 0), np.mean(yfit == 1)])
    pr = dict(np.load(rundir / "units" / "income_prediction__sex" / "preds.npz"))
    ys = pr["y_s"]
    ll, ll0 = log_loss(ys, pr["P__rep__NL"], labels=[0, 1]), log_loss(ys, np.tile(prior, (len(ys), 1)), labels=[0, 1])
    got = res["exploratory"]["income_prediction__sex|P|rep|NL|LL_skill"]["estimate"]
    C("LL_skill = 1 - LL/LL0 (LL0 = attacker_fit prior) == sklearn", abs(1 - ll / ll0 - got) < 1e-9, f"{got} vs {1 - ll / ll0}")
    got = res["exploratory"]["income_prediction__sex|P|rep|NL|LLR_nats"]["estimate"]
    C("LLR_nats = LL0 - LL == sklearn", abs(ll0 - ll - got) < 1e-9, f"{got} vs {ll0 - ll}")
    Yb = np.eye(2)[ys]
    bs = 1 - np.mean(((pr["P__rep__NL"] - Yb) ** 2).sum(1)) / np.mean(((prior - Yb) ** 2).sum(1))
    got = res["exploratory"]["income_prediction__sex|P|rep|NL|brier_skill"]["estimate"]
    C("Brier skill == numpy reference", abs(bs - got) < 1e-12, f"{got} vs {bs}")
    # G1 R² and N0 against the sklearn reference
    for uid in ("income_prediction__sex", "education_assessment__race"):
        got = res["exploratory"][f"{uid}|G1|R2"]["estimate"]
        C(f"G1 R² (SS_tot around saved prior, unclamped) == reference [{uid}]", abs(got - ref[uid]["G1"]) < 1e-12,
          f"{got} vs {ref[uid]['G1']}")
        C(f"N0 float64 == sklearn in-sample ridge [{uid}]", abs(res["native"][uid]["N0_float64"] - ref[uid]["N0"]) < 1e-9,
          f"{res['native'][uid]['N0_float64']} vs {ref[uid]['N0']}")
    # seed aggregation
    for s in ("0.25", "8"):
        q = "P|rep|NL|AUC_macro"
        vals = [res["exploratory"][f"income_prediction__sex__p0_sigma{s}_seed{k}|{q}"]["estimate"] for k in R.SEEDS]
        agg = res["exploratory"][f"noise_sigma{s}_seedmean|{q}"]
        C(f"seed aggregation sigma={s}: point == mean of seeds", abs(agg["estimate"] - np.mean(vals)) < 1e-12)
        widths = [res["exploratory"][f"income_prediction__sex__p0_sigma{s}_seed{k}|{q}"]["upper"]
                  - res["exploratory"][f"income_prediction__sex__p0_sigma{s}_seed{k}|{q}"]["lower"] for k in R.SEEDS]
        C(f"seed aggregation sigma={s}: within-replicate mean interval no wider than widest seed",
          agg["upper"] - agg["lower"] <= max(widths) + 1e-12)
    C("noise utility paired difference vs untreated present",
      "income_prediction__sex__p0_sigma1_seed0|DIFF_vs_untreated|U1|accuracy" in res["exploratory"])
    C("assessment duplicates present in synthetic data", truth["n_assess"] > truth["n_assess_clusters"],
      f"rows={truth['n_assess']} clusters={truth['n_assess_clusters']}")

    # ---------------- mutation runs (must be detected)
    def mutate(tag, fn, report_dir=None):
        mrun = work / f"run_mut_{tag}"
        if mrun.exists():
            shutil.rmtree(mrun)
        shutil.copytree(rundir, mrun)
        fn(mrun)
        p, r = R.replay(mrun, inp, None, report_dir, work / f"IV_mut_{tag}.json", b_expl=200, b_prim=400,
                        expected_role_counts=None, verbose=False)
        return p, r

    def edit_npz(path, **upd):
        d = dict(np.load(path))
        d.update(upd)
        np.savez(path, **d)

    def m_support(mrun):
        f = mrun / "units" / "employment_analysis__race" / "supported.json"
        j = json.loads(f.read_text())
        j["supported_classes"] = [0, 1, 2, 4]
        f.write_text(json.dumps(j))
    p, _ = mutate("support", m_support)
    C("mutation: wrong supported.json detected",
      any(i["status"] == "FAIL" and "employment_analysis__race: supported classes" in i["check"] for i in p["items"]))

    def m_ids(mrun):
        f = mrun / "units" / "income_prediction__race" / "preds.npz"
        d = dict(np.load(f))
        fit_row = int(np.flatnonzero(lab["role"] == "attacker_fit")[0])
        d["assess_row_id"] = d["assess_row_id"].copy()
        d["assess_row_id"][5] = fit_row
        np.savez(f, **d)
    p, _ = mutate("ids", m_ids)
    C("mutation: assessment ID replaced by attacker_fit row detected",
      any(i["status"] == "FAIL" and "income_prediction__race: assess_row_id set" in i["check"] for i in p["items"]))

    def m_missing(mrun):
        shutil.rmtree(mrun / "units" / "income_prediction__sex__p0_sigma2_seed1")
    p, _ = mutate("missing", m_missing)
    C("mutation: missing unit detected (unit IDs compared, not counts)",
      any(i["status"] == "FAIL" and "26 expected unit IDs" in i["check"] for i in p["items"]))

    def m_neg(mrun):
        f = mrun / "units" / "education_assessment__income" / "preds.npz"
        d = dict(np.load(f))
        Y = np.eye(2)[d["y_s"]]
        d["G1_pred"] = d["G1_prior"][None, :] - 0.5 * (Y - d["G1_prior"][None, :])
        np.savez(f, **d)
    p, r = mutate("negR2", m_neg)
    got = r["exploratory"]["education_assessment__income|G1|R2"]["estimate"]
    C("mutation: analytic negative held-out R² reported exactly -1.25 (unclamped)", abs(got + 1.25) < 1e-12, got)
    C("mutation: G1_pred that is not the ridge fit detected by closed-form re-derivation",
      any(i["status"] == "FAIL" and "education_assessment__income: saved G1_pred" in i["check"] for i in p["items"]))

    def m_tamper(mrun):
        f = mrun / "units" / "employment_analysis__age_group" / "fit_records.json"
        f.write_text(f.read_text().replace("true", "false"))
    p, _ = mutate("tamper", m_tamper)
    C("mutation: file changed after COMPLETE.json detected (sha256)",
      any(i["status"] == "FAIL" and "employment_analysis__age_group: COMPLETE.json" in i["check"] for i in p["items"]))
    C("v2 manifests found under run_v1/inputs (prefix stripped)",
      sum(1 for i in payload["items"] if i["check"].endswith("manifest used for closed-form check")
          and "manifest_v2_" in str(i["replay_value"])) == 26)

    mrep = work / "report_mut"
    if mrep.exists():
        shutil.rmtree(mrep)
    shutil.copytree(report, mrep)
    rows = list(csv.DictReader(open(mrep / "PRIMARY_ENDPOINTS.csv")))
    for row in rows:
        if row["id"] == "P2-income_prediction__sex":
            row["decision"] = "ESTABLISHED_BELOW"
        if row["id"] == "P1-education_assessment__race":
            row["point"] = fmt(float(row["point"]) + 1e-3)
    with open(mrep / "PRIMARY_ENDPOINTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p, _ = R.replay(rundir, inp, None, mrep, work / "IV_mut_report.json", b_expl=200, b_prim=2000,
                    expected_role_counts=None, verbose=False)
    C("mutation: flipped runner decision (far from bar) detected as FAIL",
      any(i["status"] == "FAIL" and i["check"] == "P2-income_prediction__sex decision" for i in p["items"]))
    C("mutation: point estimate perturbed by 1e-3 detected as FAIL",
      any(i["status"] == "FAIL" and i["check"] == "P1-education_assessment__race point estimate" for i in p["items"]))

    out = {"schema": "pcrl.cell_a.replay_synthetic_validation/v1",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "workdir": str(work), "B_primary": b_prim, "B_exploratory": b_expl, "B_reference_bootstrap": b_ref,
           "clean_replay_seconds": round(t_clean, 1), "clean_replay_summary": payload["summary"],
           "mc_borderline_items": border,
           "n_checks": len(C.rows), "n_pass": sum(r_["pass"] for r_ in C.rows), "checks": C.rows,
           "addendum": "FROZEN_DESIGN Addendum D1 applied (task_labels_v1.npz, LLR_nats/LL_skill, RHO_u/RHO_v, type-7 quantiles, C1 suppresses C2, tau grid, task-class support); actual saved format per coordinator 2026-10-02 (full-K columns, repPLUSoutputs, Laplace LO_P, priors, COMPLETE.json, run_v1/inputs)",
           "note": ("synthetic runner tables use the real report layouts (PRIMARY_ENDPOINTS, PRIMARY_FAMILY, "
                    "EXPLORATORY_ENDPOINTS, DECOMPOSITION, UTILITY, SUPPORT_COVERAGE, NATIVE_CHECKS) and are produced "
                    "by an implementation independent of replay.py")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(DEFAULT_WORK))
    ap.add_argument("--b-prim", type=int, default=R.B_PRIM)
    ap.add_argument("--b-expl", type=int, default=R.B_EXPL)
    ap.add_argument("--b-ref", type=int, default=20000)
    ap.add_argument("--out", default=str(HERE / "test_replay_synthetic_result.json"))
    a = ap.parse_args()
    out = run(Path(a.work), a.b_prim, a.b_expl, a.b_ref)
    Path(a.out).write_text(json.dumps(out, indent=1, default=R._jsonable))
    print(f"[test] {out['n_pass']}/{out['n_checks']} checks pass -> {a.out}")
    return 0 if out["n_pass"] == out["n_checks"] else 1


if __name__ == "__main__":
    sys.exit(main())
