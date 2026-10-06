"""Inference from saved OSF_DEVELOPMENT_ASSESSMENT predictions and EVALUATION_LOCK.json only (no refits, no selection).

Per encoder seed: recovery = SEX AUC (score = P(SEX=1), fixed orientation) of the inner-AUC-selected final attacker,
averaged over attacker seeds 0-2; accuracy = released decisions; true-label log loss = mean -log(clip(prob_y, 1e-12));
Brier = mean sum_k (prob_k - 1[y=k])^2 (the osf convention); const = OSF_DEFENSE_FIT majority class; U = the task-only
teacher's continuous deployed output (label "SRC|U"). Roles (J*, P*, T*, C_global, C_match) are GLOBAL configurations
resolved from the lock; a descriptive fallback is scored but its rows are DESCRIPTIVE_ONLY. Endpoint = mean over seeds
of the per-seed paired statistic. SE = sd (ddof 1) over B = 1999 paired multinomial bootstrap replicates of exact-record
groups (seed 20261007; the same draws for every statistic, arm and seed); interval = point +- z SE with
z = 3.1717657833516224. A primary slot with any nonfinite replicate is INVALID (never dropped).

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m dpc.infer --evaluation-lock results/pcrl_decision_preserving_compression_v1/EVALUATION_LOCK.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from jcv.infer import G, acc_stat
from stored_model_eval.bench_infer import class_auc, run
from stored_model_eval.pilot_infer import UnitBootstrap

from dpc import family as FAM
from dpc import run as R

SEEDS = (0, 1, 2)
ROLES = ("J*", "P*", "T*", "C_global", "C_match")
EPS = 1e-12


def safe(label):
    return label.replace("|", "_").replace("*", "star").replace("/", "_").replace(" ", "_")


def row_losses(prob, y):
    prob = np.asarray(prob, dtype=np.float64)
    ll = -np.log(np.clip(prob[np.arange(len(y)), y], EPS, 1.0))
    br = ((prob - np.eye(prob.shape[1])[y]) ** 2).sum(1)
    return ll, br


class Ctx:
    def __init__(self, EL):
        self.EL, self.g, self.preds = EL, G(), {}
        self.labels = list(EL["seeds"]["0"]["score"])
        z0 = None
        for k in SEEDS:
            assert list(EL["seeds"][str(k)]["score"]) == self.labels
            for lab in self.labels:
                z = np.load(R.U(f"outer__s{k}__{safe(lab)}") / "preds.npz")
                p = {x: z[x] for x in z.files}
                self.preds[(k, lab)] = p
                z0 = z0 if z0 is not None else p
                assert np.array_equal(p["assess_row_id"], z0["assess_row_id"]), "assessment rows differ"
                assert np.array_equal(p["assess_unit"], z0["assess_unit"]), "assessment groups differ"
        self.units, self.sex, self.rows = z0["assess_unit"], z0["sex"], z0["assess_row_id"]
        self.y = {0: z0["y_income"], 1: z0["y_occ"]}
        self.const = {j: int(z0["const_class"][j]) for j in (0, 1)}
        from dpc.eval_lock import prior_hash
        from dpc import data as DA
        assert prior_hash(DA.load()) == EL["sex_prior_defense_fit_sha256"], "fitting prior differs from the locked hash"

    def lab(self, x):
        if x in ROLES:
            return self.EL["resolved"].get(x)
        return x if x in self.labels else None

    def rec(self, k, lab, view, fam=None):
        key = f"P_auc_{view}" if fam is None else f"P_auc_{fam}_{view}"
        P3 = self.preds[(k, lab)][key]
        ids = [self.g.base_once(f"auc#{k}#{lab}#{fam}#{view}#{s}", f"auc#{k}#{lab}#{fam}#{view}#{s}",
                                class_auc(self.sex, P3[s], 1)) for s in range(3)]
        return self.g.add(f"R#{k}#{lab}#{fam}#{view}", "mean", ids)

    def has(self, k, lab, view, fam):
        return (f"P_auc_{view}" if fam is None else f"P_auc_{fam}_{view}") in self.preds[(k, lab)]

    def acc(self, k, lab, j):
        p = self.preds[(k, lab)]
        return self.g.base_once(f"acc#{k}#{lab}#{j}", f"acc#{k}#{lab}#{j}", acc_stat(p[f"hard{j + 1}"] == self.y[j]))

    def loss(self, k, lab, j, kind):
        p = self.preds[(k, lab)]
        ll, br = row_losses(p[f"prob{j + 1}"], self.y[j])
        return self.g.base_once(f"{kind}#{k}#{lab}#{j}", f"{kind}#{k}#{lab}#{j}", acc_stat(ll if kind == "ll" else br))

    def constacc(self, j):
        return self.g.base_once(f"const#{j}", f"const#{j}", acc_stat(self.y[j] == self.const[j]))


def build(ctx):
    g, ids = ctx.g, {}
    U = "SRC|U"
    for e in FAM.PRIMARY:
        nom = ctx.lab(e["nominee"])
        ref = ctx.lab(e["ref"]) if "ref" in e else None
        if nom is None or ("ref" in e and ref is None):
            ids[e["id"]] = None
            continue
        per = []
        for k in SEEDS:
            j = e.get("task")
            if e["kind"] == "coalition":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.rec(k, ref, "pair"), ctx.rec(k, nom, "pair")]))
            elif e["kind"] == "local":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.rec(k, nom, e["view"]), ctx.rec(k, ref, e["view"])]))
            elif e["kind"] == "acc":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.acc(k, nom, j), ctx.acc(k, U, j)]))
            elif e["kind"] == "logloss":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.loss(k, nom, j, "ll"), ctx.loss(k, U, j, "ll")]))
            elif e["kind"] == "brier":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.loss(k, nom, j, "br"), ctx.loss(k, U, j, "br")]))
            else:
                per.append(g.add(f"{e['id']}#{k}", "lin", [ctx.acc(k, nom, j), ctx.acc(k, U, j), ctx.constacc(j)]))
        ids[e["id"]] = g.add(e["id"], "mean", per)
    levels = {}
    fams = [None, "complete", "scores", "probs", "decisions", "features", "logits", "hard"]
    for lab in ctx.labels:
        for k in SEEDS:
            for fam in fams:
                for v in ("v1", "v2", "pair"):
                    if ctx.has(k, lab, v, fam):
                        levels[f"R#{k}#{lab}#{fam or 'primary'}#{v}"] = ctx.rec(k, lab, v, fam)
            for j in (0, 1):
                levels[f"acc#{k}#{lab}#{j}"] = ctx.acc(k, lab, j)
                levels[f"ll#{k}#{lab}#{j}"] = ctx.loss(k, lab, j, "ll")
                levels[f"br#{k}#{lab}#{j}"] = ctx.loss(k, lab, j, "br")
        for fam in fams:
            for v in ("v1", "v2", "pair"):
                if all(ctx.has(k, lab, v, fam) for k in SEEDS):
                    levels[f"Rmean#{lab}#{fam or 'primary'}#{v}"] = g.add(f"Rmean#{lab}#{fam}#{v}", "mean",
                                                                         [ctx.rec(k, lab, v, fam) for k in SEEDS])
        for j in (0, 1):
            for kind, fn in (("acc", lambda k: ctx.acc(k, lab, j)), ("ll", lambda k: ctx.loss(k, lab, j, "ll")),
                             ("br", lambda k: ctx.loss(k, lab, j, "br"))):
                levels[f"{kind}mean#{lab}#{j}"] = g.add(f"{kind}mean#{lab}#{j}", "mean", [fn(k) for k in SEEDS])
    for j in (0, 1):
        levels[f"const#{j}"] = ctx.constacc(j)
    return ids, levels


def decide(e, lo, hi):
    if e["side"] == "lower>":
        return "PASS" if lo > e["target"] else "NOT_ESTABLISHED"
    return "PASS" if hi < e["target"] else "NOT_ESTABLISHED"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-lock", required=True)
    a = ap.parse_args(argv)
    EL = json.loads(Path(a.evaluation_lock).read_text())
    ctx = Ctx(EL)
    ids, levels = build(ctx)
    boot = UnitBootstrap(ctx.units, FAM.B, FAM.BOOT_SEED, 250)
    want = [i for i in ids.values() if i] + list(levels.values())
    pts, reps = run(ctx.g, boot, list(dict.fromkeys(want)))
    z = FAM.Z_PRIMARY
    out = {"primary": [], "levels": {}, "B": FAM.B, "seed": FAM.BOOT_SEED, "z": z,
           "resampling_unit": "OSF_DEVELOPMENT_ASSESSMENT exact-record group", "n_assessment": int(len(ctx.units)),
           "n_groups": int(len(np.unique(ctx.units))), "resolved": EL["resolved"], "statuses": EL["statuses"]}
    invalid = False
    for e in FAM.PRIMARY:
        sid = ids[e["id"]]
        if sid is None:
            out["primary"].append({**e, "point": None, "z": z, "decision": "NOT_ESTABLISHED",
                                   "reason": "nominee or comparator absent"})
            continue
        r = reps[sid]
        nonfinite = int((~np.isfinite(r)).sum())
        if nonfinite or not np.isfinite(pts[sid]):
            invalid = True
            out["primary"].append({**e, "point": pts[sid], "z": z, "decision": "INVALID",
                                   "nonfinite_replicates": nonfinite})
            continue
        se = float(np.std(r, ddof=1))
        pt = pts[sid]
        lo, hi = pt - z * se, pt + z * se
        row = {**e, "point": pt, "se": se, "lower": lo, "upper": hi, "z": z, "decision": decide(e, lo, hi),
               "n_finite_replicates": int(len(r))}
        roles = [e["nominee"]] + ([e["ref"]] if "ref" in e else [])
        if any(EL["statuses"].get(x, {}).get("status") != "NOMINEE" for x in roles):
            row["decision_numeric"], row["decision"] = row["decision"], "DESCRIPTIVE_ONLY"
        out["primary"].append(row)
    for nm, sid in levels.items():
        r = reps[sid]
        out["levels"][nm] = {"point": pts[sid], "se": float(np.std(r[np.isfinite(r)], ddof=1)),
                             "nonfinite": int((~np.isfinite(r)).sum())}
    dec = {}
    for claim in FAM.CLAIMS:
        d = {e["id"]: e["decision"] for e in out["primary"] if e["claim"] == claim}
        dec[claim] = FAM.claim_decision(claim, d, EL["statuses"])
        out[f"claim{claim}"] = dec[claim]
    complete = bool(EL.get("U_valid", False)) and not invalid and not any(
        str(s.get("status", "")).startswith("INVALID") for s in EL["statuses"].values())
    out["complete"] = complete
    out["label"] = FAM.overall_label(dec, complete=complete)
    (R.RUN / "inference.json").write_text(json.dumps(out, indent=1, default=float))
    cols = ["id", "claim", "stat", "target", "side", "point", "se", "lower", "upper", "z", "decision",
            "decision_numeric", "alias_of"]
    with open(R.PKG / "PRIMARY_ENDPOINTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for e in out["primary"]:
            w.writerow({c: (f"{e[c]:.6f}" if isinstance(e.get(c), float) else e.get(c)) for c in cols})
    with open(R.PKG / "ALL_LEVELS.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["quantity", "seed", "label", "detail", "point", "se"])
        for nm, v in sorted(out["levels"].items()):
            parts = nm.split("#")
            if parts[0] in ("R", "acc", "ll", "br"):
                seed, lab, det = parts[1], parts[2], "|".join(parts[3:])
            elif parts[0] == "const":
                seed, lab, det = "", "const", parts[1]
            else:
                seed, lab, det = "mean", parts[1], "|".join(parts[2:])
            w.writerow([parts[0], seed, lab, det, f"{v['point']:.6f}", f"{v['se']:.6f}"])
    return out


if __name__ == "__main__":
    main()
