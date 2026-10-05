"""Inference from saved OSF_DEVELOPMENT_ASSESSMENT predictions and EVALUATION_LOCK.json only (no refits, no selection).

Per encoder seed: recovery = SEX AUC (score = P(SEX=1), fixed orientation) of the inner-AUC-selected final attacker,
averaged over attacker seeds 0, 1, 2; race = macro one-vs-rest AUC over supported classes; proper-loss recovery
LLR = 1 - mean(-log P_ce[s]) / mean(-log prior[s]) (CE-selected attacker; prior = OSF_DEFENSE_FIT SEX prior, checked
against the locked hash); accuracy = deployed hard decisions; const = OSF_DEFENSE_FIT majority class. Nominees and
comparators (N*, R*, L*, C*) are GLOBAL configurations resolved from the lock (a descriptive fallback is scored for
transparency but cannot pass). Endpoint = mean over seeds 0, 1, 2 of the per-seed paired statistic. SE = sd (ddof 1)
over B = 1999 paired multinomial bootstrap replicates of exact-record groups (seed 20261006; the same draws for every
quantity, arm and seed); interval = point +- z SE with the family's two-sided Bonferroni z (osf.family).

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m osf.infer --evaluation-lock results/pcrl_online_strength_frontier_v1/EVALUATION_LOCK.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from jcv.infer import G, acc_stat, sub_stat
from stored_model_eval.bench_infer import class_auc, run
from stored_model_eval.pilot_infer import UnitBootstrap, _ratio_skill

from osf import data as DA
from osf import family as FAM
from osf import run as R

SEEDS = (0, 1, 2)
KEY = {("prim", "v1"): "v1", ("prim", "v2"): "v2", ("prim", "pair"): "pair", ("prob", "v1"): "p1",
       ("prob", "v2"): "p2", ("prob", "pair"): "ppair", ("hard", "v1"): "h1", ("hard", "v2"): "h2",
       ("hard", "pair"): "hpair", ("feat", "v1"): "r1", ("feat", "v2"): "r2", ("feat", "pair"): "rpair",
       ("logit", "v1"): "c1", ("logit", "v2"): "c2", ("logit", "pair"): "cpair"}
ROLES = ("N*", "R*", "L*", "C*")


def safe(label):
    return label.replace("|", "_").replace("*", "star").replace("/", "_").replace(" ", "_")


class Ctx:
    def __init__(self, EL):
        self.EL, self.g, self.preds = EL, G(), {}
        z0 = None
        self.labels = list(EL["seeds"]["0"]["score"])
        for k in SEEDS:
            assert list(EL["seeds"][str(k)]["score"]) == self.labels
            for lab in self.labels:
                z = np.load(R.U(f"outer__s{k}__{safe(lab)}") / "preds.npz")
                self.preds[(k, lab)] = {x: z[x] for x in z.files}
                z0 = z0 if z0 is not None else self.preds[(k, lab)]
                assert np.array_equal(z["assess_row_id"], z0["assess_row_id"]), "assessment rows differ"
                assert np.array_equal(z["assess_unit"], z0["assess_unit"]), "assessment groups differ"
        self.units, self.sex, self.rows = z0["assess_unit"], z0["sex"], z0["assess_row_id"]
        D = DA.load()
        tr = D["idx"]["DEFENSE_FIT"]
        from osf.eval_lock import prior_hash
        assert prior_hash(D) == EL["sex_prior_defense_fit_sha256"], "fitting prior differs from the locked hash"
        prior = np.bincount(D["sex"][tr], minlength=2) / len(tr)
        self.ll_const = -np.log(prior[self.sex])
        c = {j: int(np.argmax(np.bincount(D["y"][t][tr]))) for j, t in enumerate(("income", "occupation_group"))}
        self.const_correct = {j: (z0["y_income"] if j == 0 else z0["y_occ"]) == c[j] for j in (0, 1)}

    def lab(self, arm):
        if arm in ROLES:
            return self.EL["resolved"].get(arm)
        return arm if arm in self.labels else None

    def has(self, lab, fmt, view):
        return f"P_auc_{KEY[(fmt, view)]}" in self.preds[(0, lab)]

    def rec(self, k, lab, fmt, view):
        P3 = self.preds[(k, lab)][f"P_auc_{KEY[(fmt, view)]}"]
        ids = [self.g.base_once(f"{k}|{lab}|{fmt}|{view}|{s}", f"auc|{k}|{lab}|{fmt}|{view}|{s}",
                                class_auc(self.sex, P3[s], 1)) for s in range(3)]
        return self.g.add(f"R|{k}|{lab}|{fmt}|{view}", "mean", ids)

    def llr(self, k, lab, view):
        P3 = self.preds[(k, lab)][f"P_ce_{KEY[('prim', view)]}"]
        ids = []
        for s in range(3):
            ll = -np.log(np.clip(P3[s][np.arange(len(self.sex)), self.sex], 1e-12, 1.0))
            ids.append(self.g.base_once(f"{k}|{lab}|llr|{view}|{s}", f"llr|{k}|{lab}|{view}|{s}",
                                        _ratio_skill(ll, self.ll_const)))
        return self.g.add(f"LLR|{k}|{lab}|{view}", "mean", ids)

    def rrace(self, k, lab, view):
        p = self.preds[(k, lab)]
        key = f"Prace_auc_{view}"
        if key not in p:
            return None
        pos, yr = p["race_pos"], p["race_y"]
        ids = []
        for s in range(3):
            cls = [self.g.base_once(f"{k}|{lab}|race|{view}|{s}|{c}", f"raceauc|{k}|{lab}|{view}|{s}|{c}",
                                    sub_stat(class_auc(yr, p[key][s], c), pos)) for c in range(p[key].shape[2])]
            ids.append(self.g.add(f"Rr|{k}|{lab}|{view}|{s}", "mean", cls))
        return self.g.add(f"Rrace|{k}|{lab}|{view}", "mean", ids)

    def acc(self, k, lab, j):
        p = self.preds[(k, lab)]
        y = p["y_income"] if j == 0 else p["y_occ"]
        return self.g.base_once(f"acc|{k}|{lab}|{j}", f"acc|{k}|{lab}|{j}", acc_stat(p[f"hard{j + 1}"] == y))

    def const(self, j):
        return self.g.base_once(f"const|{j}", f"const|{j}", acc_stat(self.const_correct[j]))


def build(ctx):
    g, ids = ctx.g, {}
    for e in FAM.PRIMARY:
        nom = ctx.lab(e["nominee"])
        ref = ctx.lab(e["ref"]) if "ref" in e else None
        if nom is None or ("ref" in e and ref is None):
            ids[e["id"]] = None
            continue
        per = []
        for k in SEEDS:
            if e["kind"] == "coalition":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, ref, "prim", "pair"), ctx.rec(k, nom, "prim", "pair")]))
            elif e["kind"] == "local":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, nom, "prim", e["view"]), ctx.rec(k, ref, "prim", e["view"])]))
            elif e["kind"] == "acc":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.acc(k, nom, e["task"]), ctx.acc(k, "U", e["task"])]))
            elif e["kind"] == "retain":
                per.append(g.add(f"{e['id']}|{k}", "lin", [ctx.acc(k, nom, e["task"]), ctx.acc(k, "U", e["task"]), ctx.const(e["task"])]))
            else:
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.acc(k, nom, e["task"]), ctx.const(e["task"])]))
        ids[e["id"]] = g.add(e["id"], "mean", per)
    for e in FAM.SECONDARY:
        labs = [ctx.lab(e[x]) for x in ("a", "b", "arm") if x in e]
        if any(x is None for x in labs):
            ids[e["id"]] = None
            continue
        per = []
        for k in SEEDS:
            if e["kind"] == "rec":
                a, b = labs
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, a, "prim", e["view"]), ctx.rec(k, b, "prim", e["view"])]))
            elif e["kind"] == "accdiff":
                a, b = labs
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.acc(k, a, e["task"]), ctx.acc(k, b, e["task"])]))
            elif e["kind"] == "logloss":
                a, b = labs
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.llr(k, a, e["view"]), ctx.llr(k, b, e["view"])]))
            elif e["kind"] == "synergy":
                lab = labs[0]
                mx = g.add(f"max|{k}|{lab}", "max", [ctx.rec(k, lab, "prim", "v1"), ctx.rec(k, lab, "prim", "v2")])
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, lab, "prim", "pair"), mx]))
        ids[e["id"]] = g.add(e["id"], "mean", per)
    levels = {}
    for k in SEEDS:
        for lab in ctx.labels:
            for fmt in ("prim", "prob", "hard", "feat", "logit"):
                for v in ("v1", "v2", "pair"):
                    if ctx.has(lab, fmt, v):
                        levels[f"R#{k}#{lab}#{fmt}#{v}"] = ctx.rec(k, lab, fmt, v)
            for v in ("v1", "v2", "pair"):
                rr = ctx.rrace(k, lab, v)
                if rr is not None:
                    levels[f"Rrace#{k}#{lab}#{v}"] = rr
                levels[f"LLR#{k}#{lab}#{v}"] = ctx.llr(k, lab, v)
            for j in (0, 1):
                levels[f"acc#{k}#{lab}#{j}"] = ctx.acc(k, lab, j)
    for lab in ctx.labels:
        for v in ("v1", "v2", "pair"):
            levels[f"Rmean#{lab}#{v}"] = g.add(f"Rmean|{lab}|{v}", "mean", [ctx.rec(k, lab, "prim", v) for k in SEEDS])
        for j in (0, 1):
            levels[f"accmean#{lab}#{j}"] = g.add(f"accmean|{lab}|{j}", "mean", [ctx.acc(k, lab, j) for k in SEEDS])
    for j in (0, 1):
        levels[f"const#{j}"] = ctx.const(j)
    return ids, levels


def decide(e, lo, hi):
    if e["side"] == "lower>":
        return "PASS" if lo > e["target"] else "NOT_ESTABLISHED"
    if e["side"] == "upper<":
        return "PASS" if hi < e["target"] else "NOT_ESTABLISHED"
    return "ABOVE" if lo > e["target"] else ("BELOW" if hi < e["target"] else "NOT_RESOLVED")


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
    out = {"primary": [], "secondary": [], "levels": {}, "B": FAM.B, "seed": FAM.BOOT_SEED,
           "resampling_unit": "OSF_DEVELOPMENT_ASSESSMENT exact-record group", "n_assessment": int(len(ctx.units)),
           "n_groups": int(len(np.unique(ctx.units))), "resolved": EL["resolved"], "statuses": EL["statuses"]}
    for fam, key, z in ((FAM.PRIMARY, "primary", FAM.Z_PRIMARY), (FAM.SECONDARY, "secondary", FAM.Z_SECONDARY)):
        for e in fam:
            sid = ids[e["id"]]
            if sid is None:
                out[key].append({**e, "point": None, "z": z,
                                 "decision": "NOT_ESTABLISHED" if key == "primary" else "NOT_ESTIMABLE",
                                 "reason": "nominee/comparator absent or configuration not in the locked bank"})
                continue
            r = reps[sid][np.isfinite(reps[sid])]
            se = float(np.std(r, ddof=1))
            pt = pts[sid]
            lo, hi = pt - z * se, pt + z * se
            out[key].append({**e, "point": pt, "se": se, "lower": lo, "upper": hi, "z": z, "decision": decide(e, lo, hi),
                             "n_finite_replicates": int(len(r))})
    for nm, sid in levels.items():
        r = reps[sid]
        out["levels"][nm] = {"point": pts[sid], "se": float(np.std(r[np.isfinite(r)], ddof=1))}
    for e in out["primary"]:               # review A7: clauses of a non-NOMINEE nominee/comparator are descriptive
        roles = [e.get("nominee")] + ([e["ref"]] if "ref" in e else [])
        if e["point"] is not None and any(EL["statuses"].get(x, {}).get("status") != "NOMINEE" for x in roles):
            e["decision_numeric"], e["decision"] = e["decision"], "DESCRIPTIVE_ONLY"
    dec = {}
    for claim in FAM.CLAIMS:
        d = {e["id"]: e["decision"] for e in out["primary"] if e["claim"] == claim}
        dec[claim] = FAM.claim_decision(claim, d, EL["statuses"])
        out[f"claim{claim}"] = dec[claim]
    complete = bool(EL.get("U_valid", False)) and not any(str(s.get("status", "")).startswith("INVALID")
                                                         for s in EL["statuses"].values())
    out["complete"] = complete
    out["label"] = FAM.overall_label(dec, complete=complete)          # review S2
    (R.RUN / "inference.json").write_text(json.dumps(out, indent=1, default=float))
    R.PKG.mkdir(parents=True, exist_ok=True)
    for key, fn in (("primary", "PRIMARY_ENDPOINTS.csv"), ("secondary", "SECONDARY_ENDPOINTS.csv")):
        cols = ["id", "stat", "target", "side", "point", "se", "lower", "upper", "z", "decision",
                "decision_numeric", "alias_of", "claim"]
        with open(R.PKG / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for e in out[key]:
                w.writerow({c: (f"{e[c]:.6f}" if isinstance(e.get(c), float) else e.get(c)) for c in cols})
    with open(R.PKG / "ALL_LEVELS.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["quantity", "seed", "label", "detail", "point", "se"])
        for nm, v in sorted(out["levels"].items()):
            parts = nm.split("#")
            if parts[0] in ("R", "Rrace", "LLR", "acc"):
                seed, lab, det = parts[1], parts[2], "|".join(parts[3:])
            elif parts[0] == "const":
                seed, lab, det = "", "const", parts[1]
            else:
                seed, lab, det = "mean", parts[1], "|".join(parts[2:])
            w.writerow([parts[0], seed, lab, det, f"{v['point']:.6f}", f"{v['se']:.6f}"])
    return out


if __name__ == "__main__":
    main()
