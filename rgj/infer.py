"""Inference from saved DEVELOPMENT_ASSESSMENT predictions and EVALUATION_LOCK.json only (no refits, no selection).

Per encoder seed: recovery = SEX AUC (score = P(SEX=1), fixed orientation) of the inner-AUC-selected final attacker,
averaged over attacker seeds 0, 1, 2; race = macro one-vs-rest AUC over supported classes; proper-loss recovery
LLR = 1 - mean(-log P_ce[s]) / mean(-log prior[s]) (CE-selected attacker; prior = DEFENSE_FIT SEX prior); accuracy =
deployed hard decisions. Endpoint = mean over seeds 0, 1, 2 (C* resolved per seed from the lock). SE = sd (ddof 1) over
B = 1999 paired multinomial bootstrap replicates of exact-record groups (seed 20261004; the same draws for every
quantity, arm and seed); interval = point +- z SE with the family's two-sided Bonferroni z. Claims: rgj.family.claim_decision.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m rgj.infer --evaluation-lock results/pcrl_refreshed_guarded_joint_v1/EVALUATION_LOCK.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from jcv.infer import G, acc_stat, sub_stat
from stored_model_eval.bench_infer import class_auc, run
from stored_model_eval.pilot_infer import UnitBootstrap, _ratio_skill

from rgj import data as DA
from rgj import family as FAM
from rgj import run as R

SEEDS = (0, 1, 2)
KEY = {("prim", "v1"): "v1", ("prim", "v2"): "v2", ("prim", "pair"): "pair", ("prob", "v1"): "p1",
       ("prob", "v2"): "p2", ("prob", "pair"): "ppair", ("hard", "v1"): "h1", ("hard", "v2"): "h2",
       ("hard", "pair"): "hpair"}


def safe(label):
    return label.replace("*", "star").replace("/", "_").replace(" ", "_")


class Ctx:
    def __init__(self, EL):
        self.EL, self.g, self.preds = EL, G(), {}
        z0 = None
        for k in SEEDS:
            for lab in EL["seeds"][str(k)]["score"]:
                z = np.load(R.U(f"outer__s{k}__{safe(lab)}") / "preds.npz")
                self.preds[(k, lab)] = {x: z[x] for x in z.files}
                z0 = z0 if z0 is not None else self.preds[(k, lab)]
                assert np.array_equal(z["assess_row_id"], z0["assess_row_id"]), "assessment rows differ"
        self.units, self.sex, self.rows = z0["assess_unit"], z0["sex"], z0["assess_row_id"]
        prior = np.asarray(EL["sex_prior_defense_fit"], float)
        self.ll_const = -np.log(prior[self.sex])
        D = DA.load()
        tr = D["idx"]["DEFENSE_FIT"]
        c = {j: int(np.argmax(np.bincount(D["y"][t][tr]))) for j, t in enumerate(("income", "occupation_group"))}
        self.const_correct = {j: (z0["y_income"] if j == 0 else z0["y_occ"]) == c[j] for j in (0, 1)}

    def lab(self, k, arm):
        S = self.EL["seeds"][str(k)]
        if arm == "C*":
            return S.get("comparator", {}).get("arm")
        return arm

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
        per = []
        for k in SEEDS:
            jg = "J-G"
            if e["kind"] in ("coalition", "local"):
                ref = ctx.lab(k, e["ref"])
                if ref is None:
                    per = None
                    break
                if e["kind"] == "coalition":
                    per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, ref, "prim", "pair"), ctx.rec(k, jg, "prim", "pair")]))
                else:
                    per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, jg, "prim", e["view"]), ctx.rec(k, ref, "prim", e["view"])]))
            elif e["kind"] == "acc":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.acc(k, jg, e["task"]), ctx.acc(k, "U", e["task"])]))
            elif e["kind"] == "retain":
                per.append(g.add(f"{e['id']}|{k}", "lin", [ctx.acc(k, jg, e["task"]), ctx.acc(k, "U", e["task"]), ctx.const(e["task"])]))
            else:
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.acc(k, jg, e["task"]), ctx.const(e["task"])]))
        ids[e["id"]] = None if per is None else g.add(e["id"], "mean", per)
    for e in FAM.SECONDARY:
        per = []
        for k in SEEDS:
            if e["kind"] == "out":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, e["a"], e["fmt"], e["view"]), ctx.rec(k, e["b"], e["fmt"], e["view"])]))
            elif e["kind"] == "race":
                a, b = ctx.rrace(k, e["a"], e["view"]), ctx.rrace(k, e["b"], e["view"])
                if a is None or b is None:
                    per = None
                    break
                per.append(g.add(f"{e['id']}|{k}", "diff", [a, b]))
            elif e["kind"] == "logloss":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.llr(k, e["a"], e["view"]), ctx.llr(k, e["b"], e["view"])]))
            elif e["kind"] in ("fixed", "rec"):
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, e["a"], "prim", e["view"]), ctx.rec(k, e["b"], "prim", e["view"])]))
            elif e["kind"] == "synergy":
                lab = e["arm"]
                mx = g.add(f"max|{k}|{lab}", "max", [ctx.rec(k, lab, "prim", "v1"), ctx.rec(k, lab, "prim", "v2")])
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, lab, "prim", "pair"), mx]))
        ids[e["id"]] = None if per is None else g.add(e["id"], "mean", per)
    levels = {}
    labels = list(ctx.EL["seeds"]["0"]["score"])
    for k in SEEDS:
        for lab in labels:
            for fmt in ("prim", "prob", "hard"):
                for v in ("v1", "v2", "pair"):
                    levels[f"R|{k}|{lab}|{fmt}|{v}"] = ctx.rec(k, lab, fmt, v)
            for v in ("v1", "v2", "pair"):
                rr = ctx.rrace(k, lab, v)
                if rr is not None:
                    levels[f"Rrace|{k}|{lab}|{v}"] = rr
                levels[f"LLR|{k}|{lab}|{v}"] = ctx.llr(k, lab, v)
            for j in (0, 1):
                levels[f"acc|{k}|{lab}|{j}"] = ctx.acc(k, lab, j)
    for lab in labels:
        for v in ("v1", "v2", "pair"):
            levels[f"Rmean|{lab}|{v}"] = g.add(f"Rmean|{lab}|{v}", "mean", [ctx.rec(k, lab, "prim", v) for k in SEEDS])
        for j in (0, 1):
            levels[f"accmean|{lab}|{j}"] = g.add(f"accmean|{lab}|{j}", "mean", [ctx.acc(k, lab, j) for k in SEEDS])
    for j in (0, 1):
        levels[f"const|{j}"] = ctx.const(j)
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
           "resampling_unit": "DEVELOPMENT_ASSESSMENT exact-record group", "n_assessment": int(len(ctx.units)),
           "n_groups": int(len(np.unique(ctx.units)))}
    for fam, key, z in ((FAM.PRIMARY, "primary", FAM.Z_PRIMARY), (FAM.SECONDARY, "secondary", FAM.Z_SECONDARY)):
        for e in fam:
            sid = ids[e["id"]]
            if sid is None:
                out[key].append({**e, "point": None, "decision": "NOT_ESTABLISHED" if key == "primary" else "NOT_ESTIMABLE",
                                 "reason": "comparator or supported class absent on some seed"})
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
    st = {}
    for k in SEEDS:
        S = EL["seeds"][str(k)]
        st[k] = {"valid_reference": S.get("valid_reference", False), "J-G": S.get("status", {}).get("J-G"),
                 "L-G": S.get("status", {}).get("L-G"), "C*": S.get("comparator", {}).get("arm")}
    for claim in ("A", "B"):
        dec = {e["id"]: e["decision"] for e in out["primary"] if e["claim"] == claim}
        out[f"claim{claim}"] = {**FAM.claim_decision(claim, dec, st), "seed_status": st}
    (R.RUN / "inference.json").write_text(json.dumps(out, indent=1, default=float))
    R.PKG.mkdir(parents=True, exist_ok=True)
    for key, fn in (("primary", "PRIMARY_ENDPOINTS.csv"), ("secondary", "SECONDARY_ENDPOINTS.csv")):
        cols = ["id", "stat", "target", "side", "point", "se", "lower", "upper", "z", "decision", "alias_of", "claim"]
        with open(R.PKG / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for e in out[key]:
                w.writerow({c: (f"{e[c]:.6f}" if isinstance(e.get(c), float) else e.get(c)) for c in cols})
    with open(R.PKG / "RAW_LEVELS.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["quantity", "seed", "label", "detail", "point", "se"])
        for nm, v in sorted(out["levels"].items()):
            parts = nm.split("|")
            seed = parts[1] if parts[0] in ("R", "Rrace", "LLR", "acc") else "mean"
            lab = parts[2] if seed != "mean" else (parts[1] if len(parts) > 1 else "")
            det = "|".join(parts[3:] if seed != "mean" else parts[2:])
            w.writerow([parts[0], seed, lab, det, f"{v['point']:.6f}", f"{v['se']:.6f}"])
    return out


if __name__ == "__main__":
    main()
