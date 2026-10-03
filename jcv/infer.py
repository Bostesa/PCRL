"""Outer inference from saved assessment predictions only (jcv.outer units + SELECTION_LOCK.json).

Recovery of an (arm, view) on one encoder seed = SEX AUC on assessment averaged over attacker seeds 0,1,2 (race: macro
one-vs-rest AUC over supported classes). Endpoint = mean over encoder seeds 0,1,2 of the per-seed statistic (C* is the
per-seed frozen comparator). SE = sd (ddof 1) over B = 1999 paired multinomial bootstrap replicates of assessment
record groups (unit = de-duplicated record; Adult has no households), the same draws for every quantity; interval =
point +- z SE, z = two-sided Bonferroni normal critical value of the family. 'lower>' passes iff lower > target;
'upper<' passes iff upper < target. A claim passes only if all nine clauses pass and every seed has nominees.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m jcv.infer --selection-lock <SELECTION_LOCK.json>
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from stored_model_eval.bench_infer import Graph, class_auc, run
from stored_model_eval.pilot_infer import UnitBootstrap

from jcv import family as FAM
from jcv import run as R

SEEDS = (0, 1, 2)
VIEW_KEY = {("prim", "v1"): "P_v1", ("prim", "v2"): "P_v2", ("prim", "pair"): "P_pair",
            ("prob", "v1"): "P_p1", ("prob", "v2"): "P_p2", ("prob", "pair"): "P_ppair",
            ("hard", "v1"): "P_h1", ("hard", "v2"): "P_h2", ("hard", "pair"): "P_hpair"}


class G(Graph):
    def evaluate(self, WT, bases, aggs):
        v = {s: np.asarray(self.base[s](WT), dtype=np.float64) for s in bases}
        for s in aggs:
            op, parts = self.agg[s]
            X = [v[p] for p in parts]
            if op == "mean":
                v[s] = np.mean(np.stack(X), axis=0)
            elif op == "diff":
                v[s] = X[0] - X[1]
            elif op == "max":
                v[s] = np.max(np.stack(X), axis=0)
            elif op == "lin":   # X0 - 0.8 X1 - 0.2 X2
                v[s] = X[0] - 0.8 * X[1] - 0.2 * X[2]
            else:
                raise ValueError(op)
        return v


def acc_stat(c):
    c = np.asarray(c, float)
    return lambda WT: (c @ WT) / WT.sum(0)


def sub_stat(fn, idx):
    return lambda WT: fn(WT[idx])


class Ctx:
    def __init__(self, SL):
        self.SL = SL
        self.g = G()
        self.preds = {}
        z0 = None
        for k in SEEDS:
            for arm in ("U", "E", "L", "J", "JP", "S12", "S21", "F", "F0"):
                z = np.load(R.U(f"outer__s{k}__{arm}") / "preds.npz")
                self.preds[(k, arm)] = {kk: z[kk] for kk in z.files}
                if z0 is None:
                    z0 = self.preds[(k, arm)]
                assert np.array_equal(z["assess_row_id"], z0["assess_row_id"]), "assessment rows differ"
        self.rows = z0["assess_row_id"]
        self.units = z0["assess_unit"]
        self.sex = z0["sex"]
        self.race_idx = np.searchsorted(self.rows, z0["race_row_id"])
        assert np.array_equal(self.rows[self.race_idx], z0["race_row_id"])
        self.race = z0["race"]
        D = R.DA.load()
        c = R.constants(D)
        self.const_correct = {j: (z0["y_income"] if j == 0 else z0["y_occ"]) == c[j] for j in (0, 1)}

    def arm_of(self, k, ref):
        if ref == "C*":
            return self.SL["seeds"][str(k)]["comparator"]["arm"]
        return ref

    def rec(self, k, arm, fmt, view):
        p = self.preds[(k, arm)]
        key = VIEW_KEY[(fmt, view)]
        ids = []
        for s in range(3):
            P = p[key][s]
            ids.append(self.g.base_once(f"{k}|{arm}|{key}|{s}", f"auc|{k}|{arm}|{key}|{s}", class_auc(self.sex, P, 1)))
        return self.g.add(f"R|{k}|{arm}|{fmt}|{view}", "mean", ids)

    def rrace(self, k, arm, view):
        p = self.preds[(k, arm)]
        P3 = p[f"Prace_{view}"]
        sup = sorted(set(self.race.tolist()))
        ids = []
        for s in range(3):
            cls = [self.g.base_once(f"{k}|{arm}|race|{view}|{s}|{c}", f"raceauc|{k}|{arm}|{view}|{s}|{c}",
                                    sub_stat(class_auc(self.race, P3[s], c), self.race_idx)) for c in sup]
            ids.append(self.g.add(f"Rr|{k}|{arm}|{view}|{s}", "mean", cls))
        return self.g.add(f"Rrace|{k}|{arm}|{view}", "mean", ids)

    def acc(self, k, arm, j):
        p = self.preds[(k, arm)]
        y = p["y_income"] if j == 0 else p["y_occ"]
        return self.g.base_once(f"acc|{k}|{arm}|{j}", f"acc|{k}|{arm}|{j}", acc_stat(p[f"hard{j + 1}"] == y))

    def const(self, j):
        return self.g.base_once(f"const|{j}", f"const|{j}", acc_stat(self.const_correct[j]))


def build(ctx):
    g = ctx.g
    ids = {}
    for e in FAM.PRIMARY:
        per = []
        for k in SEEDS:
            ref = ctx.arm_of(k, e.get("ref", "L")) if e["kind"] in ("coalition", "local") else None
            if e["kind"] in ("coalition", "local") and ref is None:
                per = None
                break
            if e["kind"] == "coalition":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, ref, "prim", "pair"), ctx.rec(k, "J", "prim", "pair")]))
            elif e["kind"] == "local":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, "J", "prim", e["view"]), ctx.rec(k, ref, "prim", e["view"])]))
            elif e["kind"] == "acc":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.acc(k, "J", e["task"]), ctx.acc(k, "U", e["task"])]))
            elif e["kind"] == "retain":
                per.append(g.add(f"{e['id']}|{k}", "lin", [ctx.acc(k, "J", e["task"]), ctx.acc(k, "U", e["task"]), ctx.const(e["task"])]))
            elif e["kind"] == "useful":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.acc(k, "J", e["task"]), ctx.const(e["task"])]))
        ids[e["id"]] = None if per is None else g.add(e["id"], "mean", per)
    for e in FAM.SECONDARY:
        per = []
        for k in SEEDS:
            if e["kind"] == "out":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, e["a"], e["fmt"], e["view"]), ctx.rec(k, e["b"], e["fmt"], e["view"])]))
            elif e["kind"] == "race":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rrace(k, e["a"], e["view"]), ctx.rrace(k, e["b"], e["view"])]))
            elif e["kind"] == "rec":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, e["a"], "prim", e["view"]), ctx.rec(k, e["b"], "prim", e["view"])]))
            elif e["kind"] == "accdiff":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.acc(k, e["a"], e["task"]), ctx.acc(k, e["b"], e["task"])]))
            elif e["kind"] == "synergy":
                mx = g.add(f"max|{k}|{e['arm']}", "max", [ctx.rec(k, e["arm"], "prim", "v1"), ctx.rec(k, e["arm"], "prim", "v2")])
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, e["arm"], "prim", "pair"), mx]))
        ids[e["id"]] = g.add(e["id"], "mean", per)
    levels = {}
    for k in SEEDS:
        for arm in ("U", "E", "L", "J", "JP", "S12", "S21", "F", "F0"):
            for fmt in ("prim", "prob", "hard"):
                for v in ("v1", "v2", "pair"):
                    levels[f"R|{k}|{arm}|{fmt}|{v}"] = ctx.rec(k, arm, fmt, v)
            for v in ("v1", "v2", "pair"):
                levels[f"Rrace|{k}|{arm}|{v}"] = ctx.rrace(k, arm, v)
            for j in (0, 1):
                levels[f"acc|{k}|{arm}|{j}"] = ctx.acc(k, arm, j)
    for arm in ("U", "E", "L", "J", "JP", "S12", "S21", "F", "F0"):
        for v in ("v1", "v2", "pair"):
            levels[f"Rmean|{arm}|{v}"] = g.add(f"Rmean|{arm}|{v}", "mean", [ctx.rec(k, arm, "prim", v) for k in SEEDS])
        for j in (0, 1):
            levels[f"accmean|{arm}|{j}"] = g.add(f"accmean|{arm}|{j}", "mean", [ctx.acc(k, arm, j) for k in SEEDS])
    return ids, levels


def decide(e, pt, lo, hi):
    if e["side"] == "lower>":
        return "PASS" if lo > e["target"] else "NOT_ESTABLISHED"
    return "PASS" if hi < e["target"] else "NOT_ESTABLISHED"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-lock", required=True)
    a = ap.parse_args(argv)
    SL = json.loads(Path(a.selection_lock).read_text())
    ctx = Ctx(SL)
    ids, levels = build(ctx)
    boot = UnitBootstrap(ctx.units, FAM.B, FAM.BOOT_SEED, 250)
    want = [i for i in ids.values() if i] + list(levels.values())
    pts, reps = run(ctx.g, boot, list(dict.fromkeys(want)))
    out = {"primary": [], "secondary": [], "levels": {}, "B": FAM.B, "seed": FAM.BOOT_SEED,
           "resampling_unit": "assessment record group (de-duplicated record)", "n_assessment": int(len(ctx.units)),
           "n_groups": int(len(np.unique(ctx.units)))}
    for fam, key, z in ((FAM.PRIMARY, "primary", FAM.Z_PRIMARY), (FAM.SECONDARY, "secondary", FAM.Z_SECONDARY)):
        for e in fam:
            sid = ids[e["id"]]
            if sid is None:
                out[key].append({**e, "point": None, "decision": "NOT_ESTABLISHED", "reason": "no comparator (C*) on some seed"})
                continue
            r = reps[sid][np.isfinite(reps[sid])]
            se = float(np.std(r, ddof=1))
            pt = pts[sid]
            lo, hi = pt - z * se, pt + z * se
            out[key].append({**e, "point": pt, "se": se, "lower": lo, "upper": hi, "z": z,
                             "decision": decide(e, pt, lo, hi), "n_finite_replicates": int(len(r))})
    for nm, sid in levels.items():
        r = reps[sid]
        out["levels"][nm] = {"point": pts[sid], "se": float(np.std(r[np.isfinite(r)], ddof=1))}
    # nominee status gates for the conjunctions
    st = {k: SL["seeds"][str(k)]["arms"] for k in SEEDS}
    nomJ = all(st[k]["J"]["status"] == "NOMINEE" for k in SEEDS)
    nomL = all(st[k]["L"]["status"] == "NOMINEE" for k in SEEDS)
    nomC = all(SL["seeds"][str(k)]["comparator"]["arm"] is not None for k in SEEDS)
    for claim, nom in ((1, nomJ and nomL), (2, nomJ and nomC)):
        cl = [e for e in out["primary"] if e["claim"] == claim]
        allpass = all(e["decision"] == "PASS" for e in cl)
        out[f"claim{claim}"] = {"all_nine_pass": allpass, "nominees_all_seeds": nom,
                                "decision": "PASS" if (allpass and nom) else "NOT_ESTABLISHED",
                                "clauses_passing": sum(e["decision"] == "PASS" for e in cl)}
    (R.RUN / "inference.json").write_text(json.dumps(out, indent=1, default=float))
    for key, fn in (("primary", "PRIMARY_ENDPOINTS.csv"), ("secondary", "SECONDARY_ENDPOINTS.csv")):
        cols = ["id", "stat", "target", "side", "point", "se", "lower", "upper", "z", "decision", "alias_of", "claim"]
        with open(R.PKG / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for e in out[key]:
                w.writerow({c: (f"{e[c]:.6f}" if isinstance(e.get(c), float) else e.get(c)) for c in cols})
    return out


if __name__ == "__main__":
    main()
