"""Outer inference for the focused no-erasure study, from saved assessment predictions and SELECTION_LOCK.json only.

Same estimator as the predecessor: per-seed statistic (SEX AUC averaged over attacker seeds 0,1,2; race macro OvR over
supported classes; deployed accuracy), mean over encoder seeds 0,1,2; SE = sd (ddof 1) over B = 1999 paired multinomial
bootstrap replicates of assessment record groups (seed 20261015); interval = point +- z SE.
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m pnx.infer --selection-lock <SELECTION_LOCK.json>
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from jcv.infer import G, acc_stat, sub_stat
from stored_model_eval.bench_infer import class_auc, run
from stored_model_eval.pilot_infer import UnitBootstrap

from jcv import run as JR
from pnx import family as FAM
from pnx import run as R

SEEDS = (0, 1, 2)
BETAS = ("0.1", "1", "10")
LABELS = ["U", "E", "F", "F0"] + [f"{a}_b{b}" for a in ("JP", "PN", "LN") for b in BETAS]
VIEW_KEY = {("prim", "v1"): "P_v1", ("prim", "v2"): "P_v2", ("prim", "pair"): "P_pair",
            ("prob", "v1"): "P_p1", ("prob", "v2"): "P_p2", ("prob", "pair"): "P_ppair",
            ("hard", "v1"): "P_h1", ("hard", "v2"): "P_h2", ("hard", "pair"): "P_hpair"}


def label_of(unit_or_units):
    if isinstance(unit_or_units, list):
        return None
    u = unit_or_units
    if u.endswith("__U"):
        return "U"
    if u.endswith("__E"):
        return "E"
    arm = u.split("__")[2]
    b = u.split("__b")[-1]
    return f"{arm}_b{b}"


class Ctx:
    def __init__(self, SL):
        self.SL, self.g, self.preds = SL, G(), {}
        z0 = None
        for k in SEEDS:
            for lab in LABELS:
                z = np.load(R.U(f"outer__s{k}__{lab}") / "preds.npz")
                self.preds[(k, lab)] = {x: z[x] for x in z.files}
                z0 = z0 if z0 is not None else self.preds[(k, lab)]
                assert np.array_equal(z["assess_row_id"], z0["assess_row_id"]), "assessment rows differ"
        self.units, self.sex = z0["assess_unit"], z0["sex"]
        rows = z0["assess_row_id"]
        self.race_idx = np.searchsorted(rows, z0["race_row_id"])
        assert np.array_equal(rows[self.race_idx], z0["race_row_id"])
        self.race = z0["race"]
        D = R.load_D()
        c = JR.constants(D)
        self.const_correct = {j: (z0["y_income"] if j == 0 else z0["y_occ"]) == c[j] for j in (0, 1)}

    def lab(self, k, arm):
        """Frozen selection -> scored outer label (C* resolved per seed)."""
        S = self.SL["seeds"][str(k)]
        if arm == "C*":
            arm = S["comparator"]["arm"]
            if arm is None:
                return None
        if arm in ("F", "F0", "U"):
            return arm
        a = S["arms"][arm]
        return label_of(a["unit"])

    def rec(self, k, lab, fmt, view):
        p, key = self.preds[(k, lab)], VIEW_KEY[(fmt, view)]
        ids = [self.g.base_once(f"{k}|{lab}|{key}|{s}", f"auc|{k}|{lab}|{key}|{s}", class_auc(self.sex, p[key][s], 1))
               for s in range(3)]
        return self.g.add(f"R|{k}|{lab}|{fmt}|{view}", "mean", ids)

    def rrace(self, k, lab, view):
        P3 = self.preds[(k, lab)][f"Prace_{view}"]
        sup = sorted(set(self.race.tolist()))
        ids = []
        for s in range(3):
            cls = [self.g.base_once(f"{k}|{lab}|race|{view}|{s}|{c}", f"raceauc|{k}|{lab}|{view}|{s}|{c}",
                                    sub_stat(class_auc(self.race, P3[s], c), self.race_idx)) for c in sup]
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
            pn = ctx.lab(k, "PN")
            if e["kind"] in ("coalition", "local"):
                ref = ctx.lab(k, e["ref"])
                if ref is None:
                    per = None
                    break
                if e["kind"] == "coalition":
                    per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, ref, "prim", "pair"), ctx.rec(k, pn, "prim", "pair")]))
                else:
                    per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, pn, "prim", e["view"]), ctx.rec(k, ref, "prim", e["view"])]))
            elif e["kind"] == "acc":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.acc(k, pn, e["task"]), ctx.acc(k, "U", e["task"])]))
            elif e["kind"] == "retain":
                per.append(g.add(f"{e['id']}|{k}", "lin", [ctx.acc(k, pn, e["task"]), ctx.acc(k, "U", e["task"]), ctx.const(e["task"])]))
            else:
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.acc(k, pn, e["task"]), ctx.const(e["task"])]))
        ids[e["id"]] = None if per is None else g.add(e["id"], "mean", per)
    for e in FAM.SECONDARY:
        per = []
        for k in SEEDS:
            if e["kind"] == "erase_acc":
                per += [g.add(f"{e['id']}|{k}|{b}", "diff", [ctx.acc(k, f"PN_b{b}", e["task"]), ctx.acc(k, f"JP_b{b}", e["task"])]) for b in BETAS]
            elif e["kind"] == "erase_rec":
                per += [g.add(f"{e['id']}|{k}|{b}", "diff", [ctx.rec(k, f"PN_b{b}", "prim", e["view"]), ctx.rec(k, f"JP_b{b}", "prim", e["view"])]) for b in BETAS]
            elif e["kind"] == "out":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, ctx.lab(k, "LN"), e["fmt"], e["view"]), ctx.rec(k, ctx.lab(k, "PN"), e["fmt"], e["view"])]))
            elif e["kind"] == "race":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rrace(k, ctx.lab(k, "LN"), e["view"]), ctx.rrace(k, ctx.lab(k, "PN"), e["view"])]))
            elif e["kind"] == "rec":
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, ctx.lab(k, e["a"]), "prim", "pair"), ctx.rec(k, ctx.lab(k, "PN"), "prim", "pair")]))
            elif e["kind"] == "synergy":
                lab = ctx.lab(k, e["arm"])
                mx = g.add(f"max|{k}|{lab}", "max", [ctx.rec(k, lab, "prim", "v1"), ctx.rec(k, lab, "prim", "v2")])
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, lab, "prim", "pair"), mx]))
            else:
                b = e["beta"]
                per.append(g.add(f"{e['id']}|{k}", "diff", [ctx.rec(k, f"LN_b{b}", "prim", "pair"), ctx.rec(k, f"PN_b{b}", "prim", "pair")]))
        ids[e["id"]] = g.add(e["id"], "mean", per)
    levels = {}
    for k in SEEDS:
        for lab in LABELS:
            for fmt in ("prim", "prob", "hard"):
                for v in ("v1", "v2", "pair"):
                    levels[f"R|{k}|{lab}|{fmt}|{v}"] = ctx.rec(k, lab, fmt, v)
            for v in ("v1", "v2", "pair"):
                levels[f"Rrace|{k}|{lab}|{v}"] = ctx.rrace(k, lab, v)
            for j in (0, 1):
                levels[f"acc|{k}|{lab}|{j}"] = ctx.acc(k, lab, j)
    for lab in LABELS:
        for v in ("v1", "v2", "pair"):
            levels[f"Rmean|{lab}|{v}"] = g.add(f"Rmean|{lab}|{v}", "mean", [ctx.rec(k, lab, "prim", v) for k in SEEDS])
        for j in (0, 1):
            levels[f"accmean|{lab}|{j}"] = g.add(f"accmean|{lab}|{j}", "mean", [ctx.acc(k, lab, j) for k in SEEDS])
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
    ap.add_argument("--selection-lock", required=True)
    a = ap.parse_args(argv)
    SL = json.loads(Path(a.selection_lock).read_text())
    ctx = Ctx(SL)
    ids, levels = build(ctx)
    boot = UnitBootstrap(ctx.units, FAM.B, FAM.BOOT_SEED, 250)
    want = [i for i in ids.values() if i] + list(levels.values())
    pts, reps = run(ctx.g, boot, list(dict.fromkeys(want)))
    out = {"primary": [], "secondary": [], "levels": {}, "B": FAM.B, "seed": FAM.BOOT_SEED,
           "resampling_unit": "assessment record group (de-duplicated record; not households)",
           "n_assessment": int(len(ctx.units)), "n_groups": int(len(np.unique(ctx.units)))}
    for fam, key, z in ((FAM.PRIMARY, "primary", FAM.Z_PRIMARY), (FAM.SECONDARY, "secondary", FAM.Z_SECONDARY)):
        for e in fam:
            sid = ids[e["id"]]
            if sid is None:
                out[key].append({**e, "point": None, "decision": "NOT_ESTABLISHED", "reason": "no comparator on some seed"})
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
    st = {k: SL["seeds"][str(k)] for k in SEEDS}
    pn_nonzero = all(st[k]["arms"]["PN"]["status"] == "NOMINEE" for k in SEEDS)
    ln_nonzero = all(st[k]["arms"]["LN"]["status"] == "NOMINEE" for k in SEEDS)
    has_c = all(st[k]["comparator"]["arm"] is not None for k in SEEDS)
    valid_ref = all(st[k]["valid_reference"] for k in SEEDS)
    pn_nonzero, ln_nonzero, has_c = pn_nonzero and valid_ref, ln_nonzero and valid_ref, has_c and valid_ref
    for claim, nom in (("A", pn_nonzero and ln_nonzero), ("B", pn_nonzero and has_c)):
        cl = [e for e in out["primary"] if e["claim"] == claim]
        allpass = all(e["decision"] == "PASS" for e in cl)
        out[f"claim{claim}"] = {"all_nine_pass": allpass, "status_requirements_met": nom, "valid_reference_all_seeds": valid_ref,
                                "comparator_per_seed": {k: {x: st[k]["comparator"].get(x) for x in ("arm", "status", "unit")} for k in SEEDS},
                                "decision": "PASS" if (allpass and nom) else "NOT_ESTABLISHED",
                                "clauses_passing": sum(e["decision"] == "PASS" for e in cl)}
    (R.RUN / "inference.json").write_text(json.dumps(out, indent=1, default=float))
    R.PKG.mkdir(parents=True, exist_ok=True)
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
