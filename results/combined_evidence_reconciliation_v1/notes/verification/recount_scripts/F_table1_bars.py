"""F3: AAAI Table 1 ('Seven of 42 published-method combinations pass') at AUC bars 0.52/0.55/0.60.

Combinations = 7 variants (LAFTR official, LAFTR reimpl, VFAE, DANN-scrub, LEACE, Obliviator, FARE)
x 3 cells x 2 tiers = 42. For each combination and bar we re-apply the stored selection logic
(baseline_gauntlet.py:438-456 tier_pick): a combination PASSES at bar b iff a stored 3-seed
certification point has representation tier-max <= b. If only a seed-0 sweep row is <= b and no
3-seed point was ever computed at that knob, the verdict is UNDETERMINED from stored data
(the certification was only run for points that cleared 0.55). If every stored sweep row is > b the
combination FAILS at b. A second view adds the output surface (the paper's stated rule:
'every attacker ... at or below the bar on both exposed surfaces', paper.tex:261-262).
No attacker is fit; only stored per-row AUCs are re-thresholded.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from F_common import BARS, dg_json, save

bg, i1 = dg_json("results/baseline_gauntlet.json")
lo, i2 = dg_json("results/laftr_official.json")
ob, i3 = dg_json("results/obliviator_gauntlet.json")
fa, i4 = dg_json("results/fare_gauntlet.json")
tt, i5 = dg_json("results/two_tier_certification.json")
inputs = [i1, i2, i3, i4, i5]
CELL = {"hmda/race/loan_decision": "easy", "hmda/race/loan_amount_band": "middle", "adult/sex/income": "hard"}

combos = []  # (method, cell, rows, certs list)
for c in bg["cells"]:
    for b in c["baselines"]:
        combos.append((b["baseline"] + (" (reimpl)" if b["baseline"] == "LAFTR" else ""), CELL[c["cell"]],
                       b["rows"], list(b["certs"].values()), c["clean_lift"]))
for c in lo["cells"]:
    combos.append(("LAFTR (official)", CELL[c["cell"]], c["rows"], list(c["certs"].values()), c["clean_lift"]))
for c in ob["cells"]:
    combos.append(("Obliviator", CELL[c["cell"]], c["rows"], list(c["certs"].values()), c["clean_lift"]))
for c in fa["cells"]:
    combos.append(("FARE", CELL[c["cell"]], c["rows"], list(c["certs"].values()), c["clean_lift"]))
assert len(combos) == 21


def rep_t(row, t):
    return row[f"tier{t}_max"]


def out_t(row, t):
    # sweep rows store out_xgb/out_mlp only (out_max); certs store out_tier1/2_max
    return row.get("out_max")


def verdict(rows, certs, t, b, both):
    cert_ok = [cc for cc in certs if cc[f"rep_tier{t}_max"] <= b and
               (not both or cc[f"out_tier{t}_max"] <= b)]
    if cert_ok:
        best = max(cert_ok, key=lambda cc: cc["lift_best"])
        return "pass", best["label"] if "label" in best else None, best["lift_best"]
    sweep_ok = [r for r in rows if rep_t(r, t) <= b and (not both or out_t(r, t) <= b)]
    if sweep_ok:
        return "undetermined", [r.get("label", r.get("tag")) for r in sweep_ok][:6], None
    return "fail", None, None


table = {}
for b in BARS:
    for both in (False, True):
        key = f"bar_{b:.2f}_{'both_surfaces' if both else 'representation_only'}"
        res, npass, nund = [], 0, 0
        for (m, cell, rows, certs, cl) in combos:
            for t in (1, 2):
                v, which, lift = verdict(rows, certs, t, b, both)
                npass += v == "pass"; nund += v == "undetermined"
                res.append({"method": m, "cell": cell, "tier": t, "verdict": v, "point": which,
                            "util_pct_of_clean_lift": (100 * lift / cl if lift is not None else None)})
        table[key] = {"n_combinations": len(res), "pass": npass, "undetermined": nund,
                      "fail": len(res) - npass - nund,
                      "passes": [(r["method"], r["cell"], r["tier"]) for r in res if r["verdict"] == "pass"],
                      "undetermined_list": [(r["method"], r["cell"], r["tier"], r["point"]) for r in res
                                            if r["verdict"] == "undetermined"],
                      "rows": res}

# closest approach per combination (min rep tier-max over all stored points)
closest = []
for (m, cell, rows, certs, cl) in combos:
    for t in (1, 2):
        vals = [rep_t(r, t) for r in rows] + [cc[f"rep_tier{t}_max"] for cc in certs]
        closest.append({"method": m, "cell": cell, "tier": t, "min_rep_tier_max": min(vals)})

# 'ours' rows (not part of the 42) for context: e2e full-rank sweep (seed 0) and 5-seed certified points
ours = []
for c in tt["e2e_cells"]:
    cell = CELL[c["cell"]]
    for t in (1, 2):
        pt = c[f"tier{t}"]
        means = {a: pt[f"{a}_mean"] for a in ("xgb", "mlp", "lora", "lrt")}
        t1 = max(means["xgb"], means["mlp"], means["lora"])
        tm = t1 if t == 1 else max(t1, means["lrt"])
        ps = pt["per_seed"]
        n = len(ps["xgb"])
        per_seed_t1 = [max(ps["xgb"][i], ps["mlp"][i], ps["lora"][i]) for i in range(n)]
        per_seed_tm = per_seed_t1 if t == 1 else [max(per_seed_t1[i], ps["lrt"][i]) for i in range(n)]
        ours.append({"cell": cell, "tier": t, "sigma": c["tier%d" % t].get("sigma"),
                     "rep_max_of_means": tm, "mean_of_per_seed_max": sum(per_seed_tm) / n,
                     "worst_seed": max(per_seed_tm),
                     "util_pct": 100 * pt["lift_mean"] / c["clean_lift"],
                     "passes_by_bar_max_of_means": {f"{b:.2f}": tm <= b for b in BARS},
                     "passes_by_bar_worst_seed": {f"{b:.2f}": max(per_seed_tm) <= b for b in BARS}})

out = {"inputs": inputs, "table1_counts": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                           for k, v in table.items()},
       "rows_by_view": {k: v["rows"] for k, v in table.items()},
       "closest_approach": closest, "ours_e2e_fullrank_certified_points": ours,
       "note": "Seed-0 sweep rows store only out_max = max(out_xgb, out_mlp); output LoRA/LRT exist only in 3-seed certs."}
p = save("F_table1_bars.json", out)
print(p)
for k, v in table.items():
    print(k, "pass", v["pass"], "undet", v["undetermined"], "fail", v["fail"], v["passes"], v["undetermined_list"])
for o in ours:
    print(o["cell"], o["tier"], round(o["rep_max_of_means"], 4), round(o["mean_of_per_seed_max"], 4), round(o["worst_seed"], 4), round(o["util_pct"], 1))
