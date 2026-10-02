"""E1: in-sample vs held-out utility behind AAAI Table 1, and whether per-cell utility rankings change.

Stored utility conventions (code read at dg@956f5c8):
  lift_own  = deployed head's accuracy on ALL rows it was trained on, minus majority (diagnostic.py:205-206;
              LEACE: LR head fit and scored on all rows, baseline_gauntlet.py:309-323)            -> in-sample
  lift_lr   = fresh LR head fit on 75% of the released representation, scored on the other 25%
              (targeted_noise.py:108-120), encoder/eraser still fit on all rows                       -> head-held-out
  lift_best = max(lift_own, lift_lr) (baseline_gauntlet.py:380) -> the Table 1 utility for baselines
  ours e2e  = own head in-sample (two_tier_certification.json); out of partition only for 9 'ours' points
              (fresh_partition_generalization.json, encoder trained on one half, everything scored on the other).
Denominator: clean e2e lift in-sample (Table 1 caption convention). No held-out clean lift exists for the
baseline split; the fresh-partition clean lift is a different split and model.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from F_common import dg_json, save

bg, i1 = dg_json("results/baseline_gauntlet.json")
lo, i2 = dg_json("results/laftr_official.json")
ob, i3 = dg_json("results/obliviator_gauntlet.json")
fa, i4 = dg_json("results/fare_gauntlet.json")
tt, i5 = dg_json("results/two_tier_certification.json")
fp, i6 = dg_json("results/fresh_partition_generalization.json")
CELL = {"hmda/race/loan_decision": "easy", "hmda/race/loan_amount_band": "middle", "adult/sex/income": "hard"}


def point(t):
    if t["certified"]:
        c = t["cert"]
        return {"kind": "3-seed cert", "label": c.get("label"), "own": c["lift_own_mean"], "lr": c["lift_lr_mean"]}
    c = t["closest"]
    return {"kind": "seed-0 closest sweep row (fails)", "label": c.get("label", c.get("tag")),
            "own": c["lift_own"], "lr": c["lift_lr"]}


rows = []
def add(method, cell_name, cl, t1, t2):
    for tier, t in ((1, t1), (2, t2)):
        p = point(t)
        rows.append({"method": method, "cell": CELL[cell_name], "tier": tier, **p,
                     "util_own_pct": 100 * p["own"] / cl, "util_lr_heldout_pct": 100 * p["lr"] / cl,
                     "util_table1_pct": 100 * max(p["own"], p["lr"]) / cl,
                     "table1_convention_uses": "own(in-sample)" if p["own"] >= p["lr"] else "LR(head-held-out)",
                     "certified": t["certified"]})

for c in bg["cells"]:
    for b in c["baselines"]:
        add(b["baseline"] + (" (reimpl)" if b["baseline"] == "LAFTR" else ""), c["cell"], c["clean_lift"], b["tier1"], b["tier2"])
for c in lo["cells"]:
    add("LAFTR (official)", c["cell"], c["clean_lift"], c["tier1"], c["tier2"])
for c in ob["cells"]:
    add("Obliviator", c["cell"], c["clean_lift"], c["tier1"], c["tier2"])
for c in fa["cells"]:
    add("FARE", c["cell"], c["clean_lift"], c["tier1"], c["tier2"])

# ours: in-partition (in-sample head) vs out-of-partition
ours = []
for c in fp["cells"]:
    for k, lab in (("fullrank_t1", "Ours full-rank T1"), ("fullrank_t2", "Ours full-rank T2"), ("subspace", "Ours subspace T1")):
        ours.append({"method": lab, "cell": CELL[c["cell"]], "util_in_partition_pct": c[k]["stored_inpartition"]["util_pct"],
                     "util_out_of_partition_pct": c[k]["utility_kept_eval_pct"],
                     "clean_lift_in_sample": c["clean_lift_rep_insample"], "clean_lift_eval": c["clean_lift_eval"]})
posthoc = [{"cell": pc["cell"].split()[0], "winner_t1": pc["winner_tier1"]["label"],
            "util_lr_heldout_pct": 100 * pc["winner_tier1"]["lift"] / pc["clean_lift"],
            "note": "post-hoc arm utility = LR head held-out (utility_through), clean reference = sigma=0 frozen P"}
           for pc in tt["posthoc_cells"]]

# ranking change: per (cell, tier), order methods by Table-1 utility vs by LR-held-out utility
rank_changes = {}
for cell in ("easy", "middle", "hard"):
    for tier in (1, 2):
        sub = [r for r in rows if r["cell"] == cell and r["tier"] == tier]
        a = [r["method"] for r in sorted(sub, key=lambda r: -r["util_table1_pct"])]
        b = [r["method"] for r in sorted(sub, key=lambda r: -r["util_lr_heldout_pct"])]
        c_ = [r["method"] for r in sorted(sub, key=lambda r: -r["util_own_pct"])]
        rank_changes[f"{cell}_T{tier}"] = {"order_table1": a, "order_lr_heldout": b, "order_own_insample": c_,
                                           "changed_table1_vs_heldout": a != b}
# among PASSING rows at 0.55 plus ours, per cell at T1: does the winner change held-out?
passers = {}
for cell in ("easy", "middle", "hard"):
    cand = [(r["method"], r["util_table1_pct"], r["util_lr_heldout_pct"]) for r in rows
            if r["cell"] == cell and r["tier"] == 1 and r["certified"]]
    cand += [(o["method"], o["util_in_partition_pct"], o["util_out_of_partition_pct"]) for o in ours
             if o["cell"] == cell and o["method"] != "Ours full-rank T2"]
    passers[cell] = {"in_sample_order": [m for m, _, _ in sorted(cand, key=lambda x: -x[1])],
                     "heldout_order": [m for m, _, _ in sorted(cand, key=lambda x: -x[2])], "values": cand}

p = save("E_utility_heldout.json", {"inputs": [i1, i2, i3, i4, i5, i6], "table1_rows": rows, "ours_fresh_partition": ours,
                                    "posthoc_arm": posthoc, "rank_changes_all_methods": rank_changes,
                                    "tier1_passers_plus_ours_order": passers,
                                    "rows_without_any_heldout_value": ["none of the 42 baseline combinations lacks lift_lr; "
                                                                       "ours e2e rows lack a head-held-out lift in two_tier_certification.json "
                                                                       "(only the 9 fresh-partition points have out-of-sample utility); "
                                                                       "the post-hoc arm has only head-held-out LR utility"]})
print(p)
for r in rows:
    print(f"{r['method']:<18} {r['cell']:<6} T{r['tier']} cert={r['certified']!s:<5} own={r['util_own_pct']:7.1f} lr={r['util_lr_heldout_pct']:7.1f} table={r['util_table1_pct']:7.1f} uses={r['table1_convention_uses']}")
for k, v in rank_changes.items(): print(k, v["changed_table1_vs_heldout"], v["order_table1"], v["order_lr_heldout"])
for k, v in passers.items(): print(k, v)
for o in ours: print(o)
print(posthoc)
