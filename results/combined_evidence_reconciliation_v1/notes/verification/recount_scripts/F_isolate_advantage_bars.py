"""F6: the isolate-then-noise (subspace) utility advantage over full-rank noise, at AUC bars
0.52/0.55/0.60, three ways:
  (a) frontier on the stored seed-0 sweeps: max utility s.t. Tier-1 max <= bar, per channel
      (two_tier_certification.json e2e_cells[].sweep for full-rank; e2e_surgical_winners[].rows
      for the subspace channel at its winning (rank, lam));
  (b) the paper's matched comparison (isolate_vs_fullrank.json; nearest-T1 matching rule,
      run_isolate_vs_fullrank.py:79-93) reproduced as stored;
  (c) out of partition at the paper's selected points (fresh_partition_generalization.json).
Tier 2: the subspace channel's stored LRT readings are listed (no bar <= 0.60 is met).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from F_common import BARS, dg_json, save

tt, i1 = dg_json("results/two_tier_certification.json")
iv, i2 = dg_json("results/isolate_vs_fullrank.json")
fp, i3 = dg_json("results/fresh_partition_generalization.json")
CELL = {"hmda/race/loan_decision": "easy", "hmda/race/loan_amount_band": "middle", "adult/sex/income": "hard"}

full = {CELL[c["cell"]]: (c["clean_lift"], c["sweep"]) for c in tt["e2e_cells"]}
sub = {c["cell"].split()[0]: (c["clean_lift"], c["rows"], c["config"]) for c in tt["e2e_surgical_winners"]}


def frontier(cl, rows, bar, key="tier1_max"):
    ok = [r for r in rows if r[key] <= bar]
    if not ok:
        return None
    r = max(ok, key=lambda r: r["lift"])
    return {"sigma": r["sigma"], "t1": r["tier1_max"], "t2": r["tier2_max"], "util_pct": 100 * r["lift"] / cl}


res = {}
for b in BARS:
    per = {}
    for cell in ("easy", "middle", "hard"):
        f = frontier(*full[cell], b)
        s = frontier(sub[cell][0], sub[cell][1], b)
        f2 = frontier(*full[cell], b, key="tier2_max")
        s2 = frontier(sub[cell][0], sub[cell][1], b, key="tier2_max")
        per[cell] = {"fullrank_t1": f, "subspace_t1": s,
                     "advantage_pp_t1": (s["util_pct"] - f["util_pct"]) if (f and s) else None,
                     "fullrank_t1_sweep_floor_note": ("full-rank sweep starts at sigma=%g (T1=%.4f); lower sigma not stored"
                                                      % (full[cell][1][0]["sigma"], full[cell][1][0]["tier1_max"])),
                     "fullrank_t2": f2, "subspace_t2": s2}
    res[f"{b:.2f}"] = per

matched = [{"cell": r["cell"], "isolate_t1": r["isolate"]["t1_max_mean"], "isolate_util": r["isolate"]["util_mean"],
            "fullrank_sigma": r["fullrank"]["sigma"], "fullrank_t1": r["fullrank"]["t1_max_mean"],
            "fullrank_util": r["fullrank"]["util_mean"],
            "gap_pp": r["isolate"]["util_mean"] - r["fullrank"]["util_mean"]} for r in iv["rows"]]
oop = []
for c in fp["cells"]:
    oop.append({"cell": CELL[c["cell"]],
                "fullrank_t1_util_eval_pct": c["fullrank_t1"]["utility_kept_eval_pct"],
                "subspace_util_eval_pct": c["subspace"]["utility_kept_eval_pct"],
                "advantage_pp_out_of_partition": c["subspace"]["utility_kept_eval_pct"] - c["fullrank_t1"]["utility_kept_eval_pct"],
                "in_partition_fullrank_t1": c["fullrank_t1"]["stored_inpartition"]["util_pct"],
                "in_partition_subspace": c["subspace"]["stored_inpartition"]["util_pct"],
                "advantage_pp_in_partition": c["subspace"]["stored_inpartition"]["util_pct"] - c["fullrank_t1"]["stored_inpartition"]["util_pct"],
                "t1_fullrank_eval": c["fullrank_t1"]["tier1_max_mean"], "t1_subspace_eval": c["subspace"]["tier1_max_mean"],
                "t2_subspace_eval": c["subspace"]["tier2_max_mean"]})
p = save("F_isolate_advantage_bars.json", {"inputs": [i1, i2, i3], "frontier_seed0_sweeps_by_bar": res,
                                           "paper_matched_comparison_stored": matched,
                                           "selected_points_in_vs_out_of_partition": oop})
print(p)
for b, per in res.items():
    print(b, {c: (round(v["advantage_pp_t1"], 1) if v["advantage_pp_t1"] is not None else None,
                  v["fullrank_t1"] and v["fullrank_t1"]["sigma"], v["subspace_t1"] and v["subspace_t1"]["sigma"],
                  v["subspace_t2"]) for c, v in per.items()})
for m in matched: print(m)
for o in oop: print(o)
