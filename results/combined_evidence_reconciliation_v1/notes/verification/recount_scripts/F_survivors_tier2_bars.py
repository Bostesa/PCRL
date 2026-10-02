"""F7: 'five of the eight [survivors] still leak to an attacker who knows the defense' at bars
0.52/0.55/0.60, from stored Tier-2 readings (number_reports.json c_adult_survivors_tier2 for the
3 Adult rows; tpr_extension.json tier2 suites for the 5 expansion rows). Also re-derives the
'clean accuracies 0.915/0.619/0.895' to show their composition.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from F_common import BARS, dg_json, save

nr, i1 = dg_json("results/number_reports.json")
te, i2 = dg_json("results/tpr_extension.json")
rows = [(r["config"], r["tier1_best"], r["tier2_max"]) for r in nr["c_adult_survivors_tier2"]["rows"]]
for c in te["results"]["configs"]:
    if c["population"] == "a_survivors" and c["config"].endswith("|tier2"):
        m = c["suite_auc_means"]
        rows.append((c["config"], max(m["XGB"], m["MLP"], m["LoRA"]), max(m.values())))
assert len(rows) == 8
counts = {f"{b:.2f}": {"tier2_leak": sum(t2 > b for _, _, t2 in rows),
                       "tier1_leak": sum(t1 > b for _, t1, _ in rows)} for b in BARS}
clean = [{"cell": a["cell"], "clean_accuracy_reported": a["clean_accuracy"],
          "composed_from": [a["source_lift"], a["source_majority"]],
          "majority_plus_insample_lift": a["majority_baseline_acc"] + a["clean_lift_insample"]}
         for a in nr["a_headline_accuracies"]]
p = save("F_survivors_tier2_bars.json", {"inputs": [i1, i2],
                                          "survivor_rows_[config,tier1,tier2]": rows, "counts_by_bar": counts,
                                          "clean_accuracy_composition": clean})
print(p, counts)
for r in rows: print(r)
for c in clean: print(c)
