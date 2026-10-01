# Usage: mkdir d; git -C /Users/nathansamson/PCRL show origin/main:results/v2_cross_purpose/per_seed_results.json > d/v2cp_per_seed_results.json; git -C /Users/nathansamson/PCRL show cross-purpose-rebuttal-2026-05-18:results/rebuttal/cross_purpose/unified_protocol_results.json > d/unified_protocol_results.json; python3 recount_cross_purpose.py d
"""Standalone recount of encoder-lineage cross-purpose attack flags (no repo imports).
Inputs: origin/main:results/v2_cross_purpose/per_seed_results.json (submission, R5/R7 final.pt)
        cross-purpose-rebuttal-2026-05-18:results/rebuttal/cross_purpose/unified_protocol_results.json (cross-purpose-constrained retrain, best.pt)
"""
import json, sys
import numpy as np
D = sys.argv[1]
sub = json.load(open(f"{D}/v2cp_per_seed_results.json"))["per_seed"]
uni = json.load(open(f"{D}/unified_protocol_results.json"))

def cells_sub():
    out = {}
    for e in sub:
        for r in e["rows"]:
            k = (e["dataset"], r["attribute"], r["arch"])
            out.setdefault(k, []).append(dict(seed=e["seed"], maj=r["majority"], concat=r["concat_acc"],
                                              single=r["best_single_acc"], best_p=r["best_single_purpose"],
                                              per=r["single_per_purpose"]))
    return out

def cells_uni():
    out = {}
    for ds, v in uni["per_seed"].items():
        for e in v["per_seed"]:
            for r in e["rows"]:
                k = (ds, r["attribute"], r["arch"])
                out.setdefault(k, []).append(dict(seed=e["seed"], maj=r["majority"], concat=r["concat_acc"],
                                                  single=r["best_single_acc"], best_p=r["best_single_purpose"],
                                                  per=r["single_per_purpose"]))
    return out

def summarize(cells, name):
    res = {"n_cells": len(cells)}
    inc = absn = single_abs = 0
    rows = []
    for k, rs in sorted(cells.items()):
        maj = np.mean([x["maj"] for x in rs])  # aggregate() averages majority across seeds
        c = np.array([x["concat"] for x in rs]); s = np.array([x["single"] for x in rs])
        gain = ((c - s) * 100).mean()
        absd = ((c - maj) * 100).mean()
        sabs = ((s - maj) * 100).mean()
        inc += gain > 1.0; absn += absd > 1.0; single_abs += sabs > 1.0
        rows.append(dict(cell="/".join(k), n_seeds=len(rs), gain_pp=round(gain, 2), concat_minus_majority_pp=round(absd, 2),
                         best_single_minus_majority_pp=round(sabs, 2),
                         best_single_purposes=[x["best_p"] for x in rs]))
    res.update(incremental_flags=int(inc), absolute_flags=int(absn), best_single_absolute_flags=int(single_abs), rows=rows)
    return res

out = {"submission_R5R7_final": summarize(cells_sub(), "sub"), "crosspurpose_retrain_best": summarize(cells_uni(), "uni")}
json.dump(out, open(f"{D}/D_recount_crosspurpose_out.json", "w"), indent=1)
for k, v in out.items():
    print(k, "cells", v["n_cells"], "incremental(gain>1pp)", v["incremental_flags"], "absolute(concat-maj>1pp)", v["absolute_flags"],
          "best_single-maj>1pp", v["best_single_absolute_flags"])

# --- Part 2: split single-purpose leakage by policy status (registry copied from origin/main pcrl/data/{adult,hmda,diabetes}.py) ---
DIS = {
 "adult": {"income_prediction": {"race","sex"}, "employment_analysis": {"race","age_group","marital_status"}, "education_assessment": {"sex","race","income"}},
 "hmda": {"underwriting": {"race","ethnicity"}, "pricing_analysis": {"race","sex"}, "fair_lending_audit": {"race","sex"}},
 "diabetes": {"billing_audit": {"race","gender"}, "quality_research": {"race","age_bucket"}, "clinical_decision_support": {"race","gender"}},
}
def policy_split(cells):
    rows = []
    for (ds, attr, arch), rs in sorted(cells.items()):
        maj = np.mean([x["maj"] for x in rs])
        dis_p = [p for p, a in DIS[ds].items() if attr in a]
        allow_p = [p for p in DIS[ds] if p not in dis_p]
        dis_max = np.mean([max(x["per"][p] for p in dis_p) for x in rs]) if dis_p else float("nan")
        allow_max = np.mean([max(x["per"][p] for p in allow_p) for x in rs]) if allow_p else float("nan")
        c = np.mean([x["concat"] for x in rs])
        rows.append(dict(cell=f"{ds}/{attr}/{arch}", allowed_in=allow_p, disallowed_in=dis_p,
                         concat_minus_maj_pp=round((c-maj)*100,2),
                         max_disallowed_single_minus_maj_pp=round((dis_max-maj)*100,2),
                         max_allowed_single_minus_maj_pp=(round((allow_max-maj)*100,2) if allow_p else None),
                         concat_minus_max_allowed_pp=(round((c-allow_max)*100,2) if allow_p else None)))
    return rows
out["policy_split"] = {"submission_R5R7_final": policy_split(cells_sub()), "crosspurpose_retrain_best": policy_split(cells_uni())}
for k in ("submission_R5R7_final", "crosspurpose_retrain_best"):
    rr = out["policy_split"][k]
    n_dis_flag = sum(r["max_disallowed_single_minus_maj_pp"] > 1.0 for r in rr)
    n_never_allowed = [r for r in rr if not r["allowed_in"]]
    print(k, "cells where a DISALLOWED-purpose single rep beats majority by >1pp:", n_dis_flag, "/", len(rr),
          "| attributes disallowed in every purpose:", sorted({r['cell'].rsplit('/',1)[0] for r in n_never_allowed}),
          "| of those, concat-maj>1pp:", sum(r["concat_minus_maj_pp"] > 1.0 for r in n_never_allowed), "/", len(n_never_allowed))
json.dump(out, open(f"{D}/D_recount_crosspurpose_out.json", "w"), indent=1)
