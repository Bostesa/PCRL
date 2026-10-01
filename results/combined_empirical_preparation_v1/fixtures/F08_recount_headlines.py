"""F8 -- independent recount of headline counts from STORED result files.

No model is fitted. Inputs are read-only:
  DG   durable-guarantees clone @ 956f5c8 (results/*.json)
  PX   PCRL origin/main @ 55e4cb1d1 results exported with `git archive` to scratch
  PL   local untracked PCRL results/rebuttal/erase_layer_pilot_aws (main checkout)
Pearson / Spearman are implemented here with numpy (average ranks for ties).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DG = Path(sys.argv[1]) if len(sys.argv) > 1 else None
PX = Path(sys.argv[2]) if len(sys.argv) > 2 else None
PL = Path(sys.argv[3]) if len(sys.argv) > 3 else None
INPUTS = []


def load(p):
    p = Path(p)
    INPUTS.append({"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return json.loads(p.read_text())


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def honest():
    d = load(DG / "results/honest_reaudit.json")
    m = d["master"]
    stopped = [r for r in m if r["old_verdict"] == "stopped"]
    surv = [r for r in m if r["honest_stopped"]]
    recomputed_surv = [r for r in m if max(r["xgb_auc"], r["mlp_auc"]) <= d["honest_auc_bar"]]
    key = lambda r: (r["cell"], round(r["old_r2"], 10), round(r["xgb_auc"], 10), round(r["mlp_auc"], 10))
    uniq_stopped = {key(r) for r in stopped}
    uniq_surv = {key(r) for r in surv}
    return {"stored_claim": f"{d['survived']}/{d['total']} survive", "rows_total": len(m),
            "rows_old_verdict_breach": sum(r["old_verdict"] == "breach" for r in m),
            "rows_old_verdict_stopped": len(stopped),
            "survivors_stored_flag": len(surv), "survivors_recomputed_from_aucs": len(recomputed_surv),
            "survivors_all_old_stopped": all(r["old_verdict"] == "stopped" for r in surv),
            "collapse_among_stopped": len(stopped) - sum(r["honest_stopped"] for r in stopped),
            "distinct_stopped_measurements": len(uniq_stopped), "distinct_survivor_measurements": len(uniq_surv),
            "survivors": [f"{r['experiment']}|{r['cell']}|{r['method']}|xgb={r['xgb_auc']:.4f}|mlp={r['mlp_auc']:.4f}" for r in surv],
            "near_bar_rows": [f"{r['experiment']}|{r['cell']}|{r['method']}|max={max(r['xgb_auc'], r['mlp_auc']):.4f}"
                              for r in m if abs(max(r["xgb_auc"], r["mlp_auc"]) - 0.55) < 0.0125]}


def gauntlet():
    d = load(DG / "results/master_gauntlet_table.json")
    combos = [(c, m, t, v[t]["certified"]) for c, ms in d["cells"].items() for m, v in ms.items() for t in ("tier1", "tier2")]
    passes = [x for x in combos if x[3]]
    txt = (DG / "results/master_gauntlet_table.txt").read_text()
    INPUTS.append({"path": str(DG / "results/master_gauntlet_table.txt"),
                   "sha256": hashlib.sha256(txt.encode()).hexdigest()})
    txt_pub_cert = sum(1 for line in txt.splitlines() if "CERTIFIES" in line and "OURS" not in line)
    return {"combinations": len(combos), "methods": sorted({m for _, m, _, _ in combos}),
            "cells": sorted(d["cells"]), "certified": len(passes), "certified_list": [list(p[:3]) for p in passes],
            "txt_published_rows_marked_CERTIFIES": txt_pub_cert}


def continuous():
    merged = load(DG / "results/continuous_cost.json")
    # only the adult shard is tracked in git; use the merged rows and cross-check the adult shard
    rows = merged["rows"]
    good = [r for r in rows if not r["degenerate"]]
    adult = load(DG / "results/continuous_cost_adult.json")["rows"]
    mrg = {r["cell"]: r for r in rows}
    out = {"merged_rows": len(rows), "non_degenerate": len(good),
           "shards_tracked": ["adult"], "untracked_shards": ["hmda", "diabetes"],
           "adult_shard_rows_identical_in_merged": all(mrg.get(r["cell"]) == r for r in adult),
           "adult_shard_rows": len(adult)}
    for key in ("cost_rep", "cost_durable"):
        x = [r["predictor"] for r in good]
        y = [r[key] for r in good]
        out[key + "_n20"] = {"pearson": pearson(x, y), "spearman": spearman(x, y),
                             "stored_pearson": merged["stats"][key]["pearson_r"],
                             "stored_spearman": merged["stats"][key]["spearman_r"]}
    exp = load(DG / "results/expansion_analysis.json")
    new = [r for r in exp["new_cell_pairs"] if not r["degenerate"]]
    for key in ("cost_rep", "cost_durable"):
        x = [r["predictor"] for r in good] + [r["predictor"] for r in new]
        y = [r[key] for r in good] + [r[key] for r in new]
        out[key + "_n27"] = {"n": len(x), "pearson": pearson(x, y), "spearman": spearman(x, y),
                             "stored_pearson": exp["correlation"][key]["combined"]["pearson_r"],
                             "stored_spearman": exp["correlation"][key]["combined"]["spearman_r"]}
    return out


def pcrl_encoder():
    R = PX / "results"
    out = {}
    best = {}
    for ds, tag in (("adult", "ROUND5"), ("hmda", "ROUND5"), ("diabetes", "ROUND5"), ("diabetes", "ROUND6"), ("diabetes", "ROUND7")):
        p = load(R / f"v2_{ds}_{tag}/per_seed_results.json")["per_seed"]
        best[f"{ds}_{tag}"] = {"strict_pass": sum(a["linear_r2"] <= 0.05 for s in p for a in s["attribute_results"]),
                               "cells": sum(len(s["attribute_results"]) for s in p),
                               "best_epochs": [s.get("best_epoch") for s in p], "last_epochs": [s.get("last_epoch") for s in p]}
    out["per_seed_results_selected_checkpoint"] = best
    fvb = {}
    for ds in ("adult", "hmda", "diabetes"):
        d = load(R / f"v2_{ds}_ROUND5/final_vs_best.json")["per_seed"]
        fvb[ds] = {k: sum(sum(r["linear_r2"] <= 0.05 for r in d[s][k]["rows"]) for s in d) for k in ("best", "final")}
    out["final_vs_best_ROUND5"] = fvb
    rows = []
    for ds, tag in (("adult", "ROUND5"), ("hmda", "ROUND5"), ("diabetes", "ROUND7")):
        d = load(R / f"v2_{ds}_{tag}/dominant_axis_audit.json")
        for sd, v in d["per_seed"].items():
            for r in v["rows"]:
                pc, pi = np.array(r["per_class_r2"]), np.array(r["priors"])
                w = pi * (1 - pi) / (pi * (1 - pi)).sum()
                rows.append({"ds": ds, "seed": sd, "epoch": v.get("epoch"), "pair": f"{r['purpose']}/{r['attribute']}",
                             "K": r["num_classes"], "onehot": r["r2_onehot"], "da": r["r2_da"], "f64": float(w @ pc)})
    multi = [r for r in rows if r["K"] > 2]
    amp = np.array([r["da"] / r["onehot"] for r in multi])
    hidden = [r for r in rows if r["onehot"] <= 0.05 < r["da"]]
    out["dominant_axis_audits_final_epoch"] = {
        "cells": len(rows), "epochs": sorted({r["epoch"] for r in rows}),
        "strict_pass_stored_onehot": sum(r["onehot"] <= 0.05 for r in rows),
        "strict_pass_float64_identity_onehot": sum(r["f64"] <= 0.05 for r in rows),
        "strict_pass_dominant_axis": sum(r["da"] <= 0.05 for r in rows),
        "multiclass_pair_seeds": len(multi),
        "amplification_median": float(np.median(amp)), "amplification_max": float(amp.max()),
        "n_amp_gt_1.5": int((amp > 1.5).sum()), "n_amp_gt_2": int((amp > 2).sum()),
        "hidden_cases": [f"{r['ds']} s{r['seed']} {r['pair']} onehot={r['onehot']:.4f} da={r['da']:.4f}" for r in hidden],
        "certificate_minus_float64_identity": {
            "max_abs": float(max(abs(r["f64"] - r["onehot"]) for r in rows)),
            "max_relative_understatement": float(max((r["f64"] - r["onehot"]) / r["f64"] for r in rows if r["f64"] > 0)),
            "n_cert_above_float64": int(sum(r["onehot"] > r["f64"] + 1e-12 for r in rows)),
            "n_relative_understatement_gt_1pct": int(sum((r["f64"] - r["onehot"]) / r["f64"] > 0.01 for r in rows if r["f64"] > 0)),
            "pass_fail_flips_at_0.05": int(sum((r["onehot"] <= 0.05) != (r["f64"] <= 0.05) for r in rows)),
            "worst_rows": [f"{r['ds']} s{r['seed']} {r['pair']} K={r['K']} cert={r['onehot']:.5f} f64={r['f64']:.5f}"
                           for r in sorted(rows, key=lambda r: r["onehot"] - r["f64"])[:3]]},
    }
    return out


def erase_pilot():
    out = {}
    tot = 0
    fall = []
    for ds in ("adult", "hmda", "diabetes"):
        p = load(PL / f"v2_{ds}_ERASE_PILOT/per_seed_results.json")
        ps = p["per_seed"]
        strict = sum(a["linear_r2"] <= 0.05 for s in ps for a in s["attribute_results"])
        cells = sum(len(s["attribute_results"]) for s in ps)
        clean = 0
        for s in ps:
            h = s["per_purpose_health"]
            for a in s["attribute_results"]:
                hh = h[a["purpose"]]
                clean += int(a["linear_r2"] <= 0.05 and hh["per_dim_std_mean"] >= 0.5 and hh["effective_rank"] >= 2.0)
        tot += strict
        fall += [f"{ds} s{s['seed']}: {s['cotter_selection'].get('kind')} feasible={s['cotter_selection'].get('n_feasible_post_warmup')} epoch={s['cotter_selection'].get('epoch')}" for s in ps]
        out[ds] = {"strict_pass": f"{strict}/{cells}", "cleanly_compliant": f"{clean}/{cells}",
                   "status_field": p["summary"].get("STATUS"),
                   "adj_pass_counts_per_seed": [s.get("pass_count") for s in ps],
                   "best_epoch": [s["best_epoch"] for s in ps], "last_epoch": [s["last_epoch"] for s in ps],
                   "final_and_best_both_stored": False}
    comp = load(PL / "rebuttal/comparison.json")
    out["total_strict"] = f"{tot}/60"
    out["comparison_json_pilot_strict"] = {r["dataset"]: r["pilot"]["strict_pass"] for r in comp["rows"]}
    out["comparison_json_base_strict"] = {r["dataset"]: (r["base_tag"], r["base"]["strict_pass"]) for r in comp["rows"]}
    out["checkpoint_selector"] = fall
    return out


def main():
    res = {"id": "F8", "honest_reaudit": honest(), "master_gauntlet": gauntlet(), "continuous_cost": continuous(),
           "pcrl_encoder": pcrl_encoder(), "erase_layer_pilot": erase_pilot()}
    res["input_files"] = INPUTS
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
