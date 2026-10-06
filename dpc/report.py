"""Reports (public aggregates only).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.report [--part fit|outer|all]

fit part (after TRAINING_LOCK fits; no labels beyond OSF_DEFENSE_FIT objectives):
  CLASS_PRESERVATION.json   per policy unit: decisions equal the teacher's on every row (all roles), token alphabets,
                            effective fitting states per predicted class, fallback tokens
  OPTIMIZATION_RECEIPTS.json per configuration and seed: D_i, I_i, I_12, F_* after greedy and final, JOINT starts /
                            winner / witness dominance, sequential stage coefficients, refinement work, aliases by
                            fingerprint, fine-partition convergence, smoothing differences
outer part (after the assessment and inference):
  ACTUAL_TASK_UTILITY.csv   assessment utility of every scored release (acc, log loss, Brier, balanced acc, recalls, ECE)
  COALITION_AND_RATE_RESULTS.csv  every scored label: pair/local recovery per view family (equal FINAL-slate attack
                            strength), ignore-recipient winners, code-pair coverage, utility, rate, family, fitted MI
  figures/fig_tradeoff.{pdf,png}  pair recovery against BOTH task log losses, with classification accuracy beside it
  MODEL_MANIFEST.json       packaged units, statuses, deployment
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np

from dpc import run as R


def _pol_units():
    return [d.name for d in sorted(R.UNITS.glob("pol__s*")) if R.done(d.name)]


def class_preservation():
    out = {"rule": "released decision = argmax of the decoded prototype = teacher decision, every row of every role",
           "units": {}, "all_pass": True}
    for n in _pol_units():
        r = R.rec(n)
        z = np.load(R.U(n) / "release.npz")
        T = R.teacher(r["seed"], r["cfg"]["teacher"])
        ok = {i: bool(np.array_equal(z[f"hard{i}"], T[f"d{i}"]) and np.array_equal(z[f"q{i}"].argmax(1), T[f"d{i}"]))
              for i in (1, 2)}
        out["units"][n] = {"config": r["config"], "seed": r["seed"], "rows": int(len(z["row_id"])),
                           "decisions_equal_teacher": ok, "alphabet": [int(z["alpha1"]), int(z["alpha2"])],
                           "fitting_token_states": r.get("token_states_fit")}
        out["all_pass"] &= all(ok.values())
    out["n_units"] = len(out["units"])
    (R.PKG / "CLASS_PRESERVATION.json").write_text(json.dumps(out, indent=1) + "\n")
    return out["all_pass"], out["n_units"]


def optimization_receipts():
    rows, fps = {}, defaultdict(list)
    for n in _pol_units():
        r = R.rec(n)
        rc = r["receipts"]
        key = f"{r['config']}#s{r['seed']}"
        entry = {"final": rc.get("final"), "after_greedy": rc.get("after_greedy"), "work": rc.get("work"),
                 "fingerprint": r["fingerprint"], "token_states_fit": r.get("token_states_fit"),
                 "wall_s": r.get("wall_s")}
        for x in ("winner", "witness_dominance", "stage_coefficients", "stages", "row_level_max_abs_diff",
                  "summary"):
            if x in rc:
                entry[x] = rc[x]
        if "starts" in rc:
            entry["starts"] = {s: {kk: v.get(kk) for kk in ("initial_F_joint", "refined_F_joint", "source")}
                               for s, v in rc["starts"].items()}
        rows[key] = entry
        fps[(r["seed"], r["cfg"]["teacher"], r["fingerprint"])].append(r["config"])
    aliases = [{"seed": s, "teacher": t, "configs": sorted(v)} for (s, t, _), v in fps.items() if len(v) > 1]
    fine = {}
    for d in sorted(R.UNITS.glob("fine__s*")):
        rr = R.rec(d.name)["receipts"]
        fine[d.name] = {i: [{kk: c.get(kk) for kk in ("class", "rows", "distinct_vectors", "cells", "initial_cells",
                                                      "rounds_used", "converged", "sparse_cells", "fallback")}
                            for c in rr[i]["per_class"]] for i in ("r1", "r2")}
    out = {"schema": "dpc-optimization-receipts-v1", "n_policy_units": len(rows), "alias_groups": aliases,
           "n_alias_groups": len(aliases), "fine_partitions": fine, "units": rows,
           "notes": ["objectives are fitting-row training criteria (plug-in MI, KL to the teacher); not privacy bounds",
                     "JOINT final F_joint <= every witness is asserted in each unit (witness_dominance)",
                     "exact aliases keep their manifest slots; one physical fit each"]}
    (R.PKG / "OPTIMIZATION_RECEIPTS.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return len(rows), len(aliases)


def outer_records():
    out = {}
    for d in sorted(R.UNITS.glob("outer__s*")):
        if R.done(d.name):
            r = R.rec(d.name)
            out[(r["seed"], r["label"])] = r
    return out


def write_outer():
    recs = outer_records()
    inf = json.loads((R.RUN / "inference.json").read_text())
    L = inf["levels"]
    util_rows = []
    for (k, lab), r in sorted(recs.items()):
        for i, task in (("0", "income"), ("1", "occupation_group")):
            u = r["utility"][i] if i in r["utility"] else r["utility"][int(i)]
            util_rows.append({"seed": k, "label": lab, "task": task, **{kk: u.get(kk) for kk in (
                "acc", "logloss", "brier", "const_acc", "gain", "balanced_acc", "minority_recall", "ece")},
                "recalls": json.dumps(u.get("recalls"))})
    with open(R.PKG / "ACTUAL_TASK_UTILITY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(util_rows[0]), lineterminator="\n")
        w.writeheader()
        for q in util_rows:
            w.writerow({kk: (f"{v:.6f}" if isinstance(v, float) else v) for kk, v in q.items()})
    labels = sorted({lab for (_, lab) in recs})
    fams = sorted({kk.split("#")[2] for kk in L if kk.startswith("Rmean#")} and
                  {kk.split("#")[2] for kk in L if kk.startswith("Rmean#")})
    rows = []
    for lab in labels:
        p = R.parse_id(lab)
        row = {"label": lab, "kind": p["kind"], "teacher": p.get("teacher") or "", "family": p.get("family") or
               (lab.split("|")[0]), "m": p.get("m") or "", "lam": p.get("lam") or ""}
        for fam in fams:
            for v in ("pair", "v1", "v2"):
                x = L.get(f"Rmean#{lab}#{fam}#{v}")
                if x:
                    row[f"{fam}_{v}_auc"], row[f"{fam}_{v}_se"] = x["point"], x["se"]
        for j, t in ((0, "income"), (1, "occ")):
            for kind in ("acc", "ll", "br"):
                x = L.get(f"{kind}mean#{lab}#{j}")
                row[f"{kind}_{t}"] = x["point"] if x else ""
        win = [recs[(k, lab)]["families"][recs[(k, lab)]["primary_family"]].get("scored", {}).get("pair", {})
               .get("auc", {}).get("source_view", "") for k in R.SEEDS if (k, lab) in recs]
        row["pair_winner_views"] = ";".join(str(x) for x in win)
        cov = [recs[(k, lab)]["families"][recs[(k, lab)]["primary_family"]].get("coverage") for k in R.SEEDS
               if (k, lab) in recs]
        row["pair_coverage"] = json.dumps(cov)[:300] if any(cov) else ""
        rows.append(row)
    cols = sorted({c for r in rows for c in r}, key=lambda c: (c not in ("label", "kind", "teacher", "family", "m", "lam"), c))
    with open(R.PKG / "COALITION_AND_RATE_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for q in rows:
            w.writerow({kk: (f"{v:.6f}" if isinstance(v, float) else v) for kk, v in q.items()})
    figure(L, labels)
    return len(util_rows), len(rows)


def figure(L, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = json.loads((R.RUN / "selection.json").read_text())
    elig = {c: r["task_eligible"] for c, r in sel["rows"].items()}
    col = {"SRC": "black", "REF": "tab:purple", "CLASS": "tab:gray", "FINE-TASK": "tab:green", "DIRECT-TASK": "tab:olive",
           "LOCAL": "tab:orange", "SEQ-12": "tab:cyan", "SEQ-21": "tab:blue", "JOINT": "tab:red"}
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    panels = [(0, "ll", 0, "income log loss"), (1, "ll", 1, "occupation log loss"), (2, "acc", 0, "income accuracy"),
              (3, "acc", 1, "occupation accuracy")]
    for a_i, kind, j, xl in panels:
        a = ax[a_i]
        for lab in labels:
            fam = lab.split("|")[0] if lab.startswith(("SRC", "REF")) else lab.split("|")[1]
            for k in R.SEEDS:
                x = L.get(f"{kind}#{k}#{lab}#{j}", {}).get("point")
                y = L.get(f"R#{k}#{lab}#primary#pair", {}).get("point")
                if x is None or y is None:
                    continue
                mk = "s" if lab.startswith("RAW-J") or lab == "SRC|RAW-J_b0.3" else "o"
                a.scatter(x, y, s=24, marker=mk, edgecolors=col.get(fam, "k"),
                          facecolors=col.get(fam, "k") if elig.get(lab) else "none", linewidths=1)
        a.set_xlabel(f"{xl} (assessment)")
        a.set_ylabel("pair SEX AUC (assessment; lower = less recovery)")
        a.grid(alpha=0.3)
    for fam, c in col.items():
        ax[0].scatter([], [], color=c, label=fam)
    ax[0].legend(fontsize=7)
    fig.suptitle("Pair recovery vs task log loss and accuracy; filled = inner task-eligible; squares = RAW-J teacher")
    fig.tight_layout()
    (R.PKG / "figures").mkdir(exist_ok=True)
    fig.savefig(R.PKG / "figures" / "fig_tradeoff.pdf")
    fig.savefig(R.PKG / "figures" / "fig_tradeoff.png", dpi=120)


def main(argv=None):
    part = (argv or sys.argv[1:] or ["--part", "all"])[-1]
    if part in ("fit", "all"):
        print("class preservation", class_preservation())
        print("optimization receipts", optimization_receipts())
    if part in ("outer", "all"):
        print("outer", write_outer())


if __name__ == "__main__":
    main()
