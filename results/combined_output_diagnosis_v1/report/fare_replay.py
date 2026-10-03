"""Stage 5 replay (saved predictions only): existing Adult/HMDA FARE vs untreated / target LEACE / zero-fairness twin
under the same complete contract (features + own head); task gain above the attacker_fit-majority constant and share
retained, per encoder seed (HMDA s1 one-cell nominee shown separately)."""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
import oar.study as S
U = Path.home() / "PCRL_eval_cache_private/oar_v1/run/units"
SEL = Path.home() / "PCRL_eval_cache_private/oar_v1/run/selection"
rows = []
for ds in ("adult", "hmda"):
    W = S.load_world(ds)
    a, f = W["idx"]["assessment"], W["idx"]["attacker_fit"]
    t = W["t"]
    const = float((t[a] == np.argmax(np.bincount(t[f]))).mean())
    cls = [0, 1] if ds == "adult" else [0, 1, 2]
    for k in S.SEEDS:
        sel = json.loads((SEL / f"{ds}__s{k}.json").read_text())
        src = sel["nominee_unit_source"]
        n_cells = int(np.load(U / f"{ds}__s{k}__FAREFIT_c{src}" / "cells.npy").max()) + 1
        def acc(u):
            z = np.load(U / u / "preds.npz"); return float((z["U2_P"].argmax(1) == z["y_t"]).mean())
        def rec(u):
            z = np.load(U / u / "preds.npz")
            r = json.loads((U / u / "record.json").read_text())
            if r.get("plus_selection", {}).get("alias_source"):
                z = np.load(U / r["plus_selection"]["alias_source"] / "preds.npz")
            return float(np.mean([np.mean([roc_auc_score(z["y_s"] == c, z[f"P__NL__as{s_}"][:, c]) for c in cls]) for s_ in (0, 1, 2)]))
        accs = {"A": acc(f"{ds}__s{k}__U2__A"), "B": acc(f"{ds}__s{k}__U2__B"), "F": acc(f"{ds}__s{k}__U2__Fc{src}"), "FZ": acc(f"{ds}__s{k}__U2__FZ")}
        recs = {"A": rec(f"{ds}__s{k}__A__rep+head"), "B": rec(f"{ds}__s{k}__B__rep+head"), "F": rec(f"{ds}__s{k}__F__rep+head"), "FZ": rec(f"{ds}__s{k}__FZ__rep+head")}
        for arm in ("A", "B", "F", "FZ"):
            g = accs[arm] - const
            rows.append({"dataset": ds, "seed": k, "arm": arm, "u2_accuracy": accs[arm], "constant_accuracy": const,
                         "gain_over_constant": g, "share_of_untreated_gain": g / (accs["A"] - const) if accs["A"] - const > 0 else None,
                         "complete_contract_recovery_macro_auc": recs[arm], "fare_nominee": sel["nominee"] if arm == "F" else "",
                         "fare_n_cells": n_cells if arm == "F" else "", "constant_release": (arm == "F" and n_cells == 1)})
out = WT / "results/combined_output_diagnosis_v1/FARE_REPLAY_EXISTING.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["dataset"], r["seed"], r["arm"].ljust(2), round(r["u2_accuracy"], 4), round(r["gain_over_constant"], 4),
          None if r["share_of_untreated_gain"] is None else round(r["share_of_untreated_gain"], 2), round(r["complete_contract_recovery_macro_auc"], 4), r["fare_n_cells"], r["constant_release"])
