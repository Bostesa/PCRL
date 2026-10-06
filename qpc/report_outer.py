"""Post-assessment report part (lead-owned; split from qpc/report.py so the locked report module is untouched).

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.report_outer

Reads only checked aggregates (<PRIVATE_CACHE>/qpc_v1/run/inference.json levels), the EVALUATION_LOCK scored list and
the outer unit records' utility blocks. Writes COALITION_AND_RATE_RESULTS.csv (accuracy / confidence / recovery
table), ACTUAL_TASK_UTILITY.csv, the assessment columns of CAPACITY_CURVE.csv and figures fig_capacity / fig_tradeoff.
"""
from __future__ import annotations

import csv
import json

from qpc import run as R
from qpc.report import _fig_capacity, _write_csv

def _lv(L, key):
    x = L.get(key)
    return None if x is None else x["point"]


def outer_tables():
    inf = json.loads((R.RUN / "inference.json").read_text())
    EL = json.loads((R.PKG / "EVALUATION_LOCK.json").read_text())
    L = inf["levels"]
    labels = EL["scored_labels"]
    fams = sorted({k.split("#")[2] for k in L if k.startswith("Rmean#")})
    rows, util_rows = [], []
    for lab in labels:
        p = R.parse_id(lab)
        row = {"label": lab, "kind": p["kind"], "family": p.get("family") or lab.split("|")[0],
               "m1": p.get("m1", ""), "m2": p.get("m2", ""), "lam": p.get("lam", "") if p.get("lam") is not None else "",
               "roles": ";".join(sorted(x for x, c in EL["resolved"].items() if c == lab))}
        for fam in fams:
            for v in ("pair", "v1", "v2"):
                x = L.get(f"Rmean#{lab}#{fam}#{v}")
                if x:
                    row[f"{fam}_{v}_auc"], row[f"{fam}_{v}_se"] = x["point"], x["se"]
        for j, t in ((0, "income"), (1, "occ")):
            for kind in ("acc", "ll", "br", "llx", "brx"):
                x = L.get(f"{kind}mean#{lab}#{j}")
                row[f"{kind}_{t}"] = x["point"] if x else ""
                if kind in ("llx", "brx") and x:
                    row[f"{kind}_{t}_se"] = x["se"]
        rows.append(row)
        for k in R.SEEDS:
            n = f"outer__s{k}__{lab.replace('|', '_').replace('*', 'star').replace('/', '_').replace(' ', '_')}"
            if not (R.UNITS / n / "record.json").exists():
                continue
            u = R.rec(n).get("utility") or {}
            for t in ("income", "occupation"):
                m = u.get(t) or u.get({"income": "0", "occupation": "1"}[t]) or {}
                util_rows.append({"seed": k, "label": lab, "task": t, **{kk: m.get(kk) for kk in (
                    "acc", "logloss", "brier", "balanced_acc", "const_acc", "gain", "ece")},
                    "recalls": json.dumps(m.get("recalls"))})
    _write_csv(R.PKG / "COALITION_AND_RATE_RESULTS.csv", rows)
    if util_rows:
        _write_csv(R.PKG / "ACTUAL_TASK_UTILITY.csv", util_rows)
    # assessment columns of the capacity curve (Stage A codes are all scored: PROTOCOL section 13)
    cc = list(csv.DictReader(open(R.PKG / "CAPACITY_CURVE.csv")))
    by = {r["label"]: r for r in rows}
    for r in cc:
        a = by.get(r["config"], {})
        for kk in ("llx_occ", "brx_occ", "llx_income", "brx_income", "primary_pair_auc", "primary_v1_auc",
                   "primary_v2_auc"):
            r[f"assess_{kk}"] = a.get(kk, "")
    _write_csv(R.PKG / "CAPACITY_CURVE.csv", cc)
    outer = {r["config"]: {"ll_excess_occupation": _f(r["assess_llx_occ"]),
                           "brier_excess_occupation": _f(r["assess_brx_occ"])} for r in cc}
    curve = [{**r, **{k: _f(v) for k, v in r.items() if k.startswith(("inner_", "m1_", "m2_"))}} for r in cc]
    for r in curve:
        r["m1_cap"], r["m2_cap"] = int(r["m1_cap"]), int(r["m2_cap"])
    _fig_capacity(curve, outer)
    _fig_tradeoff(rows)
    return len(rows), len(util_rows)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fig_tradeoff(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"SRC": "black", "REF": "tab:purple", "CLASS": "tab:gray", "DIRECT-TASK": "tab:olive",
           "FINE-TASK": "tab:green", "LOCAL": "tab:orange", "SEQ-12": "tab:cyan", "SEQ-21": "tab:blue",
           "JOINT": "tab:red"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (x, xl) in zip(ax, (("llx_occ", "occupation log-loss excess vs U (nats)"),
                               ("llx_income", "income log-loss excess vs U (nats)"),
                               ("acc_occ", "occupation accuracy"))):
        for r in rows:
            xv, yv = _f(r.get(x)), _f(r.get("primary_pair_auc"))
            if xv is None or yv is None:
                continue
            a.scatter(xv, yv, color=col.get(r["family"], "k"), s=28, marker="s" if r["roles"] else "o")
        if x.startswith("llx"):
            a.axvline(0.01, color="k", lw=0.8)
        a.set_xlabel(xl + " (assessment, mean over seeds)")
        a.set_ylabel("pair SEX AUC (assessment; lower = less recovery)")
        a.grid(alpha=0.3)
    for f, c in col.items():
        ax[0].scatter([], [], color=c, label=f)
    ax[0].legend(fontsize=7)
    fig.suptitle("Recovery vs confidence and accuracy (squares = selected roles; line = 0.01-nat allowance)")
    fig.tight_layout()
    (R.PKG / "figures").mkdir(exist_ok=True)
    fig.savefig(R.PKG / "figures" / "fig_tradeoff.pdf")
    fig.savefig(R.PKG / "figures" / "fig_tradeoff.png", dpi=120)


if __name__ == "__main__":
    print("outer tables", outer_tables())
