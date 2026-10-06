"""Reports (public aggregates only). Lead-owned.

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.report --part stagea|stageb|outer|all

stagea (after the gate; INNER_SELECTION + fitting-row statistics only):
  CAPACITY_CURVE.csv          per Stage A configuration: caps, actual alphabets, tokens emitted on fitting rows,
                              token entropy (fitting rows), fit KL distortion per recipient (receipts), INNER
                              log-loss / Brier excess per task (mean and worst seed), eligibility, headroom; the
                              assessment columns are appended by --part outer when (and only if) the assessment ran
  figures/fig_capacity.{pdf,png}  occupation states vs occupation confidence excess (inner; assessment when present)
stageb (after Stage B fits): CLASS_PRESERVATION.json, OPTIMIZATION_RECEIPTS.json (receipts, aliases, MI diagnostics)
outer (after the assessment and inference): ACTUAL_TASK_UTILITY.csv, COALITION_AND_RATE_RESULTS.csv,
  figures/fig_tradeoff.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np

from qpc import gate as GT
from qpc import run as R

TASKS = ("income", "occupation")


def entropy(tok):
    _, c = np.unique(np.asarray(tok), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def _fit_kl(rec, i):
    """Mean fitting KL(p_i || decoded) reported by the compressor receipts (None if absent)."""
    return (rec.get("fit_distortion") or {}).get(f"D{i}")


def capacity_curve(D=None):
    gate = json.loads((R.RUN / "gate.json").read_text())
    tr = None
    if D is not None:
        tr = np.asarray(D["idx"]["DEFENSE_FIT"])
    rows = []
    for cid in GT.stagea_ids():
        s = gate["summary"][cid]
        p = R.parse_id(cid)
        r = {"config": cid, "m1_cap": p["m1"], "m2_cap": p["m2"], "eligible_all_seeds": s["eligible_all_seeds"],
             "headroom_all_seeds": s["headroom_all_seeds"], "mean_total_states": s["mean_total_states"],
             "worst_seed_norm_excess": s["worst_seed_norm_excess"],
             "selected": cid in gate["decision"]["selected_rates"]}
        per = gate["per_config_seed"][cid]
        for t in TASKS:
            for x in ("ll_excess", "brier_excess"):
                vals = [per[str(k)]["gate"][t][x] for k in R.SEEDS]
                r[f"inner_{x}_{t}_mean"] = float(np.mean(vals))
                r[f"inner_{x}_{t}_worst"] = float(np.max(vals))
        a, ent, kl, emitted = [[], []], [[], []], [[], []], [[], []]
        for k in R.SEEDS:
            u = R.unit_for(k, cid)
            z = R.npz(u, "release.npz")
            rec = R.rec(u)
            for i in (1, 2):
                a[i - 1].append(int(z[f"alpha{i}"]))
                if tr is not None:
                    ent[i - 1].append(entropy(z[f"tok{i}"][tr]))
                    emitted[i - 1].append(int(len(np.unique(z[f"tok{i}"][tr]))))
                v = _fit_kl(rec, i)
                if v is not None:
                    kl[i - 1].append(float(v))
        for i, t in ((1, "income"), (2, "occupation")):
            r[f"alphabet_{t}_mean"] = float(np.mean(a[i - 1]))
            r[f"emitted_fit_{t}_mean"] = float(np.mean(emitted[i - 1])) if emitted[i - 1] else ""
            r[f"token_entropy_fit_{t}_mean"] = float(np.mean(ent[i - 1])) if ent[i - 1] else ""
            r[f"fit_kl_{t}_mean"] = float(np.mean(kl[i - 1])) if kl[i - 1] else ""
        rows.append(r)
    _write_csv(R.PKG / "CAPACITY_CURVE.csv", rows)
    _fig_capacity(rows)
    return rows


def _write_csv(path, rows):
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def _fig_capacity(rows, outer=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, (x, lim, lab) in enumerate((("ll_excess", 0.01, "log-loss excess (nats)"),
                                       ("brier_excess", 0.005, "Brier excess"))):
        a = ax[j]
        for m1, mk in ((4, "o"), (8, "s")):
            rr = sorted([r for r in rows if r["m1_cap"] == m1], key=lambda r: r["m2_cap"])
            xs = [r["m2_cap"] for r in rr]
            a.plot(xs, [r[f"inner_{x}_occupation_worst"] for r in rr], marker=mk, label=f"occupation, inner worst seed (income cap {m1})")
            a.plot(xs, [r[f"inner_{x}_income_worst"] for r in rr], marker=mk, ls=":", label=f"income, inner worst seed (income cap {m1})")
            if outer:
                a.plot(xs, [outer.get(r["config"], {}).get(f"{x}_occupation") for r in rr], marker=mk, ls="--",
                       label=f"occupation, assessment mean (income cap {m1})")
        a.axhline(lim, color="k", lw=0.8)
        a.set_xscale("log", base=2)
        a.set_xlabel("occupation states per predicted class (cap)")
        a.set_ylabel(lab + " vs U")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=6)
    fig.suptitle("Task-only DIRECT-TASK capacity curve (U teacher; line = registered allowance)")
    fig.tight_layout()
    (R.PKG / "figures").mkdir(exist_ok=True)
    fig.savefig(R.PKG / "figures" / "fig_capacity.pdf")
    fig.savefig(R.PKG / "figures" / "fig_capacity.png", dpi=120)


def main(argv=None):
    part = (argv or sys.argv[1:] or ["--part", "all"])[-1]
    D = None
    if part in ("stagea", "all"):
        from qpc import data as DA
        D = DA.load()
        print("capacity curve rows", len(capacity_curve(D)))


if __name__ == "__main__":
    main()
