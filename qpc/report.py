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


# ------------------------------------------------------------------ stage B part (fitting rows only)
def class_preservation(D):
    """Every fitted policy unit (Stage A and Stage B): released decisions equal its teacher's on every row of every
    role, prototypes' argmax equals the decision, alphabets and fitting-row emitted states."""
    out = {"rule": "released decision = argmax of the decoded prototype = teacher decision, every row of every role",
           "units": {}, "all_pass": True, "rows_checked": 0}
    roles = {r: np.asarray(ix) for r, ix in D["idx"].items() if r in ("DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT",
                                                                        "INNER_SELECTION", "OSF_DEVELOPMENT_ASSESSMENT")}
    tr = np.asarray(D["idx"]["DEFENSE_FIT"])
    for d in sorted(R.UNITS.glob("pol__s*")):
        if not R.done(d.name):
            continue
        z = R.npz(d.name, "release.npz")
        rec = R.rec(d.name)
        k = int(d.name.split("__")[1][1:])
        T = R.teacher(k)
        ok, mism = {}, {}
        for i in (1, 2):
            eq = (z[f"hard{i}"] == T[f"d{i}"]) & (np.asarray(z[f"q{i}"]).argmax(1) == T[f"d{i}"])
            ok[i] = bool(eq.all())
            mism[i] = {r: int((~eq[ix]).sum()) for r, ix in roles.items()}
            out["rows_checked"] += int(len(eq))
        out["units"][d.name] = {"config": rec.get("config"), "seed": k, "decisions_equal_teacher": ok,
                                "mismatches_by_role": mism, "alphabet": [int(z["alpha1"]), int(z["alpha2"])],
                                "emitted_fit": [int(len(np.unique(z[f"tok{i}"][tr]))) for i in (1, 2)]}
        out["all_pass"] &= all(ok.values())
    out["n_units"] = len(out["units"])
    (R.PKG / "CLASS_PRESERVATION.json").write_text(json.dumps(out, indent=1) + "\n")
    return out["all_pass"], out["n_units"], out["rows_checked"]


def optimization_receipts():
    out = {"schema": "qpc-optimization-receipts-v1", "stage_a": {}, "fine_partitions": {}, "stage_b": {},
           "notes": ["objectives are fitting-row training criteria (plug-in MI, KL to the teacher); not privacy bounds",
                     "plug-in MI at these alphabets is biased upward: read every fitted MI beside its permutation null",
                     "JOINT is a local search (math review: small optimiser gaps on tiny exhaustive fixtures; XOR needs "
                     "a coordinated move); final F_joint <= every unchanged witness is asserted per unit",
                     "stop reasons are reported separately: assignment fixed point, relative tolerance, cap"]}
    for d in sorted(R.UNITS.glob("dir__s*")):
        rc = R.rec(d.name)["receipts"]
        cls = [c for c in rc["per_class"] if not c["fallback"]]
        win = [next(s for s in c["starts"] if s["start"] == c["winner"]) for c in cls]
        out["stage_a"][d.name] = {"m": rc["m"], "winners": [c["winner"] for c in cls],
                                  "stop_reasons": [w["stop_reason"] for w in win],
                                  "rounds_used": [w["rounds_used"] for w in win],
                                  "effective_cells": [c["effective_cells"] for c in cls],
                                  "fallback_classes": rc.get("fallback_classes"), "mean_kl_fit": rc["mean_kl_fit"],
                                  "all_starts_converged": rc.get("all_converged"),
                                  "winners_converged": rc.get("winners_converged")}
    for d in sorted(R.UNITS.glob("fine__s*")):
        r = R.rec(d.name)
        out["fine_partitions"][d.name] = {
            f"r{i}": {"F": r[f"r{i}"]["F"], "winners": [c["winner"] for c in r[f"r{i}"]["per_class"] if not c["fallback"]],
                      "effective_cells": [c.get("effective_cells") for c in r[f"r{i}"]["per_class"]],
                      "all_converged": r[f"r{i}"].get("all_converged")} for i in (1, 2)}
    for d in sorted(R.UNITS.glob("pol__s*")):
        r = R.rec(d.name)
        if "DIRECT-TASK" in (r.get("config") or ""):
            continue
        e = {"config": r.get("config"), "final": r.get("final"), "row_level_max_abs_diff": r.get("row_level_max_abs_diff"),
             "alphabets": [r.get("pair_record", {}).get("alpha1"), r.get("pair_record", {}).get("alpha2")],
             "states_emitted_fit": r.get("pair_record", {}).get("states_emitted_fit"),
             "sparsity_fit": {i: {x: (r.get("sparsity_fit") or {}).get(i, {}).get(x) for x in
                                  ("alphabet", "occupied", "singleton_cells", "singleton_fraction_of_occupied",
                                   "unseen_fraction", "entropy_nats")} for i in ("r1", "r2", "pair")},
             "perm_null_mi_fit": r.get("perm_null_mi_fit"), "privacy_term": r.get("privacy_term"),
             "summary": r.get("summary")}
        for x in ("baseline_correction", "winner", "witness_dominance", "unresolved_local_optima", "starts_not_converged"):
            if r.get(x) is not None:
                e[x] = r[x]
        out["stage_b"][d.name] = e
    (R.PKG / "OPTIMIZATION_RECEIPTS.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return len(out["stage_a"]), len(out["stage_b"])


def main(argv=None):
    part = (argv or sys.argv[1:] or ["--part", "all"])[-1]
    D = None
    if part in ("stagea", "all"):
        from qpc import data as DA
        D = DA.load()
        print("capacity curve rows", len(capacity_curve(D)))
    if part in ("stageb", "all"):
        if D is None:
            from qpc import data as DA
            D = DA.load()
        print("class preservation", class_preservation(D))
        print("optimization receipts", optimization_receipts())


if __name__ == "__main__":
    main()
