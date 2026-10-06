"""Reports (public aggregates only). Lead-owned. Adapted from qpc/report.py + qpc/report_outer.py at d0c8a45.

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.report --part fits|inner|outer|all

fits  (after Stage 2; fitting rows / label-free role indices only):
  CLASS_PRESERVATION.json    every policy unit: released decision == argmax of the decoded prototype == teacher decision
                             on every row of every role (row checks and failed checks separately)
  OPTIMIZATION_RECEIPTS.json per privacy unit: final objective terms, section-13 diagnostics (new fits: record["cbp"];
                             reused: run/endpoint_parity.json), search work, convergence, unresolved local optima, witness
                             dominance, the JOINT / sequential computational asymmetry, and exact pair aliases per seed
inner (after selection; INNER_SELECTION only):
  figures/fig1_inner_lambda_curve.{pdf,png}  inner pair AUC and worst-seed log-loss / Brier excess vs lambda per family,
                             with the 0.006 headroom line and the 0.01 original allowance
  figures/fig4_states_vs_recovery.{pdf,png}  (inner panel) actual occupied fitting states vs inner pair AUC
outer (after the assessment and inference; locked scored list only):
  ASSESSMENT_COMPARISON.csv  absolute pair / individual AUC, both accuracies, both confidence costs, inner eligibility
  figures/fig2_pair_vs_occ_ll.{pdf,png}      assessment pair AUC vs occupation log-loss excess (point +- z SE, z of the
                             primary family), CLASS-ONLY decision-only level, 0.006 inner headroom and 0.01 allowance
  figures/fig3_individual_vs_pair.{pdf,png}  v1 / v2 / pair assessment AUC per scored release
  figures/fig4_states_vs_recovery.{pdf,png}  adds the assessment panel
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

from cbp import run as R

TASKS = ("income", "occupation")
FAM_STYLE = {"LOCAL": ("tab:blue", "o"), "SEQ-12": ("tab:orange", "s"), "SEQ-21": ("tab:green", "^"),
             "JOINT": ("tab:red", "D")}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (R.PKG / "figures").mkdir(parents=True, exist_ok=True)
    return plt


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(R.PKG / "figures" / f"{name}.pdf")
    fig.savefig(R.PKG / "figures" / f"{name}.png", dpi=120)


def _write_csv(path, rows):
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.6f}" if isinstance(r.get(c), float) else r.get(c)) for c in cols})


def _dump(name, obj):
    (R.PKG / name).write_text(json.dumps(R._finite(obj), indent=1, allow_nan=False) + "\n")


# ------------------------------------------------------------------ fits
def class_preservation(D):
    roles = {r: np.asarray(ix) for r, ix in D["idx"].items() if r in ("DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT",
                                                                        "INNER_SELECTION", "OSF_DEVELOPMENT_ASSESSMENT")}
    out = {"rule": "released decision = argmax of the decoded prototype = teacher decision, every row of every role "
                   "(label-free: role row indices and decisions only)", "units": {}, "all_pass": True,
           "rows_checked": 0, "failed_checks": 0}
    for d in sorted(R.UNITS.glob("pol__s*")):
        if not R.done(d.name):
            continue
        z, rec = R.npz(d.name, "release.npz"), R.rec(d.name)
        k = int(d.name.split("__")[1][1:])
        T = R.teacher(k)
        ok, mism = {}, {}
        for i in (1, 2):
            eq = (z[f"hard{i}"] == T[f"d{i}"]) & (np.asarray(z[f"q{i}"]).argmax(1) == T[f"d{i}"])
            ok[i] = bool(eq.all())
            mism[i] = {r: int((~eq[ix]).sum()) for r, ix in roles.items()}
            out["rows_checked"] += int(len(eq))
            out["failed_checks"] += int((~eq).sum())
        out["units"][d.name] = {"config": rec.get("config"), "seed": k, "decisions_equal_teacher": ok,
                                "mismatches_by_role": mism, "alphabet": [int(z["alpha1"]), int(z["alpha2"])]}
        out["all_pass"] &= all(ok.values())
    out["n_units"] = len(out["units"])
    _dump("CLASS_PRESERVATION.json", out)
    return out["all_pass"], out["n_units"], out["rows_checked"]


def optimization_receipts():
    from cbp import fit as FT
    par = json.loads((R.RUN / "endpoint_parity.json").read_text()) if (R.RUN / "endpoint_parity.json").exists() else {}
    out = {"schema": "cbp-optimization-receipts-v1", "units": {}, "aliases": {}, "counts": {},
           "notes": ["objectives are fitting-row training criteria (plug-in MI, KL to the teacher); not privacy bounds",
                     "plug-in MI at these alphabets is biased upward: read every fitted MI beside its permutation null",
                     "JOINT dominance is asserted on the fitting F_joint over its unchanged witnesses only (not on "
                     "components, held-out rows or DIRECT-TASK)",
                     "JOINT has 5 search paths and 4 witnesses vs one two-stage path per sequential arm: not matched "
                     "compute and not the official Taylor solver"]}
    pols = {}
    for k in R.SEEDS:
        for cid in R.privacy_ids():
            n = R.unit_for(k, cid)
            if not R.done(n):
                out["units"][n] = {"config": cid, "seed": k, "status": "MISSING"}
                continue
            r = R.rec(n)
            s13 = (r.get("cbp") or {}).get("section13") or (par.get(n) or {}).get("section13")
            e = {"config": cid, "seed": k, "origin": r.get("origin", "REUSED_AFTER_PARITY"), "final": r.get("final"),
                 "row_level_max_abs_diff": r.get("row_level_max_abs_diff"), "section13": s13,
                 "summary": r.get("summary"), "work": r.get("work"), "cpu_s": r.get("cpu_s"),
                 "computational_asymmetry": (r.get("cbp") or {}).get("computational_asymmetry")}
            for x in ("winner", "witness_dominance", "unresolved_local_optima", "starts_not_converged"):
                if r.get(x) is not None:
                    e[x] = r[x]
            out["units"][n] = e
            pols.setdefault(k, {})[cid] = json.loads((R.U(n) / "policy.json").read_text())
        for cid in R.task_ids():
            n = R.unit_for(k, cid)
            if R.done(n):
                pols.setdefault(k, {})[cid] = json.loads((R.U(n) / "policy.json").read_text())
    for k, P in pols.items():
        out["aliases"][str(k)] = FT.aliases(P)
    out["counts"] = {"units": len(out["units"]), "missing": sum(e.get("status") == "MISSING" for e in out["units"].values()),
                     "new_fits": sum(e.get("origin") == "NEW_FIT" for e in out["units"].values()),
                     "pair_aliases_per_seed": {k: v["pair_alias_count"] for k, v in out["aliases"].items()},
                     "units_with_unresolved_local_optima": sum(bool(e.get("unresolved_local_optima"))
                                                               for e in out["units"].values()),
                     "units_with_unconverged_starts": sum(bool(e.get("starts_not_converged"))
                                                         for e in out["units"].values())}
    _dump("OPTIMIZATION_RECEIPTS.json", out)
    return out["counts"]


# ------------------------------------------------------------------ inner
def _occupied(k, cid):
    n = R.unit_for(k, cid)
    try:
        r = R.rec(n)
    except Exception:                                                      # noqa: BLE001
        return None
    s13 = (r.get("cbp") or {}).get("section13")
    if s13 and s13.get("alphabets"):
        return s13["alphabets"]["occupied_fit"]["r1"] + s13["alphabets"]["occupied_fit"]["r2"]
    sp = r.get("sparsity_fit") or {}
    if "r1" in sp and "r2" in sp:
        return sp["r1"].get("occupied", 0) + sp["r2"].get("occupied", 0)
    return None


def inner_figures():
    S = json.loads((R.RUN / "selection.json").read_text())
    rows = S["rows"]
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for fam, (col, mk) in FAM_STYLE.items():
        pts = sorted((R.parse_id(c)["lam"], r) for c, r in rows.items() if r.get("family") == fam and r.get("ok"))
        lam = [p[0] for p in pts]
        ax[0].plot(lam, [p[1]["mean_pair"] for p in pts], marker=mk, color=col, label=fam)
        for j, (t, a) in enumerate((("occupation", ax[1]), ("income", ax[1]))):
            a.plot(lam, [max(s["ll_excess"][t] for s in p[1]["seeds"].values()) for p in pts], marker=mk, color=col,
                   ls="-" if t == "occupation" else ":", label=f"{fam} {t}" if fam == "LOCAL" or t == "occupation" else None)
        ax[2].plot(lam, [max(s["brier_excess"][t] for s in p[1]["seeds"].values() for t in TASKS) for p in pts],
                   marker=mk, color=col, label=fam)
    for c, ls in (("U|DIRECT-TASK|i8o64", "--"), ("U|FINE-TASK|i8o64", "-."), ("SRC|U", ":")):
        if rows.get(c, {}).get("ok"):
            ax[0].axhline(rows[c]["mean_pair"], color="grey", ls=ls, lw=1, label=c.replace("U|", "").replace("|i8o64", ""))
    ax[0].set(xlabel="lambda", ylabel="mean INNER pair SEX AUC", title="Inner privacy (pair recovery)")
    ax[1].axhline(0.006, color="k", ls="--", lw=1, label="headroom 0.006 (inner rule)")
    ax[1].axhline(0.01, color="k", ls="-", lw=1, label="original allowance 0.01")
    ax[1].set(xlabel="lambda", ylabel="worst-seed log-loss excess vs U (nats)",
              title="Inner confidence cost (solid occupation, dotted income)")
    ax[2].axhline(0.0035, color="k", ls="--", lw=1, label="headroom 0.0035")
    ax[2].axhline(0.005, color="k", ls="-", lw=1, label="allowance 0.005")
    ax[2].set(xlabel="lambda", ylabel="worst task/seed Brier excess vs U", title="Inner Brier cost")
    for a in ax:
        a.legend(fontsize=7)
    fig.suptitle("Figure 1 (INNER_SELECTION only; the assessment never scores the full grid)", fontsize=9)
    _save(fig, "fig1_inner_lambda_curve")
    plt.close(fig)
    pts = []
    for c, r in rows.items():
        if not r.get("ok") or R.parse_id(c)["kind"] != "policy":
            continue
        occ = [_occupied(k, c) for k in R.SEEDS]
        if any(o is None for o in occ):
            continue
        pts.append({"config": c, "family": r["family"], "occupied_states_fit_mean": float(np.mean(occ)),
                    "inner_pair_auc": r["mean_pair"]})
    _write_csv(R.PKG / "INNER_STATES_VS_RECOVERY.csv", pts)
    return len(rows), len(pts)


# ------------------------------------------------------------------ outer
def _lv(L, key):
    v = L.get(key)
    return (v or {}).get("point"), (v or {}).get("se")


def outer_tables():
    inf = json.loads((R.RUN / "inference.json").read_text())
    S = json.loads((R.RUN / "selection.json").read_text())
    EL = json.loads((R.PKG / "EVALUATION_LOCK.json").read_text())
    L, z = inf["levels"], inf["z"]
    roles = {}
    for x, c in EL["resolved"].items():
        if c:
            roles.setdefault(c, []).append(x + ("" if EL["statuses"][x]["status"] == "NOMINEE" else " (fallback)"))
    rows = []
    for lab in EL["scored_labels"]:
        r = {"release": lab, "roles": "; ".join(roles.get(lab, [])), "family": S["rows"].get(lab, {}).get("family"),
             "inner_ordinary": S["rows"].get(lab, {}).get("ordinary"),
             "inner_headroom": S["rows"].get(lab, {}).get("headroom")}
        for v in ("v1", "v2", "pair"):
            r[f"auc_{v}"], r[f"auc_{v}_se"] = _lv(L, f"Rmean#{lab}#primary#{v}")
        for j, t in enumerate(TASKS):
            r[f"acc_{t}"], _ = _lv(L, f"accmean#{lab}#{j}")
            if lab != "SRC|U":
                r[f"ll_excess_{t}"], r[f"ll_excess_{t}_se"] = _lv(L, f"llxmean#{lab}#{j}")
                r[f"brier_excess_{t}"], r[f"brier_excess_{t}_se"] = _lv(L, f"brxmean#{lab}#{j}")
        rows.append(r)
    _write_csv(R.PKG / "ASSESSMENT_COMPARISON.csv", rows)
    _outer_figures(rows, z)
    return len(rows)


def _outer_figures(rows, z):
    plt = _plt()
    cls = next((r for r in rows if r["release"] == "U|CLASS|i1o1"), None)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r in rows:
        if r.get("ll_excess_occupation") is None or r.get("auc_pair") is None:
            continue
        col, mk = FAM_STYLE.get(r.get("family") or "", ("grey", "x"))
        ax.errorbar(r["ll_excess_occupation"], r["auc_pair"], xerr=z * (r.get("ll_excess_occupation_se") or 0),
                    yerr=z * (r.get("auc_pair_se") or 0), fmt=mk, color=col, ms=5, capsize=2)
        ax.annotate(r["release"].replace("U|", "").replace("|i8o64", ""), (r["ll_excess_occupation"], r["auc_pair"]),
                    fontsize=6)
    if cls and cls.get("auc_pair") is not None:
        ax.axhline(cls["auc_pair"], color="grey", ls=":", label="CLASS-ONLY (decisions only)")
    ax.axvline(0.006, color="k", ls="--", lw=1, label="inner headroom 0.006")
    ax.axvline(0.01, color="k", ls="-", lw=1, label="allowance 0.01")
    ax.set(xlabel="assessment occupation log-loss excess vs U (point +- z SE)", ylabel="assessment pair SEX AUC",
           title="Figure 2: locked scored list (z of the 37-slot family)")
    ax.legend(fontsize=7)
    _save(fig, "fig2_pair_vs_occ_ll")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(rows)), 4.5))
    x = np.arange(len(rows))
    for o, v in zip((-0.27, 0, 0.27), ("v1", "v2", "pair")):
        ax.bar(x + o, [r.get(f"auc_{v}") or np.nan for r in rows], 0.27,
               yerr=[z * (r.get(f"auc_{v}_se") or 0) for r in rows], label=v, capsize=1)
    ax.set_xticks(x, [r["release"].replace("U|", "").replace("|i8o64", "") for r in rows], rotation=60, ha="right",
                  fontsize=7)
    ax.set(ylabel="assessment SEX AUC", ylim=(0.5, 1.0), title="Figure 3: individual-recipient vs pair recovery")
    ax.legend(fontsize=7)
    _save(fig, "fig3_individual_vs_pair")
    plt.close(fig)
    inner = list(csv.DictReader(open(R.PKG / "INNER_STATES_VS_RECOVERY.csv"))) \
        if (R.PKG / "INNER_STATES_VS_RECOVERY.csv").exists() else []
    occ = {r["config"]: float(r["occupied_states_fit_mean"]) for r in inner}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for r in inner:
        col, mk = FAM_STYLE.get(r["family"], ("grey", "x"))
        ax[0].scatter(float(r["occupied_states_fit_mean"]), float(r["inner_pair_auc"]), color=col, marker=mk, s=18)
    for r in rows:
        if r["release"] in occ and r.get("auc_pair") is not None:
            col, mk = FAM_STYLE.get(r.get("family") or "", ("grey", "x"))
            ax[1].errorbar(occ[r["release"]], r["auc_pair"], yerr=z * (r.get("auc_pair_se") or 0), fmt=mk, color=col)
    ax[0].set(xlabel="occupied fitting states (r1 + r2, seed mean)", ylabel="mean INNER pair AUC", title="inner")
    ax[1].set(xlabel="occupied fitting states (r1 + r2, seed mean)", ylabel="assessment pair AUC",
              title="assessment (locked list)")
    fig.suptitle("Figure 4: token counts vs recovery (is protection mostly coarser compression?)", fontsize=9)
    _save(fig, "fig4_states_vs_recovery")
    plt.close(fig)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=("fits", "inner", "outer", "all"))
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    if a.part in ("fits", "all"):
        from cbp import data as DA
        D = DA.load()
        print("class preservation", class_preservation(D))
        print("optimization receipts", optimization_receipts())
    if a.part in ("inner", "all"):
        print("inner figures", inner_figures())
    if a.part in ("outer", "all"):
        print("outer rows", outer_tables())


if __name__ == "__main__":
    main()
