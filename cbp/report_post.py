"""Post-lock REPORTING completion (lead; written after the assessment; changes no scientific procedure, selection,
endpoint or label). Disclosed in VALIDATION.md section 6 and COST_AND_CLOSEOUT.md section 5. The locked cbp/report.py and
every file it wrote are kept unchanged; this module only ADDS files:

  ENDPOINT_RECEIPTS.json        prompt section 13 diagnostics of the 9 reused task-only reference units (DIRECT-TASK,
                                FINE-TASK, CLASS-ONLY x 3 seeds), copied from the aggregate section-13 blocks of
                                run/endpoint_parity.json (DIRECT-TASK has no permutation-null block: recorded as such)
  INNER_STATES_VS_RECOVERY_v2.csv   the locked table plus the U|DIRECT-TASK|i8o64 row (Q), whose occupied fitting states
                                the locked reader skipped (they live in endpoint_parity.json, not in the unit record)
  figures/fig1b_inner_tradeoff  worst-seed inner occupation log-loss excess vs mean inner pair AUC per family (lines
                                ordered by lambda), headroom 0.006 and allowance 0.01, with T*, Q and U markers
  figures/fig2_annotated        figure 2 with a role legend and a zoom inset on the crowded task-code region
  figures/fig3_annotated        figure 3 with recipient names (1 = income, 2 = occupation)
  figures/fig4_annotated        figure 4 with Q added and a family legend

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.report_post
"""
from __future__ import annotations

import csv
import json

import numpy as np

from cbp import run as R
from cbp.report import FAM_STYLE, _dump, _plt, _save

SECTION13_FIELDS = ("alphabets", "states_per_predicted_class", "total_occupied_states_fit", "entropy_nats_fit",
                    "singletons_fit", "lt5_fit", "unseen_fraction_fit", "support_per_class_fit", "fallback_tokens",
                    "objective_terms_deployed", "mi_vs_permutation_null", "search")
OTHER_STYLE = {"FINE-TASK": ("black", "P"), "DIRECT-TASK": ("dimgray", "X"), "CLASS": ("purple", "v"),
               "SRC": ("goldenrod", "*"), "REF": ("gray", "x")}


def _style(fam):
    return FAM_STYLE.get(fam) or OTHER_STYLE.get(fam) or ("gray", "x")


def endpoint_receipts():
    par = json.loads((R.RUN / "endpoint_parity.json").read_text())
    out = {"schema": "cbp-endpoint-receipts-v1",
           "source": "run/endpoint_parity.json section-13 blocks (aggregate; written by cbp.fit.endpoint_parity at stage fit)",
           "note": "reporting completion after the assessment; no unit was refit; the 72 privacy units are in "
                   "OPTIMIZATION_RECEIPTS.json",
           "mi_null": "where present: the qpc fit-receipt permutation null (100 permutations, seed 20261006)",
           "units": {}}
    for k in R.SEEDS:
        for cid in R.task_ids():
            n = R.unit_for(k, cid)
            s13 = (par.get(n) or {}).get("section13") or {}
            e = {"config": cid, "seed": k, "parity_ok": (par.get(n) or {}).get("ok")}
            for f in SECTION13_FIELDS:
                if f in s13:
                    e[f] = s13[f]
            if "mi_vs_permutation_null" not in s13:
                e["mi_vs_permutation_null"] = {"status": "NOT_RECORDED", "reason": "the reused-unit parity block "
                                               "for this reference computed fitted terms but no permutation null"}
            out["units"][n] = e
    out["counts"] = {"endpoint_units": len(out["units"])}
    _dump("ENDPOINT_RECEIPTS.json", out)
    return out


def states_v2(rec):
    rows = list(csv.DictReader(open(R.PKG / "INNER_STATES_VS_RECOVERY.csv")))
    S = json.loads((R.RUN / "selection.json").read_text())["rows"]
    q = "U|DIRECT-TASK|i8o64"
    occ = [rec["units"][R.unit_for(k, q)]["alphabets"]["occupied_fit"] for k in R.SEEDS]
    rows.insert(0, {"config": q, "family": "DIRECT-TASK",
                    "occupied_states_fit_mean": f"{np.mean([o['r1'] + o['r2'] for o in occ]):.6f}",
                    "inner_pair_auc": f"{S[q]['mean_pair']:.6f}"})
    with open(R.PKG / "INNER_STATES_VS_RECOVERY_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[1]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


def fig1b():
    S = json.loads((R.RUN / "selection.json").read_text())["rows"]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for fam, (col, mk) in FAM_STYLE.items():
        pts = sorted((R.parse_id(c)["lam"], r) for c, r in S.items() if r.get("family") == fam and r.get("ok"))
        x = [max(s["ll_excess"]["occupation"] for s in r["seeds"].values()) for _, r in pts]
        y = [r["mean_pair"] for _, r in pts]
        ax.plot(x, y, marker=mk, color=col, label=f"{fam} (lambda 0.01 -> 0.1)")
        for (lam, _), xi, yi in zip(pts, x, y):
            ax.annotate(f"{lam:g}", (xi, yi), fontsize=6, xytext=(2, 2), textcoords="offset points")
    for c, lab in (("U|FINE-TASK|i8o64", "T* (FINE-TASK)"), ("U|DIRECT-TASK|i8o64", "Q (DIRECT-TASK)")):
        r = S[c]
        col, mk = _style(r["family"])
        ax.scatter([max(s["ll_excess"]["occupation"] for s in r["seeds"].values())], [r["mean_pair"]], color=col,
                   marker=mk, s=60, label=lab, zorder=5)
    ax.axhline(S["SRC|U"]["mean_pair"], color="goldenrod", ls=":", lw=1, label="U continuous (no protection)")
    ax.axvline(0.006, color="k", ls="--", lw=1, label="headroom 0.006 (inner rule)")
    ax.axvline(0.01, color="k", ls="-", lw=1, label="original allowance 0.01")
    ax.set(xlabel="worst-seed INNER occupation log-loss excess vs U (nats)", ylabel="mean INNER pair SEX AUC",
           title="Figure 1b: inner privacy vs confidence trade-off (all 24 configurations)")
    ax.legend(fontsize=7)
    _save(fig, "fig1b_inner_tradeoff")
    plt.close(fig)


def _rows():
    return list(csv.DictReader(open(R.PKG / "ASSESSMENT_COMPARISON.csv")))


def _f(x):
    return None if x in (None, "") else float(x)


def fig2_annotated(z):
    rows = _rows()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    axin = ax.inset_axes([0.55, 0.50, 0.42, 0.46])
    for r in rows:
        x, y = _f(r.get("ll_excess_occupation")), _f(r.get("auc_pair"))
        if x is None or y is None:
            continue
        col, mk = _style(r.get("family") or ("SRC" if r["release"].startswith("SRC") else "REF"))
        xe, ye = z * (_f(r.get("ll_excess_occupation_se")) or 0), z * (_f(r.get("auc_pair_se")) or 0)
        lab = r["release"].replace("U|", "").replace("|i8o64", "") + (f"  [{r['roles']}]" if r.get("roles") else "")
        for a in (ax, axin):
            a.errorbar(x, y, xerr=xe, yerr=ye, fmt=mk, color=col, ms=5, capsize=2,
                       label=lab if a is ax else None)
    cls = next((r for r in rows if r["release"] == "U|CLASS|i1o1"), None)
    for a in (ax, axin):
        a.axvline(0.006, color="k", ls="--", lw=1)
        a.axvline(0.01, color="k", ls="-", lw=1)
    if cls:
        ax.axhline(float(cls["auc_pair"]), color="purple", ls=":", lw=1)
    axin.set_xlim(0.0, 0.012)
    axin.set_ylim(0.80, 0.86)
    axin.tick_params(labelsize=6)
    axin.set_title("zoom: task-only and privacy codes", fontsize=7)
    ax.set(xlabel="assessment occupation log-loss excess vs U (point +- z SE; z = 3.2048)",
           ylabel="assessment pair SEX AUC",
           title="Figure 2: locked scored list\n(dashed: 0.006 inner headroom; solid: 0.01 allowance; dotted: CLASS-ONLY)")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    _save(fig, "fig2_annotated")
    plt.close(fig)


def fig3_annotated(z):
    rows = _rows()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(rows)), 4.8))
    x = np.arange(len(rows))
    for o, v, nm in zip((-0.27, 0, 0.27), ("v1", "v2", "pair"),
                        ("recipient 1 (income)", "recipient 2 (occupation)", "pair (both recipients)")):
        ax.bar(x + o, [_f(r.get(f"auc_{v}")) or np.nan for r in rows], 0.27,
               yerr=[z * (_f(r.get(f"auc_{v}_se")) or 0) for r in rows], label=nm, capsize=1)
    ax.set_xticks(x, [r["release"].replace("U|", "").replace("|i8o64", "") + (f"\n[{r['roles']}]" if r.get("roles")
                                                                               else "") for r in rows],
                  rotation=60, ha="right", fontsize=6)
    ax.set(ylabel="assessment SEX AUC", ylim=(0.5, 1.0),
           title="Figure 3: individual-recipient vs pair recovery (locked scored list; +- z SE)")
    ax.legend(fontsize=7)
    _save(fig, "fig3_annotated")
    plt.close(fig)


def fig4_annotated(states, z):
    rows = {r["release"]: r for r in _rows()}
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    seen = set()
    for r in states:
        fam = r["family"]
        col, mk = _style(fam)
        ax[0].scatter(float(r["occupied_states_fit_mean"]), float(r["inner_pair_auc"]), color=col, marker=mk, s=22,
                      label=None if fam in seen else fam)
        seen.add(fam)
        a = rows.get(r["config"])
        if a and a.get("auc_pair"):
            ax[1].errorbar(float(r["occupied_states_fit_mean"]), float(a["auc_pair"]),
                           yerr=z * (_f(a.get("auc_pair_se")) or 0), fmt=mk, color=col)
            ax[1].annotate(r["config"].replace("U|", "").replace("|i8o64", ""),
                           (float(r["occupied_states_fit_mean"]), float(a["auc_pair"])), fontsize=5.5)
    ax[0].set(xlabel="occupied fitting states (recipient 1 + 2, seed mean)", ylabel="mean INNER pair AUC",
              title="inner (all 24 privacy configurations + task-only codes)")
    ax[1].set(xlabel="occupied fitting states (recipient 1 + 2, seed mean)", ylabel="assessment pair AUC",
              title="assessment (locked scored list; +- z SE)")
    ax[0].legend(fontsize=7)
    fig.suptitle("Figure 4: token counts vs recovery (is protection mostly coarser compression?)", fontsize=9)
    _save(fig, "fig4_annotated")
    plt.close(fig)


def main():
    inf = json.loads((R.RUN / "inference.json").read_text())
    z = inf["z"]
    rec = endpoint_receipts()
    st = states_v2(rec)
    fig1b()
    fig2_annotated(z)
    fig3_annotated(z)
    fig4_annotated(st, z)
    print("endpoint receipts", rec["counts"], "| states rows", len(st))


if __name__ == "__main__":
    main()
