"""Public report tables and figures for lra (lead). Reads COMMITTED result artifacts and private unit records only.
Never refits a mapper, re-solves a decoder for a new purpose, or rewrites a scientific unit (prompt section 16).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.sema --label A:report -- \\
        env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.report <part|all>

parts: fit (fitting-row tables: DECODER_ONLY_ABLATION fit part, BUDGET_FEASIBILITY, DECODER_CERTIFICATES,
       OPTIMIZATION_RECEIPTS), work (ACTUAL_WORK_ACCOUNTING, RUN_STATUS), later parts are added after the inner and
       assessment stages.
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

from lra import run as R

PKG = R.PKG
FIG = PKG / "figures"


def _jsonable(x):
    return json.loads(json.dumps(R._finite(x), allow_nan=False))


def _write_json(name, body):
    (PKG / name).write_text(json.dumps(_jsonable(body), indent=1, allow_nan=False) + "\n")


def _write_csv(name, rows, header=None):
    header = header or (list(rows[0]) if rows else ["status"])
    with open(PKG / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _rec(unit):
    return R.rec(unit) if R.done(unit) else None


# ------------------------------------------------------------------ fitting-row tables
def decoder_only_ablation_fit():
    """D0 vs D1 on IDENTICAL tokens, fitting rows (OSF_DEFENSE_FIT), every fixed map and seed (dec__ records)."""
    rows = []
    for k in R.SEEDS:
        for cid in R.d1_fixed_ids():
            r = _rec(R.unit_for(k, cid))
            if r is None:
                rows.append({"seed": k, "d1_config": cid, "status": "MISSING"})
                continue
            for i in ("1", "2"):
                f = r["fitting"][i]
                rows.append({"scope": "fitting rows (OSF_DEFENSE_FIT)", "seed": k, "d0_config": r["d0_config"],
                             "d1_config": cid, "recipient": int(i), "tokens_bitwise_equal_d0": r["tokens_bitwise_equal_d0"],
                             "L_U": f["L_U"], "L_D0": f["L_D0"], "L_D1": f["L_D1"], "dL_D1_minus_D0": f["L_D1"] - f["L_D0"],
                             "B_U": f["B_U"], "B_D0": f["B_D0"], "B_D1": f["B_D1"], "dB_D1_minus_D0": f["B_D1"] - f["B_D0"],
                             "fit_budget_ok_D0": f["L_D0"] <= f["L_U"] + 0.005 and f["B_D0"] <= f["B_U"] + 0.003,
                             "fit_budget_ok_D1": f["L_D1"] <= f["L_U"] + 0.005 and f["B_D1"] <= f["B_U"] + 0.003,
                             "tokens": f["tokens"], "occupied_fit": f["occupied_fit"], "I_fit": f["I_fit"],
                             "I12_fit": r["I12_fit"], "status": "OK"})
    return rows


def budget_feasibility():
    """Deployed fitting budgets and local caps of every new mapping unit (C-TASK, weighted, constrained)."""
    rows = []
    for k in R.SEEDS:
        for cid in R.new_fit_ids():
            r = _rec(R.unit_for(k, cid))
            if r is None:
                rows.append({"seed": k, "config": cid, "status": "MISSING"})
                continue
            b, dep = r["budgets"], r["deployed"]
            t = dep["terms"]
            row = {"seed": k, "config": cid, "arm": r["arm"], "lam": r.get("lam"), "mapper_status": r["status"],
                   "budgets_enforced": b["enforced"], "deployed_feasible": dep["feasible"], "parity_ok": dep["parity_ok"],
                   "T": t.get("T"), "Phi": t.get("Phi"), "I12": t.get("I12")}
            for i in ("1", "2"):
                row.update({f"L{i}": t.get(f"L{i}"), f"L{i}_U": b["L_U"][i], f"L{i}_limit": b["L_limit"][i],
                            f"L{i}_slack": b["L_limit"][i] - t.get(f"L{i}"), f"B{i}": t.get(f"B{i}"),
                            f"B{i}_U": b["B_U"][i], f"B{i}_limit": b["B_limit"][i],
                            f"B{i}_slack": b["B_limit"][i] - t.get(f"B{i}"), f"I{i}": t.get(f"I{i}"),
                            f"I{i}_cap": ((r.get("refs") or {}).get("I_ctask") or {}).get(i),
                            f"tokens{i}": int(np.sum(r["token_counts"][i]))})
            rows.append(row)
    return rows


def decoder_certificates():
    from lra import decoder as DEC
    body = {"schema": "lra-decoder-certificates-v1", "tolerances": DEC.TOLERANCES, "units": {}, "aggregate": {}}
    agg = {"units": 0, "tables": 0, "supervised_tokens": 0, "fallback_tokens": 0, "all_converged": True,
           "max_stationarity_rel": 0.0, "max_dual_infeas_rel": 0.0, "max_projection_magnitude": 0.0,
           "max_sum_q_residual": 0.0, "min_margin": float("inf"), "tokens_with_tie": 0}
    for k in R.SEEDS:
        for cid in R.d1_fixed_ids() + R.new_fit_ids():
            u = R.unit_for(k, cid)
            r = _rec(u)
            if r is None:
                body["units"][u] = {"status": "MISSING"}
                continue
            cs = r["certificates"]
            body["units"][u] = {"config": cid, "seed": k, "decoder_sha256": r["decoder_sha256"], "certificates": cs}
            agg["units"] += 1
            for c in cs.values():
                agg["tables"] += 1
                agg["supervised_tokens"] += c["supervised_tokens"]
                agg["fallback_tokens"] += c["fallback_tokens"]
                agg["all_converged"] = agg["all_converged"] and bool(c["all_converged"])
                for key in ("max_stationarity_rel", "max_dual_infeas_rel", "max_projection_magnitude", "max_sum_q_residual"):
                    agg[key] = max(agg[key], float(c[key]))
                agg["min_margin"] = min(agg["min_margin"], float(c["min_margin"]))
                agg["tokens_with_tie"] += int(c["tokens_with_tie"])
    body["aggregate"] = agg
    return body


def optimization_receipts():
    body = {"schema": "lra-optimization-receipts-v1", "units": {}}
    for k in R.SEEDS:
        for cid in R.new_fit_ids():
            u = R.unit_for(k, cid)
            r = _rec(u)
            if r is None:
                body["units"][u] = {"status": "MISSING"}
                continue
            w = r["work"]
            body["units"][u] = {"config": cid, "seed": k, "status": r["status"], "winner": r["winner"], "work": w,
                                "eval_ceiling_hit": w["evals"] >= w["eval_ceiling"],
                                "deployed_feasible": r["deployed"]["feasible"], "parity_ok": r["deployed"]["parity_ok"],
                                "sex_used_in_search": r["sex_used_in_search"], "rules_sha256": r["rules_sha256"],
                                "trace_sha256": r.get("trace_sha256"), "cpu_s_mapper": r.get("cpu_s_mapper")}
    return body


def actual_work_accounting():
    led = [json.loads(l) for l in (R.RUN / "COMPUTE_LEDGER.jsonl").read_text().splitlines() if l.strip()]
    units = sorted(p.name for p in R.UNITS.iterdir() if p.is_dir())
    by = {}
    for u in units:
        by.setdefault(u.split("__")[0], []).append(u)
    done = {p: sum(1 for u in v if R.done(u)) for p, v in by.items()}
    return {"schema": "lra-actual-work-v1",
            "admitted_custody_units": {"total": 189, "note": "verified copies (SOURCE_ADMISSION.json); not new fits"},
            "private_units_by_prefix": {p: {"present": len(v), "complete": done[p]} for p, v in sorted(by.items())},
            "registered": {"d1_fixed_map_decodes": len(R.d1_jobs()), "ctask": 3, "weighted": 72, "constrained": 15,
                           "inner_audit_units": len(R.scored_ids()) * 3},
            "compute_ledger": led}


PARTS = {"fit": lambda: (_write_csv("DECODER_ONLY_ABLATION.csv", decoder_only_ablation_fit()),
                         _write_csv("BUDGET_FEASIBILITY.csv", budget_feasibility()),
                         _write_json("DECODER_CERTIFICATES.json", decoder_certificates()),
                         _write_json("OPTIMIZATION_RECEIPTS.json", optimization_receipts())),
         "work": lambda: _write_json("ACTUAL_WORK_ACCOUNTING.json", actual_work_accounting())}


# ------------------------------------------------------------------ inner-row tables and figures (after select)
def _inner_rows():
    rows = list(csv.DictReader(open(PKG / "INNER_SELECTION_TABLE.csv")))
    by = {}
    for r in rows:
        by.setdefault(r["config"], {})[int(r["seed"])] = r
    return by


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def decoder_utility_ablation_inner():
    """Same tokens, D0 vs D1: fitting-row losses (dec__ records) and inner-row losses (INNER_SELECTION_TABLE), every
    fixed map and seed; roles named by the inner selection are marked (assessment contrasts are added by 'assess')."""
    by = _inner_rows()
    sel = json.loads((R.RUN / "selection.json").read_text())
    named = {p["d1"]: p["name"] for p in sel["diagnostics"]["same_map_decoder_pairs"]}
    out = []
    for cid in R.d1_fixed_ids():
        d0 = cid[:-3]
        for k in R.SEEDS:
            r = _rec(R.unit_for(k, cid))
            a1, a0 = by.get(cid, {}).get(k, {}), by.get(d0, {}).get(k, {})
            row = {"d1_config": cid, "d0_config": d0, "seed": k, "named_pair": named.get(cid, ""),
                   "tokens_bitwise_equal": r["tokens_bitwise_equal_d0"] if r else None,
                   "exposure": "fit = OSF_DEFENSE_FIT (in-sample for the U heads and for D1); inner = INNER_SELECTION (held out)"}
            for i, t in (("1", "income"), ("2", "occupation")):
                f = r["fitting"][i] if r else {}
                row.update({f"fit_dL_{t}": (f["L_D1"] - f["L_D0"]) if f else None,
                            f"fit_dB_{t}": (f["B_D1"] - f["B_D0"]) if f else None,
                            f"inner_dL_{t}": (_f(a1.get(f"ll_{t}")) - _f(a0.get(f"ll_{t}"))) if a1 and a0 else None,
                            f"inner_dB_{t}": (_f(a1.get(f"brier_{t}")) - _f(a0.get(f"brier_{t}"))) if a1 and a0 else None,
                            f"inner_ll_excess_{t}_D0": _f(a0.get(f"ll_excess_{t}")),
                            f"inner_ll_excess_{t}_D1": _f(a1.get(f"ll_excess_{t}"))})
            row.update({"inner_pair_auc_D0": _f(a0.get("auc_pair")), "inner_pair_auc_D1": _f(a1.get("auc_pair")),
                        "inner_ordinary_D0": a0.get("ordinary_seed"), "inner_ordinary_D1": a1.get("ordinary_seed")})
            out.append(row)
    return out


def decision_floor_and_feasibility():
    """CLASS (decisions alone) under D0 and D1: fitting-budget feasibility, inner utility eligibility and MEASURED inner
    attacks. Measured AUCs are finite-attacker scores, never an MI guarantee; the fitted plug-in I12 is reported
    separately (fitting rows)."""
    by = _inner_rows()
    out = []
    for cid in (R.d0_id("CLASS"), R.CLASS_D1):
        for k in R.SEEDS:
            a = by.get(cid, {}).get(k, {})
            r = _rec(R.unit_for(k, R.CLASS_D1))
            row = {"config": cid, "decoder": "D1" if cid.endswith("|D1") else "D0", "seed": k,
                   "fitted_I12_plugin_fit_rows": r["I12_fit"] if r else None}
            for i, t in (("1", "income"), ("2", "occupation")):
                f = r["fitting"][i] if r else {}
                L, B = (f.get("L_D1"), f.get("B_D1")) if row["decoder"] == "D1" else (f.get("L_D0"), f.get("B_D0"))
                row.update({f"fit_L_{t}_slack": (f["L_U"] + 0.005 - L) if f else None,
                            f"fit_B_{t}_slack": (f["B_U"] + 0.003 - B) if f else None,
                            f"inner_ll_excess_{t}": _f(a.get(f"ll_excess_{t}")),
                            f"inner_brier_excess_{t}": _f(a.get(f"brier_excess_{t}")),
                            f"inner_acc_{t}": _f(a.get(f"acc_{t}"))})
            row.update({"inner_auc_v1": _f(a.get("auc_v1")), "inner_auc_v2": _f(a.get("auc_v2")),
                        "inner_auc_pair": _f(a.get("auc_pair")), "inner_ordinary_seed": a.get("ordinary_seed"),
                        "note": "decision disclosure floor: every class-preserving release has fitted I12 >= this CLASS value; "
                                "measured AUCs are finite-attacker scores, not MI bounds"})
            out.append(row)
    return out


def figures_inner():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    by = _inner_rows()
    pts = []
    for c, seeds in by.items():
        v = list(seeds.values())
        p = R.parse_id(c)
        pts.append({"c": c, "arm": p.get("arm", p["kind"]), "dec": p.get("decoder", "-"),
                    "pair": sum(_f(x["auc_pair"]) for x in v) / len(v), "v1": sum(_f(x["auc_v1"]) for x in v) / len(v),
                    "v2": sum(_f(x["auc_v2"]) for x in v) / len(v),
                    "llx": max(_f(x["ll_excess_occupation"]) for x in v),
                    "ok": all(x["ordinary_seed"] == "True" for x in v),
                    "states": sum(_f(x["token_states"]) or 0 for x in v) / len(v) if all(_f(x["token_states"]) is not None for x in v) else None})
    made = {}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {"d0": "tab:gray", "d1_fixed": "tab:blue", "ctask": "tab:green", "weighted": "tab:orange", "constrained": "tab:red",
              "source": "k", "reference": "tab:purple"}
    for a, col in colors.items():
        q = [x for x in pts if x["arm"] == a]
        if q:
            ax.scatter([x["llx"] for x in q], [x["pair"] for x in q], c=col, s=[40 if x["ok"] else 14 for x in q],
                       marker="o", alpha=0.75, label=a, edgecolors=["k" if x["ok"] else "none" for x in q])
    ax.axvline(0.01, ls="--", c="k", lw=0.8)
    ax.text(0.0102, ax.get_ylim()[0] + 0.005, "original inner LL limit (U + 0.01)", fontsize=7)
    ax.set_xlabel("occupation inner log-loss excess over U (worst seed)")
    ax.set_ylabel("mean inner pair AUC (complete interface)")
    ax.set_title("Inner trade-off (large outlined markers = ordinarily eligible on every seed)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig1_inner_tradeoff.png", dpi=130); plt.close(fig)
    made["fig1_inner_tradeoff.png"] = "inner trade-off with the original utility limit"
    fig, ax = plt.subplots(figsize=(7, 5))
    q = [x for x in pts if x["arm"] not in ("source", "reference")]
    ax.scatter([x["v1"] for x in q], [x["pair"] for x in q], s=12, label="recipient 1 (income) vs pair", alpha=0.7)
    ax.scatter([x["v2"] for x in q], [x["pair"] for x in q], s=12, label="recipient 2 (occupation) vs pair", alpha=0.7)
    ax.set_xlabel("individual inner AUC"); ax.set_ylabel("pair inner AUC"); ax.legend(fontsize=7)
    ax.set_title("Individual and pair recovery (inner, complete interface)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig4_recovery_inner.png", dpi=130); plt.close(fig)
    made["fig4_recovery_inner.png"] = "individual/pair recovery on inner rows"
    fig, ax = plt.subplots(figsize=(7, 5))
    q = [x for x in pts if x["states"] is not None]
    ax.scatter([x["states"] for x in q], [x["pair"] for x in q], s=12, c=[colors.get(x["arm"], "k") for x in q], alpha=0.7)
    ax.set_xscale("log"); ax.set_xlabel("actual total token states (mean over seeds)"); ax.set_ylabel("mean inner pair AUC")
    ax.set_title("Actual token/support counts", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig5_token_counts.png", dpi=130); plt.close(fig)
    made["fig5_token_counts.png"] = "actual token/support counts"
    return made


def _part_inner():
    _write_csv("DECODER_UTILITY_ABLATION.csv", decoder_utility_ablation_inner())
    _write_csv("DECISION_FLOOR_AND_FEASIBILITY.csv", decision_floor_and_feasibility())
    made = figures_inner()
    idx = FIG / "FIGURES.json"
    cur = json.loads(idx.read_text()) if idx.exists() else {"schema": "lra-figures-v1", "made": {}}
    cur["made"].update(made)
    idx.write_text(json.dumps(cur, indent=1) + "\n")


PARTS["inner"] = _part_inner


# ------------------------------------------------------------------ assessment tables and figures (after infer)
def _inference():
    return json.loads((R.RUN / "inference.json").read_text())


def _lv(inf, name):
    v = inf["levels"].get(name)
    return (v["point"], v.get("se")) if v else (None, None)


def decoder_utility_ablation_assess(inf):
    """Adds the locked-assessment same-map D1-D0 contrasts (supplementary, nominal 95%) for the INNER-named pairs."""
    rows = list(csv.DictReader(open(PKG / "DECODER_UTILITY_ABLATION.csv")))
    sm = {p["d1"]: p for p in inf["same_map_decoder_contrasts"]["pairs"]}
    for r in rows:
        p = sm.get(r["d1_config"])
        for t in ("income", "occupation"):
            for kind in ("logloss", "brier"):
                c = (p or {}).get("contrasts", {}).get(f"{kind}_{t}") or {}
                r[f"assess_{kind}_{t}_mean_over_seeds_point"] = c.get("point")
                r[f"assess_{kind}_{t}_lower95"] = c.get("lower")
                r[f"assess_{kind}_{t}_upper95"] = c.get("upper")
        r["assess_verdict"] = (p or {}).get("verdict", "not an inner-named pair (not scored)")
    return rows


def figures_assess(inf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = sorted({n.split("#")[1] for n in inf["levels"] if n.startswith("Rmean#")})
    pts = []
    for lab in labs:
        pr, pse = _lv(inf, f"Rmean#{lab}#primary#pair")
        lx, lse = _lv(inf, f"llxmean#{lab}#1")
        if pr is None:
            continue
        pts.append((lab, pr, pse or 0.0, lx if lx is not None else 0.0, lse or 0.0))
    z = 1.959963984540054
    fig, ax = plt.subplots(figsize=(9, 6))
    for lab, pr, pse, lx, lse in pts:
        ax.errorbar(lx, pr, xerr=z * lse, yerr=z * pse, fmt="o", ms=4, capsize=2, alpha=0.8)
        ax.annotate(lab[:28], (lx, pr), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axvline(0.01, ls="--", c="k", lw=0.8)
    ax.set_xlabel("occupation log-loss excess over U (assessment, mean over seeds; nominal 95% bars)")
    ax.set_ylabel("pair AUC (assessment, complete interface; nominal 95% bars)")
    ax.set_title("Locked assessment trade-off (descriptive intervals; primary verdicts use the 37-slot family)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig2_assessment_tradeoff.png", dpi=130); plt.close(fig)
    rows = list(csv.DictReader(open(PKG / "DECODER_UTILITY_ABLATION.csv")))
    named = [p for p in inf["same_map_decoder_contrasts"]["pairs"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = ["fit", "inner", "assessment"]
    for p in named:
        rr = [r for r in rows if r["d1_config"] == p["d1"]]
        fit = sum(float(r["fit_dL_occupation"]) for r in rr) / len(rr) if rr else None
        inn = sum(float(r["inner_dL_occupation"]) for r in rr) / len(rr) if rr else None
        a = (p.get("contrasts", {}).get("logloss_occupation") or {})
        ys = [fit, inn, a.get("point")]
        ax.plot(xs, ys, marker="o", label=f"{p['name']}: {p['d1']}")
        if a.get("lower") is not None:
            ax.errorbar(["assessment"], [a["point"]], yerr=[[a["point"] - a["lower"]], [a["upper"] - a["point"]]], fmt="none", capsize=3, c="k")
    ax.axhline(0, c="k", lw=0.6)
    ax.set_ylabel("occupation log loss, D1 minus D0 on identical tokens (nats)")
    ax.set_title("Fixed-token decoder ablation: fitting rows vs held-out rows", fontsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "fig3_fixed_token_decoder_ablation.png", dpi=130); plt.close(fig)
    return {"fig2_assessment_tradeoff.png": "locked assessment trade-off with nominal intervals",
            "fig3_fixed_token_decoder_ablation.png": "fixed-token decoder ablation (fit, inner, assessment)"}


def run_status(inf):
    led = [json.loads(l) for l in (R.RUN / "COMPUTE_LEDGER.jsonl").read_text().splitlines() if l.strip()]
    return {"schema": "lra-run-status-v1", "label": inf["label"], "label_headline": inf["label_headline"],
            "claim_status": inf["claim_status"], "q_status": inf["q_status"], "engineering_gate": inf["engineering_gate"],
            "technical_valid": inf["technical_valid"],
            "stages_run": [{"stage": e["stage"], "shard": e.get("shard"), "at": e["at"], "cpu_s": e["cpu_s"],
                            "wall_s": e["wall_s"]} for e in led],
            "locks": {n: "pushed" for n in ("SOURCE_ADMISSION_LOCK", "CORRECTNESS_LOCK", "SCIENCE_LOCK", "EVALUATION_LOCK")},
            "finiteness": inf["finiteness_receipt"]}


def _part_assess():
    inf = _inference()
    _write_csv("DECODER_UTILITY_ABLATION.csv", decoder_utility_ablation_assess(inf))
    made = figures_assess(inf)
    idx = FIG / "FIGURES.json"
    cur = json.loads(idx.read_text()) if idx.exists() else {"schema": "lra-figures-v1", "made": {}}
    cur["made"].update(made)
    idx.write_text(json.dumps(cur, indent=1) + "\n")
    _write_json("RUN_STATUS.json", run_status(inf))


PARTS["assess"] = _part_assess


if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    for p in (PARTS if want == ["all"] else want):
        PARTS[p]()
        print("written", p)
