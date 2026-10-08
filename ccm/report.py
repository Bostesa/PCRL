"""Post-run reporting (role A; written after FEASIBILITY_LOCK; reads ONLY the public aggregates FEASIBILITY_GEOMETRY.json,
ORACLE_RESULTS.json and PREDICTIONS.json, never private arrays or labels). Writes PRIMARY_ENDPOINTS.csv (the registered
go-rule inputs under G), GEOMETRY_METRICS.csv (every F metric under G and the G_exp diagnostic), ALL_ARMS.csv (finite-law
oracle arms) and PREDICTION_SCORES.json.

    PYTHONPATH=. python -m ccm.report
"""
from __future__ import annotations

import csv
import json

from ccm import ids as I

GO_COVERAGE, GO_FALLBACK = 0.95, 0.05
NAMES = {1: "income", 2: "occupation"}
ARMS = ["identity_release", "decision_only", "class", "task_only", "local", "seq12_nonadaptive", "seq21_nonadaptive",
        "joint", "stochastic_local"]


def _w(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def _key(key):
    s, r = key.split("|")
    return int(s[1:]), int(r[1:])


def primary(geo):
    rows = []
    for key, v in sorted(geo["go_rule"]["rows"].items()):
        k, i = _key(key)
        rows.append({"seed": k, "recipient": i, "task": NAMES[i], "contract": "G", "F3_coverage": v["F3"],
                     "F3x_exact_income": v["F3x"], "F3u_neighbourhood": v["F3u"], "F3u_packing": v["F3u_packing"],
                     "F3u_registered": v["F3u_registered"], "F4_heldout_fallback": v["F4"],
                     "go_coverage_threshold": GO_COVERAGE, "go_fallback_threshold": GO_FALLBACK, "cell_ok": v["ok"],
                     "coverage_failure_intrinsic": v["coverage_failure_intrinsic"],
                     "coverage_failure_incomplete": v["coverage_failure_incomplete"],
                     "fallback_dominated": v["fallback_dominated"]})
    return rows


def geometry(geo):
    rows = []
    for contract, res in geo["contracts"].items():
        for key, r in sorted(res.items()):
            k, i = _key(key)
            g = r["go_inputs"]
            rows.append({"contract": contract, "role": "primary" if contract == "G" else "secondary diagnostic",
                         "seed": k, "recipient": i, "task": NAMES[i], "K": r["K"], "cap_per_class": r["cap_per_class"],
                         "n_fit": r["n_fit"], "n_held": r["n_held"], "n_tied_fit": r["n_tied_fit"],
                         "n_tied_held": r["n_tied_held"], "F1_bins": r["F1"]["n_bins"],
                         "F1_compression_ratio": r["F1"]["compression_ratio"],
                         "F2_lower_bound_bins": r["F2"].get("lower_bound_bins"),
                         "F3_coverage": g["F3_coverage"], "F3_method": r["F3"].get("method", r["F3"].get("metric")),
                         "F3_greedy_coverage": r.get("F3_greedy", {}).get("coverage"), "F3x": g.get("F3x"),
                         "F3u": g.get("F3u"), "F3u_packing": g.get("F3u_packing"),
                         "F3u_registered": g.get("F3u_registered"), "F4_fallback": g["F4_fallback_rate"],
                         "F5_CLASS_eligible": r["F5"].get("eligible"), "decision_only_eligible": False,
                         "cpu_s": r.get("cpu_s")})
    return rows


def arms(orc):
    rows = []
    for law, L in orc["results"]["laws"].items():
        for a in ARMS:
            v = L["arms"].get(a, {})
            m = v.get("measures", {})
            row = {"law": law, "arm": a, "eligible": v.get("eligible"), "design": v.get("design"),
                   "tokens_r1": (v.get("tokens") or {}).get("1"), "tokens_r2": (v.get("tokens") or {}).get("2"),
                   "reason": v.get("reason")}
            for t in ("t1", "t2", "t12"):
                row[f"mi_{t}_nats"] = m.get(t, {}).get("mi")
                row[f"bayes_{t}"] = m.get(t, {}).get("bayes_acc")
            rows.append(row)
    return rows


def predictions(geo, pred):
    G = geo["go_rule"]["rows"]
    X = geo["contracts"]["G_exp"]
    occ = [G[k] for k in G if k.endswith("r2")]
    inc = [G[k] for k in G if k.endswith("r1")]
    occ_ratio = [geo["contracts"]["G"][k]["F1"]["compression_ratio"] for k in G if k.endswith("r2")]
    outcomes = {
        "occupation F3 coverage < 0.95 (all seeds)": all(v["F3"] < GO_COVERAGE for v in occ),
        "occupation compression ratio > 0.5": all(r > 0.5 for r in occ_ratio),
        "income F3 coverage < 0.95 (all seeds)": all(v["F3"] < GO_COVERAGE for v in inc),
        "F4 fallback > 0.05 for occupation": all(v["F4"] > GO_FALLBACK for v in occ),
        "go rule fails (no pilot; PREMISE_NOT_SUPPORTED)": not geo["go_rule"]["go"],
        "occupation F3 coverage >= 0.95 at capacity under G_exp": all(
            X[k]["go_inputs"]["F3_coverage"] >= GO_COVERAGE for k in X if k.endswith("r2")),
    }
    scored = []
    for f in pred["forecasts"]:
        for q, p in f.get("probabilities", {}).items():
            if q in outcomes:
                o = outcomes[q]
                scored.append({"forecast": f["id"], "question": q, "probability": p, "outcome": o,
                               "brier": round((p - (1.0 if o else 0.0)) ** 2, 6)})
    return {"schema": "ccm-prediction-scores-v1", "scored": scored,
            "not_scored_here": "F6 toy-law forecasts are scored in FEASIBILITY_RESULTS.md from ORACLE_RESULTS.json; "
                               "F7 is a fixture, not a forecast"}


def main():
    geo = json.loads((I.PKG / "FEASIBILITY_GEOMETRY.json").read_text())
    orc = json.loads((I.PKG / "ORACLE_RESULTS.json").read_text())
    pred = json.loads((I.PKG / "PREDICTIONS.json").read_text())
    p = primary(geo)
    _w(I.PKG / "PRIMARY_ENDPOINTS.csv", p, list(p[0]))
    g = geometry(geo)
    _w(I.PKG / "GEOMETRY_METRICS.csv", g, list(g[0]))
    a = arms(orc)
    _w(I.PKG / "ALL_ARMS.csv", a, ["law", "arm", "eligible", "design", "tokens_r1", "tokens_r2", "mi_t1_nats",
                                    "bayes_t1", "mi_t2_nats", "bayes_t2", "mi_t12_nats", "bayes_t12", "reason"])
    (I.PKG / "PREDICTION_SCORES.json").write_text(json.dumps(predictions(geo, pred), indent=1) + "\n")


if __name__ == "__main__":
    main()
