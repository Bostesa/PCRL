"""Independent recomputation of the prospective 2016 decisions from committed aggregates.

Reads only the published aggregate files at the pinned evidence commit
(results/pcrl_final_prospective_v1/{INFERENCE_2016,PRIMARY_CLAIMS,SECONDARY_CONTRASTS}.json)
via `git show`. No individual-level 2016 data is opened.

Recomputes every one-sided 97.5% bound (estimate + z*SE), every clause decision, both
ten-clause conjunctions, the secondary Bonferroni critical value and intervals, and a set of
interpretation checks (joint vs simultaneous, task sign vs material margin, D17 dominance).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from statistics import NormalDist

PIN = "5e154e5c4fdaeb23d327a0ebefe838525f1a19cb"
BASE = "results/pcrl_final_prospective_v1/"
N = NormalDist()


def load(name):
    raw = subprocess.check_output(["git", "show", f"{PIN}:{BASE}{name}"])
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def rows_with_prefix(obj, prefix):
    out = []
    if isinstance(obj, dict):
        if str(obj.get("id", "")).startswith(prefix) and "estimate" in obj:
            out.append(obj)
        for v in obj.values():
            out += rows_with_prefix(v, prefix)
    elif isinstance(obj, list):
        for v in obj:
            out += rows_with_prefix(v, prefix)
    return out


def main():
    inf, inf_sha = load("INFERENCE_2016.json")
    reg, reg_sha = load("PRIMARY_CLAIMS.json")
    sec_reg, sec_sha = load("SECONDARY_CONTRASTS.json")
    z1 = N.inv_cdf(1 - 0.025)
    assert abs(z1 - reg["z_one_sided"]) < 1e-12
    prim = {r["id"]: r for r in rows_with_prefix(inf, "primary|")}
    assert len(prim) == 20, len(prim)
    thresholds = {e["id"]: e["threshold"] for e in reg["endpoints"]}
    out = {"pin": PIN, "sha256": {"INFERENCE_2016.json": inf_sha, "PRIMARY_CLAIMS.json": reg_sha,
                                   "SECONDARY_CONTRASTS.json": sec_sha}, "primary": {}, "checks": []}
    max_err = 0.0
    for cand in ("Q", "D17"):
        rows = [prim[k] for k in sorted(prim) if prim[k]["candidate"] == cand]
        assert len(rows) == 10
        clauses = []
        for r in rows:
            ub = r["estimate"] + z1 * r["bootstrap_se"]
            max_err = max(max_err, abs(ub - r["upper_bound_97_5"]))
            thr = thresholds[r["id"]]
            passed = ub <= thr
            assert passed == r["passed"], r["id"]
            clauses.append({"id": r["id"], "clause": r["clause"], "weighting": r["weighting"],
                            "estimate": r["estimate"], "se": r["bootstrap_se"], "ub": ub, "threshold": thr,
                            "passed": passed, "ub_below_zero": ub < 0,
                            "estimate_meets_threshold": r["estimate"] <= thr,
                            "z_needed_to_meet_threshold": (thr - r["estimate"]) / r["bootstrap_se"]})
        task = [c for c in clauses if c["clause"] == "task"]
        sens = [c for c in clauses if c["clause"] != "task"]
        out["primary"][cand] = {
            "clauses_passed": sum(c["passed"] for c in clauses),
            "conjunction_passed": all(c["passed"] for c in clauses),
            "sensitive_passed": sum(c["passed"] for c in sens),
            "task_passed": sum(c["passed"] for c in task),
            "task_ub_negative_both": all(c["ub_below_zero"] for c in task),
            "task_point_estimate_beyond_margin_both": all(c["estimate_meets_threshold"] for c in task),
            "max_sensitive_ub": max(c["ub"] for c in sens),
            "clauses": clauses,
        }
    out["max_abs_ub_error"] = max_err
    # simultaneous coverage implied by 10 pointwise one-sided .975 bounds (Bonferroni lower bound)
    out["bonferroni_simultaneous_coverage_lower_bound_10_bounds"] = 1 - 10 * 0.025
    out["bonferroni_simultaneous_coverage_lower_bound_8_bounds"] = 1 - 8 * 0.025
    # the sensitive clauses would also pass at a Bonferroni-corrected one-sided level over 8 (and 10, 20)
    for m in (8, 10, 20):
        zm = N.inv_cdf(1 - 0.025 / m)
        out[f"sensitive_all_pass_at_one_sided_bonferroni_{m}"] = {
            c: all(x["estimate"] + zm * x["se"] <= x["threshold"] for x in out["primary"][c]["clauses"]
                   if x["clause"] != "task") for c in ("Q", "D17")}
        out[f"task_ub_negative_at_one_sided_bonferroni_{m}"] = {
            c: all(x["estimate"] + zm * x["se"] < 0 for x in out["primary"][c]["clauses"]
                   if x["clause"] == "task") for c in ("Q", "D17")}
    # secondary
    zs = N.inv_cdf(1 - 0.05 / (2 * 70))
    assert abs(zs - inf["secondary"]["z"]) < 1e-9
    sec = inf["secondary"]["rows"]
    assert len(sec) == 70
    serr = 0.0
    for r in sec:
        lo, hi = r["estimate"] - zs * r["bootstrap_se"], r["estimate"] + zs * r["bootstrap_se"]
        serr = max(serr, abs(lo - r["lower"]), abs(hi - r["upper"]))
        assert (lo > 0 or hi < 0) == r["excludes_zero"], r["id"]
    out["secondary_max_abs_interval_error"] = serr
    d17 = [r for r in sec if r["comparator"] == "D17"]
    out["Q_vs_D17"] = {
        "rows": [{k: r[k] for k in ("id", "estimate", "lower", "upper", "excludes_zero")} for r in d17],
        "D17_task_better_resolved": [r["weighting"] for r in d17 if "utility" in r["role"] and r["lower"] > 0],
        "any_sensitive_resolved": any(r["excludes_zero"] for r in d17 if "attack" in r["role"]),
        "sensitive_point_estimates_favoring_Q": sum(r["estimate"] < 0 for r in d17 if "attack" in r["role"]),
        "full_domination_by_D17": all((r["lower"] > 0) for r in d17),
    }
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
