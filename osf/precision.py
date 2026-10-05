"""PRECISION_PLANNING.json (prompt section 6), computed BEFORE any new nonzero fit from previously saved paired outcomes
only: the closed smf study's per-row assessment predictions on ITS 3,796 NEW_DEVELOPMENT_ASSESSMENT rows (committed
outcomes; permitted planning evidence) and its published endpoint SEs. Nothing here reads the consolidated assessment's
labels or produces a consolidated-cohort performance summary.

Planning rule for the one-point accuracy guard (clauses 4-5: lower bound of Acc(nominee) - Acc(U) > -0.01 with
z = Phi^-1(1 - 0.05/54)): if the true loss is d, the lower bound is about -d - z SE, so it passes only if
SE < (0.01 - d) / z. Planning SEs for the eligible cohort scale the smf paired SEs by sqrt(n_smf / n_osf) — an
assumption (pools differ in composition; a new model's paired SE may be larger or smaller), never a bound.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

SMF_UNITS = Path.home() / "PCRL_eval_cache_private" / "smf_v1" / "run" / "units"
Z27 = float(norm.ppf(1 - 0.05 / 54))
N_SMF = 3796
PAIRS = [("RAW-J", "U"), ("RAW-L", "U"), ("J-F", "U"), ("L-F", "U")]
B_PLAN, SEED_PLAN = 1000, 20261099      # planning-only bootstrap (not the registered inference seed)


def load(label, k):
    return np.load(SMF_UNITS / f"outer__s{k}__{label}" / "preds.npz")


def paired_acc_se(a, b):
    zs = {lab: [load(lab, k) for k in (0, 1, 2)] for lab in (a, b)}
    z0 = zs[a][0]
    units = z0["assess_unit"]
    ug, inv = np.unique(units, return_inverse=True)
    out = {}
    for t, (hk, yk) in {"income": ("hard1", "y_income"), "occupation_group": ("hard2", "y_occ")}.items():
        ca = np.mean([(z[hk] == z[yk]).astype(float) for z in zs[a]], 0)
        cb = np.mean([(z[hk] == z[yk]).astype(float) for z in zs[b]], 0)
        assert all(np.array_equal(z["assess_unit"], units) for zz in zs.values() for z in zz)
        d_row = ca - cb
        gsum = np.bincount(inv, d_row, minlength=len(ug))
        gcnt = np.bincount(inv, minlength=len(ug)).astype(float)
        rng = np.random.default_rng(SEED_PLAN)
        reps = []
        for _ in range(B_PLAN):
            w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
            reps.append((w * gsum).sum() / (w * gcnt).sum())
        out[t] = {"point_on_smf_rows": float(d_row.mean()), "paired_group_bootstrap_se_smf": float(np.std(reps, ddof=1))}
    return out


def main(n_osf):
    scale = float(np.sqrt(N_SMF / n_osf))
    need = {f"true_loss_{d:g}": (0.01 - d) / Z27 for d in (0.0, 0.003, 0.006)}
    pairs = {}
    for a, b in PAIRS:
        r = paired_acc_se(a, b)
        for t in r:
            r[t]["planning_se_osf"] = r[t]["paired_group_bootstrap_se_smf"] * scale
            r[t]["guard_resolvable_at_true_loss"] = {k: bool(r[t]["planning_se_osf"] < v) for k, v in need.items()}
        pairs[f"{a} - {b}"] = r
    pub = {"P01 R_pair(L-F)-R_pair(J-F)": 0.001847, "P10 R_pair(C*)-R_pair(J-F)": 0.004365,
           "P02 R_v1 diff": 0.003003, "P03 R_v2 diff": 0.001798, "P06 gain retention income": 0.001806,
           "P07 gain retention occupation": 0.002527, "P08 gain income": 0.006887, "P09 gain occupation": 0.008537}
    out = {"schema": "osf-precision-planning-v1",
           "computed_before_new_nonzero_fits": True,
           "evidence": "closed smf study: per-row assessment predictions (outer__s{0,1,2}__{label}, its 3,796 rows) and "
                       "its published PRIMARY_ENDPOINTS.csv SEs; no consolidated-assessment label read",
           "z_27_slots": Z27, "family": "27 primary slots, two-sided Bonferroni Phi^-1(1 - 0.05/54)",
           "guard": "clauses 4-5: lower bound of Acc(nominee) - Acc(U) > -0.01",
           "se_needed_for_guard": need, "worked_example": "true loss 0.006 -> SE < (0.01 - 0.006)/z = %.6f" % need["true_loss_0.006"],
           "n_smf_assessment": N_SMF, "n_osf_assessment": n_osf, "scale_sqrt_n_ratio": scale,
           "paired_accuracy_planning": pairs,
           "published_smf_se_scaled": {k: {"smf_se": v, "planning_se_osf": v * scale} for k, v in pub.items()},
           "bootstrap_for_planning": {"B": B_PLAN, "seed": SEED_PLAN, "unit": "exact-record group (assess_unit)",
                                      "note": "planning only; registered inference uses B = 1999, seed 20261006"},
           "caveats": ["scaled historical SEs are planning assumptions, not lower bounds on any new model's paired SE",
                       "the margin is not weakened, no pool is dropped and no impossibility is predeclared",
                       "if final uncertainty crosses a guard, unresolved precision is distinguished from a point effect "
                       "beyond the guard"]}
    return out


if __name__ == "__main__":
    n = int(sys.argv[1])
    o = main(n)
    p = Path(sys.argv[2])
    p.write_text(json.dumps(o, indent=1) + "\n")
    print(json.dumps({k: o[k] for k in ("z_27_slots", "se_needed_for_guard", "scale_sqrt_n_ratio")}, indent=1))
    for k, v in o["paired_accuracy_planning"].items():
        print(k, {t: (round(x["point_on_smf_rows"], 4), round(x["paired_group_bootstrap_se_smf"], 5),
                      round(x["planning_se_osf"], 5)) for t, x in v.items()})
