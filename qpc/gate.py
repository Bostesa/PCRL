"""Stage A capacity gate (A3 inner confidence, A4 conditional rate decision). Lead-owned; locked in STAGE_A_LOCK.

Inputs (all completed Stage A units under <PRIVATE_CACHE>/qpc_v1/run/units):
  pol__s{k}__U_DIRECT-TASK_i{m1}o{m2}   the 8 global rate configurations x 3 seeds (A2)
  a1__s{k}                              A1 historical convergence diagnostic (source 20-round reproduction and the
                                        same source initialization with <= 200 rounds)
  tea__s{k}__U                          U continuous teacher (the utility anchor)
Only INNER_SELECTION true task labels are read (qpc.utility through the qpc.data allowlist); no attackers, no SEX,
no assessment rows.

Eligibility of a configuration (unchanged contract): on EACH seed, for EACH task (income, occupation_group):
  accuracy >= U - 0.01; log loss <= U + 0.01 nats; Brier <= U + 0.005; gain over the OSF_DEFENSE_FIT-majority constant
  >= 0.8 x U's gain; gain >= 0.03; exact teacher decision preservation (all rows). No mean across seeds.
Headroom preference (used ONLY to order eligible rates): log loss <= U + 0.0075 and Brier <= U + 0.0035, each task,
each seed.

Rate selection (A4; locked before Stage A):
  * no eligible configuration -> CAPACITY_GATE_NOT_MET; privacy stage not run; the best-shortfall configuration
    (smallest worst-seed normalized excess, then fewer actual total states averaged over seeds, then configuration ID)
    is packaged as a runnable INELIGIBLE code. Assessment labels are never opened.
  * otherwise select up to TWO configurations, the same rate on every seed:
      key(c) = (mean over seeds of actual total token states (alpha1 + alpha2 of the deployed policy, including the
                reserved fallback token of an absent class),
                worst-seed occupation log loss on INNER_SELECTION,
                worst-seed income log loss on INNER_SELECTION,
                configuration ID (lexicographic))
      H = eligible configurations meeting both headroom preferences on all seeds, sorted by key;
      E = the other eligible configurations, sorted by key;
      selected = (H + E)[:2]   (if |H| >= 2 this is the first two of H; otherwise H then the first of E)
  Normalized excess of one seed = max over tasks of (log-loss excess / 0.01, Brier excess / 0.005).

Outputs: <PRIVATE_CACHE>/qpc_v1/run/gate.json (full) and results/pcrl_confidence_capacity_v1/CAPACITY_GATE.json
(aggregates only), plus CONVERGENCE_DIAGNOSTIC.csv (A1 rows).
"""
from __future__ import annotations

import csv
import json

import numpy as np

RATES_I = (4, 8)
RATES_O = (8, 16, 32, 64)
SEEDS = (0, 1, 2)
TASKS = ("income", "occupation")
HEADROOM = {"ll": 0.0075, "brier": 0.0035}
ALLOW = {"acc": 0.01, "ll": 0.01, "brier": 0.005, "retention": 0.8, "gain": 0.03}
MAX_SELECTED = 2


def config_id(m1, m2, family="DIRECT-TASK", lam=None):
    return f"U|{family}|i{m1}o{m2}" + (f"|l{lam:g}" if lam is not None else "")


def stagea_ids():
    return [config_id(a, b) for a in RATES_I for b in RATES_O]


def _seed_states(rec):
    return int(rec["alpha1"]) + int(rec["alpha2"])


def summarize(per_seed):
    """per_seed: {k: {"gate": utility.gate_record(...), "util": {task: metrics}, "alpha1": int, "alpha2": int}}."""
    ks = sorted(per_seed)
    g = {k: per_seed[k]["gate"] for k in ks}
    elig = all(bool(g[k]["eligible"]) for k in ks) and len(ks) == len(SEEDS)
    head = elig and all(bool(g[k]["headroom"]) for k in ks)
    norm = [max(float(g[k][t]["norm_excess"]) for t in TASKS) for k in ks]
    return {"eligible_all_seeds": elig, "headroom_all_seeds": head,
            "eligible_by_seed": {str(k): bool(g[k]["eligible"]) for k in ks},
            "headroom_by_seed": {str(k): bool(g[k]["headroom"]) for k in ks},
            "worst_seed_norm_excess": max(norm), "norm_excess_by_seed": {str(k): n for k, n in zip(ks, norm)},
            "mean_total_states": float(np.mean([_seed_states(per_seed[k]) for k in ks])),
            "worst_seed_ll": {t: max(float(per_seed[k]["util"][t]["logloss"]) for k in ks) for t in TASKS},
            "worst_seed_excess": {t: {"ll": max(float(g[k][t]["ll_excess"]) for k in ks),
                                      "brier": max(float(g[k][t]["brier_excess"]) for k in ks)} for t in TASKS}}


def rate_key(cid, s):
    return (s["mean_total_states"], s["worst_seed_ll"]["occupation"], s["worst_seed_ll"]["income"], cid)


def shortfall_key(cid, s):
    return (s["worst_seed_norm_excess"], s["mean_total_states"], cid)


def select_rates(summary):
    """summary: {cid: summarize(...)}. Returns the locked A4 decision."""
    H = sorted((c for c, s in summary.items() if s["eligible_all_seeds"] and s["headroom_all_seeds"]),
               key=lambda c: rate_key(c, summary[c]))
    E = sorted((c for c, s in summary.items() if s["eligible_all_seeds"] and not s["headroom_all_seeds"]),
               key=lambda c: rate_key(c, summary[c]))
    sel = (H + E)[:MAX_SELECTED]
    best_short = min(summary, key=lambda c: shortfall_key(c, summary[c])) if summary else None
    return {"gate": "CAPACITY_GATE_MET" if sel else "CAPACITY_GATE_NOT_MET",
            "selected_rates": sel, "headroom_eligible": H, "other_eligible": E,
            "eligible": sorted(H + E), "best_shortfall_config": None if sel else best_short,
            "best_shortfall_label": None if sel else "INELIGIBLE (best shortfall; not ready for use)",
            "rule": "selected = (H + E)[:2]; key = (mean actual total states, worst-seed occupation log loss, "
                    "worst-seed income log loss, configuration ID)"}


def write_outputs(decision, summary, a1_rows, per_cfg_seed, pkg, run):
    full = {"schema": "qpc-capacity-gate-v1", "decision": decision, "summary": summary, "per_config_seed": per_cfg_seed,
            "a1": a1_rows, "allowances": ALLOW, "headroom": HEADROOM}
    (run / "gate.json").write_text(json.dumps(full, indent=1, default=float) + "\n")
    pub = {"schema": "qpc-capacity-gate-v1", "decision": decision, "allowances": ALLOW, "headroom": HEADROOM,
           "configs": {c: {**summary[c], "per_seed": {str(k): per_cfg_seed[c][str(k)] for k in SEEDS
                                                      if str(k) in per_cfg_seed[c]}} for c in summary},
           "note": "INNER_SELECTION only; feasibility on inner selection is not confidence preservation on the "
                   "assessment and is not a fit-distortion certificate"}
    (pkg / "CAPACITY_GATE.json").write_text(json.dumps(pub, indent=1, default=float) + "\n")
    if a1_rows:
        cols = sorted({c for r in a1_rows for c in r}, key=lambda c: (c not in ("seed", "version", "recipient"), c))
        with open(pkg / "CONVERGENCE_DIAGNOSTIC.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            for r in a1_rows:
                w.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in r.items()})
    return full
