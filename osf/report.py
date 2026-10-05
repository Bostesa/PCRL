"""Reports (public aggregates only): STRENGTH_PROFILES.csv, STRENGTH_COMPARISON.json, RAW_FIDELITY.json,
CRITIC_TRACKING.csv, ACTUAL_TASK_UTILITY.csv, LINEAR_DIAGNOSTICS.csv and the frontier figure.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m osf.report [--part strength|fidelity|tracking|outer|all]

Strength statistics come from the per-step receipts (steps.npz) of every run unit. "Realized" statistics count zero
steps as 0; "conditional" statistics (nonzero steps only) are labelled as such and are never called realized strength.
The combined ratio is ||q_enc|| / ||t_enc|| over both encoder blocks; the RMS ratio is sqrt(mean_i (q_i/t_i)^2).
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

from osf import run as R
from osf import train as T

Z_ACTIVE = (T.Z_NONE,)


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "rms": None, "median": None, "p10": None, "p90": None}
    return {"n": int(len(x)), "mean": float(x.mean()), "rms": float(np.sqrt(np.mean(x * x))),
            "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}


def run_units():
    return [d.name for d in sorted(R.UNITS.glob("run__s*")) if R.done(d.name) and "nonfinite_original" not in d.name]


def strength_rows(name):
    rec = R.rec(name)
    s = np.load(R.U(name) / "steps.npz")
    cid, k = rec["config"], rec["seed"]
    per_ep = int(np.ceil(len(s["epoch"]) / max(1, int(s["epoch"].max()) + 1)))
    rows = []

    def add(scope, quantity, st):
        for key, v in st.items():
            rows.append({"config": cid, "seed": k, "scope": scope, "quantity": quantity, "stat": key, "value": v})

    for i in (0, 1):
        z = s["zero"][:, i]
        act = (z == T.Z_NONE)
        add("run", f"realized_ratio_enc{i + 1}", _stats(s["realized_ratio"][:, i]))
        add("run", f"conditional_ratio_enc{i + 1}_nonzero_steps", _stats(s["ratio"][act, i]))
        add("run", f"cos_t_p_enc{i + 1}", _stats(s["cos"][:, i]))
        add("run", f"t_norm_enc{i + 1}", _stats(s["t_norm"][:, i]))
        add("run", f"q_norm_enc{i + 1}", _stats(s["q_norm"][:, i]))
        add("run", f"post_clip_update_norm_enc{i + 1}", _stats(s["post_enc"][:, i]))
        rows.append({"config": cid, "seed": k, "scope": "run", "quantity": f"zero_fraction_enc{i + 1}", "stat": "value",
                     "value": float(((z != T.Z_NONE) & (z != T.Z_OFF)).mean())})
        rows.append({"config": cid, "seed": k, "scope": "run", "quantity": f"cap_fraction_enc{i + 1}", "stat": "value",
                     "value": float(s["cap"][:, i].mean())})
    add("run", "combined_ratio", _stats(s["comb_ratio"]))
    add("run", "rms_ratio_realized", _stats(s["rms_ratio"]))
    rows.append({"config": cid, "seed": k, "scope": "run", "quantity": "clip_fraction", "stat": "value",
                 "value": float((s["kappa"] < 1).mean())})
    for e in range(int(s["epoch"].max()) + 1):
        m = s["epoch"] == e
        for i in (0, 1):
            rows.append({"config": cid, "seed": k, "scope": f"epoch_{e:02d}", "quantity": f"realized_ratio_enc{i + 1}",
                         "stat": "mean", "value": float(s["realized_ratio"][m, i].mean())})
        rows.append({"config": cid, "seed": k, "scope": f"epoch_{e:02d}", "quantity": "combined_ratio", "stat": "mean",
                     "value": float(np.nanmean(s["comb_ratio"][m]))})
        rows.append({"config": cid, "seed": k, "scope": f"epoch_{e:02d}", "quantity": "clip_fraction", "stat": "mean",
                     "value": float((s["kappa"][m] < 1).mean())})
    return rows


def write_strength():
    rows = []
    for n in run_units():
        rows += strength_rows(n)
    with open(R.PKG / "STRENGTH_PROFILES.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "seed", "scope", "quantity", "stat", "value"], lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({kk: (f"{v:.6g}" if isinstance(v, float) else v) for kk, v in r.items()})
    comparison()
    return len(rows)


def _pooled(cid, key, i=None, fn=None):
    vals = []
    for k in R.SEEDS:
        n = R.run_name(k, cid)
        if not R.done(n):
            return None
        s = np.load(R.U(n) / "steps.npz")
        x = s[key] if i is None else s[key][:, i]
        vals.append(fn(s, x) if fn else x)
    return np.concatenate(vals)


def comparison():
    """Does a constant normalized ratio approximate the incumbent's distribution and time profile of strengths?"""
    out = {"incumbent": "RAW-J|b0.3", "note": "realized ratios pooled over seeds and steps (zeros count 0); early = "
           "epochs 0-4, late = epochs 30-39; equal average norm does not equalise direction, interference, clipping "
           "or trajectory", "configs": {}}
    for cid in ("RAW-J|b0.1", "RAW-J|b0.3", "RAW-J|b0.6", "RAW-L|b0.3", "NORM-J|r1.5|a1", "NORM-J|r3|a1",
                "NORM-J|r5|a1", "NORM-L|r3|a1"):
        r = {}
        for i in (0, 1):
            x = _pooled(cid, "realized_ratio", i)
            if x is None:
                break
            ep = _pooled(cid, "epoch")
            r[f"enc{i + 1}"] = {**_stats(x), "early_mean": float(x[ep <= 4].mean()), "late_mean": float(x[ep >= 30].mean()),
                                "cv": float(x.std() / x.mean()) if x.mean() > 0 else None}
        if not r:
            continue
        c = _pooled(cid, "comb_ratio")
        rr = _pooled(cid, "rms_ratio")
        ep = _pooled(cid, "epoch")
        r["combined"] = {**_stats(c), "early_mean": float(np.nanmean(c[ep <= 4])), "late_mean": float(np.nanmean(c[ep >= 30]))}
        r["rms"] = {**_stats(rr), "early_mean": float(rr[ep <= 4].mean()), "late_mean": float(rr[ep >= 30].mean())}
        r["cos_mean"] = [float(np.nanmean(_pooled(cid, "cos", i))) for i in (0, 1)]
        r["clip_fraction"] = float((_pooled(cid, "kappa") < 1).mean())
        out["configs"][cid] = r
    (R.PKG / "STRENGTH_COMPARISON.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def write_fidelity():
    out = {"schema": "osf-raw-fidelity-v1", "parity": {}, "fidelity_2_epoch": {}, "replays_40_epoch": {}}
    for d in sorted(R.UNITS.glob("parity__*")):
        if R.done(d.name):
            r = R.rec(d.name)
            out["parity"][d.name] = {"pass": r["pass"], "critic_online_updates": r["critic_online_updates"]}
    for d in sorted(R.UNITS.glob("fid__*")):
        if R.done(d.name):
            r = R.rec(d.name)
            out["fidelity_2_epoch"][d.name] = {kk: r[kk] for kk in ("bitwise_model", "bitwise_critics",
                                                                    "logged_norm_max_rel_dev", "pass")}
            out["fidelity_2_epoch"][d.name]["equivalence"] = {
                "equivalent": r["equivalence"]["equivalent"], "applicable": r["equivalence"]["applicable"],
                "rel_err": [e["rel_err"] for e in r["equivalence"]["encoders"]],
                "r_i": [e["r_i"] for e in r["equivalence"]["encoders"]],
                "expected_failure_common_rho_equivalent": r["expected_failure_common_rho"]["equivalent"],
                "expected_failure_cap_equivalent": r["expected_failure_cap"]["equivalent"]}
    for k in R.SEEDS:
        for cid in R.admitted_ids():
            n = R.run_name(k, cid)
            if R.done(n):
                r = R.rec(n)
                out["replays_40_epoch"][n] = {"replay_of": r["replay_of"], "bitwise": r["bitwise"],
                                              "wall_s": r["wall_s"]}
    out["all_pass"] = (all(v["pass"] for v in out["parity"].values()) and
                       all(v["pass"] for v in out["fidelity_2_epoch"].values()) and
                       all(all(v["bitwise"].values()) for v in out["replays_40_epoch"].values()))
    out["counts"] = {kk: len(out[kk]) for kk in ("parity", "fidelity_2_epoch", "replays_40_epoch")}
    (R.PKG / "RAW_FIDELITY.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def write_tracking():
    rows = []
    for d in sorted(R.UNITS.glob("track__*")):
        if not R.done(d.name):
            continue
        r = R.rec(d.name)
        for snap, views in r["snapshots"].items():
            for v, x in views.items():
                for rowset in ("calib", "inner"):
                    rows.append({"config": r["config"], "seed": r["seed"], "snapshot": snap, "view": v, "rows": rowset,
                                 "gap_registered": x[f"gap_registered_{rowset}"],
                                 "gap_best_of_bank": x[f"gap_best_of_bank_{rowset}"],
                                 "online_best_ce": x[f"online_best_{rowset}"], "fresh_best_ce": x[f"fresh_best_{rowset}"],
                                 "const_ce": x["const_ce"][rowset]})
    if rows:
        with open(R.PKG / "CRITIC_TRACKING.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            for q in rows:
                w.writerow({kk: (f"{v:.6f}" if isinstance(v, float) else v) for kk, v in q.items()})
    return len(rows)


def main(argv=None):
    part = (argv or sys.argv[1:] or ["--part", "all"])[-1]
    if part in ("strength", "all"):
        print("strength rows", write_strength())
    if part in ("fidelity", "all"):
        print("fidelity all_pass", write_fidelity()["all_pass"])
    if part in ("tracking", "all"):
        print("tracking rows", write_tracking())


if __name__ == "__main__":
    main()
