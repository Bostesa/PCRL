"""Stage 1: versioned corrections of the matched benchmark's exploratory arithmetic. No fits.

C1  Seed pairing: in a budget-truncated noise group, D-A and plus-minus-outputs contrasts must pair only encoder
    seeds present on BOTH sides at that sigma. Same release-draw aggregation, same bootstrap (B, seed, unit).
C2  Stable rho1^2: centre the saved canonical scores (a fixed shift; the squared correlation is shift-invariant)
    before forming weighted moments. Verified against an independent SVD calculation.

The locked benchmark modules are imported unchanged; the corrected rho statistic is injected only into this
process. Original tables are not modified; corrected copies are written with a correction_status column.
"""
import csv, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

assert os.environ.get("OMP_NUM_THREADS") == "1"
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
import stored_model_eval.bench_infer as BI  # noqa: E402
from stored_model_eval.bench_effective import BENCH_EFFECTIVE, Tracked, parse_unit  # noqa: E402
from stored_model_eval.pilot_infer import UnitBootstrap, rho_stat as rho_original  # noqa: E402

ROOT = Path.home() / "PCRL_eval_cache_private" / "bench_v1"
OLD = WT / "results" / "combined_matched_removal_benchmark_v1"
PKG = WT / "results" / "combined_output_aware_removal_v1"
OUTP = Path.home() / "PCRL_eval_cache_private" / "oar_v1" / "corrections"
OUTP.mkdir(parents=True, exist_ok=True)
E = Tracked(BENCH_EFFECTIVE)
I = BENCH_EFFECTIVE["inference"]
EX = I["exploratory"]
A2 = (1 - EX["level"]) / 2
SS = json.loads((ROOT / "infer" / "SIGMA_STAR.json").read_text())
SIG = {d: v["sigma_star"] for d, v in SS["datasets"].items()}


def rho_stable(u, v):
    """Same statistic as pilot_infer.rho_stat (weighted squared correlation), float64 with an explicit fixed
    centring shift so the uncentred moments no longer cancel catastrophically."""
    u = np.asarray(u, np.float64)
    v = np.asarray(v, np.float64)
    u = u - u.mean()
    v = v - v.mean()
    return rho_original(u, v)


def rho_svd(u, v):
    """Independent point check: squared correlation from the singular values of the unit-norm centred pair."""
    u = np.asarray(u, np.float64) - np.mean(u)
    v = np.asarray(v, np.float64) - np.mean(v)
    if not (np.any(u) and np.any(v)):
        return float("nan")
    Z = np.stack([u / np.linalg.norm(u), v / np.linalg.norm(v)], 1)
    s = np.linalg.svd(Z, compute_uv=False)
    return float(((s[0] ** 2 - s[1] ** 2) / 2) ** 2)


def units_of(cell):
    return sorted(p.name for p in (ROOT / "units").iterdir()
                  if p.name.startswith(cell + "__") and (p / "COMPLETE.json").exists())


def interval(r):
    iv, n_ne = BI._q(r, A2, 1 - A2, I["quantile_method"], A2)
    return (iv if iv else (None, None)), n_ne


def boot_for(U):
    any_u = next(iter(U.values()))
    return UnitBootstrap(any_u["preds"]["assess_unit"], EX["B"], EX["seed"], I["chunk"])


ledger, t0, c0 = [], time.time(), time.process_time()
orig_mc = {r["id"]: r for r in csv.DictReader(open(OLD / "MATCHED_COMPARISONS.csv"))}
orig_rec = {r["id"]: r for r in csv.DictReader(open(OLD / "RECOVERY.csv"))}

# ---------------------------------------------------------------- C1 seed pairing (all cells with noise groups)
c1_rows = []
ALL = sorted(p.name for p in (ROOT / "units").iterdir() if (p / "COMPLETE.json").exists())
BY_CELL = defaultdict(list)
for u in ALL:
    BY_CELL[parse_unit(u)["cell"]].append(u)
for full, us in sorted(BY_CELL.items()):
    ds = full.split("__")[0]
    groups = defaultdict(list)
    for u in us:
        i = parse_unit(u)
        groups[i["arm"] if i["arm"] != "D" else f"D_sigma{i['sigma']:g}"].append(u)
    dgroups = {k: v for k, v in groups.items() if k.startswith("D_")}
    if not dgroups or "A" not in groups:
        continue
    seeds_A = {parse_unit(u)["seed"] for u in groups["A"]}
    truncated = {k: v for k, v in dgroups.items() if {parse_unit(u)["seed"] for u in v} != seeds_A}
    for k, dv in sorted(dgroups.items()):
        seeds_D = sorted({parse_unit(u)["seed"] for u in dv})
        a_shared = [u for u in groups["A"] if parse_unit(u)["seed"] in seeds_D]
        rec = {"cell": full, "arm_group": k, "encoder_seeds_D": seeds_D, "encoder_seeds_A_original": sorted(seeds_A),
               "encoder_seeds_A_paired": sorted({parse_unit(u)["seed"] for u in a_shared}),
               "n_D_units": len(dv), "n_A_units_paired": len(a_shared),
               "release_seeds_per_encoder_seed": {s: sorted(parse_unit(u)["release_seed"] for u in dv if parse_unit(u)["seed"] == s) for s in seeds_D},
               "truncated": k in truncated}
        if k not in truncated:
            rec["status"] = "unchanged (complete seed set on both sides)"
            c1_rows.append(rec)
            continue
        U = {u: BI.load_unit(ROOT / "units", u) for u in a_shared + dv}
        g, meta = BI.build_cell(full, {"A": a_shared, k: dv}, U, E, SIG.get(ds))
        ids = list(meta["diffs"].values()) + [s for s in g.order if s.endswith("plus_NL_minus_outputs_NL|macro_auc")]
        pts, reps = BI.run(g, boot_for(U), ids)
        for (arm, kk), sid in meta["diffs"].items():
            (lo, hi), n_ne = interval(reps[sid])
            o = orig_mc.get(sid)
            c1_rows.append({**rec, "id": sid, "statistic": "|".join(kk), "kind": "paired_diff_vs_A",
                            "original_point": float(o["point"]) if o and o["point"] else None,
                            "original_lower90": float(o["lower90"]) if o and o["lower90"] else None,
                            "original_upper90": float(o["upper90"]) if o and o["upper90"] else None,
                            "corrected_point": float(pts[sid]), "corrected_lower90": lo, "corrected_upper90": hi,
                            "status": "CORRECTED"})
        for sid in [s for s in ids if s.endswith("plus_NL_minus_outputs_NL|macro_auc")]:
            if f"|{k}|" not in sid and f"|A|" not in sid:
                continue
            (lo, hi), n_ne = interval(reps[sid])
            o = orig_mc.get(sid)
            c1_rows.append({**rec, "id": sid, "statistic": "plus_NL_minus_outputs_NL|macro_auc",
                            "kind": "rep+outputs minus outputs (paired seeds)",
                            "original_point": float(o["point"]) if o and o["point"] else None,
                            "corrected_point": float(pts[sid]), "corrected_lower90": lo, "corrected_upper90": hi,
                            "status": "CORRECTED" if f"|{k}|" in sid else "PAIRED REFERENCE (A on shared seeds)"})
json.dump(c1_rows, open(OUTP / "C1_seed_pairing.json", "w"), indent=1, default=str)

# ---------------------------------------------------------------- C2 stable rho1^2 (every unit, every cell)
c2_rows, affected_cells = [], set()
for u in sorted(p.name for p in (ROOT / "units").iterdir() if (ROOT / "units" / p.name / "COMPLETE.json").exists()):
    with np.load(ROOT / "units" / u / "preds.npz") as z:
        if "RHO_u" not in z.files:
            continue
        ru, rv = z["RHO_u"], z["RHO_v"]
    W1 = np.ones((len(ru), 1))
    p_old = float(rho_original(ru, rv)(W1)[0])
    p_new = float(rho_stable(ru, rv)(W1)[0])
    p_svd = rho_svd(ru, rv)
    offset = float(abs(np.mean(ru)) / (np.std(ru) + 1e-300))
    rel = abs(p_old - p_new) / max(abs(p_new), 1e-300)
    row = {"unit": u, "cell": parse_unit(u)["cell"], "rho_u_abs_mean_over_sd": offset, "original_point": p_old,
           "stable_point": p_new, "svd_point": p_svd, "stable_vs_svd_abs": abs(p_new - p_svd), "rel_change": rel}
    c2_rows.append(row)
    if rel > 1e-6:
        affected_cells.add(parse_unit(u)["cell"])
# intervals for affected cells with the stable statistic (unit and group level)
BI.rho_stat = rho_stable
c2_int = []
for full in sorted(affected_cells):
    us = [r["unit"] for r in c2_rows if r["cell"] == full]
    groups = defaultdict(list)
    for u in us:
        i = parse_unit(u)
        groups[i["arm"] if i["arm"] != "D" else f"D_sigma{i['sigma']:g}"].append(u)
    U = {u: BI.load_unit(ROOT / "units", u) for v in groups.values() for u in v}
    g, meta = BI.build_cell(full, groups, U, E, SIG.get(full.split("__")[0]))
    ids = [sid for grp in meta["gid"].values() for k, sid in grp.items() if k == ("rep", "RHO1", "rho1sq")]
    ids += [o[("rep", "RHO1", "rho1sq")] for u, (o, _) in meta["per_unit"].items() if ("rep", "RHO1", "rho1sq") in o]
    pts, reps = BI.run(g, boot_for(U), ids)
    for sid in ids:
        (lo, hi), _ = interval(reps[sid])
        o = orig_rec.get(sid)
        c2_int.append({"id": sid, "cell": full, "original_point": float(o["point"]) if o and o["point"] else None,
                       "original_lower90": float(o["lower90"]) if o and o["lower90"] else None,
                       "original_upper90": float(o["upper90"]) if o and o["upper90"] else None,
                       "corrected_point": float(pts[sid]), "corrected_lower90": lo, "corrected_upper90": hi})
BI.rho_stat = rho_original
json.dump({"units": c2_rows, "intervals": c2_int}, open(OUTP / "C2_rho_stable.json", "w"), indent=1, default=str)

# ---------------------------------------------------------------- corrected copies of the exploratory tables
fix_mc = {r["id"]: r for r in c1_rows if r.get("status") == "CORRECTED" and "id" in r}
fix_rec = {r["id"]: r for r in c2_int if r["original_point"] is None or abs(r["corrected_point"] - r["original_point"]) > 1e-9 * max(1, abs(r["corrected_point"]))
           or (r["original_lower90"] is not None and r["corrected_lower90"] is not None and abs(r["corrected_lower90"] - r["original_lower90"]) > 1e-9)}
for name, fix in (("MATCHED_COMPARISONS", fix_mc), ("RECOVERY", fix_rec)):
    rows = list(csv.DictReader(open(OLD / f"{name}.csv")))
    cols = list(rows[0].keys()) + ["correction_status", "original_point", "original_lower90", "original_upper90"]
    n = 0
    with open(PKG / f"{name}_CORRECTED.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            c = fix.get(r["id"])
            if c:
                n += 1
                r = {**r, "original_point": r["point"], "original_lower90": r["lower90"], "original_upper90": r["upper90"],
                     "point": c["corrected_point"], "lower90": c["corrected_lower90"], "upper90": c["corrected_upper90"],
                     "correction_status": "C1_seed_pairing" if name == "MATCHED_COMPARISONS" else "C2_stable_rho1sq"}
                if name == "MATCHED_COMPARISONS":
                    r["decisions"] = json.dumps({"vs_0": BI._decide(c["corrected_lower90"], c["corrected_upper90"], 0.0)})
            else:
                r = {**r, "correction_status": "unchanged"}
            w.writerow(r)
    print(name, "corrected rows:", n)
summary = {"C1_truncated_groups": sorted({r["arm_group"] for r in c1_rows if r.get("truncated")}),
           "C1_corrected_rows": len(fix_mc), "C2_units": len(c2_rows),
           "C2_units_rel_change_gt_1e-6": sum(r["rel_change"] > 1e-6 for r in c2_rows),
           "C2_max_stable_vs_svd": max(r["stable_vs_svd_abs"] for r in c2_rows if np.isfinite(r["svd_point"])),
           "C2_corrected_table_rows": len(fix_rec), "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
json.dump(summary, open(OUTP / "SUMMARY.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
