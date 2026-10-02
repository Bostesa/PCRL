"""Benchmark report tables (from BENCH_INFER.json + unit records only) and the synthetic runtime calibration.

Tables written to the package directory (aggregates only; no row-level values, no private absolute paths):
  PRIMARY_ENDPOINTS.csv, NATIVE_AND_HELDOUT_CHECKS.csv, RECOVERY.csv, UTILITY.csv, MATCHED_COMPARISONS.csv,
  SUPPORT_COVERAGE.csv, SEED_VARIATION.csv, PRIMARY_FAMILY.json, MODEL_MANIFEST.json, SIGMA_STAR_SUMMARY.json,
  notes/evaluation/CONSUMPTION.json
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from .admission import sha256_file
from .bench_effective import consumption_report, parse_unit, tier1_units, tier2_units

CONTRACT = {"rep": "C_rep", "rep+outputs": "C_rep_plus_clean_out", "outputs": "outputs_only",
            "label_only": "label_only (reference)", "constant": "constant (reference)"}


def _w(path: Path, rows: list[dict], cols: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else (json.dumps(r[c]) if isinstance(r.get(c), (dict, list))
                                                        else r[c])) for c in cols})


def _meaning(stat, dec):
    if stat == "G1_r2":
        return {"ESTABLISHED_ABOVE": "held-out fixed-ridge R2 above tau (does not generalise)",
                "ESTABLISHED_BELOW": "held-out fixed-ridge R2 below tau"}.get(dec, dec)
    if stat.endswith("macro_auc"):
        return {"ESTABLISHED_ABOVE": "recovery above 0.55 established (outside the method's scope)",
                "ESTABLISHED_BELOW": "recovery below 0.55 established"}.get(dec, dec)
    return {"NONINFERIOR": "U2 accuracy non-inferior to untreated (margin 1 pp)",
            "INFERIOR": "U2 accuracy inferior to untreated beyond 1 pp"}.get(dec, dec)


def _native_by_unit(inf: dict) -> dict:
    return inf.get("native") or {}


def report_bench(root: Path, pkg: Path) -> dict:
    root, pkg = Path(root), Path(pkg)
    from .guards import inside_git_worktree
    inf = json.loads((root / "infer" / "BENCH_INFER.json").read_text())
    native = _native_by_unit(inf)
    ex = inf["exploratory"]

    # ---- PRIMARY_ENDPOINTS ------------------------------------------------------------------------------------
    prim = []
    for e in inf["primary"]["endpoints"]:
        members = e.get("members") or []
        nat = [native.get(u) or {} for u in members]
        if e["arm"] == "A":
            ns = [n.get("category") for n in nat]
            c1 = any(str(x).startswith("above tau") for x in ns)
        elif e["arm"] in ("B", "C"):
            ns = [n.get("status") for n in nat]
            c1 = any(x == "FAIL" for x in ns)
        else:
            ns, c1 = ["NA (no native certificate)"] * len(nat), False
        if e["decision"] in ("NE", "UNRESOLVED"):
            cat = "C5 (not established)"
        elif e["statistic"] == "U2_accuracy_diff_vs_A":
            cat = e["decision"]
        elif c1 and e["statistic"] != "U2_accuracy_diff_vs_A":
            cat = "C1 (native check fails as specified) in some encoder seed"
        elif e["decision"] == "ESTABLISHED_ABOVE":
            cat = "C2 (held-out quantity fails)" if e["statistic"] == "G1_r2" else "C3 (recovery outside scope)"
        else:
            cat = "none (established below)"
        prim.append({**e, "meaning": _meaning(e["statistic"], e["decision"]), "category": cat,
                     "native_status_per_seed": ns, "n_members": len(members),
                     "adjusted": f"Bonferroni simultaneous ({inf['primary']['family_size']})"})
    _w(pkg / "PRIMARY_ENDPOINTS.csv", prim,
       ["id", "dataset", "cell", "arm", "arm_group", "statistic", "bar", "point", "lower", "upper", "decision",
        "meaning", "status", "category", "native_status_per_seed", "sigma_star", "sigma_star_flagged", "alias_of",
        "n_members", "alpha_each", "B", "seed", "tail_count", "resolution", "n_ne_replicates",
        "n_replicates_below_bar", "n_replicates_above_bar", "reason", "note", "adjusted"])
    (pkg / "PRIMARY_FAMILY.json").write_text(json.dumps(
        {k: inf["primary"][k] for k in ("family", "family_size", "alpha_family", "alpha_each", "B", "seed",
                                         "tail_count", "adjustment", "validity_note")}, indent=1))

    # ---- NATIVE_AND_HELDOUT_CHECKS ----------------------------------------------------------------------------
    unit_rows = {}
    for r in ex:
        if r.get("level") == "unit":
            unit_rows[(r["unit"], r["surface"], r["recipe"], r["metric"])] = r
    nh = []
    for u in sorted(native):
        n = native[u] or {}
        i = parse_unit(u)
        row = {"unit": u, "dataset": i["dataset"], "cell": i["cell"], "encoder_seed": i["seed"], "arm": i["arm_tag"],
               "native_scope": {"A": "historical in-sample one-hot ridge R2 on the test split (tau 0.05)",
                                "B": "LEACE implementation bound on defense_fit (target concept)",
                                "C": "LEACE implementation bound on defense_fit (policy concept)",
                                "D": "none (unclipped Gaussian)"}[i["arm"]],
               "native_status": n.get("status") or n.get("category")}
        if i["arm"] == "A":
            hm, f6 = n.get("historical_mixed_precision", {}), n.get("float64", {})
            rp = n.get("reproduction", {})
            row.update(N0_mixed_clamped=hm.get("clamped"), N0_mixed_raw=hm.get("raw"), N0_float64_raw=f6.get("raw"),
                       N0_rows=n.get("n_rows"), historical_r2_onehot=rp.get("historical_r2_onehot"),
                       abs_diff_vs_historical=rp.get("abs_diff"), reproduction=rp.get("status"))
        elif i["arm"] in ("B", "C"):
            pr, ez, oos = n.get("primary", {}), n.get("exact_zero_crosscov", {}), n.get("out_of_support", {})
            row.update(bound_whitened_residual=pr.get("whitened_residual_spectral_norm"), svd_tol=pr.get("svd_tol"),
                       bound_holds=pr.get("holds"), exact_zero_status=ez.get("status"),
                       crosscov_max_abs_rel_erased=ez.get("crosscov_max_abs_rel_erased"),
                       ols_r2_joint_erased=ez.get("ols_r2_joint_erased"),
                       n_singular_values_nonzero_truncated=ez.get("n_singular_values_nonzero_truncated"),
                       defense_fit_cov_rank=oos.get("fit_cov_rank"),
                       oos_rel_median_attacker_fit=(oos.get("attacker_fit") or {}).get("oos_rel_median"),
                       oos_rel_median_assessment=(oos.get("assessment") or {}).get("oos_rel_median"),
                       oos_rel_max_assessment=(oos.get("assessment") or {}).get("oos_rel_max"))
        for nm, key in (("G1", ("rep", "G1", "r2")), ("G2", ("rep", "G2", "r2")), ("rho1sq", ("rep", "RHO1", "rho1sq")),
                        ("crosscov_target_fro", ("rep", "crosscov_target", "fro")),
                        ("crosscov_policy_fro", ("rep", "crosscov_policy", "fro"))):
            r = unit_rows.get((u,) + key)
            if r:
                row[f"heldout_{nm}"] = r["point"]
                row[f"heldout_{nm}_lower90"] = r["lower90"]
                row[f"heldout_{nm}_upper90"] = r["upper90"]
        nh.append(row)
    for r in ex:
        if r.get("level") == "group" and r["surface"] == "rep" and r["recipe"] in ("G1", "G2", "RHO1", "crosscov_target",
                                                                                  "crosscov_policy"):
            nh.append({"unit": f"{r['cell']}|{r['arm_group']} (seed mean)", "dataset": r["dataset"], "cell": r["cell"],
                       "arm": r["arm_group"], f"heldout_{r['recipe'] if r['recipe'] != 'RHO1' else 'rho1sq'}"
                       + ("_fro" if r["recipe"].startswith("crosscov") else ""): r["point"],
                       "group_metric": f"{r['recipe']}|{r['metric']}", "group_point": r["point"],
                       "group_lower90": r["lower90"], "group_upper90": r["upper90"],
                       "group_decision_tau": (r.get("decisions") or {}).get("tau=0.05")})
    _w(pkg / "NATIVE_AND_HELDOUT_CHECKS.csv", nh,
       ["unit", "dataset", "cell", "encoder_seed", "arm", "native_scope", "native_status", "N0_mixed_clamped",
        "N0_mixed_raw", "N0_float64_raw", "N0_rows", "historical_r2_onehot", "abs_diff_vs_historical", "reproduction",
        "bound_whitened_residual", "svd_tol", "bound_holds", "exact_zero_status", "crosscov_max_abs_rel_erased",
        "ols_r2_joint_erased", "n_singular_values_nonzero_truncated", "defense_fit_cov_rank",
        "oos_rel_median_attacker_fit", "oos_rel_median_assessment", "oos_rel_max_assessment",
        "heldout_G1", "heldout_G1_lower90", "heldout_G1_upper90", "heldout_G2", "heldout_G2_lower90",
        "heldout_G2_upper90", "heldout_rho1sq", "heldout_rho1sq_lower90", "heldout_rho1sq_upper90",
        "heldout_crosscov_target_fro", "heldout_crosscov_policy_fro", "group_metric", "group_point",
        "group_lower90", "group_upper90", "group_decision_tau"])

    # ---- RECOVERY / UTILITY / MATCHED ---------------------------------------------------------------------------
    rec, ut, mc = [], [], []
    for r in ex:
        surf = r.get("surface")
        base = {**r, "contract": CONTRACT.get(surf, surf), "decisions": r.get("decisions")}
        if r.get("level") == "paired_diff_vs_A" or "minus" in r["id"]:
            mc.append({**base, "comparison": r["arm_group"] if r.get("level") == "paired_diff_vs_A" else r["id"].split(
                "|", 1)[1], "paired_on": "identical assessment people (same bootstrap replicate)"})
        elif surf in ("U1", "U2", "clean_output", "Uconst"):
            note = {"U1": "frozen head on the transformed representation: compatibility diagnostic; outside LEACE's "
                          "scope (Linear-ReLU-Linear)",
                    "U2": "common LR probe (capability retained in the representation)",
                    "clean_output": "C_rep_plus_clean_out task utility = the clean head; unchanged across methods",
                    "Uconst": "attacker_fit majority class"}[surf]
            ut.append({**base, "note": note})
        elif surf in CONTRACT:
            rec.append(base)
    for w in inf.get("worst", []):
        rec.append({**w, "level": "group", "metric": w["statistic"], "lower90": w.get("lower"),
                    "upper90": w.get("upper"), "contract": CONTRACT.get(w["surface"]),
                    "interval_kind": "simultaneous over K supported classes/pairs (alpha/K)"})
    for e in prim:
        if e["statistic"] == "U2_accuracy_diff_vs_A":
            mc.append({"id": e["id"], "cell": e["cell"], "dataset": e["dataset"], "comparison": f"{e['arm_group']}-A",
                       "level": "primary", "surface": "U2", "recipe": "U2", "metric": "accuracy", "point": e["point"],
                       "lower90": None, "upper90": None, "primary_lower": e["lower"], "primary_upper": e["upper"],
                       "decision": e["decision"], "label": "primary (Bonferroni one-sided, NI margin -0.01)"})
    cols_r = ["id", "cell", "dataset", "contract", "level", "arm_group", "unit", "surface", "recipe", "metric",
              "statistic", "K", "argmax", "point", "lower90", "upper90", "decisions", "status", "n_members",
              "n_ne_replicates", "boot_B", "boot_seed", "interval_kind", "label"]
    _w(pkg / "RECOVERY.csv", rec, cols_r)
    _w(pkg / "UTILITY.csv", ut, ["id", "cell", "dataset", "contract", "level", "arm_group", "unit", "surface", "recipe", "metric",
                                 "point", "lower90", "upper90", "status", "n_members", "note", "label"])
    _w(pkg / "MATCHED_COMPARISONS.csv", mc,
       ["id", "cell", "dataset", "comparison", "level", "surface", "recipe", "metric", "point", "lower90", "upper90",
        "decisions", "primary_lower", "primary_upper", "decision", "paired_on", "status", "label"])

    # ---- SUPPORT_COVERAGE -------------------------------------------------------------------------------------
    sc = []
    for cell, s in sorted(inf["support"].items()):
        for what in ("sensitive", "task"):
            rr = s[what]
            for k in range(rr["n_classes"]):
                sc.append({"cell": cell, "what": what, "class": k,
                           **{f"n_{r}": rr["counts_per_role"][r][k] for r in rr["counts_per_role"]},
                           "supported": k in rr["supported_classes"],
                           "reason": (rr["unsupported_classes"].get(str(k)) or {}).get("detail"),
                           "cell_status": rr["status"],
                           "class_coverage": f"{rr['coverage']['classes'][0]}/{rr['coverage']['classes'][1]}",
                           "pair_coverage": f"{rr['coverage']['pairs'][0]}/{rr['coverage']['pairs'][1]}",
                           "assessment_row_coverage": rr["coverage"]["assessment_row_fraction_in_supported_classes"]})
        for arm, c in (s.get("concept") or {}).items():
            sc.append({"cell": cell, "what": f"eraser_concept_{arm}", "class": None, "supported": c["status"] == "FITTABLE",
                       "reason": c.get("classes_below_floor") or None, "cell_status": c["status"],
                       "n_defense_fit": c["defense_fit_counts"]})
    for u in inf.get("units_missing", []):
        sc.append({"cell": parse_unit(u)["cell"], "what": "unit", "unit": u, "cell_status": "MISSING_OUTPUTS"})
    for u, why in (inf.get("ne_units") or {}).items():
        sc.append({"cell": parse_unit(u)["cell"], "what": "unit", "unit": u, "cell_status": f"NE: {why}"})
    _w(pkg / "SUPPORT_COVERAGE.csv", sc,
       ["cell", "what", "unit", "class", "n_attacker_fit", "n_attacker_val", "n_assessment", "n_defense_fit",
        "supported", "reason", "cell_status", "class_coverage", "pair_coverage", "assessment_row_coverage"])

    # ---- SEED_VARIATION ---------------------------------------------------------------------------------------
    _w(pkg / "SEED_VARIATION.csv", inf.get("seed_variation", []),
       ["cell", "dataset", "arm_group", "surface", "recipe", "metric", "n_encoder_seeds", "n_release_seeds",
        "n_attacker_seeds", "encoder_seed_sd", "release_seed_sd_mean", "attacker_seed_sd_mean",
        "attacker_seed_sd_max", "per_encoder_seed_point", "per_unit_attacker_seed_points", "definition"])

    # ---- MODEL_MANIFEST ---------------------------------------------------------------------------------------
    mm = []
    for base, kind in (("units", "unit"), ("shared/outputs_only", "outputs_only"), ("shared/U2", "U2_probe")):
        d0 = root / base
        if not d0.exists():
            continue
        for d in sorted(p for p in d0.iterdir() if p.is_dir() and ".partial" not in p.name):
            comp = d / "COMPLETE.json"
            files = json.loads(comp.read_text())["files"] if comp.exists() else {}
            for rel, h in sorted(files.items()):
                if rel.startswith("models/"):
                    mm.append({"kind": f"{kind}_model", "owner": d.name, "path": f"{base}/{d.name}/{rel}",
                               "sha256": h, "model": Path(rel).stem})
    pins_p = root / "defenses" / "MAP_PINS.json"
    if pins_p.exists():
        for mid, rec_ in json.loads(pins_p.read_text())["maps"].items():
            for rel, h in rec_["files"].items():
                mm.append({"kind": "official_leace_map", "owner": mid, "path": f"defenses/{mid}/{rel}", "sha256": h})
    (pkg / "MODEL_MANIFEST.json").write_text(json.dumps(
        {"schema": "bench_model_manifest/v1", "paths_relative_to": "the private root (~/PCRL_eval_cache_private/bench_v1)",
         "n_files": len(mm), "by_kind": {k: sum(1 for m in mm if m["kind"] == k) for k in sorted({m["kind"] for m in mm})},
         "files": mm}, indent=1))
    ss = inf["sigma_star"]
    (pkg / "SIGMA_STAR_SUMMARY.json").write_text(json.dumps(
        {"sha256_of_private_SIGMA_STAR_json": ss["sha256"], "values": ss["values"],
         "datasets": {d: {k: v for k, v in x.items() if k in ("cell", "sigma_star", "flagged_fallback",
                                                              "val_curve_mean", "supported_classes")}
                      for d, x in ss["datasets"].items()},
         "selected_on": "attacker_val only (mean over encoder seeds x release seeds, attacker seed 0)"}, indent=1))
    # ---- consumption (units + stages + inference) ---------------------------------------------------------------
    consumed = set(inf.get("effective_keys_consumed_in_infer", []))
    for u in inf.get("units_found", []):
        f = root / "units" / u / "fit_records.json"
        if f.exists():
            consumed |= set(json.loads(f.read_text()).get("effective_keys_consumed_in_unit", []))
    led = root / "logs" / "BUDGET_LEDGER.json"
    ledger = json.loads(led.read_text()) if led.exists() else {"entries": []}
    for e in ledger.get("entries", []):
        consumed |= set(e.get("consumed", []))
    cons = consumption_report(consumed, json.loads((root / "units" / inf["units_found"][0] / "fit_records.json")
                                                   .read_text())["effective_protocol"]) if inf.get("units_found") else {}
    tier2_only = [k for k in cons.get("not_consumed", []) if k.startswith(("datasets.", "panel.tier2"))]
    cons["not_consumed_tier2_only"] = tier2_only
    cons["not_consumed_other"] = [k for k in cons.get("not_consumed", []) if k not in tier2_only]
    cpu = {"units": sum(float(e.get("cpu_s", 0)) for e in ledger.get("entries", []) if e.get("kind") == "unit"),
           "sigma_star": sum(float(e.get("cpu_s", 0)) for e in ledger.get("entries", []) if e.get("kind") == "sigma_star"),
           "sanity": sum(float(e.get("cpu_s", 0)) for e in ledger.get("entries", []) if e.get("kind") == "sanity"),
           "infer": sum(float(e.get("cpu_s", 0)) for e in ledger.get("entries", []) if e.get("kind") == "infer"),
           "wall_units": sum(float(e.get("wall_s", 0)) for e in ledger.get("entries", []) if e.get("kind") == "unit")}
    (pkg / "notes" / "evaluation").mkdir(parents=True, exist_ok=True)
    (pkg / "notes" / "evaluation" / "CONSUMPTION.json").write_text(json.dumps(
        {"consumption": cons, "science_cpu_s": cpu, "budget_stops": ledger.get("stops", [])}, indent=1))
    files = ["PRIMARY_ENDPOINTS.csv", "NATIVE_AND_HELDOUT_CHECKS.csv", "RECOVERY.csv", "UTILITY.csv",
             "MATCHED_COMPARISONS.csv", "SUPPORT_COVERAGE.csv", "SEED_VARIATION.csv", "PRIMARY_FAMILY.json",
             "MODEL_MANIFEST.json", "SIGMA_STAR_SUMMARY.json", "notes/evaluation/CONSUMPTION.json"]
    leak = [f for f in files if str(root) in (pkg / f).read_text()]
    if leak:
        raise ValueError(f"private root path leaked into public tables: {leak}")
    return {"mode": "report", "package_dir_in_git": inside_git_worktree(pkg) is not None, "files": files,
            "n_primary": len(prim), "n_recovery": len(rec), "n_utility": len(ut), "n_matched": len(mc),
            "n_models": len(mm), "science_cpu_s": cpu}


# ==================================================================================================================
# calibration (synthetic, real role sizes, full frozen grids; not science)
# ==================================================================================================================


def calibrate(P: dict, eff: dict, out: Path, work: Path | None = None, datasets=("adult", "hmda"),
              arms=("A", "B", "C", "D_sigma1_rs0", "D_sigma8_rs0"), log=print) -> dict:
    """Time the full Tier-1 per-unit slate on a synthetic world at the real role sizes (OMP_NUM_THREADS=1) and
    project Tier-1 / Tier-2 CPU-hours against the budget. Writes `out` (calibration.json)."""
    from .bench import (Inputs, MapStore, SharedStore, freeze_all_support, make_bench_world, run_unit, _write_json)
    from .bench_infer import infer_bench
    from .guards import FitAuthorization
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: calibration must run with OMP_NUM_THREADS=1")
    work = Path(work or tempfile.mkdtemp(prefix="bench_calibration_"))
    t0 = time.perf_counter()
    w = make_bench_world(work / "priv", d=64, n_test={"adult": 15060, "hmda": 13661},
                         n_train={"adult": 24127, "hmda": 63704}, role_counts={"adult": (7571, 2239, 5250)},
                         signal={"adult": {"sex": "direct"}, "hmda": {"race": "direct"}}, datasets=datasets)
    build_s = time.perf_counter() - t0
    root = work / "priv"
    inputs = Inputs(w["index"])
    _write_json(root / "support" / "SUPPORT_FROZEN.json", freeze_all_support(inputs, eff))
    ctx0 = {"root": root, "inputs": inputs, "auth": FitAuthorization(synthetic=True), "synthetic": True,
            "eff_plain": eff, "maps": MapStore(root), "shared": SharedStore(root), "log": log,
            "registered": set(tier1_units()) | {u for v in tier2_units().values() for u in v},
            "support_frozen": json.loads((root / "support" / "SUPPORT_FROZEN.json").read_text()),
            "support_frozen_sha256": sha256_file(root / "support" / "SUPPORT_FROZEN.json")}
    cells = {"adult": ("income_prediction", "sex"), "hmda": ("underwriting", "race")}
    probes, per = {}, {}
    for d in datasets:
        p, a = cells[d]
        per[d] = {}
        for arm in arms:
            u = f"{d}__s0__{p}__{a}__{arm}"
            c0, w0 = time.process_time(), time.perf_counter()
            run_unit(u, ctx0)
            fr = json.loads((root / "units" / u / "fit_records.json").read_text())
            probes[u] = {"cpu_s": time.process_time() - c0, "wall_s": time.perf_counter() - w0,
                         "n_model_fits": fr["n_model_fits"], "roles": fr.get("roles"),
                         "recipe_cpu_s": {f"{x['surface']}|{x['recipe']}": x.get("cpu_s") for x in fr["recipes"]
                                          if x.get("cpu_s") is not None},
                         "shared_reused": fr.get("shared")}
            log(f"[calibrate] {u}: {probes[u]['cpu_s']:.1f}s cpu")
        sh = {k: probes[f"{d}__s0__{p}__{a}__{k}"]["cpu_s"] for k in arms}
        dvals = [v for k, v in sh.items() if k.startswith("D")]
        # A carries the shared outputs_only fit; D units carry their own U2. Per-arm projection:
        per[d] = {"A": sh["A"], "B": sh["B"], "C": sh["C"], "D": float(np.mean(dvals))}
    # inference timing on the calibration units (full B), sigma* fixed at 1 for timing only
    ssd = {"schema": "calibration-only", "datasets": {d: {"sigma_star": 1.0, "flagged_fallback": False}
                                                      for d in datasets}}
    _write_json(root / "infer" / "SIGMA_STAR.json", ssd)
    c0 = time.process_time()
    infer_bench(root, eff=eff, tier="1", unit_ids=list(probes))
    infer_cpu = time.process_time() - c0
    n_cal_units = len(probes)
    infer_per_unit = infer_cpu / n_cal_units
    # projections
    nD = len(eff["defenses"]["noise"]["sigmas"]) * len(eff["defenses"]["noise"]["release_seeds"])
    ns = len(eff["panel"]["encoder_seeds"])
    t1 = {d: ns * (per[d]["A"] + per[d]["B"] + per[d]["C"] + nD * per[d]["D"]) for d in datasets}
    n_t1 = ns * (3 + nD) * len(datasets)
    t1_inf = infer_per_unit * n_t1
    # Tier 2: binary attributes priced at the Adult (sex, K=2) cost; multi-class at the HMDA (race, K=5) cost,
    # each scaled by the dataset's attacker_fit size ratio (estimate; the runner re-projects from measured units)
    size = {"adult": 7571, "hmda": 6794}
    from .bench_effective import DATASETS, TIER2_PAIRS
    cost_bin = {k: per["adult"][k] for k in ("A", "B", "C", "D")}
    cost_mc = {k: per["hmda"][k] for k in ("A", "B", "C", "D")}
    t2 = {"E1": 0.0, "E2": 0.0, "E3": 0.0}
    n_t2 = {"E1": 0, "E2": 0, "E3": 0}
    for d, pairs in TIER2_PAIRS.items():
        for p, a in pairs:
            K = DATASETS[d]["attr_dims"][a]
            base = cost_bin if K == 2 else cost_mc
            sc = size[d] / (size["adult"] if K == 2 else size["hmda"])
            t2["E1"] += ns * (base["A"] + base["B"]) * sc
            t2["E2"] += ns * base["C"] * sc
            t2["E3"] += ns * nD * base["D"] * sc
            n_t2["E1"] += 2 * ns
            n_t2["E2"] += ns
            n_t2["E3"] += ns * nD
    t2_inf = {k: infer_per_unit * n for k, n in n_t2.items()}
    budget_h = float(eff["budget"]["cpu_hours"])
    sanity = 2 * (per["adult"]["A"] + per["hmda"]["A"]) / 2   # two C_rep slates (upper bound: an A unit each)
    tier1_total = sum(t1.values()) + t1_inf + sanity
    cum, fits = tier1_total, {}
    for k in ("E1", "E2", "E3"):
        cum += t2[k] + t2_inf[k]
        fits[k] = {"cpu_h": (t2[k] + t2_inf[k]) / 3600, "cumulative_cpu_h_incl_tier1": cum / 3600,
                   "fits_in_budget": cum / 3600 <= budget_h, "n_units": n_t2[k]}
    res = {"label": "ESTIMATE from synthetic timing at the real role sizes (single thread, full frozen grids); not a "
                    "measurement of the real run; recorded separately from the science budget",
           "measured_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "machine": platform.platform(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
           "world": {"d": 64, "n_test": {"adult": 15060, "hmda": 13661}, "n_train": {"adult": 24127, "hmda": 63704},
                     "adult_roles": [7571, 2239, 5250], "hmda_roles": "registered hash rule 0.50/0.15/0.35",
                     "signal": "adult sex direct (column 0), hmda race direct; other attributes null",
                     "build_wall_s": build_s},
           "probes": probes, "per_unit_cpu_s": per,
           "inference": {"cpu_s_for_calibration_units": infer_cpu, "n_units": n_cal_units,
                         "cpu_s_per_unit": infer_per_unit,
                         "includes": "exploratory B=2000 + primary B=20000 on the calibration units"},
           "projection": {"tier1": {"units_cpu_h": {d: v / 3600 for d, v in t1.items()},
                                    "inference_cpu_h": t1_inf / 3600, "sanity_cpu_h": sanity / 3600,
                                    "total_cpu_h": tier1_total / 3600, "n_units": n_t1,
                                    "fits_in_budget": tier1_total / 3600 <= budget_h},
                          "tier2": fits, "budget_cpu_h": budget_h,
                          "tier2_note": "binary attributes priced at the Adult/sex cost, multi-class at the HMDA/race "
                                        "cost, scaled by attacker_fit size; Tier 2 runs in the fixed order and stops at "
                                        "a unit boundary by the frozen stop rule using measured per-arm costs"},
           "inference_reserve_cpu_s_per_unit_in_protocol": eff["budget"]["inference_reserve_cpu_s_per_unit"]}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    return {"mode": "calibrate", "written": str(out), "projection": res["projection"],
            "per_unit_cpu_s": per, "infer_cpu_s_per_unit": infer_per_unit, "work_dir": str(work)}


__all__ = ["report_bench", "calibrate"]
