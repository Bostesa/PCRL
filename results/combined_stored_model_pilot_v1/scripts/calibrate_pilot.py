"""Synthetic calibration of the executable CELL-A slate at the pilot sizes (no real data is read).

Builds a synthetic world with the real layout (15,060 rows; hash roles 0.50/0.15/0.35 -> ~7.5k/2.3k/5.3k; d = 64
representations; output dims 2/6/4 per purpose; binary, 4-class and 5-class attributes), runs representative units
through the SAME runner code and EFFECTIVE_PROTOCOL (pilot.run_unit, single thread), times inference, and
extrapolates to the 26 registered units by unit type.

Run (single thread):
  OMP_NUM_THREADS=1 /Users/nathansamson/PCRL/.venv/bin/python \
      results/combined_stored_model_pilot_v1/scripts/calibrate_pilot.py --work-dir <scratch dir outside git>
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))

from stored_model_eval.effective import (EFFECTIVE_PROTOCOL, NOISE_UNITS, UNTREATED_UNITS,  # noqa: E402
                                         effective_hash, unit_info)
from stored_model_eval.fixtures import make_pilot_world  # noqa: E402
from stored_model_eval.guards import FitAuthorization  # noqa: E402
from stored_model_eval.pilot import run_unit  # noqa: E402
from stored_model_eval.pilot_infer import infer_pilot  # noqa: E402
from stored_model_eval.pilot_inputs import build_inputs  # noqa: E402

# K of each untreated attribute on the real data (labels.npz class counts; no outcome involved)
REAL_K = {"race": 5, "sex": 2, "age_group": 4, "marital_status": 2, "income": 2}
PROBES = ["income_prediction__sex", "income_prediction__race", "employment_analysis__age_group",
          "employment_analysis__marital_status", "income_prediction__sex__p0_sigma1_seed0",
          "income_prediction__sex__p0_sigma8_seed0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", default=str(WT / "results/combined_stored_model_pilot_v1/notes/implementation/"
                                               "calibration.json"))
    a = ap.parse_args()
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("run with OMP_NUM_THREADS=1")
    work = Path(a.work_dir)
    t0 = time.perf_counter()
    w = make_pilot_world(work, n=15060, d=64, seed=7, sex_signal=0.6)
    build_inputs(w["orig"], work / "run_v1" / "inputs", w["source"], w["checkpoint"], log=lambda m: None)
    t_world = time.perf_counter() - t0
    auth = FitAuthorization(synthetic=True)
    per = {}
    for u in PROBES:
        c0, w0 = time.process_time(), time.perf_counter()
        run_unit(u, work / "run_v1" / "inputs" / f"manifest_{u}.json", work / "run_v1" / "units", auth,
                 log=lambda m: print(m, flush=True))
        fr = json.loads((work / "run_v1" / "units" / u / "fit_records.json").read_text())
        per[u] = {"cpu_s": time.process_time() - c0, "wall_s": time.perf_counter() - w0,
                  "roles": fr["roles"], "K_s": fr["K_s"], "K_t": fr["K_t"],
                  "recipe_cpu_s": {f"{r['surface']}|{r['recipe']}": r.get("cpu_s") for r in fr["recipes"]
                                   if r.get("cpu_s") is not None}}
        print(json.dumps({u: {k: per[u][k] for k in ("cpu_s", "wall_s", "K_s")}}), flush=True)
    c0, w0 = time.process_time(), time.perf_counter()
    inf = infer_pilot(work / "run_v1" / "units", unit_ids=PROBES)
    t_inf = {"cpu_s": time.process_time() - c0, "wall_s": time.perf_counter() - w0,
             "n_exploratory_stats": len(inf["exploratory"]), "n_units": len(PROBES)}

    # extrapolation by unit type (untreated: by K and the purpose's output dim; noise: per unit)
    cost = {2: per["income_prediction__sex"]["cpu_s"], 5: per["income_prediction__race"]["cpu_s"],
            4: per["employment_analysis__age_group"]["cpu_s"]}
    noise_cost = max(per["income_prediction__sex__p0_sigma1_seed0"]["cpu_s"],
                     per["income_prediction__sex__p0_sigma8_seed0"]["cpu_s"])
    est = {}
    for u in UNTREATED_UNITS:
        k = REAL_K[unit_info(u)["attribute"]]
        est[u] = per[u]["cpu_s"] if u in per else cost[k]
    for u in NOISE_UNITS:
        est[u] = noise_cost
    fits_s = sum(est.values())
    inf_scale = 26 / len(PROBES)
    infer_s = t_inf["cpu_s"] * inf_scale * 1.2
    total_h = (fits_s + infer_s) / 3600
    out = {"label": "ESTIMATE from synthetic timing at pilot sizes (single thread); not a measurement of the real run",
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "machine": platform.platform(),
           "processor": platform.processor(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
           "effective_protocol_sha256": effective_hash(EFFECTIVE_PROTOCOL),
           "world": {"n_rows": 15060, "d": 64, "output_dims": {"income_prediction": 2, "employment_analysis": 6,
                                                               "education_assessment": 4},
                     "build_wall_s": t_world, "synthetic_signal": "sex_signal=0.6 (moderate), others null/weak"},
           "probes": per, "inference_probe": t_inf,
           "estimate": {"per_unit_cpu_s": est, "fits_cpu_s": fits_s, "inference_cpu_s_scaled_x1.2": infer_s,
                        "total_cpu_hours": total_h, "allowance_cpu_hours": 2.0,
                        "within_allowance": total_h <= 2.0,
                        "extrapolation": "untreated units by attribute K (2/4/5) from the probe of the same K; "
                                         "noise units at max(sigma=1, sigma=8) probe; inference scaled by 26/6 "
                                         "units x 1.2"}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["estimate"] | {"per_unit_cpu_s": None}, indent=1))


if __name__ == "__main__":
    main()
