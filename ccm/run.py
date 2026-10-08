"""Stage runner of the confidence-constrained mechanism sprint (ccm; role A). Each stage refuses unless its lock (and any
later amendment) verifies and is byte-identical on origin, and every loaded study module is locked with its hash.

  FEASIBILITY_LOCK  geometry  Stage D feasibility metrics F1-F5 (+ secondary G_exp) on label-free Ucal reference vectors
                              (OSF_DEFENSE_FIT for fitting geometry, CALIBRATION_HELDOUT for the fallback rate), per seed
                              and recipient (ccm.data.reference; ccm.geometry.run_geometry); go rule (PROTOCOL section 5)
                    oracle    exact finite-law oracles on the pinned TOY_LAWS.json (ccm.oracle)
  PILOT_LOCK        pilot     (only if the go rule holds)

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m ccm.sema --label A:<stage> -- env OMP_NUM_THREADS=1 PYTHONPATH=. \\
        <python> -m ccm.run --lock results/pcrl_confidence_constrained_mechanism_v1/<LOCK>.json --stage <stage>
Outputs: private records in <PRIVATE_CACHE>/ccm_v1/run/ (bins and representatives stay private); public aggregates in
the results folder (FEASIBILITY_GEOMETRY.json, ORACLE_RESULTS.json).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np

from ccm import ids as I

STAGE_MODULES = {"geometry": ["ccm.guard", "ccm.geometry", "ccm.data"], "oracle": ["ccm.guard", "ccm.oracle"]}
GEOMETRY_ROLES = {"fit": "OSF_DEFENSE_FIT", "held": "CALIBRATION_HELDOUT"}
GO_COVERAGE, GO_FALLBACK = 0.95, 0.05
NAMES = {1: "income", 2: "occupation"}


def _finite(o):
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _finite(o.tolist())
    return o


def event(msg, **kw):
    I.RUN.mkdir(parents=True, exist_ok=True)
    with open(I.RUN / "ACTIVITY_LOG.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": msg, **_finite(kw)},
                           allow_nan=False) + "\n")


def ledger(stage, wall, cpu):
    I.RUN.mkdir(parents=True, exist_ok=True)
    with open(I.RUN / "COMPUTE_LEDGER.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, "pid": os.getpid(),
                            "wall_s": wall, "cpu_s": cpu,
                            "maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}) + "\n")


def _split_public(rec):
    """Drop private arrays (representatives, member indices) from a geometry record; keep aggregates."""
    if isinstance(rec, dict):
        return {k: _split_public(v) for k, v in rec.items() if not str(k).startswith("_")
                and k not in ("bins", "members", "reps", "rep_classes", "representatives", "assign", "served", "order",
                              "packing_rows", "neighbourhood_counts", "rows")}
    if isinstance(rec, (list, tuple)):
        return [_split_public(v) for v in rec]
    return rec


def stage_geometry():
    from ccm import data as CD
    from ccm import geometry as GE
    from ccm import guard as GU
    out = {"schema": "ccm-geometry-v1", "contracts": {}, "constants": GU.CONFIG, "roles": GEOMETRY_ROLES}
    for contract in ("G", "G_exp"):
        res = {}
        for k in I.SEEDS:
            fit = CD.reference(k, GEOMETRY_ROLES["fit"])
            held = CD.reference(k, GEOMETRY_ROLES["held"])
            for i in I.RECIPIENTS:
                t0, c0 = time.time(), time.process_time()
                cap = GU.capacity_for(recipient=NAMES[i])
                rec = GE.run_geometry(fit["Ucal"][i], held["Ucal"][i], cap, dec_fit=fit["d"][i], dec_held=held["d"][i],
                                      contract=contract)
                priv = I.RUN / "geometry" / f"{contract}__s{k}__r{i}.json"
                priv.parent.mkdir(parents=True, exist_ok=True)
                priv.write_text(json.dumps(_finite(rec), allow_nan=False))
                pub = _split_public(rec)
                pub.update({"seed": k, "recipient": i, "cap_per_class": cap, "n_fit": int(len(fit["d"][i])),
                            "n_held": int(len(held["d"][i])), "wall_s": round(time.time() - t0, 1),
                            "cpu_s": round(time.process_time() - c0, 1),
                            "private_record_sha256": hashlib.sha256(priv.read_bytes()).hexdigest()})
                res[f"s{k}|r{i}"] = _finite(pub)
                event("geometry unit", contract=contract, seed=k, recipient=i)
        out["contracts"][contract] = res
    out["go_rule"] = go_rule(out["contracts"]["G"])
    (I.PKG / "FEASIBILITY_GEOMETRY.json").write_text(json.dumps(_finite(out), indent=1, allow_nan=False) + "\n")
    event("geometry done", go=out["go_rule"]["go"])


def go_rule(gres):
    """PROTOCOL section 5 (registered): go iff for EVERY seed and BOTH recipients F3 >= 0.95 and F4 <= 0.05 under G."""
    rows = {}
    for key, r in gres.items():
        g = r["go_inputs"]
        f3v, f4v, f3x, f3ur = g["F3_coverage"], g["F4_fallback_rate"], g.get("F3x"), g.get("F3u_registered")
        ok = f3v is not None and f4v is not None and f3v >= GO_COVERAGE and f4v <= GO_FALLBACK
        bound = f3x if f3x is not None else f3ur          # income: exact F3x; occupation: min(F3u, F3u_packing)
        cov_fail = f3v is None or f3v < GO_COVERAGE
        rows[key] = {"F3": f3v, "F4": f4v, "F3x": f3x, "F3u": g.get("F3u"), "F3u_packing": g.get("F3u_packing"),
                     "F3u_registered": f3ur, "ok": bool(ok),
                     "coverage_failure": bool(cov_fail),
                     "coverage_failure_intrinsic": bool(cov_fail and bound is not None and bound < GO_COVERAGE),
                     "coverage_failure_incomplete": bool(cov_fail and (bound is None or bound >= GO_COVERAGE)),
                     "fallback_dominated": bool(f4v is None or f4v > GO_FALLBACK)}
    go = bool(rows) and all(v["ok"] for v in rows.values())
    return {"rule": "go iff F3 >= 0.95 and F4 <= 0.05 under G for every seed and both recipients (PROTOCOL section 5)",
            "rows": rows, "go": go,
            "all_coverage_failures_intrinsic": all(v["coverage_failure_intrinsic"] for v in rows.values()
                                                   if v["coverage_failure"]),
            "any_incomplete": any(v["coverage_failure_incomplete"] for v in rows.values()),
            "any_fallback_dominated": any(v["fallback_dominated"] for v in rows.values())}


def stage_oracle():
    from ccm import oracle as OR
    laws_path = I.PKG / "TOY_LAWS.json"
    laws = json.loads(laws_path.read_text())
    sha = hashlib.sha256(laws_path.read_bytes()).hexdigest()
    res = OR.run_all(laws)
    out = {"schema": "ccm-oracle-v1", "laws_sha256": sha, "results": _finite(res)}
    (I.PKG / "ORACLE_RESULTS.json").write_text(json.dumps(_finite(out), indent=1, allow_nan=False) + "\n")
    event("oracle done", laws_sha256=sha)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", required=True, choices=("geometry", "oracle"))
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from ccm import lock as LK
    v = LK.verify_lock(Path(a.lock), stage=a.stage)
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    for m in STAGE_MODULES[a.stage]:
        importlib.import_module(m)
    bad = LK.check_loaded_modules(v["locked_files"])
    if bad:
        raise SystemExit("REFUSED: unlocked or changed code loaded: " + "; ".join(bad[:10]))
    event(f"start {a.stage}", pid=os.getpid(), lock=v["lock"])
    t0, c0 = time.time(), time.process_time()
    {"geometry": stage_geometry, "oracle": stage_oracle}[a.stage]()
    bad = LK.check_loaded_modules(v["locked_files"])
    if bad:
        raise SystemExit("REFUSED after stage (lazy import of unlocked code): " + "; ".join(bad[:10]))
    ledger(a.stage, time.time() - t0, time.process_time() - c0)
    event(f"end {a.stage}", pid=os.getpid())


if __name__ == "__main__":
    main()
