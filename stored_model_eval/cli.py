"""Command line: python -m stored_model_eval <command> [...]

Commands: plan | admit | forward | recount | fit-attackers | score | infer | report | pilot | lock

CELL-A pilot (repaired runner; see pilot.py, pilot_infer.py, lock.py):
  pilot --plan | (default dry-run) | --execute-scientific-fits [--resume] [--units a,b]
  lock build|verify --inputs-dir D [--lock P]
  infer --units-dir D --out INFER.json          (pilot inference from saved predictions only)
  report --pilot-infer INFER.json --tables-dir D [--historical-native dominant_axis_audit.json]
Global: --protocol PATH (protocol config JSON, deep-merged over defaults), --dry-run, --out PATH (JSON).
Default mode performs NO scientific fits and opens NO network connections (a socket guard is installed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_protocol
from .guards import FitAuthorization, ScientificFitRefused, install_network_guard
from .metrics import to_jsonable


def _emit(obj, out: str | None):
    txt = json.dumps(to_jsonable(obj), indent=1, default=str)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(txt)
    print(txt if len(txt) < 6000 else txt[:6000] + f"\n... (truncated; full JSON in {out})")


def cmd_plan(a, cfg):
    from .plan import load_calibration, make_plan
    spec = json.loads(Path(a.spec).read_text())
    return make_plan(spec, cfg, load_calibration(a.calibration), Path(a.spec).parent)


def cmd_admit(a, cfg):
    from .admission import admit
    rec = admit(a.manifest)
    if a.dry_run:
        rec["dry_run"] = True
    return rec


def cmd_forward(a, cfg):
    from .admission import admit, _load_member
    from .forward import forward_to_cache
    man = json.loads(Path(a.manifest).read_text())
    rec = admit(man, Path(a.manifest).parent)
    if rec["status"] != "ADMITTED":
        raise SystemExit(f"forward refused: manifest {rec['status']}: {rec['errors']}")
    ck = rec["files"][a.checkpoint_name]
    fa = man["arrays"][a.features_array]
    X = _load_member(Path(rec["files"][fa["file"]]["path"]), fa["key"])
    ids = _load_member(Path(rec["files"][fa["file"]]["path"]), fa["ids"])
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    info = forward_to_cache(ck["path"], man["files"][a.checkpoint_name]["sha256"], X, ids, a.cache_dir, a.tag,
                            batch_size=a.batch_size, allow_pickle=a.allow_pickle, dry_run=a.dry_run)
    info["admission"] = {"status": rec["status"], "n_rows": rec["n_rows"]}
    return info


def cmd_recount(a, cfg):
    from .recount import recount_pcrl_strict, recount_probability_configs
    if a.kind == "probs":
        spec = json.loads(Path(a.spec).read_text())
        if a.dry_run:
            return {"dry_run": True, "n_configs": len(spec["configs"]),
                    "files": sorted({c["file"] for c in spec["configs"]})}
        return recount_probability_configs(spec, Path(a.spec).parent)
    if a.kind == "pcrl-strict":
        if a.dry_run:
            return {"dry_run": True, "repo": a.repo, "ref": a.ref}
        return recount_pcrl_strict(a.repo, a.ref, tau=cfg["r2"]["tau"])
    raise SystemExit("unknown recount kind")


def _synthetic_arrays(a, cfg):
    from .fixtures import make_synthetic
    fx = make_synthetic(a.synthetic_kind, n_units=a.n_units, d=a.dim, seed=a.seed, K=a.k)
    roles = np.where(fx["roles"] == "evaluation", cfg["roles"]["score"], fx["roles"])  # protocol's role name
    return {"representations": fx["H"], "outputs": fx["outputs"], "labels": fx["S"], "units": fx["units"],
            "roles": roles, "record_keys": fx["record_keys"], "row_ids": fx["row_ids"], "_synthetic": True}


def cmd_fit(a, cfg):
    from .pipeline import fit_attackers, save_scores
    auth = FitAuthorization(synthetic=a.synthetic, execute_scientific_fits=a.execute_scientific_fits)
    if a.synthetic:
        arrays = _synthetic_arrays(a, cfg)
    else:
        if not a.manifest:
            raise SystemExit("fit-attackers needs --manifest (or --synthetic)")
        from .admission import load_admitted
        arrays, rec = load_admitted(a.manifest)
        if arrays["_synthetic"] and not a.synthetic:
            raise SystemExit("manifest declares synthetic data; pass --synthetic")
    attackers = a.attackers.split(",")
    surfaces = a.surfaces.split(",") if a.surfaces else None
    roles = arrays["roles"].astype(str)
    summary = {"rows_per_role": {r: int((roles == r).sum()) for r in np.unique(roles)},
               "attackers": attackers, "surfaces": surfaces or cfg["surfaces"],
               "synthetic": bool(arrays["_synthetic"])}
    if a.dry_run:
        return {"dry_run": True, **summary, "fits_performed": 0}
    try:
        auth.check("attacker slate", bool(arrays["_synthetic"]))
    except ScientificFitRefused as e:
        raise SystemExit(f"REFUSED: {e}")
    t0 = time.perf_counter()
    res = fit_attackers(arrays, cfg, auth, attackers=attackers, surfaces=surfaces, random_state=a.seed)
    path = save_scores(res, a.out_dir)
    wall = time.perf_counter() - t0
    out = {**summary, "scores": str(path), "timing_s": res["timing_s"], "wall_s": wall}
    if a.timing_out:
        n_fit = summary["rows_per_role"].get(cfg["roles"]["attacker_fit"], 0)
        from .plan import grid_sizes
        GRID_SIZES = grid_sizes(cfg)
        cal = {"source": f"synthetic timing n_fit={n_fit} d={a.dim} on this machine"}
        for att in attackers:
            t = res["timing_s"].get(f"rep|{att}")
            if t is not None:
                cal[att] = t / ((n_fit / 1000) * GRID_SIZES[att] * max(a.dim / 64, 0.25))
        Path(a.timing_out).write_text(json.dumps(cal, indent=1))
        out["calibration"] = cal
    return out


def cmd_score(a, cfg):
    from .pipeline import load_scores, score
    sc = load_scores(a.scores)
    ms = a.min_support or cfg["support"]["min_class_support"]
    return {"n_eval_rows": int(len(sc["y"])), "min_support": ms,
            "scores": {f"{s}|{att}": score(sc["y"], P, ms, cfg["support"].get("macro_over", "all_declared"))
                       for (s, att), P in sc["probs"].items()}}


def cmd_infer(a, cfg):
    if a.units_dir:
        from .pilot_infer import infer_pilot
        if a.dry_run:
            return {"dry_run": True, "units_dir": a.units_dir}
        return infer_pilot(Path(a.units_dir).expanduser())
    if not a.scores:
        raise SystemExit("infer needs --scores (legacy) or --units-dir (pilot)")
    from .pipeline import infer, load_scores
    sc = load_scores(a.scores)
    if a.dry_run:
        return {"dry_run": True, "n_rows": int(len(sc["y"])), "n_boot": cfg["bootstrap"]["n_boot"]}
    return infer(sc, cfg, a.min_support)


def cmd_report(a, cfg):
    if a.pilot_infer:
        from .pilot_infer import report_pilot
        if not a.tables_dir:
            raise SystemExit("report --pilot-infer needs --tables-dir")
        hist = json.loads(Path(a.historical_native).read_text()) if a.historical_native else None
        return report_pilot(json.loads(Path(a.pilot_infer).read_text()), Path(a.tables_dir), hist)
    if not (a.scores and a.infer):
        raise SystemExit("report needs --scores and --infer (legacy) or --pilot-infer (pilot)")
    from .pipeline import report
    inf = json.loads(Path(a.infer).read_text())
    return report(a.scores, inf, cfg)


def build_parser():
    p = argparse.ArgumentParser(prog="stored_model_eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", default=None, help="protocol config JSON (merged over defaults)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None, help="write the JSON result here")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan"); s.add_argument("--spec", required=True); s.add_argument("--calibration")
    s = sub.add_parser("admit"); s.add_argument("--manifest", required=True)
    s = sub.add_parser("forward")
    s.add_argument("--manifest", required=True); s.add_argument("--checkpoint-name", default="checkpoint")
    s.add_argument("--features-array", default="features"); s.add_argument("--cache-dir", required=True)
    s.add_argument("--tag", required=True); s.add_argument("--batch-size", type=int, default=512)
    s.add_argument("--threads", type=int, default=1); s.add_argument("--allow-pickle", action="store_true")
    s = sub.add_parser("recount"); s.add_argument("kind", choices=["probs", "pcrl-strict"])
    s.add_argument("--spec"); s.add_argument("--repo", default="/Users/nathansamson/PCRL")
    s.add_argument("--ref", default="origin/main")
    s = sub.add_parser("fit-attackers")
    s.add_argument("--manifest"); s.add_argument("--out-dir", required=True)
    s.add_argument("--execute-scientific-fits", action="store_true")
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--synthetic-kind", default="direct"); s.add_argument("--n-units", type=int, default=2000)
    s.add_argument("--dim", type=int, default=8); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--k", type=int, default=2, help="classes for --synthetic-kind contrast")
    s.add_argument("--attackers", default="linear,gbt,mlp"); s.add_argument("--surfaces", default=None)
    s.add_argument("--timing-out", default=None)
    s = sub.add_parser("score"); s.add_argument("--scores", required=True)
    s.add_argument("--min-support", type=int, default=None)
    s = sub.add_parser("infer"); s.add_argument("--scores", default=None)
    s.add_argument("--min-support", type=int, default=None)
    s.add_argument("--units-dir", default=None, help="pilot: run_v1/units (saved predictions only)")
    s = sub.add_parser("report"); s.add_argument("--scores", default=None); s.add_argument("--infer", default=None)
    s.add_argument("--pilot-infer", default=None); s.add_argument("--tables-dir", default=None)
    s.add_argument("--historical-native", default=None,
                   help="origin/main results/v2_adult_ROUND4/dominant_axis_audit.json (N0 comparison)")
    s = sub.add_parser("pilot", help="CELL-A pilot unit runner")
    s.add_argument("--plan", action="store_true")
    s.add_argument("--execute-scientific-fits", action="store_true")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--units", default=None, help="comma-separated registered unit IDs (subset)")
    s.add_argument("--worktree", default=None)
    s.add_argument("--run-dir", default=None, help="default ~/PCRL_eval_cache_private/pilot_adult_s0/run_v1")
    s.add_argument("--manifest-dir", default=None, help="default <run-dir>/inputs")
    s.add_argument("--lock", default=None,
                   help="default <worktree>/results/combined_stored_model_pilot_v1/PILOT_LOCK_v2.json")
    s.add_argument("--synthetic", action="store_true", help="inputs are synthetic fixtures (tests)")
    s.add_argument("--allow-other-branch-for-tests", action="store_true")
    s = sub.add_parser("lock", help="build / verify PILOT_LOCK_v2.json")
    s.add_argument("action", choices=["build", "verify"])
    s.add_argument("--inputs-dir", default=None); s.add_argument("--lock", default=None)
    s.add_argument("--run-dir", default=None)
    s.add_argument("--access-table", default=None); s.add_argument("--features", default=None)
    s.add_argument("--effective-out", default=None, help="also write EFFECTIVE_PROTOCOL.json here (build)")
    s.add_argument("--worktree", default=None); s.add_argument("--allow-other-branch-for-tests", action="store_true")
    return p


def _pilot_paths(a):
    from .pilot import DEFAULT_RUN_DIR, resolve_worktree
    wt = resolve_worktree(a.worktree, a.allow_other_branch_for_tests)
    run = Path(a.run_dir).expanduser() if getattr(a, "run_dir", None) else DEFAULT_RUN_DIR
    lock = Path(a.lock).expanduser() if a.lock else \
        Path(wt["root"]) / "results/combined_stored_model_pilot_v1/PILOT_LOCK_v2.json"
    return wt, run, lock


def cmd_pilot(a, cfg):
    import os
    from . import pilot
    if a.plan and a.execute_scientific_fits:
        raise SystemExit("--plan and --execute-scientific-fits are exclusive")
    wt, run, lock_path = _pilot_paths(a)
    mdir = Path(a.manifest_dir).expanduser() if a.manifest_dir else run / "inputs"
    subset = [u.strip() for u in a.units.split(",")] if a.units else None
    if a.plan:
        return {"worktree": wt, **pilot.plan(mdir, subset)}
    if a.dry_run or not a.execute_scientific_fits:
        out = pilot.dry_run(mdir, subset)
        out["worktree"] = wt
        if lock_path.exists():
            from .lock import verify_lock
            out["lock_check"] = verify_lock(lock_path, Path(wt["root"]), mdir)
        else:
            out["lock_check"] = {"ok": False, "mismatches": [f"no lock at {lock_path}"]}
        return out
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: scientific execution requires OMP_NUM_THREADS=1 (one scheduler, fixed threads)")
    from .lock import verify_lock
    if not lock_path.exists():
        raise SystemExit(f"REFUSED: lock {lock_path} not found")
    v = verify_lock(lock_path, Path(wt["root"]), mdir)
    if not v["ok"]:
        raise SystemExit("REFUSED: lock verification failed:\n  " + "\n  ".join(v["mismatches"]))
    auth = FitAuthorization(synthetic=a.synthetic, execute_scientific_fits=True)
    res = pilot.execute(mdir, run / "units", auth, subset=subset, resume=a.resume,
                        log=lambda m: print(m, file=sys.stderr, flush=True))
    res.update(worktree=wt, lock_check=v)
    return res


def cmd_lock(a, cfg):
    from .lock import DEFAULT_ACCESS_TABLE, build_lock, verify_lock
    wt, run, lock_path = _pilot_paths(a)
    inputs = Path(a.inputs_dir).expanduser() if a.inputs_dir else run / "inputs"
    if a.action == "build":
        return build_lock(Path(wt["root"]), inputs, lock_path, a.access_table or DEFAULT_ACCESS_TABLE, a.features,
                          Path(a.effective_out) if a.effective_out else None)
    v = verify_lock(lock_path, Path(wt["root"]), inputs)
    if not v["ok"]:
        print(json.dumps(v, indent=1))
        raise SystemExit(3)
    return v


COMMANDS = {"plan": cmd_plan, "admit": cmd_admit, "forward": cmd_forward, "recount": cmd_recount,
            "fit-attackers": cmd_fit, "score": cmd_score, "infer": cmd_infer, "report": cmd_report,
            "pilot": cmd_pilot, "lock": cmd_lock}


def main(argv=None) -> int:
    install_network_guard()
    a = build_parser().parse_args(argv)
    cfg = load_protocol(a.protocol)
    t0 = time.perf_counter()
    res = COMMANDS[a.cmd](a, cfg)
    if isinstance(res, dict):
        res.setdefault("_runtime_s", round(time.perf_counter() - t0, 3))
        res.setdefault("_protocol", cfg["_meta"])
    _emit(res, a.out)
    if a.cmd == "admit" and res.get("status") == "REJECTED":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
