"""Stage 2: post hoc exposure sensitivity with fixed saved predictors (rule: exposure/EXPOSURE_RULE.md). No fits."""
import json, os, sys, time
from pathlib import Path
import numpy as np

assert os.environ.get("OMP_NUM_THREADS") == "1"
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
import stored_model_eval.bench_infer as BI  # noqa: E402  (locked runner code, unchanged)
from stored_model_eval.recipes import historical_native_r2  # noqa: E402
from stored_model_eval.bench_effective import BENCH_EFFECTIVE, tier1_units, tier2_units, parse_unit  # noqa: E402

HOME = Path.home()
ROOT = HOME / "PCRL_eval_cache_private" / "bench_v1"
OUT = HOME / "PCRL_eval_cache_private" / "oar_v1" / "exposure"
OUT.mkdir(parents=True, exist_ok=True)
t0, c0 = time.time(), time.process_time()

DROP, EXPOSED_TEST = {}, {}
for ds in ("adult", "hmda"):
    L = np.load(ROOT / "inputs" / f"{ds}_labels.npz")
    train_keys = set(L["canon_key"][L["split"] == "train"])
    exposed = (L["split"] == "test") & np.isin(L["canon_key"], list(train_keys))
    DROP[ds] = set(L["row_id"][exposed & (L["role"] == "assessment")].tolist())
    EXPOSED_TEST[ds] = set(L["row_id"][exposed].tolist())
print({d: len(v) for d, v in DROP.items()}, {d: len(v) for d, v in EXPOSED_TEST.items()})

PRIOR_KEYS = {"s_prior_fit", "t_prior_fit", "G1_prior", "G2_prior"}
_orig_load = BI.load_unit
SUPPORT_CHANGES = {}


def load_unit_retained(units_dir, uid):
    d = _orig_load(units_dir, uid)
    if d is None:
        return d
    ds = parse_unit(uid)["dataset"]
    p = d["preds"]
    rid = p["assess_row_id"]
    keep = ~np.isin(rid, list(DROP[ds]))
    n = len(rid)
    d["preds"] = {k: (v[keep] if (k not in PRIOR_KEYS and getattr(v, "ndim", 0) >= 1 and v.shape[0] == n) else v)
                  for k, v in p.items()}
    # recompute assessment support on retained rows against the frozen thresholds
    sup = json.loads(json.dumps(d["supported"]))
    for what, ykey in (("sensitive", "y_s"), ("task", "y_task")):
        s = sup[what]
        y = d["preds"][ykey]
        cnt = [int((y == c).sum()) for c in range(s["n_classes"])]
        thr = s["thresholds"]["assessment"]
        before = s["counts_per_role"]["assessment"]
        lost = [c for c in s["supported_classes"] if cnt[c] < thr]
        SUPPORT_CHANGES.setdefault(parse_unit(uid)["cell"], {})[what] = {
            "assessment_counts_before": before, "assessment_counts_after": cnt, "threshold": thr,
            "supported_classes": s["supported_classes"], "classes_lost": lost}
        if lost:
            raise RuntimeError(f"{uid}: {what} classes {lost} fall below the assessment threshold; report as NE")
        s["counts_per_role"]["assessment"] = cnt
    d["supported"] = sup
    d["exposure_retained_rows"] = int(keep.sum())
    return d


BI.load_unit = load_unit_retained
res = BI.infer_bench(ROOT, tier="all")
BI.load_unit = _orig_load
json.dump(res, open(OUT / "INFER_ALL_retained.json", "w"), default=str)

# native N0 on the test split without any of the training-overlapping test rows (all roles)
idx = json.loads((ROOT / "inputs" / "INPUTS_INDEX.json").read_text())
nat = BENCH_EFFECTIVE["native"]["A"]
n0 = []
for ds in ("adult", "hmda"):
    L = np.load(ROOT / "inputs" / f"{ds}_labels.npz")
    pur = idx["datasets"][ds]["purposes"]
    for s in (0, 1, 2):
        F = np.load(ROOT / "inputs" / f"{ds}_s{s}_forward.npz")
        assert np.array_equal(F["row_id"], L["row_id"])
        test = L["split"] == "test"
        keep = test & ~np.isin(L["row_id"], list(EXPOSED_TEST[ds]))
        for pname, pinfo in pur.items():
            H = F[pinfo["rep_key"]]
            for att in pinfo["disallowed_attrs"]:
                y = L[att]
                full = {v: historical_native_r2(H[test], y[test], nat["lambda"], v)["raw"] for v in nat["variants"]}
                ret = {v: historical_native_r2(H[keep], y[keep], nat["lambda"], v)["raw"] for v in nat["variants"]}
                n0.append({"dataset": ds, "seed": s, "purpose": pname, "attribute": att, "n_full": int(test.sum()),
                           "n_retained": int(keep.sum()), "N0_full": full, "N0_retained": ret,
                           "category_full": "fails" if full[nat["category_variant"]] > nat["tau"] else "passes",
                           "category_retained": "fails" if ret[nat["category_variant"]] > nat["tau"] else "passes"})
json.dump({"drop_assessment_rows": {d: len(v) for d, v in DROP.items()},
           "exposed_test_rows_all_roles": {d: len(v) for d, v in EXPOSED_TEST.items()},
           "support_changes": SUPPORT_CHANGES, "native_N0": n0, "wall_s": time.time() - t0,
           "cpu_s": time.process_time() - c0}, open(OUT / "EXPOSURE_EXTRA.json", "w"), indent=1, default=str)
print("done", time.time() - t0)
