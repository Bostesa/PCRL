"""Stage 1 custody check: re-run the LOCKED benchmark inference (Tier 1) from saved predictions and compare all 24
primary rows and decisions with the committed PRIMARY_ENDPOINTS.csv. No fits."""
import csv, json, os, sys, time
from pathlib import Path
assert os.environ.get("OMP_NUM_THREADS") == "1"
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from stored_model_eval.bench_infer import infer_bench  # locked runner code, unchanged
root = Path.home() / "PCRL_eval_cache_private" / "bench_v1"
out = Path.home() / "PCRL_eval_cache_private" / "oar_v1" / "custody"
t0, c0 = time.time(), time.process_time()
res = infer_bench(root, tier="1")
json.dump(res, open(out / "INFER_T1_custody.json", "w"), default=str)
orig = {r["id"]: r for r in csv.DictReader(open(WT / "results/combined_matched_removal_benchmark_v1/PRIMARY_ENDPOINTS.csv"))}
prim = res["primary"]["endpoints"]
rows = prim if isinstance(prim, list) else list(prim.values())
cmp = []
for r in rows:
    o = orig[r["id"]]
    d = {k: (float(o[k]) if o[k] not in ("", "nan") else None, r.get(k)) for k in ("point", "lower", "upper")}
    same = all((a is None and (b is None or b != b)) or (b is not None and abs(a - float(b)) <= 1e-12) for a, b in d.values())
    cmp.append({"id": r["id"], "decision_orig": o["decision"], "decision_now": r.get("decision"),
                "values_identical_1e-12": same, "decision_identical": o["decision"] == r.get("decision")})
summ = {"n": len(cmp), "values_identical": sum(c["values_identical_1e-12"] for c in cmp),
        "decisions_identical": sum(c["decision_identical"] for c in cmp), "wall_s": time.time() - t0,
        "cpu_s": time.process_time() - c0, "rows": cmp}
json.dump(summ, open(out / "CUSTODY_PRIMARY.json", "w"), indent=1)
print({k: summ[k] for k in ("n", "values_identical", "decisions_identical", "wall_s", "cpu_s")})
