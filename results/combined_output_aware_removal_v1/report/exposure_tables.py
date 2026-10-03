"""Build EXPOSURE_ENDPOINTS.csv: original vs exposure-retained values for the 24 primary endpoints, headline
exploratory rows and the native N0 check (all from saved predictions; rule in exposure/EXPOSURE_RULE.md)."""
import csv, json
from pathlib import Path
HOME = Path.home()
WT = Path(__file__).resolve().parents[3]
OLD = WT / "results/combined_matched_removal_benchmark_v1"
PKG = WT / "results/combined_output_aware_removal_v1"
X = HOME / "PCRL_eval_cache_private/oar_v1/exposure"
ret = json.load(open(X / "INFER_ALL_retained.json"))
extra = json.load(open(X / "EXPOSURE_EXTRA.json"))
orig = {r["id"]: r for r in csv.DictReader(open(OLD / "PRIMARY_ENDPOINTS.csv"))}
rows = []
f = lambda v: None if v in (None, "", "nan") else float(v)  # noqa: E731
for e in ret["primary"]["endpoints"]:
    o = orig[e["id"]]
    same = o["decision"] == e["decision"]
    rows.append({"kind": "primary (original family, 24)", "id": e["id"], "original_point": f(o["point"]),
                 "original_lower": f(o["lower"]), "original_upper": f(o["upper"]), "original_decision": o["decision"],
                 "retained_point": e.get("point"), "retained_lower": e.get("lower"), "retained_upper": e.get("upper"),
                 "retained_decision": e["decision"], "status": "STABLE" if same else "CHANGED"})
# headline exploratory rows (Tier-1 cells + all-pair statements)
oexp = {r["id"]: r for r in csv.DictReader(open(OLD / "MATCHED_COMPARISONS.csv"))}
orec = {r["id"]: r for r in csv.DictReader(open(OLD / "RECOVERY.csv"))}
rexp = {r["id"]: r for r in ret["exploratory"]}
for sid, r in rexp.items():
    o = oexp.get(sid) or orec.get(sid)
    if not o or r.get("level") not in ("group", "paired_diff_vs_A", "reference") and "minus" not in sid:
        continue
    if not any(k in sid for k in ("|B-A|rep|NL|macro_auc", "|C-A|rep|NL|macro_auc", "outputs_NL_minus_LO", "|A|rep|NL|macro_auc",
                                   "|B-A|U2|U2|accuracy", "|A|rep|RHO1|rho1sq", "|B|rep|RHO1|rho1sq")):
        continue
    if "|D_" in sid:
        continue
    lo, hi = r.get("lower90"), r.get("upper90")
    olo, ohi = f(o.get("lower90")), f(o.get("upper90"))
    def side(a, b, bar):
        if a is None or b is None:
            return "NE"
        return "above" if a > bar else ("below" if b < bar else "unresolved")
    bar = 0.55 if sid.endswith("|A|rep|NL|macro_auc") else 0.0
    if "rho1sq" in sid:
        bar = 0.05
    so, sr = side(olo, ohi, bar), side(lo, hi, bar)
    rows.append({"kind": "exploratory headline (90%)", "id": sid, "original_point": f(o["point"]), "original_lower": olo,
                 "original_upper": ohi, "original_decision": f"{so} {bar}", "retained_point": r.get("point"),
                 "retained_lower": lo, "retained_upper": hi, "retained_decision": f"{sr} {bar}",
                 "status": "STABLE" if so == sr else ("UNRESOLVED" if "unresolved" in (so, sr) else "CHANGED")})
for n in extra["native_N0"]:
    rows.append({"kind": "native N0 (historical in-sample, tau=0.05)", "id": f"{n['dataset']}__s{n['seed']}__{n['purpose']}__{n['attribute']}",
                 "original_point": n["N0_full"]["historical_mixed_precision"], "original_decision": n["category_full"],
                 "retained_point": n["N0_retained"]["historical_mixed_precision"], "retained_decision": n["category_retained"],
                 "status": "STABLE" if n["category_full"] == n["category_retained"] else "CHANGED",
                 "original_lower": None, "original_upper": None, "retained_lower": None, "retained_upper": None})
cols = list(rows[0].keys())
with open(PKG / "EXPOSURE_ENDPOINTS.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=cols); w.writeheader(); w.writerows(rows)
from collections import Counter
print(Counter((r["kind"], r["status"]) for r in rows))
print([ (r["id"], r["original_decision"], r["retained_decision"]) for r in rows if r["status"] != "STABLE"][:20])
print("support changes:", {c: {w: v["classes_lost"] for w, v in d.items()} for c, d in extra["support_changes"].items() if any(v["classes_lost"] for v in d.values())})
