"""Stage 0B: regenerate repaired reporting outputs (no fits) and write ORIGINAL_VS_REPAIRED.csv."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from oar import study as S
from odx.equality import feature_equality_counts
OLD = WT / "results/combined_output_aware_removal_v1"
PKG = WT / "results/combined_output_diagnosis_v1"
(PKG / "repaired").mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
rows = []
# R1 roles: rebuild with the repaired code; role row sets must be unchanged
orig = json.loads((OLD / "ROLES_AND_SUPPORT.json").read_text())
rep = {}
for ds in ("adult", "hmda"):
    W = S.load_world(ds)
    rs = S.role_summary(W)
    ex = W["role"] == "excluded_exposure"
    by = {r: int((ex & (W["role_bench"] == r)).sum()) for r in ("attacker_fit", "attacker_val", "assessment")}
    rep[ds] = {"roles": rs, "excluded_exposure_rows": int(ex.sum()), "excluded_exposure_groups": int(len(np.unique(W["unit"][ex]))),
               "excluded_by_original_role": by}
    for r in ("defense_fit", "cert", "attacker_fit", "attacker_val", "assessment"):
        assert rs[r]["row_ids_sha256"] == orig[ds]["roles"][r]["row_ids_sha256"], (ds, r)
    rows.append({"item": "R1_role_label_truncation", "dataset": ds, "location": "ROLES_AND_SUPPORT.json",
                 "original": f"excluded_exposure_rows={orig[ds]['roles']['excluded_exposure_rows']}, exposure_groups_removed={orig[ds]['exposure_groups_removed']}",
                 "repaired": f"excluded_exposure_rows={ex.sum()}, groups={rep[ds]['excluded_exposure_groups']}, by role {by}",
                 "decisions_changed": "none (role row sets identical, hashes verified)"})
(PKG / "repaired/ROLES_EXCLUSIONS_REPAIRED.json").write_text(json.dumps(rep, indent=1))
# R2 feature-equality counts (rows AND distinct vectors) for cert and assessment vs defense_fit, every seed
a1 = json.loads((S.RUN.parent / "run/certificates/AMENDMENT_A1.json").read_text()) if False else \
    json.loads((Path.home() / "PCRL_eval_cache_private/oar_v1/run/certificates/AMENDMENT_A1.json").read_text())
eq = {}
for ds in ("adult", "hmda"):
    W = S.load_world(ds)
    for k in S.SEEDS:
        H = S.load_seed(ds, k)["H"]
        ref = H[W["idx"]["defense_fit"]]
        e = {r: feature_equality_counts(H[W["idx"][r]], ref) for r in ("cert", "attacker_fit", "attacker_val", "assessment")}
        eq[f"{ds}__s{k}"] = e
        old = a1[f"{ds}__s{k}"]["amended"]["nominee"]["cert_rows_with_feature_vector_equal_to_a_fit_row"]
        rows.append({"item": "R2_feature_equality_rows_vs_vectors", "dataset": f"{ds} s{k}", "location": "AMENDMENT_A1.json field cert_rows_with_feature_vector_equal_to_a_fit_row",
                     "original": f"{old} (labelled rows; actually distinct vectors)",
                     "repaired": f"cert affected_rows={e['cert']['affected_rows']}/{e['cert']['query_rows']}, distinct shared vectors={e['cert']['distinct_vectors_shared_with_reference']}, distinct cert vectors={e['cert']['distinct_query_vectors']}; assessment affected_rows={e['assessment']['affected_rows']}/{e['assessment']['query_rows']}",
                     "decisions_changed": "none (certificate statuses/bounds unchanged)"})
(PKG / "repaired/FEATURE_EQUALITY_COUNTS.json").write_text(json.dumps(eq, indent=1))
# R3 research-decision table: HMDA P3/P4 'lower' entries were the upper endpoints
pe = {r["id"]: r for r in csv.DictReader(open(OLD / "PRIMARY_ENDPOINTS.csv"))}
for rid in ("P3-hmda", "P4-hmda"):
    r = pe[rid]
    rows.append({"item": "R3_table_endpoint_label", "dataset": "hmda", "location": f"RESEARCH_DECISION.md section 3 row {rid[:2]} (simultaneous lower bound column)",
                 "original": f"lower bound shown as {float(r['upper']):.4f} (this is the UPPER endpoint)",
                 "repaired": f"lower {float(r['lower']):.4f}, upper {float(r['upper']):.4f}; decision {r['decision']} unchanged (lower > -0.01)",
                 "decisions_changed": "none"})
with open(PKG / "ORIGINAL_VS_REPAIRED.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump({"original_file_sha256_at_f7425b1": {n: sha(OLD / n) for n in ("ROLES_AND_SUPPORT.json", "RESEARCH_DECISION.md", "PRIMARY_ENDPOINTS.csv")}},
          open(PKG / "repaired/ORIGINAL_HASHES.json", "w"), indent=1)
print(len(rows)); [print(r["item"], r["dataset"], "|", r["repaired"][:150]) for r in rows]
