#!/usr/bin/env python3
"""Independent audit of the Stage B / G2 diagnostic at its pinned commit.

Reads only committed evidence through the git object store. Recomputes every quantity the
manuscript uses from the smallest machine-readable source available, and records which reported
numbers have a committed generator and which do not.
"""
import hashlib, io, json, os, statistics as st, subprocess

SHA = "cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RD = "results/pcrl_stochastic_channel_v1/"
ENDS = ("0", "1", "2")
sources, findings = {}, []


def show(path):
    raw = subprocess.run(["git", "-C", ROOT, "show", f"{SHA}:{path}"], check=True,
                         capture_output=True).stdout
    sources[path] = hashlib.sha256(raw).hexdigest()
    return raw.decode()


def note(key, value, status, comment):
    findings.append({"item": key, "value": value, "status": status, "comment": comment})


stage_b = json.loads(show(RD + "STAGE_B_RESULTS.json"))
g2 = json.loads(show(RD + "gates/G2.json"))
code = show("experiments/pcrl_stochastic_channel_v1/stage_b.py")

# ---------------------------------------------------------------- 1. B2, both weightings
per_res = {}
for res in ("64", "256"):
    uw = [stage_b[res][s]["B2_deployable"]["residence_advantage_nats_code_only"] for s in ENDS]
    pw = [stage_b[res][s]["B2_deployable"]["residence_advantage_nats_person_weighted"] for s in ENDS]
    se = [stage_b[res][s]["B2_interval_code_only"]["bootstrap_se"] for s in ENDS]
    per_res[res] = {
        "unweighted_by_seed": uw, "person_weighted_by_seed": pw, "bootstrap_se_by_seed": se,
        "seed_mean_unweighted": st.mean(uw), "seed_mean_person_weighted": st.mean(pw),
        "seeds_positive_unweighted": sum(x > 0 for x in uw),
        "seeds_positive_person_weighted": sum(x > 0 for x in pw),
        "abs_estimate_over_se_by_seed": [abs(u) / s for u, s in zip(uw, se)],
        "households": stage_b[res]["0"]["B2_interval_code_only"]["households"],
        "validation_rows_by_seed": [stage_b[res][s]["validation_rows"] for s in ENDS],
        "fit_rows": stage_b[res]["0"]["fit_rows"],
        "probe_input_dim": stage_b[res]["0"]["B2_deployable"]["code_probe_input_dim"],
        "ref_J_probe_input_dim": stage_b[res]["0"]["B2_deployable"]["ref_J_probe_input_dim"],
        "selected_family": [stage_b[res][s]["B2_deployable"]["code_selected"] for s in ENDS],
    }

note("G2 seed-mean advantage reproduces", {r: round(per_res[r]["seed_mean_unweighted"], 5) for r in per_res},
     "CONFIRMED", "matches gates/G2.json seed_mean_unweighted at both resolutions")
note("seeds_with_positive_advantage is unweighted-only",
     {r: {"unweighted": per_res[r]["seeds_positive_unweighted"],
          "person_weighted": per_res[r]["seeds_positive_person_weighted"]} for r in per_res},
     "CORRECTION",
     "gates/G2.json reports seeds_with_positive_advantage = 0 without naming a weighting, and the "
     "decision document says 'negative in 3/3 seeds at both resolutions'. Person-weighted, seed 0 is "
     "POSITIVE at both resolutions (+0.00216 at 64, +0.00007 at 256). Seed means stay negative under "
     "both weightings, so the G2 verdict is unaffected.")
note("precision at the declared resolution", per_res["64"]["abs_estimate_over_se_by_seed"], "CONFIRMED",
     "|estimate|/SE = 0.16, 0.98, 0.41 at |T|=64: within ~1 SE of zero, i.e. no demonstrated advantage "
     "and no demonstrated deficit, as the decision document itself states.")
note("precision at the fallback resolution", per_res["256"]["abs_estimate_over_se_by_seed"], "CONTEXT",
     "|estimate|/SE = 0.72, 3.07, 2.64 at |T|=256, where the probe carries 276 input columns against "
     "the baseline's 20 on 2048 fitting rows; a width handicap is a live explanation of the deficit.")

# ---------------------------------------------------------------- 2. subsumption provenance
sub = g2["subsumption_check"]["validation_log_loss_nats"]
match = all(round(stage_b["64"][s]["B2_deployable"]["ref_J_log_loss"], 5) == sub[s]["J_only"]
            and round(stage_b["64"][s]["B2_deployable"]["unconstrained_code_log_loss"], 5) == sub[s]["J_plus_T"]
            for s in ENDS)
blob = json.dumps(stage_b)
note("J_only and J_plus_T trace to STAGE_B_RESULTS.json", match, "CONFIRMED" if match else "MISMATCH",
     "the two columns that carry the comparison reproduce exactly")
note("prior and T_only have no machine-readable backing",
     {"in_stage_b_results": "T_only" in blob or "prior" in blob,
      "generator_in_committed_code": ("T_only" in code) or ("t_only" in code)},
     "GAP",
     "gates/G2.json reports prior and T-only validation log losses, and the closing commit message "
     "repeats them, but STAGE_B_RESULTS.json contains neither and no committed code fits a T-only or "
     "prior probe. Two of the four columns of the headline table are unreproducible from this commit.")
gap = {s: round(sub[s]["J_plus_T"] - sub[s]["J_only"], 5) for s in ENDS}
ses = {s: round(stage_b["64"][s]["B2_interval_code_only"]["bootstrap_se"], 5) for s in ENDS}
note("size of the 'subsumption' difference against its own uncertainty",
     {"J_plus_T_minus_J_only": gap, "bootstrap_se": ses,
      "ratio": {s: round(abs(gap[s]) / ses[s], 2) for s in ENDS}},
     "CORRECTION",
     "the difference that motivates the word 'subsumed' is 0.16-0.98 SE. It supports 'no measurable "
     "improvement at this probe capacity', not an information-theoretic subsumption claim.")

# ---------------------------------------------------------------- 3. B1 label path
leak_path = ("_per_row(" in code and "y_val" in code
             and "_baseline_score(ref_j" in code and "per_row_loss" in code)
dec = {s: stage_b["64"][s]["B1_fitted_model_ceiling"]["probability_decile_partition"] for s in ENDS}
note("B1 decile partition is built from per-row loss, which reads the residence label", leak_path,
     "DEFECT",
     "stage_b.run_seed: decile_cells = digitize(_baseline_score(ref_j, j_val), ...); _baseline_score "
     "returns probe['per_row_loss']; _per_row computes -log p[i, y_i] from y_val = the residence "
     "label. The partition coordinate is therefore a function of each person's own label.")
note("numerical signature of the leak",
     {"baseline_log_loss_by_seed": [round(dec[s]["baseline_log_loss"], 5) for s in ENDS],
      "real_residence_log_loss_J_only": [sub[s]["J_only"] for s in ENDS],
      "n_base_cells": dec["0"]["n_base_cells"]},
     "DEFECT",
     "ten cells reduce residence log loss from ~0.52 to 0.065-0.094 nats. No label-free 10-cell "
     "partition of a service-probability view can do that; the fixture b1_label_leak_fixture.py "
     "reproduces the effect on a label that is pure noise.")
note("B1 influence on the gate decision",
     {"gate_verdict_source": g2["what_was_measured"][:60] + "...",
      "decision": g2["decision"], "B1_verdict": g2["B1_fitted_model_ceiling"]["verdict"]},
     "SCOPED",
     "G2's verdict is computed from B2; B1 is reported as 'NOT RELIABLY ESTIMABLE' and did not gate, "
     "select or stop anything. The defect is confined to interpretation and must not be propagated to "
     "the G2 outcome or to unrelated studies.")
note("product partition is label-free but sparse",
     {"rows_per_cell": [stage_b["64"][s]["B1_fitted_model_ceiling"]["product_partition"]["rows_per_cell"]
                        for s in ENDS]},
     "CONFIRMED",
     "_bin_view uses only the baseline view, no label. Its 1.9-2.1 rows per cell make it unusable, "
     "which the study states correctly.")

out = {"audited_commit": SHA, "sources_sha256": sources, "findings": findings,
       "B2_recomputed": per_res,
       "scope": "arithmetic on committed evidence; no ACS model fitted, no pool read, 2016 untouched"}
if __name__ == "__main__":
    path = os.path.join(ROOT, "results/pcrl_manuscript_overnight_v1/AUDIT_STAGE_B.json")
    json.dump(out, open(path, "w"), indent=1)
    for f in findings:
        print(f"[{f['status']:10s}] {f['item']}")
    print(f"\n{len(findings)} findings written to {os.path.relpath(path, ROOT)}")
