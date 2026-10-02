"""Real-shape SMOKE validation of the official-LEACE wrapper (method owner, 2026-10-02). NOT a benchmark fit.

Uses the admitted defense_fit representations (fit role only) with SHUFFLED sensitive labels (a seeded joint
row permutation, so class frequencies and the joint label structure are preserved but any link to H is broken).
This validates the wrapper on the real dimensionality / conditioning / block structure (B: target one-hot,
C: concatenated marginal policy one-hots) without computing any real concept map. The real B/C maps are fitted
by the coordinator. Also records the label-free scale report used to read the absolute noise sigmas (arm D).

No sensitive or task label of attacker / utility / assessment rows is read. Their representations are used only for a
label-free geometry check (variance outside the defense_fit covariance support, where the official map is the
identity). Writes notes/method/smoke_real_shape.json (aggregates only).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from stored_model_eval import defenses as D  # noqa: E402

INPUTS = Path(os.path.expanduser("~/PCRL_eval_cache_private/bench_v1/inputs"))
SHUFFLE_SEED = 20261002


def main():
    idx = json.loads((INPUTS / "INPUTS_INDEX.json").read_text())
    out = {"schema": "smoke_real_shape_v1", "date": "2026-10-02", "status": "smoke validation only; shuffled labels",
           "shuffle_seed": SHUFFLE_SEED, "provenance": D.verify_official_leace(), "cells": {}}
    out["provenance"]["installed_path"] = "<venv>/site-packages/concept_erasure"
    for ds, info in idx["datasets"].items():
        t1 = info["tier1"]
        purpose = info["purposes"][t1["purpose"]]
        L = np.load(info["labels_npz"])
        fit = np.flatnonzero(L["role"] == "defense_fit")
        perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(fit))
        dims = purpose["disallowed_attr_dims"]
        shuffled = {a: L[a][fit][perm] for a in t1["policy"]}
        for seed, enc in sorted(info["encoders"].items()):
            if enc.get("lineage_status") != "ADMITTED":
                out["cells"][f"{ds}__s{seed}"] = {"status": "encoder not admitted"}
                continue
            F = np.load(enc["forward_npz"])
            assert np.array_equal(F["row_id"], L["row_id"])
            H = F[purpose["rep_key"]][fit].astype(np.float64)
            cell = {"n_defense_fit": int(len(fit)), "dim": int(H.shape[1]), "rep_key": purpose["rep_key"],
                    "scale_report_defense_fit": {k: v for k, v in D.scale_report(H).items() if k != "per_dim_sd"}}
            cell["scale_report_defense_fit"]["per_dim_sd"] = {
                k: v for k, v in D.scale_report(H)["per_dim_sd"].items() if k != "values"}
            # label-free: do other-role rows carry variance outside the defense_fit covariance support?
            mu = H.mean(0)
            ev, V = np.linalg.eigh(np.cov(H.T))
            null = ev <= ev[-1] * H.shape[1] * np.finfo(np.float64).eps
            sup = {"defense_fit_sample_cov_rank": int((~null).sum()), "n_null_dirs": int(null.sum())}
            for r in ("attacker_fit", "attacker_val", "assessment"):
                Hr = F[purpose["rep_key"]][L["role"] == r].astype(np.float64)
                tot = float(((Hr - mu) ** 2).sum(1).mean())
                out_sup = float((((Hr - mu) @ V[:, null]) ** 2).sum(1).mean()) if null.any() else 0.0
                sup[f"{r}_mean_sq_outside_support_over_total"] = out_sup / tot
                sup[f"{r}_max_abs_coord_outside_support"] = float(
                    np.abs((Hr - mu) @ V[:, null]).max()) if null.any() else 0.0
            dead = np.flatnonzero(np.ptp(F[purpose["rep_key"]], axis=0) == 0)
            sup["dims_constant_on_all_rows"] = int(len(dead))
            cell["support_transfer_label_free"] = sup
            arms = {"B": {t1["target"]: shuffled[t1["target"]]}, "C": {a: shuffled[a] for a in t1["policy"]}}
            maps = {}
            for arm, labs in arms.items():
                Z, spec = D.concat_marginal_onehots(labs, {a: dims[a] for a in labs})
                t0 = time.process_time()
                m = D.fit_leace(H, Z, fit_row_ids=L["row_id"][fit], concept_spec=spec)
                cpu = time.process_time() - t0
                nc = m.native_check(H, Z)
                dg = m.metadata["diagnostics"]
                maps[arm] = m
                cell[arm] = {"concept_blocks": [(b["name"], b["n_classes"]) for b in spec], "fit_cpu_s": cpu,
                             "rank": m.metadata["rank"], "singular_values": dg["singular_values_whitened_xz"],
                             "n_nonzero_truncated": dg["n_singular_values_nonzero_truncated"],
                             "trace_constraint_fired": dg["cov_trace_constraint_fired"],
                             "shrinkage_alpha": dg["shrinkage_alpha"], "sample_cov_rank": dg["sample_cov_rank"],
                             "sample_cov_eig_min_max": [dg["sample_cov_eig_min"], dg["sample_cov_eig_max"]],
                             "native_status": nc["status"],
                             "crosscov_max_abs_rel_erased": nc["crosscov_max_abs_rel_erased"],
                             "per_block_ols_r2_erased": {b["name"]: b["ols_r2_fit_rows_erased"] for b in nc["blocks"]},
                             "whitened_residual_spectral_norm": nc["whitened_residual_spectral_norm"],
                             "implementation_bound_holds": nc["implementation_bound_holds"]}
            cell["alias_B_C"] = {k: v for k, v in D.alias_test(maps["B"], maps["C"]).items()}
            out["cells"][f"{ds}__s{seed}"] = cell
    (HERE / "smoke_real_shape.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    o = main()
    for k, c in o["cells"].items():
        if "B" not in c:
            print(k, c)
            continue
        sr = c["scale_report_defense_fit"]
        print(k, "rms_sd=%.4f" % sr["sigma_train_scale_rms_sd"], "median_norm=%.3f" % sr["norm_quantiles"]["q50"])
        for arm in ("B", "C"):
            a = c[arm]
            print("  ", arm, a["concept_blocks"], "rank", a["rank"], "trunc", a["n_nonzero_truncated"],
                  "sv", np.round(a["singular_values"], 4).tolist(), a["native_status"],
                  "%.2e" % a["crosscov_max_abs_rel_erased"], "cpu %.2fs" % a["fit_cpu_s"],
                  "covrank", a["sample_cov_rank"], "eig", ["%.2e" % e for e in a["sample_cov_eig_min_max"]])
        print("   alias", c["alias_B_C"]["alias"], "| wres B/C %.4f/%.4f" % (c["B"]["whitened_residual_spectral_norm"],
              c["C"]["whitened_residual_spectral_norm"]), c["B"]["implementation_bound_holds"],
              c["C"]["implementation_bound_holds"], "| support", c["support_transfer_label_free"])
